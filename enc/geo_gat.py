import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax


# GAT torch_geometric implementation
# Adapted from https://github.com/snap-stanford/pretrain-gnns
class GATConv(MessagePassing):
    """Graph attention layer with edge attribute
    Args:
        input_dim(int): the size of input feature
        embed_dim(int): the size of output feature
        num_head(int): the number of num_head in multi-head attention
        negative_slope(float): the slope in leaky relu function
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute

    """
    def __init__(self, input_dim, embed_dim, num_head=6, negative_slope=0.2, aggr = "add", num_edge_type=0):
        super(GATConv, self).__init__(node_dim=0)
        assert embed_dim%num_head==0
        self.k=embed_dim//num_head
        self.aggr = aggr

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(input_dim, embed_dim,bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, num_head, 2 * num_head * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(embed_dim))

        if num_edge_type>0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, embed_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        # Add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.weight_linear(x).view(-1, self.num_head, self.k) # N * num_head * k
        return self.propagate(edge_index, x=x)

    def message(self, edge_index, x_i, x_j):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * num_head
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.num_head, 1) #E * num_head * k

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1,self.embed_dim)
        aggr_out = aggr_out + self.bias
        return F.relu(aggr_out)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_head=2, num_classes=3):
        super(GAT, self).__init__()
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.3)
        self.out = torch.nn.Linear(embedding_dim, num_classes)


    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GATConv(input_dim=input_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_block = GATConv(input_dim=hidden_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_last = GATConv(input_dim=hidden_dim, embed_dim=embedding_dim, num_head=self.num_head)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, node_label, node_index):
        x = self.conv_first(x, edge_index)
        x = self.act2(x)

        x = self.conv_block(x, edge_index)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.act2(x)
        x_embed = x

        output = self.out(x) # [output] contains all nodes [subfeatures + subjects]
        node_output = output[node_index] # [node_output] contains only subjects
        _, ypred = torch.max(output, dim=1)
        y_nodepred = ypred[node_index]
        return x_embed, node_output, ypred, y_nodepred

    def loss(self, node_output, node_label):
        node_output = node_output.to(device='cuda')
        node_label = node_label.to(device='cuda')
        node_label_indices = torch.argmax(node_label, dim=2).view(-1)
        # Weight vector renorm label
        weight_vector = torch.zeros([3]).to(device='cuda')
        for i in range(3):
            n_samplei = torch.sum(node_label_indices == i)
            weight_vector[i] = len(node_label) / (n_samplei)

        node_output = torch.log_softmax(node_output, dim=-1)
        loss = F.nll_loss(node_output, node_label_indices, weight_vector)
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(node_output, node_label)
        return loss