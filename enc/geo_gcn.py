import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

from torch_geometric.nn import aggr

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # X: [N, in_channels]
        # edge_index: [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

         # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] OUT PUT DIMS = [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_head =2, num_classes=3):
        super(GCN, self).__init__()

        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.2)


        self.out = torch.nn.Linear(embedding_dim, num_classes)

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_block = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_last = GCNConv(in_channels=hidden_dim, out_channels=embedding_dim)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, node_label, node_index):
        # import pdb; pdb.set_trace()
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