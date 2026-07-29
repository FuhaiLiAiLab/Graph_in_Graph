import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.autograd import Variable
from torch_scatter import scatter_add
from torch_geometric import utils
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

from geo_loader.geo_readgraph import read_geodata, read_batch
from geo_loader.geograph_sampler import GeoGraphLoader

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
    def __init__(self, input_dim, embed_dim, num_head=3, negative_slope=0.2, aggr = "add", num_edge_type=0):
        super(GATConv, self).__init__(node_dim=0)
        assert embed_dim % num_head == 0
        self.k = embed_dim//num_head
        self.aggr = aggr
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(input_dim, embed_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, num_head, 2 * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(embed_dim))

        if num_edge_type>0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, embed_dim)
            glorot(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_linear.weight.data)
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


class GIG_GAT(nn.Module):
    def __init__(self, gene_input_dim, gene_hidden_dim, gene_embedding_dim,
                        gene_num_top_feature, num_gene_node,
                        gig_input_dim, gig_input_transform_dim, gig_hidden_dim, gig_embedding_dim,
                        num_classes, gene_num_head, gig_num_head, class_weight_fine,
                        class_weight, ortho_weight, link_weight, ent_weight, graph_opt):
        super(GIG_GAT, self).__init__()
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_num_top_feature = gene_num_top_feature
        self.node_weight_assign = torch.nn.Parameter(torch.Tensor(gene_num_top_feature, num_gene_node))

        self.class_weight = class_weight
        self.ortho_weight = ortho_weight
        self.link_weight = link_weight
        self.ent_weight = ent_weight
        self.graph_opt = graph_opt

        self.gene_conv_first, self.gene_conv_block, self.gene_conv_last = self.gene_build_conv_layer(
                    gene_input_dim=gene_input_dim, 
                    gene_hidden_dim=gene_hidden_dim, 
                    gene_embedding_dim=gene_embedding_dim,
                    gene_num_head=gene_num_head)
        if graph_opt == 'GinG':
            self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                        gig_input_transform_dim=gig_input_transform_dim + (gene_num_top_feature * gene_embedding_dim), 
                        gig_hidden_dim=gig_hidden_dim, gig_embedding_dim=gig_embedding_dim,
                        gig_num_head=gig_num_head)
        elif graph_opt == 'subject':
            self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                        gig_input_transform_dim=gig_input_transform_dim,
                        gig_hidden_dim=gig_hidden_dim, 
                        gig_embedding_dim=gig_embedding_dim,
                        gig_num_head=gig_num_head)

        self.act = nn.LeakyReLU(negative_slope=0.5)
        self.gene_transform = nn.Linear(gene_num_top_feature * gene_embedding_dim, gig_embedding_dim)
        self.pheno_transform = nn.Linear(gig_input_dim, gig_input_transform_dim)
        self.out = torch.nn.Linear(gig_embedding_dim, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.node_weight_assign)
        glorot(self.gene_transform.weight.data)
        glorot(self.pheno_transform.weight.data)
        glorot(self.out.weight.data)
    
    def gene_build_conv_layer(self, gene_input_dim, gene_hidden_dim, gene_embedding_dim, gene_num_head):
        gene_conv_first = GATConv(input_dim=gene_input_dim, embed_dim=gene_hidden_dim, num_head=gene_num_head)
        gene_conv_block = GATConv(input_dim=gene_hidden_dim, embed_dim=gene_hidden_dim, num_head=gene_num_head)
        gene_conv_last = GATConv(input_dim=gene_hidden_dim, embed_dim=gene_embedding_dim, num_head=gene_num_head)
        return gene_conv_first, gene_conv_block, gene_conv_last

    def build_conv_layer(self, gig_input_transform_dim, gig_hidden_dim, gig_embedding_dim, gig_num_head):
        conv_first = GATConv(input_dim=gig_input_transform_dim, embed_dim=gig_hidden_dim, num_head=gig_num_head)
        conv_block = GATConv(input_dim=gig_hidden_dim, embed_dim=gig_hidden_dim, num_head=gig_num_head)
        conv_last = GATConv(input_dim=gig_hidden_dim, embed_dim=gig_embedding_dim, num_head=gig_num_head)
        return conv_first, conv_block, conv_last

    def forward(self, num_feature, num_subfeature, num_subject, num_gene_node, 
                    gene_feature, gene_edge_index,
                    x, edge_index, node_label, node_index, args, device):
        ### Embeddings for gene features from batch to batch
        # Initialize the gene feature
        gene_x_embed = torch.zeros((num_subfeature + num_subject, (self.gene_num_top_feature * self.gene_embedding_dim)), requires_grad=True).to(device)
        clone_gene_x_embed = gene_x_embed.clone()
        upper_index = 0
        num_subject_node = gene_feature.shape[0]
        if self.graph_opt != 'subject':
            for index in tqdm(range(0, num_subject_node, args.batch_size)):
                if (index + args.batch_size) < num_subject_node:
                    upper_index = index + args.batch_size
                else:
                    upper_index = num_subject_node
                # [batch_node_weight_assign] 
                batch_node_weight_assign = self.node_weight_assign.repeat(1, upper_index - index)
                batch_node_weight_assign = batch_node_weight_assign.view(self.gene_num_top_feature, upper_index - index, num_gene_node).transpose(0,1)
                geo_datalist = read_batch(index=index, upper_index=upper_index, 
                                        x_input=gene_feature, num_feature=num_feature, 
                                        num_node=num_gene_node, edge_index=gene_edge_index)
                dataset_loader, node_num, feature_dim = GeoGraphLoader.load_graph(geo_datalist, args)
                for batch_idx, data in enumerate(dataset_loader):
                    gene_graph_x = Variable(data.x.float(), requires_grad=False).to(device)
                    gene_graph_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
                    gene_graph_x = self.gene_conv_first(gene_graph_x, gene_graph_edge_index)
                    gene_graph_x = self.act(gene_graph_x)
                    # gene_graph_x = self.gene_conv_block(gene_graph_x, gene_graph_edge_index)
                    # gene_graph_x = self.act(gene_graph_x)
                    # gene_graph_x = self.gene_conv_last(gene_graph_x, gene_graph_edge_index)
                    # gene_graph_x = self.act(gene_graph_x)
                    batch_gene_x = gene_graph_x.view(-1, num_gene_node, self.gene_embedding_dim)
                    batch_assigned_gene_x = torch.matmul(batch_node_weight_assign, batch_gene_x).reshape(upper_index - index, -1)
                # Preserve prediction of batch training data
                # import pdb; pdb.set_trace()
                clone_gene_x_embed[num_subfeature+index:num_subfeature+upper_index, :] = batch_assigned_gene_x
        
        # Embeddings for [gene features / pheno features] for GiG_GAT
        if self.graph_opt == 'GinG':
            pheno_x = self.pheno_transform(x)
            gig_x = torch.concat([clone_gene_x_embed, pheno_x], dim=1) # gene features + pheno features
            gig_x = self.conv_first(gig_x, edge_index)
            gig_x = self.act(gig_x)
            gig_x = self.conv_block(gig_x, edge_index)
            gig_x = self.act(gig_x)
            gig_x = self.conv_last(gig_x, edge_index)
            gig_x = self.act(gig_x)
            x_embed = gig_x.clone()
        elif self.graph_opt == 'subject':
            pheno_x = self.pheno_transform(x) # pheno features
            gig_x = self.conv_first(pheno_x, edge_index)
            gig_x = self.act(gig_x)
            gig_x = self.conv_block(gig_x, edge_index)
            gig_x = self.act(gig_x)
            gig_x = self.conv_last(gig_x, edge_index)
            gig_x = self.act(gig_x)
            x_embed = gig_x.clone()
        elif self.graph_opt == 'gene':
            gig_x = self.gene_transform(clone_gene_x_embed) # gene features
            x_embed = gig_x.clone()
        
        # import pdb; pdb.set_trace()
        output = self.out(gig_x) # [output] contains all nodes [subfeatures + subjects]
        node_output = output[node_index] # [node_output] contains only subjects
        _, ypred = torch.max(output, dim=1)
        y_nodepred = ypred[node_index]
        return x_embed, node_output, ypred, y_nodepred

    def loss(self, node_output, node_label, gene_edge_index, gene_num_top_feature):
        ### Classification loss
        node_output = node_output.to(device='cuda')
        node_label = node_label.to(device='cuda')
        node_label_indices = torch.argmax(node_label, dim=2).view(-1)
        # Weight vector renorm label
        weight_vector = torch.zeros([3]).to(device='cuda')
        for i in range(3):
            n_samplei = torch.sum(node_label_indices == i)
            if i == 2:
                weight_vector[i] = len(node_label) / (n_samplei * 2)
            else:
                weight_vector[i] = len(node_label) / (n_samplei)

        node_output = torch.log_softmax(node_output, dim=-1)
        class_loss = F.nll_loss(node_output, node_label_indices, weight_vector)
        ### Assignment loss
        # import pdb; pdb.set_trace()
        s = self.node_weight_assign # [gene_num_top_feature, num_gene_node]
        s = torch.softmax(s, dim=-1)
        # [orthogonality loss]
        ss = torch.matmul(s, s.transpose(0, 1))
        i_s = torch.eye(gene_num_top_feature).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-1, -2))
        ortho_loss = torch.mean(ortho_loss)
        # [link_loss (or Laplacian loss)]
        adj = utils.to_dense_adj(gene_edge_index).to(device='cuda')
        link_loss = adj - torch.matmul(s.transpose(0, 1), s)
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / adj.numel()
        # import pdb; pdb.set_trace()
        # [entropy loss]
        EPS = 1e-15
        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()
        ### Total loss
        total_loss = self.class_weight * class_loss + self.ortho_weight * ortho_loss + self.link_weight * link_loss + self.ent_weight * ent_loss
        print('class_loss: ', class_loss.item(), 'ortho_loss: ', ortho_loss.item(), 'link_loss: ', link_loss.item(), 'ent_loss: ', ent_loss.item(), 'total_loss: ', total_loss.item())
        return total_loss