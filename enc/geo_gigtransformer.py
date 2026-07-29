import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.autograd import Variable
from torch_scatter import scatter_add
from torch_geometric import utils
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax

from geo_loader.geo_readgraph import read_geodata, read_batch
from geo_loader.geograph_sampler import GeoGraphLoader


class GeneTransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        
        # import pdb; pdb.set_trace()

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # import pdb; pdb.set_trace()
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class PatientTransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)
        
        # import pdb; pdb.set_trace()

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        # import pdb; pdb.set_trace()
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # import pdb; pdb.set_trace()
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GIG_Transformer(nn.Module):
    def __init__(self, gene_input_dim, gene_hidden_dim, gene_embedding_dim,
                        gene_num_top_feature, num_gene_node,
                        gig_input_dim, gig_input_transform_dim, gig_hidden_dim, gig_embedding_dim,
                        num_classes, gene_num_head, gig_num_head, class_weight_fine,
                        class_weight, ortho_weight, link_weight, ent_weight, graph_opt):
        super(GIG_Transformer, self).__init__()
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
        gene_conv_first = GeneTransformerConv(in_channels=gene_input_dim, out_channels=gene_hidden_dim, heads=gene_num_head)
        gene_conv_block = GeneTransformerConv(in_channels=gene_hidden_dim, out_channels=gene_hidden_dim, heads=gene_num_head)
        gene_conv_last = GeneTransformerConv(in_channels=gene_hidden_dim, out_channels=gene_embedding_dim, heads=gene_num_head)
        return gene_conv_first, gene_conv_block, gene_conv_last

    def build_conv_layer(self, gig_input_transform_dim, gig_hidden_dim, gig_embedding_dim, gig_num_head):
        conv_first = PatientTransformerConv(in_channels=gig_input_transform_dim, out_channels=gig_hidden_dim, heads=gig_num_head)
        conv_block = PatientTransformerConv(in_channels=gig_hidden_dim, out_channels=gig_hidden_dim, heads=gig_num_head)
        conv_last = PatientTransformerConv(in_channels=gig_hidden_dim, out_channels=gig_embedding_dim, heads=gig_num_head)
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
                    gene_graph_x = self.gene_conv_block(gene_graph_x, gene_graph_edge_index)
                    gene_graph_x = self.act(gene_graph_x)
                    gene_graph_x = self.gene_conv_last(gene_graph_x, gene_graph_edge_index)
                    gene_graph_x = self.act(gene_graph_x)
                    batch_gene_x = gene_graph_x.view(-1, num_gene_node, self.gene_embedding_dim)
                    batch_assigned_gene_x = torch.matmul(batch_node_weight_assign, batch_gene_x).reshape(upper_index - index, -1)
                # Preserve prediction of batch training data
                clone_gene_x_embed[num_subfeature+index:num_subfeature+upper_index, :] = batch_assigned_gene_x
        
        # Embeddings for [gene features / pheno features] for GiG_Transformer
        # import pdb; pdb.set_trace()
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
            # import pdb; pdb.set_trace()
            gig_x = self.gene_transform(clone_gene_x_embed) # gene features
            x_embed = gig_x.clone()
            
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