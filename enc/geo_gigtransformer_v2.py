"""Parameterized GiG model; legacy graph direction and signed pooling are preserved."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot
from .geo_gigtransformer import GeneTransformerConv, PatientTransformerConv, GIG_Transformer
if __package__ and "." in __package__:
    from ..geo_loader.geo_readgraph import read_batch
    from ..geo_loader.geograph_sampler import GeoGraphLoader
else:
    from geo_loader.geo_readgraph import read_batch
    from geo_loader.geograph_sampler import GeoGraphLoader


class GIG_TransformerV2(nn.Module):
    # Keep the existing loss definition, including its softmax regularization.
    loss = GIG_Transformer.loss

    def __init__(self, config, num_genes):
        super().__init__()
        c=config
        self.config=dict(c)
        self.gene_num_top_feature=c['gene_num_top_feature']
        self.gene_embedding_dim=c['gene_output_dim']
        self.graph_opt=c['graph_opt']
        for key in ('class_weight','ortho_weight','link_weight','ent_weight'):
            setattr(self,key,c[key])
        self.node_weight_assign=nn.Parameter(torch.empty(c['gene_num_top_feature'],num_genes))
        glorot(self.node_weight_assign)
        self.act=nn.LeakyReLU(c['activation_negative_slope'])
        self.representation_dropout=c['representation_dropout']
        self.gene_convs=self._layers(GeneTransformerConv,c['gene_input_dim'],c['gene_hidden_dim'],
            c['gene_output_dim'],c['gene_layers'],c['gene_num_head'],c['gene_attention_dropout'],c['gene_root_weight'],c['gene_beta'])
        self.pheno_transform=nn.Linear(c['gig_input_dim'],c['gig_input_transform_dim'])
        self.gene_transform=nn.Linear(c['gene_num_top_feature']*c['gene_output_dim'],c['gig_output_dim'])
        if self.graph_opt in ('GinG','subject'):
            width=c['gig_input_transform_dim']+(c['gene_num_top_feature']*c['gene_output_dim'] if self.graph_opt=='GinG' else 0)
            self.patient_convs=self._layers(PatientTransformerConv,width,c['gig_hidden_dim'],c['gig_output_dim'],
                c['patient_layers'],c['gig_num_head'],c['patient_attention_dropout'],c['patient_root_weight'],c['patient_beta'])
        self.out=nn.Linear(c['gig_output_dim'],c['num_classes'],bias=False)
        for module in (self.pheno_transform,self.gene_transform,self.out):glorot(module.weight)

    @staticmethod
    def _layers(kind,input_dim,hidden,output,depth,heads,dropout,root,beta):
        widths=[input_dim]+[hidden]*(depth-1)+[output]
        return nn.ModuleList([kind(widths[i],widths[i+1]//heads,heads=heads,concat=True,
                                    dropout=dropout,root_weight=root,beta=beta) for i in range(depth)])

    def _activate(self,values):
        return F.dropout(self.act(values),p=self.representation_dropout,training=self.training)

    def forward(self,num_feature,num_subfeature,num_subject,num_gene_node,gene_feature,gene_edge_index,
                x,edge_index,node_label,node_index,args,device):
        if self.graph_opt!='subject':
            chunks=[]
            for start in range(0,num_subject,args.batch_size):
                stop=min(start+args.batch_size,num_subject)
                graphs=read_batch(index=start,upper_index=stop,x_input=gene_feature,num_feature=num_feature,
                                  num_node=num_gene_node,edge_index=gene_edge_index)
                loader,_,_=GeoGraphLoader.load_graph(graphs,args)
                pieces=[]
                for data in loader:
                    h=data.x.float().to(device);edges=data.edge_index.to(device)
                    for conv in self.gene_convs:h=self._activate(conv(h,edges))
                    h=h.reshape(-1,num_gene_node,self.gene_embedding_dim)
                    pieces.append(torch.matmul(self.node_weight_assign,h).reshape(h.shape[0],-1))
                chunks.append(torch.cat(pieces,dim=0))
            subject_gene=torch.cat(chunks,dim=0)
            gene_embed=torch.cat([subject_gene.new_zeros((num_subfeature,subject_gene.shape[1])),subject_gene],dim=0)
        if self.graph_opt=='GinG':
            h=torch.cat([gene_embed,self.pheno_transform(x)],dim=1)
        elif self.graph_opt=='subject':h=self.pheno_transform(x)
        else:h=self.gene_transform(gene_embed)
        if self.graph_opt in ('GinG','subject'):
            for conv in self.patient_convs:h=self._activate(conv(h,edge_index))
        logits=self.out(h);pred=logits.argmax(dim=1)
        return h,logits[node_index],pred,pred[node_index]
