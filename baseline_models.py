"""Independent joint-input GAT/GIN based on the supplied reference encoders."""
import numpy as np
import torch
from torch import nn
from enc.geo_gat import GATConv
from enc.geo_gin import GINConv

def joint_input(data):
    pheno=np.asarray(data['graph_x'],dtype=np.float32);gene=np.asarray(data['gene_x'],dtype=np.float32);idx=np.asarray(data['node_index'])
    if idx.ndim!=1 or len(idx)!=data['num_subject'] or len(set(idx.tolist()))!=len(idx) or not np.issubdtype(idx.dtype,np.integer):raise ValueError('Invalid subject node indices')
    if np.any(idx<0) or np.any(idx>=len(pheno)) or gene.shape[0]!=len(idx):raise ValueError('Invalid joint-input dimensions')
    if not np.isfinite(pheno).all() or not np.isfinite(gene).all():raise ValueError('Nonfinite input')
    result=np.zeros((len(pheno),pheno.shape[1]+gene.shape[1]),dtype=np.float32)
    result[:,:pheno.shape[1]]=pheno;result[idx,pheno.shape[1]:]=gene
    return result

class BaselineClassifier(nn.Module):
    def __init__(self,architecture,input_dim,cfg):
        super().__init__()
        if architecture not in ('gat','gin'):raise ValueError('Unknown baseline')
        self.architecture=architecture
        depth=cfg['patient_layers']
        if depth<2:raise ValueError('Need at least two graph layers')
        widths=[input_dim]+[cfg['gig_hidden_dim']]*(depth-1)+[cfg['gig_output_dim']]
        fix=cfg.get('baseline_activation_fix',False)
        if type(fix) is not bool:raise ValueError('baseline_activation_fix must be bool')
        norm=cfg.get('baseline_gin_layer_norm',False)
        if type(norm) is not bool or (architecture!='gin' and norm):raise ValueError('LayerNorm switch is only valid for GIN')
        self.layers=nn.ModuleList([GATConv(a,b,num_head=cfg['gig_num_head'],negative_slope=cfg['activation_negative_slope'],output_relu=not fix) if architecture=='gat' else GINConv(a,b,activation_slope=cfg['activation_negative_slope'] if fix else None,hidden_layer_norm=norm and i==0) for i,(a,b) in enumerate(zip(widths,widths[1:]))])
        self.act=nn.LeakyReLU(cfg['activation_negative_slope']);self.dropout=nn.Dropout(cfg['representation_dropout'])
        head=cfg.get('baseline_mlp_hidden',0)
        if type(head) is not int or head<0 or (architecture!='gat' and head):raise ValueError('MLP head is only enabled for GAT')
        self.out=nn.Linear(widths[-1],cfg['num_classes']) if head==0 else nn.Sequential(nn.Linear(widths[-1],head),nn.LeakyReLU(cfg['activation_negative_slope']),nn.Linear(head,cfg['num_classes']))
        residual=cfg.get('baseline_patient_residual',False)
        if type(residual) is not bool:raise ValueError('Patient residual switch must be boolean')
        # No RNG draw: all pre-existing parameters and the initial prediction stay unchanged.
        self.register_parameter('patient_residual_weight',nn.Parameter(torch.zeros(widths[-1],input_dim)) if residual else None)
        local=cfg.get('baseline_first_layer_residual',False)
        if type(local) is not bool:raise ValueError('First-layer residual switch must be boolean')
        self.register_parameter('first_layer_residual_weight',nn.Parameter(torch.zeros(widths[-1],widths[1])) if local else None)
        if cfg['patient_attention_dropout']!=0:raise ValueError('Reference GAT implementation supports patient_attention_dropout=0 only')
    def forward(self,x,edges,subjects):
        own_input=x[subjects] if self.patient_residual_weight is not None else None
        first_patient=None
        for i,layer in enumerate(self.layers):
            x=self.act(layer.forward_dense(x,edges) if self.architecture=='gin' else layer(x,edges))
            if i==0 and self.first_layer_residual_weight is not None:first_patient=x[subjects]
        h=x[subjects]
        self.last_graph_representation=h.detach()
        self.last_residual_representation=None
        if self.patient_residual_weight is not None:
            residual=torch.nn.functional.linear(own_input,self.patient_residual_weight)
            self.last_residual_representation=residual.detach()
            h=h+residual
        self.last_first_layer_representation=None
        self.last_first_layer_residual_representation=None
        if self.first_layer_residual_weight is not None:
            local_residual=torch.nn.functional.linear(first_patient,self.first_layer_residual_weight)
            self.last_first_layer_representation=first_patient.detach()
            self.last_first_layer_residual_representation=local_residual.detach()
            h=h+local_residual
        self.last_representation=h.detach()
        return self.out(self.dropout(h))


def graph_structure(data,architecture):
    if architecture=='gat':return np.asarray(data['edge_index'],dtype=np.int64)
    n=len(data['graph_x']);edges=np.asarray(data['edge_index']);adj=np.zeros((n,n),dtype=np.float32)
    np.add.at(adj,(edges[1],edges[0]),1.)
    return adj
