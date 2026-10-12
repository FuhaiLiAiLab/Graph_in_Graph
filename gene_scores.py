"""Descriptive gene scores from the exact forward pass; no model retraining."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

METHODS = ['attention_raw','assignment_l2','embedding_x_assignment_l2','embedding_x_softmax_sum']
DEFINITIONS = {
    'attention_raw':'head 0 raw attention; per-gene mean incoming/outgoing means over available directions; equal mean over all active gene layers',
    'assignment_l2':'L2 norm over K of raw signed assignment weights for each gene; identical across subjects within checkpoint',
    'embedding_x_assignment_l2':'post-activation last-gene-layer embedding L2 norm times raw assignment column L2 norm; Frobenius norm of that gene outer-product contribution at pooling',
    'embedding_x_softmax_sum':'post-activation last-gene-layer embedding L2 norm times sum over K of gene-axis softmax assignment; legacy descriptive comparator, not actual forward weights',
}
CLASS_NAMES = {0:'t2ds',1:'pret2ds',2:'no_t2ds'}


def pooling_scores(embedding_norm,assignment):
    norms=np.asarray(embedding_norm,dtype=np.float64)
    weights=np.asarray(assignment,dtype=np.float64)
    if norms.ndim!=2 or weights.ndim!=2 or norms.shape[1]!=weights.shape[1] or not np.isfinite(norms).all() or not np.isfinite(weights).all() or (norms<0).any():
        raise ValueError('Expected finite nonnegative N x G norms and finite K x G assignment')
    magnitude=np.linalg.norm(weights,axis=0)
    soft=np.exp(weights-weights.max(axis=1,keepdims=True))
    soft/=soft.sum(axis=1,keepdims=True)
    return {'assignment_l2':np.broadcast_to(magnitude,norms.shape),
            'embedding_x_assignment_l2':norms*magnitude,
            'embedding_x_softmax_sum':norms*soft.sum(axis=0)}


def attention_node_scores(weights,edges,num_genes):
    weights=np.asarray(weights,dtype=np.float64)
    edges=np.asarray(edges,dtype=np.int64)
    if edges.ndim!=2 or edges.shape[0]!=2 or weights.ndim!=2 or weights.shape[1]!=edges.shape[1] or not np.isfinite(weights).all():
        raise ValueError('Expected B x E attention and 2 x E edges')
    if edges.size and (edges.min()<0 or edges.max()>=num_genes):
        raise ValueError('Edge outside gene universe')
    out_degree=np.bincount(edges[0],minlength=num_genes)
    in_degree=np.bincount(edges[1],minlength=num_genes)
    directions=(out_degree>0).astype(int)+(in_degree>0).astype(int)
    result=np.full((len(weights),num_genes),np.nan)
    for row,values in enumerate(weights):
        outgoing=np.bincount(edges[0],weights=values,minlength=num_genes)/np.maximum(out_degree,1)
        incoming=np.bincount(edges[1],weights=values,minlength=num_genes)/np.maximum(in_degree,1)
        np.divide(outgoing+incoming,directions,out=result[row],where=directions>0)
    return result


class GeneScoreCollector:
    def __init__(self,model,data):
        if model.graph_opt=='subject':
            raise ValueError('No gene scores for phenotype-only model')
        self.model,self.data=model,data
        self.n,self.g=data['num_subject'],data['num_gene_node']
        self.edges=np.asarray(data['gene_edge_index'],dtype=np.int64)
        self.eligible=np.bincount(self.edges.reshape(-1),minlength=self.g)>0
        self.assignment=model.node_weight_assign.detach().cpu().numpy().copy()
        self.embedding_norm=np.full((self.n,self.g),np.nan)
        self.attention=np.zeros((self.n,self.g),dtype=np.float64)
        if __package__:
            from .analyze import analysis_layers
        else:
            from analyze import analysis_layers
        layers, _ = analysis_layers(model)
        if not layers: raise ValueError('No active gene layers')
        self.cursors={layer:0 for layer in layers}
        self.last_gene_layer=list(layers)[-1]

    def collect_gene(self,layer,edges,weights,batch_size,values):
        start=self.cursors[layer]; stop=start+batch_size
        count=self.edges.shape[1]
        if stop>self.n or weights.ndim!=2 or weights.shape[0]!=batch_size*count or weights.shape[1]<1:
            raise ValueError('Gene scoring requires aligned edge/head batches')
        for batch in range(batch_size):
            if not np.array_equal(edges[:,batch*count:(batch+1)*count]-batch*self.g,self.edges):
                raise ValueError('Gene edge ordering differs during scoring')
        self.attention[start:stop]+=attention_node_scores(weights[:,0].reshape(batch_size,count),self.edges,self.g)/len(self.cursors)
        if layer==self.last_gene_layer:
            # Convolution hooks receive pre-activation values. Apply the same activation
            # on a detached tensor; return the untouched original output to the model.
            h=self.model.act(values.detach()).reshape(batch_size,self.g,self.model.gene_embedding_dim)
            self.embedding_norm[start:stop]=np.linalg.norm(h.double().cpu().numpy(),axis=-1)
        self.cursors[layer]=stop

    def finish_arrays(self):
        if any(count!=self.n for count in self.cursors.values()) or not np.isfinite(self.embedding_norm).all():
            raise ValueError('Incomplete scoring pass')
        arrays={'attention_raw':self.attention,**pooling_scores(self.embedding_norm,self.assignment)}
        for values in arrays.values():
            if not np.isfinite(values[:,self.eligible]).all():
                raise ValueError('Nonfinite score for an eligible gene')
        return arrays

    def export(self,output,data_root,payload,checkpoint,save_subjects=False):
        output=Path(output)
        genes=pd.read_csv(Path(data_root)/'metadata/genes.csv')
        if len(genes)!=self.g or not np.array_equal(genes.gene_index,np.arange(self.g)) or genes.gene_name.duplicated().any():
            raise ValueError('Gene metadata is not aligned')
        if type(payload.get('seed')) is not int or payload.get('fold') not in range(1,6):
            raise ValueError('Scoring requires saved seed and fold identities')
        condition=payload.get('ablation') or ('full' if self.model.graph_opt=='GinG' else 'gene_only')
        arrays=self.finish_arrays()
        labels=np.asarray(self.data['labels'])
        groups={'all':np.ones(self.n,dtype=bool),**{name:labels==index for index,name in CLASS_NAMES.items()}}
        pieces=[]
        for name,mask in tqdm(groups.items(),desc='Export gene score groups',unit='group',ascii=True):
            if not mask.any():
                continue
            for method in METHODS:
                values=arrays[method][mask].mean(axis=0)
                rank=np.full(self.g,np.nan)
                rank[self.eligible]=pd.Series(values[self.eligible]).rank(ascending=False,method='average').to_numpy()
                pieces.append(pd.DataFrame(dict(Fold=payload['fold'],Seed=payload['seed'],Condition=condition,
                    Class=name,Method=method,Gene_Index=genes.gene_index,Gene=genes.gene_name,
                    Score=values,Eligible=self.eligible,n_subjects=int(mask.sum()),Rank=rank)))
        pd.concat(pieces,ignore_index=True).to_csv(output/'gene_scores.csv',index=False)
        pd.DataFrame({'subject':self.data['subject_ids'],'label':labels}).to_csv(output/'scoring_subjects.csv',index=False)
        if save_subjects:
            np.savez_compressed(output/'subject_gene_scores.npz',subject_ids=self.data['subject_ids'],labels=labels,
                                gene_indices=np.arange(self.g),eligible=self.eligible,**arrays)
        metadata=dict(schema_version=1,status='complete',data_signature=payload['data_signature'],
            model_config=payload['model_config'],condition=condition,fold=payload['fold'],seed=payload['seed'],
            methods=METHODS,score_definitions=DEFINITIONS,eligible_gene_indices=np.flatnonzero(self.eligible).tolist(),
            score_head=0,gene_layers=list(self.cursors),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            graph_note='incident genes are the common ranking universe; isolated genes have undefined attention, not zero importance',
            aggregation='mean of individual scores within true-label group, then descending average-tie ranks',
            per_subject_export=save_subjects,
            limitations=['Embedding magnitude is not RNA abundance; it includes active omics and neighbor information.',
                         'All scores are descriptive, not signed class-specific output attribution or causal effects.',
                         'Raw attention is partly determined by graph degree; stability alone does not imply biological validity.',
                         'Assignment-only scores are identical across groups within a checkpoint.',
                         'Validation folds also selected epochs/hyperparameters; not independent final evaluation.'])
        (output/'gene_scoring_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        return metadata
