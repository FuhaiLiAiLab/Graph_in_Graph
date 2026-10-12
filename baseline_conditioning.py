"""Train-subject-only centering; the fitted state travels with every checkpoint."""
import hashlib
import json
import torch


def fit_centering(x, subjects, phenotype_dim, enabled=False, pca_components=0, pca_seed=2026, pca_mode="joint"):
    if type(enabled) is not bool:
        raise ValueError('baseline_center_omics must be bool')
    if not 0 < phenotype_dim < x.shape[1] or subjects.numel() == 0:
        raise ValueError('Invalid centering dimensions')
    if type(pca_components) is not int or pca_components < 0:
        raise ValueError('PCA components must be nonnegative integer')
    if pca_mode not in ('joint','rna32_methyl32'):raise ValueError('Unknown PCA mode')
    if pca_mode=='rna32_methyl32':
        if pca_components!=64:raise ValueError('Split PCA requires32+32=64 components')
        return fit_split_pca(x,subjects,phenotype_dim,enabled,pca_seed)
    if pca_components:
        return fit_pca(x, subjects, phenotype_dim, enabled, pca_components, pca_seed)
    mean = x[subjects, phenotype_dim:].mean(0) if enabled else x.new_zeros(x.shape[1]-phenotype_dim)
    if not torch.isfinite(mean).all():
        raise ValueError('Nonfinite training mean')
    return dict(enabled=enabled, phenotype_dim=phenotype_dim, omics_mean=mean.detach().cpu(), fit_scope='training subjects only')


def apply_centering(x, subjects, state):
    if state.get('pca_components', 0):
        return apply_pca(x, subjects, state)
    split = state['phenotype_dim']
    if x.shape[1]-split != len(state['omics_mean']):
        raise ValueError('Centering width differs')
    if not state['enabled']:
        return x
    out = x.clone()
    out[subjects, split:] = x[subjects, split:] - state['omics_mean'].to(device=x.device, dtype=x.dtype)
    return out


def state_fingerprint(state):
    if state.get('pca_components', 0):
        h=hashlib.sha256()
        for key in sorted(state):
            value=state[key]
            h.update(key.encode())
            if isinstance(value, torch.Tensor):
                v=value.detach().cpu().contiguous()
                h.update(json.dumps([str(v.dtype),list(v.shape)]).encode());h.update(v.numpy().tobytes())
            else:h.update(json.dumps(value,sort_keys=True).encode())
        return h.hexdigest()
    mean = state['omics_mean'].detach().cpu().contiguous()
    header = dict(enabled=state['enabled'], phenotype_dim=state['phenotype_dim'], fit_scope=state['fit_scope'], dtype=str(mean.dtype), shape=list(mean.shape))
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode()+mean.numpy().tobytes()).hexdigest()


def representation_stats(model):
    h = model.last_representation
    variance = h.var(dim=0, unbiased=False)
    result=dict(constant_fraction=float((variance == 0).float().mean()),
                near_constant_fraction=float((variance <= 1e-12).float().mean()),
                mean_variance=float(variance.mean()),
                zero_fraction=float((h == 0).float().mean()))
    residual=getattr(model,'last_residual_representation',None)
    if residual is not None:
        graph=model.last_graph_representation
        result.update(graph_mean_l2=float(graph.norm(dim=1).mean()),residual_mean_l2=float(residual.norm(dim=1).mean()),graph_constant_fraction=float((graph.var(dim=0,unbiased=False)==0).float().mean()))
    local=getattr(model,'last_first_layer_residual_representation',None)
    if local is not None:
        first=model.last_first_layer_representation;graph=model.last_graph_representation
        result.update(first_layer_mean_l2=float(first.norm(dim=1).mean()),local_residual_mean_l2=float(local.norm(dim=1).mean()),local_residual_constant_fraction=float((local.var(0,unbiased=False)==0).float().mean()),graph_mean_l2=float(graph.norm(dim=1).mean()),graph_constant_fraction=float((graph.var(0,unbiased=False)==0).float().mean()))
    return result


def fit_pca(x, subjects, phenotype_dim, legacy_centering, components, seed):
    """Only training patient rows enter PCA; no labels or validation are accepted."""
    from sklearn.decomposition import PCA
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid PCA random seed')
    if components > min(subjects.numel()-1, x.shape[1]-phenotype_dim):
        raise ValueError('PCA components exceed training sample/feature rank')
    values=x[subjects,phenotype_dim:].detach().cpu().numpy().copy()
    if not torch.isfinite(torch.from_numpy(values)).all():raise ValueError('Nonfinite PCA training data')
    pca=PCA(n_components=components,svd_solver='randomized',random_state=seed,iterated_power=7,n_oversamples=10,whiten=False)
    pca.fit(values)
    state=dict(enabled=True,legacy_centering=legacy_centering,phenotype_dim=phenotype_dim,
        pca_components=components,pca_seed=seed,fit_scope='training subjects only',
        solver='randomized',iterated_power=7,n_oversamples=10,whiten=False,scale=False,
        n_training_subjects=int(subjects.numel()),
        omics_mean=torch.from_numpy(pca.mean_.copy()),components=torch.from_numpy(pca.components_.copy()),
        explained_variance_ratio=torch.from_numpy(pca.explained_variance_ratio_.copy()),
        singular_values=torch.from_numpy(pca.singular_values_.copy()))
    if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
        raise ValueError('Nonfinite fitted PCA state')
    return state


def apply_pca(x, subjects, state):
    split=state['phenotype_dim'];k=state['pca_components'];mean=state['omics_mean'];components=state['components']
    if x.shape[1]-split!=mean.numel() or components.shape!=(k,mean.numel()):raise ValueError('PCA width differs')
    out=x.new_zeros((x.shape[0],split+k));out[:,:split]=x[:,:split]
    # Auxiliary nodes retain zero omics, rather than projecting artificial zero rows.
    out[subjects,split:]=(x[subjects,split:]-mean.to(x)) @ components.to(x).T
    return out


def split_omics_columns(width):
    if type(width) is not int or width<=0 or width%6:raise ValueError('Expected gene-major six-channel omics')
    columns=torch.arange(width,dtype=torch.long)
    return columns[columns%6==0],columns[columns%6!=0]


def fit_split_pca(x,subjects,phenotype_dim,legacy_centering,seed):
    """Separate RNA32 and methyl32, retain full-width projection for shared inference."""
    import numpy as np
    from sklearn.decomposition import PCA
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid PCA random seed')
    width=x.shape[1]-phenotype_dim;rna,methyl=split_omics_columns(width)
    if min(subjects.numel()-1,len(rna),len(methyl))<32:raise ValueError('Insufficient training rank for split PCA32')
    values=x[subjects,phenotype_dim:].detach().cpu().numpy().copy()
    if not np.isfinite(values).all():raise ValueError('Nonfinite split PCA data')
    total_variance=float(np.var(values.astype(np.float64),axis=0,ddof=1).sum())
    if not total_variance>0:raise ValueError('Zero split PCA variance')
    mean=torch.empty(width,dtype=x.dtype);components=torch.zeros((64,width),dtype=x.dtype);ratios=[];singular=[];blocks={}
    for label,columns,start in [('rna',rna,0),('methyl',methyl,32)]:
        block=values[:,columns.numpy()]
        pca=PCA(n_components=32,svd_solver='randomized',random_state=seed,iterated_power=7,n_oversamples=10,whiten=False).fit(block)
        mean[columns]=torch.from_numpy(pca.mean_.copy());components[start:start+32,columns]=torch.from_numpy(pca.components_.copy())
        ratios.append(torch.from_numpy(pca.explained_variance_.astype(np.float64)/total_variance));singular.append(torch.from_numpy(pca.singular_values_.copy()))
        blocks[label+'_columns']=columns;blocks[label+'_explained_variance_ratio']=torch.from_numpy(pca.explained_variance_ratio_.copy())
        blocks[label+'_input_variance']=float(np.var(block.astype(np.float64),axis=0,ddof=1).sum())
    state=dict(enabled=True,legacy_centering=legacy_centering,phenotype_dim=phenotype_dim,pca_components=64,pca_seed=seed,pca_mode='rna32_methyl32',fit_scope='training subjects only',solver='randomized',iterated_power=7,n_oversamples=10,whiten=False,scale=False,n_training_subjects=int(subjects.numel()),omics_mean=mean,components=components,explained_variance_ratio=torch.cat(ratios),singular_values=torch.cat(singular),total_input_variance=total_variance,**blocks)
    if any(not torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):raise ValueError('Nonfinite split PCA state')
    return state
