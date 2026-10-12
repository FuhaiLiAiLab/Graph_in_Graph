"""Descriptive gene-rank stability across held-out folds and training seeds."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm

METHODS = {'attention_raw','assignment_l2','embedding_x_assignment_l2','embedding_x_softmax_sum'}
CLASSES = {'all','t2ds','pret2ds','no_t2ds'}
GROUP = ['Class','Method']
GENE = ['Gene_Index','Gene']


def rank_scores(scores):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Eligible gene scores must be a nonempty finite vector')
    return rankdata(-scores,method='average')


def rank_correlation(left,right):
    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.shape!=right.shape or left.ndim!=1:
        raise ValueError('Rank vectors must have matching one-dimensional shapes')
    if len(left)<2 or np.ptp(left)==0 or np.ptp(right)==0:
        return float('nan')
    left=left-left.mean(); right=right-right.mean()
    # Pearson correlation of midranks = Spearman. Elementwise reductions avoid
    # loading a second BLAS/OpenMP runtime when used beside PyTorch on Windows.
    denominator=np.sqrt(np.sum(left*left)*np.sum(right*right))
    return float(np.clip(np.sum(left*right)/denominator,-1,1))


def fractional_topk(ranks,k):
    """Allocate exactly min(K,G) membership mass, splitting cutoff ties equally."""
    ranks=np.asarray(ranks,dtype=float)
    if k<1 or not len(ranks) or not np.isfinite(ranks).all():
        raise ValueError('Expected positive K and finite nonempty ranks')
    effective=min(k,len(ranks))
    cutoff=np.sort(ranks)[effective-1]
    above=ranks<cutoff
    boundary=ranks==cutoff
    fraction=(effective-int(above.sum()))/int(boundary.sum())
    weights=above.astype(float)+boundary*fraction
    return weights,dict(effective_k=effective,boundary_tie_size=int(boundary.sum()),
                        boundary_fraction=float(fraction),trivial_topk=k>=len(ranks))


def load_runs(directories,allow_incomplete=False):
    runs=[]
    reference=None
    gene_reference=None
    seen=set()
    rosters={}
    required={'Fold','Seed','Condition','Class','Method','Gene_Index','Gene','Score','Eligible','n_subjects'}
    identity_keys=['data_signature','model_config','condition','methods','score_definitions','score_head','graph_note']
    for directory in tqdm(directories,desc='Load gene rankings',unit='run',ascii=True):
        folder=Path(directory)
        meta=json.loads((folder/'gene_scoring_metadata.json').read_text(encoding='utf-8'))
        if meta.get('schema_version')!=1 or any(k not in meta for k in identity_keys+['fold','seed','eligible_gene_indices']):
            raise ValueError(f'Invalid scoring metadata: {folder}')
        if set(meta['methods'])!=METHODS or len(meta['methods'])!=len(METHODS) or meta['score_head']!=0:
            raise ValueError(f'Unexpected scoring methods/head in metadata: {folder}')
        identity={k:meta[k] for k in identity_keys}
        identity['methods']=sorted(identity['methods'])
        if reference is None: reference=identity
        if identity!=reference: raise ValueError(f'Incompatible scoring metadata: {folder}')
        fold,seed=meta['fold'],meta['seed']
        if type(fold) is not int or fold not in range(1,6) or type(seed) is not int or not 0<=seed<2**32:
            raise ValueError('Invalid fold/seed identity')
        if (fold,seed) in seen: raise ValueError('Duplicate fold/seed run')
        seen.add((fold,seed))
        frame=pd.read_csv(folder/'gene_scores.csv')
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f'Missing score columns: {folder}')
        if frame[['Fold','Seed','Condition','Class','Method','Gene_Index','Gene','Eligible','n_subjects']].isna().any().any():
            raise ValueError(f'Missing score identity values: {folder}')
        if not ((frame.Fold==fold)&(frame.Seed==seed)&(frame.Condition==meta['condition'])).all():
            raise ValueError('Score rows disagree with metadata identity')
        if set(frame.Class)!=CLASSES or set(frame.Method)!=METHODS or frame.duplicated(GROUP+['Gene_Index']).any():
            raise ValueError('Invalid class/method/gene grid')
        if not frame.Eligible.isin([True,False]).all():
            raise ValueError('Eligible must be boolean')
        frame['Eligible']=frame.Eligible.astype(bool)
        if not np.equal(frame.Gene_Index,frame.Gene_Index.astype(int)).all() or (frame.Gene_Index<0).any():
            raise ValueError('Gene indices must be nonnegative integers')
        frame['Gene_Index']=frame.Gene_Index.astype(int)
        frame['Gene']=frame.Gene.astype(str)
        for _,group in frame.groupby(GROUP):
            genes=group[GENE+['Eligible']].sort_values('Gene_Index').reset_index(drop=True)
            if genes.Gene.duplicated().any(): raise ValueError('Duplicate gene names')
            if gene_reference is None: gene_reference=genes
            if not genes.equals(gene_reference): raise ValueError('Inconsistent gene identity/eligibility universe')
        if len(frame.groupby(GROUP))!=len(CLASSES)*len(METHODS): raise ValueError('Incomplete class/method grid')
        eligible=gene_reference.loc[gene_reference.Eligible,'Gene_Index'].tolist()
        if not eligible or sorted(meta['eligible_gene_indices'])!=eligible:
            raise ValueError('Eligible genes disagree with metadata')
        roster=pd.read_csv(folder/'scoring_subjects.csv')
        if not {'subject','label'}.issubset(roster.columns) or roster.empty or roster[['subject','label']].isna().any().any() or roster.subject.duplicated().any() or not roster.label.isin([0,1,2]).all():
            raise ValueError('Invalid held-out subject roster')
        roster=roster[['subject','label']].sort_values('subject').reset_index(drop=True)
        if fold in rosters and not roster.equals(rosters[fold]): raise ValueError('Within-fold subject rosters differ across seeds')
        rosters[fold]=roster
        for class_name,label in [('all',None),('t2ds',0),('pret2ds',1),('no_t2ds',2)]:
            count=len(roster) if label is None else int((roster.label==label).sum())
            if not (frame.loc[frame.Class==class_name,'n_subjects']==count).all():
                raise ValueError('Score subject counts disagree with roster')
            if count==0:
                raise ValueError('Every scoring class must have subjects for ranking comparison')
        selected=frame[frame.Eligible].copy()
        selected['Rank']=selected.groupby(GROUP)['Score'].transform(lambda x:rank_scores(x.to_numpy()))
        runs.append(dict(folder=str(folder),metadata=meta,rankings=selected.sort_values(GROUP+['Gene_Index']),roster=roster))
    if not runs: raise ValueError('No gene score files found')
    folds=sorted(rosters)
    for a,b in itertools.combinations(folds,2):
        if set(rosters[a].subject)&set(rosters[b].subject): raise ValueError('Cross-fold subject overlap')
    seedsets={f:{s for ff,s in seen if ff==f} for f in folds}
    if len({tuple(sorted(s)) for s in seedsets.values()})!=1:
        raise ValueError('Every fold must contain the same seed set')
    if not allow_incomplete and (folds!=[1,2,3,4,5] or any(len(s)!=5 for s in seedsets.values())):
        raise ValueError('Require five folds and five distinct seeds per fold; use --allow-incomplete for diagnostics')
    return runs


def compare_runs(runs,top_k):
    top_k=sorted(set(top_k))
    if not top_k or min(top_k)<1: raise ValueError('Top-K values must be positive')
    rankings=pd.concat([r['rankings'] for r in runs],ignore_index=True)
    tables={}
    resolutions=[]
    memberships=[]
    grouped=rankings.groupby(['Fold','Seed']+GROUP,sort=True)
    for identity,group in tqdm(grouped,total=grouped.ngroups,desc='Build tie-aware rankings',unit='ranking',ascii=True):
        group=group.sort_values('Gene_Index')
        ranks=group.Rank.to_numpy()
        counts=group.Score.value_counts()
        context=dict(zip(['Fold','Seed']+GROUP,identity))
        resolutions.append(dict(**context,n_genes=len(group),unique_scores=len(counts),
                                tie_fraction=float(counts[counts>1].sum()/len(group)),
                                constant_scores=len(counts)==1))
        tables[identity]=(group,ranks)
        for k in top_k:
            weights,details=fractional_topk(ranks,k)
            for gene,weight,rank in zip(group[GENE].itertuples(index=False,name=None),weights,ranks):
                memberships.append(dict(**context,Gene_Index=gene[0],Gene=gene[1],TopK=k,Membership=weight,Rank=rank,**details))
    pairs=[]
    run_ids=sorted({(r['metadata']['fold'],r['metadata']['seed']) for r in runs})
    candidates=[(a,b) for a,b in itertools.combinations(run_ids,2) if a[0]==b[0] or a[1]==b[1]]
    for a,b in tqdm(candidates,desc='Compare ranking pairs',unit='pair',ascii=True):
        level='same_fold_different_seed' if a[0]==b[0] else 'same_seed_different_fold'
        for cls,method in itertools.product(sorted(CLASSES),sorted(METHODS)):
            left,lr=tables[(*a,cls,method)]
            right,rr=tables[(*b,cls,method)]
            if not left[GENE].reset_index(drop=True).equals(right[GENE].reset_index(drop=True)):
                raise ValueError('Pair gene alignment differs')
            correlation=rank_correlation(lr,rr)
            for k in top_k:
                lw,ld=fractional_topk(lr,k)
                rw,rd=fractional_topk(rr,k)
                pairs.append(dict(level=level,Fold_A=a[0],Seed_A=a[1],Fold_B=b[0],Seed_B=b[1],
                    Class=cls,Method=method,TopK=k,n_genes=len(lr),spearman=correlation,
                    jaccard=float(np.minimum(lw,rw).sum()/np.maximum(lw,rw).sum()),
                    effective_k=ld['effective_k'],trivial_topk=ld['trivial_topk'],
                    boundary_tie_size_A=ld['boundary_tie_size'],boundary_tie_size_B=rd['boundary_tie_size'],
                    boundary_fraction_A=ld['boundary_fraction'],boundary_fraction_B=rd['boundary_fraction'],
                    constant_A=np.ptp(lr)==0,constant_B=np.ptp(rr)==0,
                    informative_topk=not ld['trivial_topk'] and np.ptp(lr)>0 and np.ptp(rr)>0))
    membership=pd.DataFrame(memberships)
    frequencies=membership.groupby(['Fold']+GROUP+GENE+['TopK'],as_index=False).agg(
        frequency=('Membership','mean'),mean_rank=('Rank','mean'),rank_std=('Rank','std'),
        n_seeds=('Seed','nunique'),effective_k=('effective_k','first'),trivial_topk=('trivial_topk','first'),
        mean_boundary_fraction=('boundary_fraction','mean'))
    fold_ranks=rankings.groupby(['Fold']+GROUP+GENE,as_index=False).agg(
        seed_mean_rank=('Rank','mean'),seed_rank_std=('Rank','std'),n_seeds=('Seed','nunique'))
    consensus=fold_ranks.groupby(GROUP+GENE,as_index=False).agg(
        mean_rank=('seed_mean_rank','mean'),between_fold_std=('seed_mean_rank','std'),
        between_fold_min=('seed_mean_rank','min'),between_fold_max=('seed_mean_rank','max'),n_folds=('Fold','nunique'))
    consensus['Consensus_Rank']=consensus.groupby(GROUP).mean_rank.transform(lambda x:rankdata(x,method='average'))
    pairwise=pd.DataFrame(pairs)
    if pairs:
        pairwise['informative_jaccard']=pairwise.jaccard.where(pairwise.informative_topk)
        summary=pairwise.groupby(['level']+GROUP+['TopK'],as_index=False).agg(
            spearman_mean=('spearman','mean'),spearman_std=('spearman','std'),n_defined=('spearman','count'),
            n_pairs=('spearman','size'),jaccard_mean=('informative_jaccard','mean'),jaccard_std=('informative_jaccard','std'),
            raw_jaccard_mean=('jaccard','mean'),raw_jaccard_std=('jaccard','std'),
            n_informative_topk=('informative_topk','sum'),trivial_topk=('trivial_topk','first'))
    else:
        summary=pd.DataFrame(columns=['level']+GROUP+['TopK','spearman_mean','n_defined','n_pairs'])
        pairwise=pd.DataFrame(columns=['level','Fold_A','Seed_A','Fold_B','Seed_B']+GROUP+['TopK','spearman','jaccard'])
    return dict(rankings=rankings,pairwise_stability=pairwise,frequencies=frequencies,consensus=consensus,
                fold_consensus=fold_ranks,memberships=membership,ranking_resolution=pd.DataFrame(resolutions),summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',nargs='+',type=Path)
    source.add_argument('--analysis-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--top-k',nargs='+',type=int,default=[20,50,100])
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new stability output directory; existing results are not overwritten')
    directories=args.input if args.input is not None else sorted(p.parent for p in args.analysis_root.rglob('gene_scores.csv'))
    runs=load_runs(directories,args.allow_incomplete)
    outputs=compare_runs(runs,args.top_k)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,frame in tqdm(outputs.items(),desc='Write stability tables',unit='file',ascii=True):
        frame.to_csv(args.output/f'{name}.csv',index=False)
    metadata=dict(schema_version=1,inputs=[r['folder'] for r in runs],n_runs=len(runs),
        data_signature=runs[0]['metadata']['data_signature'],model_config=runs[0]['metadata']['model_config'],
        condition=runs[0]['metadata']['condition'],top_k=sorted(set(args.top_k)),allow_incomplete=args.allow_incomplete,
        ranking='descending average ranks; eligible incident genes only; supplied ranks ignored',
        membership='fractional cutoff ties; total membership=min(K,G); Jaccard=sum(min)/sum(max)',
        overlap_summary='jaccard_mean/std exclude constant-score and K>=G pairs; raw_jaccard_mean/std retain all diagnostics',
        aggregation='seeds averaged within fold, then folds equally averaged; sample SD is descriptive',
        limitations=['Ranks only: raw score scales across checkpoints are not comparable.',
            'Fold/seed comparisons are correlated; no confidence intervals or p-values are inferred.',
            'Stable constants and K>=gene-count are uninformative; no automatic winner recommendation.',
            'Stability is not validity or causal attribution.',
            'Within-fold comparisons hold the scored people fixed; cross-fold comparisons also change people.'])
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),n_runs=len(runs),n_pairs=len(outputs['pairwise_stability'])),indent=2))


if __name__=='__main__':
    main()
