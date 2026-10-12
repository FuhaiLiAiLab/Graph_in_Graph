"""Export explicitly selected Full checkpoints into current RNA-scored Shiny inputs."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifact_io import write_json,sha
from runtime import ROOT,load_run,forward,seed_everything,select_device
from analyze import AttentionCollector
from normalize_edges import process
from rna_scores import SignedRNACollector,CLASSES
from network_average import equal_fold_seed_average


def export_case(checkpoint,data_root,output,device='cpu',batch_size=8):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model,payload,data=load_run(checkpoint,data_root,device)
    if payload.get('selection')!='checkpoint_rule:balanced_guard':raise ValueError('Expected explicitly guard-selected Full checkpoint')
    if payload.get('ablation') not in (None,'full') or model.graph_opt!='GinG':raise ValueError('Full checkpoints required')
    seed_everything(payload['seed'])
    genes=pd.read_csv(Path(data_root)/'metadata/genes.csv')
    categories=pd.read_csv(Path(data_root)/'metadata/subfeature_nodes.csv')
    attention=output/'attention';attention.mkdir()
    with torch.no_grad():plain=forward(model,data,device,batch_size)[1].detach().cpu()
    scorer=SignedRNACollector(model,data)
    collector=AttentionCollector(data,payload['fold'],attention,categories.subfeature_name.tolist(),gene_scorer=scorer).attach(model)
    try:
        with torch.no_grad():hooked=forward(model,data,device,batch_size)[1].detach().cpu()
        torch.testing.assert_close(plain,hooked,rtol=0,atol=0)
        collector.finish()
        scorer.export_signed(output,genes,payload['seed'],payload['fold'])
    finally:collector.remove()
    layers=[f'gene_layer_{i+1:02d}' for i in range(len(model.gene_convs))]
    process([attention],data_root,output/'edges',modes=['raw','rownorm'],expected_layers=layers)
    write_json(output/'case.json',dict(seed=payload['seed'],fold=payload['fold'],epoch=payload['epoch'],checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),hook_logits_unchanged=True,model_config=payload['model_config']))
    return payload,data


def make_shiny_inputs(nodes,edges,genes,output,pvalues):
    output=Path(output)
    required=['TvsNO_min_pvalues.csv','PrevsNO_min_pvalues.csv']
    for name in required:
        if not (Path(pvalues)/name).is_file():raise FileNotFoundError(name)
    for kind in ['single_group','inter']:
        app=output/kind;shutil.copytree(ROOT/'shinyapp'/kind,app)
        dest=app/'analysis/gigtransformer-rownorm';dest.mkdir(parents=True)
        mapping=app/'data/filtered_data';mapping.mkdir(parents=True)
        genes.rename(columns={'gene_index':'gene_node_idx','gene_name':'gene_node_name'}).to_csv(mapping/'gene_num_dict_df.csv',index=False)
        for group in CLASSES.values():
            nodes[nodes.Class==group][['Node_idx','Weight']].to_csv(dest/f'{group}_layer_norm_average_fold_node_weight_df.csv',index=False)
            edges[edges.Class==group][['Actual_From','Actual_To','Weight']].to_csv(dest/f'{group}_layer_norm_average_fold_gene_edge_weight_df.csv',index=False)
        (app/'pvalues_output').mkdir()
        for name in required:shutil.copy2(Path(pvalues)/name,app/'pvalues_output'/name)
        (app/'t6.txt').write_text('Locus\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True,help='JSON list of {seed, fold, checkpoint}; complete five-fold grid per seed')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pvalues',type=Path,required=True,help='Existing annotation files; never fabricated or recomputed silently')
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args();cases=json.loads(args.manifest.read_text());keys=[(int(x['seed']),int(x['fold'])) for x in cases]
    seeds=sorted({s for s,f in keys});folds=list(range(1,6))
    if len(seeds)<2 or len(keys)!=len(set(keys)) or set(keys)!={(s,f) for s in seeds for f in folds}:p.error('Need complete unique five-fold cases for at least two seeds')
    args.output.mkdir(parents=True,exist_ok=False);nodes={};edges={};coverage={s:[] for s in seeds};config=None
    for case in cases:
        key=(case['seed'],case['fold']);target=args.output/f'seed_{key[0]}'/f'fold_{key[1]:02d}'
        cp=Path(case['checkpoint']);cp=cp if cp.is_absolute() else args.manifest.parent/cp
        payload,data=export_case(cp,args.data_root,target,str(select_device(args.device)),args.batch_size)
        if (payload['seed'],payload['fold'])!=key:raise ValueError('Manifest and checkpoint identity differ')
        if config is not None and config!=payload['model_config']:raise ValueError('Model configurations differ')
        config=payload['model_config'];coverage[key[0]].extend(data['subject_ids'].tolist())
        nodes[key]=pd.read_csv(target/'rna_gene_scores.csv');edges[key]=pd.read_csv(target/'edges/rownorm/gene_edges.csv')
    reference=None
    for ids in coverage.values():
        if len(ids)!=len(set(ids)):raise ValueError('Repeated held-out subjects within a seed')
        if reference is not None and set(ids)!=reference:raise ValueError('Seed cohort coverage differs')
        reference=set(ids)
    _,n=equal_fold_seed_average(nodes,['Class','Node_idx'],seeds,folds)
    _,e=equal_fold_seed_average(edges,['Class','Head','Actual_From','Actual_To'],seeds,folds)
    genes=pd.read_csv(args.data_root/'metadata/genes.csv');n['Gene']=genes.gene_name.to_numpy()[n.Node_idx.to_numpy()]
    n.to_csv(args.output/'gene_nodes.csv',index=False);e.to_csv(args.output/'gene_edges.csv',index=False)
    make_shiny_inputs(n,e,genes,args.output/'shiny',args.pvalues)
    write_json(args.output/'provenance.json',dict(seeds=seeds,folds=folds,method='embedding_norm * assignment_column_norm * signed_model_input_RNA',aggregation='equal folds then equal seeds',edges='mean gene-layer attention divided by outgoing degree',pvalues='external annotations copied unchanged',external_evidence='disabled through empty t6 Locus table'))
if __name__=='__main__':main()
