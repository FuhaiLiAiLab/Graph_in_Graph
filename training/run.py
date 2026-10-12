"""Fixed-configuration training; no parameter or seed search."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from runtime import ROOT,select_device,load_run,forward
from train import run_fold,save_predictions,metrics
from artifact_io import write_json

MODELS=('full','phenotype_only','gene_only','rna_only','methylation_only','gat','gin')

def task_grid(models,seeds,folds):
    if any(x not in MODELS for x in models) or not models or len(set(models))!=len(models):raise ValueError('Invalid model list')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 or s>=2**32 for s in seeds):raise ValueError('Invalid unique uint32 seeds')
    if not folds or len(set(folds))!=len(folds) or any(f not in range(1,6) for f in folds):raise ValueError('Invalid folds')
    return [(m,s,f) for m in models for s in seeds for f in folds]

def execute(model,seed,fold,args):
    device=str(select_device(args.device));folder=args.output/model/f'seed_{seed}'/f'fold_{fold:02d}'
    config_name=model if model in ('full','gat','gin') else 'ablation'
    cfg=json.loads((ROOT/f'configs/{config_name}.json').read_text())
    if model in ('gat','gin'):
        from baseline_train import train_fold
        if folder.exists() and any(folder.iterdir()) and not args.resume:raise ValueError('Output exists: use --resume with the same contract')
        train_fold(model,cfg,fold,seed,args.data_root,folder,device,epochs=args.epochs)
    else:
        policy=json.loads((ROOT/'configs/full_training.json').read_text())
        # A short software smoke run is allowed; production defaults remain exactly 500.
        policy['scheduler_horizon']=args.epochs
        policy['early_stopping']['min_epochs']=min(policy['early_stopping']['min_epochs'],args.epochs)
        run_fold(cfg,fold,folder.parent,args.data_root,device,args.epochs,seed,args.batch_size,
                 ablation=model,resume=args.resume,training_policy=policy,checkpoint_rules=['balanced_guard'])
        checkpoint=folder/'rule_balanced_guard.pt'
        net,payload,data=load_run(checkpoint,args.data_root,device)
        with torch.no_grad():logits=forward(net,data,device,args.batch_size)[1]
        observed,_=metrics(data['labels'],logits)
        for name,value in payload['metrics'].items():
            if value is not None and abs(observed[name]-value)>1e-6:raise ValueError('Reloaded selected checkpoint metrics differ')
        save_predictions(folder/'selected_predictions.csv',data,logits)
        write_json(folder/'selected_checkpoint.json',dict(checkpoint=checkpoint.name,epoch=payload['epoch'],metrics=observed,selection='balanced_guard'))

def main(default_models=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=default_models or ['full'])
    p.add_argument('--seeds',nargs='+',type=int,default=None)
    p.add_argument('--folds',nargs='+',type=int,default=None)
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--data-root',type=Path,default=ROOT/'processed_data')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--task-index',type=int,help='Run one zero-based task, suitable for independent Slurm array tasks')
    args=p.parse_args();plan=json.loads((ROOT/'configs/evaluation.json').read_text())
    tasks=task_grid(args.models,args.seeds if args.seeds is not None else plan['seeds'],args.folds if args.folds is not None else plan['folds'])
    if args.task_index is not None:
        if not 0<=args.task_index<len(tasks):p.error('task-index outside grid')
        tasks=[tasks[args.task_index]]
    for model,seed,fold in tasks:
        print(json.dumps(dict(event='training',model=model,seed=seed,fold=fold)),flush=True)
        execute(model,seed,fold,args)
    print('All requested training tasks complete.',flush=True)
if __name__=='__main__':main()
