"""Fixed-parameter standalone GAT/GIN, auditable metrics and exact state resume."""
import csv,gc,hashlib,json,math,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from baseline_models import BaselineClassifier,joint_input,graph_structure
from baseline_conditioning import fit_centering,apply_centering,state_fingerprint,representation_stats
from runtime import ROOT,load_pair,apply_graph_policy,patient_graph_summary,data_signature,seed_everything,training_class_weights
from train import metrics,save_predictions
from training_state import atomic_torch_save,cpu_copy,capture_rng,restore_rng,build_optimization
from checkpoint_rules import rule_score
from artifact_io import write_json,sha,audit
import fcntl

SOURCE_NAMES=['artifact_io.py','baseline_models.py','baseline_train.py','baseline_conditioning.py','enc/geo_gat.py','enc/geo_gin.py','runtime.py','model_config.py','train.py','training_state.py','training_policy.py','checkpoint_rules.py','processed_data/scripts/load_fold.py']
def fingerprints():return {n:sha(ROOT/n) for n in SOURCE_NAMES}

def verify_result(folder,contract=None):
    folder=Path(folder);p=folder/'summary.json'
    if not p.exists():return None
    row=json.loads(p.read_text())
    if contract is not None and row['contract']!=contract:raise ValueError('Completed contract differs')
    if not row['training_complete']:return None
    if row['checkpoint_sha256']!=sha(folder/'rule_balanced_guard.pt') or row['predictions_sha256']!=sha(folder/'test_predictions.csv'):raise ValueError('Completed artifacts changed')
    if row['preprocessing_sha256']!=sha(folder/'preprocessing.pt') or row['train_predictions_sha256']!=sha(folder/'train_predictions.csv'):raise ValueError('Preprocessing/train predictions changed')
    return row


def validate_early_stopping(policy,epochs):
    if policy is None:return None
    if not isinstance(policy,dict) or set(policy)!={'min_epochs','patience'}:raise ValueError('Invalid early stopping policy')
    if any(type(policy[k]) is not int or policy[k]<1 for k in policy):raise ValueError('Invalid early stopping integers')
    if policy['min_epochs']>epochs:raise ValueError('Early stopping minimum exceeds budget')
    return dict(policy)

def stop_due(epoch,best_epoch,policy):
    return policy is not None and epoch>=policy['min_epochs'] and best_epoch>0 and epoch-best_epoch>=policy['patience']

def train_fold(architecture,cfg,fold,seed,data_root,folder,device,epochs=500,checkpoint_interval=10,callback=None,early_stopping=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(architecture,cfg,fold,seed,Path(data_root),folder,device,epochs,checkpoint_interval,callback,early_stopping)

def _train(architecture,cfg,fold,seed,data_root,folder,device,epochs,checkpoint_interval,callback,early_stopping):
    if epochs<1 or checkpoint_interval<1:raise ValueError('Invalid epoch budget')
    early_stopping=validate_early_stopping(early_stopping,epochs)
    reproducibility=seed_everything(seed)
    tr,va=load_pair(fold,data_root);tr=apply_graph_policy(tr,cfg['graph_policy']);va=apply_graph_policy(va,cfg['graph_policy'])
    import importlib.metadata
    contract=dict(architecture=architecture,config=cfg,fold=fold,seed=seed,epochs=epochs,checkpoint_interval=checkpoint_interval,checkpoint_rule='balanced_guard',data_signature=data_signature(data_root),source_sha256=fingerprints(),reproducibility=reproducibility,device=device,versions={n:importlib.metadata.version(n) for n in ['torch','torch-geometric','numpy','scikit-learn']},patient_graph={r:patient_graph_summary(d) for r,d in [('train',tr),('validation',va)]})
    if early_stopping is not None:contract['early_stopping']=early_stopping
    saved_config=folder/'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text())!=contract:raise ValueError('Resume contract differs')
    complete=verify_result(folder,contract)
    if complete is not None:return complete
    write_json(saved_config,contract)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    def tensors(d):return (torch.as_tensor(joint_input(d),device=device),torch.as_tensor(graph_structure(d,architecture),device=device),torch.as_tensor(d['node_index'],dtype=torch.long,device=device))
    raw_train,raw_val=tensors(tr),tensors(va)
    if cfg.get('baseline_pca_mode','joint')=='rna32_methyl32' and tr['num_feature']!=6:raise ValueError('Split PCA requires six gene channels')
    input_transform=fit_centering(raw_train[0],raw_train[2],tr['graph_x'].shape[1],cfg.get('baseline_center_omics',False),cfg.get('baseline_pca_components',0),cfg.get('baseline_pca_seed',2026),cfg.get('baseline_pca_mode','joint'))
    transform_sha=state_fingerprint(input_transform)
    preprocessing_file=folder/'preprocessing.pt'
    if preprocessing_file.exists():
        stored=torch.load(preprocessing_file,map_location='cpu',weights_only=True)
        if state_fingerprint(stored)!=transform_sha:raise ValueError('Stored training centering differs')
    else:atomic_torch_save(input_transform,preprocessing_file)
    def apply_inputs(raw,state):return (apply_centering(raw[0],raw[2],state),raw[1],raw[2])
    train_inputs,val_inputs=apply_inputs(raw_train,input_transform),apply_inputs(raw_val,input_transform)
    labels=torch.as_tensor(tr['labels'],dtype=torch.long,device=device)
    model=BaselineClassifier(architecture,train_inputs[0].shape[1],cfg).to(device)
    optimizer,scheduler=build_optimization(model,cfg,epochs)
    weights=training_class_weights(tr['labels'],device,cfg['class_frequency_exponent'],cfg['majority_class_multiplier'],cfg['class_weighting'])
    history=[];best=None;best_score=-math.inf;start_epoch=1;resume_count=0;prior_elapsed=0.;started=time.monotonic()
    last=folder/'last_model.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['contract']!=contract:raise ValueError('Snapshot contract differs')
        if state_fingerprint(saved['input_transform'])!=transform_sha:raise ValueError('Snapshot preprocessing differs')
        if [r['epoch'] for r in saved['history']]!=list(range(1,saved['epoch']+1)):raise ValueError('Snapshot history differs')
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
        if scheduler is not None:scheduler.load_state_dict(saved['scheduler_state'])
        history=saved['history'];best=saved['best'];best_score=saved['best_score'];start_epoch=saved['epoch']+1;resume_count=saved['resume_count']+1;prior_elapsed=saved['elapsed_seconds'];restore_rng(saved['rng_state'])
        if best is not None:atomic_torch_save(best,folder/'rule_balanced_guard.pt')
        print(json.dumps(dict(event='RESUME',architecture=architecture,seed=seed,fold=fold,saved_epoch=saved['epoch'],next_epoch=start_epoch)),flush=True)
    def snapshot(epoch):
        atomic_torch_save(dict(contract=contract,input_transform=cpu_copy(input_transform),epoch=epoch,model_state=cpu_copy(model.state_dict()),optimizer_state=cpu_copy(optimizer.state_dict()),scheduler_state=cpu_copy(scheduler.state_dict()) if scheduler else None,rng_state=capture_rng(),history=history,best=best,best_score=best_score,resume_count=resume_count,elapsed_seconds=prior_elapsed+time.monotonic()-started),last)
    with (folder/'history.csv').open('w',newline='') as f:
        writer=None
        if history:writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history);f.flush()
        for epoch in range(start_epoch,epochs+1):
            if stop_due(epoch-1,best['epoch'] if best else 0,early_stopping):break
            model.train();optimizer.zero_grad(set_to_none=True);logits=model(*train_inputs)
            loss=cfg['class_weight']*F.cross_entropy(logits,labels,weight=weights)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            first_layer_gradient_norm=sum(float(p.grad.norm()) for p in model.layers[0].parameters() if p.grad is not None)
            residual_parameter=model.patient_residual_weight
            residual_gradient_norm=float(residual_parameter.grad.norm()) if residual_parameter is not None and residual_parameter.grad is not None else 0.
            local_parameter=model.first_layer_residual_weight
            local_gradient_norm=float(local_parameter.grad.norm()) if local_parameter is not None and local_parameter.grad is not None else 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip']);optimizer.step()
            if scheduler:scheduler.step()
            model.eval()
            with torch.no_grad():
                train_metrics=metrics(tr['labels'],model(*train_inputs))[0]
                train_representation=representation_stats(model)
                val_metrics=metrics(va['labels'],model(*val_inputs))[0]
                validation_representation=representation_stats(model)
            score=rule_score(val_metrics,'balanced_guard')
            if score>best_score:
                best_score=score;best=dict(contract=contract,input_transform=cpu_copy(input_transform),model_state=cpu_copy(model.state_dict()),epoch=epoch,validation=val_metrics,train=train_metrics,score=score,train_representation=train_representation,validation_representation=validation_representation)
                atomic_torch_save(best,folder/'rule_balanced_guard.pt')
            row=dict(epoch=epoch,train_loss=float(loss.detach().cpu()),**{'train_'+k:v for k,v in train_metrics.items()},**{'validation_'+k:v for k,v in val_metrics.items()},first_layer_gradient_norm=first_layer_gradient_norm,**{'train_representation_'+k:v for k,v in train_representation.items()},**{'validation_representation_'+k:v for k,v in validation_representation.items()});history.append(row)
            row['patient_residual_gradient_norm']=residual_gradient_norm
            if local_parameter is not None:row['first_layer_residual_gradient_norm']=local_gradient_norm
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);f.flush()
            if callback:
                try:callback(epoch)
                except BaseException:snapshot(epoch);raise
            if epoch==1 or epoch%checkpoint_interval==0 or epoch==epochs:snapshot(epoch)
            if epoch==1 or epoch%25==0:print(json.dumps(dict(architecture=architecture,seed=seed,fold=fold,epoch=epoch,loss=row['train_loss'],BA=val_metrics['balanced_accuracy'],F1=val_metrics['macro_f1'],AP=val_metrics['macro_auprc'])),flush=True)
            if stop_due(epoch,best['epoch'],early_stopping):
                snapshot(epoch)
                print(json.dumps(dict(event='EARLY_STOP',epoch=epoch,best_epoch=best['epoch'],policy=early_stopping)),flush=True)
                break
    # Reload the persisted selected checkpoint for the reported metrics/predictions.
    payload=torch.load(folder/'rule_balanced_guard.pt',map_location='cpu',weights_only=True);model.load_state_dict(payload['model_state']);model.eval()
    if state_fingerprint(payload['input_transform'])!=transform_sha:raise ValueError('Selected preprocessing differs')
    with torch.no_grad():
        measured=save_predictions(folder/'test_predictions.csv',va,model(*apply_inputs(raw_val,payload['input_transform'])))
        measured_train=save_predictions(folder/'train_predictions.csv',tr,model(*apply_inputs(raw_train,payload['input_transform'])))
    for k,v in measured.items():
        if v is not None and not math.isclose(v,payload['validation'][k],rel_tol=1e-6,abs_tol=1e-7):raise ValueError('Reloaded selected-checkpoint metric mismatch')
    for k,v in measured_train.items():
        if v is not None and not math.isclose(v,payload['train'][k],rel_tol=1e-6,abs_tol=1e-7):raise ValueError('Reloaded training metric mismatch')
    result=dict(contract=contract,architecture=architecture,seed=seed,fold=fold,training_complete=True,completed_epochs=history[-1]['epoch'],requested_epochs=epochs,stop_reason='early_stopping' if history[-1]['epoch']<epochs else 'max_epochs',selected_epoch=payload['epoch'],validation=measured,train=payload['train'],resume_count=resume_count,elapsed_seconds=prior_elapsed+time.monotonic()-started,class_weights=weights.detach().cpu().tolist(),parameter_count=sum(p.numel() for p in model.parameters()),input_dim=train_inputs[0].shape[1],n_train=len(tr['labels']),n_validation=len(va['labels']),checkpoint_sha256=sha(folder/'rule_balanced_guard.pt'),predictions_sha256=sha(folder/'test_predictions.csv'),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU')
    result.update(input_transform_fingerprint=transform_sha,preprocessing_sha256=sha(preprocessing_file),train_predictions_sha256=sha(folder/'train_predictions.csv'),train_representation=payload['train_representation'],validation_representation=payload['validation_representation'],last_representation={k:v for k,v in history[-1].items() if 'representation_' in k},last_first_layer_gradient_norm=history[-1]['first_layer_gradient_norm'])
    result.update(pca_components=input_transform.get('pca_components',0), pca_explained_variance_ratio_sum=float(input_transform['explained_variance_ratio'].sum()) if input_transform.get('pca_components',0) else None)
    result.update(pca_mode=input_transform.get('pca_mode','joint'),rna_explained_variance_ratio_sum=float(input_transform['rna_explained_variance_ratio'].sum()) if 'rna_explained_variance_ratio' in input_transform else None,methyl_explained_variance_ratio_sum=float(input_transform['methyl_explained_variance_ratio'].sum()) if 'methyl_explained_variance_ratio' in input_transform else None)
    if model.first_layer_residual_weight is not None:result.update(first_layer_residual_weight_norm=float(model.first_layer_residual_weight.detach().norm()),last_first_layer_residual_gradient_norm=history[-1]['first_layer_residual_gradient_norm'])
    write_json(folder/'summary.json',result)
    del model,optimizer,train_inputs,val_inputs,labels,payload,best;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    audit(f'Baseline {architecture} seed={seed} fold={fold} verified selected_epoch={result["selected_epoch"]} BA={measured["balanced_accuracy"]} F1={measured["macro_f1"]} AP={measured["macro_auprc"]}')
    return result
