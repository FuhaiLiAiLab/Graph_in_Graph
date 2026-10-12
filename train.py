"""Train on fixed family folds with fold-local arrays and reconstructable checkpoints."""
import argparse
import csv
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
if __package__:
    from .checkpoint_rules import normalize_rules, rule_score
    from .runtime import (ROOT, DEFAULT_DATA_ROOT, build_model, data_signature, forward, load_pair,
                          read_config, seed_everything, select_device, training_class_weights,
                          apply_graph_policy, patient_graph_summary)
    from .ablation_data import CONDITIONS, configure_ablation, transform_ablation
    from .model_config import resolve_model_config
    from .training_policy import normalize_policy, StopTracker
    from .evaluation_policy import PRIMARY_METRIC, validation_selection_score, generalization_gaps
    from .training_state import (fold_lock, cpu_copy, atomic_torch_save, capture_rng, restore_rng, build_optimization, source_fingerprint)
else:
    from checkpoint_rules import normalize_rules, rule_score
    from runtime import (ROOT, DEFAULT_DATA_ROOT, build_model, data_signature, forward, load_pair,
                         read_config, seed_everything, select_device, training_class_weights,
                         apply_graph_policy, patient_graph_summary)
    from ablation_data import CONDITIONS, configure_ablation, transform_ablation
    from model_config import resolve_model_config
    from training_policy import normalize_policy, StopTracker
    from evaluation_policy import PRIMARY_METRIC, validation_selection_score, generalization_gaps
    from training_state import (fold_lock, cpu_copy, atomic_torch_save, capture_rng, restore_rng, build_optimization, source_fingerprint)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def metrics(labels, logits):
    return probability_metrics(labels,torch.softmax(logits.detach(),dim=-1).cpu().numpy())


def probability_metrics(labels, prob):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, [0,1,2]).all():
        raise ValueError('Expected nonempty three-class labels')
    prob = np.asarray(prob)
    if prob.shape != (len(labels),3) or not np.isfinite(prob).all() or (prob<0).any() or (prob>1).any() or not np.allclose(prob.sum(axis=1),1,atol=1e-6):
        raise ValueError('Expected finite N x 3 class probabilities')
    pred = prob.argmax(axis=1)
    result = dict(accuracy=float(accuracy_score(labels, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                  macro_f1=float(f1_score(labels, pred, labels=[0,1,2], average='macro', zero_division=0)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,pred,labels=[0,1,2],zero_division=0)
    names = ['t2ds','pret2ds','no_t2ds']
    for k,name in enumerate(names):
        binary = (labels == k).astype(int)
        defined = 0 < support[k] < len(labels)
        result.update({f'{name}_precision':float(precision[k]),f'{name}_recall':float(recall[k]),
                       f'{name}_f1':float(f1[k]),f'{name}_support':int(support[k]),
                       f'{name}_auroc':float(roc_auc_score(binary,prob[:,k])) if defined else None,
                       f'{name}_auprc':float(average_precision_score(binary,prob[:,k])) if defined else None})
    for metric in ['auroc','auprc']:
        values = [result[f'{name}_{metric}'] for name in names]
        result[f'macro_{metric}'] = float(np.mean(values)) if all(v is not None for v in values) else None
    return result, prob





def save_predictions(path, data, logits):
    result, probs = metrics(data['labels'], logits)
    frame = pd.DataFrame({'subject': data['subject_ids'], 'local_node_index': data['node_index'],
                          'label': data['labels'], 'prediction': probs.argmax(axis=1)})
    for k, name in enumerate(['t2ds','pret2ds','no_t2ds']):
        frame[f'prob_{name}'] = probs[:, k]
    frame.to_csv(path, index=False)
    names = ['t2ds','pret2ds','no_t2ds']
    matrix = confusion_matrix(data['labels'],probs.argmax(axis=1),labels=[0,1,2])
    pd.DataFrame(matrix,index=pd.Index(names,name='true_class'),columns=names).to_csv(
        Path(path).with_name(Path(path).stem+'_confusion.csv'))
    return result


def run_fold(config, fold, output_root, data_root=DEFAULT_DATA_ROOT, device='cpu',
             epochs=500, seed=2026, batch_size=32, selection=PRIMARY_METRIC, progress_position=1, ablation=None,
             resume=False, training_policy=None, progress_callback=None, checkpoint_rules=None):
    with fold_lock(output_root,fold):
        return _run_fold(config,fold,output_root,data_root,device,epochs,seed,batch_size,selection,
                         progress_position,ablation,resume,training_policy,progress_callback,checkpoint_rules)


def _run_fold(config,fold,output_root,data_root,device,epochs,seed,batch_size,selection,
              progress_position,ablation,resume,training_policy,progress_callback,checkpoint_rules):
    started=perf_counter()
    if epochs<1 or batch_size<1:
        raise ValueError('epochs and batch_size must be positive')
    if selection not in ['accuracy','macro_f1','balanced_accuracy','macro_auroc','macro_auprc']:
        raise ValueError('Unsupported checkpoint selection metric')
    policy=normalize_policy(training_policy,epochs)
    checkpoint_rules=normalize_rules(checkpoint_rules)
    stop_monitor=policy['early_stopping'].get('monitor','selection')
    if stop_monitor!='selection' and stop_monitor not in checkpoint_rules:
        raise ValueError('Early stopping monitor must have a saved checkpoint rule')
    rule_best_scores={r:float("-inf") for r in checkpoint_rules}
    rule_best_payloads={}
    config=resolve_model_config(config)
    if ablation is not None:config=resolve_model_config(configure_ablation(config,ablation))
    folder=Path(output_root)/f'fold_{fold:02d}'
    if resume:
        if not (folder/'last_model.pt').is_file():raise ValueError('No last_model.pt to resume')
    else:folder.mkdir(parents=True,exist_ok=False)
    device=select_device(str(device));reproducibility=seed_everything(seed)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    signature=data_signature(data_root)
    versions={n:importlib.metadata.version(n) for n in ['torch','torch-geometric','numpy','pandas','scikit-learn']}
    train,test=load_pair(fold,data_root)
    if ablation is not None:train,test=transform_ablation(train,ablation),transform_ablation(test,ablation)
    graph_policy=config.get('graph_policy','legacy_directed')
    train,test=apply_graph_policy(train,graph_policy),apply_graph_policy(test,graph_policy)
    patient_graph={'train':patient_graph_summary(train),'test':patient_graph_summary(test)}
    contract=dict(model_config=config,fold=fold,seed=seed,epochs=epochs,batch_size=batch_size,
                  selection=selection,ablation=ablation,data_signature=signature,
                  device=str(device),versions=versions,source_sha256=source_fingerprint(ROOT),training_policy=policy,
                  reproducibility=reproducibility,checkpoint_rules=checkpoint_rules,patient_graph=patient_graph)
    saved=None
    if resume:
        saved=torch.load(folder/'last_model.pt',map_location='cpu',weights_only=True)
        if saved.get('resume_schema_version')!=1 or saved['contract']!=contract:
            raise ValueError('Resume contract differs (configuration/data/code/environment/device)')
    if train['num_feature']!=config['gene_input_dim'] or train['graph_x'].shape[1]!=config['gig_input_dim']:
        raise ValueError('Model input dimensions differ from prepared data')
    model=build_model(config,train['num_gene_node'],device);model.progress_position=progress_position
    weights=training_class_weights(train['labels'],device,config.get('class_frequency_exponent',1.),
                                   config.get('majority_class_multiplier',.5),mode=config.get('class_weighting','power'))
    optimizer,scheduler=build_optimization(model,config,policy['scheduler_horizon'])
    run_config=dict(contract,class_weights=weights.cpu().tolist(),metric_schema_version=2,
        class_weighting=config.get('class_weighting','power'),class_weight_source='training labels only',
        training_class_counts=np.bincount(train['labels'],minlength=3).tolist(),
        selection_split='validation',training_metrics_role='diagnostic only; not optimization target',
        evaluation_role='validation fold used for epoch/hyperparameter selection; not independent final test',
        loop_policy='epoch cap; optional early stopping; best validation metric, earliest tie',
        metric_definitions='F1/precision/recall: argmax, zero_division=0; AUC: one-vs-rest; AUPRC is noninterpolated AP',
        graph_policy=graph_policy)
    if not resume:write_json(folder/'config.json',run_config)
    best_score=-float('inf');best_payload=None;history=[];start_epoch=1;prior_elapsed=0.;resume_count=0
    peak_allocated=peak_reserved=0
    stopper=StopTracker(policy['early_stopping'])
    training_complete=False;stop_reason='epoch_cap'
    if saved is not None:
        model.load_state_dict(saved['model_state'],strict=True);optimizer.load_state_dict(saved['optimizer_state'])
        if scheduler is not None:scheduler.load_state_dict(saved['scheduler_state'])
        history=saved['history'];best_payload=saved['best_payload'];best_score=saved['best_score']
        start_epoch=saved['epoch']+1;prior_elapsed=saved['elapsed_seconds'];resume_count=saved['resume_count']+1
        peak_allocated=saved.get('cuda_peak_allocated_bytes',0);peak_reserved=saved.get('cuda_peak_reserved_bytes',0)
        stopper=StopTracker(policy['early_stopping'],saved.get('stopper_state'))
        training_complete=saved.get('training_complete',saved['epoch']==epochs)
        stop_reason=saved.get('stop_reason','epoch_cap')
        if len(history)!=saved['epoch'] or [int(r['epoch']) for r in history]!=list(range(1,start_epoch)):
            raise ValueError('Invalid resume history/epoch contract')
        atomic_torch_save(best_payload,folder/'best_model.pt')
        rule_best_scores=saved.get('rule_best_scores',rule_best_scores)
        rule_best_payloads=saved.get('rule_best_payloads',{})
        for name,payload in rule_best_payloads.items():
            atomic_torch_save(payload,folder/f'rule_{name}.pt')
        # Building/loading the model consumes random numbers. Restore RNG only afterwards.
        restore_rng(saved['rng_state'])
    with (folder/'history.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=None
        if history:
            writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history);handle.flush()
        epoch_bar=tqdm(range(start_epoch,epochs+1) if not training_complete else [],desc=f'Fold {fold} epochs',unit='epoch',
            position=progress_position if sys.stderr.isatty() else 0,leave=True,mininterval=2 if sys.stderr.isatty() else 10,ascii=True)
        for epoch in epoch_bar:
            epoch_started=perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            logits=forward(model,train,device,batch_size)[1]
            loss=model.loss(logits,torch.as_tensor(train['labels_onehot'][None],device=device),
                            torch.as_tensor(train['gene_edge_index'],dtype=torch.long),class_weights=weights)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf') if config['clip'] is None else config['clip'],error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:scheduler.step()
            train_metrics,_=metrics(train['labels'],logits);loss_value=float(loss.detach());del logits,loss
            model.eval()
            with torch.no_grad():validation_logits=forward(model,test,device,batch_size)[1]
            val_metrics,_=metrics(test['labels'],validation_logits)
            score=validation_selection_score(val_metrics,selection)
            if device.type=='cuda':torch.cuda.synchronize(device)
            row=dict(epoch=epoch,loss=loss_value,grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],
                epoch_seconds=perf_counter()-epoch_started,elapsed_seconds=prior_elapsed+perf_counter()-started,
                **model.last_loss_components,**{f'train_{k}':v for k,v in train_metrics.items()},
                **{f'validation_{k}':v for k,v in val_metrics.items()})
            if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);handle.flush();history.append(row)
            improved=[name for name in checkpoint_rules if rule_score(val_metrics,name)>rule_best_scores[name]]
            if improved:
                shared_state=cpu_copy(model.state_dict())
                for name in improved:
                    rule_best_scores[name]=rule_score(val_metrics,name)
                    rule_payload=dict(schema_version=1,model_state=shared_state,model_config=config,
                        fold=fold,seed=seed,epoch=epoch,data_signature=signature,class_weights=weights.cpu(),
                        metrics=dict(val_metrics),selection='checkpoint_rule:'+name,metric_schema_version=2,
                        ablation=ablation,reproducibility=reproducibility,patient_graph=patient_graph)
                    rule_best_payloads[name]=rule_payload
                    atomic_torch_save(rule_payload,folder/f'rule_{name}.pt')
            if score>best_score:
                best_score=score
                best_payload=dict(schema_version=1,model_state=cpu_copy(model.state_dict()),model_config=config,
                    fold=fold,seed=seed,epoch=epoch,data_signature=signature,class_weights=weights.cpu(),
                    metrics=val_metrics,selection=selection,metric_schema_version=2,ablation=ablation,
                    reproducibility=reproducibility,patient_graph=patient_graph)
                atomic_torch_save(best_payload,folder/'best_model.pt')
            if device.type=='cuda':
                peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
                peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
            stop_score=score if stop_monitor=='selection' else rule_score(val_metrics,stop_monitor)
            early_stop=stopper.observe(epoch,stop_score)
            training_complete=early_stop or epoch==epochs
            stop_reason='early_stopping' if early_stop else 'epoch_cap'
            def snapshot():
                return dict(resume_schema_version=1,contract=contract,epoch=epoch,model_state=cpu_copy(model.state_dict()),
                    optimizer_state=cpu_copy(optimizer.state_dict()),scheduler_state=scheduler.state_dict() if scheduler else None,
                    rng_state=capture_rng(),history=history,best_payload=best_payload,best_score=best_score,
                    rule_best_scores=rule_best_scores,rule_best_payloads=rule_best_payloads,
                    elapsed_seconds=prior_elapsed+perf_counter()-started,resume_count=resume_count,
                    cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved,
                    stopper_state=stopper.state_dict(),training_complete=training_complete,stop_reason=stop_reason)
            if progress_callback is not None:
                try:progress_callback(epoch,score,best_score)
                except BaseException:
                    # Callback pruning or interruption gets its own current-epoch snapshot.
                    training_complete=False;stop_reason='callback_interruption'
                    atomic_torch_save(snapshot(),folder/'last_model.pt')
                    raise
            if epoch==1 or epoch%policy['checkpoint_interval']==0 or training_complete:
                atomic_torch_save(snapshot(),folder/'last_model.pt')
            epoch_bar.set_postfix(loss=f'{loss_value:.4f}',refresh=False,**{selection:f'{score:.4f}'})
            if early_stop:break
    payload=torch.load(folder/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(payload['model_state'],strict=True);model.eval()
    with torch.no_grad():
        final_test=save_predictions(folder/'test_predictions.csv',test,forward(model,test,device,batch_size)[1])
        final_train=save_predictions(folder/'train_predictions.csv',train,forward(model,train,device,batch_size)[1])
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated(device))
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved(device))
    summary=dict(fold=fold,seed=seed,ablation=ablation,selected_epoch=payload['epoch'],selection=selection,
        train=final_train,validation=final_test,generalization_gap=generalization_gaps(final_train,final_test),
        gap_role='diagnostic only; scores from the same selected checkpoint',selection_split='validation',
        class_weighting=config.get('class_weighting','power'),class_weights=weights.cpu().tolist(),n_train=train['num_subject'],n_test=test['num_subject'],
        elapsed_seconds=prior_elapsed+perf_counter()-started,completed_epochs=len(history),epoch_cap=epochs,training_complete=True,stop_reason=stop_reason,resume_count=resume_count,
        model_parameter_count=sum(p.numel() for p in model.parameters()),graph_policy=graph_policy,patient_graph=patient_graph,
        cuda_peak_allocated_bytes=peak_allocated if device.type=='cuda' else None,
        cuda_peak_reserved_bytes=peak_reserved if device.type=='cuda' else None)
    if checkpoint_rules:
        write_json(folder/'rule_checkpoints.json',{name:dict(epoch=p['epoch'],metrics=p['metrics'],
            checkpoint=f'rule_{name}.pt',score=rule_best_scores[name]) for name,p in rule_best_payloads.items()})
    write_json(folder/'summary.json',summary)
    return summary

def main():
    from training.run import main as entry
    entry(["full"])
    # Fold-local training artifacts are written by run_fold; no cross-fold report export.


if __name__ == '__main__':
    main()
