"""Atomic epoch snapshots, safe RNG serialization and configurable optimization."""
import copy
import hashlib
import os
import random
from contextlib import contextmanager
from pathlib import Path
import fcntl
import numpy as np
import torch


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def atomic_torch_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),state[2],state[3],state[4]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs from resume snapshot')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def build_optimization(model,cfg,epochs):
    cls={'adam':torch.optim.Adam,'adamw':torch.optim.AdamW}[cfg.get('optimizer','adam')]
    optimizer=cls(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],
                  betas=(cfg.get('beta1',.8),cfg.get('beta2',.999)),eps=cfg.get('optimizer_eps',1e-7))
    kind=cfg.get('scheduler','multistep')
    if kind=='none':scheduler=None
    elif kind=='cosine':scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    else:scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg['milestones'],gamma=cfg['gamma'])
    return optimizer,scheduler


def source_fingerprint(root):
    names=['checkpoint_rules.py','runtime.py','train.py','model_config.py','training_state.py','training_policy.py','evaluation_policy.py','ablation_data.py',
           'enc/geo_gigtransformer.py','enc/geo_gigtransformer_v2.py','processed_data/scripts/load_fold.py',
           'geo_loader/geo_readgraph.py','geo_loader/geograph_sampler.py']
    return {name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in names}


@contextmanager
def fold_lock(output_root,fold):
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    with (root/f'.fold_{fold:02d}.lock').open('a') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Fold {fold} already has an active writer') from None
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
