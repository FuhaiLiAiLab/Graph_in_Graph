"""A stopped training process must resume without resetting optimizer/scheduler/RNG."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pytest
import torch
from runtime import default_config
from train import run_fold
from tools.smoke_pipeline import make_fixture


def test_interrupted_fold_matches_continuous_training(tmp_path,monkeypatch):
    data=tmp_path/'data';make_fixture(data)
    cfg=dict(default_config(),model_schema_version=2,representation_dropout=.2)
    full=tmp_path/'full'; interrupted=tmp_path/'interrupted'
    run_fold(cfg,1,full,data,'cpu',epochs=3,seed=7,batch_size=4)
    original=torch.optim.Adam.step;calls=0
    def stop_on_second_step(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated process interruption')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',stop_on_second_step)
    with pytest.raises(RuntimeError,match='simulated process'):
        run_fold(cfg,1,interrupted,data,'cpu',epochs=3,seed=7,batch_size=4)
    assert (interrupted/'fold_01/last_model.pt').is_file()
    monkeypatch.setattr(torch.optim.Adam,'step',original)
    run_fold(cfg,1,interrupted,data,'cpu',epochs=3,seed=7,batch_size=4,resume=True)
    a=torch.load(full/'fold_01/last_model.pt',weights_only=True)
    b=torch.load(interrupted/'fold_01/last_model.pt',weights_only=True)
    assert a['epoch']==b['epoch']==3
    for k in a['model_state']:torch.testing.assert_close(a['model_state'][k],b['model_state'][k],rtol=0,atol=0)
    assert a['scheduler_state']==b['scheduler_state']
    assert (full/'fold_01/test_predictions.csv').read_text()==(interrupted/'fold_01/test_predictions.csv').read_text()
    with pytest.raises(ValueError,match='contract'):
        run_fold(dict(cfg,lr=.01),1,interrupted,data,'cpu',epochs=3,seed=7,batch_size=4,resume=True)


@pytest.mark.parametrize('scheduler',['none','cosine','multistep'])
def test_optimizer_scheduler_configuration(tmp_path,scheduler):
    data=tmp_path/'data';make_fixture(data)
    cfg=dict(default_config(),model_schema_version=2,optimizer='adamw',scheduler=scheduler,
             beta1=.9,beta2=.98,optimizer_eps=1e-6,clip=None)
    out=tmp_path/'out'
    run_fold(cfg,1,out,data,'cpu',epochs=2,seed=8,batch_size=4)
    state=torch.load(out/'fold_01/last_model.pt',weights_only=True)
    group=state['optimizer_state']['param_groups'][0]
    assert group['betas']==(.9,.98) and group['eps']==1e-6
    assert state['contract']['model_config']['optimizer']=='adamw'
    assert (state['scheduler_state'] is None)==(scheduler=='none')
    summary=json.loads((out/'fold_01/summary.json').read_text())
    assert summary['completed_epochs']==2 and summary['model_parameter_count']>0
