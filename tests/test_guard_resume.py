import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from checkpoint_rules import RULES

def test_rule_payloads_survive_resume_without_changing_training(tmp_path):
    import json
    from tools.smoke_pipeline import make_fixture
    from runtime import default_config
    from train import run_fold
    data=tmp_path/'data';make_fixture(data)
    cfg=dict(default_config(),model_schema_version=2,representation_dropout=.2)
    args=dict(data_root=data,device='cpu',epochs=3,seed=7,batch_size=4,checkpoint_rules=RULES,
              training_policy={'checkpoint_interval':10,'early_stopping':{'enabled':False}})
    run_fold(cfg,1,tmp_path/'full',**args)
    plain=dict(args);plain.pop('checkpoint_rules')
    run_fold(cfg,1,tmp_path/'plain',**plain)
    control=torch.load(tmp_path/'plain/fold_01/last_model.pt',weights_only=True)
    tracked=torch.load(tmp_path/'full/fold_01/last_model.pt',weights_only=True)
    for name in control['model_state']:torch.testing.assert_close(control['model_state'][name],tracked['model_state'][name],rtol=0,atol=0)
    def interrupt(epoch,*unused):
        if epoch==2:raise RuntimeError('intentional test interruption')
    with pytest.raises(RuntimeError,match='intentional'):
        run_fold(cfg,1,tmp_path/'resume',progress_callback=interrupt,**args)
    stale=tmp_path/'resume/fold_01/rule_f1.pt';torch.save({'epoch':999},stale)
    run_fold(cfg,1,tmp_path/'resume',resume=True,**args)
    a=json.loads((tmp_path/'full/fold_01/rule_checkpoints.json').read_text())
    b=json.loads((tmp_path/'resume/fold_01/rule_checkpoints.json').read_text())
    assert a==b
    for rule in RULES:
        x=torch.load(tmp_path/f'full/fold_01/rule_{rule}.pt',weights_only=True)
        y=torch.load(tmp_path/f'resume/fold_01/rule_{rule}.pt',weights_only=True)
        assert x['epoch']==y['epoch'] and x['metrics']==y['metrics']
        for name in x['model_state']:torch.testing.assert_close(x['model_state'][name],y['model_state'][name],rtol=0,atol=0)
