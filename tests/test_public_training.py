"""Portable fixed-config training and selected-checkpoint regressions on synthetic data."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from tools.smoke_pipeline import make_fixture
from baseline_train import train_fold
from runtime import ROOT

spec=importlib.util.spec_from_file_location('public_training',ROOT/'training/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def baseline_fixture(root):
    make_fixture(root)
    cfg=json.loads((root/'config.json').read_text());cfg['n_genes']=12;(root/'config.json').write_text(json.dumps(cfg))
    pd.DataFrame({'gene_index':range(12),'gene_name':['g'+str(i) for i in range(12)]}).to_csv(root/'metadata/genes.csv',index=False)
    np.save(root/'shared/gene_edge_index.npy',np.array([np.arange(12),np.roll(np.arange(12),-1)]))
    rng=np.random.default_rng(4)
    for role,n,start in [('train',96,1000),('test',24,2000)]:
        folder=root/'folds/fold_01'/role;ids=np.arange(start,start+n);y=np.arange(n)%3
        x=rng.normal(size=(n,42)).astype('float32')
        arrays=dict(subject_ids=ids,labels=y,labels_onehot=np.eye(3)[y],gene_x=rng.normal(size=(n,72)).astype('float32'),phenotype_x=x,phenotype_bins=np.ones((n,40)),graph_x=np.vstack([np.zeros((2,42),dtype='float32'),x]),edge_index=np.array([np.arange(2,n+2),np.arange(n)%2]),node_index=np.arange(2,n+2))
        for name,value in arrays.items():np.save(folder/(name+'.npy'),value)
        pd.DataFrame(dict(subject=ids,family_id=ids,label=y)).to_csv(folder/'subjects.csv',index=False)

@pytest.mark.parametrize('arch',['gat','gin'])
def test_fixed_baseline_pca64_resume(tmp_path,arch):
    data=tmp_path/'data';baseline_fixture(data);cfg=json.loads((ROOT/f'configs/{arch}.json').read_text())
    full=tmp_path/'continuous';stopped=tmp_path/'stopped'
    train_fold(arch,cfg,1,2026,data,full,'cpu',epochs=3,checkpoint_interval=1)
    def interrupt(*args):raise RuntimeError('test interruption')
    with pytest.raises(RuntimeError,match='test interruption'):
        train_fold(arch,cfg,1,2026,data,stopped,'cpu',epochs=3,checkpoint_interval=1,callback=interrupt)
    train_fold(arch,cfg,1,2026,data,stopped,'cpu',epochs=3,checkpoint_interval=1)
    a=torch.load(full/'rule_balanced_guard.pt',weights_only=True);b=torch.load(stopped/'rule_balanced_guard.pt',weights_only=True)
    for k in a['model_state']:torch.testing.assert_close(a['model_state'][k],b['model_state'][k],rtol=0,atol=0)
    assert (full/'test_predictions.csv').read_text()==(stopped/'test_predictions.csv').read_text()
    result=json.loads((full/'summary.json').read_text());assert result['input_dim']==106

@pytest.mark.parametrize('model',['full','phenotype_only','gene_only','rna_only','methylation_only'])
def test_fixed_full_and_ablation_selected_output(tmp_path,model):
    data=tmp_path/'data';make_fixture(data)
    args=SimpleNamespace(device='cpu',data_root=data,output=tmp_path/'run',epochs=1,batch_size=4,resume=False)
    runner.execute(model,2026,1,args)
    folder=args.output/model/'seed_2026/fold_01';p=torch.load(folder/'rule_balanced_guard.pt',weights_only=True)
    assert (folder/'selected_predictions.csv').is_file()
    if model!='full':
        assert p['model_config']['class_weighting']=='none'
        assert all(p['model_config'][x]==0 for x in ['ortho_weight','link_weight','ent_weight'])
    args.resume=True;runner.execute(model,2026,1,args)

def test_grid_has_all_175_unique_tasks():
    plan=json.loads((ROOT/'configs/evaluation.json').read_text());tasks=runner.task_grid(list(runner.MODELS),plan['seeds'],plan['folds'])
    assert len(tasks)==len(set(tasks))==175
    with pytest.raises(ValueError):runner.task_grid(['gat'],[1,1],[1])
