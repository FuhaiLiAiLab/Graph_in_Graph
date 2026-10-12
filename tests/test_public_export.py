"""Selected Full checkpoint to RNA nodes/edges and two portable Shiny folders."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from runtime import ROOT
from train import run_fold
from tools.smoke_pipeline import make_fixture
from export_full_network import export_case,make_shiny_inputs
from network_average import equal_fold_seed_average

def test_selected_full_export_and_apps(tmp_path):
    data=tmp_path/'data';make_fixture(data)
    cfg=json.loads((ROOT/'configs/full.json').read_text())
    run_fold(cfg,1,tmp_path/'fit',data,'cpu',epochs=1,seed=2026,batch_size=4,checkpoint_rules=['balanced_guard'])
    folder=tmp_path/'export';payload,_=export_case(tmp_path/'fit/fold_01/rule_balanced_guard.pt',data,folder,'cpu',4)
    assert json.loads((folder/'case.json').read_text())['hook_logits_unchanged']
    scores=np.load(folder/'subject_rna_scores.npz')
    np.testing.assert_allclose(scores['score'],scores['embedding_norm']*np.linalg.norm(scores['assignment'],axis=0)*scores['rna'])
    nodes=pd.read_csv(folder/'rna_gene_scores.csv');edges=pd.read_csv(folder/'edges/rownorm/gene_edges.csv')
    genes=pd.read_csv(data/'metadata/genes.csv');p=tmp_path/'pvalues';p.mkdir()
    for name,col in [('TvsNO','t2ds_no_t2ds_pvalue'),('PrevsNO','pret2ds_no_t2ds_pvalue')]:
        pd.DataFrame({'gene':genes.gene_name,col:.02}).to_csv(p/(name+'_min_pvalues.csv'),index=False)
    make_shiny_inputs(nodes,edges,genes,tmp_path/'shiny',p)
    for kind in ['single_group','inter']:
        assert (tmp_path/'shiny'/kind/'app.R').is_file()
        assert (tmp_path/'shiny'/kind/'t6.txt').read_text()=='Locus\n'
    frames={(s,f):nodes.sample(frac=1,random_state=s+f) for s in [1,2] for f in range(1,6)}
    _,averaged=equal_fold_seed_average(frames,['Class','Node_idx'],[1,2],list(range(1,6)))
    np.testing.assert_allclose(averaged.Weight,nodes.sort_values(['Class','Node_idx']).Weight)
