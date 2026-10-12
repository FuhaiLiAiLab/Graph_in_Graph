import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
import torch
from runtime import default_config,build_model,forward
from test_runtime import synthetic_data
from analyze import AttentionCollector
from gene_scores import GeneScoreCollector,attention_node_scores

@pytest.mark.parametrize('depth,heads',[(1,1),(4,1),(4,2)])
def test_v2_all_layers_and_exact_scores(tmp_path,depth,heads):
    data=synthetic_data()
    cfg=dict(default_config(),model_schema_version=2,gene_layers=depth,patient_layers=2,
             gene_hidden_dim=12,gene_output_dim=6,gene_num_head=heads)
    model=build_model(cfg,4,'cpu').eval()
    with torch.no_grad(): original=forward(model,data,'cpu',2)[1]
    scorer=GeneScoreCollector(model,data)
    collector=AttentionCollector(data,1,tmp_path,['a','b'],gene_scorer=scorer).attach(model)
    try:
        with torch.no_grad(): actual=forward(model,data,'cpu',2)[1]
        edges=collector.finish()
        scores=scorer.finish_arrays()
    finally: collector.remove()
    torch.testing.assert_close(original,actual,rtol=0,atol=0)
    assert edges.Layer.nunique()==depth
    assert edges.Head.nunique()==heads
    assert len(collector.pheno_seen)==2
    x=torch.from_numpy(data['gene_x'][0].reshape(4,6))
    edge=torch.as_tensor(data['gene_edge_index'],dtype=torch.long)
    expected=np.zeros(4)
    with torch.no_grad():
        for layer in model.gene_convs:
            x,(_,alpha)=layer(x,edge,return_attention_weights=True)
            expected+=attention_node_scores(alpha[:,0].numpy()[None],edge.numpy(),4)[0]/depth
            x=model.act(x)
    np.testing.assert_allclose(scores['attention_raw'][0],expected,rtol=1e-6)
    np.testing.assert_allclose(scorer.embedding_norm[0],np.linalg.norm(x.numpy(),axis=1),rtol=1e-6)
