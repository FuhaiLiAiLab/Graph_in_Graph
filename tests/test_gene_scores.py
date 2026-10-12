import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from gene_scores import pooling_scores, attention_node_scores, GeneScoreCollector
from normalize_edges import node_scores
from analyze import AttentionCollector
from runtime import build_model, default_config, forward
from test_runtime import synthetic_data


def test_pooling_product_is_outer_product_magnitude_and_not_softmax_weight():
    w=np.array([[-1.,2.,0.],[2.,1.,0.]])
    h=np.array([[[3.,4.],[1.,2.],[0.,0.]],[[1.,0.],[2.,3.],[1.,1.]]])
    result=pooling_scores(np.linalg.norm(h,axis=-1),w)
    for subject in range(2):
        for gene in range(3):
            expected=np.linalg.norm(np.outer(w[:,gene],h[subject,gene]))
            np.testing.assert_allclose(result['embedding_x_assignment_l2'][subject,gene],expected)
    np.testing.assert_array_equal(result['assignment_l2'][0],result['assignment_l2'][1])
    soft=np.exp(w-w.max(axis=1,keepdims=True)); soft/=soft.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(result['embedding_x_softmax_sum'],np.linalg.norm(h,axis=-1)*soft.sum(axis=0))


def test_attention_baseline_matches_existing_node_score_and_marks_isolates():
    edges=np.array([[0,0,1],[1,2,2]])
    weights=np.array([[.2,.4,.6],[.1,.3,.8]])
    actual=attention_node_scores(weights,edges,4)
    for subject in range(2):
        frame=pd.DataFrame({'Actual_From':edges[0],'Actual_To':edges[1],'Weight':weights[subject]})
        expected=node_scores(frame).set_index('Node_idx').Weight
        np.testing.assert_allclose(actual[subject,expected.index],expected)
    assert np.isnan(actual[:,3]).all()


def test_hook_scoring_uses_post_activation_and_preserves_predictions(tmp_path):
    data=synthetic_data()
    model=build_model(default_config(),4,'cpu').eval()
    with torch.no_grad():
        original=forward(model,data,'cpu',2)[1]
    scorer=GeneScoreCollector(model,data)
    attention=AttentionCollector(data,1,tmp_path,['a','b'],gene_scorer=scorer).attach(model)
    try:
        with torch.no_grad(): scored=forward(model,data,'cpu',2)[1]
    finally: attention.remove()
    torch.testing.assert_close(original,scored,rtol=0,atol=0)
    scores=scorer.finish_arrays()
    assert set(scores)=={'attention_raw','assignment_l2','embedding_x_assignment_l2','embedding_x_softmax_sum'}
    for arr in scores.values(): assert arr.shape==(3,4) and np.isfinite(arr).all()
    x=torch.from_numpy(data['gene_x'][0].reshape(4,6))
    edge=torch.as_tensor(data['gene_edge_index'],dtype=torch.long)
    with torch.no_grad():
        for layer in [model.gene_conv_first,model.gene_conv_block,model.gene_conv_last]:
            x=model.act(layer(x,edge))
    np.testing.assert_allclose(scorer.embedding_norm[0],np.linalg.norm(x.numpy(),axis=1),rtol=1e-6)
