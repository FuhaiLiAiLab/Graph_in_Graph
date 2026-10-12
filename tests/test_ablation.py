import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from ablation_data import configure_ablation, transform_ablation
from runtime import build_model, default_config, forward, training_class_weights
from test_runtime import synthetic_data


@pytest.mark.parametrize('condition,channels', [('rna_only',[0]),('methylation_only',[1,2,3,4,5]),('gene_only',[0,1,2,3,4,5])])
def test_omics_branch_removes_phenotype_and_uses_exact_channels(condition,channels):
    raw = synthetic_data()
    result = transform_ablation(raw,condition)
    expected = raw['gene_x'].reshape(3,4,6)[:,:,channels]
    np.testing.assert_array_equal(result['gene_x'].reshape(expected.shape),expected)
    assert result['num_feature'] == len(channels)
    assert not result['graph_x'].any() and result['edge_index'].shape == (2,0)
    assert raw['graph_x'].any() and raw['edge_index'].shape[1] > 0
    cfg = configure_ablation(default_config(),condition)
    assert cfg['graph_opt'] == 'gene' and cfg['gene_input_dim']==len(channels)


def test_phenotype_only_removes_all_omics_and_gene_regularization():
    raw = synthetic_data()
    cfg = configure_ablation(default_config(),'phenotype_only')
    data = transform_ablation(raw,'phenotype_only')
    assert cfg['graph_opt']=='subject'
    assert cfg['ortho_weight']==cfg['link_weight']==cfg['ent_weight']==0
    assert not data['gene_x'].any() and data['gene_edge_index'].shape==(2,0)
    np.testing.assert_array_equal(data['graph_x'],raw['graph_x'])
    np.testing.assert_array_equal(data['edge_index'],raw['edge_index'])


@pytest.mark.parametrize('condition',['full','rna_only','methylation_only','gene_only','phenotype_only'])
def test_each_ablation_forward_backward_and_removed_modality_invariance(condition):
    data = synthetic_data()
    cfg = configure_ablation(default_config(),condition)
    model = build_model(cfg,4,'cpu')
    model.eval()
    transformed = transform_ablation(data,condition)
    logits = forward(model,transformed,'cpu',batch_size=2)[1]
    assert logits.shape==(3,3)
    loss = model.loss(logits,torch.tensor(data['labels_onehot'])[None],
                      torch.as_tensor(transformed['gene_edge_index']),
                      class_weights=training_class_weights(data['labels'],'cpu'))
    loss.backward()
    assert torch.isfinite(loss)
    if condition=='phenotype_only':
        assert model.node_weight_assign.grad is None
        assert model.gene_conv_first.lin_key.weight.grad is None
    changed = dict(data)
    if condition in ['rna_only','methylation_only','gene_only']:
        changed['graph_x'] = data['graph_x']+1000
        changed['edge_index'] = np.empty((2,0),dtype=np.int64)
    if condition in ['rna_only','methylation_only','phenotype_only']:
        gene = data['gene_x'].reshape(3,4,6).copy()
        excluded = [1,2,3,4,5] if condition=='rna_only' else ([0] if condition=='methylation_only' else list(range(6)))
        gene[:,:,excluded] += 1000
        changed['gene_x'] = gene.reshape(3,-1)
    with torch.no_grad():
        altered = forward(model,transform_ablation(changed,condition),'cpu',batch_size=2)[1]
    torch.testing.assert_close(logits,altered)
