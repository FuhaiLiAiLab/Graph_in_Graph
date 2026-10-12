import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from runtime import build_model, default_config, forward, training_class_weights


def synthetic_data():
    return dict(subject_ids=np.array([101, 205, 302]), labels=np.array([0, 1, 2]),
                labels_onehot=np.eye(3, dtype=np.int64), num_subject=3,
                num_gene_node=4, num_feature=6, num_subfeature=2,
                gene_x=np.arange(72, dtype=np.float32).reshape(3, 24)/72,
                gene_edge_index=np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
                graph_x=np.arange(210, dtype=np.float32).reshape(5, 42)/210,
                edge_index=np.array([[2, 3, 4], [0, 0, 1]]),
                node_index=np.array([2, 3, 4]))


@pytest.mark.parametrize('k', [24, 48, 64, 96, 128])
def test_cpu_forward_backward_batch_equivalence(k):
    cfg = default_config()
    cfg['gene_num_top_feature'] = k
    model = build_model(cfg, 4, 'cpu')
    data = synthetic_data()
    model.eval()
    a = forward(model, data, 'cpu', batch_size=2)[1]
    b = forward(model, data, 'cpu', batch_size=3)[1]
    assert a.shape == (3, 3)
    torch.testing.assert_close(a, b)
    assert model.node_weight_assign.shape == (k, 4)
    model.train()
    w = training_class_weights(data['labels'], 'cpu')
    loss = model.loss(a, torch.tensor(data['labels_onehot'])[None],
                      torch.tensor(data['gene_edge_index']), k, class_weights=w)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.node_weight_assign.grad is not None
    assert torch.isfinite(model.node_weight_assign.grad).all()


def test_class_weights_from_training_counts():
    weights = training_class_weights(np.array([0, 0, 1, 2, 2, 2]), 'cpu')
    # Legacy relative weights: inverse class counts, class 2 additionally /2.
    torch.testing.assert_close(weights/weights[1], torch.tensor([.5, 1., 1/6]))
    with pytest.raises(ValueError, match='class'):
        training_class_weights(np.array([0, 1]), 'cpu')


def test_explicit_invalid_heads_rejected():
    cfg = default_config()
    cfg['gene_num_head'] = 2
    with pytest.raises(ValueError, match='head'):
        build_model(cfg, 4, 'cpu')
