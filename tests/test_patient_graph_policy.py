"""Protect the single graph-direction change, old behavior, and saved-model contract."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
import torch

import runtime
from analyze import AttentionCollector
from test_runtime import synthetic_data
from tools.smoke_pipeline import make_fixture
from train import run_fold


NEW_POLICY = 'patient_category_bidirectional'


def config(policy='legacy_directed'):
    return dict(runtime.default_config(), model_schema_version=2,
                gene_num_top_feature=2, gene_layers=2, patient_layers=2,
                graph_policy=policy)


def test_only_subject_category_reverse_edges_are_added_without_mutating_inputs():
    # C0->C1 remains one-way. S2/S3->C0 and S4->C1 gain reverse edges.
    data = synthetic_data()
    data['edge_index'] = np.array([[0, 2, 3, 4], [1, 0, 0, 1]])
    before = copy.deepcopy(data)
    result = runtime.apply_graph_policy(data, NEW_POLICY)
    np.testing.assert_array_equal(result['edge_index'],
        [[0, 2, 3, 4, 0, 0, 1], [1, 0, 0, 1, 2, 3, 4]])
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(data[key], value)
        if key != 'edge_index':
            np.testing.assert_array_equal(result[key], value)
    np.testing.assert_array_equal(
        runtime.apply_graph_policy(result, NEW_POLICY)['edge_index'], result['edge_index'])
    # Untagged, already bidirectional input must not accumulate duplicate edges either.
    raw_again = dict(before, edge_index=result['edge_index'].copy())
    np.testing.assert_array_equal(
        runtime.apply_graph_policy(raw_again, NEW_POLICY)['edge_index'], result['edge_index'])


def test_legacy_policy_preserves_input_and_rejects_relabeling_an_effective_new_graph():
    data = synthetic_data()
    assert runtime.apply_graph_policy(data, 'legacy_directed') is data
    changed = runtime.apply_graph_policy(data, NEW_POLICY)
    with pytest.raises(ValueError, match='policy'):
        runtime.apply_graph_policy(changed, 'legacy_directed')
    with pytest.raises(ValueError, match='policy'):
        runtime.apply_graph_policy(data, 'all_edges_undirected')


@pytest.mark.parametrize('bad_edges', [
    np.array([[2, 3], [0, 5]]),  # node 5 is outside this local graph
    np.array([[2, 3], [0, -1]]),
    np.array([[2.5], [0.0]]),
    np.array([2, 0]),
])
def test_new_policy_rejects_invalid_edge_identity(bad_edges):
    data = dict(synthetic_data(), edge_index=bad_edges)
    with pytest.raises(ValueError):
        runtime.apply_graph_policy(data, NEW_POLICY)


def test_new_policy_does_not_change_initial_weights_or_parameter_count():
    torch.manual_seed(17)
    old = runtime.build_model(config(), 4, 'cpu')
    torch.manual_seed(17)
    new = runtime.build_model(config(NEW_POLICY), 4, 'cpu')
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())


def test_two_patient_layers_receive_shared_category_neighbor_information():
    data = synthetic_data()
    data['graph_x'][:2] = 0  # prepared category nodes start at zero, as in real data
    perturbed = copy.deepcopy(data)
    perturbed['graph_x'][3] += .7  # change S3; inspect the unchanged S2 sharing C0
    perturbed['gene_x'][1] += .7
    for policy in ('legacy_directed', NEW_POLICY):
        torch.manual_seed(17)
        model = runtime.build_model(config(policy), 4, 'cpu').eval()
        original = runtime.forward(model, data, 'cpu')[1]
        altered = runtime.forward(model, perturbed, 'cpu')[1]
        if policy == 'legacy_directed':
            torch.testing.assert_close(original[0], altered[0], rtol=0, atol=0)
        else:
            assert (original[0] - altered[0]).abs().max().item() > 1e-7
            original[0].square().sum().backward()
            gradients = [layer.lin_value.weight.grad for layer in model.patient_convs]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.abs().sum().item() for g in gradients) > 0
        # S4 belongs only to C1; neither policy creates a two-hop path from S3 to S4.
        torch.testing.assert_close(original[2], altered[2], rtol=0, atol=0)


def test_attention_export_uses_effective_edges_and_preserves_predictions(tmp_path):
    data = synthetic_data()
    model = runtime.build_model(config(NEW_POLICY), 4, 'cpu').eval()
    with torch.no_grad():
        expected = runtime.forward(model, data, 'cpu')[1]
    collector = AttentionCollector(data, 1, tmp_path, ['C0', 'C1']).attach(model)
    try:
        with torch.no_grad():
            actual = runtime.forward(model, data, 'cpu')[1]
        collector.finish()
    finally:
        collector.remove()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    files = list(tmp_path.glob('*_patient_edge_weight.csv'))
    assert len(files) == 2
    wanted = {(2, 0), (3, 0), (4, 1), (0, 2), (0, 3), (1, 4)}
    for path in files:
        frame = pd.read_csv(path)
        assert len(frame) == 6
        assert set(zip(frame.from_node_idx, frame.to_node_idx)) == wanted
    np.testing.assert_array_equal(data['edge_index'], [[2, 3, 4], [0, 0, 1]])


def test_new_graph_training_resume_reload_and_policy_identity(tmp_path):
    data_root = tmp_path/'data'
    make_fixture(data_root)
    cfg = dict(config(NEW_POLICY), representation_dropout=.2)
    full, interrupted = tmp_path/'full', tmp_path/'interrupted'
    options = dict(data_root=data_root, device='cpu', epochs=3, seed=17, batch_size=4,
                   checkpoint_rules=['balanced_guard'])
    run_fold(cfg, 1, full, **options)

    def stop_after_epoch_one(epoch, *_):
        if epoch == 1:
            raise RuntimeError('controlled interruption')

    with pytest.raises(RuntimeError, match='controlled interruption'):
        run_fold(cfg, 1, interrupted, progress_callback=stop_after_epoch_one, **options)
    with pytest.raises(ValueError, match='contract'):
        run_fold(dict(cfg, graph_policy='legacy_directed'), 1, interrupted, resume=True, **options)
    run_fold(cfg, 1, interrupted, resume=True, **options)
    a = torch.load(full/'fold_01/last_model.pt', weights_only=True)
    b = torch.load(interrupted/'fold_01/last_model.pt', weights_only=True)
    for name, value in a['model_state'].items():
        torch.testing.assert_close(value, b['model_state'][name], rtol=0, atol=0)
    assert (full/'fold_01/test_predictions.csv').read_bytes() == (
        interrupted/'fold_01/test_predictions.csv').read_bytes()
    saved = json.loads((full/'fold_01/config.json').read_text())
    assert saved['graph_policy'] == NEW_POLICY
    assert saved['patient_graph']['train']['edge_count'] == 24
    assert saved['patient_graph']['test']['edge_count'] == 6
    assert saved['patient_graph']['test']['subject_in_degree_min'] == 1
    model, payload, loaded = runtime.load_run(full/'fold_01/best_model.pt', data_root, 'cpu')
    assert loaded['edge_index'].shape == (2, 6)
    with torch.no_grad():
        prob = torch.softmax(runtime.forward(model, loaded, 'cpu', 4)[1], -1).numpy()
    predicted = pd.read_csv(full/'fold_01/test_predictions.csv')
    np.testing.assert_allclose(prob, predicted[['prob_t2ds','prob_pret2ds','prob_no_t2ds']],
                               rtol=1e-6, atol=1e-7)
