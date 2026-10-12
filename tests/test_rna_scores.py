from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch
from rna_scores import signed_rna_scores, class_score_table, SignedRNACollector
from runtime import default_config, build_model, forward
from gene_scores import GeneScoreCollector
from analyze import AttentionCollector


def test_signed_product_matches_literal_pooling_contribution():
    h = np.array([[[3.,4.],[1.,-2.],[4.,0.]], [[-1.,2.],[2.,1.],[0.,3.]]])
    assignment = np.array([[-1.,2.,0.],[2.,-1.,3.]])
    rna = np.array([[-2.5,0.,2.75],[.2,-.1,1.5]])
    score = signed_rna_scores(np.linalg.norm(h, axis=-1), assignment, rna)
    expected = np.empty_like(rna)
    for person in range(2):
        for gene in range(3):
            expected[person,gene] = np.linalg.norm(np.outer(assignment[:,gene], h[person,gene]))*rna[person,gene]
    np.testing.assert_allclose(score, expected, rtol=1e-14)
    assert score[0,0] < 0 and score[0,1] == 0 and score[0,2] > 0


def test_multiply_before_class_averaging():
    h = np.array([[1.],[3.],[2.],[4.],[1.],[5.]])
    rna = np.array([[-2.],[1.],[.25],[2.],[-1.],[.5]])
    labels = np.array([0,0,1,1,2,2])
    genes = pd.DataFrame({'gene_index':[0], 'gene_name':['geneA']})
    table = class_score_table(signed_rna_scores(h, np.ones((1,1)), rna), labels, genes, 7, 1)
    assert table.iloc[0].Weight == .5
    assert table.iloc[0].Weight != float(h[:2].mean()*rna[:2].mean())
    assert table.n_subjects.tolist() == [2,2,2]


@pytest.mark.parametrize('bad', ['negative_norm','nan_rna','infinite_weight','wrong_rna_shape','wrong_weight_genes'])
def test_invalid_components_rejected(bad):
    h, w, r = np.ones((3,4)), np.ones((2,4)), np.ones((3,4))
    if bad == 'negative_norm': h[0,0] = -1
    if bad == 'nan_rna': r[0,0] = np.nan
    if bad == 'infinite_weight': w[0,0] = np.inf
    if bad == 'wrong_rna_shape': r = np.ones((3,1))
    if bad == 'wrong_weight_genes': w = np.ones((2,3))
    with pytest.raises(ValueError): signed_rna_scores(h,w,r)


@pytest.mark.parametrize('bad', ['missing_class','gene_order','duplicate_gene','label_length'])
def test_misaligned_group_export_rejected(bad):
    labels = np.array([0,1,2]); genes = pd.DataFrame({'gene_index':[0,1], 'gene_name':['A','B']})
    if bad == 'missing_class': labels[:] = 0
    if bad == 'gene_order': genes = genes.iloc[::-1]
    if bad == 'duplicate_gene': genes['gene_name'] = ['A','A']
    if bad == 'label_length': labels = labels[:2]
    with pytest.raises(ValueError): class_score_table(np.ones((3,2)),labels,genes,1,1)


def test_v2_scoring_preserves_logits_edges_and_roundtrips_signed_values(tmp_path):
    torch.manual_seed(71)
    data = dict(subject_ids=np.array([101,205,302]), labels=np.array([0,1,2]),
        labels_onehot=np.eye(3,dtype=np.int64), num_subject=3, num_gene_node=4,
        num_feature=6, num_subfeature=2, gene_x=np.arange(72,dtype=np.float32).reshape(3,24)/72,
        gene_edge_index=np.array([[0,1,2,3],[1,2,3,0]]), graph_x=np.arange(210,dtype=np.float32).reshape(5,42)/210,
        edge_index=np.array([[2,3,4],[0,0,1]]), node_index=np.array([2,3,4]))
    data['gene_x'][0,0], data['gene_x'][1,6], data['gene_x'][2,12] = -2.5, 0., 2.75
    original_input = data['gene_x'].copy()
    cfg = dict(default_config(), model_schema_version=2, gene_layers=4, patient_layers=2,
               gene_hidden_dim=12, gene_output_dim=6, gene_num_top_feature=3,
               graph_policy='patient_category_bidirectional', representation_dropout=.3)
    model = build_model(cfg,4,'cpu').eval()
    with torch.no_grad(): baseline = forward(model,data,'cpu',2)[1]
    legacy_dir, new_dir = tmp_path/'legacy', tmp_path/'signed'
    legacy_dir.mkdir(); new_dir.mkdir()
    legacy = GeneScoreCollector(model,data)
    collector = AttentionCollector(data,1,legacy_dir,['a','b'],gene_scorer=legacy).attach(model)
    try:
        with torch.no_grad(): old_logits = forward(model,data,'cpu',2)[1]
        old_edges = collector.finish()
    finally: collector.remove()
    scorer = SignedRNACollector(model,data)
    with pytest.raises(ValueError,match='Incomplete'): scorer.finish_signed_scores()
    collector = AttentionCollector(data,1,new_dir,['a','b'],gene_scorer=scorer).attach(model)
    try:
        with torch.no_grad(): logits = forward(model,data,'cpu',2)[1]
        edges = collector.finish()
    finally: collector.remove()
    torch.testing.assert_close(logits,baseline,rtol=0,atol=0)
    torch.testing.assert_close(logits,old_logits,rtol=0,atol=0)
    pd.testing.assert_frame_equal(edges,old_edges,check_exact=True)
    np.testing.assert_array_equal(data['gene_x'],original_input)
    genes = pd.DataFrame({'gene_index':range(4),'gene_name':['A','B','C','D']})
    table, scores, meta = scorer.export_signed(new_dir,genes,71,1)
    np.testing.assert_allclose(scores,legacy.finish_arrays()['embedding_x_assignment_l2']*scorer.rna,rtol=1e-14)
    assert scores[0,0]<0 and scores[1,1]==0 and meta['negative_rna_values']==1
    with np.load(new_dir/'subject_rna_scores.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['score'],scores)
        np.testing.assert_array_equal(saved['rna'],original_input.reshape(3,4,6)[:,:,0])
    reread = pd.read_csv(new_dir/'rna_gene_scores.csv')
    np.testing.assert_allclose(reread.Weight,table.Weight,rtol=1e-12,atol=1e-12)
    assert reread.Weight.min()<0 and not reread.duplicated(['Class','Node_idx']).any()
    model.train()
    with pytest.raises(ValueError,match='evaluation'): SignedRNACollector(model,data)
