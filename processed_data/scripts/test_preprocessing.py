import numpy as np
import pandas as pd
import pytest

if __package__:
    from . import preprocessing as p
else:
    import preprocessing as p


def test_held_out_extremes_never_change_training_parameters():
    train = np.array([[0., np.nan], [2., 10.], [4., 20.]])
    fitted = p.fit_numeric(train)
    before = p.transform_numeric(train, fitted)
    test = np.array([[1000., -500.], [np.nan, np.nan]])
    after_test = p.transform_numeric(test, fitted)
    np.testing.assert_allclose(before, p.transform_numeric(train, fitted))
    np.testing.assert_allclose(fitted['mean'], [2., 15.])
    np.testing.assert_allclose(after_test[0], [250., -51.])
    np.testing.assert_allclose(after_test[1], [.5, .5])


def test_bins_use_training_thresholds_and_legacy_right_closed_boundaries():
    fitted = p.fit_numeric(np.arange(11, dtype=float)[:, None])
    assert p.bin_numeric(np.array([[1.], [9.], [10.]]), fitted).ravel().tolist() == [1, 2, 3]
    tied = p.fit_numeric(np.ones((5, 1)))
    assert p.bin_numeric(np.array([[0.], [1.], [2.]]), tied).ravel().tolist() == [1, 2, 3]
    np.testing.assert_array_equal(p.transform_numeric(np.array([[1.], [7.]]), tied), 0.)


def test_all_missing_training_feature_rejected():
    with pytest.raises(ValueError, match='all.missing'):
        p.fit_numeric(np.array([[np.nan], [np.nan]]))


def test_earliest_visit_keeps_complete_row_without_backfilling():
    frame = pd.DataFrame({'subject': [8, 8, 9, 9], 'visit': [2, 1, 3, 2], 'gene': [5., np.nan, 9., 2.]})
    result = p.earliest_visit(frame).set_index('subject')
    assert result.loc[8, 'visit'] == 1 and np.isnan(result.loc[8, 'gene'])
    assert result.loc[9, 'visit'] == 2 and result.loc[9, 'gene'] == 2


def test_ambiguous_subject_visit_is_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        p.earliest_visit(pd.DataFrame({'subject': [8, 8], 'visit': [1, 1]}))


def test_diabetes_class_definitions_do_not_turn_unknown_into_control():
    labels = pd.DataFrame({'t2ds': ['1', '0', '0', np.nan, '0'], 'pret2ds': ['.', '1', '0', np.nan, '.']})
    np.testing.assert_array_equal(p.diabetes_labels(labels), [0, 1, 2, -1, -1])
    with pytest.raises(ValueError, match='conflict'):
        p.diabetes_labels(pd.DataFrame({'t2ds': ['1'], 'pret2ds': ['1']}))


def test_groups_remain_whole_and_folds_reproduce():
    groups = np.repeat(np.arange(20), 3)
    labels = np.tile([0, 1, 2], 20)
    fold, seed, score = p.assign_folds(labels, groups, base_seed=42, candidates=5)
    repeat, _, _ = p.assign_folds(labels, groups, base_seed=42, candidates=5)
    np.testing.assert_array_equal(fold, repeat)
    assert set(fold) == {1, 2, 3, 4, 5}
    for group in np.unique(groups):
        assert len(set(fold[groups == group])) == 1
    for k in range(1, 6):
        assert set(labels[fold == k]) == {0, 1, 2}


def test_conflicting_family_sources_form_one_conservative_group():
    ped = pd.DataFrame({'subject': [1, 2, 3, 4], 'gpedid': [10, 10, 20, 30]})
    pheno = pd.DataFrame({'subject': [1, 2, 3, 4], 'gpedid': [100, 100, 100, 300]})
    result = p.reconcile_families(ped, pheno).set_index('subject')
    assert result.loc[1, 'family_id'] == result.loc[3, 'family_id']
    assert result.loc[1, 'family_id'] != result.loc[4, 'family_id']


def test_graph_edges_and_labels_use_only_local_subjects():
    bins = np.array([[1, 3], [2, 1]])
    sex = np.array([1, 2])
    graph_x, edges, idx = p.phenotype_graph(np.ones((2, 4)), bins, sex)
    assert graph_x.shape == (10, 4)
    np.testing.assert_array_equal(idx, [8, 9])
    assert edges.min() >= 0 and edges.max() < len(graph_x)
    assert not np.any(np.isin(edges, [10, 11]))
    # Legacy graph reserves sex category nodes but does not connect subjects to them.
    assert not np.isin(edges, [0, 1]).any()
