import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rank_stability import rank_scores, fractional_topk, rank_correlation, load_runs, compare_runs
from rank_stability import main


def test_midrank_and_fractional_boundary_tie():
    ranks = rank_scores(np.array([8.,4.,4.,1.]))
    np.testing.assert_array_equal(ranks,[1,2.5,2.5,4])
    weights,detail = fractional_topk(ranks,2)
    np.testing.assert_array_equal(weights,[1,.5,.5,0])
    assert weights.sum()==2 and detail['boundary_fraction']==.5
    assert not detail['trivial_topk']


def test_constant_rank_is_not_perfect_correlation():
    assert np.isnan(rank_correlation(np.ones(4),np.ones(4)))
    weights,detail = fractional_topk(rank_scores(np.ones(4)),2)
    np.testing.assert_array_equal(weights,[.5]*4)
    assert detail['boundary_tie_size']==4
    assert fractional_topk(np.arange(4),10)[1]['trivial_topk']


def make_run(root,fold,seed,order=None,signature='cohort'):
    folder=root/f'f{fold}s{seed}'
    folder.mkdir()
    methods=['attention_raw','assignment_l2','embedding_x_assignment_l2','embedding_x_softmax_sum']
    classes=['all','t2ds','pret2ds','no_t2ds']
    metadata=dict(schema_version=1,data_signature=signature,model_config={'K':128},condition='full',
                  fold=fold,seed=seed,methods=methods,score_definitions={m:m for m in methods},
                  eligible_gene_indices=[0,1,2],score_head=0,graph_note='incident')
    (folder/'gene_scoring_metadata.json').write_text(json.dumps(metadata))
    pd.DataFrame({'subject':[fold*10,fold*10+1,fold*10+2],'label':[0,1,2]}).to_csv(folder/'scoring_subjects.csv',index=False)
    rows=[dict(Fold=fold,Seed=seed,Condition='full',Class=c,Method=m,Gene_Index=g,Gene=f'G{g}',
               Score=float(3-g),Eligible=True,n_subjects=3 if c=='all' else 1,Rank=99)
          for c in classes for m in methods for g in range(3)]
    frame=pd.DataFrame(rows)
    if order is not None: frame=frame.iloc[order]
    frame.to_csv(folder/'gene_scores.csv',index=False)
    return folder


def test_permutation_alignment_and_separate_comparison_levels(tmp_path):
    dirs=[make_run(tmp_path,f,s,order=np.arange(48)[::-1] if s==2 else None) for f in [1,2] for s in [1,2]]
    runs=load_runs(dirs,allow_incomplete=True)
    output=compare_runs(runs,[1,3])
    pairs=output['pairwise_stability']
    assert set(pairs.level)=={'same_fold_different_seed','same_seed_different_fold'}
    assert set(pairs.spearman.dropna())=={1.}
    assert set(pairs.jaccard)=={1.}
    assert output['rankings'].Rank.max()==3
    assert output['consensus'].shape[0]==48


def test_identity_and_completeness_rejections(tmp_path):
    one=make_run(tmp_path,1,1)
    with pytest.raises(ValueError,match='five'): load_runs([one])
    with pytest.raises(ValueError,match='Duplicate'): load_runs([one,one],allow_incomplete=True)
    other=make_run(tmp_path,2,1,signature='different')
    with pytest.raises(ValueError,match='metadata'): load_runs([one,other],allow_incomplete=True)


def test_roster_overlap_rejected(tmp_path):
    one,two=make_run(tmp_path,1,1),make_run(tmp_path,2,1)
    (two/'scoring_subjects.csv').write_text((one/'scoring_subjects.csv').read_text())
    with pytest.raises(ValueError,match='overlap'): load_runs([one,two],allow_incomplete=True)


def test_constant_scores_have_fractional_frequencies_and_no_informative_pairs(tmp_path):
    dirs=[make_run(tmp_path,1,s) for s in [1,2]]
    for folder in dirs:
        frame=pd.read_csv(folder/'gene_scores.csv')
        frame['Score']=1.
        frame.to_csv(folder/'gene_scores.csv',index=False)
    output=compare_runs(load_runs(dirs,allow_incomplete=True),[1,3])
    pairs=output['pairwise_stability']
    assert pairs.spearman.isna().all()
    assert not pairs.informative_topk.any()
    assert (pairs.jaccard==1).all()  # Symmetric uncertainty, explicitly flagged uninformative.
    assert output['summary'].jaccard_mean.isna().all()
    assert (output['summary'].raw_jaccard_mean==1).all()
    freq=output['frequencies']
    np.testing.assert_allclose(freq.loc[freq.TopK==1,'frequency'],1/3)
    assert (output['ranking_resolution'].tie_fraction==1).all()
    assert (output['summary'].n_defined==0).all()


@pytest.mark.parametrize('change',['eligibility','gene_name','method','roster','counts'])
def test_changed_comparison_universe_or_roster_is_rejected(tmp_path,change):
    one,two=make_run(tmp_path,1,1),make_run(tmp_path,1,2)
    frame=pd.read_csv(two/'gene_scores.csv')
    if change=='eligibility': frame.loc[frame.Gene_Index==0,'Eligible']=False
    elif change=='gene_name': frame.loc[frame.Gene_Index==0,'Gene']='changed'
    elif change=='method': frame=frame[frame.Method!='attention_raw']
    elif change=='counts': frame['n_subjects']=99
    else:
        roster=pd.read_csv(two/'scoring_subjects.csv')
        roster.loc[0,'label']=2
        roster.to_csv(two/'scoring_subjects.csv',index=False)
    frame.to_csv(two/'gene_scores.csv',index=False)
    with pytest.raises(ValueError): load_runs([one,two],allow_incomplete=True)


def test_missing_seed_fold_cell_rejected_even_for_diagnostics(tmp_path):
    dirs=[make_run(tmp_path,f,s) for f,s in [(1,1),(1,2),(2,1)]]
    with pytest.raises(ValueError,match='same seed set'): load_runs(dirs,allow_incomplete=True)


def test_hand_calculated_fractional_overlap_and_consensus(tmp_path):
    one,two=make_run(tmp_path,1,1),make_run(tmp_path,1,2)
    for folder,scores in [(one,[5.,2.,2.]),(two,[2.,5.,2.])]:
        frame=pd.read_csv(folder/'gene_scores.csv')
        frame['Score']=frame.Gene_Index.map(dict(enumerate(scores)))
        frame.to_csv(folder/'gene_scores.csv',index=False)
    result=compare_runs(load_runs([one,two],allow_incomplete=True),[2])
    np.testing.assert_allclose(result['pairwise_stability'].spearman,-.5)
    np.testing.assert_allclose(result['pairwise_stability'].jaccard,1.5/2.5)
    ranks=result['consensus'].query("Class=='all' and Method=='attention_raw'").sort_values('Gene_Index')
    np.testing.assert_allclose(ranks.mean_rank,[1.75,1.75,2.5])
    np.testing.assert_allclose(ranks.Consensus_Rank,[1.5,1.5,3.])


def test_cli_writes_auditable_outputs_and_refuses_overwrite(tmp_path,monkeypatch):
    folder=make_run(tmp_path,1,1)
    output=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['rank_stability','--input',str(folder),'--output',str(output),
                                  '--top-k','1','3','--allow-incomplete'])
    main()
    expected={'rankings.csv','pairwise_stability.csv','frequencies.csv','consensus.csv',
              'fold_consensus.csv','memberships.csv','ranking_resolution.csv','summary.csv','metadata.json'}
    assert {p.name for p in output.iterdir()}==expected
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['n_runs']==1 and metadata['allow_incomplete']
    with pytest.raises(FileExistsError): main()
