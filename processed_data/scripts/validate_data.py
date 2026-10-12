"""Independent checks: recompute training statistics without preprocessing helpers."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

OUT = Path(os.environ.get('GIG_DATA_ROOT',Path(__file__).resolve().parents[1])).resolve()
ROOT = Path(__file__).resolve().parents[2]
from source_layout import resolve_source
BASE = Path(os.environ.get('GIG_RAW_ROOT',ROOT/'base_data_1575')).resolve()


def load(path):
    return np.load(path, allow_pickle=False)


def verify_parameters(values, indices, path):
    with np.load(path, allow_pickle=False) as archive:
        saved = dict(archive)
    train = values[indices]
    assert not np.all(np.isnan(train), axis=0).any()
    mean = np.nanmean(train, axis=0)
    filled = np.where(np.isnan(train), mean, train)
    expected = {'mean': mean, 'min': filled.min(axis=0), 'max': filled.max(axis=0),
                'q10': np.quantile(filled, .1, axis=0, method='linear'),
                'q90': np.quantile(filled, .9, axis=0, method='linear')}
    expected['constant'] = expected['min'] == expected['max']
    assert set(saved) == set(expected)
    for key in expected:
        np.testing.assert_allclose(saved[key], expected[key], rtol=1e-11, atol=1e-11)
    return expected


def expected_transform(raw, params):
    filled = np.where(np.isnan(raw), params['mean'], raw)
    width = params['max'] - params['min']
    width[width == 0] = 1
    result = (filled - params['min']) / width
    result[:, params['constant']] = 0
    return result.astype(np.float32)


def check_source_alignment(cohort, raw, genes):
    """Independently sample actual source values across people, genes and all channels."""
    rows = [0, len(cohort)//2, len(cohort)-1]
    gene_positions = [0, len(genes)//2, len(genes)-1]
    subjects = cohort.subject.iloc[rows].to_numpy()
    names = genes.gene_name.iloc[gene_positions].tolist()
    mapping = pd.read_csv(OUT/'metadata/rna_gene_mapping.csv')
    subset = mapping[mapping.gene_name.isin(names)]
    columns = ['subject', 'visit'] + sorted(subset.rna_column.unique().tolist())
    rna = pd.read_csv(BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv', usecols=columns)
    rna = rna[rna.subject.isin(subjects)].sort_values(['subject', 'visit']).drop_duplicates('subject').set_index('subject')
    for row, subject in zip(rows, subjects):
        assert rna.loc[subject, 'visit'] == cohort.iloc[row].rna_visit
        for g, name in zip(gene_positions, names):
            cols = subset[subset.gene_name == name].rna_column.tolist()
            expected = np.nanmean(rna.loc[subject, cols].to_numpy(dtype=float))
            np.testing.assert_allclose(raw[row, g, 0], expected, rtol=1e-12, atol=1e-12)
    paths = ['Core_Promoter_final.csv', 'Proximal_Promoter_final.csv', 'Distal_Promoter_final.csv', 'Upstream_final.csv', 'Downstream_final.csv']
    for channel, filename in enumerate(tqdm(paths, desc='Methylation source alignment', unit='file'), 1):
        table = pd.read_csv(BASE/'methylation'/filename, usecols=['gene_name'] + [str(s) for s in subjects])
        for row, subject in zip(rows, subjects):
            for g, name in zip(gene_positions, names):
                expected = table.loc[table.gene_name == name, str(subject)].mean()
                np.testing.assert_allclose(raw[row, g, channel], expected, rtol=1e-12, atol=1e-12)
    return len(rows) * len(gene_positions) * 6


def main():
    with tqdm(total=5, desc='Validation: read artifacts and check identities', unit='stage') as progress:
        _validate(progress)


def _validate(progress):
    config = json.loads((OUT/'config.json').read_text(encoding='utf-8'))
    cohort = pd.read_csv(OUT/'metadata/cohort.csv')
    starting = pd.read_csv(OUT/'metadata/starting_cohort.csv')
    excluded = pd.read_csv(OUT/'metadata/excluded_subjects.csv')
    splits = pd.read_csv(OUT/'splits/subject_folds.csv')
    genes = pd.read_csv(OUT/'metadata/genes.csv')
    raw = load(OUT/'shared/gene_raw.npy')
    flat = raw.reshape(len(cohort), -1)
    pheno_frame = pd.read_csv(OUT/'shared/phenotype_raw.csv')
    pheno = pheno_frame.drop(columns='subject').to_numpy()
    assert list(pheno_frame.columns[1:]) == config['continuous_phenotypes']
    np.testing.assert_array_equal(pheno_frame.subject, cohort.subject)
    np.testing.assert_array_equal(splits.subject, cohort.subject)
    assert len(starting) == starting.subject.nunique() == 1575
    assert set(cohort.subject) | set(excluded.subject) == set(starting.subject)
    assert not set(cohort.subject) & set(excluded.subject)
    assert len(cohort) == cohort.subject.nunique() == config['n_subjects']
    assert raw.shape == (len(cohort), len(genes), 6)
    assert len(genes) == genes.gene_name.nunique() == config['n_genes']
    assert genes.gene_name.tolist() == sorted(genes.gene_name)
    assert not np.isinf(raw).any() and not np.isinf(pheno).any()
    source_pheno = pd.read_excel(BASE/'pheno/LLFS_phenos_21JUN2022.xlsx', sheet_name='Phenodata').set_index('subject')
    np.testing.assert_allclose(pheno, source_pheno.loc[cohort.subject, config['continuous_phenotypes']].to_numpy(), equal_nan=True)
    source_labels = pd.read_csv(resolve_source('data/label_data/t2dpret2d.txt'), sep='\t', dtype=str).set_index('subject')
    src = source_labels.loc[cohort.subject.astype(str)]
    expected_labels = np.full(len(src), -1)
    expected_labels[(src.t2ds == '1').to_numpy()] = 0
    expected_labels[((src.t2ds == '0') & (src.pret2ds == '1')).to_numpy()] = 1
    expected_labels[((src.t2ds == '0') & (src.pret2ds == '0')).to_numpy()] = 2
    np.testing.assert_array_equal(expected_labels, cohort.label)
    assert (expected_labels >= 0).all()
    for column in ['family_id', 'gpedid_pedigree', 'gpedid_phenotype']:
        assert splits.groupby(column).fold.nunique().max() == 1, column
    assert set(splits.fold) == {1, 2, 3, 4, 5}
    rna_index = pd.read_csv(BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv', usecols=['subject', 'visit'])
    assert not rna_index.duplicated(['subject', 'visit']).any()
    np.testing.assert_array_equal(cohort.rna_visit, rna_index.groupby('subject').visit.min().loc[cohort.subject])
    methyl_ids = None
    for path in (BASE/'methylation').glob('*_final.csv'):
        with path.open(encoding='utf-8-sig') as handle:
            header = next(csv.reader(handle))
        current = {int(x) for x in header[1:]}
        if methyl_ids is not None:
            assert current == methyl_ids
        methyl_ids = current
    common = set(rna_index.subject) & set(source_pheno.index) & methyl_ids
    assert common == set(starting.subject)
    gene_edges = load(OUT/'shared/gene_edge_index.npy')
    assert gene_edges.shape[0] == 2 and gene_edges.min() >= 0 and gene_edges.max() < len(genes)
    edge_names = pd.read_csv(OUT/'shared/gene_edges.csv')
    np.testing.assert_array_equal(genes.gene_name.to_numpy()[gene_edges.T], edge_names[['src','dest']].to_numpy())
    # Verify the shared graph against the actual external signaling resource.
    kegg = pd.read_csv(resolve_source('data/kg_data/KEGG/full_kegg_pathway_list.csv'), usecols=['source','target','pathway_name'])
    kegg = kegg[kegg.pathway_name.str.contains('signaling pathway|signaling pathways', case=False, na=False)].dropna()
    kegg.source, kegg.target = kegg.source.str.upper(), kegg.target.str.upper()
    wanted = kegg[kegg.source.isin(genes.gene_name) & kegg.target.isin(genes.gene_name)][['source','target']].drop_duplicates().sort_values(['source','target'])
    np.testing.assert_array_equal(wanted.to_numpy(), edge_names[['src','dest']].to_numpy())
    folds_checked, all_test = [], []
    progress.update(1)
    progress.set_description('Validation: recompute fold parameters and transforms')
    for k in tqdm(range(1, 6), desc='Validate prepared folds', unit='fold'):
        folder = OUT/f'folds/fold_{k:02d}'
        train = np.flatnonzero(splits.fold.to_numpy() != k)
        test = np.flatnonzero(splits.fold.to_numpy() == k)
        assert not set(cohort.family_id.iloc[train]) & set(cohort.family_id.iloc[test])
        np.testing.assert_array_equal(load(folder/'preprocessing/fit_subject_ids.npy'), cohort.subject.to_numpy()[train])
        gp = verify_parameters(flat, train, folder/'preprocessing/gene_parameters.npz')
        pp = verify_parameters(pheno, train, folder/'preprocessing/phenotype_parameters.npz')
        for role, idx in [('train', train), ('test', test)]:
            d = folder/role
            np.testing.assert_array_equal(load(d/'subject_ids.npy'), cohort.subject.to_numpy()[idx])
            np.testing.assert_array_equal(load(d/'labels.npy'), cohort.label.to_numpy()[idx])
            np.testing.assert_array_equal(load(d/'labels_onehot.npy'), np.eye(3, dtype=np.int64)[cohort.label.to_numpy()[idx]])
            gx = load(d/'gene_x.npy')
            np.testing.assert_allclose(gx, expected_transform(flat[idx], gp), atol=2e-6, rtol=2e-6)
            assert gx.shape == (len(idx), len(genes)*6) and np.isfinite(gx).all()
            px = load(d/'phenotype_x.npy')
            np.testing.assert_allclose(px[:, :-2], expected_transform(pheno[idx], pp), atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(px[:, -2:], np.column_stack([cohort.sex.to_numpy()[idx] == 1, cohort.sex.to_numpy()[idx] == 2]))
            filled = np.where(np.isnan(pheno[idx]), pp['mean'], pheno[idx])
            bins = np.full(filled.shape, 2, dtype=np.int8)
            bins[filled <= pp['q10']] = 1
            bins[filled > pp['q90']] = 3
            ties = pp['q10'] == pp['q90']
            bins[:, ties] = np.sign(filled[:, ties] - pp['q10'][ties]).astype(np.int8) + 2
            np.testing.assert_array_equal(load(d/'phenotype_bins.npy'), bins)
            graph = load(d/'graph_x.npy')
            nsub = config['n_subfeature_nodes']
            assert graph.shape == (nsub + len(idx), px.shape[1])
            np.testing.assert_array_equal(graph[:nsub], 0.)
            np.testing.assert_array_equal(graph[nsub:], px)
            np.testing.assert_array_equal(load(d/'node_index.npy'), np.arange(nsub, nsub+len(idx)))
            edges = load(d/'edge_index.npy')
            assert edges.shape == (2, 2*pheno.shape[1] + len(idx)*pheno.shape[1])
            assert edges.min() >= 0 and edges.max() < len(graph)
            subject_edges = edges[:, 2*pheno.shape[1]:]
            np.testing.assert_array_equal(subject_edges[0], np.repeat(np.arange(nsub, nsub+len(idx)), pheno.shape[1]))
            np.testing.assert_array_equal(subject_edges[1].reshape(len(idx), -1), 2+3*np.arange(pheno.shape[1])[None,:]+bins-1)
            assert not np.isin(edges, [0, 1]).any()
        all_test.extend(cohort.subject.to_numpy()[test].tolist())
        folds_checked.append({'fold': k, 'train_n': len(train), 'test_n': len(test), 'family_overlap': 0,
                              'all_training_parameters_recomputed': True, 'all_saved_transforms_checked': True})
        tqdm.write(f'Validated fold {k}')
    assert len(all_test) == len(set(all_test)) == len(cohort)
    progress.update(1)
    progress.set_description('Validation: check original source alignment')
    sample_count = check_source_alignment(cohort, raw, genes)
    progress.update(1)
    progress.set_description('Validation: verify source hashes')
    manifest = json.loads((OUT/'metadata/source_manifest.json').read_text(encoding='utf-8'))
    for record in tqdm(manifest, desc='Verify source files', unit='file'):
        path = resolve_source(record['path'])
        digest = hashlib.sha256()
        with path.open('rb') as handle, tqdm(total=path.stat().st_size, desc=f'Hash {path.name}', unit='B', unit_scale=True, leave=False) as hash_progress:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
                hash_progress.update(len(block))
        assert path.stat().st_size == record['bytes']
        assert digest.hexdigest() == record['sha256'], path
    progress.update(1)
    progress.set_description('Validation: write reports')
    report = {'status': 'PASS', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'prepared artifacts, cohort/family integrity, source alignment, fold-local preprocessing; not model performance or upstream residualization',
              'n_starting_subjects': len(starting), 'n_included': len(cohort), 'n_excluded': len(excluded),
              'n_family_groups': int(cohort.family_id.nunique()), 'n_genes': len(genes),
              'n_phenotype_features': pheno.shape[1]+2, 'source_alignment_values_checked': sample_count,
              'source_files_hash_verified': len(manifest), 'folds': folds_checked,
              'limitations': config['limitations']}
    (OUT/'reports/validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    summary_path = OUT/'reports/preparation_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary['status'] = 'validated'
    summary['validation_report'] = 'reports/validation.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    code_manifest = []
    for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
        code_manifest.append({'path': 'processed_data/scripts/'+path.name,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/'reports/code_manifest.json').write_text(json.dumps(code_manifest, indent=2)+'\n', encoding='utf-8')
    with (ROOT/'change.log').open('a', encoding='utf-8') as handle:
        handle.write(f"\n[{report['timestamp_utc']}] 独立数据核验 PASS：{len(cohort)} 人、{len(genes)} 基因、5 折；训练参数逐项重算、输出变换全量核对、家系零交叉、源文件哈希一致。范围及上游处理限制见 reports/validation.json。\n")
    print(json.dumps({key: report[key] for key in ['status','n_included','n_family_groups','n_genes','n_phenotype_features']}), flush=True)
    progress.update(1)


if __name__ == '__main__':
    main()
