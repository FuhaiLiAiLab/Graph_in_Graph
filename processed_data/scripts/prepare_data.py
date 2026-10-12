"""Prepare the approved LLFS family-grouped five-fold inputs. Run from any cwd."""
import argparse
import os
import ast
import csv
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import (assign_folds, bin_numeric, diabetes_labels, earliest_visit,
                           fit_numeric, phenotype_graph, reconcile_families, transform_numeric)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('GIG_DATA_ROOT', ROOT/'processed_data')).resolve()
from source_layout import resolve_source
BASE = Path(os.environ.get('GIG_RAW_ROOT', ROOT/'base_data_1575')).resolve()
CHANNELS = ['rna', 'core_promoter', 'proximal_promoter', 'distal_promoter', 'upstream', 'downstream']
METH_FILES = ['Core_Promoter_final.csv', 'Proximal_Promoter_final.csv',
              'Distal_Promoter_final.csv', 'Upstream_final.csv', 'Downstream_final.csv']
CLASSES = ['t2ds', 'pret2ds', 'no_t2ds']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def log(message):
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with (ROOT / 'change.log').open('a', encoding='utf-8') as handle:
        handle.write(f'\n[{stamp}] {message}\n')
    tqdm.write(message)


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)


def source_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle, tqdm(total=path.stat().st_size, desc=f'Hash {path.name}', unit='B', unit_scale=True, leave=False) as progress:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def source_paths():
    return [BASE/'subject_id_1575.csv', BASE/'pedigree/triplet_visit3.csv',
            BASE/'pheno/LLFS_phenos_21JUN2022.xlsx', BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv',
            *[BASE/'methylation'/name for name in METH_FILES],
            resolve_source('data/label_data/t2dpret2d.txt'), resolve_source('data/kg_data/ensembl/mart_export_genename.txt'),
            resolve_source('data/kg_data/KEGG/full_kegg_pathway_list.csv'), ROOT/'configs/phenotype_features.json']


def notebook_features(phenotype):
    spec=json.loads((ROOT/'configs/phenotype_features.json').read_text())
    features=[name for name in phenotype if '_v1' in name and name not in spec['exclusions']]
    if features != spec['features']:raise ValueError('Phenotype feature order differs from fixed schema')
    return features,spec['exclusions']


def prepare_metadata():
    ids = pd.read_csv(BASE/'subject_id_1575.csv')
    assert len(ids) == ids.subject.nunique() == 1575
    phenotype = pd.read_excel(BASE/'pheno/LLFS_phenos_21JUN2022.xlsx', sheet_name='Phenodata')
    pedigree = pd.read_csv(BASE/'pedigree/triplet_visit3.csv')
    labels = pd.read_csv(resolve_source('data/label_data/t2dpret2d.txt'), sep='\t', dtype=str)
    labels['subject'] = pd.to_numeric(labels.subject).astype('int64')
    if labels.subject.duplicated().any():
        raise ValueError('duplicate label records')
    family = reconcile_families(pedigree, phenotype)
    cohort = ids.merge(family, on='subject', validate='one_to_one')
    assert len(cohort) == 1575
    cohort = cohort.merge(labels[['subject', 't2ds', 'pret2ds']], on='subject', how='left', validate='one_to_one')
    cohort['label'] = diabetes_labels(cohort)
    cohort = cohort.merge(phenotype[['subject', 'sex', 'age_v1', 'fc']], on='subject', validate='one_to_one')
    rna_index = pd.read_csv(BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv', usecols=['subject', 'visit'])
    earliest = earliest_visit(rna_index)
    cohort = cohort.merge(earliest.rename(columns={'visit': 'rna_visit'}), on='subject', validate='one_to_one')
    assert len(cohort) == 1575
    cohort['included'] = cohort.label >= 0
    cohort['exclusion_reason'] = np.where(cohort.included, '', 'missing_or_unknown_diabetes_label')
    cohort = cohort.sort_values('subject').reset_index(drop=True)
    cohort.to_csv(OUT/'metadata/starting_cohort.csv', index=False)
    cohort[~cohort.included].to_csv(OUT/'metadata/excluded_subjects.csv', index=False)
    conflict = cohort[cohort.gpedid_pedigree != cohort.gpedid_phenotype]
    conflict.to_csv(OUT/'metadata/family_conflicts.csv', index=False)
    included = cohort[cohort.included].copy().reset_index(drop=True)
    included.insert(0, 'row_index', np.arange(len(included)))
    included['class_name'] = included.label.map(dict(enumerate(CLASSES)))
    if not included.sex.isin([1, 2]).all():
        raise ValueError('missing or unexpected sex code; do not silently impute categories')
    included.to_csv(OUT/'metadata/cohort.csv', index=False)
    features, exclusions = notebook_features(phenotype)
    raw_pheno = phenotype.set_index('subject').loc[included.subject, features]
    raw_pheno.reset_index().to_csv(OUT/'shared/phenotype_raw.csv', index=False)
    pheno_features = pd.DataFrame({'feature_index': range(len(features) + 2),
                                  'feature_name': features + ['sex_1', 'sex_2'],
                                  'kind': ['continuous'] * len(features) + ['one_hot'] * 2})
    pheno_features.to_csv(OUT/'metadata/phenotype_features.csv', index=False)
    nodes = ['sex-1', 'sex-2'] + [f'{feature}-{k}' for feature in features for k in (1, 2, 3)]
    pd.DataFrame({'subfeature_node_idx': range(len(nodes)), 'subfeature_name': nodes}).to_csv(OUT/'metadata/subfeature_nodes.csv', index=False)
    missing = pd.DataFrame({'feature': features, 'missing_count': raw_pheno.isna().sum().values})
    missing['missing_fraction'] = missing.missing_count / len(included)
    missing.to_csv(OUT/'reports/phenotype_missingness.csv', index=False)
    return included, raw_pheno.to_numpy(dtype=float), features, exclusions


def gene_resources():
    mapping = pd.read_csv(resolve_source('data/kg_data/ensembl/mart_export_genename.txt')).rename(
        columns={'Gene stable ID': 'gene_id', 'Gene name': 'gene_name'})[['gene_id', 'gene_name']].dropna().drop_duplicates()
    mapping.gene_name = mapping.gene_name.astype(str).str.strip()
    mapping = mapping[mapping.gene_name != '']
    kegg = pd.read_csv(resolve_source('data/kg_data/KEGG/full_kegg_pathway_list.csv'), usecols=['source', 'target', 'pathway_name'])
    kegg = kegg[kegg.pathway_name.str.contains('signaling pathway|signaling pathways', case=False, na=False)].dropna()
    kegg = kegg.rename(columns={'source': 'src', 'target': 'dest'})
    for col in ['src', 'dest']:
        kegg[col] = kegg[col].str.upper()
    with (BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv').open(encoding='utf-8-sig') as handle:
        rna_columns = next(csv.reader(handle))
    rna_names = pd.DataFrame({'rna_column': rna_columns[2:]})
    rna_names['gene_id'] = rna_names.rna_column.str.split('.').str[0]
    mapped = rna_names.merge(mapping, on='gene_id', how='inner').drop_duplicates(['rna_column', 'gene_name'])
    common = set(mapped.gene_name)
    inventory = [{'source': 'rna_mapped', 'unique_genes': len(common)}]
    for name in tqdm(METH_FILES, desc='Methylation gene inventories', unit='file'):
        frame = pd.read_csv(BASE/'methylation'/name, usecols=['gene_name'])
        names = set(frame.gene_name.dropna().astype(str))
        common &= names
        inventory.append({'source': name, 'unique_genes': len(names)})
    edges = kegg[kegg.src.isin(common) & kegg.dest.isin(common)][['src', 'dest']].drop_duplicates().sort_values(['src', 'dest'])
    genes = sorted(set(edges.src) | set(edges.dest))
    if not genes:
        raise ValueError('empty gene intersection with KEGG signaling graph')
    mapping_used = mapped[mapped.gene_name.isin(genes)].sort_values(['gene_name', 'rna_column'])
    mapping_used.to_csv(OUT/'metadata/rna_gene_mapping.csv', index=False)
    pd.DataFrame(inventory).to_csv(OUT/'reports/gene_inventory.csv', index=False)
    pd.DataFrame({'gene_index': range(len(genes)), 'gene_name': genes}).to_csv(OUT/'metadata/genes.csv', index=False)
    pd.DataFrame({'channel_index': range(6), 'channel_name': CHANNELS}).to_csv(OUT/'metadata/channels.csv', index=False)
    edges.to_csv(OUT/'shared/gene_edges.csv', index=False)
    numbers = {name: i for i, name in enumerate(genes)}
    edge_index = edges.replace({'src': numbers, 'dest': numbers}).to_numpy(dtype=np.int64).T
    save_array(OUT/'shared/gene_edge_index.npy', edge_index)
    return genes, mapping_used


def prepare_genes(cohort):
    genes, mapping = gene_resources()
    subjects = cohort.subject.to_numpy()
    cols = ['subject', 'visit'] + sorted(mapping.rna_column.unique().tolist())
    log(f'读取最早访视 RNA，映射至 {len(genes)} 个 KEGG signaling 基因。')
    rna = pd.read_csv(BASE/'rna/RNA_seq_residuals_v3_allsubjects.csv', usecols=cols)
    rna = earliest_visit(rna[rna.subject.isin(subjects)]).set_index('subject').loc[subjects]
    np.testing.assert_array_equal(rna.visit, cohort.rna_visit)
    transposed = rna.drop(columns='visit').T
    transposed.index.name = 'rna_column'
    named = mapping[['rna_column', 'gene_name']].merge(transposed.reset_index(), on='rna_column', validate='many_to_one')
    rna_values = named.drop(columns='rna_column').groupby('gene_name', sort=True).mean().reindex(genes).T.to_numpy(dtype=float)
    matrices = [rna_values]
    for name in tqdm(METH_FILES, desc='Prepare methylation channels', unit='channel'):
        log(f'分块读取甲基化 {name}，按 gene_name 汇总并对齐受试者。')
        pieces = []
        subject_columns = [str(s) for s in subjects]
        for chunk in tqdm(pd.read_csv(BASE/'methylation'/name, usecols=['gene_name'] + subject_columns, chunksize=2000), desc=f'Read {name}', unit='chunk', leave=False):
            selected = chunk[chunk.gene_name.isin(genes)]
            if len(selected):
                pieces.append(selected)
        merged = pd.concat(pieces, ignore_index=True).groupby('gene_name', sort=True)[subject_columns].mean()
        values = merged.reindex(genes).T.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and (finite.min() < -1e-8 or finite.max() > 1 + 1e-8):
            raise ValueError(f'non-beta values in {name}')
        matrices.append(values)
    # Explicit gene-major layout, directly compatible with reshape(N, G, 6).
    gene = np.stack(matrices, axis=2)
    if gene.shape != (len(cohort), len(genes), 6) or np.isinf(gene).any():
        raise ValueError('invalid aligned gene tensor')
    save_array(OUT/'shared/gene_raw.npy', gene)
    write_json(OUT/'reports/omics_quality.json', {
        channel: {'missing_values': int(np.isnan(gene[:, :, i]).sum()),
                  'min': float(np.nanmin(gene[:, :, i])), 'max': float(np.nanmax(gene[:, :, i]))}
        for i, channel in enumerate(CHANNELS)})
    return gene


def export_fold(cohort, gene, raw_pheno, features, fold, k):
    train_index, test_index = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    directory = OUT/f'folds/fold_{k:02d}'
    params = directory/'preprocessing'
    params.mkdir(parents=True, exist_ok=True)
    gene_flat = gene.reshape(len(gene), -1)
    gene_fit, pheno_fit = fit_numeric(gene_flat[train_index]), fit_numeric(raw_pheno[train_index])
    np.savez(params/'gene_parameters.npz', **gene_fit)
    np.savez(params/'phenotype_parameters.npz', **pheno_fit)
    save_array(params/'fit_subject_ids.npy', cohort.subject.to_numpy()[train_index])
    pd.DataFrame({'feature': features, **{name: value for name, value in pheno_fit.items()}}).to_csv(params/'phenotype_parameters.csv', index=False)
    quality = {}
    for name, indices in [('train', train_index), ('test', test_index)]:
        target = directory/name
        target.mkdir(parents=True, exist_ok=True)
        subset = cohort.iloc[indices].copy()
        subset['local_row_index'] = np.arange(len(subset))
        labels = subset.label.to_numpy(dtype=np.int64)
        sex = subset.sex.to_numpy(dtype=np.int64)
        pheno = np.column_stack([transform_numeric(raw_pheno[indices], pheno_fit), sex == 1, sex == 2]).astype(np.float32)
        gene_x = transform_numeric(gene_flat[indices], gene_fit).astype(np.float32)
        bins = bin_numeric(raw_pheno[indices], pheno_fit)
        graph_x, edge_index, node_index = phenotype_graph(pheno, bins, sex)
        subset['local_node_index'] = node_index
        subset.to_csv(target/'subjects.csv', index=False)
        arrays = {'subject_ids': subset.subject.to_numpy(dtype=np.int64), 'labels': labels,
                  'labels_onehot': np.eye(3, dtype=np.int64)[labels], 'gene_x': gene_x,
                  'phenotype_x': pheno, 'phenotype_bins': bins, 'graph_x': graph_x,
                  'edge_index': edge_index, 'node_index': node_index}
        for filename, array in arrays.items():
            save_array(target/f'{filename}.npy', array)
        quality[name] = {'n_subjects': len(subset), 'n_families': int(subset.family_id.nunique()),
                         'class_counts': subset.class_name.value_counts().to_dict(),
                         'gene_scaled_below_zero': int((gene_x < 0).sum()),
                         'gene_scaled_above_one': int((gene_x > 1).sum()),
                         'phenotype_scaled_below_zero': int((pheno < 0).sum()),
                         'phenotype_scaled_above_one': int((pheno > 1).sum())}
    quality['constant_gene_features'] = int(gene_fit['constant'].sum())
    quality['constant_phenotype_features'] = int(pheno_fit['constant'].sum())
    write_json(directory/'summary.json', quality)
    log(f'导出 fold_{k:02d}：训练 {len(train_index)} 人，评估 {len(test_index)} 人；独立保存训练拟合参数。')
    return quality


def main():
    with tqdm(total=6, desc='Preparation: verify inputs', unit='stage') as progress:
        _prepare(progress)


def _prepare(progress):
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-shared', action='store_true', help='Reuse aligned gene_raw only after verifying source hashes')
    args = parser.parse_args()
    for directory in ['metadata', 'shared', 'splits', 'reports']:
        (OUT/directory).mkdir(parents=True, exist_ok=True)
    if (OUT/'reports/validation.json').exists():
        raise RuntimeError('validated output exists; use a new versioned directory rather than overwrite it')
    log('开始运行 prepare_data.py；数据产物限定在 revision/processed_data。')
    manifest = []
    for path in tqdm(source_paths(), desc='Hash source files', unit='file'):
        if not path.exists():
            raise FileNotFoundError(path)
        manifest.append({'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': source_hash(path)})
    old_manifest = OUT/'metadata/source_manifest.json'
    if args.reuse_shared and (not old_manifest.exists() or json.loads(old_manifest.read_text(encoding='utf-8')) != manifest):
        raise RuntimeError('cannot reuse shared data: input manifest mismatch')
    write_json(old_manifest, manifest)
    progress.update(1)
    progress.set_description('Preparation: read cohort metadata and RNA index')
    cohort, pheno, features, exclusions = prepare_metadata()
    progress.update(1)
    progress.set_description('Preparation: align omics channels')
    gene = np.load(OUT/'shared/gene_raw.npy') if args.reuse_shared else prepare_genes(cohort)
    assert len(gene) == len(cohort)
    progress.update(1)
    progress.set_description('Preparation: assign family folds and write configuration')
    fold, selected_seed, score = assign_folds(cohort.label, cohort.family_id)
    split = cohort[['subject', 'family_id', 'gpedid_pedigree', 'gpedid_phenotype', 'label', 'class_name']].copy()
    split['fold'] = fold
    split.to_csv(OUT/'splits/subject_folds.csv', index=False)
    family_folds = split.groupby('family_id').agg(fold=('fold', 'first'), n_subjects=('subject', 'size')).reset_index()
    assert split.groupby('family_id').fold.nunique().max() == 1
    family_folds.to_csv(OUT/'splits/family_folds.csv', index=False)
    config = {'schema_version': 1, 'n_splits': 5, 'base_seed': 20260914, 'split_candidates': 100,
              'selected_seed': selected_seed, 'split_balance_score': score,
              'split_score': 'mean squared relative class-count deviation + mean squared relative fold-size deviation; lowest score, first seed breaks ties',
              'target_classes': dict(enumerate(CLASSES)), 'continuous_phenotypes': features,
              'legacy_excluded_phenotypes': exclusions, 'sex_categories': [1, 2],
              'rna_visit_rule': 'earliest numeric visit per subject; whole row; no per-feature visit backfill',
              'family_rule': 'connected membership across source-namespaced pedigree/phenotype family IDs plus recorded parent links',
              'channels': CHANNELS, 'gene_layout': 'gene-major; gene_x.reshape(n_subjects,n_genes,6)',
              'imputation': 'per-training-feature mean', 'scaling': 'per-training-feature min-max; no clipping; constant training features map to zero',
              'binning': 'training-imputed q10/q90; right-closed; tied thresholds use equality as middle; keep three categories',
              'phenotype_graph': 'legacy directed continuous-category chains and subject->category links; sex nodes reserved and disconnected as in old notebook; separate train/test subject nodes',
              'n_starting_subjects': 1575, 'n_subjects': len(cohort), 'n_genes': gene.shape[1],
              'n_continuous_phenotypes': len(features), 'n_phenotype_features': len(features) + 2,
              'n_subfeature_nodes': 2 + 3 * len(features), 'n_family_groups': int(cohort.family_id.nunique()),
              'source_manifest': 'metadata/source_manifest.json',
              'limitations': ['No independent final evaluation if these five folds are also used for hyperparameter selection.',
                              'Existing RNA residualization fit scope unverified: generation script unavailable in this workspace.',
                              'Official precomputed phenotype transformations inherited; their fitting scope is not refitted here.',
                              'Earliest available RNA visit can differ from phenotype visit 1; methylation visit metadata is unavailable in supplied matrices.']}
    write_json(OUT/'config.json', config)
    progress.update(1)
    progress.set_description('Preparation: fit and export folds')
    summaries = [export_fold(cohort, gene, pheno, features, fold, k) for k in tqdm(range(1, 6), desc='Export prepared folds', unit='fold')]
    progress.update(1)
    progress.set_description('Preparation: write reports')
    rows = []
    for k, summary in enumerate(summaries, 1):
        for role in ['train', 'test']:
            part = summary[role]
            rows.append({'fold': k, 'role': role, 'n_subjects': part['n_subjects'], 'n_families': part['n_families'], **part['class_counts']})
    pd.DataFrame(rows).to_csv(OUT/'splits/fold_summary.csv', index=False)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'openpyxl']}
    write_json(OUT/'reports/software_versions.json', versions)
    write_json(OUT/'reports/preparation_summary.json', {'status': 'prepared_pending_independent_validation',
               'n_subjects': len(cohort), 'class_counts': cohort.class_name.value_counts().to_dict(),
               'rna_visit_counts': cohort.rna_visit.value_counts().to_dict(),
               'n_genes': gene.shape[1], 'n_family_groups': int(cohort.family_id.nunique()),
               'selected_seed': selected_seed, 'split_score': score})
    progress.update(1)
    log('五折数据生成结束，等待独立 validate_data.py 核验；此阶段不训练模型。')


if __name__ == '__main__':
    main()
