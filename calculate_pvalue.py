"""Exploratory independent-subject MWU tests on prepared, unscaled omics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / 'processed_data'
COMPARISONS = [('TvsNO', 0, 2), ('TvsPre', 0, 1), ('PrevsNO', 1, 2)]


def calculate_comparison(raw, labels, gene_names, group_one, group_two, progress_label=None):
    """Two-sided MWU; BH separately over finite gene tests for this comparison."""
    raw, labels = np.asarray(raw), np.asarray(labels)
    if raw.shape != (len(labels), len(gene_names)) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('raw data, gene names and class labels are not aligned')
    if np.isinf(raw).any():
        raise ValueError('infinite raw values are invalid')
    rows = []
    for column, name in enumerate(tqdm(gene_names, desc=progress_label, unit='gene',
                                     disable=progress_label is None, leave=False)):
        left, right = raw[labels == group_one, column], raw[labels == group_two, column]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        stat, pvalue = (mannwhitneyu(left, right, alternative='two-sided', method='auto')
                        if len(left) and len(right) else (np.nan, np.nan))
        rows.append({'gene': name, 'p_value': pvalue, 'U_statistic': stat,
                     'n_group_one': len(left), 'n_group_two': len(right),
                     'status': 'tested' if len(left) and len(right) else 'insufficient_observations'})
    result = pd.DataFrame(rows)
    result['q_value_BH'] = np.nan
    finite = np.isfinite(result.p_value)
    if finite.any():
        result.loc[finite, 'q_value_BH'] = multipletests(result.loc[finite, 'p_value'], method='fdr_bh')[1]
    return result


def process_prepared(data_root, output, subject_ids=None):
    data_root, output = Path(data_root), Path(output)
    cohort = pd.read_csv(data_root / 'metadata/cohort.csv', dtype={'subject': str})
    genes = pd.read_csv(data_root / 'metadata/genes.csv')
    channels = pd.read_csv(data_root / 'metadata/channels.csv')
    raw = np.load(data_root / 'shared/gene_raw.npy', mmap_mode='r', allow_pickle=False)
    if (cohort.subject.duplicated().any() or genes.gene_name.duplicated().any()
            or not np.array_equal(cohort.row_index, np.arange(len(cohort)))
            or not np.array_equal(genes.gene_index, np.arange(len(genes)))
            or not np.array_equal(channels.channel_index, np.arange(len(channels)))
            or raw.shape != (len(cohort), len(genes), len(channels))):
        raise ValueError('prepared raw tensor and metadata identities are inconsistent')
    selected = np.ones(len(cohort), dtype=bool)
    if subject_ids is not None:
        wanted = [str(value) for value in subject_ids]
        if len(wanted) != len(set(wanted)) or not set(wanted) <= set(cohort.subject):
            raise ValueError('subject subset must contain unique known subject IDs')
        selected = cohort.subject.isin(wanted).to_numpy()
    if not selected.any():
        raise ValueError('empty subject subset')
    output.mkdir(parents=True, exist_ok=True)
    labels = cohort.loc[selected, 'label'].to_numpy()
    with tqdm(total=len(channels) * len(COMPARISONS), desc='Omics comparisons', unit='comparison') as comparisons:
        for channel in channels.itertuples(index=False):
            # Keep the old RNA output name while retaining the prepared channel name in metadata.
            region = 'tran_v1' if channel.channel_name == 'rna' else channel.channel_name
            values = np.asarray(raw[:, :, channel.channel_index])[selected]
            for name, left, right in COMPARISONS:
                folder = output / name
                folder.mkdir(exist_ok=True)
                result = calculate_comparison(values, labels, genes.gene_name.tolist(), left, right,
                                              progress_label=f'{region}: {name}')
                result.to_csv(folder / f'{name}_{region}_pvalues.csv', index=False)
                comparisons.update(1)
    cohort.loc[selected, ['subject', 'family_id', 'label', 'class_name']].to_csv(output / 'subjects.csv', index=False)
    report = {
        'scope': 'exploratory independent-subject association; not causal or family-adjusted inference',
        'data': 'shared/gene_raw.npy; no fold scaling, imputation or attention weights',
        'n_subjects': int(selected.sum()), 'n_family_groups': int(cohort.loc[selected, 'family_id'].nunique()),
        'subset': 'all prepared cohort subjects' if subject_ids is None else 'explicit subject list',
        'test': 'scipy.stats.mannwhitneyu, two-sided, method=auto, continuity correction default',
        'missing_values': 'omit NaNs separately per gene and group; report usable counts; no imputation',
        'multiplicity': 'Benjamini-Hochberg within each region/comparison over finite tested p-values',
        'channels': channels.to_dict(orient='records'), 'rna_filename_alias': 'tran_v1',
        'limitations': ['Family-grouped model splits do not make related subjects statistically independent.',
                       'These MWU p-values are exploratory; no family dependence correction is applied.',
                       'BH is not applied jointly across regions or comparisons.'],
    }
    (output / 'analysis_metadata.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'pvalues_output')
    parser.add_argument('--subjects', type=Path, help='Optional CSV containing a subject column; default entire cohort')
    args = parser.parse_args()
    subjects = pd.read_csv(args.subjects, dtype={'subject': str}).subject.tolist() if args.subjects else None
    print(json.dumps(process_prepared(args.data_root, args.output, subjects), indent=2))


if __name__ == '__main__':
    main()
