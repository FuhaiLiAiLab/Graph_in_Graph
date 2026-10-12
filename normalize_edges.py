"""Keyed gene-edge normalization and explicit averaging of held-out fold attention."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / 'processed_data'
EDGE_KEYS = ['Actual_From', 'Actual_To']
GROUP_KEYS = ['Class', 'Head']


def normalize_frame(frame, edge_index, num_nodes, mode):
    """Normalize one class/layer/head; degrees are indexed by actual endpoints."""
    if mode not in {'raw', 'minmax', 'binorm', 'rownorm'}:
        raise ValueError(f'unknown mode: {mode}')
    result = frame.copy()
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.size == 0:
        raise ValueError('gene edge_index must be nonempty 2 x E')
    if edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError('gene edge index outside node range')
    if result.duplicated(EDGE_KEYS).any():
        raise ValueError('duplicate edges within one class/layer/head')
    wanted = set(map(tuple, edges.T.tolist()))
    actual = set(map(tuple, result[EDGE_KEYS].to_numpy().tolist()))
    if actual != wanted:
        raise ValueError('attention edges do not match the shared gene graph')
    if not np.isfinite(result.Weight).all():
        raise ValueError('nonfinite attention weights')
    source, target = result.Actual_From.to_numpy(dtype=int), result.Actual_To.to_numpy(dtype=int)
    result['Degree_Row'] = np.bincount(edges[0], minlength=num_nodes)[source]
    result['Degree_Col'] = np.bincount(edges[1], minlength=num_nodes)[target]
    if mode == 'minmax':
        low = result.groupby('Actual_From').Weight.transform('min')
        width = result.groupby('Actual_From').Weight.transform('max') - low
        result['Weight'] = np.divide(result.Weight - low, width,
                                      out=np.zeros(len(result)), where=width.to_numpy() != 0)
    elif mode == 'binorm':
        result['Weight'] /= np.sqrt(result.Degree_Row * result.Degree_Col)
    elif mode == 'rownorm':
        result['Weight'] /= result.Degree_Row
    return result


def average_folds(frame, weighting='equal'):
    """Average edge weights by identity; never average integer IDs or row positions."""
    if weighting not in {'equal', 'subjects'}:
        raise ValueError('weighting must be equal or subjects')
    keys = ['Layer'] + GROUP_KEYS + EDGE_KEYS
    if frame.duplicated(['Fold'] + keys).any():
        raise ValueError('duplicate fold/class/layer/head/edge; supply one run per fold')
    data = frame.copy()
    data['_weight'] = data.n_subjects if weighting == 'subjects' else 1
    data['_weighted_value'] = data.Weight * data._weight
    result = data.groupby(keys, as_index=False).agg(
        _weighted_value=('_weighted_value', 'sum'), _weight=('_weight', 'sum'),
        n_subjects=('n_subjects', 'sum'), n_folds=('Fold', 'nunique'))
    result['Weight'] = result._weighted_value / result._weight
    return result.drop(columns=['_weighted_value', '_weight'])


def node_scores(edges):
    """Legacy score: equal mean of incoming mean and outgoing mean when available."""
    incoming = edges.groupby('Actual_To').Weight.mean().rename_axis('Node_idx')
    outgoing = edges.groupby('Actual_From').Weight.mean().rename_axis('Node_idx')
    return pd.concat([incoming, outgoing]).groupby(level=0).mean().reset_index()


def process(inputs, data_root, output, modes=('raw', 'minmax', 'binorm', 'rownorm'),
            weighting='equal', head=0, percentile=85, expected_layers=None):
    if not 0 <= percentile <= 100:
        raise ValueError('percentile must be within 0..100')
    paths = [Path(path) / 'gene_attention_summary.csv' if Path(path).is_dir() else Path(path) for path in inputs]
    frame = pd.concat([pd.read_csv(path) for path in tqdm(paths, desc='Read attention files', unit='file')], ignore_index=True)
    frame = frame[frame.Head == head].copy()
    if frame.empty:
        raise ValueError(f'no gene attention for head {head}; subject-only models have no gene attention')
    if not (frame.n_subjects > 0).all() or not np.isfinite(frame.Weight).all():
        raise ValueError('invalid class counts or attention weights')
    # All observed folds must provide the same class/layer/head/edge keys.
    key_columns = ['Layer'] + GROUP_KEYS + EDGE_KEYS
    fold_keys = [set(map(tuple, part[key_columns].to_numpy().tolist())) for _, part in frame.groupby('Fold')]
    if any(keys != fold_keys[0] for keys in fold_keys):
        raise ValueError('folds have different classes/layers/edges; cannot silently average missing groups')
    required_layers = list(expected_layers) if expected_layers is not None else list(['gene_first', 'gene_block', 'gene_last'])
    if not required_layers or len(set(required_layers)) != len(required_layers) or set(frame.Layer) != set(required_layers):
        raise ValueError('exported layers differ from explicit expected layers')
    if frame.duplicated(['Fold'] + key_columns).any():
        raise ValueError('duplicate fold/layer/class/head/edge; normalize each seed separately')
    edge_index = np.load(Path(data_root) / 'shared/gene_edge_index.npy', allow_pickle=False)
    genes = pd.read_csv(Path(data_root) / 'metadata/genes.csv')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for mode in tqdm(modes, desc='Normalization modes', unit='mode'):
        folder = output / mode
        folder.mkdir(exist_ok=True)
        parts = []
        groups = frame.groupby(['Fold', 'Layer'] + GROUP_KEYS)
        for _, part in tqdm(groups, total=groups.ngroups, desc=f'{mode}: fold/layer/class groups', unit='group', leave=False):
            parts.append(normalize_frame(part, edge_index, len(genes), mode))
        normalized = pd.concat(parts, ignore_index=True)
        normalized.to_csv(folder / 'fold_gene_edges.csv', index=False)
        averaged = average_folds(normalized, weighting)
        averaged.to_csv(folder / 'layer_gene_edges.csv', index=False)
        final = averaged.groupby(GROUP_KEYS + EDGE_KEYS, as_index=False).agg(Weight=('Weight', 'mean'))
        final['From_Gene'] = genes.gene_name.to_numpy()[final.Actual_From.to_numpy(dtype=int)]
        final['To_Gene'] = genes.gene_name.to_numpy()[final.Actual_To.to_numpy(dtype=int)]
        final.to_csv(folder / 'gene_edges.csv', index=False)
        for (class_name, selected_head), part in final.groupby(GROUP_KEYS):
            scores = node_scores(part)
            scores['Gene'] = genes.gene_name.to_numpy()[scores.Node_idx.to_numpy(dtype=int)]
            prefix = f'{class_name}_head{selected_head}'
            scores.to_csv(folder / f'{prefix}_node_weights.csv', index=False)
            part[part.Weight >= np.percentile(part.Weight, percentile)].to_csv(folder / f'{prefix}_filtered_edges.csv', index=False)
            scores[scores.Weight >= np.percentile(scores.Weight, percentile)].to_csv(folder / f'{prefix}_filtered_nodes.csv', index=False)
    report = {'inputs': [str(path.resolve()) for path in paths], 'folds': sorted(frame.Fold.unique().tolist()),
        'fold_weighting': weighting, 'layer_weighting': 'equal arithmetic mean of expected layers', 'expected_layers': required_layers,
        'head': head, 'modes': list(modes), 'percentile': percentile,
        'order': 'subject mean within class/fold/layer; normalize; average folds; average layers',
        'scope': 'held-out test attention only; descriptive, not causal or significance inference',
        'minmax': 'within observed outgoing edges; constant rows become zero',
        'raw': 'unmodified attention; no additional dense graph min-max',
        'node_score': 'equal mean of incoming-edge mean and outgoing-edge mean, available directions only'}
    (output / 'normalization_metadata.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', type=Path, required=True, help='Analysis directories or summary CSVs, one per fold')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['raw', 'minmax', 'binorm', 'rownorm'], default=['raw', 'minmax', 'binorm', 'rownorm'])
    parser.add_argument('--fold-weighting', choices=['equal', 'subjects'], default='equal')
    parser.add_argument('--head', type=int, default=0, help='Legacy exports used head 0; no silent averaging across heads')
    parser.add_argument('--percentile', type=float, default=85)
    args = parser.parse_args()
    print(json.dumps(process(args.input, args.data_root, args.output, args.modes, args.fold_weighting, args.head, args.percentile), indent=2))


if __name__ == '__main__':
    main()
