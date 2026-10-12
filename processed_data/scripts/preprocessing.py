"""Deterministic cohort and train-fitted transforms; no filesystem side effects."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def earliest_visit(frame):
    data = frame.copy()
    for column in ('subject', 'visit'):
        data[column] = pd.to_numeric(data[column], errors='raise')
        if data[column].isna().any():
            raise ValueError(f'missing {column}')
    if data.duplicated(['subject', 'visit']).any():
        raise ValueError('duplicate subject/visit records')
    return data.sort_values(['subject', 'visit'], kind='stable').drop_duplicates('subject').reset_index(drop=True)


def diabetes_labels(frame):
    t2d = pd.to_numeric(frame['t2ds'], errors='coerce').to_numpy()
    pre = pd.to_numeric(frame['pret2ds'], errors='coerce').to_numpy()
    if np.any((t2d == 1) & (pre == 1)):
        raise ValueError('conflicting diabetes labels')
    result = np.full(len(frame), -1, dtype=np.int64)
    result[(t2d == 0) & (pre == 0)] = 2
    result[(t2d == 0) & (pre == 1)] = 1
    result[t2d == 1] = 0
    return result


def reconcile_families(pedigree, phenotype):
    """Union observed family membership across sources without conflating ID namespaces."""
    if pedigree.subject.duplicated().any() or phenotype.subject.duplicated().any():
        raise ValueError('duplicate subjects in family sources')
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    merged = pedigree[['subject', 'gpedid']].merge(
        phenotype[['subject', 'gpedid']], on='subject', how='left',
        suffixes=('_pedigree', '_phenotype'), validate='one_to_one')
    if merged.gpedid_pedigree.isna().any():
        raise ValueError('missing pedigree family')
    for row in merged.itertuples():
        a = ('pedigree', int(row.gpedid_pedigree))
        find(a)
        if pd.notna(row.gpedid_phenotype) and row.gpedid_phenotype > 0:
            union(a, ('phenotype', int(row.gpedid_phenotype)))
    # Additional documented parent links remain in the same splitting component.
    by_subject = pedigree.set_index('subject').gpedid.to_dict()
    for column in ('dadsubj', 'momsubj'):
        if column in pedigree:
            for subject, parent in zip(pedigree.subject, pedigree[column]):
                if pd.notna(parent) and parent != 0 and parent in by_subject:
                    union(('pedigree', int(by_subject[subject])), ('pedigree', int(by_subject[parent])))
    roots = {}
    for g in merged.gpedid_pedigree.unique():
        roots.setdefault(find(('pedigree', int(g))), []).append(int(g))
    identifiers = {root: f'family_{min(values)}' for root, values in roots.items()}
    merged['family_id'] = [identifiers[find(('pedigree', int(g)))] for g in merged.gpedid_pedigree]
    return merged


def fit_numeric(train):
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError('expected finite-or-missing 2D training features')
    if np.any(np.all(np.isnan(values), axis=0)):
        raise ValueError('all-missing training feature')
    mean = np.nanmean(values, axis=0)
    filled = np.where(np.isnan(values), mean, values)
    minimum, maximum = filled.min(axis=0), filled.max(axis=0)
    return {'mean': mean, 'min': minimum, 'max': maximum,
            'constant': maximum == minimum,
            'q10': np.percentile(filled, 10, axis=0, method='linear'),
            'q90': np.percentile(filled, 90, axis=0, method='linear')}


def impute_numeric(values, fitted):
    data = np.asarray(values, dtype=np.float64)
    if np.isinf(data).any():
        raise ValueError('infinite transform input')
    return np.where(np.isnan(data), fitted['mean'], data)


def transform_numeric(values, fitted):
    filled = impute_numeric(values, fitted)
    width = fitted['max'] - fitted['min']
    transformed = (filled - fitted['min']) / np.where(width == 0, 1., width)
    transformed[:, fitted['constant']] = 0.
    return transformed


def bin_numeric(values, fitted):
    filled = impute_numeric(values, fitted)
    bins = np.where(filled <= fitted['q10'], 1, np.where(filled <= fitted['q90'], 2, 3))
    # For tied thresholds, equality is the middle category; empty categories are retained.
    tied = fitted['q10'] == fitted['q90']
    bins[:, tied] = np.where(filled[:, tied] < fitted['q10'][tied], 1,
                            np.where(filled[:, tied] > fitted['q90'][tied], 3, 2))
    return bins.astype(np.int8)


def assign_folds(labels, groups, base_seed=20260914, candidates=100):
    labels, groups = np.asarray(labels), np.asarray(groups)
    best = None
    classes, total = np.unique(labels, return_counts=True)
    for seed in range(base_seed, base_seed + candidates):
        fold = np.zeros(len(labels), dtype=np.int8)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for k, (_, test) in enumerate(cv.split(np.zeros(len(labels)), labels, groups), 1):
            fold[test] = k
        counts = np.array([[np.sum((fold == k) & (labels == c)) for c in classes] for k in range(1, 6)])
        if np.any(counts == 0):
            continue
        target = total / 5
        score = float(np.mean(((counts - target) / target) ** 2)
                      + np.mean(((counts.sum(axis=1) - len(labels) / 5) / (len(labels) / 5)) ** 2))
        if best is None or score < best[2]:
            best = (fold, seed, score)
    if best is None:
        raise ValueError('no valid five-fold split containing every class in every fold')
    return best


def phenotype_graph(features, bins, sex):
    """Legacy directed category chains and subject-to-category links, local subjects only."""
    features, bins = np.asarray(features), np.asarray(bins)
    if not np.isin(sex, [1, 2]).all():
        raise ValueError('unknown sex code')
    n, p = bins.shape
    subnodes = 2 + 3 * p
    graph_x = np.vstack([np.zeros((subnodes, features.shape[1])), features]).astype(np.float32)
    idx = np.arange(subnodes, subnodes + n, dtype=np.int64)
    chains = np.array([(2 + 3*j + k, 2 + 3*j + k + 1) for j in range(p) for k in (0, 1)], dtype=np.int64).T
    subject_src = np.repeat(idx, p)
    subject_dst = (2 + 3 * np.arange(p)[None, :] + bins - 1).ravel()
    edges = np.concatenate([chains, np.vstack([subject_src, subject_dst])], axis=1)
    return graph_x, edges, idx
