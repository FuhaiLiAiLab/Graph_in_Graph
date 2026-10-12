"""Average node/edge weights by identity; not predictive-metric reporting."""
import numpy as np
import pandas as pd

def require(value, message):
    if not value:
        raise ValueError(message)

def align(frame, keys, expected_index=None):
    require(not frame[keys].isna().any().any(), 'Missing key values')
    require(not frame.duplicated(keys).any(), 'Duplicate identity keys')
    require(np.isfinite(frame.Weight.to_numpy(dtype=np.float64)).all(), 'Nonfinite weights')
    indexed = frame.set_index(keys).sort_index()
    if expected_index is not None:
        require(indexed.index.equals(expected_index), 'Identity keys differ')
    return indexed

def equal_fold_seed_average(frames, keys, seeds, folds):
    require(len(set(seeds))==len(seeds) and len(set(folds))==len(folds), 'Duplicate seed/fold plan')
    require(set(frames)=={(s,f) for s in seeds for f in folds}, 'Incomplete or extra seed/fold cases')
    reference = None
    seed_parts, seed_values, all_values = [], [], []
    for seed in seeds:
        fold_values = []
        for fold in folds:
            frame = align(frames[(seed,fold)], keys, reference)
            if reference is None:
                reference = frame.index
            values = frame.Weight.to_numpy(dtype=np.float64)
            fold_values.append(values)
            all_values.append(values)
        values = np.stack(fold_values).mean(axis=0)
        part = reference.to_frame(index=False)
        part['Seed'], part['Weight'], part['n_folds'] = seed, values, len(folds)
        seed_parts.append(part)
        seed_values.append(values)
    final = reference.to_frame(index=False)
    matrix = np.stack(seed_values)
    final['Weight'] = matrix.mean(axis=0)
    final['Weight_SD_across_seeds'] = matrix.std(axis=0, ddof=1)
    final['n_folds_per_seed'], final['n_seeds'] = len(folds), len(seeds)
    # All cells are present and equally weighted, so the direct 25-case mean must agree.
    np.testing.assert_allclose(final.Weight, np.stack(all_values).mean(axis=0), rtol=1e-12, atol=1e-12)
    return pd.concat(seed_parts,ignore_index=True), final
