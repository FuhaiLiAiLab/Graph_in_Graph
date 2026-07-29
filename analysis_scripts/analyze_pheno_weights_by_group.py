"""
Analyze GIG Transformer phenotype feature (subfeature node) attention weights
by patient group (t2ds / pret2ds / no_t2ds).

Source: analysis/gigtransformer/fold_N/pheno_{first|block|last}_patient_edge_weight.csv
  - from_node_idx >= 122 : subject -> subfeature edges
  - to_node_idx < 122    : subfeature node (phenotype bin)
  - Weight               : attention weight from PatientTransformerConv

Output: Top-20 subfeature weights per group, barplot, heatmap, CSV
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDS      = [1, 2, 3, 4, 5]
GROUPS     = ['t2ds', 'pret2ds', 'no_t2ds']
LAYERS     = ['last']    # last layer: closest to classification output
TOP_K      = 20
OUTPUT_DIR = os.path.join(BASE_DIR, 'enrichment_output', 'pheno_weights_by_group')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHENO_FILE_TPL = os.path.join(
    BASE_DIR, 'analysis/gigtransformer/fold_{fold}/pheno_{layer}_patient_edge_weight.csv')

LABEL_FILE   = os.path.join(BASE_DIR, 'data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
SUBFEAT_FILE = os.path.join(BASE_DIR, 'data/filtered_data/subfeature_dict_df.csv')


# ── helpers ───────────────────────────────────────────────────────────────────

def group_name(subfeature_name: str) -> str:
    """'age_v1-3' -> 'age_v1'  |  'sex-1' -> 'sex'"""
    return '-'.join(subfeature_name.split('-')[:-1])


def load_fold_layer(fold: int, layer: str,
                    group_idx: dict,          # group -> set of node_idx
                    n_subfeatures: int) -> dict:
    """
    Returns {group: np.array([n_subfeatures])} mean attention weight
    for subject->subfeature edges in each patient group.
    """
    path = PHENO_FILE_TPL.format(fold=fold, layer=layer)
    df = pd.read_csv(path)

    # Keep only subject -> subfeature edges
    df = df[df['from_node_idx'] >= 122].copy()

    result = {}
    for g, idx_set in group_idx.items():
        grp_df = df[df['from_node_idx'].isin(idx_set)]
        # mean weight per subfeature node (to_node_idx)
        agg = grp_df.groupby('to_node_idx')['Weight'].mean()
        weights = np.zeros(n_subfeatures, dtype=np.float64)
        for sf_idx, w in agg.items():
            weights[sf_idx] = w
        result[g] = weights
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading metadata ...")
    label_df   = pd.read_csv(LABEL_FILE)
    subfeat_df = pd.read_csv(SUBFEAT_FILE)   # subfeature_node_idx, subfeature_names
    n_sf = len(subfeat_df)
    print(f"  Subfeature nodes: {n_sf}")

    # Build group -> set of subject node_idx
    group_idx = {}
    for g in GROUPS:
        group_idx[g] = set(label_df.loc[label_df[g] == 1, 'node_idx'].tolist())
    for g, s in group_idx.items():
        print(f"  {g}: {len(s)} subjects")

    # ── Accumulate weights across folds and layers ────────────────────────────
    # final_acc[group] = list of (n_sf,) arrays, one per fold-layer combination
    final_acc = {g: [] for g in GROUPS}

    for fold in FOLDS:
        for layer in LAYERS:
            print(f"  fold={fold} layer={layer}")
            fold_layer = load_fold_layer(fold, layer, group_idx, n_sf)
            for g in GROUPS:
                final_acc[g].append(fold_layer[g])

    # Average across all fold-layer combinations
    final_scores = {g: np.mean(final_acc[g], axis=0) for g in GROUPS}

    # ── Build top-20 tables ───────────────────────────────────────────────────
    sf_names = subfeat_df['subfeature_names'].tolist()   # index == subfeature_node_idx
    results  = {}

    for g in GROUPS:
        top_idx    = np.argsort(final_scores[g])[-TOP_K:][::-1]
        top_sf     = [sf_names[i] for i in top_idx]
        top_group  = [group_name(n) for n in top_sf]
        top_scores = final_scores[g][top_idx].tolist()

        results[g] = pd.DataFrame({
            'rank':         list(range(1, TOP_K + 1)),
            'Group Name':   top_group,
            'Subgroup Name':top_sf,
            'Weight':       [round(w, 7) for w in top_scores],
        })
        print(f"\n-- {g} Top-{TOP_K} --")
        print(results[g].to_string(index=False))

        csv_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_pheno_{g}.csv')
        results[g].to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}")

    # ── Combined table ────────────────────────────────────────────────────────
    combined = pd.concat(
        [df.assign(group=g) for g, df in results.items()], ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_pheno_all_groups.csv')
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined table saved -> {combined_path}")

    # ── Group-unique subfeature analysis ─────────────────────────────────────
    sf_sets = {g: set(results[g]['Subgroup Name']) for g in GROUPS}
    print("\n=== Top-20 subfeature overlap ===")
    for g in GROUPS:
        others = set().union(*[sf_sets[g2] for g2 in GROUPS if g2 != g])
        print(f"  {g} unique: {sorted(sf_sets[g] - others)}")

    # ── Barplot (1 row × 3 groups) ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    colors = {'t2ds': '#d62728', 'pret2ds': '#ff7f0e', 'no_t2ds': '#1f77b4'}

    for ax, g in zip(axes, GROUPS):
        df = results[g]
        labels = df['Subgroup Name'][::-1]
        vals   = df['Weight'][::-1]
        ax.barh(labels, vals, color=colors[g], alpha=0.85)
        ax.set_title(f'{g}\n(Top {TOP_K} phenotype features)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean Attention Weight', fontsize=10)
        ax.tick_params(axis='y', labelsize=7)

    plt.suptitle(
        'GIG Model — Top 20 Phenotype Feature Weights by Group\n'
        '(subject→subfeature attention, averaged across layers & folds)',
        fontsize=11, y=1.01)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_pheno_barplot.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"\nBarplot saved -> {bar_path}")

    # ── Heatmap: groups × union of top-20 subfeatures ────────────────────────
    union_sf = sorted(set().union(*[sf_sets[g] for g in GROUPS]))
    sf_to_idx = {n: i for i, n in enumerate(sf_names)}

    hm_data = pd.DataFrame(
        {g: [final_scores[g][sf_to_idx[sf]] for sf in union_sf] for g in GROUPS},
        index=union_sf
    )
    fig2, ax2 = plt.subplots(figsize=(6, max(6, len(union_sf) * 0.32)))
    im = ax2.imshow(hm_data.values.astype(float), aspect='auto', cmap='YlOrRd')
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(GROUPS, fontsize=11)
    ax2.set_yticks(range(len(union_sf)))
    ax2.set_yticklabels(union_sf, fontsize=7)
    plt.colorbar(im, ax=ax2, label='Mean Attention Weight')
    ax2.set_title('Phenotype Feature Heatmap\n(Top-20 union across groups)', fontsize=10)
    plt.tight_layout()
    hm_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_pheno_heatmap.png')
    plt.savefig(hm_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved -> {hm_path}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
