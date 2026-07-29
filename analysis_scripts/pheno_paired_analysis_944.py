import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, 'gsea_output', 'phenotype_944')
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Step 1: Build 944 paired couples
# ============================================================
print("Building 944 paired couples...")

couples = pd.read_csv(os.path.join(BASE_DIR, 'couples_longevity_v2.csv'))
one_lon = couples[couples['couple_type'] == 'one_longevity'].copy()

def get_lon_mar(row):
    if row['husband_longevity'] == 'longevity' and row['wife_longevity'] == 'marryin':
        return row['husband_subject'], row['wife_subject']
    elif row['husband_longevity'] == 'marryin' and row['wife_longevity'] == 'longevity':
        return row['wife_subject'], row['husband_subject']
    return None, None

one_lon[['lon_subj', 'mar_subj']] = one_lon.apply(lambda r: pd.Series(get_lon_mar(r)), axis=1)
one_lon = one_lon.dropna(subset=['lon_subj', 'mar_subj'])
one_lon['lon_subj'] = one_lon['lon_subj'].astype(int)
one_lon['mar_subj'] = one_lon['mar_subj'].astype(int)

pheno_full = pd.read_excel(os.path.join(BASE_DIR, 'data', 'pheno_data', 'LLFS_phenos_21JUN2022.xlsx'), sheet_name='Phenodata')
pheno_subjects = set(pheno_full['subject'].dropna().astype(int))

paired_944 = one_lon[
    one_lon['lon_subj'].isin(pheno_subjects) &
    one_lon['mar_subj'].isin(pheno_subjects)
].copy().reset_index(drop=True)

age_map = pheno_full.set_index('subject')['age_v1']
paired_944['lon_age'] = paired_944['lon_subj'].map(age_map)
paired_944['age_group'] = paired_944['lon_age'].apply(
    lambda x: '<60' if x < 60 else ('>=60' if x >= 60 else 'unknown')
)

print(f"  Total pairs: {len(paired_944)}")
print(f"  <60 pairs:   {(paired_944['age_group'] == '<60').sum()}")
print(f"  >=60 pairs:  {(paired_944['age_group'] == '>=60').sum()}")

# ============================================================
# Step 2: Biological theme definitions (same as original)
# ============================================================
THEMES = {
    'Cardiovascular':    ['_sbp2z_v1','_dbp2z_v1','_Pulsez_v1','TSadj_BP_bcz_v1','pp2z_v1','ntbnpe_logz_v1','q_plaque_v2','nlogplaque_sevq_v2','abi_bcz_invn_v1'],
    'Lipids':            ['cholz_v1','hdlz_v1','ldlz_v1','tg_logz_v1'],
    'Metabolic':         ['glucz_v1','hba1cz_v1','A1Cz_v1','_ins_logz_v1','_BMI_logz_v1','_Waistz_v1','weightz_v1'],
    'Pulmonary':         ['fev1z_v1','fev1z_lk_v1','fev6z_v1','fvcz_invn_v1','fev1fvc_bcz_v1','tmpPPFEV1z_v1','tmpPPFEV6z_v1'],
    'Cognitive':         ['animaltotz_v1','digitbwdtotz_v1','digitfwdtotz_v1','digitsymtotz_v1','logmemdlydtotz_v1','logmemimtotz_v1','mmsetot_bcz_v1','_totscorez_invn_v1'],
    'Physical_Function': ['gaitspeedz_v1','gripz_invn_v1','stand5time_invnz_v1'],
    'Inflammatory':      ['hscrp_logz_v1','il6_logz_invn_v1','srage_logz_invn_v1'],
    'Renal':             ['creatr_bcz_invn_v1','cysc_logz_invn_v1'],
    'Biomarkers':        ['albz_v1','d2_logz_v1','d3z_v1','dhea_logz_v1','igf1_invnz_v1','transferrin_invnz_v1','teststrnz_invn_v1'],
}

HEALTHY_DIRECTION = {
    'Cardiovascular':    -1,
    'Lipids':            +1,
    'Metabolic':         -1,
    'Pulmonary':         +1,
    'Cognitive':         +1,
    'Physical_Function': +1,
    'Inflammatory':      -1,
    'Renal':             +1,
    'Biomarkers':        +1,
}

# ============================================================
# Step 3: Compute within-couple differences
# ============================================================
demo_drop = {'id', 'subject', 'fc', 'gpedid', 'sex', 'age_v1', 'age_v2', 'longevity_class'}
feat_cols = [c for c in pheno_full.columns if c not in demo_drop]
pheno_feat = pheno_full.set_index('subject')[feat_cols]

def get_diff_matrix(pair_df):
    lon_vals = pheno_feat.reindex(pair_df['lon_subj'].values)
    mar_vals = pheno_feat.reindex(pair_df['mar_subj'].values)
    lon_vals.index = range(len(pair_df))
    mar_vals.index = range(len(pair_df))
    diff = lon_vals - mar_vals
    n_complete = diff.notna().all(axis=1).sum()
    print(f"  Couples with complete phenotype for both: {n_complete}/{len(pair_df)}")
    return diff

print("\nComputing within-couple differences...")
diff_all   = get_diff_matrix(paired_944)
diff_young = get_diff_matrix(paired_944[paired_944['age_group'] == '<60'].reset_index(drop=True))
diff_old   = get_diff_matrix(paired_944[paired_944['age_group'] == '>=60'].reset_index(drop=True))

# ============================================================
# Step 4: Direction Consistency Test
# ============================================================
def direction_consistency(diff_df, label):
    print(f"\n{'='*60}")
    print(f"Direction Consistency Test: {label} (N={len(diff_df)} couples)")
    print('='*60)

    rows = []
    for theme, feats in THEMES.items():
        available = [f for f in feats if f in diff_df.columns]
        if not available:
            continue
        theme_diff = diff_df[available].mean(axis=0)
        n_positive = (theme_diff > 0).sum()
        n_total    = len(theme_diff)
        prop_pos   = n_positive / n_total
        binom      = binomtest(int(n_positive), n_total, p=0.5)
        healthy_dir = HEALTHY_DIRECTION[theme]
        consistent  = (prop_pos > 0.5) == (healthy_dir > 0)

        rows.append({
            'Theme'       : theme,
            'N_features'  : n_total,
            'N_lon>mar'   : int(n_positive),
            'Prop_lon>mar': round(prop_pos, 3),
            'Binomial_p'  : round(binom.pvalue, 4),
            'Healthy_dir' : '+' if healthy_dir > 0 else '-',
            'Consistent'  : 'YES' if consistent else 'NO',
            'Mean_diff'   : round(theme_diff.mean(), 4),
        })
        print(f"  {theme:20s}: {n_positive}/{n_total} features lon>mar "
              f"({prop_pos:.0%})  binomial p={binom.pvalue:.4f}  "
              f"{'[consistent with longevity healthier]' if consistent else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, f'direction_consistency_{label}.csv'), index=False)
    return df

dc_all   = direction_consistency(diff_all,   'primary_944pairs')
dc_young = direction_consistency(diff_young, 'age_lt60')
dc_old   = direction_consistency(diff_old,   'age_ge60')

# ============================================================
# Step 5: Direction Consistency Plot (lollipop chart)
# ============================================================
def plot_direction_consistency(dc_df, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    themes   = dc_df['Theme']
    props    = dc_df['Prop_lon>mar']
    binom_ps = dc_df['Binomial_p']
    healthy  = dc_df['Healthy_dir']

    colors = []
    for p, h in zip(props, healthy):
        if h == '+':
            colors.append('#2166ac' if p > 0.5 else '#d73027')
        else:
            colors.append('#2166ac' if p < 0.5 else '#d73027')

    y_pos = range(len(themes))
    ax.axvline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    for i, (y, p, c, bp) in enumerate(zip(y_pos, props, colors, binom_ps)):
        ax.plot([0.5, p], [y, y], color=c, lw=2, alpha=0.7)
        ax.scatter(p, y, color=c, s=80, zorder=5)
        star = '***' if bp < 0.001 else ('**' if bp < 0.01 else ('*' if bp < 0.05 else ('.' if bp < 0.10 else '')))
        if star:
            ax.text(p + 0.01, y, star, va='center', fontsize=10, color=c)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(themes, fontsize=11)
    ax.set_xlabel('Proportion of features where Longevity > Marryin', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_title(f'Direction Consistency by Phenotype Theme\n({label})', fontsize=12)

    patch_blue = mpatches.Patch(color='#2166ac', label='Longevity healthier (consistent)')
    patch_red  = mpatches.Patch(color='#d73027', label='Marryin healthier (inconsistent)')
    ax.legend(handles=[patch_blue, patch_red], loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTDIR, f'direction_consistency_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

plot_direction_consistency(dc_all,   'primary_944pairs')
plot_direction_consistency(dc_young, 'age_lt60')
plot_direction_consistency(dc_old,   'age_ge60')

# ============================================================
# Step 6: PCA on within-couple differences
# ============================================================
def plot_pca_diff(diff_df, pair_df, label):
    valid_feats = diff_df.columns[diff_df.isna().mean() < 0.2]
    X = diff_df[valid_feats].copy().fillna(diff_df[valid_feats].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    pcs = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    age_col   = pair_df['age_group'].values
    color_map = {'<60': '#e41a1c', '>=60': '#377eb8', 'unknown': '#999999'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for grp, col in color_map.items():
        mask = age_col == grp
        if mask.sum() == 0:
            continue
        ax.scatter(pcs[mask, 0], pcs[mask, 1], c=col, alpha=0.6,
                   s=30, label=f'Longevity age {grp} (N={mask.sum()})', edgecolors='none')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} var)', fontsize=11)
    ax.set_title(f'PCA of Within-Couple Phenotype Differences\n({label})', fontsize=11)
    ax.legend(fontsize=9)

    for grp, col in color_map.items():
        mask = age_col == grp
        if mask.sum() < 5:
            continue
        pts  = pcs[mask, :2]
        cov  = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h  = 2 * 1.5 * np.sqrt(vals)
        ell   = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=angle,
                        edgecolor=col, fc='None', lw=1.5, alpha=0.6)
        ax.add_patch(ell)

    ax2 = axes[1]
    ax2.bar(range(1, 11), var_exp[:10] * 100, color='#4393c3', alpha=0.8)
    ax2.plot(range(1, 11), np.cumsum(var_exp[:10]) * 100,
             'o-', color='#d6604d', lw=1.5, markersize=5, label='Cumulative')
    ax2.set_xlabel('Principal Component', fontsize=11)
    ax2.set_ylabel('Variance Explained (%)', fontsize=11)
    ax2.set_title('Scree Plot', fontsize=11)
    ax2.set_xticks(range(1, 11))
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(OUTDIR, f'pca_diff_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    loadings = pd.DataFrame(pca.components_[:2].T, index=valid_feats, columns=['PC1', 'PC2'])
    print(f"\n  Top PC1 loadings:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(10).to_string())
    print(f"\n  Top PC2 loadings:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(10).to_string())
    loadings.to_csv(os.path.join(OUTDIR, f'pca_loadings_{label}.csv'))
    return pca, pcs, var_exp

print("\nRunning PCA on within-couple differences...")
pca_all, pcs_all, var_all = plot_pca_diff(diff_all, paired_944, 'primary_944pairs')

# PCA colored by continuous age
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(pcs_all[:, 0], pcs_all[:, 1],
                c=paired_944['lon_age'], cmap='RdYlBu_r',
                alpha=0.8, s=35, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Longevity member age (Visit 1)')
ax.axhline(0, color='grey', lw=0.5, ls='--')
ax.axvline(0, color='grey', lw=0.5, ls='--')
ax.set_xlabel(f'PC1 ({var_all[0]:.1%} var)', fontsize=11)
ax.set_ylabel(f'PC2 ({var_all[1]:.1%} var)', fontsize=11)
ax.set_title('PCA of Phenotype Differences (lon-mar)\nColored by longevity member age', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'pca_diff_age_gradient.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pca_diff_age_gradient.png")

print("\nAll analyses complete. Results in:", OUTDIR)
