import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, 'gsea_output_194')
os.makedirs(OUTDIR, exist_ok=True)

GENE_SETS = {
    'Hallmark': 'MSigDB_Hallmark_2020',
    'KEGG'    : 'KEGG_2021_Human',
    'GO_BP'   : 'GO_Biological_Process_2023',
}

# ============================================================
# Step 1: Build 194 paired couples (both have raw meth data)
# ============================================================
print("Building 194 paired couples...")

# Raw methylation subjects (936 total)
meth_ref = pd.read_csv(os.path.join(BASE_DIR, 'data', 'omics_data', 'epigenomics', 'Core_Promoter_final.csv'),
                        index_col=0, nrows=1)
meth_subjects = set(meth_ref.columns.astype(int))

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

paired_194 = one_lon[
    one_lon['lon_subj'].isin(meth_subjects) &
    one_lon['mar_subj'].isin(meth_subjects)
].copy().reset_index(drop=True)

# Age from LLFS_phenos_21JUN2022.xlsx
pheno_full = pd.read_excel(os.path.join(BASE_DIR, 'data', 'pheno_data', 'LLFS_phenos_21JUN2022.xlsx'),
                            sheet_name='Phenodata')
age_map = pheno_full.set_index('subject')['age_v1']
paired_194['lon_age'] = paired_194['lon_subj'].map(age_map)

print(f"  Total methylation pairs: {len(paired_194)}")
print(f"  <60 pairs:  {(paired_194['lon_age'] < 60).sum()}")
print(f"  >=60 pairs: {(paired_194['lon_age'] >= 60).sum()}")

young_194 = paired_194[paired_194['lon_age'] < 60].copy().reset_index(drop=True)
old_194   = paired_194[paired_194['lon_age'] >= 60].copy().reset_index(drop=True)

# ============================================================
# Step 2: GSEA helper — uses raw meth files (genes x subjects)
# ============================================================
REGION_FILES = {
    'core_promoter'    : 'Core_Promoter_final.csv',
    'distal_promoter'  : 'Distal_Promoter_final.csv',
    'proximal_promoter': 'Proximal_Promoter_final.csv',
    'downstream'       : 'Downstream_final.csv',
    'upstream'         : 'Upstream_final.csv',
}
METH_DIR = os.path.join(BASE_DIR, 'data', 'omics_data', 'epigenomics') + os.sep

def run_paired_gsea(meth_subj_df, pair_df, label, data_label):
    """
    meth_subj_df: rows=subjects (int index), cols=gene names
    pair_df:      DataFrame with lon_subj (int) and mar_subj (int)
    """
    print(f"\n{'='*60}")
    print(f"[{data_label}] {label}  (N = {len(pair_df)} pairs)")
    print('='*60)

    lon_mat = meth_subj_df.loc[pair_df['lon_subj'].values].values
    mar_mat = meth_subj_df.loc[pair_df['mar_subj'].values].values
    diff    = lon_mat - mar_mat

    t_stats, _ = stats.ttest_1samp(diff, 0, axis=0)
    ranked = pd.Series(t_stats, index=meth_subj_df.columns).dropna().sort_values(ascending=False)
    print(f"  Ranked genes: {len(ranked)}")

    safe_label = f"{data_label}_{label}"
    rnk_path = os.path.join(OUTDIR, f"{safe_label}.rnk")
    ranked.reset_index().to_csv(rnk_path, sep='\t', index=False, header=False)

    results = {}
    for gs_name, gs_key in GENE_SETS.items():
        out_sub = os.path.join(OUTDIR, safe_label, gs_name)
        os.makedirs(out_sub, exist_ok=True)
        print(f"  Running GSEA [{gs_name}]...")
        try:
            res = gp.prerank(
                rnk            = ranked,
                gene_sets      = gs_key,
                threads        = 4,
                min_size       = 15,
                max_size       = 500,
                permutation_num= 1000,
                outdir         = out_sub,
                seed           = 42,
                verbose        = False,
            )
            df = res.res2d.sort_values('NES', ascending=False)
            csv_path = os.path.join(OUTDIR, f"{safe_label}_{gs_name}.csv")
            df.to_csv(csv_path, index=False)
            results[gs_name] = df

            sig = df[df['FDR q-val'].astype(float) < 0.25]
            print(f"    Tested: {len(df)} | FDR<0.25: {len(sig)} | |NES|>1.3: {(df['NES'].astype(float).abs() > 1.3).sum()}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[gs_name] = None

    if results.get('Hallmark') is not None:
        df = results['Hallmark']
        df['NES'] = df['NES'].astype(float)
        df['FDR q-val'] = df['FDR q-val'].astype(float)
        pos = df[df['NES'] > 0][['Term', 'NES', 'NOM p-val', 'FDR q-val']].head(8)
        neg = df[df['NES'] < 0][['Term', 'NES', 'NOM p-val', 'FDR q-val']].tail(8)
        print(f"\n  Top Hallmark (longevity UP):\n{pos.to_string(index=False)}")
        print(f"\n  Top Hallmark (marryin UP):\n{neg.to_string(index=False)}")

    return results

# ============================================================
# Step 3: Methylation GSEA — 5 regions x 3 comparisons
# ============================================================
for region, fname in REGION_FILES.items():
    print(f"\n{'#'*60}")
    print(f"# REGION: {region}")
    print('#'*60)

    # Load raw meth: rows=gene_name, cols=subject_id (str)
    meth_raw = pd.read_csv(os.path.join(METH_DIR, fname), index_col=0)
    meth_raw.columns = meth_raw.columns.astype(int)
    # Transpose: rows=subjects, cols=gene_names
    meth_subj = meth_raw.T
    meth_subj.index = meth_subj.index.astype(int)

    # Keep only subjects in paired_194
    needed = set(paired_194['lon_subj']) | set(paired_194['mar_subj'])
    meth_subj = meth_subj.loc[meth_subj.index.isin(needed)]

    # Per-group pairs: verify both subjects present in this region's data
    avail = set(meth_subj.index)
    p_all   = paired_194[paired_194['lon_subj'].isin(avail) & paired_194['mar_subj'].isin(avail)].copy()
    p_young = young_194[young_194['lon_subj'].isin(avail) & young_194['mar_subj'].isin(avail)].copy()
    p_old   = old_194[old_194['lon_subj'].isin(avail) & old_194['mar_subj'].isin(avail)].copy()

    run_paired_gsea(meth_subj, p_all,   'primary_194pairs', region)
    run_paired_gsea(meth_subj, p_young, 'age_lt60',         region)
    run_paired_gsea(meth_subj, p_old,   'age_ge60',         region)

# ============================================================
# Step 4: Phenotype paired t-test (190 pairs with pheno data)
# ============================================================
print("\n" + "="*60)
print("PHENOTYPE: paired t-test")
print("="*60)

pheno_subjects = set(pheno_full['subject'].dropna().astype(int))
paired_pheno = paired_194[
    paired_194['lon_subj'].isin(pheno_subjects) &
    paired_194['mar_subj'].isin(pheno_subjects)
].copy().reset_index(drop=True)
print(f"Pairs with both in phenotype data: {len(paired_pheno)}")

demo_drop = {'id', 'subject', 'fc', 'gpedid', 'sex', 'age_v1', 'age_v2', 'longevity_class'}
feat_cols = [c for c in pheno_full.columns if c not in demo_drop]
pheno_feat = pheno_full.set_index('subject')[feat_cols]

def pheno_paired_ttest(pair_df, label):
    lon_vals = pheno_feat.reindex(pair_df['lon_subj'].values)
    mar_vals = pheno_feat.reindex(pair_df['mar_subj'].values)
    lon_vals.index = range(len(pair_df))
    mar_vals.index = range(len(pair_df))

    diff = lon_vals.values - mar_vals.values
    t_stats, p_vals = stats.ttest_1samp(diff, 0, axis=0)
    _, q_vals, _, _ = multipletests(p_vals, method='fdr_bh')
    mean_diff = np.nanmean(diff, axis=0)

    result = pd.DataFrame({
        'feature'  : lon_vals.columns,
        'mean_diff': mean_diff,
        't_stat'   : t_stats,
        'p_val'    : p_vals,
        'FDR_q'    : q_vals,
    }).sort_values('t_stat', ascending=False)

    csv_path = os.path.join(OUTDIR, f"phenotype_paired_{label}.csv")
    result.to_csv(csv_path, index=False)

    sig = result[result['FDR_q'] < 0.25]
    nom = result[result['p_val'] < 0.05]
    print(f"\n  [{label}] N pairs={len(pair_df)} | FDR<0.25: {len(sig)} | nominal p<0.05: {len(nom)}")
    print(f"\n  Top features HIGHER in longevity (t>0):")
    print(result[result['t_stat'] > 0][['feature', 'mean_diff', 't_stat', 'p_val', 'FDR_q']].head(10).to_string(index=False))
    print(f"\n  Top features HIGHER in marryin (t<0):")
    print(result[result['t_stat'] < 0][['feature', 'mean_diff', 't_stat', 'p_val', 'FDR_q']].tail(10).to_string(index=False))
    return result

young_pheno = paired_pheno[paired_pheno['lon_age'] < 60].reset_index(drop=True)
old_pheno   = paired_pheno[paired_pheno['lon_age'] >= 60].reset_index(drop=True)

pheno_paired_ttest(paired_pheno, 'primary_190pairs')
pheno_paired_ttest(young_pheno,  'age_lt60')
pheno_paired_ttest(old_pheno,    'age_ge60')

print("\nAll analyses complete. Results in:", OUTDIR)
