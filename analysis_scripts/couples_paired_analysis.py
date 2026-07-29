import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'pvalues_output', 'couples_paired')
os.makedirs(OUT_DIR, exist_ok=True)

# ── STEP 1: Update longevity labels (add gen=1) ──────────────────────────────
print('=== Step 1: Update longevity labels ===')
triplet = pd.read_csv(os.path.join(BASE_DIR, 'triplet_visit3.csv'))

def classify(row):
    g, r, c = row['gen'], row['relative'], row['control']
    if   g == 1 and r == 1 and c == 0: return 'longevity'   # gen1 blood relatives
    elif g == 1 and r == 0 and c == 0: return 'marryin'
    elif g == 2 and r == 1 and c == 0: return 'longevity'
    elif g == 2 and r == 0 and c == 0: return 'marryin'
    elif g == 3 and r == 1 and c == 0: return 'longevity'
    elif g == 3 and r == 0 and c == 1: return 'marryin'
    elif g == 4 and r == 1 and c == 1: return 'longevity'
    elif g == 4 and r == 0 and c == 0: return 'marryin'
    else:                               return 'unclassified'

triplet['longevity_class_v2'] = triplet.apply(classify, axis=1)
print(triplet['longevity_class_v2'].value_counts().to_string())

subj_label = triplet[['subject','longevity_class_v2']].drop_duplicates(subset='subject')
subj_label.columns = ['subject','longevity_class']
subj_label.to_csv(os.path.join(BASE_DIR, 'subject_longevity_labels_v2.csv'), index=False)
print(f'Saved subject_longevity_labels_v2.csv ({len(subj_label)} subjects)')

# ── STEP 2: Rebuild couples ───────────────────────────────────────────────────
print('\n=== Step 2: Rebuild couples with updated labels ===')
subj_info = triplet[['subject','sex','longevity_class_v2']].drop_duplicates(subset='subject').set_index('subject')

children = triplet[(triplet['dadsubj'] != 0) & (triplet['momsubj'] != 0)][
    ['gpedid','dadsubj','momsubj']].drop_duplicates().copy()

children['husband_subject']   = children['dadsubj']
children['wife_subject']      = children['momsubj']
children['husband_longevity'] = children['dadsubj'].map(subj_info['longevity_class_v2'])
children['wife_longevity']    = children['momsubj'].map(subj_info['longevity_class_v2'])

def couple_type(h, w):
    if h == 'longevity' and w == 'longevity': return 'both_longevity'
    if (h == 'longevity' and w == 'marryin') or (h == 'marryin' and w == 'longevity'): return 'one_longevity'
    if h == 'marryin'   and w == 'marryin':   return 'both_marryin'
    return 'other'

children['couple_type'] = [couple_type(h, w) for h, w in
                            zip(children['husband_longevity'], children['wife_longevity'])]

couples = children[['gpedid','husband_subject','wife_subject',
                     'husband_longevity','wife_longevity','couple_type']].reset_index(drop=True)
couples.to_csv(os.path.join(BASE_DIR, 'couples_longevity_v2.csv'), index=False)
print(couples['couple_type'].value_counts().to_string())

# Working set: one_longevity couples only
one_lon = couples[couples['couple_type'] == 'one_longevity'].copy()
# Standardize so longevity is always col A, marryin col B
one_lon['lon_subj']    = np.where(one_lon['husband_longevity']=='longevity',
                                   one_lon['husband_subject'], one_lon['wife_subject'])
one_lon['marryin_subj'] = np.where(one_lon['husband_longevity']=='marryin',
                                    one_lon['husband_subject'], one_lon['wife_subject'])
one_lon = one_lon[['gpedid','lon_subj','marryin_subj']].reset_index(drop=True)
print(f'one_longevity pairs for analysis: {len(one_lon)}')


# ── Helper: paired Wilcoxon across features ───────────────────────────────────
def paired_wilcoxon(df_lon, df_mar, feature_names, label):
    """
    df_lon, df_mar: DataFrames with same index (pair_id) and columns = features
    Returns result DataFrame sorted by q-value.
    """
    results = []
    n_feat = len(feature_names)
    for i, feat in enumerate(feature_names):
        if (i+1) % 2000 == 0:
            print(f'  {label}: {i+1}/{n_feat}')
        a = df_lon[feat].values
        b = df_mar[feat].values
        # Drop pairs where either is NaN
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        n = mask.sum()
        if n < 10:
            results.append((feat, np.nan, np.nan, np.nan, n))
            continue
        d = a - b
        if np.all(d == 0):
            results.append((feat, 0.0, 0.0, 1.0, n))
            continue
        try:
            stat, p = wilcoxon(d, zero_method='zsplit')
        except Exception:
            p = np.nan
        results.append((feat, float(np.mean(d)), float(np.median(d)), float(p), int(n)))

    res = pd.DataFrame(results, columns=['feature','mean_diff','median_diff','pvalue','n_pairs'])
    valid = res['pvalue'].notna()
    res.loc[valid, 'qvalue'] = multipletests(res.loc[valid,'pvalue'], method='fdr_bh')[1]
    res.loc[~valid, 'qvalue'] = np.nan
    res = res.sort_values('qvalue').reset_index(drop=True)
    return res


# ── STEP 3: Phenotype analysis ────────────────────────────────────────────────
print('\n=== Step 3: Phenotype analysis ===')
pheno = pd.read_excel(os.path.join(BASE_DIR, 'data', 'pheno_data', 'LLFS_phenos_21JUN2022.xlsx'),
                      sheet_name='Phenodata')
SKIP_COLS = {'id','subject','gpedid','fc','sex','longevity_class','longevity_class_v2'}
pheno_features = [c for c in pheno.select_dtypes(include='number').columns if c not in SKIP_COLS]
print(f'Phenotype features: {len(pheno_features)}')

pheno_sub = pheno.set_index('subject')
lon_in_pheno = one_lon['lon_subj'].isin(pheno_sub.index)
mar_in_pheno = one_lon['marryin_subj'].isin(pheno_sub.index)
pheno_pairs = one_lon[lon_in_pheno & mar_in_pheno].reset_index(drop=True)
print(f'Pairs with both in phenotype data: {len(pheno_pairs)}')

df_lon_ph = pheno_sub.loc[pheno_pairs['lon_subj'], pheno_features].values
df_mar_ph = pheno_sub.loc[pheno_pairs['marryin_subj'], pheno_features].values
df_lon_ph = pd.DataFrame(df_lon_ph, columns=pheno_features)
df_mar_ph = pd.DataFrame(df_mar_ph, columns=pheno_features)

res_pheno = paired_wilcoxon(df_lon_ph, df_mar_ph, pheno_features, 'Phenotype')
res_pheno.to_csv(f'{OUT_DIR}/phenotype_paired.csv', index=False)
sig = res_pheno[res_pheno['qvalue'] < 0.05]
print(f'Saved phenotype_paired.csv | q<0.05: {len(sig)}/{len(res_pheno)} features')


# ── STEP 4: RNA-seq analysis ──────────────────────────────────────────────────
print('\n=== Step 4: RNA-seq analysis ===')
rna = pd.read_csv(os.path.join(BASE_DIR, 'data', 'omics_data', 'residuals', 'RNA_seq_residuals_v1_allsubjects.csv'))
rna = rna.set_index('subject')
NON_GENE = {'longevity_class','longevity_class_v2'}
rna_genes = [c for c in rna.columns if c not in NON_GENE]
print(f'RNA-seq genes: {len(rna_genes)}')

lon_in_rna = one_lon['lon_subj'].isin(rna.index)
mar_in_rna = one_lon['marryin_subj'].isin(rna.index)
rna_pairs = one_lon[lon_in_rna & mar_in_rna].reset_index(drop=True)
print(f'Pairs with both in RNA-seq: {len(rna_pairs)}')

df_lon_rna = rna.loc[rna_pairs['lon_subj'], rna_genes].reset_index(drop=True)
df_mar_rna = rna.loc[rna_pairs['marryin_subj'], rna_genes].reset_index(drop=True)

res_rna = paired_wilcoxon(df_lon_rna, df_mar_rna, rna_genes, 'RNA-seq')
res_rna.to_csv(f'{OUT_DIR}/rnaseq_paired.csv', index=False)
sig = res_rna[res_rna['qvalue'] < 0.05]
print(f'Saved rnaseq_paired.csv | q<0.05: {len(sig)}/{len(res_rna)} genes')


# ── STEP 5: Methylation (5 regions) ──────────────────────────────────────────
print('\n=== Step 5: Methylation analysis (5 regions) ===')
METH_DIR = os.path.join(BASE_DIR, 'data', 'omics_data', 'epigenomics') + os.sep
REGIONS = ['Downstream','Proximal_Promoter','Core_Promoter','Distal_Promoter','Upstream']

for region in REGIONS:
    print(f'\n-- {region} --')
    meth = pd.read_csv(f'{METH_DIR}{region}_final.csv')
    # Rows = genes/CpG, cols = gene_name + subject_ids (as strings)
    meth = meth.set_index('gene_name')
    meth.columns = meth.columns.astype(int)  # subject IDs as int

    lon_in_m = one_lon['lon_subj'].isin(meth.columns)
    mar_in_m = one_lon['marryin_subj'].isin(meth.columns)
    m_pairs = one_lon[lon_in_m & mar_in_m].reset_index(drop=True)
    print(f'Pairs with both in {region}: {len(m_pairs)}')

    if len(m_pairs) < 10:
        print(f'Too few pairs, skipping.')
        continue

    # Transpose to subjects x features for consistent interface
    lon_vals = meth[m_pairs['lon_subj'].values].T.reset_index(drop=True)
    mar_vals = meth[m_pairs['marryin_subj'].values].T.reset_index(drop=True)
    features = meth.index.tolist()

    res_m = paired_wilcoxon(lon_vals, mar_vals, features, region)
    fname = f'{OUT_DIR}/methylation_{region.lower()}_paired.csv'
    res_m.to_csv(fname, index=False)
    sig = res_m[res_m['qvalue'] < 0.05]
    print(f'Saved {os.path.basename(fname)} | q<0.05: {len(sig)}/{len(res_m)} features')

print('\n=== All analyses complete ===')
