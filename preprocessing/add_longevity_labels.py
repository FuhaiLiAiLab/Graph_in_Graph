import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

def classify(row):
    g, r, c = row['gen'], row['relative'], row['control']
    if g == 2 and r == 1 and c == 0: return 'longevity'
    if g == 2 and r == 0 and c == 0: return 'marryin'
    if g == 3 and r == 1 and c == 0: return 'longevity'
    if g == 3 and r == 0 and c == 1: return 'marryin'
    if g == 4 and r == 1 and c == 1: return 'longevity'
    if g == 4 and r == 0 and c == 0: return 'marryin'
    return 'unclassified'

# ── Step 1: Master label file ──────────────────────────────────────────────────
print("=== Step 1: Generate master label file ===")
triplet = pd.read_csv('triplet_visit3.csv')
triplet['longevity_class'] = triplet.apply(classify, axis=1)
labels = triplet[['subject', 'longevity_class']].copy()
labels.to_csv('subject_longevity_labels.csv', index=False)
print(f"Saved subject_longevity_labels.csv  ({len(labels)} rows)")
print(labels['longevity_class'].value_counts().to_string())
print()

# ── Step 2a: triplet_visit3.csv ────────────────────────────────────────────────
print("=== Step 2a: triplet_visit3.csv ===")
triplet.to_csv('triplet_visit3.csv', index=False)
check = pd.read_csv('triplet_visit3.csv', usecols=['subject', 'longevity_class'])
print(f"Columns present, rows={len(check)}, sample:\n{check.head(3).to_string()}")
print()

# ── Step 2b: RNA-seq residuals ─────────────────────────────────────────────────
print("=== Step 2b: RNA_seq_residuals_v1_allsubjects.csv ===")
rna_path = 'data/omics_data/residuals/RNA_seq_residuals_v1_allsubjects.csv'
rna = pd.read_csv(rna_path)
rna = rna.merge(labels, on='subject', how='left')
rna.to_csv(rna_path, index=False)
check = pd.read_csv(rna_path, usecols=['subject', 'longevity_class'])
print(f"rows={len(check)}, class dist:\n{check['longevity_class'].value_counts().to_string()}")
print()

# ── Step 2c: label_data/t2dpret2d.txt ─────────────────────────────────────────
print("=== Step 2c: label_data/t2dpret2d.txt ===")
lbl_path = 'data/label_data/t2dpret2d.txt'
lbl = pd.read_csv(lbl_path, sep='\t')
lbl = lbl.merge(labels, on='subject', how='left')
lbl.to_csv(lbl_path, sep='\t', index=False)
check = pd.read_csv(lbl_path, sep='\t', usecols=['subject', 'longevity_class'])
print(f"rows={len(check)}, class dist:\n{check['longevity_class'].value_counts().to_string()}")
print()

# ── Step 2d: v1_label_phenodata_onehot_nodeidx_df.csv ─────────────────────────
print("=== Step 2d: v1_label_phenodata_onehot_nodeidx_df.csv ===")
p = 'data/filtered_data/v1_label_phenodata_onehot_nodeidx_df.csv'
df = pd.read_csv(p)
df = df.merge(labels, on='subject', how='left')
df.to_csv(p, index=False)
check = pd.read_csv(p, usecols=['subject', 'longevity_class'])
print(f"rows={len(check)}, class dist:\n{check['longevity_class'].value_counts().to_string()}")
print()

# ── Step 2e: subject_dict_df.csv ──────────────────────────────────────────────
print("=== Step 2e: subject_dict_df.csv ===")
p = 'data/filtered_data/subject_dict_df.csv'
df = pd.read_csv(p)
# subject_node_idx IS the subject ID
df = df.merge(labels, left_on='subject_node_idx', right_on='subject', how='left').drop(columns='subject')
df.to_csv(p, index=False)
check = pd.read_csv(p)
print(f"rows={len(check)}, columns={check.columns.tolist()}")
print(f"class dist:\n{check['longevity_class'].value_counts().to_string()}")
print()

# ── Step 2f: merged_*_nodeidx_df.csv (6 files) ────────────────────────────────
merged_files = [
    'data/filtered_data/merged_tran_v1_nodeidx_df.csv',
    'data/filtered_data/merged_downstream_nodeidx_df.csv',
    'data/filtered_data/merged_upstream_nodeidx_df.csv',
    'data/filtered_data/merged_core_promoter_nodeidx_df.csv',
    'data/filtered_data/merged_distal_promoter_nodeidx_df.csv',
    'data/filtered_data/merged_proximal_promoter_nodeidx_df.csv',
]
for p in merged_files:
    name = p.split('/')[-1]
    print(f"=== Step 2f: {name} ===")
    df = pd.read_csv(p)
    df = df.merge(labels, left_on='subject_nodeidx', right_on='subject', how='left').drop(columns='subject')
    df.to_csv(p, index=False)
    check = pd.read_csv(p, usecols=['subject_nodeidx', 'longevity_class'])
    print(f"rows={len(check)}, class dist:\n{check['longevity_class'].value_counts().to_string()}")
    print()

print("=== All A-class files done ===")
