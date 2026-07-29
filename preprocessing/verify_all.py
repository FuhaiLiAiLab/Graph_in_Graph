import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

def check(path, id_col, sep=None):
    kw = {'sep': sep} if sep else {}
    df = pd.read_csv(path, usecols=[id_col, 'longevity_class'], **kw)
    n = len(df)
    lo = (df['longevity_class'] == 'longevity').sum()
    ma = (df['longevity_class'] == 'marryin').sum()
    un = (df['longevity_class'] == 'unclassified').sum()
    name = path.split('/')[-1]
    print(f"{name:<55s} rows={n:5d}  longevity={lo:4d}  marryin={ma:4d}  unclassified={un:4d}")

check('subject_longevity_labels.csv', 'subject')
check('triplet_visit3.csv', 'subject')
check('data/omics_data/residuals/RNA_seq_residuals_v1_allsubjects.csv', 'subject')
check('data/label_data/t2dpret2d.txt', 'subject', sep='\t')
check('data/filtered_data/v1_label_phenodata_onehot_nodeidx_df.csv', 'subject')
check('data/filtered_data/subject_dict_df.csv', 'subject_node_idx')
check('data/filtered_data/merged_tran_v1_nodeidx_df.csv', 'subject_nodeidx')
check('data/filtered_data/merged_downstream_nodeidx_df.csv', 'subject_nodeidx')
check('data/filtered_data/merged_upstream_nodeidx_df.csv', 'subject_nodeidx')
check('data/filtered_data/merged_core_promoter_nodeidx_df.csv', 'subject_nodeidx')
check('data/filtered_data/merged_distal_promoter_nodeidx_df.csv', 'subject_nodeidx')
check('data/filtered_data/merged_proximal_promoter_nodeidx_df.csv', 'subject_nodeidx')

for nm in ['Downstream','Proximal_Promoter','Core_Promoter','Distal_Promoter','Upstream']:
    check(f'data/omics_data/epigenomics/{nm}_final_subject_labels.csv', 'subject')

# Phenotype xlsx
df = pd.read_excel('data/pheno_data/LLFS_phenos_21JUN2022.xlsx', sheet_name='Phenodata',
                   usecols=['subject', 'longevity_class'])
lo = (df['longevity_class'] == 'longevity').sum()
ma = (df['longevity_class'] == 'marryin').sum()
un = (df['longevity_class'] == 'unclassified').sum()
print(f"{'LLFS_phenos_21JUN2022.xlsx (Phenodata)':<55s} rows={len(df):5d}  longevity={lo:4d}  marryin={ma:4d}  unclassified={un:4d}")
