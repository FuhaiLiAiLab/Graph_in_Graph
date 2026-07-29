import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

labels = pd.read_csv('subject_longevity_labels.csv').set_index('subject')['longevity_class']

def count(ids):
    cl = labels.reindex(ids)
    lo = (cl == 'longevity').sum()
    ma = (cl == 'marryin').sum()
    un = (cl == 'unclassified').sum()
    return len(ids), lo, ma, un

def row(name, ids):
    n, lo, ma, un = count(ids)
    print(f"  {name:<35s} total={n:5d}  longevity={lo:4d}  marryin={ma:4d}  unclassified={un:4d}")
    return set(ids)

# --- RNA-seq (transcriptomics) ---
rna_ids = pd.read_csv('data/omics_data/residuals/RNA_seq_residuals_v1_allsubjects.csv',
                      usecols=['subject'])['subject']
s_rna = row("RNA-seq (transcriptomics)", rna_ids)

# --- Epigenomics (all 5 regions have same subjects) ---
epi_ids = pd.read_csv('data/omics_data/epigenomics/Downstream_final_subject_labels.csv')['subject']
s_epi = row("Epigenomics (methylation, 5 regions)", epi_ids)

# --- Phenotype ---
pheno_ids = pd.read_excel('data/pheno_data/LLFS_phenos_21JUN2022.xlsx',
                          sheet_name='Phenodata', usecols=['subject'])['subject']
s_pheno = row("Phenotype (LLFS_phenos xlsx)", pheno_ids)

# --- GNN model input (intersection used in training) ---
gnn_ids = pd.read_csv('data/filtered_data/subject_dict_df.csv')['subject_node_idx']
s_gnn = row("GNN model input (filtered subset)", gnn_ids)

print()

# --- Union of all raw data ---
s_union = s_rna | s_epi | s_pheno
n, lo, ma, un = count(list(s_union))
print(f"  {'Union (any data type)':<35s} total={n:5d}  longevity={lo:4d}  marryin={ma:4d}  unclassified={un:4d}")

# --- Intersection: RNA-seq AND epigenomics AND phenotype ---
s_inter = s_rna & s_epi & s_pheno
n, lo, ma, un = count(list(s_inter))
print(f"  {'Intersection (all 3 data types)':<35s} total={n:5d}  longevity={lo:4d}  marryin={ma:4d}  unclassified={un:4d}")
