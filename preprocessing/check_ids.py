import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

triplet = pd.read_csv("triplet_visit3.csv")
triplet_subjects = set(triplet['subject'].astype(str))
print(f"Triplet: {len(triplet_subjects)} unique subjects")

# RNA-seq
rna = pd.read_csv("data/omics_data/residuals/RNA_seq_residuals_v1_allsubjects.csv", usecols=['subject'])
rna_ids = set(rna['subject'].astype(str))
overlap_rna = rna_ids & triplet_subjects
print(f"\nRNA-seq subjects: {len(rna_ids)}, overlap with triplet: {len(overlap_rna)}")

# Epigenomics - subjects are column headers
epi = pd.read_csv("data/omics_data/epigenomics/Downstream_final.csv", nrows=0)
epi_ids = set(c for c in epi.columns if c != 'gene_name')
overlap_epi = epi_ids & triplet_subjects
print(f"Epigenomics (Downstream) subjects: {len(epi_ids)}, overlap with triplet: {len(overlap_epi)}")

# Label data
label = pd.read_csv("data/label_data/t2dpret2d.txt", sep='\t')
label_ids = set(label['subject'].astype(str))
overlap_label = label_ids & triplet_subjects
print(f"Label data subjects: {len(label_ids)}, overlap with triplet: {len(overlap_label)}")

# Subject dict (filtered data)
sd = pd.read_csv("data/filtered_data/subject_dict_df.csv")
print(f"\nSubject dict head:\n{sd.head(5).to_string()}")
sd_ids = set(sd['subject_node_idx'].astype(str))
print(f"Subject dict node_idx: {len(sd_ids)}")

# Check if subject_node_idx in subject_dict matches triplet subjects
sd_subj = set(sd['subject_node_idx'].astype(str))
overlap_sd = sd_subj & triplet_subjects
print(f"subject_node_idx overlap with triplet: {len(overlap_sd)}")

# Now classify RNA-seq subjects
rna_with_class = rna.merge(triplet[['subject','gen','relative','control','Proband_status']], on='subject', how='left')
def classify(row):
    if row['Proband_status'] == 1:
        return "Proband"
    g, r, c = row['gen'], row['relative'], row['control']
    if g == 1 and r == 1: return "Gen1_parent_LONGEVITY"
    if g == 2 and r == 1 and c == 0: return "Gen2_sibling_LONGEVITY"
    if g == 2 and r == 0 and c == 0: return "Gen2_spouse_MARRYIN"
    if g == 3 and r == 1 and c == 0: return "Gen3_offspring_LONGEVITY"
    if g == 3 and r == 0 and c == 1: return "Gen3_offspring_spouse_MARRYIN"
    if g == 4 and r == 1 and c == 1: return "Gen4_grandchild_LONGEVITY"
    if g == 4 and r == 0 and c == 0: return "Gen4_grandchild_spouse_MARRYIN"
    return "Unclassified"
rna_with_class['class'] = rna_with_class.apply(classify, axis=1)
print(f"\nRNA-seq subjects classification:")
print(rna_with_class['class'].value_counts())
