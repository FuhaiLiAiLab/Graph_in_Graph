import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

labels = pd.read_csv('subject_longevity_labels.csv')
label_map = labels.set_index('subject')['longevity_class'].to_dict()

epi_files = [
    'data/omics_data/epigenomics/Downstream_final.csv',
    'data/omics_data/epigenomics/Proximal_Promoter_final.csv',
    'data/omics_data/epigenomics/Core_Promoter_final.csv',
    'data/omics_data/epigenomics/Distal_Promoter_final.csv',
    'data/omics_data/epigenomics/Upstream_final.csv',
]

for path in epi_files:
    name = path.split('/')[-1].replace('.csv', '')
    # Read only the header row to get subject IDs (column names)
    header = pd.read_csv(path, nrows=0)
    subject_cols = [c for c in header.columns if c != 'gene_name']
    subjects_int = [int(c) for c in subject_cols]

    annot = pd.DataFrame({
        'subject': subjects_int,
        'longevity_class': [label_map.get(s, 'unclassified') for s in subjects_int]
    })
    out_path = path.replace('.csv', '_subject_labels.csv')
    annot.to_csv(out_path, index=False)

    counts = annot['longevity_class'].value_counts()
    print(f"{name}_subject_labels.csv: {len(annot)} subjects")
    print(counts.to_string())
    print()

print("=== All B-class annotation files done ===")
