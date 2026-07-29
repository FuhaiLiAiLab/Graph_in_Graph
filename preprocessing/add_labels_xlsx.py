import pandas as pd
import sys
from openpyxl import load_workbook
sys.stdout.reconfigure(encoding='utf-8')

labels = pd.read_csv('subject_longevity_labels.csv')
xlsx_path = 'data/pheno_data/LLFS_phenos_21JUN2022.xlsx'

xl = pd.ExcelFile(xlsx_path)
print("Sheets:", xl.sheet_names)

phenodata = xl.parse('Phenodata')
print(f"Phenodata shape before: {phenodata.shape}")
print(f"Columns 0-4: {phenodata.columns[:5].tolist()}")

phenodata_labeled = phenodata.merge(labels, on='subject', how='left')
print(f"Phenodata shape after: {phenodata_labeled.shape}")
print(f"longevity_class dist:\n{phenodata_labeled['longevity_class'].value_counts().to_string()}")

# Write back: preserve other sheets, only update Phenodata
with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    phenodata_labeled.to_excel(writer, sheet_name='Phenodata', index=False)

# Verify
check = pd.read_excel(xlsx_path, sheet_name='Phenodata', usecols=['subject', 'longevity_class'])
print(f"\nVerify: rows={len(check)}, sample:\n{check.head(3).to_string()}")
print("Done.")
