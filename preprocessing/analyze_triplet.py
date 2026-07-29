import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("triplet_visit3.csv")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("\ngen distribution:")
print(df['gen'].value_counts().sort_index())
print("\nProband_status=1 count:", int(df['Proband_status'].sum()))
print("\ngen x relative x control combinations:")
print(df.groupby(['gen','relative','control']).size().reset_index(name='count').to_string())

def classify(row):
    if row['Proband_status'] == 1:
        return "Proband"
    g, r, c = row['gen'], row['relative'], row['control']
    if g == 1 and r == 1:
        return "Gen1_parent_LONGEVITY"
    if g == 2 and r == 1 and c == 0:
        return "Gen2_sibling_LONGEVITY"
    if g == 2 and r == 0 and c == 0:
        return "Gen2_spouse_MARRYIN"
    if g == 3 and r == 1 and c == 0:
        return "Gen3_offspring_LONGEVITY"
    if g == 3 and r == 0 and c == 1:
        return "Gen3_offspring_spouse_MARRYIN"
    if g == 4 and r == 1 and c == 1:
        return "Gen4_grandchild_LONGEVITY"
    if g == 4 and r == 0 and c == 0:
        return "Gen4_grandchild_spouse_MARRYIN"
    return "Unclassified"

df['longevity_class'] = df.apply(classify, axis=1)
print("\nClassification results:")
print(df['longevity_class'].value_counts())

longevity = df[df['longevity_class'].str.contains("LONGEVITY|Proband")]
marryin = df[df['longevity_class'].str.contains("MARRYIN")]
unclassified = df[df['longevity_class'] == "Unclassified"]
print(f"\nLongevity family (incl. Proband): {len(longevity)}")
print(f"Marry-in (non-longevity): {len(marryin)}")
print(f"Unclassified: {len(unclassified)}")
if len(unclassified) > 0:
    print("Unclassified samples:")
    print(unclassified[['subject','gpedid','gen','relative','control','Proband_status']].to_string())
