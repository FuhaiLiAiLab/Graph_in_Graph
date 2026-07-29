import pandas as pd
from scipy.stats import mannwhitneyu
import os

# Function to separate the data based on 't2ds', 'pret2ds', and 'no_t2ds'
def separate_patient_groups(file_path):
    data = pd.read_csv(file_path)
    
    # Separate into T, Pre, and No groups based on the one-hot encoded columns
    t_group = data[data['t2ds'] == 1]
    pre_group = data[data['pret2ds'] == 1]
    no_group = data[data['no_t2ds'] == 1]
    
    return t_group, pre_group, no_group

# Function to calculate p-values for two groups
def calculate_pvalues(group_one, group_two, upstream_file):
    # Read the upstream file and merge on 'subject_nodeidx'
    upstream_data = pd.read_csv(upstream_file)
    
    # Merge groups based on 'subject_nodeidx'
    group_one_data = group_one.merge(upstream_data, left_on="node_idx", right_on="subject_nodeidx")
    group_two_data = group_two.merge(upstream_data, left_on="node_idx", right_on="subject_nodeidx")

    NON_GENE_COLS = {'subject_nodeidx', 'longevity_class'}
    results = []
    # Loop through all gene columns (skipping 'subject_nodeidx' and annotation columns)
    for gene in upstream_data.columns[1:]:
        if gene in NON_GENE_COLS:
            continue
        group_one_values = group_one_data[gene].values.flatten()
        group_two_values = group_two_data[gene].values.flatten()
        
        # Skip if any values are missing
        if len(group_one_values) == 0 or len(group_two_values) == 0:
            print(f"Skipping {gene}: Missing values")
            continue
        
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group_one_values, group_two_values)
        results.append({'gene': gene, 'p_value': p_value})

    return pd.DataFrame(results)

def process_files(random_label_file, upstream_files, regions, output_folder):
    # Separate the groups
    t_group, pre_group, no_group = separate_patient_groups(random_label_file)
    
    # Define the comparisons
    outcome_pairs = [
        ('TvsNO', t_group, no_group),
        ('TvsPre', t_group, pre_group),
        ('PrevsNO', pre_group, no_group)
    ]
    
    # Create separate folders for each comparison
    for comparison, group_one, group_two in outcome_pairs:
        comparison_folder = os.path.join(output_folder, comparison)
        os.makedirs(comparison_folder, exist_ok=True)
        
        for upstream_file, region_name in zip(upstream_files, regions):
            results = calculate_pvalues(group_one, group_two, upstream_file)
            # Modify the output file name to include comparison name
            output_file_name = f"{comparison}_{region_name}_pvalues.csv"
            output_file = os.path.join(comparison_folder, output_file_name)
            results.to_csv(output_file, index=False)
            print(f"Saved p-values for {region_name} in {comparison_folder} as {output_file_name}")

# File paths
random_label_file = './data/filtered_data/random_label_phenodata_onehot_nodeidx_df.csv'
upstream_files = [
    './data/filtered_data/merged_core_promoter_nodeidx_df.csv',
    './data/filtered_data/merged_distal_promoter_nodeidx_df.csv',
    './data/filtered_data/merged_downstream_nodeidx_df.csv',
    './data/filtered_data/merged_proximal_promoter_nodeidx_df.csv',
    './data/filtered_data/merged_tran_v1_nodeidx_df.csv',
    './data/filtered_data/merged_upstream_nodeidx_df.csv'
]

# Regions
regions = ['core_promoter', 'distal_promoter', 'downstream', 'proximal_promoter', 'tran_v1', 'upstream']

output_folder = './pvalues_output/'

# Process and calculate p-values for all comparisons
process_files(random_label_file, upstream_files, regions, output_folder)
