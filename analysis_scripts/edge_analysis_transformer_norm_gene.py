import os
import pdb
import torch
import numpy as np
import pandas as pd

from torch_geometric.utils import degree


def read_concat_files(model_type, layer_name, fold_n, patient_type):
    # Read [gene_edge_weight] on each layer [first, block, last]
    patient_type_fold_n_gene_edge_weight_df = pd.read_csv('./analysis/gigtransformer/fold_' + str(fold_n) + '/' + patient_type + '_' + layer_name + '.csv')
    add_node_degree(patient_type_fold_n_gene_edge_weight_df)
    # Add normalization on each node
    # Make directory for each fold under the model type
    if not os.path.exists('./analysis/' + model_type + '/fold_' + str(fold_n)):
        os.makedirs('./analysis/' + model_type + '/fold_' + str(fold_n))
    Actual_From_list = patient_type_fold_n_gene_edge_weight_df['Actual_From'].unique().tolist()
    # [from]
    from_node_df_list = []
    for Actual_From in Actual_From_list:
        from_node_df = patient_type_fold_n_gene_edge_weight_df[patient_type_fold_n_gene_edge_weight_df['Actual_From']==Actual_From]
        # import pdb; pdb.set_trace()
        if model_type == 'gigtransformer-norm':
            from_node_max_weight = from_node_df['Weight'].max()
            from_node_min_weight = from_node_df['Weight'].min()
            from_node_df['Weight'] = (from_node_df['Weight'] - from_node_min_weight)/ (from_node_max_weight - from_node_min_weight)
        elif model_type == 'gigtransformer-binorm':
            from_node_df['Weight'] = from_node_df['Weight'] / np.sqrt(from_node_df['Degree_Row'] * from_node_df['Degree_Col'])
        elif model_type == 'gigtransformer-rownorm':
            from_node_df['Weight'] = from_node_df['Weight'] / from_node_df['Degree_Row']
        from_node_df_list.append(from_node_df)
    from_patient_type_fold_n_gene_edge_weight_df = pd.concat(from_node_df_list, axis=0)
    from_patient_type_fold_n_gene_edge_weight_df = from_patient_type_fold_n_gene_edge_weight_df.fillna(0.0)
    # [to]
    norm_patient_type_fold_n_gene_edge_weight_df = from_patient_type_fold_n_gene_edge_weight_df.copy()
    norm_patient_type_fold_n_gene_edge_weight_df.to_csv('./analysis/' + model_type + '/fold_' + str(fold_n) + '/' + patient_type + '_norm_' + layer_name + '.csv', index=False, header=True)
    return patient_type_fold_n_gene_edge_weight_df, norm_patient_type_fold_n_gene_edge_weight_df


def edge_minmax_nromalization(model_type, patient_type):
    # Load original [edge_weight]
    layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/'+ patient_type + '_layer_average_fold_gene_edge_weight_df.csv')
    num_edges = layer_average_fold_edge_weight_df.shape[0]
    edge_index = np.zeros((2, num_edges))
    edge_weight = np.zeros((1, num_edges))
    edge_index[0, :] = layer_average_fold_edge_weight_df['Actual_From'].values
    edge_index[1, :] = layer_average_fold_edge_weight_df['Actual_To'].values
    edge_weight = layer_average_fold_edge_weight_df['Weight'].values + 1e-15
    # Normalize the [edge_weight]
    # Read [gene_num_dict_df]
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    s = torch.sparse_coo_tensor(edge_index, edge_weight, (num_gene_node, num_gene_node))
    s1 = s.to_dense()
    min_val = torch.min(s1, dim=1, keepdim=True)[0]
    max_val = torch.max(s1, dim=1, keepdim=True)[0]
    s1_normalized = (s1 - min_val) / (max_val - min_val)
    s1_normalized[np.isnan(s1_normalized)] = 0
    norm_edge_index = s1_normalized.to_sparse()._indices()
    norm_edge_weight = s1_normalized.to_sparse()._values()
    norm_layer_average_fold_edge_weight_df = layer_average_fold_edge_weight_df.copy()
    norm_layer_average_fold_edge_weight_df['Weight'] = norm_edge_weight.tolist()
    norm_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/'+ patient_type + '_norm_layer_average_fold_gene_edge_weight_df.csv', index=False, header=True)
    return norm_layer_average_fold_edge_weight_df


class NetAnalyse():
    def __init__(self):
        pass

    def first_average_fold(self, model_type, patient_type):
        fold_1_gene_first_edge_weight_df, fold_1_norm_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_first_edge_weight_df, fold_2_norm_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_first_edge_weight_df, fold_3_norm_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_first_edge_weight_df, fold_4_norm_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_first_edge_weight_df, fold_5_norm_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=5, patient_type=patient_type)
        first_average_fold_gene_first_edge_weight_df = (fold_1_gene_first_edge_weight_df+fold_2_gene_first_edge_weight_df+fold_3_gene_first_edge_weight_df+fold_4_gene_first_edge_weight_df+fold_5_gene_first_edge_weight_df)/5
        first_norm_average_fold_gene_first_edge_weight_df = (fold_1_norm_gene_first_edge_weight_df+fold_2_norm_gene_first_edge_weight_df+fold_3_norm_gene_first_edge_weight_df+fold_4_norm_gene_first_edge_weight_df+fold_5_norm_gene_first_edge_weight_df)/5
        add_node_degree(first_average_fold_gene_first_edge_weight_df)
        add_node_degree(first_norm_average_fold_gene_first_edge_weight_df)
        cols = ['Actual_From', 'Actual_To', 'Degree_Row', 'Degree_Col']
        first_average_fold_gene_first_edge_weight_df[cols] = first_average_fold_gene_first_edge_weight_df[cols].astype(int)
        first_norm_average_fold_gene_first_edge_weight_df[cols] = first_norm_average_fold_gene_first_edge_weight_df[cols].astype(int)
        # [weight]
        first_norm_average_fold_gene_first_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'first_norm_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def block_average_fold(self, model_type, patient_type):
        fold_1_gene_block_edge_weight_df, fold_1_norm_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_block_edge_weight_df, fold_2_norm_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_block_edge_weight_df, fold_3_norm_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_block_edge_weight_df, fold_4_norm_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_block_edge_weight_df, fold_5_norm_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=5, patient_type=patient_type)
        block_average_fold_gene_block_edge_weight_df = (fold_1_gene_block_edge_weight_df+fold_2_gene_block_edge_weight_df+fold_3_gene_block_edge_weight_df+fold_4_gene_block_edge_weight_df+fold_5_gene_block_edge_weight_df)/5
        block_norm_average_fold_gene_block_edge_weight_df = (fold_1_norm_gene_block_edge_weight_df+fold_2_norm_gene_block_edge_weight_df+fold_3_norm_gene_block_edge_weight_df+fold_4_norm_gene_block_edge_weight_df+fold_5_norm_gene_block_edge_weight_df)/5
        add_node_degree(block_average_fold_gene_block_edge_weight_df)
        add_node_degree(block_norm_average_fold_gene_block_edge_weight_df)
        cols = ['Actual_From', 'Actual_To', 'Degree_Row', 'Degree_Col']
        block_average_fold_gene_block_edge_weight_df[cols] = block_average_fold_gene_block_edge_weight_df[cols].astype(int)
        block_norm_average_fold_gene_block_edge_weight_df[cols] = block_norm_average_fold_gene_block_edge_weight_df[cols].astype(int)
        # [weight]
        block_norm_average_fold_gene_block_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'block_norm_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def last_average_fold(self, model_type, patient_type):
        fold_1_gene_last_edge_weight_df, fold_1_norm_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_last_edge_weight_df, fold_2_norm_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_last_edge_weight_df, fold_3_norm_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_last_edge_weight_df, fold_4_norm_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_last_edge_weight_df, fold_5_norm_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=5, patient_type=patient_type)
        last_average_fold_gene_last_edge_weight_df = (fold_1_gene_last_edge_weight_df+fold_2_gene_last_edge_weight_df+fold_3_gene_last_edge_weight_df+fold_4_gene_last_edge_weight_df+fold_5_gene_last_edge_weight_df)/5
        last_norm_average_fold_gene_last_edge_weight_df = (fold_1_norm_gene_last_edge_weight_df+fold_2_norm_gene_last_edge_weight_df+fold_3_norm_gene_last_edge_weight_df+fold_4_norm_gene_last_edge_weight_df+fold_5_norm_gene_last_edge_weight_df)/5
        add_node_degree(last_average_fold_gene_last_edge_weight_df)
        add_node_degree(last_norm_average_fold_gene_last_edge_weight_df)
        cols = ['Actual_From', 'Actual_To', 'Degree_Row', 'Degree_Col']
        last_average_fold_gene_last_edge_weight_df[cols] = last_average_fold_gene_last_edge_weight_df[cols].astype(int)
        last_norm_average_fold_gene_last_edge_weight_df[cols] = last_norm_average_fold_gene_last_edge_weight_df[cols].astype(int)
        # [weight]
        last_norm_average_fold_gene_last_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'last_norm_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def layer_average_fold(self, model_type, patient_type):
        first_norm_average_fold_gene_first_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'first_norm_average_fold_gene_edge_weight_df.csv')
        block_norm_average_fold_gene_block_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'block_norm_average_fold_gene_edge_weight_df.csv')
        last_norm_average_fold_gene_last_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'last_norm_average_fold_gene_edge_weight_df.csv')
        layer_norm_average_fold_gene_first_edge_weight_df = (first_norm_average_fold_gene_first_edge_weight_df + block_norm_average_fold_gene_block_edge_weight_df + last_norm_average_fold_gene_last_edge_weight_df)/3
        add_node_degree(layer_norm_average_fold_gene_first_edge_weight_df)
        cols = ['Actual_From', 'Actual_To', 'Degree_Row', 'Degree_Col']
        layer_norm_average_fold_gene_first_edge_weight_df[cols] = layer_norm_average_fold_gene_first_edge_weight_df[cols].astype(int)
        # [weight]
        layer_norm_average_fold_gene_first_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_layer_norm_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def norm_edge_node_analysis(self, model_type, patient_type):
        layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/'+ patient_type + '_layer_norm_average_fold_gene_edge_weight_df.csv')
        to_layer_average_fold_edge_weight_df = layer_average_fold_edge_weight_df.groupby(['Actual_To']).agg({'Weight':'mean'}).reset_index()
        from_layer_average_fold_edge_weight_df = layer_average_fold_edge_weight_df.groupby(['Actual_From']).agg({'Weight':'mean'}).reset_index()
        to_layer_average_fold_edge_weight_df = to_layer_average_fold_edge_weight_df.rename(columns={'Actual_To': 'Node_idx'})
        from_layer_average_fold_edge_weight_df = from_layer_average_fold_edge_weight_df.rename(columns={'Actual_From': 'Node_idx'})
        node_layer_average_fold_edge_weight_df = pd.concat([to_layer_average_fold_edge_weight_df, from_layer_average_fold_edge_weight_df], axis=0)
        node_layer_average_fold_edge_weight_df = node_layer_average_fold_edge_weight_df.groupby(['Node_idx']).agg({'Weight':'mean'}).reset_index()
        node_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_layer_norm_average_fold_node_weight_df.csv', index=False, header=True)

    def norm_core_signaling_analysis(self, model_type, patient_type, edge_percentile, node_percentile):
        # Percentile edge weight
        norm_layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/'+ patient_type + '_layer_norm_average_fold_gene_edge_weight_df.csv')
        edge_weight_array = np.array(norm_layer_average_fold_edge_weight_df['Weight'].values)
        edge_percentile_number = np.percentile(edge_weight_array, edge_percentile)
        print('edge_percentile_number: ', edge_percentile_number)
        print(edge_weight_array.max(), edge_weight_array.min())
        filtered_edge_layer_average_fold_edge_weight_df = norm_layer_average_fold_edge_weight_df[norm_layer_average_fold_edge_weight_df['Weight']>=edge_percentile_number]
        filtered_edge_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_norm_filtered_edge_layer_average_fold_edge_weight_df.csv', index=False, header=True)
        # Percentile node weight
        node_layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_layer_norm_average_fold_node_weight_df.csv')
        node_weight_array = np.array(node_layer_average_fold_edge_weight_df['Weight'].values)
        node_percentile_number = np.percentile(node_weight_array, node_percentile)
        print('node_percentile_number: ', node_percentile_number)
        print(node_weight_array.max(), node_weight_array.min())


def add_node_degree(input_df):
    # Read [gene_num_edge_df]
    gene_num_edge_df = pd.read_csv('./data/filtered_data/gene_num_edge_df.csv')
    # Read [gene_num_dict_df]
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    # Convert the 2 columns to edge index
    edge_index = np.zeros((2, gene_num_edge_df.shape[0]))
    edge_index[0, :] = gene_num_edge_df['From'].values
    edge_index[1, :] = gene_num_edge_df['To'].values
    # print('edge_index: ', edge_index)
    # Convert to torch tensor
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    row, col = edge_index_tensor
    deg_col = degree(col, num_gene_node, dtype=torch.float)
    deg_row = degree(row, num_gene_node, dtype=torch.float)
    input_df['Degree_Row'] = deg_row[row].cpu().detach().numpy()
    input_df['Degree_Col'] = deg_col[col].cpu().detach().numpy()
    return None


if __name__ == "__main__":
    ## Model Parameters
    model_type_list = ['gigtransformer-norm', 'gigtransformer-binorm', 'gigtransformer-rownorm']
    patient_type_list = ['t2ds', 'pret2ds', 'no_t2ds']

    for model_type in model_type_list:
        # Make directory if model directory does not exist
        if not os.path.exists('./analysis/' + model_type):
            os.makedirs('./analysis/' + model_type)
        # Generate edge weight files for each patient type in each layer
        for patient_type in patient_type_list:
            ### Average Fold
            NetAnalyse().first_average_fold(model_type=model_type, patient_type=patient_type)
            NetAnalyse().block_average_fold(model_type=model_type, patient_type=patient_type)
            NetAnalyse().last_average_fold(model_type=model_type, patient_type=patient_type)
            NetAnalyse().layer_average_fold(model_type=model_type, patient_type=patient_type)
            ### Patient specific
            NetAnalyse().norm_edge_node_analysis(model_type=model_type, patient_type=patient_type)
            NetAnalyse().norm_core_signaling_analysis(model_type=model_type, patient_type=patient_type, edge_percentile=85, node_percentile=85)