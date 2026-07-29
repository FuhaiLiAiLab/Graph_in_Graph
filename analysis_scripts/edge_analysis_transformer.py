import os
import pdb
import torch
import numpy as np
import pandas as pd


def read_concat_files(model_type, layer_name, fold_n, patient_type):
    folder_path = './analysis/' + model_type + '/fold_' + str(fold_n)
    filelist = [file_name for file_name in os.listdir(folder_path) if layer_name in file_name]
    fold_gene_first_edge_weight_df_list = []
    for file_name in filelist:
        fold_gene_first_edge_weight_df = pd.read_csv(folder_path + '/' + file_name)
        fold_gene_first_edge_weight_df_list.append(fold_gene_first_edge_weight_df)
    fold_n_gene_first_edge_weight_df = pd.concat(fold_gene_first_edge_weight_df_list, axis=0)
    # Select Patient Type
    # import pdb; pdb.set_trace()
    label_phenodata_onehot_nodeidx_df = pd.read_csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
    patient_type_df = label_phenodata_onehot_nodeidx_df[label_phenodata_onehot_nodeidx_df[patient_type]==1]
    patient_index_list = patient_type_df['node_idx'].tolist()
    patient_type_fold_n_gene_first_edge_weight_df = fold_n_gene_first_edge_weight_df[fold_n_gene_first_edge_weight_df['Patient_Index'].isin(patient_index_list)]
    patient_type_fold_n_gene_first_edge_weight_df = patient_type_fold_n_gene_first_edge_weight_df.groupby(['Actual_From', 'Actual_To']).agg({'Weight':'mean'}).reset_index()
    patient_type_fold_n_gene_first_edge_weight_df = patient_type_fold_n_gene_first_edge_weight_df.sort_values(by=['Actual_From', 'Actual_To']).reset_index(drop=True)
    patient_type_fold_n_gene_first_edge_weight_df.to_csv('./analysis/' + model_type + '/fold_' + str(fold_n) + '/' + patient_type + '_' + layer_name + '.csv', index=False, header=True)
    return patient_type_fold_n_gene_first_edge_weight_df


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
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_node = gene_num_dict_df.shape[0]
    s = torch.sparse_coo_tensor(edge_index, edge_weight, (num_node, num_node))
    s1 = s.to_dense()
    min_val = torch.min(s1, dim=1, keepdim=True)[0]
    max_val = torch.max(s1, dim=1, keepdim=True)[0]
    s1_normalized = (s1 - min_val) / (max_val - min_val)
    s1_normalized[torch.isnan(s1_normalized)] = 0
    norm_edge_index = s1_normalized.to_sparse()._indices()
    norm_edge_weight = s1_normalized.to_sparse()._values()
    norm_layer_average_fold_edge_weight_df = layer_average_fold_edge_weight_df.copy()
    norm_layer_average_fold_edge_weight_df['Weight'] = norm_edge_weight.numpy().tolist()
    norm_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/'+ patient_type + '_norm_layer_average_fold_gene_edge_weight_df.csv', index=False, header=True)
    return norm_layer_average_fold_edge_weight_df



class NetAnalyse():
    def __init__(self):
        pass

    def first_average_fold(self, model_type, patient_type):
        fold_1_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_first_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_first_edge_weight', fold_n=5, patient_type=patient_type)
        first_average_fold_gene_first_edge_weight_df = (fold_1_gene_first_edge_weight_df+fold_2_gene_first_edge_weight_df+fold_3_gene_first_edge_weight_df+fold_4_gene_first_edge_weight_df+fold_5_gene_first_edge_weight_df)/5
        cols = ['Actual_From', 'Actual_To']
        first_average_fold_gene_first_edge_weight_df[cols] = first_average_fold_gene_first_edge_weight_df[cols].astype(int)
        # [weight]
        first_average_fold_gene_first_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'first_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def block_average_fold(self, model_type, patient_type):
        fold_1_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_block_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_block_edge_weight', fold_n=5, patient_type=patient_type)
        block_average_fold_gene_block_edge_weight_df = (fold_1_gene_block_edge_weight_df+fold_2_gene_block_edge_weight_df+fold_3_gene_block_edge_weight_df+fold_4_gene_block_edge_weight_df+fold_5_gene_block_edge_weight_df)/5
        cols = ['Actual_From', 'Actual_To']
        block_average_fold_gene_block_edge_weight_df[cols] = block_average_fold_gene_block_edge_weight_df[cols].astype(int)
        # [weight]
        block_average_fold_gene_block_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'block_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def last_average_fold(self, model_type, patient_type):
        fold_1_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=1, patient_type=patient_type)
        fold_2_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=2, patient_type=patient_type)
        fold_3_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=3, patient_type=patient_type)
        fold_4_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=4, patient_type=patient_type)
        fold_5_gene_last_edge_weight_df = read_concat_files(model_type=model_type, layer_name='gene_last_edge_weight', fold_n=5, patient_type=patient_type)
        last_average_fold_gene_last_edge_weight_df = (fold_1_gene_last_edge_weight_df+fold_2_gene_last_edge_weight_df+fold_3_gene_last_edge_weight_df+fold_4_gene_last_edge_weight_df+fold_5_gene_last_edge_weight_df)/5
        cols = ['Actual_From', 'Actual_To']
        last_average_fold_gene_last_edge_weight_df[cols] = last_average_fold_gene_last_edge_weight_df[cols].astype(int)
        # [weight]
        last_average_fold_gene_last_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'last_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def layer_average_fold(self, model_type, patient_type):
        first_average_fold_gene_first_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'first_average_fold_gene_edge_weight_df.csv')
        block_average_fold_gene_block_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'block_average_fold_gene_edge_weight_df.csv')
        last_average_fold_gene_last_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_' +  'last_average_fold_gene_edge_weight_df.csv')
        layer_average_fold_gene_first_edge_weight_df = (first_average_fold_gene_first_edge_weight_df + block_average_fold_gene_block_edge_weight_df + last_average_fold_gene_last_edge_weight_df)/3
        cols = ['Actual_From', 'Actual_To']
        layer_average_fold_gene_first_edge_weight_df[cols] = layer_average_fold_gene_first_edge_weight_df[cols].astype(int)
        # [weight]
        layer_average_fold_gene_first_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_layer_average_fold_gene_edge_weight_df.csv', index=False, header=True)

    def edge_node_analysis(self, model_type, patient_type):
        norm_layer_average_fold_edge_weight_df = edge_minmax_nromalization(model_type, patient_type)
        # import pdb; pdb.set_trace()
        to_layer_average_fold_edge_weight_df = norm_layer_average_fold_edge_weight_df.groupby(['Actual_To']).agg({'Weight':'mean'}).reset_index()
        from_layer_average_fold_edge_weight_df = norm_layer_average_fold_edge_weight_df.groupby(['Actual_From']).agg({'Weight':'mean'}).reset_index()
        to_layer_average_fold_edge_weight_df = to_layer_average_fold_edge_weight_df.rename(columns={'Actual_To': 'Node_idx'})
        from_layer_average_fold_edge_weight_df = from_layer_average_fold_edge_weight_df.rename(columns={'Actual_From': 'Node_idx'})
        node_layer_average_fold_edge_weight_df = pd.concat([to_layer_average_fold_edge_weight_df, from_layer_average_fold_edge_weight_df], axis=0)
        node_layer_average_fold_edge_weight_df = node_layer_average_fold_edge_weight_df.groupby(['Node_idx']).agg({'Weight':'mean'}).reset_index()
        node_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_layer_average_fold_node_weight_df.csv', index=False, header=True)

    def core_signaling_analysis(self, model_type, patient_type, edge_percentile, node_percentile):
        # Percentile edge weight
        norm_layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/'+ patient_type + '_norm_layer_average_fold_gene_edge_weight_df.csv')
        edge_weight_array = np.array(norm_layer_average_fold_edge_weight_df['Weight'].values)
        edge_percentile_number = np.percentile(edge_weight_array, edge_percentile)
        print('edge_percentile_number: ', edge_percentile_number)
        print(edge_weight_array.max(), edge_weight_array.min())
        filtered_edge_layer_average_fold_edge_weight_df = norm_layer_average_fold_edge_weight_df[norm_layer_average_fold_edge_weight_df['Weight']>=edge_percentile_number]
        filtered_edge_layer_average_fold_edge_weight_df.to_csv('./analysis/' + model_type + '/' + patient_type + '_filtered_edge_layer_average_fold_edge_weight_df.csv', index=False, header=True)
        # Percentile node weight
        node_layer_average_fold_edge_weight_df = pd.read_csv('./analysis/' + model_type + '/' + patient_type + '_layer_average_fold_node_weight_df.csv')
        node_weight_array = np.array(node_layer_average_fold_edge_weight_df['Weight'].values)
        node_percentile_number = np.percentile(node_weight_array, node_percentile)
        print('node_percentile_number: ', node_percentile_number)
        print(node_weight_array.max(), node_weight_array.min())




if __name__ == "__main__":
    model_type = 'gigtransformer'
    patient_type_list = ['t2ds', 'pret2ds', 'no_t2ds']

    for patient_type in patient_type_list:
        print(f'--- Processing patient type: {patient_type} ---')
        NetAnalyse().first_average_fold(model_type=model_type, patient_type=patient_type)
        NetAnalyse().block_average_fold(model_type=model_type, patient_type=patient_type)
        NetAnalyse().last_average_fold(model_type=model_type, patient_type=patient_type)
        NetAnalyse().layer_average_fold(model_type=model_type, patient_type=patient_type)
        NetAnalyse().edge_node_analysis(model_type=model_type, patient_type=patient_type)
        NetAnalyse().core_signaling_analysis(model_type=model_type, patient_type=patient_type, edge_percentile=85, node_percentile=85)