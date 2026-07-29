import os
import glob
import torch
import argparse
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from geo_loader.geo_readgraph import read_geodata
from enc.geo_gigtransformer_analysis import GIG_Transformer



def build_geogig_model(args, num_gene_node, device):
    print('--- BUILDING UP GNN MODEL ... ---')
    # Get parameters
    model = GIG_Transformer(gene_input_dim=args.gene_input_dim, 
                gene_hidden_dim=args.gene_hidden_dim, 
                gene_embedding_dim=args.gene_output_dim,
                gene_num_top_feature=args.gene_num_top_feature,
                num_gene_node=num_gene_node,
                gig_input_dim=args.gig_input_dim,
                gig_input_transform_dim=args.gig_input_transform_dim,
                gig_hidden_dim=args.gig_hidden_dim, 
                gig_embedding_dim=args.gig_output_dim,
                num_classes=args.num_classes, 
                gene_num_head=args.gene_num_head,
                gig_num_head=args.gig_num_head,
                class_weight_fine=args.class_weight_fine,
                class_weight=args.class_weight,
                ortho_weight=args.ortho_weight,
                link_weight=args.link_weight,
                ent_weight=args.ent_weight,
                graph_opt=args.graph_opt)
    model = model.to(device)
    return model


def test_geogig_model(fold_n, data, num_feature, num_subfeature, num_subject, num_gene_node,
                        gene_feature, gene_edge_index, model, device, args):
    loss = 0
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(fold_n, num_feature=num_feature, num_subfeature=num_subfeature, 
                                                    num_subject=num_subject, num_gene_node=num_gene_node, 
                                                    gene_feature=gene_feature, gene_edge_index=gene_edge_index,
                                                    x=x, edge_index=edge_index, 
                                                    node_label=node_label, node_index=node_index,
                                                    args=args, device=device)
    loss = model.loss(node_output, node_label, gene_edge_index, args.gene_num_top_feature)
    loss = loss.item()
    return model, loss, x_embed, node_output, ypred, y_nodepred


def test_geogig(fold_n, model, device, args):
    # Test model on training dataset with [gene features]
    num_feature = 6
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    # gene_feature = np.load('./data/post_data/gene_x.npy', allow_pickle=True)
    gene_feature = np.load('./data/post_data/norm_gene_x.npy', allow_pickle=True)
    gene_feature = gene_feature.astype(np.float32)  # or np.float64 depending on your data
    gene_edge_index = np.load('./data/post_data/gene_edge_index.npy', allow_pickle=True)
    gene_edge_index = gene_edge_index.astype(np.int64)  # or np.float64 depending on your data
    gene_edge_index = torch.from_numpy(gene_edge_index).long()

    # Test model on training dataset with [subject features]
    subfeature_dict_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
    num_subfeature = subfeature_dict_df.shape[0]
    subject_dict_df = pd.read_csv('./data/filtered_data/subject_dict_df.csv')
    num_subject = subject_dict_df.shape[0]
    # graph_feature = np.load('./data/post_data/x.npy')
    graph_feature = np.load('./data/post_data/norm_x.npy')
    edge_index = np.load('./data/post_data/edge_index.npy')
    node_label = np.load('./data/post_data/test_label_' + str(fold_n)  + '.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load('./data/post_data/test_idx_' + str(fold_n)  + '.npy')

    # Run test model
    model.eval()
    geo_data = read_geodata(graph_feature, edge_index, node_label, node_idx)
    print('TEST MODEL ...')
    model, test_loss, x_embed, node_output, ypred, y_nodepred = test_geogig_model(fold_n, geo_data, 
                                                                num_feature, num_subfeature, num_subject, num_gene_node,
                                                                gene_feature, gene_edge_index,
                                                                model, device, args)
    print('TEST LOSS: ', test_loss)
    y_nodepred = y_nodepred.cpu().detach().numpy()
    test_correct_count = (y_nodepred == node_label_indices).sum()
    test_accuracy = float(test_correct_count) / len(node_label_indices)
    print('TEST CORRECT: ', test_correct_count)
    print('TEST ACCURACY: ', test_accuracy)
    # Compute confusion matrix
    test_confusion_matrix = confusion_matrix(node_label_indices, y_nodepred)
    print(test_confusion_matrix)
    # Create [test_label_df] for saving preparation
    test_label_df = pd.DataFrame({'test_node_idx': list(node_idx),
                                'test_label': list(node_label_indices),
                                'test_pred_label': list(y_nodepred)})
    return test_accuracy, test_confusion_matrix, test_label_df, test_loss

# Parse arguments from command line
def arg_parse():
    parser = argparse.ArgumentParser(description='GEO-WEBGNN ARGUMENTS.')
    # Set default input argument
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        add_self = '0', # 'add'
                        adj = '0', # 'sym'
                        model = '0', # 'load'
                        lr = 0.003938420135801312,
                        weight_decay = 0.00018042132714427627,
                        milestones = [150, 300, 450],
                        gamma = 0.95,
                        clip = 2.0,
                        batch_size = 256,
                        num_epochs = 500,
                        unchanged_threshold = 100,
                        change_wave = 0.75,
                        num_workers = 0,
                        # graph_opt = 'GinG',
                        # graph_opt = 'subject',
                        graph_opt = 'gene',
                        gene_input_dim = 6, # gene embedding parameters
                        gene_hidden_dim = 18,
                        gene_output_dim = 18,
                        gene_num_top_feature = 18,
                        gig_input_dim = 42, # gig embedding parameters
                        gig_input_transform_dim = 18,
                        gig_hidden_dim = 18,
                        gig_output_dim = 18,
                        class_weight_fine = 0.7746597442093394,
                        class_weight = 0.8209978466887934, # loss function parameters
                        ortho_weight = 0.007968476155213464,
                        link_weight = 0.002195183531312866,
                        ent_weight = 0.09039903257209198,
                        num_classes = 3,
                        gene_num_head = 1,
                        gig_num_head = 1,
                        dropout = 0.011645650071916447)
    return parser.parse_args()


def analysis_model(fold_n, args, device):
    # Build [GinG Transformer] Model
    fold_to_unit = {1: 1, 2: 4, 3: 2, 4: 2, 5: 2}
    test_load_path = './gnn_result/gigtransformer_gene/epoch_' + str(args.num_epochs) + '_fold' + str(fold_n) + '-' + str(fold_to_unit[fold_n]) + '/best_train_model.pth'
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    model = build_geogig_model(prog_args, num_gene_node, device)
    model.load_state_dict(torch.load(test_load_path, map_location=device))
    test_accuracy, test_confusion_matrix, test_label_df, test_loss = test_geogig(fold_n, model, device, prog_args)



def average_sample_gene(fold_n):
    ### Collect files name with 'gene_first_edge_weight' under folder [analysis/gigtransformer/]
    # Define the folder path
    folder_path = 'analysis/gigtransformer/fold_' + str(fold_n)
    # Use glob to find files containing 'gene_first_edge_weight' in their names
    file_pattern = os.path.join(folder_path, '*gene_first_edge_weight*')
    print('FILE PATTERN: ', file_pattern)
    matching_files = glob.glob(file_pattern)
    first_edge_weight_dflist = []
    for file_name in matching_files:
        first_edge_weight_df = pd.read_csv(file_name)
        first_edge_weight_dflist.append(first_edge_weight_df)
    # Average the edge weights on the 'weight' column
    concat_first_edge_weight_df = pd.concat(first_edge_weight_dflist)
    concat_first_edge_weight_df = concat_first_edge_weight_df.drop(columns=['Patient_Index'])
    avg_first_edge_weight_df = concat_first_edge_weight_df.groupby(['Actual_From', 'Actual_To'], as_index=False).mean()
    print('AVERAGE FIRST EDGE WEIGHT SHAPE: ', avg_first_edge_weight_df.shape)
    print('AVERAGE FIRST EDGE WEIGHT: ', avg_first_edge_weight_df)
    avg_first_edge_weight_df.to_csv(folder_path + '/average_first_edge_weight.csv', index=False)

    ### Collect files name with 'gene_block_edge_weight' under folder [analysis/gigtransformer/]
    # Define the folder path
    folder_path = 'analysis/gigtransformer/fold_' + str(fold_n)
    # Use glob to find files containing 'gene_block_edge_weight' in their names
    file_pattern = os.path.join(folder_path, '*gene_block_edge_weight*')
    print('FILE PATTERN: ', file_pattern)
    matching_files = glob.glob(file_pattern)
    block_edge_weight_dflist = []
    for file_name in matching_files:
        block_edge_weight_df = pd.read_csv(file_name)
        block_edge_weight_dflist.append(block_edge_weight_df)
    # Average the edge weights on the 'weight' column
    concat_block_edge_weight_df = pd.concat(block_edge_weight_dflist)
    concat_block_edge_weight_df = concat_block_edge_weight_df.drop(columns=['Patient_Index'])
    avg_block_edge_weight_df = concat_block_edge_weight_df.groupby(['Actual_From', 'Actual_To'], as_index=False).mean()
    print('AVERAGE BLOCK EDGE WEIGHT SHAPE: ', avg_block_edge_weight_df.shape)
    print('AVERAGE BLOCK EDGE WEIGHT: ', avg_block_edge_weight_df)
    avg_block_edge_weight_df.to_csv(folder_path + '/average_block_edge_weight.csv', index=False)

    ### Collect files name with 'gene_last_edge_weight' under folder [analysis/gigtransformer/]
    # Define the folder path
    folder_path = 'analysis/gigtransformer/fold_' + str(fold_n)
    # Use glob to find files containing 'gene_last_edge_weight' in their names
    file_pattern = os.path.join(folder_path, '*gene_last_edge_weight*')
    print('FILE PATTERN: ', file_pattern)
    matching_files = glob.glob(file_pattern)
    last_edge_weight_dflist = []
    for file_name in matching_files:
        last_edge_weight_df = pd.read_csv(file_name)
        last_edge_weight_dflist.append(last_edge_weight_df)
    # Average the edge weights on the 'weight' column
    concat_last_edge_weight_df = pd.concat(last_edge_weight_dflist)
    concat_last_edge_weight_df = concat_last_edge_weight_df.drop(columns=['Patient_Index'])
    avg_last_edge_weight_df = concat_last_edge_weight_df.groupby(['Actual_From', 'Actual_To'], as_index=False).mean()
    print('AVERAGE LAST EDGE WEIGHT SHAPE: ', avg_last_edge_weight_df.shape)
    print('AVERAGE LAST EDGE WEIGHT: ', avg_last_edge_weight_df)
    avg_last_edge_weight_df.to_csv(folder_path + '/average_last_edge_weight.csv', index=False)
    return avg_first_edge_weight_df, avg_block_edge_weight_df, avg_last_edge_weight_df

if __name__ == "__main__":
    ### Prepare the hyperparameters
    prog_args = arg_parse() # Parse argument from terminal or default parameters
    device, prog_args.gpu_ids = utils.get_available_devices() # Check and allocate resources
    device = torch.device('cuda:0') # Manual set
    torch.cuda.set_device(device)
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('MAIN DEVICE: ', device)

    ### Run the model analysis
    k = 5
    for fold_n in range(1, k + 1):
        analysis_model(fold_n, prog_args, device)

    ### Average the edge weights on gene graph on each fold
    k = 5
    for fold_n in range(1, k + 1):
        avg_first_edge_weight_df, avg_block_edge_weight_df, avg_last_edge_weight = average_sample_gene(fold_n)
