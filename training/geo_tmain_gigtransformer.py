import os
import pdb
import torch
import shutil
import argparse
from datetime import datetime
import tensorboardX
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from geo_loader.geo_readgraph import read_geodata
from enc.geo_gigtransformer import GIG_Transformer


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


def train_geogig_model(data, num_feature, num_subfeature, num_subject, num_gene_node,
                    gene_feature, gene_edge_index,
                    model, device, args, optimizer, scheduler):
    loss = 0
    optimizer.zero_grad()
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(num_feature=num_feature, num_subfeature=num_subfeature, 
                                                    num_subject=num_subject, num_gene_node=num_gene_node, 
                                                    gene_feature=gene_feature, gene_edge_index=gene_edge_index,
                                                    x=x, edge_index=edge_index, 
                                                    node_label=node_label, node_index=node_index,
                                                    args=args, device=device)
    loss = model.loss(node_output, node_label, gene_edge_index, args.gene_num_top_feature)
    loss.backward()
    loss = loss.item()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    torch.cuda.empty_cache()
    return model, loss, x_embed, node_output, ypred, y_nodepred


def train_geogig(args, fold_n, nth_training_fold_num, device, run_path):
    # Train model on training dataset with [gene features]
    num_feature = 6
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]
    # gene_feature = np.load('./data/post_data/gene_x.npy', allow_pickle=True)
    gene_feature = np.load('./data/post_data/norm_gene_x.npy', allow_pickle=True)
    gene_feature = gene_feature.astype(np.float32)  # or np.float64 depending on your data
    gene_edge_index = np.load('./data/post_data/gene_edge_index.npy', allow_pickle=True)
    gene_edge_index = gene_edge_index.astype(np.int64)  # or np.float64 depending on your data
    gene_edge_index = torch.from_numpy(gene_edge_index).long()
    
    # Train model on training dataset with [subject features]
    subfeature_dict_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
    num_subfeature = subfeature_dict_df.shape[0]
    subject_dict_df = pd.read_csv('./data/filtered_data/subject_dict_df.csv')
    num_subject = subject_dict_df.shape[0]
    # graph_feature = np.load('./data/post_data/x.npy')
    graph_feature = np.load('./data/post_data/norm_x.npy')
    edge_index = np.load('./data/post_data/edge_index.npy')
    node_label = np.load('./data/post_data/train_label_' + str(fold_n)  + '.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load('./data/post_data/train_idx_' + str(fold_n)  + '.npy')

    # Build [Graph in Graph] model
    model = build_geogig_model(args, num_gene_node, device)

    # Other parameters
    epoch_num = args.num_epochs
    # Record epoch loss and pearson correlation
    max_test_acc = 0
    max_test_acc_loss = 0
    max_test_training_acc = 0
    max_test_acc_training_loss = 0
    best_test_id = 0

    unchanged_count = 0

    # Create fold-specific result directory
    import yaml
    path = run_path + '/fold' + str(fold_n)
    os.makedirs(path, exist_ok=True)

    # Save all hyperparameters to config.yaml
    config_snapshot = {
        'model': {
            'type': 'gigtransformer',
            'graph_opt': args.graph_opt,
            'gene_input_dim': args.gene_input_dim,
            'gene_hidden_dim': args.gene_hidden_dim,
            'gene_output_dim': args.gene_output_dim,
            'gene_num_top_feature': args.gene_num_top_feature,
            'gig_input_dim': args.gig_input_dim,
            'gig_input_transform_dim': args.gig_input_transform_dim,
            'gig_hidden_dim': args.gig_hidden_dim,
            'gig_output_dim': args.gig_output_dim,
            'num_classes': args.num_classes,
            'gene_num_head': args.gene_num_head,
            'gig_num_head': args.gig_num_head,
            'dropout': args.dropout,
        },
        'training': {
            'epochs': epoch_num,
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'batch_size': args.batch_size,
            'milestones': args.milestones,
            'gamma': args.gamma,
            'clip': args.clip,
            'unchanged_threshold': args.unchanged_threshold,
            'change_wave': args.change_wave,
        },
        'loss_weights': {
            'class_weight_fine': float(args.class_weight_fine),
            'class_weight': float(args.class_weight),
            'ortho_weight': float(args.ortho_weight),
            'link_weight': float(args.link_weight),
            'ent_weight': float(args.ent_weight),
        },
        'experiment': {
            'fold': fold_n,
            'run_path': run_path,
        }
    }
    with open(path + '/config.yaml', 'w') as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.8,0.999], eps=1e-7, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    for i in range(1, epoch_num + 1):
        if unchanged_count > args.unchanged_threshold and training_loss > max_test_acc_training_loss * args.change_wave:
            break
        elif unchanged_count > args.unchanged_threshold and training_loss <= max_test_acc_training_loss * args.change_wave:
            unchanged_count = unchanged_count - 50
        print('-----------------------------------')
        print('------------- EPOCH: ' + str(i) + ' -------------')
        print('-----------------------------------')
        model.train()
        learning_rate = 0.005
        geo_data = read_geodata(graph_feature, edge_index, node_label, node_idx)
        print('TRAINING MODEL...')
        # import pdb; pdb.set_trace()
        model, training_loss, x_embed, node_output, ypred, y_nodepred = train_geogig_model(geo_data, 
                                                                num_feature, num_subfeature, num_subject, num_gene_node,
                                                                gene_feature, gene_edge_index,
                                                                model, device, args, optimizer, scheduler)
        print('TRAIN LOSS: ', training_loss)
        y_nodepred = y_nodepred.cpu().detach().numpy()
        train_correct_count = (y_nodepred == node_label_indices).sum()
        train_accuracy = float(train_correct_count) / len(node_label_indices)
        print('TRAIN CORRECT: ', train_correct_count)
        print('TRAIN ACCURACY: ', train_accuracy)
        # Compute confusion matrix
        train_confusion_matrix = confusion_matrix(node_label_indices, y_nodepred)
        print(train_confusion_matrix)
        # Create [train_label_df] for saving preparation
        train_label_df = pd.DataFrame({'train_node_idx': list(node_idx),
                                    'train_label': list(node_label_indices),
                                    'train_pred_label': list(y_nodepred)})
        # Save best test model with trained model
        test_accuracy, test_confusion_matrix, test_label_df, test_loss = test_geogig(fold_n, model, device, args)
        if test_accuracy > max_test_acc and test_accuracy <= train_accuracy:
            max_test_acc_training_loss = training_loss
            max_test_acc = test_accuracy
            max_test_acc_loss = test_loss
            max_test_training_acc = train_accuracy
            best_test_id = i
            torch.save(model.state_dict(), path + '/best_train_model.pth')
            train_label_df.to_csv(path + '/result_train_label_' + str(fold_n) + '.csv', index=False, header=True)
            test_label_df.to_csv(path + '/result_test_label_' + str(fold_n) + '.csv', index=False, header=True)
            unchanged_count = 0
        else:
            unchanged_count += 1
        print('--------------------------------------------- BEST TRAIN/TEST ID: ', best_test_id)
        print('--------------------------------------------- BEST TEST TRAINING LOSS: ', max_test_acc_training_loss)
        print('--------------------------------------------- BEST TEST TRAINING ACCURACY: ', max_test_training_acc)
        print('--------------------------------------------- BEST TEST LOSS: ', max_test_acc_loss)
        print('--------------------------------------------- BEST TEST ACCURACY: ', max_test_acc)
        write_best_model_info(fold_n, path, best_test_id, max_test_acc_training_loss, max_test_training_acc, max_test_acc_loss, max_test_acc)

    return max_test_acc


def test_geogig_model(data, num_feature, num_subfeature, num_subject, num_gene_node,
                        gene_feature, gene_edge_index, model, device, args):
    loss = 0
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(num_feature=num_feature, num_subfeature=num_subfeature, 
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
    # import pdb; pdb.set_trace()
    print('TEST MODEL ...')
    model, test_loss, x_embed, node_output, ypred, y_nodepred = test_geogig_model(geo_data, 
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

def write_best_model_info(fold_n, path, max_test_acc_id, max_test_acc_training_loss, max_test_training_acc, max_test_acc_loss, max_test_acc):
    best_model_info = (
        f'\n-------------Fold: {fold_n} -------------\n'
        f'\n-------------BEST TEST ACCURACY MODEL ID INFO: {max_test_acc_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {max_test_acc_training_loss}\n'
        f'BEST MODEL TRAIN ACCURACY: {max_test_training_acc}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {max_test_acc_loss}\n'
        f'BEST MODEL TEST ACCURACY: {max_test_acc}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)

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
                        # lr = 0.075,
                        weight_decay = 0.00018042132714427627,
                        milestones = [150, 300, 450],
                        gamma = 0.95,
                        clip = 2.0,
                        batch_size = 256,
                        num_epochs = 500,
                        unchanged_threshold = 100,
                        change_wave = 0.75,
                        num_workers = 0,
                        # graph_opt = 'gene',
                        # graph_opt = 'subject',
                        graph_opt = 'GinG',
                        gene_input_dim = 6, # gene embedding parameters
                        gene_hidden_dim = 9,
                        gene_output_dim = 9,
                        gene_num_top_feature = 24,
                        gig_input_dim = 42, # gig embedding parameters
                        gig_input_transform_dim = 24,
                        gig_hidden_dim = 32,
                        gig_output_dim = 24,
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


FOLD_PARAM_OVERRIDES = {
    # gig_hidden_dim=32 已设为所有fold的默认值
}

def run_model(k, fold_n, nth_training_fold_num, run_path):
    ### Prepare the hyperparameters
    prog_args = arg_parse() # Parse argument from terminal or default parameters

    # Apply per-fold parameter overrides if any
    if fold_n in FOLD_PARAM_OVERRIDES:
        overrides = FOLD_PARAM_OVERRIDES[fold_n]
        print(f'[Fold {fold_n}] Applying parameter overrides: {overrides}')
        for key, val in overrides.items():
            setattr(prog_args, key, val)

    device, prog_args.gpu_ids = utils.get_available_devices() # Check and allocate resources
    device = torch.device('cuda:0') # Manual set
    torch.cuda.set_device(device)
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('MAIN DEVICE: ', device)

    ### Train the model
    max_test_acc = train_geogig(prog_args, fold_n, nth_training_fold_num, device, run_path)
    return max_test_acc, prog_args


if __name__ == "__main__":
    k = 5
    fold_num_train = 1

    # Generate run_tag and create parent result directory once
    prog_args_init = arg_parse()
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
    lr_str = f'{prog_args_init.lr:.6f}'.replace('.', '')
    epoch_num = prog_args_init.num_epochs
    run_tag = f'Epoch-{epoch_num}_BS-{prog_args_init.batch_size}_LR-{lr_str}_{timestamp}'
    run_path = './gnn_result/gigtransformer_' + prog_args_init.graph_opt + '/' + run_tag
    os.makedirs(run_path, exist_ok=True)
    if os.listdir(run_path):
        suffix = 2
        while os.path.exists(run_path + f'_{suffix}'):
            suffix += 1
        run_path = run_path + f'_{suffix}'
        os.makedirs(run_path, exist_ok=True)

    # Set fold number
    for fold_n in range(1, k+1):
        # Record the best performance on each fold
        fold_n_max_test_acc = 0
        fold_n_max_unit = 0
        for nth_training_fold_num in range(1, fold_num_train + 1):
            # Run the model
            max_test_acc, prog_args = run_model(k, fold_n, nth_training_fold_num, run_path)
            if max_test_acc > fold_n_max_test_acc:
                fold_n_max_test_acc = max_test_acc
                fold_n_max_unit = nth_training_fold_num