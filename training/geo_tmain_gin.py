import os
import pdb
import torch
from datetime import datetime
import argparse
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
from enc.geo_gin import GIN

# Parse arguments from command line
def arg_parse():
    parser = argparse.ArgumentParser(description='GEO-WEBGNN ARGUMENTS.')
    # Set default input argument
    parser.set_defaults(cuda = '0',
                        parallel = False,
                        unchanged_threshold = 100,
                        change_wave = 0.75,
                        add_self = '0', # 'add'
                        adj = '0', # 'sym'
                        model = '0', # 'load'
                        lr = 0.01,
                        # lr = 0.0075,
                        clip = 5.0,
                        num_epochs = 500,
                        num_workers = 0,
                        input_dim = 8382,
                        hidden_dim = 48,
                        output_dim = 48,
                        num_classes = 3,
                        dropout = 0.01)
    return parser.parse_args()


def build_geogin_model(args, device):
    print('--- BUILDING UP GNN MODEL ... ---')
    # Get parameters
    model = GIN(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                embedding_dim=args.output_dim)
    model = model.to(device)
    return model


def train_geogin_model(data, model, device, args, optimizer, scheduler):
    loss = 0
    optimizer.zero_grad()
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(x=x, edge_index=edge_index, node_label=node_label, node_index=node_index)
    loss = model.loss(node_output, node_label)
    loss.backward()
    loss = loss.item()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    torch.cuda.empty_cache()
    return model, loss, x_embed, node_output, ypred, y_nodepred


def train_geogin(args, fold_n, nth_training_fold_num, device):
    # Build [WeightBiGNN, Decoder] model
    model = build_geogin_model(args, device)

    # Train model on training dataset
    # graph_all_feature = np.load('./data/post_data/all_x.npy', allow_pickle=True)
    graph_all_feature = np.load('./data/post_data/norm_all_x.npy', allow_pickle=True)
    graph_all_feature = graph_all_feature.astype(np.float32)  # or np.float64 depending on your data
    edge_index = np.load('./data/post_data/edge_index.npy')
    node_label = np.load('./data/post_data/train_label_' + str(fold_n)  + '.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load('./data/post_data/train_idx_' + str(fold_n)  + '.npy')
    # import pdb; pdb.set_trace()

    # Other parameters
    epoch_num = args.num_epochs
    # Record epoch loss and pearson correlation
    max_test_acc = 0
    max_test_acc_loss = 0
    max_test_training_acc = 0
    max_test_acc_training_loss = 0
    best_test_id = 0

    unchanged_count = 0

    # Clean result previous epoch_i_pred files
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    run_tag = f'Ep{epoch_num}_LR{args.lr:.0e}_{timestamp}'
    folder_name = 'gin/' + run_tag + '_fold' + str(fold_n)
    unit = nth_training_fold_num
    path = './gnn_result/%s-%d' % (folder_name, unit)
    while os.path.exists('./gnn_result') == False:
        os.mkdir('./gnn_result')
    while os.path.exists('./gnn_result/gin') == False:
        os.mkdir('./gnn_result/gin')
    while os.path.exists(path):
        unit += 1
        path = './gnn_result/%s-%d' % (folder_name, unit)
    os.mkdir(path)

    # Get optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.8,0.999], eps=1e-7, weight_decay=1e-10)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 200, 300, 500, 750], gamma=0.9)

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
        geo_data = read_geodata(graph_all_feature, edge_index, node_label, node_idx)
        print('TRAINING MODEL...')
        model, training_loss, x_embed, node_output, ypred, y_nodepred = train_geogin_model(geo_data, model, device, args, optimizer, scheduler)
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
        test_accuracy, test_confusion_matrix, test_label_df, test_loss = test_geogin(fold_n, model, device)
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


def test_geogin_model(data, model, device):
    loss = 0
    x = Variable(data.x, requires_grad=False).to(device)
    edge_index = Variable(data.edge_index, requires_grad=False).to(device)
    node_label = Variable(data.node_label, requires_grad=False).to(device)
    node_index = Variable(data.node_index, requires_grad=False).to(device)
    x_embed, node_output, ypred, y_nodepred = model(x=x, edge_index=edge_index, node_label=node_label, node_index=node_index)
    loss = model.loss(node_output, node_label)
    loss.backward()
    loss = loss.item()
    return model, loss, x_embed, node_output, ypred, y_nodepred


def test_geogin(fold_n, model, device):
    # Test model on test dataset
    # graph_all_feature = np.load('./data/post_data/all_x.npy', allow_pickle=True)
    graph_all_feature = np.load('./data/post_data/norm_all_x.npy', allow_pickle=True)
    graph_all_feature = graph_all_feature.astype(np.float32)  # or np.float64 depending on your data
    edge_index = np.load('./data/post_data/edge_index.npy')
    node_label = np.load('./data/post_data/test_label_' + str(fold_n)  + '.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load('./data/post_data/test_idx_' + str(fold_n)  + '.npy')

    # Run test model
    model.eval()
    geo_data = read_geodata(graph_all_feature, edge_index, node_label, node_idx)
    print('TEST MODEL...')
    model, test_loss, x_embed, node_output, ypred, y_nodepred = test_geogin_model(geo_data, model, device)
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

def run_model(k, fold_n, nth_training_fold_num):
    ### Prepare the hyperparameters
    prog_args = arg_parse() # Parse argument from terminal or default parameters
    device, prog_args.gpu_ids = utils.get_available_devices() # Check and allocate resources
    device = torch.device('cuda:0') # Manual set
    torch.cuda.set_device(device)
    prog_args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('MAIN DEVICE: ', device)

    ### Train the model
    max_test_acc = train_geogin(prog_args, fold_n, nth_training_fold_num, device)
    return max_test_acc, prog_args


if __name__ == "__main__":
    k = 5
    fold_num_train = 5
    # Set fold number
    for fold_n in range(1, k + 1):
        # Record the best performance on each fold
        fold_n_max_test_acc = 0
        fold_n_max_unit = 0
        for nth_training_fold_num in range(1, fold_num_train + 1):
            # Run the model
            max_test_acc, prog_args = run_model(k, fold_n, nth_training_fold_num)
            if max_test_acc > fold_n_max_test_acc:
                fold_n_max_test_acc = max_test_acc
                fold_n_max_unit = nth_training_fold_num