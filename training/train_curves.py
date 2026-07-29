#!/usr/bin/env python3
"""
手动调参 + Loss 曲线可视化
==========================
用法:
  # 默认参数 50 epochs, fold 1
  python train_curves.py

  # 指定 lr 和 epochs
  python train_curves.py --lr 0.005 --epochs 50

  # 跑多个 fold
  python train_curves.py --lr 0.01 --epochs 50 --folds 1 2 3

  # 调其他参数
  python train_curves.py --lr 0.005 --dropout 0.15 --weight_decay 1e-4 --batch_size 64

  # 不画图只看数字
  python train_curves.py --no_plot
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from geo_loader.geo_readgraph import read_geodata
from enc.geo_gigtransformer import GIG_Transformer


def build_model(args, num_gene_node, device):
    model = GIG_Transformer(
        gene_input_dim=args.gene_input_dim,
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
        graph_opt=args.graph_opt,
    )
    return model.to(device)


def load_data():
    """加载数据（只加载一次）"""
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]

    gene_feature = np.load('./data/post_data/norm_gene_x.npy', allow_pickle=True).astype(np.float32)
    gene_edge_index = np.load('./data/post_data/gene_edge_index.npy', allow_pickle=True).astype(np.int64)
    gene_edge_index = torch.from_numpy(gene_edge_index).long()

    subfeature_dict_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
    num_subfeature = subfeature_dict_df.shape[0]
    subject_dict_df = pd.read_csv('./data/filtered_data/subject_dict_df.csv')
    num_subject = subject_dict_df.shape[0]

    graph_feature = np.load('./data/post_data/norm_x.npy')
    edge_index = np.load('./data/post_data/edge_index.npy')

    return {
        'num_gene_node': num_gene_node,
        'gene_feature': gene_feature,
        'gene_edge_index': gene_edge_index,
        'num_subfeature': num_subfeature,
        'num_subject': num_subject,
        'graph_feature': graph_feature,
        'edge_index': edge_index,
        'num_feature': 6,
    }


def train_one_epoch(model, data, geo_data, device, args, optimizer, scheduler):
    """训练一个 epoch，返回 loss 和 accuracy"""
    model.train()
    optimizer.zero_grad()

    x = Variable(geo_data.x, requires_grad=False).to(device)
    edge_index = Variable(geo_data.edge_index, requires_grad=False).to(device)
    node_label = Variable(geo_data.node_label, requires_grad=False).to(device)
    node_index = Variable(geo_data.node_index, requires_grad=False).to(device)

    x_embed, node_output, ypred, y_nodepred = model(
        num_feature=data['num_feature'], num_subfeature=data['num_subfeature'],
        num_subject=data['num_subject'], num_gene_node=data['num_gene_node'],
        gene_feature=data['gene_feature'], gene_edge_index=data['gene_edge_index'],
        x=x, edge_index=edge_index,
        node_label=node_label, node_index=node_index,
        args=args, device=device,
    )

    loss = model.loss(node_output, node_label, data['gene_edge_index'], args.gene_num_top_feature)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    scheduler.step()

    loss_val = loss.item()
    y_pred = y_nodepred.cpu().detach().numpy()
    return loss_val, y_pred


def evaluate(model, data, fold_n, device, args):
    """测试，返回 loss 和 accuracy"""
    model.eval()

    node_label = np.load(f'./data/post_data/test_label_{fold_n}.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load(f'./data/post_data/test_idx_{fold_n}.npy')

    geo_data = read_geodata(data['graph_feature'], data['edge_index'], node_label, node_idx)

    with torch.no_grad():
        x = Variable(geo_data.x, requires_grad=False).to(device)
        edge_index_t = Variable(geo_data.edge_index, requires_grad=False).to(device)
        node_label_t = Variable(geo_data.node_label, requires_grad=False).to(device)
        node_index = Variable(geo_data.node_index, requires_grad=False).to(device)

        x_embed, node_output, ypred, y_nodepred = model(
            num_feature=data['num_feature'], num_subfeature=data['num_subfeature'],
            num_subject=data['num_subject'], num_gene_node=data['num_gene_node'],
            gene_feature=data['gene_feature'], gene_edge_index=data['gene_edge_index'],
            x=x, edge_index=edge_index_t,
            node_label=node_label_t, node_index=node_index,
            args=args, device=device,
        )

        loss = model.loss(node_output, node_label_t, data['gene_edge_index'], args.gene_num_top_feature)

    y_pred = y_nodepred.cpu().detach().numpy()
    test_acc = float((y_pred == node_label_indices).sum()) / len(node_label_indices)
    return loss.item(), test_acc, y_pred, node_label_indices


def train_with_curves(args, fold_n, data, device):
    """训练并记录每个 epoch 的 loss/acc 曲线"""
    # 加载训练数据
    node_label = np.load(f'./data/post_data/train_label_{fold_n}.npy')
    node_label_indices = np.argmax(node_label, axis=1)
    node_idx = np.load(f'./data/post_data/train_idx_{fold_n}.npy')

    # 建模
    model = build_model(args, data['num_gene_node'], device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.8, 0.999], eps=1e-7, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # 记录
    history = {
        'epoch': [], 'lr': [],
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
    }

    best_test_acc = 0
    best_epoch = 0
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"Fold {fold_n} | lr={args.lr} | epochs={args.num_epochs} | dropout={args.dropout}")
    print(f"{'='*65}")
    print(f"{'Epoch':>5} {'LR':>10} {'TrainLoss':>10} {'TestLoss':>10} {'TrainAcc':>9} {'TestAcc':>9} {'Best':>9}")
    print(f"{'-'*65}")

    for epoch in range(1, args.num_epochs + 1):
        # Train
        geo_data = read_geodata(data['graph_feature'], data['edge_index'], node_label, node_idx)
        train_loss, train_pred = train_one_epoch(model, data, geo_data, device, args, optimizer, scheduler)
        train_acc = float((train_pred == node_label_indices).sum()) / len(node_label_indices)

        # Test
        test_loss, test_acc, _, _ = evaluate(model, data, fold_n, device, args)

        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch)
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # 更新 best
        if test_acc > best_test_acc and test_acc <= train_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        # 每 5 个 epoch 或最后一个 epoch 打印
        if epoch % 5 == 0 or epoch == 1 or epoch == args.num_epochs:
            print(f"{epoch:>5} {current_lr:>10.6f} {train_loss:>10.4f} {test_loss:>10.4f} "
                  f"{train_acc:>9.4f} {test_acc:>9.4f} {best_test_acc:>9.4f}{marker}")

        torch.cuda.empty_cache()

    print(f"\nFold {fold_n} 结果: best_test_acc={best_test_acc:.4f} @ epoch {best_epoch}")
    return history, best_test_acc, best_epoch


def plot_curves(all_histories, args, save_path='loss_curves.png'):
    """画 loss 和 accuracy 曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过画图")
        return

    n_folds = len(all_histories)
    fig, axes = plt.subplots(2, n_folds, figsize=(6 * n_folds, 8), squeeze=False)

    for i, (fold_n, hist) in enumerate(all_histories.items()):
        epochs = hist['epoch']

        # Loss curves
        ax = axes[0][i]
        ax.plot(epochs, hist['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        ax.plot(epochs, hist['test_loss'], 'r-', label='Test Loss', alpha=0.7)
        ax.set_title(f'Fold {fold_n} - Loss (lr={args.lr})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[1][i]
        ax.plot(epochs, hist['train_acc'], 'b-', label='Train Acc', alpha=0.7)
        ax.plot(epochs, hist['test_acc'], 'r-', label='Test Acc', alpha=0.7)
        ax.set_title(f'Fold {fold_n} - Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n曲线已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='手动调参 + Loss 曲线可视化')

    # 核心调参参数
    parser.add_argument('--lr', type=float, default=0.01, help='学习率 (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数 (default: 50)')
    parser.add_argument('--folds', type=int, nargs='+', default=[1], help='要跑的 fold (default: [1])')

    # 常调的参数
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--class_weight', type=float, default=0.9)

    # 网络结构参数
    parser.add_argument('--gene_hidden_dim', type=int, default=18)
    parser.add_argument('--gene_num_top_feature', type=int, default=18)
    parser.add_argument('--gig_input_transform_dim', type=int, default=18)
    parser.add_argument('--gig_hidden_dim', type=int, default=18)

    # 损失函数权重
    parser.add_argument('--ortho_weight', type=float, default=0.05)
    parser.add_argument('--link_weight', type=float, default=0.05)

    # 其他
    parser.add_argument('--no_plot', action='store_true', help='不画图')
    parser.add_argument('--save', type=str, default='loss_curves.png', help='图片保存路径')
    parser.add_argument('--milestone_step', type=int, default=100, help='milestone 间隔')

    cli_args = parser.parse_args()

    # 构建 model args
    milestones = list(range(cli_args.milestone_step, cli_args.epochs + 1, cli_args.milestone_step))
    if not milestones:
        milestones = [cli_args.epochs // 2] if cli_args.epochs > 1 else [1]

    args = argparse.Namespace(
        cuda='0', parallel=False, add_self='0', adj='0', model='0',
        lr=cli_args.lr, weight_decay=cli_args.weight_decay,
        milestones=milestones,
        gamma=cli_args.gamma, clip=cli_args.clip, batch_size=cli_args.batch_size,
        num_epochs=cli_args.epochs,
        unchanged_threshold=9999,  # 不使用 early stopping，跑完所有 epoch
        change_wave=0.75, num_workers=0,
        graph_opt='GinG',
        gene_input_dim=6,
        gene_hidden_dim=cli_args.gene_hidden_dim,
        gene_output_dim=cli_args.gene_hidden_dim,  # output = hidden
        gene_num_top_feature=cli_args.gene_num_top_feature,
        gig_input_dim=42,
        gig_input_transform_dim=cli_args.gig_input_transform_dim,
        gig_hidden_dim=cli_args.gig_hidden_dim,
        gig_output_dim=cli_args.gig_hidden_dim,  # output = hidden
        class_weight_fine=0.5,
        class_weight=cli_args.class_weight,
        ortho_weight=cli_args.ortho_weight,
        link_weight=cli_args.link_weight,
        ent_weight=0.0,
        num_classes=3,
        gene_num_head=1,
        gig_num_head=1,
        dropout=cli_args.dropout,
        gpu_ids=[0],
    )

    # 打印参数摘要
    gig_first_input = args.gig_input_transform_dim + args.gene_num_top_feature * args.gene_output_dim
    print(f"架构: gene[6→{args.gene_hidden_dim}→{args.gene_output_dim}] "
          f"K={args.gene_num_top_feature} "
          f"gig[{gig_first_input}→{args.gig_hidden_dim}→{args.gig_output_dim}]")
    print(f"参数: lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}, "
          f"batch={args.batch_size}, gamma={args.gamma}, clip={args.clip}")
    print(f"Loss权重: class={args.class_weight}, ortho={args.ortho_weight}, link={args.link_weight}")

    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    print("加载数据...")
    data = load_data()

    # 训练
    all_histories = {}
    all_results = []

    for fold_n in cli_args.folds:
        history, best_acc, best_epoch = train_with_curves(args, fold_n, data, device)
        all_histories[fold_n] = history
        all_results.append((fold_n, best_acc, best_epoch))
        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*50}")
    print("汇总:")
    for fold_n, best_acc, best_epoch in all_results:
        print(f"  Fold {fold_n}: best_test_acc = {best_acc:.4f} @ epoch {best_epoch}")
    if len(all_results) > 1:
        mean_acc = np.mean([r[1] for r in all_results])
        print(f"  平均: {mean_acc:.4f}")
    print(f"{'='*50}")

    # 画图
    if not cli_args.no_plot:
        plot_curves(all_histories, args, save_path=cli_args.save)


if __name__ == '__main__':
    main()
