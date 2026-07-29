"""
verify_gene_attention.py
========================
直接从训练好的模型提取基因层 attention weights，逐患者分析，
绕过所有现有分析代码（edge_analysis_transformer.py、message() CSV保存等），
严谨判断模型在基因层面是否学到了患者类型差异。

用法:
    python verify_gene_attention.py --fold 1        # 只跑第1折
    python verify_gene_attention.py --fold all       # 跑全部5折
    python verify_gene_attention.py --fold all --skip_extract  # 跳过提取，直接统计

输出目录: verify_output/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复 Windows 下 OpenMP 双重加载冲突
import sys
import io
import json
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from collections import defaultdict

# ============================================================
# 修复 Windows 终端中文编码问题
# ============================================================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================
# 导入模型定义（只用于构建模型结构，不使用其分析逻辑）
# ============================================================
from enc.geo_gigtransformer_analysis import GIG_Transformer


# ============================================================
# Phase 0: 模型构建与数据加载
# ============================================================

def build_model(num_gene_node, device):
    """
    用指定的参数构建 GIG_Transformer 模型。
    这些参数严格对应 geo_analysis_tran.py arg_parse() 中的默认值，
    即产出 80.20%±0.0165 结果的那组参数。
    """
    model = GIG_Transformer(
        gene_input_dim=6,           # gene embedding: 输入维度
        gene_hidden_dim=18,         # gene embedding: 隐藏层维度（原来是9）
        gene_embedding_dim=18,      # gene_output_dim（原来是9）
        gene_num_top_feature=18,    # 顶部基因特征数（原来是24）
        num_gene_node=num_gene_node,
        gig_input_dim=42,           # GiG embedding: 输入维度
        gig_input_transform_dim=18, # GiG 输入变换维度（原来是24）
        gig_hidden_dim=18,          # GiG 隐藏层维度（原来是24）
        gig_embedding_dim=18,       # gig_output_dim（原来是24）
        num_classes=3,
        gene_num_head=1,
        gig_num_head=1,
        class_weight_fine=0.5,      # 损失函数参数（原来是0.7746）
        class_weight=0.9,           # （原来是0.8209）
        ortho_weight=0.05,          # （原来是0.00796）
        link_weight=0.05,           # （原来是0.00219）
        ent_weight=0.00,            # （原来是0.09039）
        graph_opt='GinG'
    )
    model = model.to(device)
    return model


def load_data():
    """
    加载所有需要的数据文件。
    返回: gene_feature, gene_edge_index, label_df, num_gene_node
    """
    # 基因节点数量
    gene_num_dict_df = pd.read_csv('./data/filtered_data/gene_num_dict_df.csv')
    num_gene_node = gene_num_dict_df.shape[0]  # 1390
    print(f'[数据] 基因节点数: {num_gene_node}')

    # 基因特征: shape (813, 8340) = 813患者 × (1390节点 × 6特征)
    # 使用 norm_gene_x.npy（与 geo_analysis_tran.py 一致）
    gene_feature = np.load('./data/post_data/norm_gene_x.npy', allow_pickle=True)
    gene_feature = gene_feature.astype(np.float32)
    print(f'[数据] 基因特征 shape: {gene_feature.shape}, dtype: {gene_feature.dtype}')

    # 基因边索引: shape (2, num_edges)
    gene_edge_index = np.load('./data/post_data/gene_edge_index.npy', allow_pickle=True)
    gene_edge_index = gene_edge_index.astype(np.int64)
    gene_edge_index = torch.from_numpy(gene_edge_index).long()
    print(f'[数据] 基因边索引 shape: {gene_edge_index.shape}')

    # 患者标签: 包含 t2ds, pret2ds, no_t2ds 列
    label_df = pd.read_csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
    n_t2ds = label_df['t2ds'].sum()
    n_pret2ds = label_df['pret2ds'].sum()
    n_no_t2ds = label_df['no_t2ds'].sum()
    print(f'[数据] 患者类型分布: t2ds={n_t2ds}, pret2ds={n_pret2ds}, no_t2ds={n_no_t2ds}')

    return gene_feature, gene_edge_index, label_df, num_gene_node


# ============================================================
# Phase 1: 逐患者提取 attention weights
# ============================================================

def extract_patient_attentions(fold_n, gene_feature, gene_edge_index,
                                num_gene_node, device):
    """
    对指定 fold，加载训练好的模型，逐个患者提取基因层 attention weights。

    核心策略:
    - 一次只处理一个患者（不用 Batch.from_data_list）
    - 传 index=None 跳过 message() 中的 CSV 保存逻辑
    - 用 return_attention_weights=True 直接获取 (edge_index, alpha)

    返回: dict[patient_idx] = {
        'first_alpha': np.array,   # 第一层 attention weights
        'block_alpha': np.array,   # 中间层 attention weights
        'last_alpha':  np.array    # 最后层 attention weights
    }
    """
    # 加载模型（路径与 geo_analysis_tran.py 的 analysis_model() 一致）
    model_path = f'./gnn_result/gigtransformer/5-fold/epoch_500_fold{fold_n}/best_train_model.pth'
    print(f'\n[Phase 1] 加载模型: {model_path}')

    if not os.path.exists(model_path):
        print(f'  [错误] 模型文件不存在: {model_path}')
        return None

    model = build_model(num_gene_node, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'  模型加载成功')

    num_patients = gene_feature.shape[0]
    num_features = 6
    edge_index_device = gene_edge_index.to(device)

    # 存储所有患者的 attention weights
    patient_alphas = {}

    print(f'  开始逐患者提取 attention (共 {num_patients} 人) ...')
    with torch.no_grad():
        for patient_i in range(num_patients):
            # 取出该患者的基因特征，reshape 为 (num_gene_node, num_features)
            patient_gene_x = gene_feature[patient_i].reshape(num_gene_node, num_features)
            x = torch.from_numpy(patient_gene_x).float().to(device)

            # ---- 第一层: gene_conv_first ----
            # index=None → message() 中 self.index is None → 跳过CSV保存
            # return_attention_weights=True → 返回 (out, (edge_index, alpha))
            out1, (_, alpha1) = model.gene_conv_first(
                x, edge_index_device,
                index=None, upper_index=None, batch_number=None,
                fold_n=fold_n, layer='gene_first',
                return_attention_weights=True
            )
            out1 = model.act(out1)

            # ---- 中间层: gene_conv_block ----
            out2, (_, alpha2) = model.gene_conv_block(
                out1, edge_index_device,
                index=None, upper_index=None, batch_number=None,
                fold_n=fold_n, layer='gene_block',
                return_attention_weights=True
            )
            out2 = model.act(out2)

            # ---- 最后层: gene_conv_last ----
            out3, (_, alpha3) = model.gene_conv_last(
                out2, edge_index_device,
                index=None, upper_index=None, batch_number=None,
                fold_n=fold_n, layer='gene_last',
                return_attention_weights=True
            )

            # alpha shape: (num_edges, num_heads=1) → 取第0个head → (num_edges,)
            patient_alphas[patient_i] = {
                'first_alpha': alpha1.cpu().numpy().squeeze(),
                'block_alpha': alpha2.cpu().numpy().squeeze(),
                'last_alpha':  alpha3.cpu().numpy().squeeze()
            }

            # 每100个患者打印一次进度
            if (patient_i + 1) % 100 == 0 or patient_i == num_patients - 1:
                print(f'    已处理 {patient_i + 1}/{num_patients} 个患者')

    print(f'  [Phase 1 完成] fold {fold_n}: 提取了 {len(patient_alphas)} 个患者的 attention')
    return patient_alphas


def save_patient_alphas(fold_n, patient_alphas, output_dir):
    """将提取的 attention weights 保存到 npz 文件"""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'fold{fold_n}_patient_alphas.npz')

    # 将 dict 转为 numpy 数组以便保存
    patient_indices = sorted(patient_alphas.keys())
    first_alphas = np.array([patient_alphas[i]['first_alpha'] for i in patient_indices])
    block_alphas = np.array([patient_alphas[i]['block_alpha'] for i in patient_indices])
    last_alphas  = np.array([patient_alphas[i]['last_alpha']  for i in patient_indices])

    np.savez(save_path,
             patient_indices=np.array(patient_indices),
             first_alphas=first_alphas,   # shape: (num_patients, num_edges)
             block_alphas=block_alphas,
             last_alphas=last_alphas)
    print(f'  [保存] {save_path}  shape: {last_alphas.shape}')
    return save_path


def load_patient_alphas(fold_n, output_dir):
    """从 npz 文件加载之前提取的 attention weights"""
    load_path = os.path.join(output_dir, f'fold{fold_n}_patient_alphas.npz')
    if not os.path.exists(load_path):
        print(f'  [错误] 文件不存在: {load_path}')
        return None
    data = np.load(load_path)
    patient_indices = data['patient_indices']
    # 重建 dict 结构
    patient_alphas = {}
    for idx_pos, patient_i in enumerate(patient_indices):
        patient_alphas[int(patient_i)] = {
            'first_alpha': data['first_alphas'][idx_pos],
            'block_alpha': data['block_alphas'][idx_pos],
            'last_alpha':  data['last_alphas'][idx_pos]
        }
    print(f'  [加载] {load_path}: {len(patient_alphas)} 个患者')
    return patient_alphas


# ============================================================
# Phase 2: 按患者类型分组统计比较
# ============================================================

def get_patient_groups(label_df):
    """
    从标签文件中获取三类患者的索引列表。
    注意: node_idx 是患者在整体图中的索引 (含122个subfeature偏移)，
    而 gene_feature 中的索引是 0..812 (纯患者序号)。
    需要用 node_idx - 122 来映射。
    """
    subfeature_df = pd.read_csv('./data/filtered_data/subfeature_dict_df.csv')
    num_subfeature = subfeature_df.shape[0]  # 122

    groups = {}
    for ptype in ['t2ds', 'pret2ds', 'no_t2ds']:
        node_indices = label_df[label_df[ptype] == 1]['node_idx'].values
        # 转换为 gene_feature 中的患者序号 (0-based)
        patient_indices = node_indices - num_subfeature
        groups[ptype] = patient_indices.tolist()

    print(f'\n[分组] t2ds: {len(groups["t2ds"])}人, '
          f'pret2ds: {len(groups["pret2ds"])}人, '
          f'no_t2ds: {len(groups["no_t2ds"])}人')
    return groups


def compute_group_statistics(patient_alphas, groups, layer_name='last_alpha'):
    """
    计算各组的 attention 统计量。

    返回 dict 包含:
    - group_means: 每组的平均 attention 向量
    - cosine_similarities: 组间余弦相似度
    - snr: 信噪比 (组间方差 / 组内方差)
    - edge_pvalues: 每条边的 Wilcoxon 检验 p-value (t2ds vs no_t2ds)
    """
    results = {}

    # ---- 1. 计算每组的平均 attention 向量 ----
    group_means = {}
    group_all_alphas = {}  # 存储每组所有患者的 alpha 矩阵
    for ptype, indices in groups.items():
        # 收集该组所有患者的 attention 向量
        alphas = []
        for idx in indices:
            if idx in patient_alphas:
                alphas.append(patient_alphas[idx][layer_name])
        if len(alphas) == 0:
            print(f'  [警告] {ptype} 组没有匹配到任何患者')
            continue
        alphas = np.array(alphas)  # shape: (num_patients_in_group, num_edges)
        group_means[ptype] = np.mean(alphas, axis=0)
        group_all_alphas[ptype] = alphas
        print(f'  [{layer_name}] {ptype}: {alphas.shape[0]}人, '
              f'mean_alpha={np.mean(alphas):.8f}, std={np.std(alphas):.8f}')

    results['group_means'] = {k: v.tolist() for k, v in group_means.items()}

    # ---- 2. 组间余弦相似度 ----
    cosine_sims = {}
    type_pairs = [('t2ds', 'no_t2ds'), ('t2ds', 'pret2ds'), ('pret2ds', 'no_t2ds')]
    for t1, t2 in type_pairs:
        if t1 in group_means and t2 in group_means:
            m1, m2 = group_means[t1], group_means[t2]
            cos_sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-15)
            cosine_sims[f'{t1}_vs_{t2}'] = float(cos_sim)
            print(f'  余弦相似度 {t1} vs {t2}: {cos_sim:.10f}')
    results['cosine_similarities'] = cosine_sims

    # ---- 3. 信噪比 (SNR) ----
    # SNR = 组间方差 / 组内平均方差
    if len(group_all_alphas) >= 2:
        # 组间方差: 各组均值的方差
        all_group_means = np.array([group_means[k] for k in group_means])
        var_between = np.mean(np.var(all_group_means, axis=0))
        # 组内方差: 各组内部方差的加权平均
        var_within_list = []
        for ptype, alphas in group_all_alphas.items():
            var_within_list.append(np.mean(np.var(alphas, axis=0)))
        var_within = np.mean(var_within_list)
        snr = var_between / (var_within + 1e-15)
        results['snr'] = float(snr)
        results['var_between'] = float(var_between)
        results['var_within'] = float(var_within)
        print(f'  SNR = {snr:.6f} (组间方差={var_between:.10f}, 组内方差={var_within:.10f})')

    # ---- 4. 逐边 Wilcoxon 检验 (t2ds vs no_t2ds) ----
    if 't2ds' in group_all_alphas and 'no_t2ds' in group_all_alphas:
        t2ds_alphas = group_all_alphas['t2ds']
        no_t2ds_alphas = group_all_alphas['no_t2ds']
        num_edges = t2ds_alphas.shape[1]

        pvalues = np.ones(num_edges)
        significant_count = 0

        print(f'  正在计算 {num_edges} 条边的 Wilcoxon 检验 ...')
        for edge_i in range(num_edges):
            t2ds_vals = t2ds_alphas[:, edge_i]
            no_t2ds_vals = no_t2ds_alphas[:, edge_i]
            try:
                _, pval = stats.mannwhitneyu(t2ds_vals, no_t2ds_vals, alternative='two-sided')
                pvalues[edge_i] = pval
                if pval < 0.05:
                    significant_count += 1
            except ValueError:
                # 当两组值完全相同时会报错
                pvalues[edge_i] = 1.0

        results['edge_pvalues'] = pvalues
        results['significant_edges_005'] = int(significant_count)
        results['total_edges'] = int(num_edges)
        results['significant_ratio'] = float(significant_count / num_edges)
        print(f'  t2ds vs no_t2ds: {significant_count}/{num_edges} 条边 p<0.05 '
              f'({significant_count/num_edges*100:.2f}%)')

        # Bonferroni 校正后的显著边数
        bonferroni_threshold = 0.05 / num_edges
        bonferroni_count = int(np.sum(pvalues < bonferroni_threshold))
        results['significant_edges_bonferroni'] = bonferroni_count
        print(f'  Bonferroni 校正后: {bonferroni_count}/{num_edges} 条边显著')

    # ---- 5. 逐边 Wilcoxon 检验 (pret2ds vs no_t2ds) ----
    if 'pret2ds' in group_all_alphas and 'no_t2ds' in group_all_alphas:
        pret2ds_alphas = group_all_alphas['pret2ds']
        no_t2ds_alphas = group_all_alphas['no_t2ds']
        num_edges = pret2ds_alphas.shape[1]

        pvalues_pre = np.ones(num_edges)
        significant_count_pre = 0

        print(f'  正在计算 pret2ds vs no_t2ds 的 {num_edges} 条边 Wilcoxon 检验 ...')
        for edge_i in range(num_edges):
            pre_vals = pret2ds_alphas[:, edge_i]
            no_vals = no_t2ds_alphas[:, edge_i]
            try:
                _, pval = stats.mannwhitneyu(pre_vals, no_vals, alternative='two-sided')
                pvalues_pre[edge_i] = pval
                if pval < 0.05:
                    significant_count_pre += 1
            except ValueError:
                pvalues_pre[edge_i] = 1.0

        results['pret2ds_significant_edges_005'] = int(significant_count_pre)
        results['pret2ds_significant_ratio'] = float(significant_count_pre / num_edges)
        print(f'  pret2ds vs no_t2ds: {significant_count_pre}/{num_edges} 条边 p<0.05 '
              f'({significant_count_pre/num_edges*100:.2f}%)')

    # ---- 6. 组均值绝对差异 ----
    for t1, t2 in type_pairs:
        if t1 in group_means and t2 in group_means:
            diff = np.abs(group_means[t1] - group_means[t2])
            results[f'mean_abs_diff_{t1}_vs_{t2}'] = float(np.mean(diff))
            results[f'max_abs_diff_{t1}_vs_{t2}'] = float(np.max(diff))
            print(f'  |mean_diff| {t1} vs {t2}: mean={np.mean(diff):.10f}, max={np.max(diff):.10f}')

    return results


# ============================================================
# Phase 3: Sanity Check（随机置换检验）
# ============================================================

def permutation_test(patient_alphas, groups, layer_name='last_alpha',
                     n_permutations=100):
    """
    随机打乱患者类型标签，重新计算 SNR 和显著边数，
    与真实结果对比。如果真实结果和随机结果相似，说明没有真实差异。
    """
    print(f'\n[Phase 3] 随机置换检验 ({n_permutations} 次) ...')

    # 收集所有患者索引
    all_indices = []
    for ptype, indices in groups.items():
        all_indices.extend(indices)

    # 各组大小
    group_sizes = {ptype: len(indices) for ptype, indices in groups.items()}

    permuted_snrs = []
    permuted_sig_counts = []

    for perm_i in range(n_permutations):
        # 随机打乱所有索引
        shuffled = np.random.permutation(all_indices)

        # 按原来的组大小重新分组
        fake_groups = {}
        offset = 0
        for ptype in ['t2ds', 'pret2ds', 'no_t2ds']:
            size = group_sizes[ptype]
            fake_groups[ptype] = shuffled[offset:offset+size].tolist()
            offset += size

        # 计算 fake SNR
        group_means = {}
        group_vars = []
        for ptype, indices in fake_groups.items():
            alphas = np.array([patient_alphas[idx][layer_name] for idx in indices
                              if idx in patient_alphas])
            if len(alphas) > 0:
                group_means[ptype] = np.mean(alphas, axis=0)
                group_vars.append(np.mean(np.var(alphas, axis=0)))

        if len(group_means) >= 2:
            all_means = np.array(list(group_means.values()))
            var_between = np.mean(np.var(all_means, axis=0))
            var_within = np.mean(group_vars)
            snr = var_between / (var_within + 1e-15)
            permuted_snrs.append(snr)

        # 计算 fake 显著边数 (t2ds vs no_t2ds, 只抽样100条边加速)
        if 't2ds' in fake_groups and 'no_t2ds' in fake_groups:
            t2_alphas = np.array([patient_alphas[idx][layer_name] for idx in fake_groups['t2ds']
                                 if idx in patient_alphas])
            no_alphas = np.array([patient_alphas[idx][layer_name] for idx in fake_groups['no_t2ds']
                                 if idx in patient_alphas])
            num_edges = t2_alphas.shape[1]
            # 随机抽样200条边做检验（全部做太慢）
            sample_edges = np.random.choice(num_edges, size=min(200, num_edges), replace=False)
            sig_count = 0
            for ei in sample_edges:
                try:
                    _, pval = stats.mannwhitneyu(t2_alphas[:, ei], no_alphas[:, ei],
                                                 alternative='two-sided')
                    if pval < 0.05:
                        sig_count += 1
                except ValueError:
                    pass
            permuted_sig_counts.append(sig_count / len(sample_edges))

        if (perm_i + 1) % 20 == 0:
            print(f'    置换检验进度: {perm_i + 1}/{n_permutations}')

    perm_results = {
        'n_permutations': n_permutations,
        'permuted_snr_mean': float(np.mean(permuted_snrs)) if permuted_snrs else None,
        'permuted_snr_std': float(np.std(permuted_snrs)) if permuted_snrs else None,
        'permuted_snr_max': float(np.max(permuted_snrs)) if permuted_snrs else None,
        'permuted_sig_ratio_mean': float(np.mean(permuted_sig_counts)) if permuted_sig_counts else None,
        'permuted_sig_ratio_std': float(np.std(permuted_sig_counts)) if permuted_sig_counts else None,
    }

    print(f'  随机置换 SNR: mean={perm_results["permuted_snr_mean"]:.6f}, '
          f'std={perm_results["permuted_snr_std"]:.6f}, '
          f'max={perm_results["permuted_snr_max"]:.6f}')
    print(f'  随机置换 显著边比例: mean={perm_results["permuted_sig_ratio_mean"]:.4f}, '
          f'std={perm_results["permuted_sig_ratio_std"]:.4f}')

    return perm_results


# ============================================================
# Phase 4: 输入特征差异性检查
# ============================================================

def check_input_feature_variance(gene_feature, groups, num_gene_node):
    """
    检查不同类型患者的原始基因输入特征是否本身就有差异。
    如果输入特征没有差异，attention 自然不可能有差异。
    """
    print(f'\n[Phase 4] 检查输入基因特征的组间差异 ...')

    num_features = 6
    for ptype, indices in groups.items():
        features = gene_feature[indices]  # shape: (n_patients, num_gene_node * num_features)
        print(f'  {ptype}: mean={np.mean(features):.6f}, std={np.std(features):.6f}, '
              f'min={np.min(features):.6f}, max={np.max(features):.6f}')

    # t2ds vs no_t2ds 的输入特征比较
    t2ds_feat = gene_feature[groups['t2ds']]    # (68, 8340)
    no_feat = gene_feature[groups['no_t2ds']]   # (641, 8340)

    # 对每个特征维度做 Mann-Whitney U 检验
    num_dims = t2ds_feat.shape[1]
    sig_count = 0
    for dim_i in range(num_dims):
        try:
            _, pval = stats.mannwhitneyu(t2ds_feat[:, dim_i], no_feat[:, dim_i],
                                          alternative='two-sided')
            if pval < 0.05:
                sig_count += 1
        except ValueError:
            pass

    print(f'  输入特征 t2ds vs no_t2ds: {sig_count}/{num_dims} 维 p<0.05 '
          f'({sig_count/num_dims*100:.2f}%)')

    # 余弦相似度
    mean_t2ds = np.mean(t2ds_feat, axis=0)
    mean_no = np.mean(no_feat, axis=0)
    cos_sim = np.dot(mean_t2ds, mean_no) / (np.linalg.norm(mean_t2ds) * np.linalg.norm(mean_no) + 1e-15)
    print(f'  输入特征余弦相似度 t2ds vs no_t2ds: {cos_sim:.10f}')

    return {
        'input_significant_dims': int(sig_count),
        'input_total_dims': int(num_dims),
        'input_cosine_sim': float(cos_sim)
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='直接验证基因层 attention 的患者类型差异')
    parser.add_argument('--fold', type=str, default='all',
                        help='要分析的折数: 1-5 或 "all"')
    parser.add_argument('--skip_extract', action='store_true',
                        help='跳过提取步骤，直接从已保存的 npz 文件加载')
    parser.add_argument('--output_dir', type=str, default='verify_output',
                        help='输出目录 (默认: verify_output)')
    parser.add_argument('--n_permutations', type=int, default=100,
                        help='随机置换检验次数 (默认: 100)')
    parser.add_argument('--cuda', type=str, default='0',
                        help='GPU 编号 (默认: 0)')
    args = parser.parse_args()

    # 设备设置
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f'[设备] {device}')

    # 确定要处理的折数
    if args.fold == 'all':
        fold_list = [1, 2, 3, 4, 5]
    else:
        fold_list = [int(args.fold)]
    print(f'[配置] 处理折数: {fold_list}')

    # 加载数据
    gene_feature, gene_edge_index, label_df, num_gene_node = load_data()

    # 获取患者分组
    groups = get_patient_groups(label_df)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # ======== Phase 4: 先检查输入特征差异 ========
    input_stats = check_input_feature_variance(gene_feature, groups, num_gene_node)

    # ======== Phase 1 & 2: 逐折处理 ========
    all_fold_results = {}
    for fold_n in fold_list:
        print(f'\n{"="*60}')
        print(f'  FOLD {fold_n}')
        print(f'{"="*60}')

        # Phase 1: 提取 attention weights
        if args.skip_extract:
            print(f'[跳过提取] 从文件加载 fold {fold_n} ...')
            patient_alphas = load_patient_alphas(fold_n, args.output_dir)
            if patient_alphas is None:
                print(f'  无法加载 fold {fold_n}，跳过')
                continue
        else:
            patient_alphas = extract_patient_attentions(
                fold_n, gene_feature, gene_edge_index, num_gene_node, device)
            if patient_alphas is None:
                continue
            save_patient_alphas(fold_n, patient_alphas, args.output_dir)

        # Phase 2: 统计比较（三层都分析）
        fold_results = {}
        for layer_name in ['first_alpha', 'block_alpha', 'last_alpha']:
            print(f'\n--- {layer_name} 层分析 ---')
            layer_results = compute_group_statistics(patient_alphas, groups, layer_name)
            # 去掉大数组，只保留标量统计量
            scalar_results = {k: v for k, v in layer_results.items()
                             if not isinstance(v, np.ndarray)}
            fold_results[layer_name] = scalar_results

        # Phase 3: 置换检验（只用 last_alpha 层）
        perm_results = permutation_test(patient_alphas, groups,
                                        layer_name='last_alpha',
                                        n_permutations=args.n_permutations)
        fold_results['permutation_test'] = perm_results

        all_fold_results[f'fold_{fold_n}'] = fold_results

    # ======== 保存汇总结果 ========
    all_fold_results['input_feature_check'] = input_stats

    summary_path = os.path.join(args.output_dir, 'global_stats.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_fold_results, f, indent=2, ensure_ascii=False)
    print(f'\n[保存] 汇总统计: {summary_path}')

    # ======== 打印最终判断 ========
    print('\n' + '='*60)
    print('  最终结果摘要')
    print('='*60)

    for fold_key, fold_data in all_fold_results.items():
        if not fold_key.startswith('fold_'):
            continue
        print(f'\n{fold_key}:')
        last = fold_data.get('last_alpha', {})
        print(f'  SNR = {last.get("snr", "N/A")}')
        for key in ['cosine_similarities']:
            if key in last:
                for pair, val in last[key].items():
                    print(f'  cos_sim ({pair}) = {val}')
        print(f'  显著边 (t2ds vs no_t2ds, p<0.05): '
              f'{last.get("significant_edges_005", "N/A")}/{last.get("total_edges", "N/A")} '
              f'= {last.get("significant_ratio", "N/A")}')
        perm = fold_data.get('permutation_test', {})
        print(f'  随机置换 SNR: {perm.get("permuted_snr_mean", "N/A")} ± {perm.get("permuted_snr_std", "N/A")}')

    print('\n判断标准:')
    print('  - 如果 SNR ≈ 0 且余弦相似度 ≈ 1.0 → 模型未学到基因层患者差异')
    print('  - 如果 SNR > 1 且有大量显著边 → 模型有区分，之前是代码管道的问题')
    print('  - 如果真实 SNR 与随机置换 SNR 无显著差异 → 差异只是噪声')


if __name__ == '__main__':
    main()
