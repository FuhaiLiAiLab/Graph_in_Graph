"""
分析 GIG Transformer 模型中各基因特征的权重
方法：提取 gene_conv_last 层输出的基因嵌入 L2 norm × node_weight_assign 软权重
按三组人群 (t2ds / pret2ds / no_t2ds) 分别汇报 Top-20 权重基因
覆盖 5-fold 模型，结果取平均后输出
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict

# ── 路径设置 ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from enc.geo_gigtransformer import GIG_Transformer

# ── 超参数（从 checkpoint state_dict 形状反推） ─────────────────────────────────
MODEL_ARGS = dict(
    gene_input_dim        = 6,
    gene_hidden_dim       = 18,
    gene_embedding_dim    = 18,   # gene_output_dim
    gene_num_top_feature  = 18,
    gig_input_dim         = 42,
    gig_input_transform_dim = 18,
    gig_hidden_dim        = 18,
    gig_embedding_dim     = 18,   # gig_output_dim
    num_classes           = 3,
    gene_num_head         = 1,
    gig_num_head          = 1,
    class_weight_fine     = 1.0,  # 推理阶段不用，占位
    class_weight          = 1.0,
    ortho_weight          = 0.0,
    link_weight           = 0.0,
    ent_weight            = 0.0,
    graph_opt             = 'GinG',
)

NUM_GENE_NODE = 1390
NUM_FOLDS     = 5
TOP_K         = 20
GROUPS        = ['t2ds', 'pret2ds', 'no_t2ds']
OUTPUT_DIR    = os.path.join(BASE_DIR, 'enrichment_output', 'gene_weights_by_group')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')   # 分析脚本用 CPU 足够


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_model(fold: int) -> GIG_Transformer:
    ckpt_path = os.path.join(
        BASE_DIR,
        f'gnn_result/gigtransformer/5-fold/epoch_500_fold{fold}/best_train_model.pth'
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model = GIG_Transformer(num_gene_node=NUM_GENE_NODE, **MODEL_ARGS)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_gene_scores_all_subjects(model: GIG_Transformer,
                                     gene_x: np.ndarray,
                                     gene_edge_index: torch.Tensor) -> np.ndarray:
    """
    对每个受试者跑 gene_conv_first→block→last，
    返回形状 (num_subjects, num_gene_node) 的权重分矩阵。
    权重分 = L2_norm(gene_embedding) × node_weight_assign 软权重
    """
    num_subjects = gene_x.shape[0]

    # node_weight_assign 软权重：[18, 1390] → softmax → sum → [1390]
    nwa = F.softmax(model.node_weight_assign, dim=-1)   # [18, 1390]
    nwa_sum = nwa.sum(dim=0).detach().cpu().numpy()     # [1390]

    all_scores = np.zeros((num_subjects, NUM_GENE_NODE), dtype=np.float32)

    with torch.no_grad():
        for i in range(num_subjects):
            x = torch.tensor(
                gene_x[i].reshape(NUM_GENE_NODE, 6).astype(np.float32)
            )  # [1390, 6]

            h = model.gene_conv_first(x, gene_edge_index)
            h = model.act(h)
            h = model.gene_conv_block(h, gene_edge_index)
            h = model.act(h)
            h = model.gene_conv_last(h, gene_edge_index)
            h = model.act(h)
            # h: [1390, 18]

            gene_norm = h.norm(dim=-1).cpu().numpy()        # [1390] L2 norm
            all_scores[i] = gene_norm * nwa_sum             # 加权

    return all_scores   # (813, 1390)


def rank_aggregate(fold_scores_list: list) -> np.ndarray:
    """
    Borda count rank aggregation：
    每个 fold 给每个基因打分→取 5-fold 平均
    """
    return np.mean(fold_scores_list, axis=0)    # (1390,)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=== Loading data ===")
    gene_x = np.load(os.path.join(BASE_DIR, 'data/post_data/norm_gene_x.npy'),
                     allow_pickle=True).astype(np.float32)          # (813, 8340)
    gene_edge_index_np = np.load(
        os.path.join(BASE_DIR, 'data/post_data/gene_edge_index.npy'),
        allow_pickle=True).astype(np.int64)                         # (2, 9199)
    gene_edge_index = torch.from_numpy(gene_edge_index_np).long()

    label_df  = pd.read_csv(
        os.path.join(BASE_DIR, 'data/filtered_data/label_phenodata_onehot_nodeidx_df.csv'))
    gene_df   = pd.read_csv(
        os.path.join(BASE_DIR, 'data/filtered_data/gene_num_dict_df.csv'))

    # Subject grouping (row order 0..812 matches gene_x row order)
    group_indices = {}
    for g in GROUPS:
        group_indices[g] = label_df.index[label_df[g] == 1].tolist()
    for g, idx in group_indices.items():
        print(f"  {g}: {len(idx)} subjects")

    # ── 5-fold loop ───────────────────────────────────────────────────────────
    fold_group_scores = {g: [] for g in GROUPS}

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n=== Fold {fold} ===")
        model = load_model(fold)

        scores_all = compute_gene_scores_all_subjects(
            model, gene_x, gene_edge_index)   # (813, 1390)

        for g in GROUPS:
            idx = group_indices[g]
            group_mean = scores_all[idx].mean(axis=0)  # (1390,)
            fold_group_scores[g].append(group_mean)
            top_idx = np.argsort(group_mean)[-5:][::-1]
            top_names = gene_df.iloc[top_idx]['gene_node_name'].tolist()
            print(f"  {g} fold{fold} Top-5: {top_names}")

    # ── Aggregate across folds ────────────────────────────────────────────────
    print("\n=== 5-fold aggregated results ===")
    final_scores = {}
    for g in GROUPS:
        final_scores[g] = rank_aggregate(fold_group_scores[g])  # (1390,)

    # ── Report & save Top-20 ─────────────────────────────────────────────────
    results = {}
    for g in GROUPS:
        top_idx = np.argsort(final_scores[g])[-TOP_K:][::-1]
        top_names  = gene_df.iloc[top_idx]['gene_node_name'].tolist()
        top_scores = final_scores[g][top_idx].tolist()
        results[g] = pd.DataFrame({
            'rank':      list(range(1, TOP_K + 1)),
            'gene':      top_names,
            'score':     [round(s, 6) for s in top_scores],
        })
        print(f"\n-- {g} Top-{TOP_K} --")
        print(results[g].to_string(index=False))

        csv_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_{g}.csv')
        results[g].to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}")

    # ── Combined table ────────────────────────────────────────────────────────
    combined = pd.concat(
        [df.assign(group=g) for g, df in results.items()],
        ignore_index=True
    )
    combined_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_all_groups.csv')
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined table saved -> {combined_path}")

    # ── Group-unique gene analysis ────────────────────────────────────────────
    gene_sets = {g: set(results[g]['gene']) for g in GROUPS}
    print("\n=== Top-20 gene overlap analysis ===")
    for g in GROUPS:
        others = set()
        for g2 in GROUPS:
            if g2 != g:
                others |= gene_sets[g2]
        unique = gene_sets[g] - others
        shared = gene_sets[g] & others
        print(f"  {g}: unique={sorted(unique)}, shared={sorted(shared)}")

    # ── 可视化：横向柱状图 ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
    colors = {'t2ds': '#d62728', 'pret2ds': '#ff7f0e', 'no_t2ds': '#1f77b4'}

    for ax, g in zip(axes, GROUPS):
        df = results[g]
        ax.barh(df['gene'][::-1], df['score'][::-1], color=colors[g], alpha=0.85)
        ax.set_title(f'{g}\n(Top {TOP_K} genes)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Weighted Gene Score', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('GIG Model — Top 20 Gene Feature Weights by Group\n(gene_conv_last embedding norm × node_weight_assign)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_gene_weights_barplot.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nBarplot saved -> {fig_path}")

    # ── Heatmap: 3 groups x union of top-20 genes ────────────────────────────
    union_genes = sorted(set().union(*[gene_sets[g] for g in GROUPS]))
    heatmap_data = pd.DataFrame(index=union_genes, columns=GROUPS, dtype=float)
    gene_name_to_idx = {name: i for i, name in enumerate(gene_df['gene_node_name'])}
    for g in GROUPS:
        for gene in union_genes:
            gi = gene_name_to_idx.get(gene)
            heatmap_data.loc[gene, g] = float(final_scores[g][gi]) if gi is not None else 0.0

    fig2, ax2 = plt.subplots(figsize=(6, max(6, len(union_genes) * 0.35)))
    im = ax2.imshow(heatmap_data.values.astype(float), aspect='auto', cmap='YlOrRd')
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(GROUPS, fontsize=11)
    ax2.set_yticks(range(len(union_genes)))
    ax2.set_yticklabels(union_genes, fontsize=7)
    plt.colorbar(im, ax=ax2, label='Weighted Gene Score')
    ax2.set_title('Gene Score Heatmap (Top-20 union across groups)', fontsize=11)
    plt.tight_layout()
    hm_path = os.path.join(OUTPUT_DIR, f'top{TOP_K}_gene_heatmap.png')
    plt.savefig(hm_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved -> {hm_path}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
