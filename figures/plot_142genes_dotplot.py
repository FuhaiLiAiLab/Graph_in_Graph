import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import pandas as pd

# ── 1. 数据载入 ────────────────────────────────────────────────────────────────
df = pd.read_csv("pvalues_output/stable_142genes_all_pvalues.csv")

comparisons  = ["TvsNO", "TvsPre", "PrevsNO"]
comp_labels  = {"TvsNO": "T vs NO", "TvsPre": "T vs Pre", "PrevsNO": "Pre vs NO"}
regions      = ["upstream", "core_promoter", "proximal_promoter",
                "distal_promoter", "tran_v1", "downstream"]
region_short = {
    "upstream":          "Upstream",
    "core_promoter":     "Core promoter",
    "proximal_promoter": "Proximal promoter",
    "distal_promoter":   "Distal promoter",
    "tran_v1":           "Transcript",
    "downstream":        "Downstream",
}

n_genes = len(df)
n_reg   = len(regions)

# ── 2. 基因排序：TvsNO 平均 adj_p 升序 → 显著性强的在上方（配合 invert_yaxis）──
tvs_adj_cols = [f"TvsNO_{r}_adj_pvalue" for r in regions]
gene_score   = df.set_index("gene")[tvs_adj_cols].mean(axis=1)
gene_order   = gene_score.sort_values(ascending=False).index.tolist()

# ── 3. 颜色归一化：以 p=0.05 为中心的双坡色阶，复现参考风格 ────────────────────
norm = TwoSlopeNorm(vmin=0, vmax=1.0, vcenter=0.05)

# ── 4. 画布：3个子图并排，高图适配 142 基因 ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 68), dpi=130,
                         sharey=True, constrained_layout=False)
fig.subplots_adjust(left=0.13, right=0.91, top=0.975, bottom=0.005, wspace=0.04)

# ── 5. 逐比较组绘图 ────────────────────────────────────────────────────────────
for ax, comp in zip(axes, comparisons):

    # 提取 adj_p 矩阵：(n_genes × n_reg)，按 gene_order 排列
    adj_cols = [f"{comp}_{r}_adj_pvalue" for r in regions]
    mat = df.set_index("gene").reindex(gene_order)[adj_cols].fillna(1.0).values

    # 生成坐标与颜色一维数组
    xs = np.tile(np.arange(n_reg), n_genes)       # x: 区域索引
    ys = np.repeat(np.arange(n_genes), n_reg)     # y: 基因索引
    cs = mat.flatten()                             # 颜色值 = adj_p_value

    # PatchCollection 圆形点
    R       = 0.38
    circles = [plt.Circle((xs[i], ys[i]), radius=R) for i in range(len(xs))]
    col     = PatchCollection(circles, array=cs, cmap="Oranges_r",
                              norm=norm, edgecolor="white", linewidth=0.6)
    ax.add_collection(col)

    # 坐标轴范围
    ax.set_xlim(-0.5, n_reg - 0.5)
    ax.set_ylim(-0.5, n_genes - 0.5)

    # x 轴标签置顶，旋转 90°（与参考风格一致）
    ax.set_xticks(np.arange(n_reg))
    ax.set_xticklabels([region_short[r] for r in regions],
                       rotation=90, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # minor grid 作为单元格分隔线
    ax.set_xticks(np.arange(n_reg) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_genes) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9, zorder=4)

    # 基因排在上方（显著性强 → 顶部）
    ax.invert_yaxis()

    ax.tick_params(axis="x", labelsize=8,  length=3, pad=4)
    ax.tick_params(axis="y", labelsize=5.5, length=0)

    ax.set_title(comp_labels[comp], fontsize=10, pad=76, loc="center")

    sns.despine(ax=ax, left=False, bottom=False, top=False, right=True)

# ── 6. y 轴基因标签（仅最左侧子图）─────────────────────────────────────────────
axes[0].set_yticks(np.arange(n_genes))
axes[0].set_yticklabels(gene_order, fontsize=5)

# ── 7. Colorbar ────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap="Oranges_r", norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.925, 0.32, 0.013, 0.32])
cbar    = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("adj. p-value", fontsize=8.5, labelpad=6)
cbar.set_ticks([0, 0.05, 0.10, 0.25, 0.50, 1.0])
cbar.ax.tick_params(labelsize=7.5)

# ── 8. 输出 ────────────────────────────────────────────────────────────────────
out = "image_storage/stable_142genes_region_pvalue_dotplot.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
