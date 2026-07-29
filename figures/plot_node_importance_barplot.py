"""
绘制节点重要性柱状图 + 底部p值点图
参考图风格：柱状图按重要性排序，颜色表示显著性类别；底部显示各比较的p值
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── 参数 ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "analysis", "gigtransformer-rownorm", "inner_join_norm_refilter_node_weight_df.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "figures", "outputs", "node_importance_barplot.png")
ALPHA = 0.05          # 显著性阈值
PVAL_MAX = 0.5        # colorbar 上限

# 颜色定义（与网络可视化保持一致）
COLOR_NONE   = '#6BAED6'   # 蓝色：无显著差异
COLOR_T2DS   = '#74C476'   # 绿色：仅 T2D vs No_T2D 显著
COLOR_PRE    = '#9E9AC8'   # 紫色：仅 Pre_T2D vs No_T2D 显著
COLOR_BOTH   = '#FC8D59'   # 橙红：两者均显著

# ─── 读取数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# 计算每个基因的重要性得分（取 Weight1 和 Weight2 的最大有效值）
def get_importance(row):
    w1 = row['Weight1'] if row['node1'] != 'no' else -np.inf
    w2 = row['Weight2']
    return max(w1, w2)

df['importance'] = df.apply(get_importance, axis=1)

# 判断节点类型（用于图例标注）
df['has_node1'] = df['node1'] == 'node_type1'
df['has_node2'] = df['node2'] == 'node_type2'

# 分配显著性类别
def sig_category(row):
    t = row['t2ds_no_t2ds_pvalue'] < ALPHA
    p = row['pret2ds_no_t2ds_pvalue'] < ALPHA
    if t and p:
        return 'both'
    elif t:
        return 't2ds'
    elif p:
        return 'pre'
    else:
        return 'none'

df['sig_cat'] = df.apply(sig_category, axis=1)

# 按重要性降序排列
df = df.sort_values('importance', ascending=False).reset_index(drop=True)

n_genes = len(df)
x = np.arange(n_genes)

# ─── 颜色映射 ──────────────────────────────────────────────────────────────────
cat_color = {'none': COLOR_NONE, 't2ds': COLOR_T2DS, 'pre': COLOR_PRE, 'both': COLOR_BOTH}
bar_colors = [cat_color[c] for c in df['sig_cat']]

# p值 colormap（深橙→白，代表低→高p值）
cmap_pval = plt.cm.YlOrBr_r
norm_pval  = Normalize(vmin=0, vmax=PVAL_MAX)

# ─── 创建画布 ──────────────────────────────────────────────────────────────────
fig_width  = max(16, n_genes * 0.15)
fig_height = 9

fig = plt.figure(figsize=(fig_width, fig_height))
gs  = gridspec.GridSpec(
    2, 1,
    height_ratios=[3.5, 1],
    hspace=0.04,
    left=0.05, right=0.95, top=0.93, bottom=0.18
)

ax_bar = fig.add_subplot(gs[0])
ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)

# ─── 上面板：柱状图 ────────────────────────────────────────────────────────────
ax_bar.bar(x, df['importance'], color=bar_colors, width=0.85, linewidth=0)
ax_bar.set_xlim(-0.5, n_genes - 0.5)
ax_bar.set_ylim(0, df['importance'].max() * 1.12)
ax_bar.set_ylabel('Node Importance', fontsize=11)
ax_bar.set_xticks([])
ax_bar.spines['bottom'].set_visible(False)
ax_bar.tick_params(axis='y', labelsize=9)

# 图例（显著性分类）
legend_patches = [
    mpatches.Patch(color=COLOR_NONE,  label='No significant difference'),
    mpatches.Patch(color=COLOR_T2DS,  label='Significant in T2ds vs No_T2ds'),
    mpatches.Patch(color=COLOR_PRE,   label='Significant in Pre_T2ds vs No_T2ds'),
    mpatches.Patch(color=COLOR_BOTH,  label='Significant in both'),
]
ax_bar.legend(handles=legend_patches, loc='upper right', fontsize=8,
              framealpha=0.8, edgecolor='gray')

# p值 colorbar（放在柱状图右上角）
sm = ScalarMappable(cmap=cmap_pval, norm=norm_pval)
sm.set_array([])
cax = fig.add_axes([0.61, 0.72, 0.14, 0.022])   # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks([0, 0.1, 0.2, 0.4, 0.5])
cbar.ax.tick_params(labelsize=7)
# 加框
for spine in cax.spines.values():
    spine.set_visible(True)

# ─── 下面板：p值点图 ────────────────────────────────────────────────────────────
pval_rows = [
    ('T2ds vs No_T2ds',      'pret2ds_no_t2ds_pvalue'),
    ('Pre_T2ds vs No_T2ds',  't2ds_no_t2ds_pvalue'),
]

dot_size  = max(4, 180 / n_genes)   # 基因越多点越小

for row_i, (row_label, col) in enumerate(pval_rows):
    pvals = df[col].values
    for gene_i, pval in enumerate(pvals):
        clipped = min(pval, PVAL_MAX)
        color   = cmap_pval(norm_pval(clipped))
        ax_dot.scatter(gene_i, row_i, color=color, s=dot_size,
                       marker='o', linewidths=0)

ax_dot.set_xlim(-0.5, n_genes - 0.5)
ax_dot.set_ylim(-0.6, 1.6)
ax_dot.set_yticks([0, 1])
ax_dot.set_yticklabels([r[0] for r in pval_rows], fontsize=8)
ax_dot.spines['top'].set_visible(False)
ax_dot.tick_params(axis='x', which='both', length=0)

# X轴基因名标签
ax_dot.set_xticks(x)
ax_dot.set_xticklabels(
    df['gene_node_name'].tolist(),
    rotation=90, fontsize=max(4, 6.5 - n_genes * 0.02),
    ha='center'
)

# --- Summary stats ---
print(f"\n=== Node Importance Summary ===")
print(f"Total genes: {n_genes}")
for cat, label in [('none','No sig diff'), ('t2ds','T2D sig only'), ('pre','Pre_T2D sig only'), ('both','Both sig')]:
    cnt = (df['sig_cat'] == cat).sum()
    print(f"  {label}: {cnt} ({cnt/n_genes*100:.1f}%)")
print(f"\nOnly in node_type2 (not in node_type1): {(~df['has_node1']).sum()}")
print(f"In both node_type1 and node_type2: {(df['has_node1']).sum()}")

# ─── 保存 ─────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_FILE.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved to:\n  {OUTPUT_FILE}")
plt.show()
