import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(BASE, "analysis", "gigtransformer-rownorm")
PVAL_DIR     = os.path.join(BASE, "pvalues_output")
OUT_DIR      = os.path.join(BASE, "image_storage")

t2ds_node_path  = os.path.join(ANALYSIS_DIR, "t2ds_norm_refilter_node_weight_df.csv")
pre_node_path   = os.path.join(ANALYSIS_DIR, "pret2ds_norm_refilter_node_weight_df.csv")
tvsno_path      = os.path.join(PVAL_DIR, "TvsNO_min_pvalues.csv")
prevsno_path    = os.path.join(PVAL_DIR, "PrevsNO_min_pvalues.csv")

# ── 读取数据 ──────────────────────────────────────────────────────────────────
t2ds_node  = pd.read_csv(t2ds_node_path,  index_col=0)[["gene_node_name", "Weight"]].rename(columns={"Weight": "t2ds_weight"})
pre_node   = pd.read_csv(pre_node_path,   index_col=0)[["gene_node_name", "Weight"]].rename(columns={"Weight": "pret2ds_weight"})
tvsno_pv   = pd.read_csv(tvsno_path).rename(columns={"gene": "gene_node_name"})
prevsno_pv = pd.read_csv(prevsno_path).rename(columns={"gene": "gene_node_name"})

print(f"T2D 网络基因数: {len(t2ds_node)}")
print(f"Pre-T2D 网络基因数: {len(pre_node)}")
print(f"TvsNO p-value 基因数: {len(tvsno_pv)}")
print(f"PrevsNO p-value 基因数: {len(prevsno_pv)}")

# ── 合并：以两个网络的并集为基础 ─────────────────────────────────────────────
merged = pd.merge(t2ds_node, pre_node, on="gene_node_name", how="outer")
merged = pd.merge(merged, tvsno_pv,   on="gene_node_name", how="left")
merged = pd.merge(merged, prevsno_pv, on="gene_node_name", how="left")

# 只保留两个 p-value 都有的基因
merged = merged.dropna(subset=["t2ds_no_t2ds_pvalue", "pret2ds_no_t2ds_pvalue"])
print(f"\n合并后（含双侧p-value）基因数: {len(merged)}")

# ── 计算辅助列 ────────────────────────────────────────────────────────────────
ALPHA = 0.05
merged["in_t2ds_net"]     = merged["t2ds_weight"].notna()
merged["in_pre_net"]      = merged["pret2ds_weight"].notna()
merged["t2ds_weight"]     = merged["t2ds_weight"].fillna(0)
merged["pret2ds_weight"]  = merged["pret2ds_weight"].fillna(0)
merged["delta_weight"]    = merged["t2ds_weight"] - merged["pret2ds_weight"]
merged["avg_weight"]      = (merged["t2ds_weight"] + merged["pret2ds_weight"]) / 2

def classify(row, alpha=ALPHA):
    t_sig = row["t2ds_no_t2ds_pvalue"]    <= alpha
    p_sig = row["pret2ds_no_t2ds_pvalue"] <= alpha
    if t_sig and p_sig:
        return "Both_sig"
    elif t_sig and not p_sig:
        return "T2D_specific"
    elif not t_sig and p_sig:
        return "PreT2D_specific"
    else:
        return "Non_sig"

merged["quadrant"] = merged.apply(classify, axis=1)

# ── 汇总统计 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("【四象限基因数量】")
print(merged["quadrant"].value_counts().to_string())

print("\n【各象限：在 GiG 网络中的基因数】")
for q in ["Both_sig", "T2D_specific", "PreT2D_specific", "Non_sig"]:
    sub = merged[merged["quadrant"] == q]
    in_net = sub[(sub["in_t2ds_net"]) | (sub["in_pre_net"])]
    print(f"  {q}: {len(in_net)}/{len(sub)} 在网络中")

for quad in ["Both_sig", "T2D_specific", "PreT2D_specific"]:
    subset = merged[merged["quadrant"] == quad].copy()
    if len(subset) == 0:
        continue
    in_net = subset[(subset["in_t2ds_net"]) | (subset["in_pre_net"])]
    print(f"\n{'='*60}")
    print(f"【{quad}】共 {len(subset)} 基因，其中 {len(in_net)} 个在 GiG 网络中")
    print(f"  Delta weight 均值: {subset['delta_weight'].mean():.4f}  std: {subset['delta_weight'].std():.4f}")
    cols = ["gene_node_name", "t2ds_weight", "pret2ds_weight", "delta_weight",
            "t2ds_no_t2ds_pvalue", "pret2ds_no_t2ds_pvalue", "in_t2ds_net", "in_pre_net"]
    print(subset[cols].sort_values("avg_weight", ascending=False).head(30).to_string(index=False))

# ── 绘图：四象限散点图 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))

color_map = {
    "Both_sig":        "#FB9A99",
    "T2D_specific":    "#B2DF8A",
    "PreT2D_specific": "#CAB2D6",
    "Non_sig":         "#CCCCCC",
}
size_map = {"Both_sig": 90, "T2D_specific": 90, "PreT2D_specific": 90, "Non_sig": 15}
zorder_map = {"Both_sig": 4, "T2D_specific": 4, "PreT2D_specific": 4, "Non_sig": 2}

for quad, grp in merged.groupby("quadrant"):
    ax.scatter(
        -np.log10(grp["t2ds_no_t2ds_pvalue"]),
        -np.log10(grp["pret2ds_no_t2ds_pvalue"]),
        c=color_map[quad], s=size_map[quad],
        alpha=0.85, label=quad, edgecolors="none", zorder=zorder_map[quad]
    )

sig_line = -np.log10(ALPHA)
ax.axvline(sig_line, color="gray", linestyle="--", linewidth=1)
ax.axhline(sig_line, color="gray", linestyle="--", linewidth=1)

# 标注：在网络中 + 显著 的 top 基因
label_df = merged[(merged["quadrant"] != "Non_sig") &
                  ((merged["in_t2ds_net"]) | (merged["in_pre_net"]))].copy()
label_df = label_df.nlargest(20, "avg_weight")
for _, row in label_df.iterrows():
    ax.annotate(
        row["gene_node_name"],
        xy=(-np.log10(row["t2ds_no_t2ds_pvalue"]),
            -np.log10(row["pret2ds_no_t2ds_pvalue"])),
        fontsize=7.5, ha="left", va="bottom",
        xytext=(3, 3), textcoords="offset points"
    )

ax.set_xlabel("-log10(p-value)  T2D vs No_T2D", fontsize=12)
ax.set_ylabel("-log10(p-value)  Pre-T2D vs No_T2D", fontsize=12)
ax.set_title("Gene Quadrant Analysis\n(network presence × p-value significance)", fontsize=13)

label_dict = {
    "Both_sig": "Both significant (shared markers)",
    "T2D_specific": "T2D-specific (progression markers)",
    "PreT2D_specific": "Pre-T2D-specific",
    "Non_sig": "Non-significant"
}
patches = [mpatches.Patch(color=color_map[q], label=label_dict[q]) for q in color_map]
ax.legend(handles=patches, loc="upper left", fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "gene_pvalue_quadrant_TvsNO_PrevsNO.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n图已保存: {out_path}")

out_csv = os.path.join(ANALYSIS_DIR, "gene_quadrant_analysis.csv")
merged.to_csv(out_csv, index=False)
print(f"完整结果已保存: {out_csv}")
