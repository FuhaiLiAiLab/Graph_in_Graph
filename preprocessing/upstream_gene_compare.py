"""
upstream_gene_compare.py

将 upstream_gene_results.csv（eQTL上游基因）与 t6.txt（GWAS T2D显著基因）进行比对，
合并模型基因的权重和p-value信息，输出汇总CSV和控制台摘要。
"""

import os
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_CSV = os.path.join(BASE_DIR, "analysis_lu", "upstream_gene_results.csv")
T6_TXT       = os.path.join(BASE_DIR, "t6.txt")
NODE_CSV     = os.path.join(BASE_DIR, "analysis_lu", "gigtransformer-rownorm", "inner_join_norm_refilter_pvalue_filtered_node_weight_df.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "analysis_lu", "upstream_gwas_overlap_summary.csv")

# ── 1. 读取数据 ────────────────────────────────────────────────────────────────
print("读取数据...")

upstream_df = pd.read_csv(UPSTREAM_CSV)
# 列: target_gene | upstream_genes (逗号分隔字符串) | upstream_gene_count

t6_df = pd.read_csv(T6_TXT, sep="\t")
gwas_genes = set(t6_df["Locus"].dropna().str.strip().tolist())
print(f"  GWAS T2D 显著基因数（t6.txt）: {len(gwas_genes)}")

node_df = pd.read_csv(NODE_CSV, index_col=0)
# 列: gene_node_idx | gene_node_name | Weight1 | node1 | Weight2 | node2
#     | t2ds_no_t2ds_pvalue | pret2ds_no_t2ds_pvalue

# ── 2. 计算每个目标基因的 GWAS 上游重叠 ─────────────────────────────────────
def calc_gwas_overlap(upstream_str):
    """返回 (overlap基因逗号字符串, overlap数量)"""
    if not upstream_str or pd.isna(upstream_str):
        return "", 0
    genes   = [g.strip() for g in str(upstream_str).split(",") if g.strip()]
    overlap = [g for g in genes if g in gwas_genes]
    return ",".join(sorted(overlap)), len(overlap)

overlap_results = upstream_df["upstream_genes"].apply(
    lambda x: pd.Series(calc_gwas_overlap(x), index=["gwas_t2d_upstream_genes", "gwas_t2d_upstream_count"])
)
upstream_df = pd.concat([upstream_df, overlap_results], axis=1)

# GWAS上游基因占该基因所有上游基因的比例
upstream_df["gwas_overlap_ratio"] = upstream_df.apply(
    lambda r: round(r["gwas_t2d_upstream_count"] / r["upstream_gene_count"], 4)
              if r["upstream_gene_count"] > 0 else 0.0,
    axis=1,
)

# 目标基因本身是否是 GWAS T2D 显著基因
upstream_df["is_self_gwas_t2d"] = upstream_df["target_gene"].isin(gwas_genes)

# ── 3. 合并模型信息（权重、p-value）──────────────────────────────────────────
node_df = node_df.rename(columns={"gene_node_name": "target_gene"})

result_df = node_df[
    ["target_gene", "gene_node_idx",
     "Weight1", "node1", "Weight2", "node2",
     "t2ds_no_t2ds_pvalue", "pret2ds_no_t2ds_pvalue"]
].merge(upstream_df, on="target_gene", how="left")

# ── 4. 排序：GWAS上游命中数↓ → t2ds显著性↑ ──────────────────────────────────
result_df = result_df.sort_values(
    by=["gwas_t2d_upstream_count", "t2ds_no_t2ds_pvalue"],
    ascending=[False, True],
).reset_index(drop=True)

# ── 5. 输出主 CSV ─────────────────────────────────────────────────────────────
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n主汇总文件已保存: {OUTPUT_CSV}")

# ── 6. 控制台摘要 ─────────────────────────────────────────────────────────────
total        = len(result_df)
has_upstream = int((result_df["upstream_gene_count"] > 0).sum())
has_overlap  = int((result_df["gwas_t2d_upstream_count"] > 0).sum())
self_gwas    = int(result_df["is_self_gwas_t2d"].sum())

print("\n" + "="*60)
print("  GWAS T2D × 上游基因  重叠分析摘要")
print("="*60)
print(f"  模型目标基因总数              : {total}")
print(f"  找到上游基因的目标基因数       : {has_upstream}  ({has_upstream/total*100:.1f}%)")
print(f"  ── 其中上游含GWAS T2D基因      : {has_overlap}  ({has_overlap/total*100:.1f}%)")
print(f"  目标基因本身是GWAS T2D基因     : {self_gwas}  ({self_gwas/total*100:.1f}%)")
print("="*60)

# 显示命中最多的前20个目标基因
print("\n【上游GWAS T2D基因命中数 Top 20（按命中数↓ + p-value↑排序）】")
top_cols = [
    "target_gene", "gwas_t2d_upstream_count", "gwas_overlap_ratio",
    "gwas_t2d_upstream_genes", "t2ds_no_t2ds_pvalue", "pret2ds_no_t2ds_pvalue",
    "is_self_gwas_t2d"
]
top20 = result_df[result_df["gwas_t2d_upstream_count"] > 0][top_cols].head(20)
print(top20.to_string(index=False))

# 统计：哪些 GWAS T2D 基因作为上游基因出现频率最高
print("\n【最常出现为上游调控基因的 GWAS T2D 基因 Top 20】")
all_upstream_gwas = []
for row in result_df["gwas_t2d_upstream_genes"].dropna():
    if row:
        all_upstream_gwas.extend([g.strip() for g in row.split(",") if g.strip()])

gwas_upstream_freq = (
    pd.Series(all_upstream_gwas)
    .value_counts()
    .reset_index()
    .rename(columns={"index": "gwas_gene", 0: "frequency", "count": "frequency"})
    .head(20)
)
# 补充 t6 中的染色体和信号数信息
gwas_upstream_freq = gwas_upstream_freq.merge(
    t6_df[["Locus", "Chr", "Number of distinct signals"]].rename(columns={"Locus": "gwas_gene"}),
    on="gwas_gene", how="left"
)
print(gwas_upstream_freq.to_string(index=False))

print("\n完成。")
