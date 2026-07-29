import os
import sys
import pandas as pd
import mygene
import pyranges as pr

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENE_LIST_CSV = os.path.join(BASE_DIR, "analysis_lu", "gigtransformer-rownorm", "inner_join_norm_refilter_pvalue_filtered_node_weight_df.csv")
EQTL_FILE     = os.path.join(
    BASE_DIR, "eqtl_data",
    "2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt",
    "2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt",
)
OUTPUT_CSV    = os.path.join(BASE_DIR, "analysis_lu", "upstream_gene_results.csv")

# ── 1. 读取目标基因列表 ────────────────────────────────────────────────────────
print("[1/5] 读取目标基因列表...")
target_df    = pd.read_csv(GENE_LIST_CSV)
target_genes = set(target_df["gene_node_name"].dropna().tolist())
print(f"      目标基因数量: {len(target_genes)}")

# ── 2. 读取 eQTLGen 数据，只保留必要列，筛选目标基因 ──────────────────────────
print("[2/5] 读取 eQTLGen 数据（大文件，仅加载必要列）...")
usecols  = ["SNP", "SNPChr", "SNPPos", "GeneSymbol"]
eqtl_df  = pd.read_csv(
    EQTL_FILE,
    sep="\t",
    usecols=usecols,
    dtype={"SNPChr": str, "SNPPos": int},
)
print(f"      eQTL 总行数: {len(eqtl_df):,}")

eqtl_target = eqtl_df[eqtl_df["GeneSymbol"].isin(target_genes)].copy()
print(f"      目标基因相关 eQTL 行数: {len(eqtl_target):,}")

if eqtl_target.empty:
    print("警告：目标基因在 eQTL 文件中无匹配记录，请检查基因名格式。")
    sys.exit(1)

# ── 3. 用 mygene 批量查询所有在 eQTL 文件中出现的基因坐标（hg19）──────────────
print("[3/5] 用 mygene 批量查询基因坐标（hg19）...")
all_eqtl_genes = eqtl_df["GeneSymbol"].unique().tolist()
print(f"      需要查询坐标的基因数: {len(all_eqtl_genes)}")

mg           = mygene.MyGeneInfo()
query_result = mg.querymany(
    all_eqtl_genes,
    scopes="symbol",
    fields="genomic_pos_hg19",
    species="human",
    verbose=False,
)

# 构建基因坐标表
gene_coord_records = []
for item in query_result:
    if item.get("notfound") or "genomic_pos_hg19" not in item:
        continue
    pos = item["genomic_pos_hg19"]
    if isinstance(pos, list):      # 多个映射位置时取第一个
        pos = pos[0]
    chrom = str(pos.get("chr", "")).replace("chr", "")
    start = pos.get("start")
    end   = pos.get("end")
    if chrom and start is not None and end is not None:
        gene_coord_records.append(
            {
                "gene_symbol": item["query"],
                "chrom":       chrom,
                "start":       int(start),
                "end":         int(end),
            }
        )

gene_coord_df = pd.DataFrame(gene_coord_records).drop_duplicates(subset="gene_symbol")
print(f"      成功获取坐标的基因数: {len(gene_coord_df)}")

if gene_coord_df.empty:
    print("错误：mygene 未返回任何坐标，请检查网络或基因名格式。")
    sys.exit(1)

# ── 4. 用 pyranges 做基因组区间重叠（SNP坐标 vs 基因体）──────────────────────
print("[4/5] 基因组区间重叠分析...")

# 基因体区间
gene_pr = pr.PyRanges(
    pd.DataFrame(
        {
            "Chromosome":    gene_coord_df["chrom"],
            "Start":         gene_coord_df["start"],
            "End":           gene_coord_df["end"],
            "upstream_gene": gene_coord_df["gene_symbol"],
        }
    )
)

# SNP 点区间（End = Start + 1）；去重后只保留唯一 (SNPChr, SNPPos, GeneSymbol)
snp_df = (
    eqtl_target[["SNPChr", "SNPPos", "GeneSymbol"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
snp_pr = pr.PyRanges(
    pd.DataFrame(
        {
            "Chromosome":  snp_df["SNPChr"].astype(str),
            "Start":       snp_df["SNPPos"],
            "End":         snp_df["SNPPos"] + 1,
            "target_gene": snp_df["GeneSymbol"],
        }
    )
)

# 区间 join：每个 SNP 落在哪些基因体内
overlaps    = snp_pr.join(gene_pr, how="left")
overlap_df  = overlaps.as_df()[["target_gene", "upstream_gene"]].dropna()

# 排除 SNP 落在目标基因自身基因体内的情况
overlap_df = overlap_df[overlap_df["target_gene"] != overlap_df["upstream_gene"]]

# ── 5. 聚合并输出 CSV ─────────────────────────────────────────────────────────
print("[5/5] 聚合结果并写出 CSV...")

grouped = (
    overlap_df
    .groupby("target_gene")["upstream_gene"]
    .apply(lambda x: ",".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"upstream_gene": "upstream_genes"})
)

# 补全没有找到上游基因的目标基因（upstream_genes 留空）
all_target_df = pd.DataFrame({"target_gene": sorted(target_genes)})
result_df     = all_target_df.merge(grouped, on="target_gene", how="left")
result_df["upstream_genes"]      = result_df["upstream_genes"].fillna("")
result_df["upstream_gene_count"] = result_df["upstream_genes"].apply(
    lambda x: len(x.split(",")) if x else 0
)

result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n完成！结果已保存至: {OUTPUT_CSV}")
print(f"统计: {len(result_df)} 个目标基因")
print(f"      {(result_df['upstream_gene_count'] > 0).sum()} 个找到上游基因")
print(f"      {(result_df['upstream_gene_count'] == 0).sum()} 个无匹配（eQTL SNP落在基因间区）")
print(result_df[result_df["upstream_gene_count"] > 0].head(10).to_string(index=False))
