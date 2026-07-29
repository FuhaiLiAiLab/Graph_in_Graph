"""
generate_pheno_feature_tables.py
=================================
生成三张表格：Top 20 phenotypic features for t2ds / pret2ds / no_t2ds patients。

数据来源：
  - analysis/gigtransformer/fold_{1-5}/pheno_{first,block,last}_patient_edge_weight.csv
  - data/filtered_data/label_phenodata_onehot_nodeidx_df.csv
  - data/filtered_data/subfeature_dict_df.csv

方法：
  1. 读取 pheno attention weights（from=患者节点 >=122, to=subfeature节点 0-121）
  2. 合并全部 5 fold × 3 层，按 (patient_type, to_node_idx) 取均值
  3. 按 Weight 降序取 Top 20，映射 subfeature 名称并提取 Group Name

运行：
  python generate_pheno_feature_tables.py
"""

import os
import pandas as pd


# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBFEATURE_FILE = os.path.join(BASE_DIR, 'data', 'filtered_data', 'subfeature_dict_df.csv')
LABEL_FILE      = os.path.join(BASE_DIR, 'data', 'filtered_data', 'label_phenodata_onehot_nodeidx_df.csv')
ANALYSIS_DIR    = os.path.join(BASE_DIR, 'analysis', 'gigtransformer')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'output')

NUM_FOLDS  = 5
LAYERS     = ['first', 'block', 'last']
TOP_N      = 20
PATIENT_TYPES = ['t2ds', 'pret2ds', 'no_t2ds']


# ── Step 1: 加载映射表 ────────────────────────────────────────────────────────
def load_mappings():
    # subfeature idx → name
    sf_df = pd.read_csv(SUBFEATURE_FILE)
    sf_map = dict(zip(sf_df['subfeature_node_idx'], sf_df['subfeature_names']))

    # patient node_idx → patient_type
    lb_df = pd.read_csv(LABEL_FILE)
    node_to_type = {}
    for ptype in PATIENT_TYPES:
        for nidx in lb_df.loc[lb_df[ptype] == 1, 'node_idx']:
            node_to_type[nidx] = ptype

    return sf_map, node_to_type


# ── Step 2-5: 读取并合并所有 pheno attention weights ─────────────────────────
def load_all_weights(node_to_type):
    records = []
    for fold in range(1, NUM_FOLDS + 1):
        for layer in LAYERS:
            path = os.path.join(
                ANALYSIS_DIR, f'fold_{fold}',
                f'pheno_{layer}_patient_edge_weight.csv'
            )
            df = pd.read_csv(path)

            # 只保留 from = 患者节点 (node_idx >= 122)
            df = df[df['from_node_idx'] >= 122].copy()

            # 打上 patient_type 标签
            df['patient_type'] = df['from_node_idx'].map(node_to_type)

            # 丢弃不在 label 文件里的节点（理论上不应有）
            df = df.dropna(subset=['patient_type'])

            records.append(df[['patient_type', 'to_node_idx', 'Weight']])

    all_df = pd.concat(records, ignore_index=True)
    return all_df


# ── Step 6: 按 (patient_type, to_node_idx) 取均值 ────────────────────────────
def compute_mean_weights(all_df):
    mean_df = (
        all_df
        .groupby(['patient_type', 'to_node_idx'], sort=False)['Weight']
        .median()
        .reset_index()
    )
    return mean_df


# ── Step 7-9: 生成 Top 20 表 ─────────────────────────────────────────────────
def build_top20(mean_df, sf_map, ptype):
    sub = mean_df[mean_df['patient_type'] == ptype].copy()

    # 映射 subfeature 名称
    sub['Subgroup Name'] = sub['to_node_idx'].map(sf_map)

    # Group Name = 去掉末尾 -1 / -2 / -3
    sub['Group Name'] = sub['Subgroup Name'].str.replace(r'-\d+$', '', regex=True)

    # 按 Weight 降序，取 Top N
    sub = sub.sort_values('Weight', ascending=False).head(TOP_N).reset_index(drop=True)
    sub.index += 1
    sub.index.name = 'Rank'

    return sub[['Group Name', 'Subgroup Name', 'Weight']].reset_index()


# ── 打印表格 ──────────────────────────────────────────────────────────────────
def print_table(df, title):
    print(f'\n{title}')
    print('-' * 70)
    header = f"{'Rank':>4}  {'Group Name':<28}  {'Subgroup Name':<30}  {'Weight'}"
    print(header)
    print('-' * 70)
    for _, row in df.iterrows():
        print(f"{row['Rank']:>4}  {row['Group Name']:<28}  {row['Subgroup Name']:<30}  {row['Weight']:.8g}")
    print()


# ── 保存 CSV ──────────────────────────────────────────────────────────────────
def save_tables(tables):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ptype, df in tables.items():
        out_path = os.path.join(OUTPUT_DIR, f'top20_pheno_{ptype}.csv')
        df.to_csv(out_path, index=False)
        print(f'Saved: {out_path}')


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print('Loading mappings...')
    sf_map, node_to_type = load_mappings()
    print(f'  Subfeatures: {len(sf_map)}, Patients labeled: {len(node_to_type)}')

    print('Loading pheno attention weights (5 folds × 3 layers)...')
    all_df = load_all_weights(node_to_type)
    print(f'  Total records: {len(all_df):,}')

    print('Computing mean weights per (patient_type, subfeature)...')
    mean_df = compute_mean_weights(all_df)

    titles = {
        't2ds':    'Table 1. Top 20 features for t2ds patients',
        'pret2ds': 'Table 2. Top 20 features for pre_t2ds patients',
        'no_t2ds': 'Table 3. Top 20 features for no_t2ds patients',
    }

    tables = {}
    for ptype in PATIENT_TYPES:
        tables[ptype] = build_top20(mean_df, sf_map, ptype)
        print_table(tables[ptype], titles[ptype])

    save_tables(tables)


if __name__ == '__main__':
    main()
