# Graph in Graph (GiG)

Code for **"Graph in Graph (GiG): A novel graph AI framework for integrating and interpreting whole-person health/medical and omics data"**.

GiG is a dual-graph neural network that jointly models a *gene-level graph* (multi-omics: transcriptomics, methylation, eQTL-derived regulatory regions) nested inside a *patient-level graph* (clinical/phenotype features), to predict disease status and surface interpretable gene- and phenotype-level attention weights.

## Repository structure

```
enc/                Model definitions: GAT, GCN, GIN, GIG-GAT, GIG-Transformer (nn.Module classes)
geo_loader/         Graph data loading / batching utilities (torch_geometric)
utils.py            Shared helpers (GPU selection etc.), imported by scripts in every stage below

preprocessing/      Stage 1 — raw multi-omics ingestion, subject/label construction, graph-node/edge assembly
training/           Stage 2 — model training entrypoints (one per architecture)
  legacy/             superseded training script variants, kept for provenance only
hpo/                Stage 3 — Optuna hyperparameter search
  results/            logged HPO trial outputs (json/csv)
  legacy/             superseded/exploratory HPO scripts (some depend on files no longer in this repo — see note below)
analysis_scripts/   Stage 4 — post-hoc statistics on trained models (edge weights, p-values, GSEA, etc.)
  legacy/             alternate-parameter or superseded analysis variants
figures/            Stage 5 — paper figure generation (Python + R + notebooks)
  outputs/            regenerated PNGs (not source, kept for convenience)
  legacy/             precursor plotting scripts
shinyapp/           Interactive R/Shiny app for exploring the gene-phenotype network
  legacy/             earlier normalization/feature variants of the app
scripts/            Misc one-off utilities
image_storage/      Figure assets referenced by the paper
```

Anything not listed above (`data/`, `analysis/`, `gnn_result/`, `*_output/` directories, virtual envs, etc.) is either a regeneratable intermediate artifact or IRB-restricted subject-level data, and is excluded via `.gitignore` — see **Data availability** below.

## Pipeline

1. **Preprocessing** (`preprocessing/`) — parse raw LLFS clinical/omics data (`pre_stat.ipynb`), attach longevity/disease labels (`add_labels_*.py`, `add_longevity_labels.py`), and build the node/edge tables consumed by the graph loader (`post_parse.py`, `generate_pheno_feature_tables.py`). `ROSMAP_union_raw_data_process_AD.ipynb` / `UCSC_union_raw_data_process.ipynb` prepare the two external replication cohorts. See **Technical notes** below for the exact data → graph transformation steps.
2. **Training** (`training/`) — `geo_tmain_gigtransformer.py` and `geo_tmain_giggat.py` train the paper's two GiG variants; `geo_tmain_gat.py` / `_gcn.py` / `_gin.py` train the single-graph baselines used for comparison. Each is a standalone CLI script (`python training/geo_tmain_gigtransformer.py --help`).
3. **Hyperparameter search** (`hpo/`) — `optuna_giggat_v2.py` runs the GIG-GAT search. There is currently no working end-to-end HPO script for GIG-Transformer in this repo (see note below); `training/geo_tmain_gigtransformer.py` was tuned manually / via `training/train_curves.py`.
4. **Analysis** (`analysis_scripts/`) — load a trained checkpoint and compute attention-weight statistics, group comparisons, and enrichment tests (`geo_analysis_tran.py` → `edge_analysis_transformer*.py` → `calculate_pvalue*.py` / `gene_quadrant_analysis.py` / `gsea_meth_194.py`, etc.).
5. **Figures** (`figures/`) — turn analysis outputs into the paper's plots.
6. **Shiny app** (`shinyapp/`) — `shinyapp_graph-rownorm-inter-pvalue.R` is the current interactive network explorer.

### ⚠️ Known gaps (flagged during repo cleanup, not fixed)

- `hpo/legacy/hpo_scheme1_20260318.py` and `hpo/legacy/optuna_gigtransformerV3.py` import a `geo_tmain_giggat_try3` / `geo_tmain_gigtransformer_copy2` module that no longer exists anywhere in the project history available at cleanup time — both scripts are currently non-runnable and are kept only for reference.
- Everything else in `training/`, `hpo/` (non-legacy), and `analysis_scripts/` (non-legacy) was verified to import correctly after the reorganization (`python <script>.py --help` succeeds).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

`requirements.txt` was exported from the environment used to run these experiments (`pip freeze`); not every package is required for every script.

## Data availability

This repository does **not** include subject-level data. `data/`, `couples_longevity*.csv`, `subject_longevity_labels*.csv`, and related files are excluded (see `.gitignore`) because they contain Long Life Family Study (LLFS) subject-level clinical/omics data under IRB-restricted access. Researchers seeking access to LLFS data should contact the LLFS data coordinating center.

## Authors

Heming Zhang, Kaiwen Fang, Yifei Lu — FuhaiLi AI Lab

## Citation

If you use this code, please cite:

> Zhang H., Fang K., Lu Y., et al. "Graph in Graph (GiG): A novel graph AI framework for integrating and interpreting whole-person health/medical and omics data." [journal, year, DOI — fill in once available]

## License

[MIT](LICENSE) — see `LICENSE` for details.

---

## Technical notes: graph construction pipeline (original team notes)

### 1. Data Processing
#### 1.1 Multi-omics Features
* tran_v1_df ->(ensembl_data)-> merged_tran_v1_df -> filter (all_edge_gene_list) -> merged_tran_v1_df
* [core_promoter/distal_promoter/proximal_promoter/downstream/upstream]_df ->(ensembl_data)-> merged_[]_df -> filter (all_edge_gene_list) -> merged_[]_df
* Overlapping of (merged_tran_v1_df & merged_[]_df ) -> intersected_gene
* kegg_df -> up_kegg_df -> Overlapping nodes -> all_edge_gene_list

#### 1.2 Clinical Features / Labels
* phenodata_df -> subject_list
* t2ds_label_df -> t2ds_label_subject_list
* merged_tran_v1_df -> merged_tran_v1_subject_list
* merged_[]_df -> merged_[upstream]_subject_list
* Overlapping of (subject_list & t2ds_label_subject_list & merged_tran_v1_subject_list & merged_upstream_subject_list) -> intersected_subject_list

Hence,
* phenodata_df -> filter (intersected_subject_list) -> phenodata_df
* t2ds_label_df -> filter (intersected_subject_list) -> t2ds_label_df
* merged_tran_v1_df -> filter (intersected_subject_list) -> merged_tran_v1_df
* merged_[]_df -> filter (intersected_subject_list) -> merged_[]_df

Cleaning clinical data
* phenodata_df -> label_phenodata_df ->  (v1) -> v1_label_phenodata_df

### 2. Graph Construction
#### 2.1 Patient Graph Nodes
* v1_label_phenodata_df -> (onehot encoding) -> v1_label_phenodata_onehot_df
* v1_label_phenodata_onehot_df -> (10/90 percentile categorize) -> v1_label_phenodata_category_df -> subfeature_dict_df
* v1_label_phenodata_category_df -> subfeature_dict_df (phenotypes mapping)
* (subject_list / subject_name_list) -> subject_dict_df (subject mapping) -> subject_number_dict_df
* concatenate [subfeature_dict_df, subject_number_dict_df] ->  node_idx_name_map_df (all nodes mapping)

#### 2.2 Patient Graph Edges
* v1_label_phenodata_category_df -> (replacing categorized values into node num) -> v1_label_phenodata_category_name_df -> mapping with (node_idx_name_map_df) -> v1_label_phenodata_category_num_df -> v1_label_phenodata_category_num_dflist -> v1_label_phenodata_category_edge_df (this is part for constructing connections between patient subjects and corresponding categorized features)
* v1_label_phenodata_category_df / v1_label_phenodata_feature_list -> subfeature_name_edge_df -> (node_name_idx_dict) -> subfeature_num_edge_df
* concatenate [subfeature_num_edge_df, v1_label_phenodata_category_edge_df] -> num_edge_df -> edge_index

#### 2.3 Patient Graph Features
* v1_label_phenodata_onehot_df ->(subject_node_dict)-> v1_label_phenodata_onehot_nodeidx_df
* v1_label_phenodata_onehot_nodeidx_df -> x_v1_label_phenodata_onehot_nodeidx_df -> subject_phenodata_x
* np.zeros((num_subfeature, num_feature)) -> subfeature_phenodata_x
* concatenate [subfeature_phenodata_x, subject_phenodata_x] -> x

#### 2.4 Gene Graph Multi-omics Features
* merged_tran_v1_df ->(subject_node_dict)-> merged_tran_v1_nodeidx_df
* merged_[]_df ->(subject_node_dict)->  merged_[]_nodeidx_df

#### 2.5 Gene Graph Nodes
* merged_tran_v1_df -> gene_name_list -> gene_node_idx_list -> gene_name_dict -> gene_num_dict_df

#### 2.6 Gene Graph Edges
* up_kegg_df ->(gene_name_dict)-> gene_num_edge_df -> reverse_gene_num_edge_df -> gene_edge_index

#### 2.7 Gene Graph Features
* merged_tran_v1_nodeidx_df -> gene_tran_x_df -> gene_tran_x -> norm_gene_tran_x
* merged_[]_nodeidx_df -> gene_[]_x_df -> gene_[]_x -> norm_[]_x
* concatenate [gene_tran_x, gene_core_promoter_x, gene_proximal_promoter_x, gene_distal_promoter_x, gene_upstream_x, gene_downstream_x] -> gene_x
* concatenate [norm_gene_tran_x, norm_gene_core_promoter_x, norm_gene_proximal_promoter_x, norm_gene_distal_promoter_x, norm_gene_upstream_x, norm_gene_downstream_x] -> norm_gene_x

### 3. Common GNN Graph Construction
#### 3.1 Graph Node Features
* v1_label_phenodata_onehot_df -> pheno_x_df -> pheno_x -> norm_pheno_x
* geno_pheno_x = np.hstack((gene_x, pheno_x))
* fill np.zeros((num_subfeature, dim_geno_pheno_x)) -> subfeature_all_x
* all_x = np.vstack((subfeature_all_x, geno_pheno_x))   [935, 8340+42]
* norm_geno_pheno_x = np.hstack((norm_gene_x, norm_pheno_x))
* fill np.zeros((num_subfeature, dim_norm_geno_pheno_x)) -> norm_subfeature_all_x
* norm_all_x = np.vstack((norm_subfeature_all_x, norm_geno_pheno_x))

#### 3.2 Graph Edge Features
* Can Just USE: num_edge_df -> edge_index
