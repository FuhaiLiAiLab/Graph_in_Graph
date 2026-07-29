# Graph in Graph (GiG)

Code for **"Graph in Graph (GiG): A novel graph AI framework for integrating and interpreting whole-person health/medical and omics data"**.

GiG is a dual-graph neural network that jointly models a *gene-level graph* (multi-omics: transcriptomics, methylation, eQTL-derived regulatory regions) nested inside a *patient-level graph* (clinical/phenotype features), to predict disease status and surface interpretable gene- and phenotype-level attention weights.

## Repository structure

```
enc/                Model definitions (GAT, GCN, GIN, GIG-GAT, GIG-Transformer)
geo_loader/         Graph data loading / batching utilities
utils.py            Shared helpers used across all stages

preprocessing/      Stage 1 — build the patient and gene graphs from raw data
training/           Stage 2 — train each model architecture
hpo/                Stage 3 — hyperparameter search
analysis_scripts/   Stage 4 — interpret a trained model's predictions
figures/            Stage 5 — generate the paper's figures
shinyapp/           Interactive app for exploring the gene-phenotype network
scripts/            Misc utilities
image_storage/      Figure assets

Each stage folder has a legacy/ subfolder holding earlier iterations, kept for provenance.
```

Data folders, model checkpoints, and other regenerated results are not included in this repository (see **Data availability**).

## Pipeline

The codebase follows a straightforward five-stage workflow:

1. **Build the graphs.** Clinical, transcriptomic, methylation, and eQTL data are cleaned, matched across subjects, and assembled into two linked graphs — a gene graph (nodes = genes, edges = regulatory/pathway relationships) nested inside a patient graph (nodes = subjects and phenotype features).
2. **Train a model.** Each script in `training/` trains one architecture (GIG-Transformer and GIG-GAT are the paper's dual-graph models; GAT/GCN/GIN are single-graph baselines used for comparison).
3. **Tune hyperparameters** (optional). Scripts in `hpo/` run Optuna searches over model/training settings.
4. **Interpret the model.** Scripts in `analysis_scripts/` load a trained checkpoint and extract attention/edge weights, then run statistical comparisons (p-values, enrichment tests) across patient groups.
5. **Generate figures.** Scripts and notebooks in `figures/` turn the analysis output into the paper's plots; `shinyapp/` provides an interactive alternative for exploring the network.

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
