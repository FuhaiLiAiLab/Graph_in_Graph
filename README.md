# Graph in Graph (GiG)

Current code for **Graph in Graph (GiG): A novel graph AI framework for integrating and interpreting medical and omics data**. This revision replaces the earlier training and analysis entry points with family-separated evaluation, fixed configurations, current GiG/GAT/GIN implementations, and RNA-weighted network interpretation.

## Installation

Linux, Python 3.10:

```bash
conda env create -f environment/environment.yml
conda activate llfs-gig-revision
bash environment/install_cpu.sh       # CPU
# OR: bash environment/install_cuda121.sh  # compatible CUDA 12.1 host
python environment/check_environment.py --device cpu
```

Run compute commands on allocated compute nodes when using Slurm. GPU IDs are taken from the environment; scripts do not request allocations or overwrite CUDA visibility. The provided wheels are a fixed tested software stack, not an automatic match for every machine.

## Models and fixed parameters

| Model | Configuration | Inputs and loss |
|---|---|---|
| GiG Full | `configs/full.json` | Phenotype + RNA + methylation; class-weighted CE and three auxiliary losses |
| Phenotype-only | `configs/ablation.json` | Phenotype; ordinary unweighted CE, no auxiliary losses |
| RNA+methylation (`gene_only`) | `configs/ablation.json` | Both gene modalities; ordinary unweighted CE, no auxiliary losses |
| RNA-only | `configs/ablation.json` | RNA; ordinary unweighted CE, no auxiliary losses |
| Methylation-only | `configs/ablation.json` | Methylation; ordinary unweighted CE, no auxiliary losses |
| GAT | `configs/gat.json` | Joint inputs, train-fitted PCA64 for omics, patient-input residual; weighted CE |
| GIN | `configs/gin.json` | Joint inputs, train-fitted PCA64, first-MLP LayerNorm and first-layer patient residual; weighted CE |

GiG Full uses K=24, four gene layers and two patient layers with bidirectional patient/category edges. GAT and GIN are the updated independent baselines, not unchanged original architectures or replacements of the Full encoder. The modality ablations also differ from Full in loss, so they are not strictly loss-matched comparisons.

Only fixed parameter sets are distributed. No hyperparameter search code, search history, model-score tables, or report aggregation pipeline is included.

## Data preparation

Subject-level LLFS data are not distributed. Researchers must obtain appropriate data access through the LLFS data coordinating center. See [data format](docs/DATA_FORMAT.md) and [raw preparation](docs/PREPROCESSING.md). Training and validation subjects/families must be disjoint; preprocessing is fit on the training fold and reused on validation.

## Training

```bash
# Five folds × the five specified seeds for Full
python training/geo_tmain_gigtransformer.py --data-root /path/to/prepared --output runs/current --device cuda

# Four modality ablations, each five folds × five seeds
python training/run_ablation.py --data-root /path/to/prepared --output runs/current --device cuda

# Updated independent baselines
python training/geo_tmain_gat.py --data-root /path/to/prepared --output runs/current --device cuda
python training/geo_tmain_gin.py --data-root /path/to/prepared --output runs/current --device cuda

# One task, or all seven models (175 fits)
python training/run.py --models full --seeds 2026 --folds 1 --data-root /path/to/prepared --output runs/current --device cpu
python training/run_5x5.py --data-root /path/to/prepared --output runs/current --device cuda
```

Use a separate output root for each experiment. Repeat an identical command with `--resume` after interruption; optimizer, scheduler and random-state contracts are checked. Use `--task-index N` to assign distinct zero-based tasks to independently allocated workers. This does not launch a search or choose seeds by their performance. The default run uses 500 epochs and no early stopping. `--epochs` is available for explicitly shortened software checks.

The model, seed and fold are part of every output path. Training writes local per-fold histories, metrics, predictions and checkpoints; these are ignored by Git. **Use `rule_balanced_guard.pt`**: validation selection score is `(macro-F1 + macro-AUPRC)/2 + 0.5*BA`. It is not a hard BA threshold. For GiG, `selected_predictions.csv` and `selected_checkpoint.json` are re-evaluated from that checkpoint. Legacy `best_model.pt`/`summary.json`/`test_predictions.csv` in GiG runs refer to the separately tracked BA checkpoint. For GAT/GIN, `test_predictions.csv` and `summary.json.validation` already refer to the guard checkpoint.

The default seed list is explicit in `configs/evaluation.json`. It was selected during earlier Full development, and one seed was used for configuration selection; these repeated development folds are **not an independent final test**. Shared validation graphs also permit interactions within a validation cohort. Identical results across different hardware/software are not promised. Historical checkpoint source fingerprints should not be bypassed to resume under changed code.

## Node and edge interpretation

```bash
python analysis_scripts/analyze.py --checkpoint /path/to/rule_balanced_guard.pt --data-root /path/to/prepared --output runs/attention --device cpu
python analysis_scripts/export_network.py --manifest /path/to/checkpoints.json --data-root /path/to/prepared --pvalues /path/to/annotations --output runs/network --device cuda
```

The network manifest is a JSON list of `{ "seed": 2026, "fold": 1, "checkpoint": "relative/or/absolute/checkpoint.pt" }` records. Supply complete five-fold grids for at least two seeds (five for the reported protocol). Relative checkpoint paths are resolved against the manifest directory. Network export verifies checkpoint identity, configuration consistency, validation cohort coverage and unchanged logits under interpretation hooks.

The node score is `L2(last gene representation) × L2(assignment column) × signed model-input RNA`. Negative values are retained. Class means are averaged equally over folds and seeds; this is aggregation of network weights, not predictive-score reporting. Gene edges use row-normalized attention with the original outgoing-degree convention. Raw attention can tie because of topology/normalization; zero incoming edges can produce zero representations. Neither a high attention score nor a zero node score establishes biological relevance or irrelevance.

The exporter builds independent `single_group/` and `inter/` Shiny folders. See [Shiny instructions](shinyapp/README.md). Existing p-value annotations are copied unchanged and do not test the new model scores. External-evidence orange marking defaults to an empty `t6.txt` Locus table. The original sliders, filtering, sorting and plot/export behavior are retained; only unused R imports and the standard server function signature were cleaned up.

## Tests

```bash
python -m pytest -q tests processed_data/scripts/test_preprocessing.py
```

Tests use synthetic data. They cover preprocessing, modalities, checkpoint recovery, fixed baseline PCA, selected-checkpoint export and RNA-score behavior. No real cohort or trained checkpoint is bundled. Test outcomes and development audit logs are maintained outside this repository.

## Repository layout

`enc/`, `geo_loader/`: models and graph loading; root Python modules: shared runtime, training and interpretation; `training/`: fixed-configuration CLI; `preprocessing/`, `processed_data/scripts/`: raw preparation and prepared-input loading; `configs/`: fixed parameters; `analysis_scripts/`: interpretation CLI; `shinyapp/`: two apps; `environment/`, `tests/`, `docs/`: setup and documentation.

Historical scripts, notebooks, generated figures and search output are available in Git history rather than shipped as a second implementation. [Change notes](docs/CHANGELOG.md).

## Citation and license

Zhang, H., Lu, Y., Fang, K., Xu, Z., Akabry Moghaddam, V., An, P., Jin, S., Wojczynski, M., Province, M., & Li, F. (2026). Graph in Graph (GiG): A novel graph AI framework for integrating and interpreting medical and omics data. *bioRxiv*. https://doi.org/10.64898/2026.07.20.739657

[MIT license](LICENSE).
