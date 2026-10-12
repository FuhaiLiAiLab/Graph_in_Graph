# Prepared input for Full training

Supply `--data-root` pointing to prepared, family-separated data. See PREPROCESSING.md for the raw-cohort preparation entry point. Never fit preprocessing on the validation fold.

Required root files: `config.json`, `splits/subject_folds.csv`,
`metadata/genes.csv`, `metadata/channels.csv`, `metadata/source_manifest.json`,
and `shared/gene_edge_index.npy`.

For each `folds/fold_01` through `fold_05`, both `train/` and `test/` contain
`subjects.csv` plus NumPy arrays: `subject_ids`, `labels`, `labels_onehot`,
`gene_x`, `phenotype_x`, `phenotype_bins`, `graph_x`, `edge_index`, `node_index`
(each with `.npy` suffix). Here `test` is the historical filename for the
validation fold used in model selection, not an independent final test.
`subjects.csv` supplies subject/family identity for disjointness checks.

`config.json` supplies `n_genes`, `channels`, `n_subfeature_nodes`, and
`n_phenotype_features`. Labels are T2D=0, PreT2D=1, No-T2D=2.
Gene channels are RNA followed by the five methylation channels; gene_x is
flattened gene-major (subjects × genes*channels). Graph node_index maps subjects
into graph_x; edges use the same node indexing. The fixed configuration expects
6 gene channels and 42 phenotype features. See `processed_data/scripts/load_fold.py`
for shape checks and `runtime.load_pair` for family/subject checks.

Only source code is included. Keep prepared data, patient identifiers,
labels and predictions outside version control. Local output artifacts are
excluded by `.gitignore`.
