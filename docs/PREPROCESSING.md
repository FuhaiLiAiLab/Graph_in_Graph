# Raw-data preparation

The preparation implementation preserves the current cohort's data alignment, family grouping, feature order and train-only transformations. It expects the source cohort schema and its 1,575-ID input list; it is not a general-purpose converter for arbitrary cohorts. Missing disease labels are excluded. Existing RNA residualization and upstream official phenotype transformations are inherited, with their original fit scope not established by this code.

```bash
export GIG_RAW_ROOT=/path/to/raw_bundle
export GIG_DATA_ROOT=/path/to/new_prepared_output
export GIG_SOURCE_MAP=/path/to/source_map.json
python preprocessing/prepare_data.py
python preprocessing/validate_data.py
```

Raw bundle paths:

- `subject_id_1575.csv`, `pedigree/triplet_visit3.csv`
- `pheno/LLFS_phenos_21JUN2022.xlsx` (Phenodata worksheet)
- `rna/RNA_seq_residuals_v3_allsubjects.csv`
- `methylation/Core_Promoter_final.csv`, `Proximal_Promoter_final.csv`, `Distal_Promoter_final.csv`, `Upstream_final.csv`, `Downstream_final.csv`

The external source map is a JSON object mapping these logical keys to local input paths:

```json
{
  "data/label_data/t2dpret2d.txt": "/path/to/labels.txt",
  "data/kg_data/ensembl/mart_export_genename.txt": "/path/to/gene_mapping.txt",
  "data/kg_data/KEGG/full_kegg_pathway_list.csv": "/path/to/gene_edges.csv"
}
```

`configs/phenotype_features.json` stores the exact feature order and exclusion names; it replaces reading an executed notebook's output. No individual data are included. Local provenance records contain source paths and hashes and must not be published without review.

Preparation uses five family-grouped folds, reconciles recorded family/parent links, fits imputation/scaling/binning on training subjects, then applies those fits to validation. Initial graphs retain the prepared directed convention; the current fixed model applies its bidirectional patient/category policy during loading. Changing configuration does not justify editing prepared data in place. Reuse already validated folds when reproducing existing experiments; do not regenerate a different split silently.
