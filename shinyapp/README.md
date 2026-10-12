# Network apps

Use `analysis_scripts/export_network.py` to produce complete application folders. The source templates here intentionally contain no patient data, p-value annotations or trained-model results. The exported `single_group` and `inter` folders are independent apps.

```r
install.packages(c("shiny", "dplyr", "igraph"))
shiny::runApp("/path/to/network/shiny/single_group")
# or: shiny::runApp("/path/to/network/shiny/inter")
```

Each generated app includes:

- `analysis/gigtransformer-rownorm/<class>_layer_norm_average_fold_node_weight_df.csv`: `Node_idx,Weight`.
- Matching `<class>_layer_norm_average_fold_gene_edge_weight_df.csv`: `Actual_From,Actual_To,Weight`.
- `data/filtered_data/gene_num_dict_df.csv`: `gene_node_idx,gene_node_name`.
- `pvalues_output/TvsNO_min_pvalues.csv`: `gene,t2ds_no_t2ds_pvalue`.
- `pvalues_output/PrevsNO_min_pvalues.csv`: `gene,pret2ds_no_t2ds_pvalue`.
- Header-only `t6.txt` with `Locus`; no external-evidence genes are invented. Supply genuine evidence separately if that marking is needed.

Class filenames are `t2ds`, `pret2ds`, `no_t2ds`. Node scores retain the sign of the model-input RNA. Existing p-values are external annotations, not significance tests of the new model scores. Original ranking, edge thresholds, component-size filters, p-value coloring and CSV exports are retained. The UI's existing p-value controls mainly mark evidence; they should not be described as a new gene-score significance test or an automatic optimized feature selector.

Unused imports of networkD3, kdensity, ggplot2 and Jmisc were removed; the server now explicitly accepts `session`. Plot/filter code and control defaults otherwise match the current original-style package. For small graphs, the default minimum component size can filter out everything; use the existing component-size control. No automatic threshold tuning or claims of biological importance are added.
