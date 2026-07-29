library(dplyr)
library(igraph)

# Default parameter values (matching Shiny UI defaults)
diabetes_type     <- 1       # T2ds
type_comparison   <- 2       # T2ds and No_t2ds
edge_threshold    <- 0.05
marking_edge_threshold <- 0.16
node_threshold    <- 0.3
pvalue_threshold  <- 0.05
giant_comp_threshold <- 20.0
gene_node_size    <- 5.0
imgene_node_size  <- 8.0
gene_label_size   <- 0.6
imgene_label_size <- 0.8

# Resolve repo root relative to this script's location (works with `Rscript`;
# falls back to the current working directory when sourced interactively --
# in that case, set the working directory to the repo root first).
.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("--file=", "", .args[grep("--file=", .args)])
.script_dir <- if (length(.script_path) > 0) dirname(normalizePath(.script_path)) else getwd()
setwd(dirname(.script_dir))

t6_data <- read.table("t6.txt", sep = "\t", header = TRUE)
t6_gene_list <- t6_data$Locus

if (diabetes_type == 1) {
  type <- 't2ds'
} else if (diabetes_type == 2) {
  type <- 'pret2ds'
} else {
  type <- 'no_t2ds'
}

type_path  <- paste('./analysis/gigtransformer-rownorm/', type, sep = '')
edge_path  <- paste(type_path, '_layer_norm_average_fold_gene_edge_weight_df.csv', sep = '')
node_path  <- paste(type_path, '_layer_norm_average_fold_node_weight_df.csv', sep = '')

net_edge_weight <- read.csv(edge_path)
all_net_node    <- read.csv('./data/filtered_data/gene_num_dict_df.csv')
type_net_node   <- read.csv(node_path)
net_node <- merge(x = all_net_node, y = type_net_node,
                  by.x = 'gene_node_idx', by.y = 'Node_idx')

filter_net_edge      <- filter(net_edge_weight, Weight > edge_threshold)
filter_net_edge_node <- unique(c(filter_net_edge$Actual_From, filter_net_edge$Actual_To))
filter_net_node      <- net_node[net_node$gene_node_idx %in% filter_net_edge_node, ]

tmp_net        <- graph_from_data_frame(d = filter_net_edge, vertices = filter_net_node, directed = FALSE)
all_components <- groups(components(tmp_net))

giant_comp_node <- c()
for (x in seq_along(all_components)) {
  each_comp <- all_components[[x]]
  if (length(each_comp) >= giant_comp_threshold) {
    giant_comp_node <- c(giant_comp_node, each_comp)
  }
}

refilter_net_edge      <- subset(filter_net_edge, (Actual_From %in% giant_comp_node | Actual_To %in% giant_comp_node))
refilter_net_edge_node <- unique(c(refilter_net_edge$Actual_From, refilter_net_edge$Actual_To))
refilter_net_node      <- filter_net_node[filter_net_node$gene_node_idx %in% refilter_net_edge_node, ]

sorted_refilter_net_node <- refilter_net_node[order(refilter_net_node$Weight, decreasing = TRUE), ]

t2ds_no_t2ds_pvalue    <- read.csv('./pvalues_output/TvsNO_min_pvalues.csv')
pret2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/PrevsNO_min_pvalues.csv')

t2ds_no_t2ds_pvalue    <- t2ds_no_t2ds_pvalue    %>% rename(gene_node_name = gene)
pret2ds_no_t2ds_pvalue <- pret2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)

sorted_refilter_net_node <- left_join(sorted_refilter_net_node, t2ds_no_t2ds_pvalue,    by = "gene_node_name")
sorted_refilter_net_node <- left_join(sorted_refilter_net_node, pret2ds_no_t2ds_pvalue, by = "gene_node_name")

net <- graph_from_data_frame(d = refilter_net_edge, vertices = sorted_refilter_net_node, directed = FALSE)

vertex_fcol <- rep(NA, vcount(net))
vertex_col  <- rep('lightblue', vcount(net))

if (type_comparison == 1) {
  vertex_col[V(net)$t2ds_pret2ds_test_result <= pvalue_threshold] <- '#FB9A99'
} else if (type_comparison == 2) {
  vertex_col[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold] <- '#FB9A99'
} else {
  vertex_col[V(net)$pret2ds_no_t2ds_test_result <= pvalue_threshold] <- '#FB9A99'
}

vertex_size <- rep(gene_node_size,    vcount(net))
vertex_cex  <- rep(gene_label_size,   vcount(net))
vertex_size[V(net)$Weight >= node_threshold] <- imgene_node_size
vertex_cex [V(net)$Weight >= node_threshold] <- imgene_label_size

for (i in seq_len(vcount(net))) {
  gene <- V(net)$gene_node_name[i]
  if (!is.na(gene) && gene %in% t6_gene_list) {
    vertex_fcol[i] <- '#FF7b00'
  }
}

edge_width <- E(net)$Weight * 15.0
edge_width[E(net)$Weight >= marking_edge_threshold] <- E(net)$Weight[E(net)$Weight >= marking_edge_threshold] * 10.0
edge_color <- rep('#666666', ecount(net))
edge_color[E(net)$Weight >= marking_edge_threshold] <- 'black'

# Save as high-resolution PNG (300 dpi, 4600x4600 px ≈ 15.3 inch square at 300 dpi)
out_path <- "image_storage/figure2b.png"
png(out_path, width = round(1150 * 1200/72), height = round(1150 * 1200/72), res = 1200)

set.seed(18)
plot(net,
     vertex.frame.width  = 4,
     vertex.frame.color  = vertex_fcol,
     vertex.color        = vertex_col,
     vertex.size         = vertex_size,
     vertex.label        = V(net)$gene_node_name,
     vertex.label.color  = 'black',
     vertex.label.cex    = vertex_cex,
     edge.width          = edge_width,
     edge.color          = edge_color,
     edge.curved         = 0.2,
     layout              = layout_nicely)

legend(x = -1.05, y = 1.13,
       legend = c('Genes',
                  'Important Genes with P value < 0.1',
                  'Genes overlapped with genome-wide significant evidence'),
       pch   = c(21, 21, 21),
       col   = c('lightblue', '#FB9A99', '#FF7b00'),
       pt.bg = c('lightblue', '#FB9A99', 'white'),
       pt.lwd = c(1, 1, 4),
       pt.cex = 2, cex = 1.2, bty = 'n')

legend(x = -1.06, y = 0.98,
       legend = c('Gene-Gene', 'Important Gene-Gene'),
       col  = c('gray', 'black'),
       lwd  = c(5, 7),
       cex  = 1.2, bty = 'n')

dev.off()

cat("Saved:", out_path, "\n")
