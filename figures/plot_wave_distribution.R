# Panel a: Wave Distribution of Features with Different Datasets (ridgeline plot)
#
# Faithful re-run of the original test.R / ggridges rendering. The only change
# from test.R is the y-axis tick labels: raw column names (e.g. "hai_rl_v1")
# are swapped for standardized clinical abbreviations from
# ./image_storage/feature_label_mapping.tsv via scale_y_discrete(labels=...).
# The underlying `feature` factor order is left completely untouched, so the
# row layout is pixel-identical to the original ggridges output.

library(ggplot2)
library(ggridges)
library(tidyr)

bar_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_t2ds_phenodata_list.csv')
bar_no_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_no_t2ds_phenodata_list.csv')
bar_pret2ds_phenodata_list <- read.csv('./data/stat_data/bar_pret2ds_phenodata_list.csv')

bar_t2ds_phenodata_list$dataset <- 'T2DS'
bar_no_t2ds_phenodata_list$dataset <- 'No T2DS'
bar_pret2ds_phenodata_list$dataset <- 'Pre-T2DS'

combined_data <- rbind(bar_t2ds_phenodata_list, bar_no_t2ds_phenodata_list, bar_pret2ds_phenodata_list)

long_data <- pivot_longer(combined_data,
                          cols = -dataset,
                          names_to = "feature",
                          values_to = "value")

# Raw (R-sanitized) column name -> standardized clinical label, display-only.
label_map <- read.delim('./image_storage/feature_label_mapping.tsv', stringsAsFactors = FALSE)
label_lookup <- setNames(label_map$standardized_label, make.names(label_map$old_abbreviation))
relabel <- function(x) unname(label_lookup[x])

ggsave(file.path("image_storage", "wave_distribution_plot.png"),
       ggplot(long_data, aes(x = value, y = feature, fill = dataset, color = dataset)) +
         geom_density_ridges(alpha = 0.2, scale = 1, rel_min_height = 0.01) +
         scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +
         scale_color_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +
         scale_y_discrete(labels = relabel) +
         theme_ridges() +
         theme(
           panel.background = element_rect(fill = "white", colour = "white"),
           plot.background = element_rect(fill = "white", colour = "white"),
           panel.grid.major = element_line(colour = "grey90"),
           panel.grid.minor = element_blank(),
           axis.text = element_text(colour = "black"),
           axis.title = element_text(colour = "black"),
           plot.title = element_text(hjust = 0.5, colour = "black")
         ) +
         labs(title = "Wave Distribution of Features with Different Datasets",
              x = "Value Distribution",
              y = "Features") +
         xlim(-1, 1),
       width = 10, height = 12, dpi = 1200
)
