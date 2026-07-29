# Install necessary libraries if not already installed
# install.packages(c("ggplot2", "ggridges", "tidyr"))

# Load required libraries
library(ggplot2)
library(ggridges)
library(tidyr)

# Read the data from CSV files
bar_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_t2ds_phenodata_list.csv')
bar_no_t2ds_phenodata_list <- read.csv('./data/stat_data/bar_no_t2ds_phenodata_list.csv')
bar_pret2ds_phenodata_list <- read.csv('./data/stat_data/bar_pret2ds_phenodata_list.csv')

# Add a column to identify each dataset
bar_t2ds_phenodata_list$dataset <- 'T2DS'
bar_no_t2ds_phenodata_list$dataset <- 'No T2DS'
bar_pret2ds_phenodata_list$dataset <- 'Pre-T2DS'

# Combine all datasets into one dataframe
combined_data <- rbind(bar_t2ds_phenodata_list, bar_no_t2ds_phenodata_list, bar_pret2ds_phenodata_list)

# Reshape data to long format for ggplot2
long_data <- pivot_longer(combined_data, 
                          cols = -dataset,       # All columns except the 'dataset' column
                          names_to = "feature",  # Create a 'feature' column for the variable names
                          values_to = "value")   # Create a 'value' column for the values

ggsave("wave_distribution_plot_v2.png", 
       ggplot(long_data, aes(x = value, y = feature, fill = dataset, color = dataset)) + 
         geom_density_ridges(alpha = 0.2, scale = 1, rel_min_height = 0.01) +  # Map both fill and color to dataset
         scale_fill_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +      # Distinct fill colors
         scale_color_manual(values = c("#1f77b4", "#ff7f0e", "#2ca02c")) +     # Match line color with region
         theme_ridges() +                                                      # Clean ridge plot theme
         theme(
           panel.background = element_rect(fill = "white", colour = "white"),   # White background
           plot.background = element_rect(fill = "white", colour = "white"),    # Plot area background
           panel.grid.major = element_line(colour = "grey90"),                  # Light grey grid lines
           panel.grid.minor = element_blank(),                                  # No minor grid lines
           axis.text = element_text(colour = "black"),                          # Black axis text
           axis.title = element_text(colour = "black"),                         # Black axis titles
           plot.title = element_text(hjust = 0.5, colour = "black")             # Centered black title
         ) +
         labs(title = "Wave Distribution of Features with Different Datasets",  # Plot title
              x = "Value Distribution",                                        # X-axis label
              y = "Features") +                                                # Y-axis label
         xlim(-1, 1)                                                           # Set x-axis limits to -1 and 1                                                 # Y-axis label
       
)
