library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)
library(Jmisc)


ui <- fluidPage(
  # titlePanel('Whole Network Interaction'),
  sidebarLayout(
    sidebarPanel(
      selectInput('diabetes_type', label = 'Selection of the type of diabetes',
                  choices = list('T2ds' = 1,
                                 'Pret2ds' = 2,
                                 'No_t2ds' = 3),
                  selected = 1),

      sliderInput('edge_threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0, max = 1.0,
                  value = 0.05),

      sliderInput('marking_edge_threshold',
                  'Select the threshold of marking important edge weight',
                  min = 0.1, max = 1.0,
                  value = 0.16),

      sliderInput('pvalue_threshold',
                  'Select the threshold of marking important genes by p-values',
                  min = 0, max = 0.3,
                  value = 0.05),

      sliderInput('giant_comp_threshold',
                  'Select the threshold of each component',
                  min = 0.0, max = 20.0,
                  value = 20.0),

      sliderInput('gene_node_size',
                  'Select the gene node size',
                  min = 2, max = 10,
                  value = 5.0),

      sliderInput('imgene_node_size',
                  'Select the important gene node size',
                  min = 5, max = 10,
                  value = 8.0),

      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.2, max = 1.0,
                  value = 0.6),

      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.4, max = 1.5,
                  value = 0.8),
    ),
    mainPanel(
      plotOutput(outputId = 'network', height = 1150, width = 1150)
    )
  )
)

server <- function(input, output) {
  edge_threshold <- reactive({
    input$edge_threshold
  })
  marking_edge_threshold <- reactive({
    input$marking_edge_threshold
  })
  pvalue_threshold <- reactive({
    input$pvalue_threshold
  })
  giant_comp_threshold <- reactive({
    input$giant_comp_threshold
  })

  t6_data <- read.table("t6.txt", sep = "\t", header = TRUE)
  t6_gene_list <- t6_data$Locus

  output$network <- renderPlot({
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$diabetes_type)
    if (input$diabetes_type == 1){
      type = 't2ds'
    } else if (input$diabetes_type == 2){
      type = 'pret2ds'
    } else if (input$diabetes_type == 3){
      type = 'no_t2ds'
    }
    type_path = paste('./analysis/gigtransformer-rownorm/', as.character(type), sep='')
    edge_path = paste(type_path, '_layer_norm_average_fold_gene_edge_weight_df.csv', sep='')
    net_edge_weight = read.csv(edge_path)
    all_net_node = read.csv('./data/filtered_data/gene_num_dict_df.csv') # NODE LABEL
    node_path = paste(type_path, '_layer_norm_average_fold_node_weight_df.csv', sep='')
    type_net_node = read.csv(node_path)
    net_node = merge(x = all_net_node, y = type_net_node, by.x = c('gene_node_idx'), by.y =c('Node_idx'))

    ### 2.1 FILTER EDGE BY [edge_weight]
    filter_net_edge = filter(net_edge_weight, Weight > edge_threshold())
    filter_net_edge_node = unique(c(filter_net_edge$Actual_From, filter_net_edge$Actual_To))
    filter_net_node = net_node[net_node$gene_node_idx %in% filter_net_edge_node, ]
    print(filter_net_node)
    print(filter_net_edge)

    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
    all_components = groups(components(tmp_net))
    # COLLECT ALL LARGE COMPONENTS
    giant_comp_node = c()
    for (x in 1:length(all_components)){
      each_comp = all_components[[x]]
      if (length(each_comp) >= giant_comp_threshold()){
        giant_comp_node = c(giant_comp_node, each_comp)
      }
    }

    refilter_net_edge <- subset(filter_net_edge, (Actual_From %in% giant_comp_node | Actual_To %in% giant_comp_node))
    refilter_net_edge_node = unique(c(refilter_net_edge$Actual_From, refilter_net_edge$Actual_To))
    refilter_net_node = filter_net_node[filter_net_node$gene_node_idx %in% refilter_net_edge_node,]

    print('Number of edges')
    print(nrow(refilter_net_edge))
    print('Number of nodes')
    print(nrow(refilter_net_node))

    sorted_refilter_net_node <- refilter_net_node[order(refilter_net_node$Weight, decreasing = TRUE), ]
    refilter_edge_path = paste(type_path, '_norm_refilter_edge_weight_df.csv', sep='')
    write.csv(refilter_net_edge, refilter_edge_path)
    refilter_node_path = paste(type_path, '_norm_refilter_node_weight_df.csv', sep='')
    write.csv(sorted_refilter_net_node, refilter_node_path)

    ### 3.3 LOAD PRE-CALCULATED P-VALUES
    t2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/TvsNO_min_pvalues.csv')
    pret2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/PrevsNO_min_pvalues.csv')

    t2ds_no_t2ds_pvalue <- t2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)
    pret2ds_no_t2ds_pvalue <- pret2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)

    sorted_refilter_net_node <- left_join(sorted_refilter_net_node, t2ds_no_t2ds_pvalue, by = "gene_node_name")
    sorted_refilter_net_node <- left_join(sorted_refilter_net_node, pret2ds_no_t2ds_pvalue, by = "gene_node_name")
    print(colnames(sorted_refilter_net_node))

    net = graph_from_data_frame(d=refilter_net_edge, vertices=sorted_refilter_net_node, directed=F)

    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color (orange = in t6 gene list)
    vertex_fcol = rep(NA, vcount(net))

    # vertex color: 4-color scheme by p-value significance
    vertex_col = rep('#A6CEE3', vcount(net))  # blue: no significant p-value
    vertex_col[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold()] = '#B2DF8A'   # green: T2D vs No_T2D sig
    vertex_col[V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = '#CAB2D6' # purple: Pre_T2D vs No_T2D sig
    vertex_col[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold() &
               V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = '#FB9A99' # pink: both sig

    # vertex size: p-value driven
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_node_size
    vertex_size[V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_node_size
    vertex_size[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold() &
                V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_node_size

    # vertex label size: p-value driven
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_label_size
    vertex_cex[V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_label_size
    vertex_cex[V(net)$t2ds_no_t2ds_pvalue <= pvalue_threshold() &
               V(net)$pret2ds_no_t2ds_pvalue <= pvalue_threshold()] = input$imgene_label_size

    # Loop through all nodes: t6 gene list → orange frame + force large size
    for (i in 1:vcount(net)) {
      gene <- V(net)$gene_node_name[i]
      if (!is.na(gene)) {
        if (gene %in% t6_gene_list) {
          vertex_fcol[i] <- '#FF7b00'
          vertex_size[i] <- input$imgene_node_size
        }
      } else {
        print(paste("Missing gene name at node", i))
      }
    }

    # edge width: normalize weights to a bounded range to ensure thin edges remain visible
    edge_weight_vals = E(net)$Weight
    w_min <- min(edge_weight_vals)
    w_max <- max(edge_weight_vals)
    above_thresh = edge_weight_vals >= marking_edge_threshold()

    if (w_max > w_min) {
      norm_w <- (edge_weight_vals - w_min) / (w_max - w_min)
    } else {
      norm_w <- rep(0.5, length(edge_weight_vals))
    }
    # normal edges: width in [1.0, 5.0]; important edges: width in [3.0, 10.0]
    edge_width <- 1.0 + norm_w * (5.0 - 1.0)
    edge_width[above_thresh] <- 3.0 + norm_w[above_thresh] * (10.0 - 3.0)

    # edge color: gray gradient for normal edges, black gradient for important edges
    # alpha in [0.35, 0.85] so even the lightest edges remain visible
    edge_alpha <- 0.35 + norm_w * (0.85 - 0.35)
    edge_color <- rgb(0.55, 0.55, 0.55, alpha = edge_alpha)
    edge_color[above_thresh] <- rgb(0, 0, 0, alpha = edge_alpha[above_thresh])

    set.seed(18)
    plot(net,
         vertex.frame.width = 4,
         vertex.frame.color = vertex_fcol,
         vertex.color = vertex_col,
         vertex.size = vertex_size,
         vertex.label = V(net)$gene_node_name,
         vertex.label.color = 'black',
         vertex.label.cex = vertex_cex,
         edge.width = edge_width,
         edge.color = edge_color,
         edge.curved = 0.2,
         layout=layout_nicely)

    ### ADD LEGEND
    legend(x=-1.05, y=1.13,
           legend=c('Genes',
                    'T2D Significant Genes (p-value)',
                    'Pre_T2D Significant Genes (p-value)',
                    'T2D and Pre_T2D Significant Genes',
                    'Genes overlapped with genome-wide significant evidence'),
           pch=c(21, 21, 21, 21, 21),
           col=c('#A6CEE3', '#B2DF8A', '#CAB2D6', '#FB9A99', '#FF7b00'),
           pt.bg=c('#A6CEE3', '#B2DF8A', '#CAB2D6', '#FB9A99', 'white'),
           pt.lwd=c(1, 1, 1, 1, 4),
           pt.cex=2, cex=1.2, bty='n')
    legend(x=-1.06, y=0.92,
           legend=c('Gene-Gene', 'Important Gene-Gene'),
           col=c('gray', 'black'), lwd=c(2, 6), cex=1.2, bty='n')
  })
}

# layout=layout_with_graphopt
# layout=layout_with_sugiyama
# layout=layout_with_lgl
# layout = layout.random
# layout=layout_nicely
# layout=layout_as_tree
# layout_with_kk
# layout=layout_with_dh
# layout=layout_with_gem

shinyApp(ui = ui, server = server)
