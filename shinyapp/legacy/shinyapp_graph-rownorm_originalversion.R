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
      
      selectInput('type_comparison', label = 'Selection of the type comparisons',
                  choices = list('T2ds and Pret2ds' = 1,
                                 'T2ds and No_t2ds' = 2,
                                 'Pret2ds and No_t2ds' = 3),
                  selected = 2),
      
      sliderInput('edge_threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0, max = 1.0,
                  value = 0.05),
      
      sliderInput('marking_edge_threshold',
                  'Select the threshold of marking important edge weight',
                  min = 0.1, max = 1.0,
                  value = 0.16),

      sliderInput('node_threshold',
                  'Select the threshold of marking important genes',
                  min = 0, max = 1.0,
                  value = 0.3),
      
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
                  min = 5, max = 10,
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
  node_threshold <- reactive({
    input$node_threshold
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
    
    # # # ### 2.1 FILTER AGAIN NODE BY [node_weight]
    # filter_net_node = filter(filter_net_node, Weight > node_threshold())
    # filter_net_edge = filter_net_edge[filter_net_edge$Actual_From %in% filter_net_node$gene_node_idx & filter_net_edge$Actual_To %in% filter_net_node$gene_node_idx, ]
    print(filter_net_node)
    print(filter_net_edge)
    
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
    # tmp_net = graph_from_data_frame(d=hop_net_edge, vertices=net_node, directed=F)
    all_components = groups(components(tmp_net))
    # COLLECT ALL LARGE COMPONENTS
    giant_comp_node = c()
    for (x in 1:length(all_components)){
      each_comp = all_components[[x]]
      if (length(each_comp) >= giant_comp_threshold()){
        giant_comp_node = c(giant_comp_node, each_comp)
      }
    }
    
    refilter_net_edge<-subset(filter_net_edge, (Actual_From %in% giant_comp_node | Actual_To %in% giant_comp_node))
    refilter_net_edge_node = unique(c(refilter_net_edge$Actual_From, refilter_net_edge$Actual_To))
    refilter_net_node = filter_net_node[filter_net_node$gene_node_idx %in% refilter_net_edge_node,]
    
    
    # 3.2 USE REWEIGHTED IDF NODE DEGREE AS NODE DEGREE
    print(nrow(refilter_net_edge))
    print(nrow(refilter_net_node))
    sorted_refilter_net_node <- refilter_net_node[order(refilter_net_node$Weight, decreasing = TRUE), ]
    refilter_edge_path = paste(type_path, '_norm_refilter_edge_weight_df.csv', sep='')
    write.csv(refilter_net_edge, refilter_edge_path)
    refilter_node_path = paste(type_path, '_norm_refilter_node_weight_df.csv', sep='')
    write.csv(sorted_refilter_net_node, refilter_node_path)
    
    # # 3.3 SELECT AND CALCULATE P-VALUE
    # subject_nodeidx_gene_df = read.csv('./data/filtered_data/merged_tran_v1_nodeidx_df.csv')
    # label_patient_nodeidx_df = read.csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
    # t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$t2ds == 1, ]
    # pret2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$pret2ds == 1, ]
    # no_t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$no_t2ds == 1, ]
    # 
    # t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% t2ds_nodeidx_df$node_idx, ]
    # pret2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% pret2ds_nodeidx_df$node_idx, ]
    # no_t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% no_t2ds_nodeidx_df$node_idx, ]
    # 
    # for(i in 1:nrow(sorted_refilter_net_node)) {
    #   gene_name <- sorted_refilter_net_node[i, 'gene_node_name']
    #   t2ds_gene_value_list <- t2ds_subject_nodeidx_gene_df[[gene_name]]
    #   pret2ds_gene_value_list <- pret2ds_subject_nodeidx_gene_df[[gene_name]]
    #   no_t2ds_gene_value_list <- no_t2ds_subject_nodeidx_gene_df[[gene_name]]
    # 
    #   t2ds_pret2ds_test_result <- wilcox.test(t2ds_gene_value_list, pret2ds_gene_value_list)
    #   print(t2ds_pret2ds_test_result$p.value)
    #   sorted_refilter_net_node$t2ds_pret2ds_pvalue[i] = t2ds_pret2ds_test_result$p.value
    # 
    #   t2ds_no_t2ds_test_result <- wilcox.test(t2ds_gene_value_list, no_t2ds_gene_value_list)
    #   print(t2ds_no_t2ds_test_result$p.value)
    #   sorted_refilter_net_node$t2ds_no_t2ds_pvalue[i] = t2ds_no_t2ds_test_result$p.value
    # 
    #   pret2ds_no_t2ds_test_result <- wilcox.test(pret2ds_gene_value_list, no_t2ds_gene_value_list)
    #   print(pret2ds_no_t2ds_test_result$p.value)
    #   sorted_refilter_net_node$pret2ds_no_t2ds_pvalue[i] = pret2ds_no_t2ds_test_result$p.value
    # }
    # Load pre-calculated p-values (with 'gene' as the column name)
    t2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/TvsNO_min_pvalues.csv')
    pret2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/PrevsNO_min_pvalues.csv')
    
    # Rename the 'gene' column to 'gene_node_name' to match refilter_net_node
    t2ds_no_t2ds_pvalue <- t2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)
    pret2ds_no_t2ds_pvalue <- pret2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)
    
    # Merge the pre-calculated p-values with the network nodes (using 'gene_node_name' as the key)
    sorted_refilter_net_node <- left_join(sorted_refilter_net_node, t2ds_no_t2ds_pvalue, by = "gene_node_name")
    sorted_refilter_net_node <- left_join(sorted_refilter_net_node, pret2ds_no_t2ds_pvalue, by = "gene_node_name")
    print(colnames(refilter_net_node))
    
    
    net = graph_from_data_frame(d=refilter_net_edge, vertices=sorted_refilter_net_node, directed=F)
    
    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color
    # vertex_fcol = rep('black', vcount(net))
    vertex_fcol = rep(NA, vcount(net))
    # vertex color
    vertex_col = rep('lightblue', vcount(net))
    # vertex_col[V(net)$Weight>=node_threshold()] = '#FB9A99' # 'tomato'
    if (input$type_comparison == 1){
      vertex_col[V(net)$t2ds_pret2ds_test_result<=pvalue_threshold()] = '#FB9A99' # 'tomato'
    }else if(input$type_comparison == 2){
      vertex_col[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold()] = '#FB9A99' # 'tomato'
    }else if(input$type_comparison == 3){
      vertex_col[V(net)$pret2ds_no_t2ds_test_result<=pvalue_threshold()] = '#FB9A99' # 'tomato'
    }
    
    # vertex size
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$Weight>=node_threshold()] = input$imgene_node_size
    # vertex cex
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$Weight>=node_threshold()] = input$imgene_label_size
    # Loop through all nodes (vertices) in the network
    for (i in 1:vcount(net)) {
      gene <- V(net)$gene_node_name[i]  # Get the gene name from the vertex
      print(paste("Gene at node", i, ":", gene))
      # Ensure no NA values before comparing
      if (!is.na(gene)) {
        if (gene %in% t6_gene_list) {
          # If the gene is in the t6 gene list, make the border thicker and color it
          vertex_fcol[i] <- '#FF7b00'  # Make the border red
        }
      } else {
        # Optional: print or log missing values for debugging
        print(paste("Missing gene name at node", i))
      }
    }
    
    
    # edge width
    edge_width = (E(net)$Weight)*(15.0)
    edge_width[E(net)$Weight>=marking_edge_threshold()] = (E(net)$Weight)*(10.0)
    # edge color
    edge_color = rep('#666666', ecount(net))
    edge_color[E(net)$Weight>=marking_edge_threshold()] = 'black'

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
    legend(x=-1.05, y=1.13, # y= -0.72,
           legend=c('Genes', 'Important Genes with P value < 0.1', 'Genes overlapped with genome-wide significant evidence'), pch=c(21,21,21), 
           col=c('lightblue', '#FB9A99', '#FF7b00'),
           pt.bg=c('lightblue', '#FB9A99', 'white'), 
           pt.lwd=c(1, 1, 4),
           pt.cex=2, cex=1.2, bty='n')
    legend(x=-1.06, y=0.98, # y= -0.85, 
           legend=c('Gene-Gene', 'Important Gene-Gene'),
           col=c('gray', 'black'), lwd=c(5,7), cex=1.2, bty='n')
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


