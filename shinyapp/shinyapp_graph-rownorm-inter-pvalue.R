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
    sidebarPanel(width = 2,
      selectInput('option1_diabetes_type', label = 'Selection of the first type of diabetes',
                  choices = list('T2ds' = 1, 
                                 'Pret2ds' = 2, 
                                 'No_t2ds' = 3),
                  selected = 1),
      
      selectInput('option2_diabetes_type', label = 'Selection of the second type of diabetes',
                  choices = list('T2ds' = 1, 
                                 'Pret2ds' = 2, 
                                 'No_t2ds' = 3),
                  selected = 2),
      
      sliderInput('edge_threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0.0, max = 1.0,
                  value = 0.13),
      
      selectInput('omics_type', label = 'Selection of the omics to be compared',
                  choices = list('Epigenomics-Upstream' = 1,
                                 'Epigenomics-Distal' = 2,
                                 'Epigenomics-Proximal' = 3,
                                 'Epigenomics-Core' = 4,
                                 'Epigenomics-Downstream' = 5,
                                 'Transcriptomics' = 6),
                  selected = 6),
      
      sliderInput('pvalue_threshold',
                  'Select the threshold of marking important genes by p-values',
                  min = 0, max = 0.3,
                  value = 0.05),

      sliderInput('giant_comp_threshold',
                  'Select the threshold of each component',
                  min = 0.0, max = 30.0,
                  value = 10.0),
      
      sliderInput('gene_node_size',
                  'Select the gene node size',
                  min = 2, max = 10,
                  value = 6),

      sliderInput('imgene_node_size',
                  'Select the important gene node size',
                  min = 5, max = 10,
                  value = 8),

      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.4, max = 1.0,
                  value = 0.88),

      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.4, max = 1.5,
                  value = 1.28),
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
    ############################################################################################################################################################################
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$option1_diabetes_type)
    if (input$option1_diabetes_type == 1){
      type1 = 't2ds'
    } else if (input$option1_diabetes_type == 2){
      type1 = 'pret2ds'
    } else if (input$option1_diabetes_type == 3){
      type1 = 'no_t2ds'
    }
    
    type_path1 = paste('./analysis/gigtransformer-rownorm/', as.character(type1), sep='')
    edge_path1 = paste(type_path1, '_layer_norm_average_fold_gene_edge_weight_df.csv', sep='')
    net_edge_weight1 = read.csv(edge_path1)
    all_net_node = read.csv('./data/filtered_data/gene_num_dict_df.csv') # NODE LABEL
    node_path1 = paste(type_path1, '_layer_norm_average_fold_node_weight_df.csv', sep='')
    type_net_node1 = read.csv(node_path1)
    net_node1 = merge(x = all_net_node, y = type_net_node1, by.x = c('gene_node_idx'), by.y =c('Node_idx'))
    
    ### 2.1 FILTER EDGE BY [edge_weight]
    filter_net_edge1 = filter(net_edge_weight1, Weight > edge_threshold())
    filter_net_edge_node1 = unique(c(filter_net_edge1$Actual_From, filter_net_edge1$Actual_To))
    filter_net_node1 = net_node1[net_node1$gene_node_idx %in% filter_net_edge_node1, ]
    filter_net_node1 <- filter_net_node1 %>% rename(Weight1 = Weight)
    node1_value <- 'node_type1'
    node1_values <- rep(node1_value, nrow(filter_net_node1))
    filter_net_node1$node1 <- node1_values
    f_net_edge1 = net_edge_weight1[net_edge_weight1$Actual_From %in% filter_net_node1$gene_node_idx & net_edge_weight1$Actual_To %in% filter_net_node1$gene_node_idx, ]
    f_net_edge1 <- f_net_edge1 %>% rename(Weight1 = Weight)
    edge1_value <- 'edge_type1'
    edge1_values <- rep(edge1_value, nrow(f_net_edge1))
    f_net_edge1$edge1 <- edge1_values
    # print(f_net_edge1)
    
    ### 1. READ GRAPH [edge_index, node] FROM FILES
    print(input$option2_diabetes_type)
    if (input$option2_diabetes_type == 1){
      type2 = 't2ds'
    } else if (input$option2_diabetes_type == 2){
      type2 = 'pret2ds'
    } else if (input$option2_diabetes_type == 3){
      type2 = 'no_t2ds'
    }
    
    type_path2 = paste('./analysis/gigtransformer-rownorm/', as.character(type2), sep='')
    edge_path2 = paste(type_path2, '_layer_norm_average_fold_gene_edge_weight_df.csv', sep='')
    net_edge_weight2 = read.csv(edge_path2)
    all_net_node = read.csv('./data/filtered_data/gene_num_dict_df.csv') # NODE LABEL
    node_path2 = paste(type_path2, '_layer_norm_average_fold_node_weight_df.csv', sep='')
    type_net_node2 = read.csv(node_path2)
    net_node2 = merge(x = all_net_node, y = type_net_node2, by.x = c('gene_node_idx'), by.y =c('Node_idx'))
    
    ### #02 2.1 FILTER EDGE BY [edge_weight]
    filter_net_edge2 = filter(net_edge_weight2, Weight > edge_threshold())
    filter_net_edge_node2 = unique(c(filter_net_edge2$Actual_From, filter_net_edge2$Actual_To))
    filter_net_node2 = net_node2[net_node2$gene_node_idx %in% filter_net_edge_node2, ]
    filter_net_node2 <- filter_net_node2 %>% rename(Weight2 = Weight)
    node2_value <- 'node_type2'
    node2_values <- rep(node2_value, nrow(filter_net_node2))
    filter_net_node2$node2 <- node2_values
    f_net_edge2 = net_edge_weight2[net_edge_weight2$Actual_From %in% filter_net_node2$gene_node_idx & net_edge_weight2$Actual_To %in% filter_net_node2$gene_node_idx, ]
    f_net_edge2 <- f_net_edge2 %>% rename(Weight2 = Weight)
    edge2_value <- 'edge_type2'
    edge2_values <- rep(edge2_value, nrow(f_net_edge2))
    f_net_edge2$edge2 <- edge2_values
    # print(f_net_edge2)
    ############################################################################################################################################################################
    ### MERGE
    # Inner join filter_net_node1 and filter_net_node1 on col1 and col2
    inner_net_node <- inner_join(filter_net_node1, filter_net_node2, by = c("gene_node_idx", "gene_node_name"))
    inner_net_node_list <- inner_net_node$gene_node_idx
    
    
    # node1
    filter_unique_net_node1 <- filter_net_node1 %>% filter(!(gene_node_idx %in% inner_net_node_list))
    filter_unique_net_node1$Weight2 <- rep(-1, nrow(filter_unique_net_node1))
    filter_unique_net_node1$node2 <- rep('no', nrow(filter_unique_net_node1))
    # node2
    filter_unique_net_node2 <- filter_net_node2 %>% filter(!(gene_node_idx %in% inner_net_node_list))
    filter_unique_net_node2$Weight1 <- rep(-1, nrow(filter_unique_net_node2))
    filter_unique_net_node2$node1 <- rep('no', nrow(filter_unique_net_node2))
    filter_unique_net_node2 <- filter_unique_net_node2 %>% select(gene_node_idx, gene_node_name, Weight1, node1, Weight2, node2)
    bind_node <- bind_rows(filter_unique_net_node1, filter_unique_net_node2, inner_net_node)
    # Inner join f_net_edge1 and f_net_edge2 on Actual_From and Actual_To
    f_net_edge1 <- f_net_edge1 %>% select(Actual_From, Actual_To, Weight1, edge1)
    f_net_edge2 <- f_net_edge2 %>% select(Actual_From, Actual_To, Weight2, edge2)
    # edge1
    inner_net_edge <- inner_join(f_net_edge1, f_net_edge2, by = c("Actual_From", "Actual_To"))
    f_net_unique_edge1 <- f_net_edge1 %>% anti_join(inner_net_edge, by = c("Actual_From", "Actual_To"))
    f_net_unique_edge1$Weight2 <- rep(-1, nrow(f_net_unique_edge1))
    f_net_unique_edge1$edge2 <- rep('no', nrow(f_net_unique_edge1))
    # edge2
    f_net_unique_edge2 <- f_net_edge2 %>% anti_join(inner_net_edge, by = c("Actual_From", "Actual_To"))
    f_net_unique_edge2$Weight1 <- rep(-1, nrow(f_net_unique_edge2))
    f_net_unique_edge2$edge1 <- rep('no', nrow(f_net_unique_edge2))
    f_net_unique_edge2 <- f_net_unique_edge2 %>% select(Actual_From, Actual_To, Weight1, edge1, Weight2, edge2)
    bind_edge <- bind_rows(f_net_unique_edge1, f_net_unique_edge2, inner_net_edge)
    print(bind_node)
    print(bind_edge)
    
    # # # ### 2.1 FILTER AGAIN NODE BY [node_weight]
    # filter_net_node = filter(filter_net_node, Weight > node_threshold())
    # filter_net_edge = filter_net_edge[filter_net_edge$Actual_From %in% filter_net_node$gene_node_idx & filter_net_edge$Actual_To %in% filter_net_node$gene_node_idx, ]
    # print(filter_net_node)
    # print(f_net_edge)
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=bind_edge, vertices=bind_node, directed=F)
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
    refilter_net_edge <- subset(bind_edge, (Actual_From %in% giant_comp_node | Actual_To %in% giant_comp_node))
    refilter_net_edge_node = unique(c(refilter_net_edge$Actual_From, refilter_net_edge$Actual_To))
    refilter_net_node = bind_node[bind_node$gene_node_idx %in% refilter_net_edge_node,]
    
    # 3.2 USE REWEIGHTED IDF NODE DEGREE AS NODE DEGREE
    print('Number of edges')
    print(nrow(refilter_net_edge))
    print('Number of nodes')
    print(nrow(refilter_net_node))
    ############################################################################################################################################################################
    
    # # 3.3 SELECT AND CALCULATE P-VALUE
    # # 3.3.1 Pre-seletion of the omics data to be compared for the p-values
    # if (input$omics_type == 6){
    #   subject_nodeidx_gene_df = read.csv('./data/filtered_data/merged_tran_v1_nodeidx_df.csv')
    #   label_patient_nodeidx_df = read.csv('./data/filtered_data/label_phenodata_onehot_nodeidx_df.csv')
    #   t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$t2ds == 1, ]
    #   pret2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$pret2ds == 1, ]
    #   no_t2ds_nodeidx_df = label_patient_nodeidx_df[label_patient_nodeidx_df$no_t2ds == 1, ]
    #   
    #   t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% t2ds_nodeidx_df$node_idx, ]
    #   pret2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% pret2ds_nodeidx_df$node_idx, ]
    #   no_t2ds_subject_nodeidx_gene_df = subject_nodeidx_gene_df[subject_nodeidx_gene_df$subject_nodeidx %in% no_t2ds_nodeidx_df$node_idx, ]
    #   
    #   for(i in 1:nrow(refilter_net_node)) {
    #     gene_name <- refilter_net_node[i, 'gene_node_name']
    #     t2ds_gene_value_list <- t2ds_subject_nodeidx_gene_df[[gene_name]]
    #     pret2ds_gene_value_list <- pret2ds_subject_nodeidx_gene_df[[gene_name]]
    #     no_t2ds_gene_value_list <- no_t2ds_subject_nodeidx_gene_df[[gene_name]]
    #     
    #     t2ds_pret2ds_test_result <- wilcox.test(t2ds_gene_value_list, pret2ds_gene_value_list)
    #     refilter_net_node$t2ds_pret2ds_pvalue[i] = t2ds_pret2ds_test_result$p.value
    #     
    #     t2ds_no_t2ds_test_result <- wilcox.test(t2ds_gene_value_list, no_t2ds_gene_value_list)
    #     refilter_net_node$t2ds_no_t2ds_pvalue[i] = t2ds_no_t2ds_test_result$p.value
    #     
    #     pret2ds_no_t2ds_test_result <- wilcox.test(pret2ds_gene_value_list, no_t2ds_gene_value_list)
    #     refilter_net_node$pret2ds_no_t2ds_pvalue[i] = pret2ds_no_t2ds_test_result$p.value
    #   }
    # } 
    # Load pre-calculated p-values (with 'gene' as the column name)
    t2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/TvsNO_min_pvalues.csv')
    pret2ds_no_t2ds_pvalue <- read.csv('./pvalues_output/PrevsNO_min_pvalues.csv')
    
    # Rename the 'gene' column to 'gene_node_name' to match refilter_net_node
    t2ds_no_t2ds_pvalue <- t2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)
    pret2ds_no_t2ds_pvalue <- pret2ds_no_t2ds_pvalue %>% rename(gene_node_name = gene)
    
    # Merge the pre-calculated p-values with the network nodes (using 'gene_node_name' as the key)
    refilter_net_node <- left_join(refilter_net_node, t2ds_no_t2ds_pvalue, by = "gene_node_name")
    refilter_net_node <- left_join(refilter_net_node, pret2ds_no_t2ds_pvalue, by = "gene_node_name")
    print(colnames(refilter_net_node))
    
    # browser()
    
    inner_join = 'inner_join'
    inner_join_type_path = paste('./analysis/gigtransformer-rownorm/', as.character(inner_join), sep='')
    refilter_edge_path = paste(inner_join_type_path, '_norm_refilter_edge_weight_df.csv', sep='')
    write.csv(refilter_net_edge, refilter_edge_path)
    refilter_node_path = paste(inner_join_type_path, '_norm_refilter_node_weight_df.csv', sep='')
    write.csv(refilter_net_node, refilter_node_path)
    pvalue_filtered_net_node = refilter_net_node %>% filter(
      (!is.na(t2ds_no_t2ds_pvalue) & t2ds_no_t2ds_pvalue <= pvalue_threshold()) |
      (!is.na(pret2ds_no_t2ds_pvalue) & pret2ds_no_t2ds_pvalue <= pvalue_threshold())
    )
    pvalue_filtered_node_path = paste(inner_join_type_path, '_norm_refilter_pvalue_filtered_node_weight_df.csv', sep='')
    write.csv(pvalue_filtered_net_node, pvalue_filtered_node_path)
    net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)

    # 标记属于 inner_join（两个人群 Weight > 阈值 的交集）的节点
    V(net)$is_inner_join <- V(net)$name %in% as.character(inner_net_node_list)

    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color
    # vertex_fcol = rep('black', vcount(net))
    vertex_fcol = rep(NA, vcount(net))
    
    # vertex color
    # 先赋原色（交集基因用饱和色）
    vertex_col = rep('#A6CEE3', vcount(net))
    vertex_col[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold()] = '#B2DF8A'
    vertex_col[V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = '#CAB2D6'
    vertex_col[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold() & V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = '#FB9A99'
    # 非交集基因：保留原色但加透明度淡化（50 = ~31% 不透明度）
    not_inner = !V(net)$is_inner_join
    vertex_col[not_inner & vertex_col == '#A6CEE3'] = '#A6CEE350'
    vertex_col[not_inner & vertex_col == '#B2DF8A'] = '#B2DF8A50'
    vertex_col[not_inner & vertex_col == '#CAB2D6'] = '#CAB2D650'
    vertex_col[not_inner & vertex_col == '#FB9A99'] = '#FB9A9950'

    # vertex size
    vertex_size = rep(input$gene_node_size, vcount(net))
    vertex_size[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_node_size
    vertex_size[V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_node_size
    vertex_size[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold() & V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_node_size
    # 非交集基因缩小
    vertex_size[not_inner] = input$gene_node_size

    # vertex cex
    vertex_cex = rep(input$gene_label_size, vcount(net))
    vertex_cex[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_label_size
    vertex_cex[V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_label_size
    vertex_cex[V(net)$t2ds_no_t2ds_pvalue<=pvalue_threshold() & V(net)$pret2ds_no_t2ds_pvalue<=pvalue_threshold()] = input$imgene_label_size
    # 非交集基因标签缩小
    vertex_cex[not_inner] = input$gene_label_size
    
    
    # Loop through all nodes (vertices) in the network
    for (i in 1:vcount(net)) {
      gene <- V(net)$gene_node_name[i]  # Get the gene name from the vertex
      print(paste("Gene at node", i, ":", gene))
      # Ensure no NA values before comparing
      if (!is.na(gene)) {
        if (gene %in% t6_gene_list) {
          # If the gene is in the t6 gene list, make the border thicker and color it
          vertex_fcol[i] <- '#FF7b00'  # Make the border red
          vertex_size[i] <- input$imgene_node_size  # Make the node bigger
        }
      } else {
        # Optional: print or log missing values for debugging
        print(paste("Missing gene name at node", i))
      }
    }
    
    
    # edge width
    edge_width = rep(1.0, ecount(net))
    # edge color
    edge_color = rep('#C0C0C0', ecount(net))
    
    set.seed(18)
    plot(net,
         vertex.frame.width = 4,
         vertex.frame.color = vertex_fcol,
         vertex.color = vertex_col,
         vertex.size = vertex_size,
         vertex.label = V(net)$gene_node_name,
         vertex.label.color = '#000000',
         vertex.label.cex = vertex_cex,
         edge.width = edge_width,
         edge.color = edge_color,
         edge.curved = 0.2,
         layout=layout_with_graphopt)
    ### ADD LEGEND
    legend(x=-1.05, y=1.13, # y= -0.72,
           legend=c('Genes (saturated=overlap, faded=unique)',
                    'T2D Significant Genes (p-value)',
                    'Pre_T2D Significant Genes (p-value)',
                    'T2D and Pre_T2D Significant Genes',
                    'Genes overlapped with genome-wide significant evidence'),
           pch=c(21, 21, 21, 21, 21),
           col = c('#A6CEE3','#B2DF8A', '#CAB2D6', '#FB9A99', '#FF7b00'),
           pt.bg=c('#A6CEE3','#B2DF8A', '#CAB2D6', '#FB9A99', 'white'),
           pt.lwd=c(1, 1, 1, 1, 4),
           pt.cex=2, cex=1.2, bty='n')
    legend(x=-1.06, y=0.92, # y= -0.85, 
           legend=c('Gene-Gene Interactions'),
           col=c('#C0C0C0'), lwd=c(2, 3), cex=1.2, bty='n')
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


