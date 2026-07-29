import torch
import numpy as np
import pandas as pd
import networkx as nx

from numpy import inf
from torch_geometric.data import Data


def read_geodata(graph_feature, edge_index, node_label, node_idx):
    # CONVERT [numpy] TO [torch]
    graph_feature = torch.from_numpy(graph_feature).float()
    edge_index = torch.from_numpy(edge_index).to(torch.long)
    node_label = torch.from_numpy(np.array([node_label])).to(torch.long)
    node_idx = torch.from_numpy(node_idx).to(torch.long)
    geo_data = Data(x=graph_feature, edge_index=edge_index, node_label=node_label, node_index=node_idx)
    return geo_data


class ReadGeoGraph():
    def __init__(self):
        pass

    def read_feature(self, num_graph, num_feature, num_node, xBatch):
        # FORM [graph_feature_list]
        xBatch = xBatch.reshape(num_graph, num_node, num_feature)
        graph_feature_list = []
        for i in range(num_graph):
            graph_feature_list.append(xBatch[i, :, :])
        return graph_feature_list

    def form_geo_datalist(self, num_graph, graph_feature_list, edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            geo_data = Data(x=graph_feature, edge_index=edge_index)
            geo_datalist.append(geo_data)
        return geo_datalist


def read_batch(index, upper_index, x_input, num_feature, num_node, edge_index):
    # FORMING BATCH FILES
    # print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    # print(xBatch.shape)
    # PREPARE LOADING LISTS OF [features, edge_index]
    # print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist(num_graph, graph_feature_list, edge_index)
    return geo_datalist