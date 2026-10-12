"""Explicit modality/branch selection; never modify prepared arrays on disk."""
import numpy as np

CONDITIONS = ('full','methylation_only','gene_only','phenotype_only','rna_only')
DEFAULT_CONDITIONS = CONDITIONS[:4]
CHANNELS = {'full':(0,1,2,3,4,5),'gene_only':(0,1,2,3,4,5),
            'methylation_only':(1,2,3,4,5),'rna_only':(0,)}


def configure_ablation(base,condition):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown ablation condition: {condition}')
    config = dict(base)
    if condition=='phenotype_only':
        config.update(graph_opt='subject',gene_input_dim=6,ortho_weight=0.,link_weight=0.,ent_weight=0.)
    else:
        config.update(graph_opt='GinG' if condition=='full' else 'gene',gene_input_dim=len(CHANNELS[condition]))
    return config


def transform_ablation(data,condition):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown ablation condition: {condition}')
    if data['num_feature'] != 6:
        raise ValueError('Ablation expects original six-channel prepared data')
    result = dict(data)
    if condition=='phenotype_only':
        result['gene_x'] = np.zeros_like(data['gene_x'])
        result['gene_edge_index'] = np.empty((2,0),dtype=np.int64)
    else:
        gene = data['gene_x'].reshape(data['num_subject'],data['num_gene_node'],6)
        selected = np.ascontiguousarray(gene[:,:,CHANNELS[condition]])
        result['num_feature'] = len(CHANNELS[condition])
        result['gene_x'] = selected.reshape(data['num_subject'],-1)
        if condition!='full':
            # Removing feature values alone would still leave phenotype-derived graph topology.
            result['graph_x'] = np.zeros_like(data['graph_x'])
            result['edge_index'] = np.empty((2,0),dtype=np.int64)
            result['phenotype_x'] = np.zeros_like(data['phenotype_x']) if 'phenotype_x' in data else np.empty((0,))
            result['phenotype_bins'] = np.zeros_like(data['phenotype_bins']) if 'phenotype_bins' in data else np.empty((0,))
    result['gene_tensor'] = result['gene_x'].reshape(result['num_subject'],result['num_gene_node'],result['num_feature'])
    return result
