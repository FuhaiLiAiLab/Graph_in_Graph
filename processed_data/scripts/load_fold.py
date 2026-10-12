"""Read a prepared split without fitting transforms or mixing participant graphs."""
import json
from pathlib import Path

import numpy as np


def load_fold(fold, role, root=None):
    if fold not in range(1, 6) or role not in ('train', 'test'):
        raise ValueError('fold must be 1..5 and role must be train or test')
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    folder = root/f'folds/fold_{fold:02d}'/role
    names = ['subject_ids', 'labels', 'labels_onehot', 'gene_x', 'phenotype_x',
             'phenotype_bins', 'graph_x', 'edge_index', 'node_index']
    data = {name: np.load(folder/f'{name}.npy', allow_pickle=False) for name in names}
    data['gene_edge_index'] = np.load(root/'shared/gene_edge_index.npy', allow_pickle=False)
    data['num_subject'] = len(data['subject_ids'])
    data['num_gene_node'] = config['n_genes']
    data['num_feature'] = len(config['channels'])
    data['num_subfeature'] = config['n_subfeature_nodes']
    data['num_phenotype_feature'] = config['n_phenotype_features']
    data['gene_tensor'] = data['gene_x'].reshape(data['num_subject'], data['num_gene_node'], data['num_feature'])
    assert data['gene_x'].shape == (data['num_subject'], data['num_gene_node'] * data['num_feature'])
    assert data['graph_x'].shape == (data['num_subfeature'] + data['num_subject'], data['num_phenotype_feature'])
    return data
