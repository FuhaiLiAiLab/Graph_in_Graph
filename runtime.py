"""Shared portable data/model/checkpoint contract for training and analysis."""
import hashlib
import os
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
if __package__:
    from .enc.geo_gigtransformer import GIG_Transformer, GENE_NUM_TOP_FEATURE
    from .processed_data.scripts.load_fold import load_fold
    from .ablation_data import configure_ablation, transform_ablation
    from .model_config import resolve_model_config
    from .enc.geo_gigtransformer_v2 import GIG_TransformerV2
else:
    from enc.geo_gigtransformer import GIG_Transformer, GENE_NUM_TOP_FEATURE
    from processed_data.scripts.load_fold import load_fold
    from ablation_data import configure_ablation, transform_ablation
    from model_config import resolve_model_config
    from enc.geo_gigtransformer_v2 import GIG_TransformerV2

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT/'processed_data'


def default_config():
    return dict(gene_input_dim=6, gene_hidden_dim=9, gene_output_dim=9,
                gene_num_top_feature=GENE_NUM_TOP_FEATURE, gig_input_dim=42,
                gig_input_transform_dim=24, gig_hidden_dim=24, gig_output_dim=24,
                num_classes=3, gene_num_head=1, gig_num_head=1, graph_opt='GinG',
                class_weight_fine=0.7746597442093394, class_weight=0.8209978466887934,
                ortho_weight=0.007968476155213464, link_weight=0.002195183531312866,
                ent_weight=0.09039903257209198, lr=0.003938420135801312,
                weight_decay=0.00018042132714427627, milestones=[150,300,450],
                gamma=0.95, clip=2.0)


def read_config(path=None):
    custom = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    return resolve_model_config(custom, default_config())


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; check GPU allocation and installed PyTorch wheels')
    return device


def seed_everything(seed):
    """Configure and record repeatability before initializing CUDA or model weights."""
    mode = os.environ.get('REVISION_DETERMINISTIC', '1')
    if mode not in ('0', '1'):
        raise ValueError('REVISION_DETERMINISTIC must be 0 or 1')
    strict = mode == '1'
    if strict:
        workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        valid = (':4096:8', ':16:8')
        if torch.cuda.is_initialized() and workspace not in valid:
            raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization; restart the process')
        if workspace is not None and workspace not in valid:
            raise ValueError('Invalid CUBLAS_WORKSPACE_CONFIG for deterministic training')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(strict, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'cuda_version': torch.version.cuda,
    }


def build_model(config, num_gene_node, device):
    cfg = resolve_model_config(config, default_config())
    if cfg.get('model_schema_version') == 2:
        return GIG_TransformerV2(cfg, num_gene_node).to(select_device(str(device)))
    if cfg['gene_num_head'] != 1 or cfg['gig_num_head'] != 1:
        raise ValueError('Current architecture requires single-head gene and patient layers')
    for key in ['gene_input_dim','gene_hidden_dim','gene_output_dim','gene_num_top_feature',
                'gig_input_dim','gig_input_transform_dim','gig_hidden_dim','gig_output_dim']:
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['graph_opt'] not in ['GinG','gene','subject'] or cfg['num_classes'] != 3:
        raise ValueError('Expected GinG/gene/subject and three diabetes classes')
    return GIG_Transformer(gene_input_dim=cfg['gene_input_dim'], gene_hidden_dim=cfg['gene_hidden_dim'],
        gene_embedding_dim=cfg['gene_output_dim'], gene_num_top_feature=cfg['gene_num_top_feature'],
        num_gene_node=num_gene_node, gig_input_dim=cfg['gig_input_dim'],
        gig_input_transform_dim=cfg['gig_input_transform_dim'], gig_hidden_dim=cfg['gig_hidden_dim'],
        gig_embedding_dim=cfg['gig_output_dim'], num_classes=cfg['num_classes'],
        gene_num_head=cfg['gene_num_head'], gig_num_head=cfg['gig_num_head'],
        class_weight_fine=cfg['class_weight_fine'], class_weight=cfg['class_weight'],
        ortho_weight=cfg['ortho_weight'], link_weight=cfg['link_weight'],
        ent_weight=cfg['ent_weight'], graph_opt=cfg['graph_opt']).to(select_device(str(device)))


def training_class_weights(labels, device, frequency_exponent=1., majority_multiplier=.5, mode='power'):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if len(counts) != 3 or np.any(counts == 0):
        raise ValueError('Training labels must contain each of the three classes')
    if not 0 <= frequency_exponent <= 1 or not 0 < majority_multiplier <= 1:
        raise ValueError("Invalid class weight exponent or majority multiplier")
    if mode not in ('none','inverse_frequency','power'):
        raise ValueError('Unknown class weighting mode')
    if mode=='none':return torch.ones(3,dtype=torch.float32,device=device)
    if mode=='inverse_frequency':frequency_exponent=1.;majority_multiplier=1.
    weights = counts.astype(float)**(-frequency_exponent)
    weights[2] *= majority_multiplier
    return torch.tensor(weights, dtype=torch.float32, device=device)


def apply_graph_policy(data, policy='legacy_directed'):
    """Append missing category->subject edges without changing saved input arrays."""
    if policy not in ('legacy_directed', 'patient_category_bidirectional'):
        raise ValueError('Unknown graph policy')
    prior = data.get('_patient_graph_policy', 'legacy_directed')
    if prior != 'legacy_directed' and prior != policy:
        raise ValueError('Effective patient graph policy differs from requested policy')
    if policy == 'legacy_directed':
        return data
    categories, subjects = int(data['num_subfeature']), int(data['num_subject'])
    total = categories + subjects
    edges = np.asarray(data['edge_index'])
    if (categories < 0 or subjects < 0 or edges.ndim != 2 or edges.shape[0] != 2
            or not np.issubdtype(edges.dtype, np.integer)):
        raise ValueError('Invalid patient edge_index or node counts')
    if edges.size and (edges.min() < 0 or edges.max() >= total):
        raise ValueError('Patient edge_index contains nodes outside this split-local graph')
    if (data['graph_x'].shape[0] != total or
            not np.array_equal(data['node_index'], np.arange(categories, total))):
        raise ValueError('Patient/category node identities differ from graph layout')
    if prior == policy:
        return data
    edges = np.asarray(edges, dtype=np.int64)
    mask = (edges[0] >= categories) & (edges[1] < categories)
    reverse = edges[::-1, mask]
    keys = edges[0] * total + edges[1]
    reverse_keys = reverse[0] * total + reverse[1]
    missing = np.flatnonzero(~np.isin(reverse_keys, keys))
    if missing.size:
        _, first = np.unique(reverse_keys[missing], return_index=True)
        missing = missing[np.sort(first)]
    effective = np.concatenate([edges, reverse[:, missing]], axis=1)
    return dict(data, edge_index=effective, _patient_graph_policy=policy)


def patient_graph_summary(data):
    """Record the effective graph and its split-local subject identities."""
    edges = np.ascontiguousarray(data['edge_index'], dtype=np.int64)
    categories, subjects = int(data['num_subfeature']), int(data['num_subject'])
    incoming = np.bincount(edges[1], minlength=categories + subjects)[categories:]
    digest = hashlib.sha256(edges.tobytes())
    digest.update(np.ascontiguousarray(data['node_index'], dtype=np.int64).tobytes())
    digest.update(json.dumps(np.asarray(data['subject_ids']).tolist(), separators=(',', ':')).encode())
    return dict(policy=data.get('_patient_graph_policy', 'legacy_directed'),
                edge_count=int(edges.shape[1]), num_subjects=subjects, num_categories=categories,
                subject_in_degree_min=int(incoming.min()) if subjects else 0,
                subject_in_degree_max=int(incoming.max()) if subjects else 0,
                sha256=digest.hexdigest())


def forward(model, data, device, batch_size=32):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    data = apply_graph_policy(data, getattr(model, 'config', {}).get('graph_policy', 'legacy_directed'))
    return model(num_feature=data['num_feature'], num_subfeature=data['num_subfeature'],
        num_subject=data['num_subject'], num_gene_node=data['num_gene_node'],
        gene_feature=data['gene_x'], gene_edge_index=torch.as_tensor(data['gene_edge_index'], dtype=torch.long),
        x=torch.as_tensor(data['graph_x'], dtype=torch.float32, device=device),
        edge_index=torch.as_tensor(data['edge_index'], dtype=torch.long, device=device),
        node_label=torch.as_tensor(data['labels_onehot'][None], dtype=torch.long, device=device),
        node_index=torch.as_tensor(data['node_index'], dtype=torch.long, device=device),
        args=SimpleNamespace(batch_size=batch_size, num_workers=0,
                             progress_position=getattr(model,'progress_position',0)+1), device=device)


def data_signature(data_root):
    root = Path(data_root)
    digest = hashlib.sha256()
    for name in ['config.json','splits/subject_folds.csv','metadata/genes.csv',
                 'metadata/channels.csv','metadata/source_manifest.json']:
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def load_pair(fold, data_root=DEFAULT_DATA_ROOT):
    train = load_fold(fold, 'train', data_root)
    test = load_fold(fold, 'test', data_root)
    import pandas as pd
    tr = pd.read_csv(Path(data_root)/f'folds/fold_{fold:02d}/train/subjects.csv')
    te = pd.read_csv(Path(data_root)/f'folds/fold_{fold:02d}/test/subjects.csv')
    np.testing.assert_array_equal(tr.subject, train['subject_ids'])
    np.testing.assert_array_equal(te.subject, test['subject_ids'])
    if set(train['subject_ids']) & set(test['subject_ids']) or set(tr.family_id) & set(te.family_id):
        raise ValueError('Train/test subject or family overlap')
    return train, test


def load_run(checkpoint, data_root=DEFAULT_DATA_ROOT, device='cpu'):
    device = select_device(str(device))
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    if payload['data_signature'] != data_signature(data_root):
        raise ValueError('Checkpoint data signature differs: wrong cohort, folds or feature order')
    test = load_fold(payload['fold'], 'test', data_root)
    if payload.get('ablation') is not None:
        condition = payload['ablation']
        if configure_ablation(payload['model_config'],condition) != payload['model_config']:
            raise ValueError('Checkpoint modality and model configuration disagree')
        test = transform_ablation(test,condition)
    graph_policy = payload['model_config'].get('graph_policy', 'legacy_directed')
    test = apply_graph_policy(test, graph_policy)
    expected_graph = payload.get('patient_graph')
    if graph_policy != 'legacy_directed' and expected_graph is None:
        raise ValueError('Checkpoint lacks effective patient graph identity')
    if expected_graph is not None and expected_graph['test'] != patient_graph_summary(test):
        raise ValueError('Checkpoint effective patient graph identity differs')
    model = build_model(payload['model_config'], test['num_gene_node'], device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return model, payload, test
