"""Export checkpoint attention on its held-out fold using stable subject identities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / 'processed_data'
CLASS_NAMES = {0: 't2ds', 1: 'pret2ds', 2: 'no_t2ds'}
GENE_LAYERS = {'gene_conv_first': 'gene_first', 'gene_conv_block': 'gene_block', 'gene_conv_last': 'gene_last'}
PHENO_LAYERS = {'conv_first': 'pheno_first', 'conv_block': 'pheno_block', 'conv_last': 'pheno_last'}


def analysis_layers(model):
    """Active layer names, preserving legacy names and indexing every v2 layer."""
    if hasattr(model, 'gene_convs'):
        gene = ({f'gene_layer_{i+1:02d}': layer for i, layer in enumerate(model.gene_convs)}
                if getattr(model, 'graph_opt', 'GinG') != 'subject' else {})
        patient = {f'pheno_layer_{i+1:02d}': layer for i, layer in enumerate(getattr(model, 'patient_convs', []))}
    else:
        gene = {name: getattr(model, attr) for attr, name in GENE_LAYERS.items()
                if hasattr(model, attr) and getattr(model, 'graph_opt', 'GinG') != 'subject'}
        patient = {name: getattr(model, attr) for attr, name in PHENO_LAYERS.items()
                   if hasattr(model, attr) and getattr(model, 'graph_opt', 'GinG') in ('GinG', 'subject')}
    return gene, patient


def node_map(subject_ids, categories):
    """Fold-local graph indices must never be confused with cohort subject IDs."""
    rows = [{'Node_Index': index, 'Node_Type': 'category', 'Subject_ID': '', 'Category': str(name)}
            for index, name in enumerate(categories)]
    rows += [{'Node_Index': len(categories) + index, 'Node_Type': 'subject',
              'Subject_ID': str(subject), 'Category': ''} for index, subject in enumerate(subject_ids)]
    return pd.DataFrame(rows)


class AttentionCollector:
    """Request attention from existing layers, returning their original tensor to the model."""
    def __init__(self, data, fold, output, categories, full_subjects=False, gene_scorer=None):
        self.data, self.fold, self.output = data, int(fold), Path(output)
        self.full_subjects = full_subjects
        self.gene_scorer = gene_scorer
        self.nodes = node_map(data['subject_ids'], categories)
        self.edges = np.asarray(data['gene_edge_index'], dtype=np.int64)
        self.subject_ids = np.asarray(data['subject_ids'])
        self.labels = np.asarray(data['labels'])
        self.num_genes = int(data['num_gene_node'])
        self.num_categories = int(data['num_subfeature'])
        if len(categories) != self.num_categories or len(set(map(str, self.subject_ids))) != len(self.subject_ids):
            raise ValueError('invalid category/subject identities')
        if not np.isin(self.labels, list(CLASS_NAMES)).all():
            raise ValueError('unknown diabetes class')
        self.cursors = {name: 0 for name in GENE_LAYERS.values()}
        self.sums, self.counts, self.handles, self.pheno_seen = {}, {}, [], set()

    @staticmethod
    def request_attention(module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs['return_attention_weights'] = True
        return args, kwargs

    def attach(self, model):
        if __package__:
            from .runtime import apply_graph_policy
        else:
            from runtime import apply_graph_policy
        self.data = apply_graph_policy(
            self.data, getattr(model, 'config', {}).get('graph_policy', 'legacy_directed'))
        gene, patient = analysis_layers(model)
        self.cursors = {name: 0 for name in gene}
        self.expected_pheno = set(patient)
        for layer, module in {**gene, **patient}.items():
            self.handles.append(module.register_forward_pre_hook(self.request_attention, with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.make_hook(layer)))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def make_hook(self, layer):
        def capture(module, args, result):
            values, (edge_index, alpha) = result
            edges = edge_index.detach().cpu().numpy()
            weights = alpha.detach().cpu().numpy()
            if layer in self.cursors:
                batch_size = args[0].shape[0] // self.num_genes
                self.collect_gene(layer, edges, weights, batch_size)
                if self.gene_scorer is not None:
                    self.gene_scorer.collect_gene(layer,edges,weights,batch_size,values)
            else:
                self.collect_pheno(layer, edges, weights)
            # The rest of the network sees exactly the same tensor as without hooks.
            return values
        return capture

    def collect_gene(self, layer, edges, weights, batch_size):
        count_edges = self.edges.shape[1]
        start = self.cursors[layer]
        stop = start + batch_size
        if batch_size <= 0 or stop > len(self.subject_ids) or weights.shape[0] != batch_size * count_edges:
            raise ValueError('gene attention batch cannot be aligned to held-out subjects')
        for batch in range(batch_size):
            actual = edges[:, batch * count_edges:(batch + 1) * count_edges] - batch * self.num_genes
            if not np.array_equal(actual, self.edges):
                raise ValueError('gene graph edge ordering/identity differs within batch')
        weights = weights.reshape(batch_size, count_edges, -1)
        labels = self.labels[start:stop]
        for label in np.unique(labels):
            key = (layer, int(label))
            total = weights[labels == label].sum(axis=0, dtype=np.float64)
            self.sums[key] = self.sums.get(key, np.zeros_like(total)) + total
            self.counts[key] = self.counts.get(key, 0) + int((labels == label).sum())
        if self.full_subjects:
            folder = self.output / 'subject_gene_attention'
            folder.mkdir(exist_ok=True)
            heads = weights.shape[2]
            table = pd.DataFrame({
                'Fold': self.fold, 'Layer': layer,
                'Subject_ID': np.repeat(self.subject_ids[start:stop], count_edges * heads),
                'Patient_Index': np.repeat(np.arange(start, stop) + self.num_categories, count_edges * heads),
                'Class': np.repeat([CLASS_NAMES[int(value)] for value in labels], count_edges * heads),
                'Actual_From': np.tile(np.repeat(self.edges[0], heads), batch_size),
                'Actual_To': np.tile(np.repeat(self.edges[1], heads), batch_size),
                'Head': np.tile(np.arange(heads), batch_size * count_edges),
                'Weight': weights.reshape(-1),
            })
            table.to_csv(folder / f'{layer}_{start:06d}_{stop:06d}.csv.gz', index=False)
        self.cursors[layer] = stop

    def collect_pheno(self, layer, edges, weights):
        if layer in self.pheno_seen or not np.array_equal(edges, np.asarray(self.data['edge_index'])):
            raise ValueError('phenotype attention graph identity differs or layer ran twice')
        self.pheno_seen.add(layer)
        heads = weights.shape[1]
        source, target = np.repeat(edges[0], heads), np.repeat(edges[1], heads)
        table = pd.DataFrame({'Fold': self.fold, 'Layer': layer, 'from_node_idx': source,
            'to_node_idx': target, 'Head': np.tile(np.arange(heads), edges.shape[1]), 'Weight': weights.reshape(-1)})
        for prefix, indices in [('From', source), ('To', target)]:
            for column in ['Node_Type', 'Subject_ID', 'Category']:
                table[f'{prefix}_{column}'] = self.nodes[column].to_numpy()[indices]
        table.to_csv(self.output / f'{layer}_patient_edge_weight.csv', index=False)

    def finish(self):
        used = list(self.cursors.values())
        if any(count != len(self.subject_ids) for count in used):
            raise ValueError('not all held-out subjects were exported in every gene layer')
        if hasattr(self, 'expected_pheno') and self.pheno_seen != self.expected_pheno:
            raise ValueError('Incomplete patient-layer export')
        pieces = []
        for (layer, label), sums in sorted(self.sums.items()):
            count, heads = self.counts[(layer, label)], sums.shape[1]
            pieces.append(pd.DataFrame({'Fold': self.fold, 'Layer': layer,
                'Class': CLASS_NAMES[label], 'Head': np.tile(np.arange(heads), self.edges.shape[1]),
                'Actual_From': np.repeat(self.edges[0], heads), 'Actual_To': np.repeat(self.edges[1], heads),
                'Weight': (sums / count).reshape(-1), 'Weight_Sum': sums.reshape(-1), 'n_subjects': count}))
        columns = ['Fold', 'Layer', 'Class', 'Head', 'Actual_From', 'Actual_To', 'Weight', 'Weight_Sum', 'n_subjects']
        summary = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
        summary.to_csv(self.output / 'gene_attention_summary.csv', index=False)
        self.nodes.to_csv(self.output / 'node_map.csv', index=False)
        return summary


def analyze_checkpoint(checkpoint, data_root, output, device='cpu', batch_size=32, full_subjects=False,
                       subject_gene_scores=False, skip_gene_scores=False):
    with tqdm(total=4, desc='Attention analysis: load dependencies', unit='stage') as progress:
        return _analyze_checkpoint(checkpoint, data_root, output, device, batch_size, full_subjects, progress,
                                   subject_gene_scores,skip_gene_scores)


def _analyze_checkpoint(checkpoint, data_root, output, device, batch_size, full_subjects, progress,
                        subject_gene_scores=False,skip_gene_scores=False):
    # Lazy imports keep --help and pure data utilities usable without a GPU environment.
    import torch
    if __package__:
        from . import runtime
        from .gene_scores import GeneScoreCollector
    else:
        import runtime
        from gene_scores import GeneScoreCollector
    if batch_size <= 0:
        raise ValueError('batch size must be positive')
    device = runtime.select_device(str(device))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('choose an empty output directory to avoid mixing checkpoint exports')
    progress.update(1)
    progress.set_description('Attention analysis: load checkpoint and fold')
    model, payload, data = runtime.load_run(Path(checkpoint), Path(data_root), device)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(Path(data_root) / 'metadata/subfeature_nodes.csv')
    if not np.array_equal(metadata.subfeature_node_idx, np.arange(len(metadata))):
        raise ValueError('category node map is not in node-index order')
    model.eval()
    scorer=GeneScoreCollector(model,data) if model.graph_opt!='subject' and not skip_gene_scores else None
    collector = AttentionCollector(data, payload['fold'], output, metadata.subfeature_name.tolist(), full_subjects,gene_scorer=scorer)
    with torch.no_grad():
        baseline_logits = runtime.forward(model, data, device, batch_size=batch_size)[1].detach().cpu()
    collector.attach(model)
    progress.update(1)
    progress.set_description('Attention analysis: inference and attention capture')
    try:
        with torch.no_grad():
            _, captured_logits, _, predicted = runtime.forward(model, data, device, batch_size=batch_size)
        captured_logits = captured_logits.detach().cpu()
        torch.testing.assert_close(captured_logits, baseline_logits, rtol=1e-5, atol=1e-6)
        if not torch.equal(captured_logits.argmax(-1), baseline_logits.argmax(-1)):
            raise ValueError('Attention hooks changed predicted labels')
        max_logit_diff = float((captured_logits - baseline_logits).abs().max())
        probabilities = torch.softmax(captured_logits, dim=-1).numpy()
        progress.update(1)
        progress.set_description('Attention analysis: write summaries and metadata')
        summary = collector.finish()
        if scorer is not None:
            scorer.export(output,data_root,payload,checkpoint,subject_gene_scores)
        elif not skip_gene_scores:
            (output/'gene_scoring_metadata.json').write_text(json.dumps(dict(schema_version=1,
                status='not_applicable',reason='phenotype-only model has no active gene branch',
                condition=payload.get('ablation') or 'phenotype_only',fold=payload['fold'],seed=payload['seed']),indent=2)+'\n',encoding='utf-8')
    finally:
        collector.remove()
    predictions = pd.DataFrame({'Subject_ID': data['subject_ids'], 'Patient_Index': data['node_index'],
                                'Label': data['labels'], 'Predicted_Label': predicted.detach().cpu().numpy()})
    for index, name in CLASS_NAMES.items():
        predictions[f'Probability_{name}'] = probabilities[:, index]
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    report = {'checkpoint': str(Path(checkpoint).resolve()), 'fold': int(payload['fold']),
        'seed': payload.get('seed'), 'epoch': payload.get('epoch'), 'model_config': payload['model_config'],
        'data_signature': payload.get('data_signature'), 'ablation': payload.get('ablation'),
        'scope': 'held-out fold test graph only', 'n_subjects': len(data['subject_ids']),
        'subject_identity': 'Subject_ID is the original subject ID; Patient_Index is fold-local graph index',
        'attention': 'incoming-neighbor softmax per head, evaluation mode (no attention dropout)',
        'gene_aggregation': 'arithmetic mean over test subjects within each class/layer/edge/head',
        'head_policy': 'all heads exported separately; legacy normalization defaults to head 0',
        'gene_summary_rows': len(summary), 'full_subject_export': full_subjects,
        'gene_scoring':'complete' if scorer is not None else ('skipped' if skip_gene_scores else 'not_applicable'),
        'phenotype_layers': sorted(collector.pheno_seen),
        'gene_layers': list(collector.cursors),
        'prediction_invariance': {'checked': True, 'max_abs_logit_difference': max_logit_diff, 'rtol': 1e-5, 'atol': 1e-6, 'predicted_labels_identical': True},
        'limitations': ['Attention is descriptive model behavior, not causal or significance evidence.',
                       'These held-out folds are also used for checkpoint/hyperparameter selection; they are not an independent final evaluation.',
                       'Test-only attention differs in scope from legacy whole-cohort transductive exports.']}
    (output / 'analysis_metadata.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    progress.update(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--full-subjects', action='store_true', help='Also write potentially large per-subject gene CSV.gz files')
    parser.add_argument('--subject-gene-scores',action='store_true',help='Save per-subject four-method score matrices as compressed NPZ')
    parser.add_argument('--skip-gene-scores',action='store_true',help='Export only the original attention analysis')
    args = parser.parse_args()
    print(json.dumps(analyze_checkpoint(args.checkpoint,args.data_root,args.output,args.device,args.batch_size,args.full_subjects,
                                       args.subject_gene_scores,args.skip_gene_scores),indent=2))


if __name__ == '__main__':
    main()
