"""Small numerical/identity regression tests; never train a model."""
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


class DownstreamTests(unittest.TestCase):
    def test_normalization_uses_edge_ids_and_constant_rows_are_zero(self):
        module = importlib.import_module('normalize_edges')
        # Deliberately unrelated CSV order to graph order.
        edges = np.array([[0, 0, 1], [1, 2, 2]])
        frame = pd.DataFrame({'Actual_From': [1, 0, 0], 'Actual_To': [2, 2, 1],
                              'Weight': [0.5, 0.8, 0.2]})
        norm = module.normalize_frame(frame, edges, 3, 'minmax')
        np.testing.assert_allclose(norm.Weight, [0, 1, 0])
        row = module.normalize_frame(frame, edges, 3, 'rownorm')
        np.testing.assert_allclose(row.Weight, [0.5, 0.4, 0.1])
        np.testing.assert_array_equal(row.Degree_Row, [1, 2, 2])
        np.testing.assert_array_equal(row.Degree_Col, [2, 2, 1])

    def test_fold_averaging_is_keyed_and_weighting_is_explicit(self):
        module = importlib.import_module('normalize_edges')
        frame = pd.DataFrame({'Fold': [1, 1, 2, 2], 'Layer': ['gene_first']*4,
            'Class': ['t2ds']*4, 'Head': [0]*4, 'Actual_From': [0, 1, 1, 0],
            'Actual_To': [1, 0, 0, 1], 'Weight': [0.1, 0.3, 0.9, 0.5],
            'n_subjects': [1, 1, 3, 3]})
        equal = module.average_folds(frame, 'equal')
        weighted = module.average_folds(frame, 'subjects')
        np.testing.assert_allclose(equal.Weight, [0.3, 0.6])
        np.testing.assert_allclose(weighted.Weight, [0.4, 0.75])

    def test_mwu_uses_raw_subject_samples_and_bh_with_missing_counts(self):
        module = importlib.import_module('calculate_pvalue')
        raw = np.array([[1, np.nan], [2, 4], [8, 8], [9, 9]], dtype=float)
        result = module.calculate_comparison(raw, np.array([0, 0, 2, 2]),
                                             ['A', 'B'], 0, 2)
        self.assertAlmostEqual(result.iloc[0].p_value, mannwhitneyu([1, 2], [8, 9], alternative='two-sided').pvalue)
        self.assertEqual(result.iloc[1].n_group_one, 1)
        self.assertTrue((result.q_value_BH >= result.p_value).all())

    def test_subject_node_mapping_keeps_global_identity(self):
        module = importlib.import_module('analyze')
        nodes = module.node_map(np.array([410, 99]), ['sex-1', 'sex-2', 'age-1'])
        self.assertEqual(nodes.iloc[3].Subject_ID, '410')
        self.assertEqual(nodes.iloc[4].Subject_ID, '99')
        self.assertEqual(nodes.iloc[2].Category, 'age-1')
        self.assertEqual(nodes.iloc[3].Node_Index, 3)

    def test_attention_hook_preserves_real_convolution_and_subject_order(self):
        import torch
        from analyze import AttentionCollector
        from enc.geo_gigtransformer import GeneTransformerConv
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gene_conv_first = GeneTransformerConv(2, 2)
                self.gene_conv_block = GeneTransformerConv(2, 2)
                self.gene_conv_last = GeneTransformerConv(2, 2)

            def forward(self, x, edges):
                for layer in [self.gene_conv_first, self.gene_conv_block, self.gene_conv_last]:
                    x = layer(x, edges)
                return x

        torch.manual_seed(19)
        model = Network().eval()
        graph_edges = np.array([[0, 0, 1], [1, 2, 2]])
        data = {'subject_ids': np.array([410, 99]), 'labels': np.array([0, 2]),
                'gene_edge_index': graph_edges, 'num_gene_node': 3, 'num_subfeature': 2}
        x = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10
        edges = torch.as_tensor(np.concatenate([graph_edges, graph_edges + 3], axis=1), dtype=torch.long)
        with torch.no_grad():
            expected = model(x, edges)
            with tempfile.TemporaryDirectory() as directory:
                collector = AttentionCollector(data, 2, directory, ['sex-1', 'sex-2'], True)
                collector.attach(model)
                try:
                    observed = model(x, edges)
                    summary = collector.finish()
                finally:
                    collector.remove()
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                self.assertEqual(len(summary), 18)
                self.assertEqual(set(summary.n_subjects), {1})
                full = pd.read_csv(next((Path(directory) / 'subject_gene_attention').glob('gene_first*.csv.gz')))
                self.assertEqual(full.Subject_ID.tolist(), [410]*3 + [99]*3)
                self.assertEqual(full.Patient_Index.tolist(), [2]*3 + [3]*3)
                # Incoming edge weights are probabilities for each target and subject.
                sums = full.groupby(['Subject_ID', 'Actual_To']).Weight.sum()
                np.testing.assert_allclose(sums, 1, atol=1e-6)

    def test_normalization_pipeline_writes_keyed_fold_and_layer_outputs(self):
        from normalize_edges import process
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/shared').mkdir(parents=True)
            (root / 'data/metadata').mkdir()
            edges = np.array([[0, 0, 1], [1, 2, 2]])
            np.save(root / 'data/shared/gene_edge_index.npy', edges)
            pd.DataFrame({'gene_index': [0, 1, 2], 'gene_name': ['A', 'B', 'C']}).to_csv(root / 'data/metadata/genes.csv', index=False)
            inputs = []
            for fold in [1, 2]:
                rows = []
                for layer in ['gene_first', 'gene_block', 'gene_last']:
                    for source, target, weight in [(1, 2, .5), (0, 2, .8), (0, 1, .2)]:
                        rows.append({'Fold': fold, 'Layer': layer, 'Class': 't2ds', 'Head': 0,
                            'Actual_From': source, 'Actual_To': target, 'Weight': weight, 'n_subjects': fold})
                path = root / f'fold{fold}.csv'
                pd.DataFrame(rows).to_csv(path, index=False)
                inputs.append(path)
            process(inputs, root / 'data', root / 'output')
            result = pd.read_csv(root / 'output/minmax/gene_edges.csv')
            np.testing.assert_allclose(result.Weight, [0, 1, 0])
            self.assertEqual(result.From_Gene.tolist(), ['A', 'A', 'B'])
            self.assertTrue((root / 'output/rownorm/t2ds_head0_filtered_edges.csv').exists())


if __name__ == '__main__':
    unittest.main()
