"""One signed RNA-weighted node score; the frozen model and attention stay unchanged."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from gene_scores import GeneScoreCollector

METHOD = 'embedding_x_assignment_l2_x_signed_input_rna'
CLASSES = {0: 't2ds', 1: 'pret2ds', 2: 'no_t2ds'}


def signed_rna_scores(embedding_norm, assignment, rna):
    h, w, r = (np.asarray(x, dtype=np.float64) for x in (embedding_norm, assignment, rna))
    if (h.ndim != 2 or w.ndim != 2 or r.shape != h.shape or w.shape[1] != h.shape[1]
            or not all(np.isfinite(x).all() for x in (h, w, r)) or (h < 0).any()):
        raise ValueError('Expected finite nonnegative N x G embedding norms, K x G weights, signed N x G RNA')
    score = h * np.linalg.norm(w, axis=0)[None, :] * r
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite RNA-weighted score')
    return score


def class_score_table(scores, labels, genes, seed, fold):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if (scores.ndim != 2 or labels.shape != (scores.shape[0],) or len(genes) != scores.shape[1]
            or not np.isfinite(scores).all() or set(labels.tolist()) != set(CLASSES)
            or not np.array_equal(genes.gene_index, np.arange(len(genes)))
            or genes.gene_name.duplicated().any()):
        raise ValueError('Incomplete or misaligned subject/class/gene score table')
    parts = []
    for label, name in CLASSES.items():
        mask = labels == label
        mean = scores[mask].mean(axis=0, dtype=np.float64)
        parts.append(pd.DataFrame(dict(Seed=seed, Fold=fold, Class=name, Method=METHOD,
            Node_idx=genes.gene_index, Gene=genes.gene_name, Weight=mean,
            Rank=pd.Series(mean).rank(ascending=False, method='average').to_numpy(),
            n_subjects=int(mask.sum()))))
    return pd.concat(parts, ignore_index=True)


class SignedRNACollector(GeneScoreCollector):
    def __init__(self, model, data):
        if model.training:
            raise ValueError('Scoring requires evaluation mode')
        if data['num_feature'] != 6 or model.graph_opt != 'GinG':
            raise ValueError('Expected the full six-channel model')
        super().__init__(model, data)
        self.rna = np.asarray(data['gene_x']).reshape(self.n, self.g, 6)[:, :, 0].copy()

    def finish_signed_scores(self):
        if any(n != self.n for n in self.cursors.values()):
            raise ValueError('Incomplete gene-layer subject coverage')
        return signed_rna_scores(self.embedding_norm, self.assignment, self.rna)

    def export_signed(self, output, genes, seed, fold):
        output = Path(output)
        scores = self.finish_signed_scores()
        table = class_score_table(scores, self.data['labels'], genes, seed, fold)
        table['In_gene_graph'] = np.tile(self.eligible, len(CLASSES))
        table.to_csv(output/'rna_gene_scores.csv', index=False)
        np.savez_compressed(output/'subject_rna_scores.npz', subject_ids=self.data['subject_ids'],
            labels=self.data['labels'], gene_indices=genes.gene_index.to_numpy(),
            embedding_norm=self.embedding_norm, assignment=self.assignment, rna=self.rna, score=scores)
        pd.DataFrame(dict(subject=self.data['subject_ids'], label=self.data['labels'])).to_csv(
            output/'scoring_subjects.csv', index=False)
        metadata = dict(method=METHOD,
            formula='norm2(last post-activation gene embedding) * norm2(actual signed assignment column) * signed RNA',
            rna_source='exact model input gene_x.reshape(N,G,6)[:,:,0]',
            negative_policy='retain sign and value; no abs/clipping/shifting/refitting',
            aggregation='multiply per individual, then mean within true-label group',
            gene_universe='all input genes; isolated genes retained and marked In_gene_graph=False',
            n_subjects=self.n, n_genes=self.g, gene_layers=list(self.cursors),
            negative_rna_values=int((self.rna<0).sum()), negative_scores=int((scores<0).sum()),
            rna_above_one=int((self.rna>1).sum()), score_minimum=float(scores.min()), score_maximum=float(scores.max()),
            interpretation='RNA-weighted descriptive pooling score; sign is not a disease-effect attribution')
        (output/'rna_scoring_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return table, scores, metadata
