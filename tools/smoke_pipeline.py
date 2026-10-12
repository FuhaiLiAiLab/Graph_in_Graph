"""Synthetic fixture for training tests; no patient data or search runner."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def make_fixture(root):
    for d in ['metadata','shared','splits']:
        (root/d).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'gene_index':range(4),'gene_name':['A','B','C','D']}).to_csv(root/'metadata/genes.csv',index=False)
    channels = ['rna','core_promoter','proximal_promoter','distal_promoter','upstream','downstream']
    pd.DataFrame({'channel_index':range(6),'channel_name':channels}).to_csv(root/'metadata/channels.csv',index=False)
    pd.DataFrame({'subfeature_node_idx':[0,1],'subfeature_name':['feature-1','feature-2']}).to_csv(root/'metadata/subfeature_nodes.csv',index=False)
    (root/'metadata/source_manifest.json').write_text('[]\n',encoding='utf-8')
    (root/'config.json').write_text(json.dumps(dict(n_genes=4,channels=channels,n_subfeature_nodes=2,n_phenotype_features=42)),encoding='utf-8')
    subjects = np.arange(100,115,dtype=np.int64)
    labels = np.tile([0,1,2],5)
    folds = np.repeat(np.arange(1,6),3)
    cohort = pd.DataFrame({'row_index':np.arange(15),'subject':subjects,'label':labels,
                          'class_name':np.array(['t2ds','pret2ds','no_t2ds'])[labels],
                          'family_id':subjects,'fold':folds})
    cohort.to_csv(root/'splits/subject_folds.csv',index=False)
    cohort.to_csv(root/'metadata/cohort.csv',index=False)
    rng = np.random.default_rng(12)
    genes = rng.normal(size=(15,4,6)).astype(np.float32)
    np.save(root/'shared/gene_raw.npy',genes)
    np.save(root/'shared/gene_edge_index.npy',np.array([[0,1,2,3],[1,2,3,0]],dtype=np.int64))
    for fold in tqdm(range(1,6), desc='Synthetic fixture folds', unit='fold'):
        for role in ['train','test']:
            idx = np.flatnonzero(folds == fold if role == 'test' else folds != fold)
            folder = root/f'folds/fold_{fold:02d}'/role
            folder.mkdir(parents=True, exist_ok=True)
            cohort.iloc[idx].to_csv(folder/'subjects.csv',index=False)
            n = len(idx)
            x = rng.normal(size=(n,42)).astype(np.float32)
            arrays = dict(subject_ids=subjects[idx],labels=labels[idx],labels_onehot=np.eye(3,dtype=np.int64)[labels[idx]],
                          gene_x=genes[idx].reshape(n,-1),phenotype_x=x,phenotype_bins=np.ones((n,40),dtype=np.int8),
                          graph_x=np.vstack([np.zeros((2,42),np.float32),x]),
                          edge_index=np.array([np.arange(2,n+2),np.zeros(n)],dtype=np.int64),node_index=np.arange(2,n+2))
            for name, arr in arrays.items():
                np.save(folder/f'{name}.npy',arr)
