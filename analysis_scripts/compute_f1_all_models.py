import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gnn_result')

models = {
    'GiG': {
        'dir': 'gigtransformer_GinG',
        'folds': [('epoch_500_fold1-31','result_test_label_1.csv'),
                  ('epoch_500_fold2-24','result_test_label_2.csv'),
                  ('epoch_500_fold3-24','result_test_label_3.csv'),
                  ('epoch_500_fold4-23','result_test_label_4.csv'),
                  ('epoch_500_fold5-21','result_test_label_5.csv')]
    },
    'GiG-Pheno': {
        'dir': 'gigtransformer_subject',
        'folds': [('epoch_500_fold1-5','result_test_label_1.csv'),
                  ('epoch_500_fold2-1','result_test_label_2.csv'),
                  ('epoch_500_fold3-4','result_test_label_3.csv'),
                  ('epoch_500_fold4-1','result_test_label_4.csv'),
                  ('epoch_500_fold5-4','result_test_label_5.csv')]
    },
    'GiG-Gene': {
        'dir': 'gigtransformer_gene',
        'folds': [('epoch_500_fold1-5','result_test_label_1.csv'),
                  ('epoch_500_fold2-4','result_test_label_2.csv'),
                  ('epoch_500_fold3-2','result_test_label_3.csv'),
                  ('epoch_500_fold4-2','result_test_label_4.csv'),
                  ('epoch_500_fold5-4','result_test_label_5.csv')]
    },
    'GAT': {
        'dir': 'gat',
        'folds': [('epoch_500_fold1-5','result_test_label_1.csv'),
                  ('epoch_500_fold2-2','result_test_label_2.csv'),
                  ('epoch_500_fold3-3','result_test_label_3.csv'),
                  ('epoch_500_fold4-5','result_test_label_4.csv'),
                  ('epoch_500_fold5-5','result_test_label_5.csv')]
    },
    'GIN': {
        'dir': 'gin',
        'folds': [('epoch_500_fold1-5','result_test_label_1.csv'),
                  ('epoch_500_fold2-4','result_test_label_2.csv'),
                  ('epoch_500_fold3-4','result_test_label_3.csv'),
                  ('epoch_500_fold4-5','result_test_label_4.csv'),
                  ('epoch_500_fold5-5','result_test_label_5.csv')]
    }
}

print(f"{'Model':<12} | {'Fold1':>6} | {'Fold2':>6} | {'Fold3':>6} | {'Fold4':>6} | {'Fold5':>6} | {'Mean':>6} | {'SD':>6}")
print('-' * 80)
for model_name, info in models.items():
    f1_list = []
    for folder, fname in info['folds']:
        path = f"{base}\\{info['dir']}\\{folder}\\{fname}"
        df = pd.read_csv(path)
        f1 = f1_score(df['test_label'], df['test_pred_label'], average='weighted')
        f1_list.append(f1)
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    fold_str = ' | '.join([f'{v:.4f}' for v in f1_list])
    print(f"{model_name:<12} | {fold_str} | {mean_f1:.4f} | {std_f1:.4f}")
