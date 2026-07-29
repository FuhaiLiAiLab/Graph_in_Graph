"""快速诊断: 单 fold 不抑制输出，看 train_geogig 到底返回什么"""
import torch
import numpy as np

# 导入 optuna_gigtransformer 会自动执行 monkey-patch
from optuna_gigtransformer import make_default_args, evaluate_single_fold

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 用默认参数，只跑 3 个 epoch 看看
args = make_default_args(num_epochs=3)
print(f"Args: gene_hidden={args.gene_hidden_dim}, K={args.gene_num_top_feature}, "
      f"gig_input_dim={args.gig_input_dim}, gig_transform={args.gig_input_transform_dim}, "
      f"gig_hidden={args.gig_hidden_dim}")

print("\n=== Running fold 1 WITHOUT suppressing output ===")
acc, success = evaluate_single_fold(args, fold_n=1, device=device, suppress_output=False)
print(f"\n=== Result: acc={acc}, success={success} ===")
