"""
第二次 Optuna 超参数优化 for GIG_GAT model.
基于第一次失败经验 + 导师建议改进：
  1. weight_decay = e^-15 ~ e^-10（约 3e-7 ~ 5e-5）
  2. 不用 warmup，直接用 MultiStepLR（LR 从最高开始，按 milestone 衰减）
  3. 基于 Phase 1 分析收窄网络结构搜索空间
  4. Phase 2 收窄逻辑改进：只用高质量 trial（acc > 阈值）

Phase 1 (Coarse):  80 trials, 1 fold,  200 epochs  -> 粗搜索
Phase 2 (Refine):  40 trials, 2 folds, 300 epochs  -> 精搜索
Phase 3 (Final):   Top 3,    5 folds,  5 reps x 500 epochs -> 最终验证

Usage:
    python optuna_giggat_try2.py --phase 1 --n_trials 80
    python optuna_giggat_try2.py --phase 2 --n_trials 40
    python optuna_giggat_try2.py --phase 3
    python optuna_giggat_try2.py --analyze --study_name giggat_try2_phase1
"""
import os
import sys
import json
import argparse
import contextlib
import io
import numpy as np
import torch
import optuna
from optuna.trial import TrialState

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from geo_tmain_giggat import train_geogig  # 直接用原始训练函数，不需要 warmup
import utils

# ================================================================
#  工具函数
# ================================================================

def make_default_args(**overrides):
    """创建默认参数，和 geo_tmain_giggat.py 中的 arg_parse() 一致"""
    defaults = dict(
        cuda='0', parallel=False, add_self='0', adj='0', model='0',
        lr=0.01, weight_decay=1e-10,
        milestones=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250],
        gamma=0.9, clip=5.0, batch_size=16, num_epochs=500,
        unchanged_threshold=100, change_wave=0.8, num_workers=0,
        graph_opt='GinG',
        gene_input_dim=6, gene_hidden_dim=6, gene_output_dim=6,
        gene_num_top_feature=6,
        gig_input_dim=42, gig_input_transform_dim=18,
        gig_hidden_dim=18, gig_output_dim=18,
        class_weight_fine=0.5, class_weight=0.9,
        ortho_weight=0.05, link_weight=0.05, ent_weight=0.00,
        num_classes=3, gene_num_head=1, gig_num_head=6,
        dropout=0.01, gpu_ids=[0],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def scaled_milestones(num_epochs):
    """生成与总 epoch 数成比例的学习率调整 milestone"""
    ratios = [0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    return sorted(set(max(1, int(num_epochs * r)) for r in ratios))


def setup_device(args):
    """设置 CUDA 设备"""
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    args.gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    return device


@contextlib.contextmanager
def suppress_stdout():
    """抑制标准输出"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

# ================================================================
#  搜索空间定义（基于第一次失败经验 + 导师建议调整）
# ================================================================

# 第一次 Phase 1 分析结论：
#   - 好的 trial (76.5%): lr=0.0001~0.006, wd=1e-7~2e-6, gene_dim=6/12, gig_dim=12/24
#   - weight_decay > 1e-5 基本全失败
#   - 导师建议 wd = e^-15(3e-7) ~ e^-10(5e-5)
#   - 不需要 warmup，原始 MultiStepLR 已经是"前面LR大，后面小"

FULL_SPACE = {
    # === 训练参数 ===
    'lr':                      ('float_log', 1e-4, 0.01),      # 第一次好的 trial: 0.0001~0.006
    'weight_decay':            ('float_log', 3e-7, 5e-5),      # 导师建议：e^-15(3e-7) 到 e^-10(5e-5)
    'gamma':                   ('float', 0.8, 0.95),           # 第一次好的 trial 在 0.74~0.97

    # === Gene-level 网络结构（基于 Phase 1 分析收窄） ===
    'gene_dim':                ('cat', [6, 12]),                # Phase 1: 6 和 12 占据所有 top 结果
    'gene_num_head':           ('cat', [2, 3]),                 # Phase 1: 2 最好，3 也可以
    'gene_num_top_feature':    ('cat', [4, 6, 8, 10]),         # 保持宽泛探索

    # === GiG-level 网络结构（基于 Phase 1 分析收窄） ===
    'gig_dim':                 ('cat', [12, 24]),               # Phase 1: 12 和 24 占据 top 结果
    'gig_num_head':            ('cat', [2, 3, 6]),              # 保持宽泛
    'gig_input_transform_dim': ('cat', [12, 18, 24, 36]),      # 保持宽泛

    # === 损失权重 ===
    'class_weight':            ('float', 0.5, 1.0),
    'class_weight_fine':       ('float', 0.1, 1.0),
    'ortho_weight':            ('float_log', 0.001, 0.2),
    'link_weight':             ('float_log', 0.001, 0.2),
    'ent_weight':              ('float', 0.0, 0.1),
}

ORIG_BOUNDS = {
    'lr': (1e-4, 0.01), 'weight_decay': (3e-7, 5e-5),
    'gamma': (0.8, 0.95), 'class_weight': (0.5, 1.0),
    'class_weight_fine': (0.1, 1.0), 'ortho_weight': (0.001, 0.2),
    'link_weight': (0.001, 0.2), 'ent_weight': (0.0, 0.1),
}


def _suggest(trial, name, spec):
    """从 spec 元组中建议一个参数"""
    ptype = spec[0]
    if ptype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    elif ptype == 'float_log':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif ptype == 'cat':
        return trial.suggest_categorical(name, spec[1])


def apply_trial_params(trial, args, space):
    """从搜索空间采样超参数并写入 args"""
    args.lr = _suggest(trial, 'lr', space['lr'])
    args.weight_decay = _suggest(trial, 'weight_decay', space['weight_decay'])
    args.gamma = _suggest(trial, 'gamma', space['gamma'])

    gene_dim = _suggest(trial, 'gene_dim', space['gene_dim'])
    args.gene_hidden_dim = gene_dim
    args.gene_output_dim = gene_dim
    args.gene_num_head = _suggest(trial, 'gene_num_head', space['gene_num_head'])
    args.gene_num_top_feature = _suggest(trial, 'gene_num_top_feature', space['gene_num_top_feature'])

    gig_dim = _suggest(trial, 'gig_dim', space['gig_dim'])
    args.gig_hidden_dim = gig_dim
    args.gig_output_dim = gig_dim
    args.gig_num_head = _suggest(trial, 'gig_num_head', space['gig_num_head'])
    args.gig_input_transform_dim = _suggest(trial, 'gig_input_transform_dim', space['gig_input_transform_dim'])

    args.class_weight = _suggest(trial, 'class_weight', space['class_weight'])
    args.class_weight_fine = _suggest(trial, 'class_weight_fine', space['class_weight_fine'])
    args.ortho_weight = _suggest(trial, 'ortho_weight', space['ortho_weight'])
    args.link_weight = _suggest(trial, 'link_weight', space['link_weight'])
    args.ent_weight = _suggest(trial, 'ent_weight', space['ent_weight'])


def apply_params_dict(args, params):
    """将已完成 Trial 的参数 dict 应用到 args"""
    args.lr = params['lr']
    args.weight_decay = params['weight_decay']
    args.gamma = params['gamma']
    args.gene_hidden_dim = params['gene_dim']
    args.gene_output_dim = params['gene_dim']
    args.gene_num_head = params['gene_num_head']
    args.gene_num_top_feature = params['gene_num_top_feature']
    args.gig_hidden_dim = params['gig_dim']
    args.gig_output_dim = params['gig_dim']
    args.gig_num_head = params['gig_num_head']
    args.gig_input_transform_dim = params['gig_input_transform_dim']
    args.class_weight = params['class_weight']
    args.class_weight_fine = params['class_weight_fine']
    args.ortho_weight = params['ortho_weight']
    args.link_weight = params['link_weight']
    args.ent_weight = params['ent_weight']

# ================================================================
#  Phase 2 收窄逻辑（改进版：加入质量阈值过滤）
# ================================================================

def compute_narrowed_space(study, top_n=15, min_acc=0.3):
    """
    从 Phase 1 最优 trial 收窄搜索空间。
    改进：只用 accuracy > min_acc 的 trial，避免低质量参数污染。
    第一次失败原因：top 15 混入了 10-14% 准确率的 trial，导致收窄方向错误。
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print('[WARN] 没有完成的 trial，使用完整搜索空间')
        return FULL_SPACE

    # 只保留准确率高于阈值的 trial
    quality_trials = [t for t in completed if t.value >= min_acc]
    if len(quality_trials) < 3:
        print(f'[WARN] 只有 {len(quality_trials)} 个 trial 准确率 > {min_acc}，使用完整搜索空间')
        return FULL_SPACE

    top_n = min(top_n, len(quality_trials))
    top_trials = sorted(quality_trials, key=lambda t: t.value, reverse=True)[:top_n]

    print(f'  收窄使用 {len(top_trials)} 个高质量 trial（acc > {min_acc}）')
    print(f'  这些 trial 的准确率范围: {top_trials[-1].value:.4f} ~ {top_trials[0].value:.4f}')

    narrowed = {}

    # --- Log 空间连续参数 ---
    for name in ['lr', 'weight_decay', 'ortho_weight', 'link_weight']:
        vals = [t.params[name] for t in top_trials if name in t.params]
        if vals:
            lo = max(min(vals) * 0.3, ORIG_BOUNDS[name][0])
            hi = min(max(vals) * 3.0, ORIG_BOUNDS[name][1])
            if lo >= hi:
                lo, hi = ORIG_BOUNDS[name]
            narrowed[name] = ('float_log', lo, hi)

    # --- 线性空间连续参数 ---
    for name in ['gamma', 'class_weight', 'class_weight_fine', 'ent_weight']:
        vals = [t.params[name] for t in top_trials if name in t.params]
        if vals:
            spread = max(vals) - min(vals)
            buf = max(spread * 0.3, 0.02)
            lo = max(min(vals) - buf, ORIG_BOUNDS[name][0])
            hi = min(max(vals) + buf, ORIG_BOUNDS[name][1])
            if lo >= hi:
                lo, hi = ORIG_BOUNDS[name]
            narrowed[name] = ('float', lo, hi)

    # --- 类别参数 ---
    for name in ['gene_dim', 'gene_num_head', 'gene_num_top_feature',
                 'gig_dim', 'gig_num_head', 'gig_input_transform_dim']:
        vals = sorted(set(t.params[name] for t in top_trials if name in t.params))
        if len(vals) >= 2:
            narrowed[name] = ('cat', vals)
        elif len(vals) == 1:
            narrowed[name] = ('cat', vals)

    # 补全缺失键
    for name in FULL_SPACE:
        if name not in narrowed:
            narrowed[name] = FULL_SPACE[name]

    return narrowed

# ================================================================
#  目标函数
# ================================================================

def objective(trial, k_folds, num_epochs, unchanged_threshold, verbose, space):
    """Optuna 目标函数：采样参数 -> 训练 -> 返回平均准确率"""
    args = make_default_args(
        num_epochs=num_epochs,
        unchanged_threshold=unchanged_threshold,
        milestones=scaled_milestones(num_epochs),
    )
    apply_trial_params(trial, args, space)
    device = setup_device(args)

    fold_accs = []
    for fold_n in range(1, k_folds + 1):
        try:
            if verbose:
                acc = train_geogig(args, fold_n, 1, device)
            else:
                with suppress_stdout():
                    acc = train_geogig(args, fold_n, 1, device)
        except Exception as e:
            print(f'  [Trial {trial.number}] Fold {fold_n} FAILED: {e}')
            raise optuna.exceptions.TrialPruned()

        fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        print(f'  [Trial {trial.number}] Fold {fold_n}/{k_folds} '
              f'acc={acc:.4f}  running_mean={mean_acc:.4f}  '
              f'(lr={args.lr:.4g}, wd={args.weight_decay:.2g})')

        if k_folds > 1:
            trial.report(mean_acc, fold_n)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return np.mean(fold_accs)

# ================================================================
#  各阶段运行器
# ================================================================

def run_phase1(opts):
    """Phase 1 -- 粗搜索: 1 fold, 200 epochs"""
    n_trials = opts.n_trials or 80
    print('=' * 60)
    print('PHASE 1: 粗搜索（导师建议 + 第一次经验改进）')
    print(f'  Trials={n_trials}  Folds=1  Epochs=200  Threshold=40')
    print(f'  改进: wd=e^-15~e^-10, lr=1e-4~0.01, 收窄网络结构')
    print('=' * 60)

    study = optuna.create_study(
        direction='maximize',
        study_name='giggat_try2_phase1',
        sampler=optuna.samplers.TPESampler(seed=opts.seed),
        pruner=optuna.pruners.NopPruner(),
        storage=opts.db_path,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, k_folds=1, num_epochs=200,
            unchanged_threshold=40, verbose=opts.verbose, space=FULL_SPACE,
        ),
        n_trials=n_trials,
    )
    _print_summary(study, 'Phase 1')
    return study


def run_phase2(opts):
    """Phase 2 -- 精搜索: 2 folds, 300 epochs, 收窄空间"""
    try:
        p1 = optuna.load_study(study_name='giggat_try2_phase1', storage=opts.db_path)
    except Exception:
        print('ERROR: Phase 1 study not found. Run --phase 1 first.')
        sys.exit(1)

    narrowed = compute_narrowed_space(p1, top_n=10, min_acc=0.3)
    n_trials = opts.n_trials or 40

    print('=' * 60)
    print('PHASE 2: 精搜索（从 Phase 1 高质量 trial 收窄）')
    print(f'  Trials={n_trials}  Folds=2  Epochs=300  Threshold=50')
    _print_narrowed_diff(narrowed)
    print('=' * 60)

    study = optuna.create_study(
        direction='maximize',
        study_name='giggat_try2_phase2',
        sampler=optuna.samplers.TPESampler(seed=opts.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=opts.db_path,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        completed = [t for t in p1.trials if t.state == TrialState.COMPLETE]
        top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        for t in top5:
            study.enqueue_trial(t.params)
        print(f'  Seeded {len(top5)} trials from Phase 1 best.')

    study.optimize(
        lambda trial: objective(
            trial, k_folds=2, num_epochs=300,
            unchanged_threshold=50, verbose=opts.verbose, space=narrowed,
        ),
        n_trials=n_trials,
    )
    _print_summary(study, 'Phase 2')
    return study


def run_phase3(opts):
    """Phase 3 -- 最终验证: top 3, 5 folds x 5 reps x 500 epochs"""
    if opts.from_phase1:
        study_name = 'giggat_try2_phase1'
        source_label = 'Phase 1'
    else:
        study_name = 'giggat_try2_phase2'
        source_label = 'Phase 2'

    try:
        study = optuna.load_study(study_name=study_name, storage=opts.db_path)
    except Exception:
        print(f'ERROR: {source_label} study not found.')
        sys.exit(1)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    top3 = sorted(completed, key=lambda t: t.value, reverse=True)[:3]

    print('=' * 60)
    print('PHASE 3: Final Validation')
    print(f'  Top 3 from {source_label}  |  5 folds x 5 repeats x 500 epochs')
    print('=' * 60)

    results = []
    for rank, trial in enumerate(top3, 1):
        print(f'\n--- Rank {rank}  ({source_label} Trial #{trial.number}, '
              f'score={trial.value:.4f}) ---')
        for k, v in trial.params.items():
            print(f'    {k}: {v}')

        args = make_default_args(num_epochs=500, unchanged_threshold=100)
        apply_params_dict(args, trial.params)
        device = setup_device(args)

        all_accs = []
        for fold_n in range(1, 6):
            for nth in range(1, 6):
                if opts.verbose:
                    acc = train_geogig(args, fold_n, nth, device)
                else:
                    with suppress_stdout():
                        acc = train_geogig(args, fold_n, nth, device)
                all_accs.append(acc)
                print(f'    Fold {fold_n} Rep {nth}: acc={acc:.4f}')

        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        results.append(dict(
            rank=rank, source_trial=trial.number, source_phase=source_label,
            params=trial.params,
            mean_acc=float(mean_acc), std_acc=float(std_acc),
            all_accs=[float(a) for a in all_accs],
        ))
        print(f'  => Rank {rank} RESULT: {mean_acc:.4f} +/- {std_acc:.4f}')

    print('\n' + '=' * 60)
    print('PHASE 3 FINAL RESULTS')
    print('=' * 60)
    for r in results:
        print(f"  Rank {r['rank']}  ({r['source_phase']} Trial #{r['source_trial']}): "
              f"{r['mean_acc']:.4f} +/- {r['std_acc']:.4f}")

    best = max(results, key=lambda x: x['mean_acc'])
    print(f"\nBEST: Rank {best['rank']}  accuracy={best['mean_acc']:.4f} +/- {best['std_acc']:.4f}")
    print('Parameters:')
    for k, v in best['params'].items():
        print(f'  {k}: {v}')

    out_path = 'optuna_try2_phase3_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')

# ================================================================
#  分析工具
# ================================================================

def analyze_study(opts):
    """加载并分析 study"""
    name = opts.study_name or 'giggat_try2_phase1'
    try:
        study = optuna.load_study(study_name=name, storage=opts.db_path)
    except Exception:
        print(f'ERROR: Study "{name}" not found')
        sys.exit(1)

    _print_summary(study, name)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(completed) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            print('\nParameter Importance:')
            for pname, imp in importance.items():
                bar = '#' * int(imp * 50)
                print(f'  {pname:30s} {imp:.4f}  {bar}')
        except Exception:
            print('\n(Parameter importance unavailable)')

    if completed:
        top10 = sorted(completed, key=lambda t: t.value, reverse=True)[:10]
        print(f'\nTop 10 Trials:')
        print(f'  {"#":<5} {"Trial":<7} {"Acc":<8} {"lr":<10} {"wd":<12} '
              f'{"gamma":<8} {"gene_d":<8} {"gig_d":<8} {"g_head":<8}')
        print(f'  {"-"*5} {"-"*7} {"-"*8} {"-"*10} {"-"*12} '
              f'{"-"*8} {"-"*8} {"-"*8} {"-"*8}')
        for i, t in enumerate(top10, 1):
            p = t.params
            print(f'  {i:<5} {t.number:<7} {t.value:<8.4f} '
                  f'{p.get("lr", 0):<10.4g} '
                  f'{p.get("weight_decay", 0):<12.2g} '
                  f'{p.get("gamma", 0):<8.3f} '
                  f'{p.get("gene_dim", "?"):<8} '
                  f'{p.get("gig_dim", "?"):<8} '
                  f'{p.get("gig_num_head", "?"):<8}')

    zero_trials = [t for t in completed if t.value < 0.01]
    print(f'\n0% acc trials: {len(zero_trials)} / {len(completed)}')

    df = study.trials_dataframe()
    csv_path = f'optuna_{name}_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'Results saved to: {csv_path}')

# ================================================================
#  打印辅助
# ================================================================

def _print_summary(study, label=''):
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == TrialState.FAIL]

    print(f'\n{"=" * 60}')
    print(f'{label} STUDY SUMMARY')
    print(f'{"=" * 60}')
    print(f'  Total trials:  {len(study.trials)}')
    print(f'    Completed:   {len(complete)}')
    print(f'    Pruned:      {len(pruned)}')
    print(f'    Failed:      {len(failed)}')

    if complete:
        zero_count = sum(1 for t in complete if t.value < 0.01)
        print(f'    0% acc:      {zero_count}')
        print(f'\n  Best accuracy: {study.best_trial.value:.4f}  '
              f'(Trial #{study.best_trial.number})')
        print('  Best params:')
        for k, v in study.best_trial.params.items():
            if isinstance(v, float):
                print(f'    {k}: {v:.6g}')
            else:
                print(f'    {k}: {v}')
    print('=' * 60)


def _print_narrowed_diff(narrowed):
    print('  Narrowed parameters:')
    for name in sorted(narrowed):
        if narrowed[name] != FULL_SPACE[name]:
            full = FULL_SPACE[name]
            narrw = narrowed[name]
            print(f'    {name}: {full} -> {narrw}')

# ================================================================
#  命令行入口
# ================================================================

def parse_optuna_args():
    parser = argparse.ArgumentParser(
        description='Optuna HPO Try2 for GIG_GAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage:
  python optuna_giggat_try2.py --phase 1 --n_trials 80
  python optuna_giggat_try2.py --phase 2 --n_trials 40
  python optuna_giggat_try2.py --phase 3
  python optuna_giggat_try2.py --analyze --study_name giggat_try2_phase1
""")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3])
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--from_phase1', action='store_true')
    parser.add_argument('--db_path', type=str, default='sqlite:///optuna_giggat_try2.db')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_optuna_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if opts.analyze:
        analyze_study(opts)
    elif opts.phase == 1:
        run_phase1(opts)
    elif opts.phase == 2:
        run_phase2(opts)
    elif opts.phase == 3:
        run_phase3(opts)
    else:
        print('Optuna HPO Try2 for GIG_GAT')
        print('=' * 50)
        print()
        print('Changes vs Try1:')
        print('  1. weight_decay: e^-15(3e-7) ~ e^-10(5e-5)')
        print('  2. lr: 1e-4 ~ 0.01 (no warmup)')
        print('  3. gene_dim=[6,12], gig_dim=[12,24]')
        print('  4. Phase 2 only uses trials with acc > 30%')
        print()
        print('Phase 1:  python optuna_giggat_try2.py --phase 1 --n_trials 80')
        print('Phase 2:  python optuna_giggat_try2.py --phase 2 --n_trials 40')
        print('Phase 3:  python optuna_giggat_try2.py --phase 3')
        print('Analyze:  python optuna_giggat_try2.py --analyze')
