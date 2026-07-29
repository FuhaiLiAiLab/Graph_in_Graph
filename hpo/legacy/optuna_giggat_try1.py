"""
3-Phase Optuna hyperparameter optimization for GIG_GAT model.

Phase 1 (Coarse):  80 trials, 1 fold,  150 epochs  -> fast broad exploration
Phase 2 (Refine):  40 trials, 2 folds, 300 epochs  -> narrowed from Phase 1
Phase 3 (Final):   Top 3,    5 folds,  5 reps x 500 epochs -> full validation

Usage (conda):
    conda activate your_env
    python optuna_giggat.py --phase 1 --n_trials 80
    python optuna_giggat.py --phase 2 --n_trials 40
    python optuna_giggat.py --phase 3
    python optuna_giggat.py --analyze --study_name giggat_hpo_phase2

All progress is persisted to optuna_giggat.db (SQLite).
Interrupted runs resume automatically on relaunch.
"""
import os  # 导入操作系统相关模块
import sys  # 导入系统相关模块
import json  # 导入用于处理JSON数据的模块
import argparse  # 导入命令行参数解析模块
import contextlib  # 导入上下文管理器工具模块
import io  # 导入io，主要用于输入输出相关操作
import numpy as np  # 导入numpy库，主要用于科学计算
import torch  # 导入PyTorch深度学习框架
import optuna  # 导入Optuna用于超参数优化
from optuna.trial import TrialState  # 导入Optuna中Trial状态枚举

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from geo_tmain_giggat import train_geogig  # 从geo_tmain_giggat模块导入训练方法
import utils  # 导入工具模块

# ================================================================
#  Utilities  工具函数部分
# ================================================================

def make_default_args(**overrides):
    """Create default args namespace (mirrors arg_parse() in geo_tmain_giggat.py)."""
    # 创建默认参数（和geo_tmain_giggat.py中的arg_parse函数一致），可用overrides覆盖
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
    defaults.update(overrides)  # 用传入参数覆盖默认参数
    return argparse.Namespace(**defaults)  # 返回命名空间对象


def scaled_milestones(num_epochs):
    """Generate scheduler milestones proportional to total epochs."""
    # 生成与总epoch数成比例的学习率调整里程碑
    ratios = [0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    return sorted(set(max(1, int(num_epochs * r)) for r in ratios))  # 每个比例对应的epoch整数


def setup_device(args):
    # 设置cuda设备为0号卡
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    args.gpu_ids = [0]  # 只使用0号卡
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    return device  # 返回device对象


@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to devnull to silence training prints."""
    # 抑制标准输出，用于屏蔽训练中的打印输出
    with contextlib.redirect_stdout(io.StringIO()):
        yield

# ================================================================
#  Search space definitions   搜索空间定义
# ================================================================

FULL_SPACE = {
    # Training 训练参数
    'lr':                      ('float_log', 1e-4, 0.1),  # 学习率，log空间
    'weight_decay':            ('float_log', 1e-12, 1e-4),  # 权重衰减
    'gamma':                   ('float', 0.5, 0.99),  # scheduler衰减因子
    # Gene-level architecture 基因级网络结构相关参数
    'gene_dim':                ('cat', [6, 12, 18, 24]),  # 隐藏层维度备选
    'gene_num_head':           ('cat', [1, 2, 3, 6]),  # 基因层head数
    'gene_num_top_feature':    ('cat', [4, 6, 8, 10]),  # 基因层top特征数
    # GiG-level architecture GiG网络结构参数
    'gig_dim':                 ('cat', [12, 18, 24, 36]),  # 隐藏层维度
    'gig_num_head':            ('cat', [2, 3, 6]),  # GiG Head数
    'gig_input_transform_dim': ('cat', [12, 18, 24, 36]),  # GiG输入转换维度
    # Loss weights 损失权重
    'class_weight':            ('float', 0.5, 1.0),  # 分类损失权重
    'class_weight_fine':       ('float', 0.1, 1.0),  # 细分类损失权重
    'ortho_weight':            ('float_log', 0.001, 0.2),  # 正交损失权重
    'link_weight':             ('float_log', 0.001, 0.2),  # 链接损失权重
    'ent_weight':              ('float', 0.0, 0.1),  # 熵损失权重
}

ORIG_BOUNDS = {
    'lr': (1e-4, 0.1), 'weight_decay': (1e-12, 1e-4),
    'gamma': (0.5, 0.99), 'class_weight': (0.5, 1.0),
    'class_weight_fine': (0.1, 1.0), 'ortho_weight': (0.001, 0.2),
    'link_weight': (0.001, 0.2), 'ent_weight': (0.0, 0.1),
}  # 各参数的原始范围


def _suggest(trial, name, spec):
    """Suggest a single parameter from its spec tuple."""
    # 从参数的spec元组中建议一个参数
    ptype = spec[0]
    if ptype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])  # 连续型参数
    elif ptype == 'float_log':
        return trial.suggest_float(name, spec[1], spec[2], log=True)  # log空间的连续型参数
    elif ptype == 'cat':
        return trial.suggest_categorical(name, spec[1])  # 离散型参数


def apply_trial_params(trial, args, space):
    """Sample all hyperparameters from *space* and write them onto *args*."""
    # 从搜索空间采样超参数并写入参数对象
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
    """Apply a plain dict (from a completed trial) onto *args*."""
    # 将已完成Trial的超参数dict应用到参数对象args上
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
#  Phase 2: narrow search space from Phase 1 top trials  第二阶段：从第一阶段结果缩小参数空间
# ================================================================

def compute_narrowed_space(study, top_n=15):
    """Build a tighter search space from the best *top_n* trials of a study."""
    # 依据study中最优top_n个Trial，收窄参数空间
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print('[WARN] No completed trials found, falling back to full space.')
        return FULL_SPACE

    top_n = min(top_n, len(completed))
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]
    narrowed = {}

    # --- Continuous params (log-scale) ---   部分参数使用log空间
    for name in ['lr', 'weight_decay', 'ortho_weight', 'link_weight']:
        vals = [t.params[name] for t in top_trials if name in t.params]
        if vals:
            lo = max(min(vals) * 0.3, ORIG_BOUNDS[name][0])
            hi = min(max(vals) * 3.0, ORIG_BOUNDS[name][1])
            if lo >= hi:
                lo, hi = ORIG_BOUNDS[name]
            narrowed[name] = ('float_log', lo, hi)

    # --- Continuous params (linear scale) --- 线性空间的参数
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

    # --- Categorical params --- 离散类别参数收敛
    for name in ['gene_dim', 'gene_num_head', 'gene_num_top_feature',
                 'gig_dim', 'gig_num_head', 'gig_input_transform_dim']:
        vals = sorted(set(t.params[name] for t in top_trials if name in t.params))
        if len(vals) >= 2:
            narrowed[name] = ('cat', vals)
        # single value -> keep full space for exploration 单类别时保持全空间

    # Fill missing keys from full space 补全缺失键
    for name in FULL_SPACE:
        if name not in narrowed:
            narrowed[name] = FULL_SPACE[name]

    return narrowed

# ================================================================
#  Generic objective function  通用optuna目标函数
# ================================================================

def objective(trial, k_folds, num_epochs, unchanged_threshold, verbose, space):
    args = make_default_args(
        num_epochs=num_epochs,
        unchanged_threshold=unchanged_threshold,
        milestones=scaled_milestones(num_epochs),
    )  # 创建参数
    apply_trial_params(trial, args, space)  # 超参数采样
    device = setup_device(args)  # 设置设备

    fold_accs = []  # 记录每折准确率
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
              f'acc={acc:.4f}  running_mean={mean_acc:.4f}')

        if k_folds > 1:
            trial.report(mean_acc, fold_n)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return np.mean(fold_accs)  # 返回平均准确率

# ================================================================
#  Phase runners   各阶段运行器
# ================================================================

def run_phase1(opts):
    """Phase 1 -- Coarse search: 1 fold, 150 epochs, full space."""
    n_trials = opts.n_trials or 80  # 阶段一默认80次Trial
    print('=' * 60)
    print('PHASE 1: Coarse Search')
    print(f'  Trials={n_trials}  Folds=1  Epochs=150  Threshold=30')
    print('=' * 60)

    study = optuna.create_study(
        direction='maximize',
        study_name='giggat_hpo_phase1',
        sampler=optuna.samplers.TPESampler(seed=opts.seed),
        pruner=optuna.pruners.NopPruner(),          # no fold-level pruning (1 fold) 不做中途裁剪
        storage=opts.db_path,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, k_folds=1, num_epochs=150,
            unchanged_threshold=30, verbose=opts.verbose, space=FULL_SPACE,
        ),
        n_trials=n_trials,
    )
    _print_summary(study, 'Phase 1')
    return study


def run_phase2(opts):
    """Phase 2 -- Refined search: 2 folds, 300 epochs, narrowed space."""
    # Load Phase 1 results 加载阶段一结果
    try:
        p1 = optuna.load_study(study_name='giggat_hpo_phase1', storage=opts.db_path)
    except Exception:
        print('ERROR: Phase 1 study not found. Please run --phase 1 first.')
        sys.exit(1)

    narrowed = compute_narrowed_space(p1, top_n=15)  # 收窄空间
    n_trials = opts.n_trials or 40  # 默认40轮Trial

    print('=' * 60)
    print('PHASE 2: Refined Search (narrowed from Phase 1)')
    print(f'  Trials={n_trials}  Folds=2  Epochs=300  Threshold=50')
    _print_narrowed_diff(narrowed)
    print('=' * 60)

    study = optuna.create_study(
        direction='maximize',
        study_name='giggat_hpo_phase2',
        sampler=optuna.samplers.TPESampler(seed=opts.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=opts.db_path,
        load_if_exists=True,
    )

    # Seed Phase 2 with Phase 1's top 5 (only on fresh start)
    # 阶段二优先用阶段一最优的5组参数进行预填充（仅在新建study时）
    if len(study.trials) == 0:
        completed = [t for t in p1.trials if t.state == TrialState.COMPLETE]
        top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        for t in top5:
            study.enqueue_trial(t.params)
        print(f'  Seeded {len(top5)} trials from Phase 1 best results.')

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
    """Phase 3 -- Final validation: top 3 from Phase 1 or 2, 5 folds x 5 reps x 500 epochs."""
    # 阶段三：验证阶段，取最优的3组，5折交叉验证，每折做5次（总计25次）
    # 支持 --from_phase1 参数，直接读取 Phase 1 的结果
    if opts.from_phase1:
        study_name = 'giggat_hpo_phase1'
        source_label = 'Phase 1'
    else:
        study_name = 'giggat_hpo_phase2'
        source_label = 'Phase 2'

    try:
        study = optuna.load_study(study_name=study_name, storage=opts.db_path)
    except Exception:
        print(f'ERROR: {source_label} study not found. Please run the corresponding phase first.')
        sys.exit(1)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    top3 = sorted(completed, key=lambda t: t.value, reverse=True)[:3]

    print('=' * 60)
    print('PHASE 3: Final Validation')
    print(f'  Top 3 from {source_label}  |  5 folds x 5 repeats x 500 epochs')
    print('=' * 60)

    results = []  # 最终保存排名结果
    for rank, trial in enumerate(top3, 1):
        print(f'\n--- Rank {rank}  ({source_label} Trial #{trial.number}, '
              f'{source_label} score={trial.value:.4f}) ---')
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

    # Final comparison 结果对比汇总
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

    out_path = 'optuna_phase3_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nDetailed results saved to {out_path}')

# ================================================================
#  Analysis  分析相关工具
# ================================================================

def analyze_study(opts):
    """Load a study and print statistics, importance, top-10."""
    # 加载并分析study，打印统计、参数重要性、top10排名
    name = opts.study_name or 'giggat_hpo_phase2'
    try:
        study = optuna.load_study(study_name=name, storage=opts.db_path)
    except Exception:
        print(f'ERROR: Study "{name}" not found in {opts.db_path}')
        sys.exit(1)

    _print_summary(study, name)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    # Parameter importance   参数重要性评估
    if len(completed) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            print('\nParameter Importance:')
            for pname, imp in importance.items():
                bar = '#' * int(imp * 50)
                print(f'  {pname:30s} {imp:.4f}  {bar}')
        except Exception:
            print('\n(Parameter importance unavailable)')

    # Top-10 table    Top10结果表
    if completed:
        top10 = sorted(completed, key=lambda t: t.value, reverse=True)[:10]
        print(f'\nTop 10 Trials:')
        print(f'  {"#":<5} {"Trial":<7} {"Accuracy":<10} {"lr":<10} '
              f'{"gene_dim":<10} {"gig_dim":<10} {"gig_head":<10}')
        print(f'  {"-"*5} {"-"*7} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
        for i, t in enumerate(top10, 1):
            p = t.params
            print(f'  {i:<5} {t.number:<7} {t.value:<10.4f} '
                  f'{p.get("lr", 0):<10.4g} '
                  f'{p.get("gene_dim", "?"):<10} '
                  f'{p.get("gig_dim", "?"):<10} '
                  f'{p.get("gig_num_head", "?"):<10}')

    # Save CSV 结果保存为CSV
    df = study.trials_dataframe()
    csv_path = f'optuna_{name}_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nFull results saved to: {csv_path}')

# ================================================================
#  Printing helpers  打印辅助函数
# ================================================================

def _print_summary(study, label=''):
    # 打印study的总结信息
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
    """Show which parameters were narrowed compared to FULL_SPACE."""
    # 打印哪些参数空间被收窄了
    print('  Narrowed parameters:')
    for name in sorted(narrowed):
        if narrowed[name] != FULL_SPACE[name]:
            full = FULL_SPACE[name]
            narrw = narrowed[name]
            print(f'    {name}: {full} -> {narrw}')

# ================================================================
#  CLI 命令行解析入口
# ================================================================

def parse_optuna_args():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='3-Phase Optuna HPO for GIG_GAT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python optuna_giggat.py --phase 1 --n_trials 80
  python optuna_giggat.py --phase 2 --n_trials 40
  python optuna_giggat.py --phase 3
  python optuna_giggat.py --analyze --study_name giggat_hpo_phase2
""")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Phase to run: 1=coarse, 2=refined, 3=final validation')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze a completed study (use --study_name to pick)')
    parser.add_argument('--n_trials', type=int, default=None,
                        help='Override number of trials (Phase 1 default=80, Phase 2 default=40)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show full training output (extremely noisy)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Study name for --analyze mode')
    parser.add_argument('--from_phase1', action='store_true',
                        help='For Phase 3: use Phase 1 results instead of Phase 2')
    parser.add_argument('--db_path', type=str, default='sqlite:///optuna_giggat.db',
                        help='SQLite storage path (default: sqlite:///optuna_giggat.db)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for TPE sampler (default: 42)')
    return parser.parse_args()  # 解析并返回参数对象

if __name__ == "__main__":
    # 程序主入口
    opts = parse_optuna_args()  # 解析命令行
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # 降低Optuna日志等级

    if opts.analyze:
        analyze_study(opts)  # 如果选分析，执行分析
    elif opts.phase == 1:
        run_phase1(opts)  # 阶段一
    elif opts.phase == 2:
        run_phase2(opts)  # 阶段二
    elif opts.phase == 3:
        run_phase3(opts)  # 阶段三
    else:
        # 打印帮助信息和阶段说明
        print('3-Phase Optuna HPO for GIG_GAT')
        print('=' * 45)
        print('Phase 1: Coarse search')
        print('  python optuna_giggat.py --phase 1 --n_trials 80')
        print('  1 fold, 150 epochs, full parameter space')
        print()
        print('Phase 2: Refined search')
        print('  python optuna_giggat.py --phase 2 --n_trials 40')
        print('  2 folds, 300 epochs, narrowed from Phase 1')
        print()
        print('Phase 3: Final validation')
        print('  python optuna_giggat.py --phase 3')
        print('  Top 3 params, 5 folds x 5 reps x 500 epochs')
        print()
        print('Analyze results:')
        print('  python optuna_giggat.py --analyze --study_name giggat_hpo_phase2')
