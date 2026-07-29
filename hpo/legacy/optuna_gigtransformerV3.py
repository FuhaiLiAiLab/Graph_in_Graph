#!/usr/bin/env python3
"""
GiG-Transformer (copy2) 超参数优化脚本 V3
==========================================
基于 V2 失败经验重新设计：
  - 全程 500 epochs，避免 epoch 不一致导致的筛选偏差
  - 搜索空间从 15 维缩减到 6 维（固定低重要性参数）
  - Phase 1 用纯 mean_acc 排名，不做方差惩罚
  - 每 fold 跑 2 次取中位数，自动过滤训练崩溃
  - Pruner 放宽（PercentilePruner 75%），减少误杀
  - 两套已验证成功参数 (80.57%, 80.20%) 作为 seed

策略：筛选 → 验证（两阶段）
  Phase 1 (筛选): 3-fold × 2-run, 500 epochs — 50 trials, 聚焦搜索空间
  Phase 2 (验证): 5-fold × 3-run, 500 epochs — Top 5, 失败重试

架构约束：
  TransformerConv 使用 concat=True 时，输出维度 = heads × out_channels。
  gene_num_head 和 gig_num_head 固定为 1。

用法:
  python optuna_gigtransformerV3.py --phase 1 --n_trials 50   # 3-fold×2-run 筛选
  python optuna_gigtransformerV3.py --phase 2                  # 5-fold×3-run 验证
  python optuna_gigtransformerV3.py --analyze                  # 汇总分析
"""

import os
import sys
import json
import logging
import argparse
import traceback
import contextlib
import io
import numpy as np
from datetime import datetime
from contextlib import contextmanager

# 多进程参数解析修复
if __name__ != "__main__" and len(sys.argv) > 1:
    sys.argv = [sys.argv[0]]

# ============================================================
# 配置
# ============================================================
DB_PATH = "optuna_gigtransformer_v3.db"
STUDY_PREFIX = "gigtransformer_v3"
LOG_DIR = "./hpo_logs"
RESULT_DIR = "./hpo_results"

NUM_EPOCHS = 500  # 全程统一 epoch 数

# ============================================================
# 固定参数（V2 参数重要性 < 0.03 或两套成功参数一致的）
# ============================================================
FIXED_PARAMS = {
    "gamma": 0.95,
    "milestone_step": 150,
    "clip": 2.0,
    "dropout": 0.01,
    "weight_decay": 1e-10,
    "ortho_weight": 0.008,
    "link_weight": 0.002,
    "gig_hidden_dim": 24,
    "gig_input_transform_dim": 24,
}

# ============================================================
# 搜索空间（仅 6 个关键参数）
# 范围基于两套成功参数: 80.57% 和 80.20%
# ============================================================
SEARCH_SPACE = {
    # lr: 成功套1=0.004, 套2=0.01, V2重要性=62%
    "lr": {"type": "loguniform", "low": 0.002, "high": 0.015},

    # gene_hidden_dim: 套1=9, 套2=18
    "gene_hidden_dim": {"type": "categorical", "choices": [9, 12, 18]},

    # gene_num_top_feature (K): 套1=24, 套2=18
    "gene_num_top_feature": {"type": "categorical", "choices": [18, 24]},

    # class_weight: 套1=0.821, 套2=0.9
    "class_weight": {"type": "uniform", "low": 0.75, "high": 0.95},

    # class_weight_fine: 套1=0.775, 套2=0.5
    "class_weight_fine": {"type": "uniform", "low": 0.3, "high": 0.85},

    # ent_weight: 套1=0.09, 套2=0.0, V2重要性=10%
    "ent_weight": {"type": "uniform", "low": 0.0, "high": 0.1},
}

# 打印排名表的参数列
TABLE_PARAMS = [
    "lr", "gene_hidden_dim", "gene_num_top_feature",
    "class_weight", "class_weight_fine", "ent_weight",
]

# 两套已验证成功参数作为 seed（只含搜索空间内的 key）
SEED_BEST1 = {  # 80.57% ± 0.71%
    "lr": 0.003938,
    "gene_hidden_dim": 9,
    "gene_num_top_feature": 24,
    "class_weight": 0.821,
    "class_weight_fine": 0.775,
    "ent_weight": 0.090,
}

SEED_BEST2 = {  # 80.20% ± 1.65%
    "lr": 0.01,
    "gene_hidden_dim": 18,
    "gene_num_top_feature": 18,
    "class_weight": 0.9,
    "class_weight_fine": 0.5,
    "ent_weight": 0.0,
}

# 在两套成功参数中间插值的探索 seed
SEED_MID = {
    "lr": 0.006,
    "gene_hidden_dim": 12,
    "gene_num_top_feature": 18,
    "class_weight": 0.86,
    "class_weight_fine": 0.65,
    "ent_weight": 0.045,
}


# ============================================================
# 工具函数
# ============================================================

def setup_logging(phase):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"transformer_v3_phase{phase}_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def generate_milestones(step, num_epochs):
    milestones = list(range(step, num_epochs + 1, step))
    if not milestones:
        milestones = [num_epochs // 2]
    return milestones


@contextmanager
def suppress_stdout(suppress=True):
    if suppress:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def make_default_args(**overrides):
    """创建默认参数，和 geo_tmain_gigtransformer_copy2.py 的 arg_parse() 一致。"""
    defaults = dict(
        cuda='0', parallel=False, add_self='0', adj='0', model='0',
        lr=0.01, weight_decay=1e-10,
        milestones=[150, 300, 450],
        gamma=0.95, clip=2.0, batch_size=256,
        num_epochs=500,
        unchanged_threshold=100, change_wave=0.75, num_workers=0,
        graph_opt='GinG',
        gene_input_dim=6, gene_hidden_dim=18, gene_output_dim=18,
        gene_num_top_feature=18,
        gig_input_dim=42, gig_input_transform_dim=24,
        gig_hidden_dim=24, gig_output_dim=24,
        class_weight_fine=0.5, class_weight=0.9,
        ortho_weight=0.008, link_weight=0.002, ent_weight=0.00,
        num_classes=3,
        gene_num_head=1,
        gig_num_head=1,
        dropout=0.01, gpu_ids=[0],
        logdir='runs',
        run_name=None,
        log_interval=10,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ============================================================
# 参数采样与应用
# ============================================================

def sample_params(trial, space):
    params = {}
    for name, config in space.items():
        if config["type"] == "loguniform":
            params[name] = trial.suggest_float(name, config["low"], config["high"], log=True)
        elif config["type"] == "uniform":
            params[name] = trial.suggest_float(name, config["low"], config["high"])
        elif config["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, config["choices"])
        elif config["type"] == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])
    return params


def create_args_from_params(params, num_epochs):
    """将搜索参数 + 固定参数合并到完整的 args 对象中。"""
    merged = {**FIXED_PARAMS, **params}
    args = make_default_args(num_epochs=num_epochs)

    for key, value in merged.items():
        if key == "milestone_step":
            args.milestones = generate_milestones(value, num_epochs)
        elif hasattr(args, key):
            setattr(args, key, value)

    # 维度一致性：output_dim = hidden_dim
    if "gig_hidden_dim" in merged:
        args.gig_output_dim = merged["gig_hidden_dim"]
    if "gene_hidden_dim" in merged:
        args.gene_output_dim = merged["gene_hidden_dim"]

    # 强制固定 heads=1
    args.gene_num_head = 1
    args.gig_num_head = 1

    return args


# ============================================================
# 训练评估
# ============================================================

def evaluate_single_fold(args, fold_n, nth_train, device, suppress_output=True):
    """单次训练，返回 (acc, success)"""
    from geo_tmain_gigtransformer_copy2 import train_geogig

    try:
        with suppress_stdout(suppress=suppress_output):
            max_test_acc = train_geogig(args, fold_n, nth_train, device)
        return max_test_acc, True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Fold {fold_n} train {nth_train} failed: {e}")
        return 0.0, False


def evaluate_fold_with_median(args, fold_n, device, n_runs=2, phase_tag=""):
    """对同一 fold 跑 n_runs 次，返回中位数准确率。
    自动过滤训练崩溃（acc < 0.05 的结果不计入）。
    """
    import torch
    logger = logging.getLogger(__name__)
    accs = []

    for nth_train in range(1, n_runs + 1):
        args.run_name = f"{phase_tag}_fold{fold_n}_run{nth_train}"
        acc, success = evaluate_single_fold(args, fold_n, nth_train, device)

        if success and acc > 0.05:
            accs.append(acc)
            logger.info(f"    Fold {fold_n} run {nth_train}: acc={acc:.4f}")
        else:
            logger.warning(f"    Fold {fold_n} run {nth_train}: FAILED (acc={acc:.4f}), skipped")

        torch.cuda.empty_cache()

    if not accs:
        return 0.0
    return float(np.median(accs))


def evaluate_with_retry(args, fold_n, nth_train, device, max_retries=3, phase_tag=""):
    """Phase 2 用：单次训练，失败则重试，最多 max_retries 次。"""
    import torch
    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        args.run_name = f"{phase_tag}_fold{fold_n}_train{nth_train}"
        if attempt > 0:
            args.run_name += f"_retry{attempt}"

        try:
            with suppress_stdout(suppress=True):
                from geo_tmain_gigtransformer_copy2 import train_geogig
                acc = train_geogig(args, fold_n, nth_train, device)
            run_dir = os.path.join(args.logdir, args.run_name)
            if os.path.exists(run_dir):
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)
            if acc > 0.05:
                return acc
            logger.warning(f"    Fold {fold_n} train {nth_train}: acc={acc:.4f}, "
                           f"retry {attempt + 1}/{max_retries}")
        except Exception as e:
            run_dir = os.path.join(args.logdir, args.run_name)
            if os.path.exists(run_dir):
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)
            logger.warning(f"    Fold {fold_n} train {nth_train}: failed ({e}), "
                           f"retry {attempt + 1}/{max_retries}")
        torch.cuda.empty_cache()

    logger.error(f"    Fold {fold_n} train {nth_train}: all {max_retries} retries failed")
    return None  # 全部失败则排除此 run


# ============================================================
# Optuna Objective (Phase 1)
# ============================================================

def make_objective(space, eval_folds, n_runs_per_fold=2):
    """Phase 1 objective: 每 fold 跑 n_runs 次取中位数，返回纯 mean_acc。"""
    import torch
    import optuna

    def objective(trial):
        logger = logging.getLogger(__name__)

        # 1. 采样参数
        params = sample_params(trial, space)
        logger.info(f"Trial {trial.number}: {json.dumps({k: round(v, 8) if isinstance(v, float) else v for k, v in params.items()})}")

        # 2. 构建 args
        args = create_args_from_params(params, NUM_EPOCHS)

        phase_tag = f"hpo_v3_p1_t{trial.number}"

        # 3. 打印关键维度信息
        gig_first_input = args.gig_input_transform_dim + args.gene_num_top_feature * args.gene_output_dim
        logger.info(f"  Architecture: gene[{args.gene_input_dim}->{args.gene_hidden_dim}->{args.gene_output_dim}] "
                     f"K={args.gene_num_top_feature} "
                     f"gig_input=[{args.gig_input_transform_dim}+{args.gene_num_top_feature}x{args.gene_output_dim}={gig_first_input}]->"
                     f"{args.gig_hidden_dim}->{args.gig_output_dim}")

        # 4. 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 5. Per-fold 评估 + 中间剪枝
        fold_medians = []
        for step, fold_n in enumerate(eval_folds):
            median_acc = evaluate_fold_with_median(
                args, fold_n, device,
                n_runs=n_runs_per_fold,
                phase_tag=phase_tag,
            )

            if median_acc < 0.05:
                logger.info(f"  Early termination: fold {fold_n} median_acc={median_acc:.4f}")
                return 0.0

            fold_medians.append(median_acc)

            # 用已完成 fold 的均值做中间汇报（供 pruner 判断）
            trial.report(np.mean(fold_medians), step=step)
            if trial.should_prune():
                logger.info(f"Trial {trial.number}: PRUNED after fold {fold_n} "
                            f"(mean_so_far={np.mean(fold_medians):.4f})")
                raise optuna.TrialPruned()

        mean_acc = float(np.mean(fold_medians))
        std_acc = float(np.std(fold_medians))

        trial.set_user_attr("mean_acc", mean_acc)
        trial.set_user_attr("std_acc", std_acc)
        trial.set_user_attr("fold_medians", [float(a) for a in fold_medians])

        logger.info(f"Trial {trial.number}: mean={mean_acc:.4f}, std={std_acc:.4f}")
        # 返回纯 mean_acc，不做方差惩罚
        return mean_acc

    return objective


# ============================================================
# Phase 1: 筛选
# ============================================================

def run_phase1(n_trials=50):
    """Phase 1: 3-fold × 2-run, 500 epochs, 聚焦搜索空间"""
    import optuna

    logger = setup_logging(1)
    logger.info("=" * 60)
    logger.info(f"PHASE 1: 筛选 (3-fold × 2-run, {NUM_EPOCHS} epochs)")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"搜索空间: {len(SEARCH_SPACE)} 个参数 (固定了 {len(FIXED_PARAMS)} 个)")
    logger.info(f"固定参数: {json.dumps({k: v for k, v in FIXED_PARAMS.items()})}")
    logger.info("=" * 60)

    study_name = f"{STUDY_PREFIX}_phase1"
    storage = f"sqlite:///{DB_PATH}"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42,
        multivariate=True,
    )
    # 放宽 pruner：只剪掉最差 25%，前 15 个 trial 不剪，至少跑 2 个 fold 再剪
    pruner = optuna.pruners.PercentilePruner(
        percentile=75.0,
        n_startup_trials=15,
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Seed: 两套成功参数 + 中间探索点
    if len(study.trials) == 0:
        study.enqueue_trial(SEED_BEST1)
        study.enqueue_trial(SEED_BEST2)
        study.enqueue_trial(SEED_MID)
        logger.info("Enqueued 3 seed trials: best1 (80.57%), best2 (80.20%), midpoint")

    objective = make_objective(
        space=SEARCH_SPACE,
        eval_folds=[1, 3, 5],
        n_runs_per_fold=2,
    )

    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    save_phase_results(study, phase=1)
    print_all_trials_table(study, phase=1)
    if study.best_trial:
        logger.info(f"Phase 1 完成。最佳 mean_acc: {study.best_value:.4f}")
        logger.info(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

    return study


# ============================================================
# Phase 2: 验证 (Top 5)
# ============================================================

def run_phase2():
    """Phase 2: 5-fold × 3-run, 500 epochs, Top 5, 失败重试"""
    import optuna
    import torch

    N_TOP = 5
    N_TRAINS = 3

    logger = setup_logging(2)
    logger.info("=" * 60)
    logger.info(f"PHASE 2: 验证 (5-fold × {N_TRAINS}-run, {NUM_EPOCHS} epochs, Top {N_TOP})")
    logger.info("=" * 60)

    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=f"{STUDY_PREFIX}_phase1",
            storage=storage,
        )
    except Exception:
        print("ERROR: Phase 1 study not found. Run --phase 1 first.")
        sys.exit(1)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:N_TOP]

    if len(top_trials) < N_TOP:
        logger.warning(f"Only {len(top_trials)} completed trials available (need {N_TOP})")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results = []
    for rank, trial in enumerate(top_trials):
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating Rank #{rank+1} (Phase 1 mean_acc: {trial.value:.4f})")
        logger.info(f"Params: {json.dumps(trial.params, indent=2)}")
        logger.info(f"{'='*50}")

        args = create_args_from_params(trial.params, num_epochs=NUM_EPOCHS)

        gig_first_input = args.gig_input_transform_dim + args.gene_num_top_feature * args.gene_output_dim
        logger.info(f"Architecture: gene[{args.gene_input_dim}->{args.gene_hidden_dim}->{args.gene_output_dim}] "
                     f"K={args.gene_num_top_feature} "
                     f"gig[{gig_first_input}->{args.gig_hidden_dim}->{args.gig_output_dim}]")

        phase_tag = f"hpo_v3_p2_rank{rank+1}"

        fold_results = {}
        for fold_n in range(1, 6):
            fold_accs = []
            for nth_train in range(1, N_TRAINS + 1):
                logger.info(f"  Fold {fold_n}, Train {nth_train}...")

                acc = evaluate_with_retry(
                    args, fold_n, nth_train, device,
                    max_retries=3,
                    phase_tag=phase_tag,
                )

                if acc is not None:
                    fold_accs.append(acc)
                    logger.info(f"    -> acc = {acc:.4f}")
                else:
                    logger.error(f"    -> ALL RETRIES FAILED, excluded from mean")

                torch.cuda.empty_cache()

            fold_mean = float(np.mean(fold_accs)) if fold_accs else 0.0
            fold_results[f"fold_{fold_n}"] = {
                "all_accs": [float(a) for a in fold_accs],
                "n_valid": len(fold_accs),
                "mean": fold_mean,
                "std": float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0,
            }
            logger.info(f"  Fold {fold_n} mean: {fold_mean:.4f} ({len(fold_accs)}/{N_TRAINS} valid runs)")

        fold_means = [fold_results[f"fold_{i}"]["mean"] for i in range(1, 6)]
        overall_mean = float(np.mean(fold_means))
        overall_std = float(np.std(fold_means))

        result = {
            "rank": rank + 1,
            "source_trial": trial.number,
            "source_mean_acc": float(trial.value),
            "params": {k: (float(v) if isinstance(v, float) else v) for k, v in trial.params.items()},
            "fixed_params": FIXED_PARAMS,
            "fold_results": fold_results,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
        }
        results.append(result)

        logger.info(f"\nRank #{rank+1} 总结:")
        logger.info(f"  5-fold 各 fold mean: {[f'{x:.4f}' for x in fold_means]}")
        logger.info(f"  overall: mean={overall_mean:.4f} +/- {overall_std:.4f}")

    # 保存最终结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULT_DIR, f"transformer_v3_phase2_final_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2 最终结果 (Top {N_TOP}):")
    logger.info(f"{'='*60}")
    for r in results:
        logger.info(f"\n  Rank #{r['rank']}: "
                     f"mean={r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%  "
                     f"(Phase 1 Trial #{r['source_trial']})")
        logger.info(f"  搜索参数:")
        for k, v in r['params'].items():
            logger.info(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")
        for fold_key in sorted(r['fold_results'].keys()):
            fr = r['fold_results'][fold_key]
            logger.info(f"    {fold_key}: mean={fr['mean']:.4f}, std={fr['std']:.4f}, "
                        f"runs={fr['all_accs']} ({fr['n_valid']}/{N_TRAINS} valid)")
    logger.info(f"\n结果已保存到: {result_file}")

    # stdout
    print(f"\n{'='*60}")
    print(f"PHASE 2 FINAL RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  Rank #{r['rank']}: "
              f"mean={r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
        print(f"  搜索参数:")
        for k, v in r['params'].items():
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")
        print(f"  固定参数:")
        for k, v in FIXED_PARAMS.items():
            print(f"    {k}: {v}")
        for fold_key in sorted(r['fold_results'].keys()):
            fr = r['fold_results'][fold_key]
            print(f"    {fold_key}: mean={fr['mean']:.4f}, std={fr['std']:.4f}, "
                  f"runs={fr['all_accs']} ({fr['n_valid']}/{N_TRAINS} valid)")
    best = max(results, key=lambda x: x['overall_mean'])
    print(f"\nBEST: Rank #{best['rank']}  "
          f"accuracy={best['overall_mean']:.4f} +/- {best['overall_std']:.4f}")
    print(f"\nResults saved to {result_file}")

    return results


# ============================================================
# 结果输出
# ============================================================

def print_all_trials_table(study, phase):
    import optuna

    logger = logging.getLogger(__name__)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]

    completed.sort(key=lambda t: t.value, reverse=True)

    logger.info(f"\n{'='*100}")
    logger.info(f"Phase {phase} 全部 Trials 排名 (完成: {len(completed)}, 剪枝: {len(pruned)})")
    logger.info(f"{'='*100}")

    header = (f"{'Rank':>4} {'Trial':>5} {'Mean':>7} {'Std':>6} | "
              f"{'lr':>9} {'g_hid':>5} {'K':>3} {'cls_w':>5} {'cls_f':>5} {'ent':>5}")
    logger.info(header)
    logger.info("-" * len(header))

    for rank, t in enumerate(completed):
        p = t.params
        mean_acc = t.user_attrs.get("mean_acc", t.value)
        std_acc = t.user_attrs.get("std_acc", 0.0)
        marker = " *" if rank == 0 else ""
        line = (
            f"{rank+1:>4} "
            f"T{t.number:>4} "
            f"{mean_acc:.4f} {std_acc:.4f} | "
            f"{p.get('lr', 0):.6f} "
            f"{p.get('gene_hidden_dim', 0):>5} "
            f"{p.get('gene_num_top_feature', 0):>3} "
            f"{p.get('class_weight', 0):.3f} "
            f"{p.get('class_weight_fine', 0):.3f} "
            f"{p.get('ent_weight', 0):.3f}"
            f"{marker}"
        )
        logger.info(line)

    # stdout
    print(f"\n{'='*70}")
    print(f"Phase {phase} Results (completed: {len(completed)}, pruned: {len(pruned)})")
    print(f"{'='*70}")
    print(header)
    print("-" * len(header))
    for rank, t in enumerate(completed):
        p = t.params
        mean_acc = t.user_attrs.get("mean_acc", t.value)
        std_acc = t.user_attrs.get("std_acc", 0.0)
        marker = " *" if rank == 0 else ""
        print(
            f"{rank+1:>4} "
            f"T{t.number:>4} "
            f"{mean_acc:.4f} {std_acc:.4f} | "
            f"{p.get('lr', 0):.6f} "
            f"{p.get('gene_hidden_dim', 0):>5} "
            f"{p.get('gene_num_top_feature', 0):>3} "
            f"{p.get('class_weight', 0):.3f} "
            f"{p.get('class_weight_fine', 0):.3f} "
            f"{p.get('ent_weight', 0):.3f}"
            f"{marker}"
        )
    if pruned:
        print(f"\n  ({len(pruned)} trials pruned)")

    # 保存到 CSV
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULT_DIR, f"transformer_v3_phase{phase}_all_trials_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write("rank,trial,mean_acc,std_acc,status," + ",".join(TABLE_PARAMS) + "\n")
        for rank, t in enumerate(completed):
            p = t.params
            mean_acc = t.user_attrs.get("mean_acc", t.value)
            std_acc = t.user_attrs.get("std_acc", 0.0)
            values = [str(p.get(k, "")) for k in TABLE_PARAMS]
            f.write(f"{rank+1},{t.number},{mean_acc:.6f},{std_acc:.6f},completed," + ",".join(values) + "\n")
        for t in pruned:
            p = t.params
            values = [str(p.get(k, "")) for k in TABLE_PARAMS]
            f.write(f",,0,0,pruned," + ",".join(values) + "\n")

    logger.info(f"\n完整结果已保存到 CSV: {csv_file}")
    print(f"\n完整结果已保存到 CSV: {csv_file}")


def save_phase_results(study, phase):
    import optuna

    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)

    results = {
        "phase": phase,
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "fixed_params": FIXED_PARAMS,
        "top_10": [
            {"trial_number": t.number, "value": t.value, "params": t.params}
            for t in completed[:10]
        ],
        "timestamp": timestamp,
    }

    result_file = os.path.join(RESULT_DIR, f"transformer_v3_phase{phase}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {result_file}", file=sys.stderr)


# ============================================================
# --analyze: 汇总分析
# ============================================================

def analyze_results():
    import optuna

    storage = f"sqlite:///{DB_PATH}"

    print("\n" + "=" * 70)
    print("GiG-Transformer V3 超参数优化结果汇总")
    print("=" * 70)
    print(f"\n固定参数:")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")

    # Phase 1
    study_name = f"{STUDY_PREFIX}_phase1"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        print(f"\nPhase 1: 未找到数据")
        return

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]

    if not completed:
        print(f"\nPhase 1: 无完成的 trials")
        return

    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\n{'='*60}")
    print(f"Phase 1: {len(completed)} completed, {len(pruned)} pruned")
    print(f"{'='*60}")
    print(f"  Best mean_acc: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6g}")
        else:
            print(f"    {k}: {v}")

    # 排名表
    print(f"\n  {'Rank':>4} {'Trial':>5} {'Mean':>7} | {'lr':>9} {'g_hid':>5} {'K':>3} "
          f"{'cls_w':>5} {'cls_f':>5} {'ent':>5}")
    print("  " + "-" * 65)
    for rank, t in enumerate(completed):
        p = t.params
        marker = " *" if rank == 0 else ""
        print(
            f"  {rank+1:>4} "
            f"T{t.number:>4} "
            f"{t.value:.4f} | "
            f"{p.get('lr', 0):.6f} "
            f"{p.get('gene_hidden_dim', 0):>5} "
            f"{p.get('gene_num_top_feature', 0):>3} "
            f"{p.get('class_weight', 0):.3f} "
            f"{p.get('class_weight_fine', 0):.3f} "
            f"{p.get('ent_weight', 0):.3f}"
            f"{marker}"
        )
    if pruned:
        print(f"\n  (另有 {len(pruned)} 个 trials 被剪枝)")

    # 参数重要性
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n  参数重要性:")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 30)
            print(f"    {param:<28s} {imp:.4f} {bar}")
    except Exception:
        pass

    # Phase 2 结果
    if os.path.exists(RESULT_DIR):
        phase2_files = [f for f in os.listdir(RESULT_DIR) if f.startswith("transformer_v3_phase2_final")]
        if phase2_files:
            latest = sorted(phase2_files)[-1]
            with open(os.path.join(RESULT_DIR, latest)) as f:
                results = json.load(f)

            print(f"\n{'='*60}")
            print(f"Phase 2 (验证 - Top {len(results)}):")
            print(f"{'='*60}")
            for r in results:
                print(f"  Rank #{r['rank']}: "
                      f"mean={r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
                print(f"    搜索参数:")
                for k, v in r.get('params', {}).items():
                    print(f"      {k}: {v:.6g}" if isinstance(v, float) else f"      {k}: {v}")
                for fold_key in sorted(r.get('fold_results', {}).keys()):
                    fr = r['fold_results'][fold_key]
                    print(f"      {fold_key}: mean={fr['mean']:.4f}, std={fr.get('std', 0):.4f}, "
                          f"runs={fr.get('all_accs', [])} ({fr.get('n_valid', '?')} valid)")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GiG-Transformer V3 Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法:
  python optuna_gigtransformerV3.py --phase 1 --n_trials 50   # 3-fold×2-run 筛选
  python optuna_gigtransformerV3.py --phase 2                  # 5-fold×3-run 验证
  python optuna_gigtransformerV3.py --analyze                  # 汇总分析
        """
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Phase to run (1=筛选, 2=验证)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials (Phase 1)")
    parser.add_argument("--analyze", action="store_true", help="Analyze all results")

    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.analyze:
        analyze_results()
    elif args.phase == 1:
        run_phase1(n_trials=args.n_trials)
    elif args.phase == 2:
        run_phase2()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
