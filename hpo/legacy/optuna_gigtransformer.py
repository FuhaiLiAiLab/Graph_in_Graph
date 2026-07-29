#!/usr/bin/env python3
"""
GiG-Transformer (copy2) 超参数优化脚本
========================================
基于 geo_tmain_gigtransformer_copy2.py 的 Optuna HPO。
不修改 geo_tmain_gigtransformer_copy2.py 本身，通过 argparse.Namespace 注入参数。

策略：初筛 → 细筛 → Top 3 验证
  Phase 1 (初筛): 2-fold, 100 epochs — 100 trials + MedianPruner, 宽搜索空间
  Phase 2 (细筛): 3-fold, 200 epochs — 30 trials, 缩窄空间
  Phase 3 (最终验证): 5-fold × 3-train, 500 epochs — Top 3

架构约束：
  TransformerConv 使用 concat=True 时，输出维度 = heads × out_channels。
  gene_num_head 和 gig_num_head 固定为 1 以避免维度不匹配。

用法：
  python optuna_gigtransformer.py --phase 1 --n_trials 100
  python optuna_gigtransformer.py --phase 2 --n_trials 30
  python optuna_gigtransformer.py --phase 3
  python optuna_gigtransformer.py --analyze
"""

import os
import sys
import json
import math
import logging
import argparse
import traceback
import contextlib
import io
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from collections import Counter

# 多进程参数解析修复
if __name__ != "__main__" and len(sys.argv) > 1:
    sys.argv = [sys.argv[0]]

# ============================================================
# 配置
# ============================================================
DB_PATH = "optuna_gigtransformer_copy2.db"
STUDY_PREFIX = "gigtransformer_c2"
LOG_DIR = "./hpo_logs"
RESULT_DIR = "./hpo_results"


# ============================================================
# Phase 1 搜索空间 (宽搜索)
# ============================================================
PHASE1_SPACE = {
    # === 学习率 ===
    "lr": {"type": "loguniform", "low": 5e-4, "high": 0.03},

    # === Weight decay ===
    "weight_decay": {"type": "loguniform", "low": 1e-8, "high": 1e-2},

    # === 学习率衰减 ===
    "gamma": {"type": "categorical", "choices": [0.85, 0.9, 0.95]},
    "milestone_step": {"type": "categorical", "choices": [50, 100, 150]},

    # === Gene-level 网络结构 ===
    # gene_hidden_dim = gene_output_dim (保持一致)
    "gene_hidden_dim": {"type": "categorical", "choices": [6, 9, 12, 18]},
    # K: assignment matrix 行数
    "gene_num_top_feature": {"type": "categorical", "choices": [8, 12, 16, 18, 24]},

    # === GiG-level 网络结构 ===
    "gig_input_transform_dim": {"type": "categorical", "choices": [18, 24, 36, 42]},
    # gig_hidden_dim = gig_output_dim (保持一致)
    "gig_hidden_dim": {"type": "categorical", "choices": [12, 18, 24, 36]},

    # === 损失函数权重 ===
    "class_weight": {"type": "uniform", "low": 0.5, "high": 1.0},
    "class_weight_fine": {"type": "uniform", "low": 0.1, "high": 1.0},
    "ortho_weight": {"type": "loguniform", "low": 1e-3, "high": 0.15},
    "link_weight": {"type": "loguniform", "low": 1e-3, "high": 0.15},
    "ent_weight": {"type": "uniform", "low": 0.0, "high": 0.1},

    # === 正则化 ===
    "dropout": {"type": "uniform", "low": 0.0, "high": 0.35},
    "clip": {"type": "categorical", "choices": [2.0, 5.0, 10.0]},
}

# 用于 Phase 2 收窄时的原始边界
ORIG_BOUNDS = {
    "lr": (5e-4, 0.03), "weight_decay": (1e-8, 1e-2),
    "class_weight": (0.5, 1.0), "class_weight_fine": (0.1, 1.0),
    "ortho_weight": (1e-3, 0.15), "link_weight": (1e-3, 0.15),
    "ent_weight": (0.0, 0.1), "dropout": (0.0, 0.35),
}

# 打印排名表的参数列
TABLE_PARAMS = [
    "lr", "dropout", "gene_hidden_dim", "gene_num_top_feature",
    "gig_input_transform_dim", "gig_hidden_dim", "class_weight", "class_weight_fine",
    "weight_decay", "ortho_weight", "link_weight", "ent_weight",
    "gamma", "milestone_step", "clip",
]


# ============================================================
# 工具函数
# ============================================================

def setup_logging(phase):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"transformer_c2_phase{phase}_{timestamp}.log")

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
        milestones=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250],
        gamma=0.9, clip=5.0, batch_size=256,
        num_epochs=500,
        unchanged_threshold=100, change_wave=0.75, num_workers=0,
        graph_opt='GinG',
        gene_input_dim=6, gene_hidden_dim=18, gene_output_dim=18,
        gene_num_top_feature=18,
        gig_input_dim=42, gig_input_transform_dim=18,
        gig_hidden_dim=18, gig_output_dim=18,
        class_weight_fine=0.5, class_weight=0.9,
        ortho_weight=0.05, link_weight=0.05, ent_weight=0.00,
        num_classes=3,
        gene_num_head=1,   # 固定: TransformerConv concat 架构约束
        gig_num_head=1,    # 固定: 同上
        dropout=0.01, gpu_ids=[0],
        # copy2 新增的 TensorBoard 参数
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
    """将采样参数合并到完整的 args 对象中。"""
    args = make_default_args(num_epochs=num_epochs)

    for key, value in params.items():
        if key == "milestone_step":
            args.milestones = generate_milestones(value, num_epochs)
        elif hasattr(args, key):
            setattr(args, key, value)

    # 维度一致性：output_dim = hidden_dim
    if "gig_hidden_dim" in params:
        args.gig_output_dim = params["gig_hidden_dim"]
    if "gene_hidden_dim" in params:
        args.gene_output_dim = params["gene_hidden_dim"]

    # 强制固定 heads=1
    args.gene_num_head = 1
    args.gig_num_head = 1

    return args


# ============================================================
# 训练评估 (调用 geo_tmain_gigtransformer_copy2)
# ============================================================

def evaluate_single_fold(args, fold_n, device, suppress_output=True):
    from geo_tmain_gigtransformer_copy2 import train_geogig

    try:
        with suppress_stdout(suppress=suppress_output):
            max_test_acc = train_geogig(args, fold_n, nth_training_fold_num=1, device=device)
        return max_test_acc, True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Fold {fold_n} failed: {e}")
        return 0.0, False


# ============================================================
# Optuna Objective
# ============================================================

def make_objective(phase, space, num_epochs, eval_folds):
    import torch
    import optuna

    def objective(trial):
        logger = logging.getLogger(__name__)

        # 1. 采样参数
        params = sample_params(trial, space)
        logger.info(f"Trial {trial.number}: {json.dumps({k: round(v, 8) if isinstance(v, float) else v for k, v in params.items()})}")

        # 2. 构建 args
        try:
            args = create_args_from_params(params, num_epochs)
        except ValueError as e:
            logger.warning(f"Trial {trial.number}: 参数不合法 - {e}")
            return 0.0

        # 为 HPO 设置独立的 TensorBoard run_name，避免冲突
        args.run_name = f"hpo_phase{phase}_trial{trial.number}"

        # 3. 打印关键维度信息
        gig_first_input = args.gig_input_transform_dim + args.gene_num_top_feature * args.gene_output_dim
        logger.info(f"  Architecture: gene[{args.gene_input_dim}->{args.gene_hidden_dim}->{args.gene_output_dim}] "
                     f"K={args.gene_num_top_feature} "
                     f"gig_input=[{args.gig_input_transform_dim}+{args.gene_num_top_feature}x{args.gene_output_dim}={gig_first_input}]->"
                     f"{args.gig_hidden_dim}->{args.gig_output_dim}")

        # 4. 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 5. Per-fold 评估 + 中间剪枝
        accuracies = []
        for step, fold_n in enumerate(eval_folds):
            try:
                acc, success = evaluate_single_fold(args, fold_n, device, suppress_output=True)
            except Exception as e:
                logger.error(f"Trial {trial.number}: fold {fold_n} ERROR - {e}")
                traceback.print_exc(file=sys.stderr)
                return 0.0

            if not success or acc < 0.05:
                logger.info(f"  Early termination: fold {fold_n} acc={acc:.4f}")
                return 0.0

            accuracies.append(acc)
            torch.cuda.empty_cache()

            trial.report(np.mean(accuracies), step=step)
            if trial.should_prune():
                logger.info(f"Trial {trial.number}: PRUNED after fold {fold_n} (acc_so_far={np.mean(accuracies):.4f})")
                raise optuna.TrialPruned()

        mean_acc = np.mean(accuracies)
        logger.info(f"Trial {trial.number}: mean_acc={mean_acc:.4f} (folds={eval_folds})")
        return mean_acc

    return objective


# ============================================================
# Phase 1: 初筛
# ============================================================

def run_phase1(n_trials=100):
    """Phase 1: 2-fold, 100 epochs, 100 trials + MedianPruner, 宽搜索空间"""
    import optuna

    logger = setup_logging(1)
    logger.info("=" * 60)
    logger.info("PHASE 1: 初筛 (2-fold, 100 epochs, MedianPruner)")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"搜索空间: {len(PHASE1_SPACE)} 个参数")
    logger.info("=" * 60)

    study_name = f"{STUDY_PREFIX}_phase1"
    storage = f"sqlite:///{DB_PATH}"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # 种子 trial 1: 当前代码默认参数
    seed_current = {
        "lr": 0.01, "weight_decay": 1e-8, "gamma": 0.9, "milestone_step": 100,
        "gene_hidden_dim": 18, "gene_num_top_feature": 18,
        "gig_input_transform_dim": 18, "gig_hidden_dim": 18,
        "class_weight": 0.9, "class_weight_fine": 0.5,
        "ortho_weight": 0.05, "link_weight": 0.05, "ent_weight": 0.0,
        "dropout": 0.01, "clip": 5.0,
    }
    # 种子 trial 2: 论文适配参数
    seed_paper = {
        "lr": 0.01, "weight_decay": 1e-8, "gamma": 0.9, "milestone_step": 100,
        "gene_hidden_dim": 6, "gene_num_top_feature": 18,
        "gig_input_transform_dim": 42, "gig_hidden_dim": 18,
        "class_weight": 0.9, "class_weight_fine": 0.5,
        "ortho_weight": 0.05, "link_weight": 0.05, "ent_weight": 0.0,
        "dropout": 0.01, "clip": 5.0,
    }
    # 种子 trial 3: 抗过拟合配置
    seed_regularized = {
        "lr": 0.005, "weight_decay": 1e-4, "gamma": 0.95, "milestone_step": 150,
        "gene_hidden_dim": 6, "gene_num_top_feature": 12,
        "gig_input_transform_dim": 24, "gig_hidden_dim": 12,
        "class_weight": 0.8, "class_weight_fine": 0.5,
        "ortho_weight": 0.03, "link_weight": 0.03, "ent_weight": 0.02,
        "dropout": 0.15, "clip": 5.0,
    }

    if len(study.trials) == 0:
        study.enqueue_trial(seed_current)
        study.enqueue_trial(seed_paper)
        study.enqueue_trial(seed_regularized)
        logger.info("Enqueued 3 seed trials (current defaults, paper adapted, anti-overfit)")

    objective = make_objective(
        phase=1,
        space=PHASE1_SPACE,
        num_epochs=100,
        eval_folds=[1, 3],  # 2-fold
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
        logger.info(f"Phase 1 完成。最佳准确率: {study.best_value:.4f}")
        logger.info(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

    return study


# ============================================================
# Phase 2: 细筛
# ============================================================

def narrow_space_from_study(study, top_k=15, min_acc=0.30):
    """基于 Phase 1 结果缩窄搜索空间"""
    import optuna

    logger = logging.getLogger(__name__)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None
                 and t.value > min_acc]

    if len(completed) < 5:
        logger.warning(f"Only {len(completed)} trials with acc > {min_acc}. Using Phase 1 space.")
        return PHASE1_SPACE

    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:top_k]

    logger.info(f"Narrowing space based on top {len(top_trials)} trials "
                f"(acc range: {top_trials[-1].value:.4f} ~ {top_trials[0].value:.4f})")

    narrowed = {}
    for param_name, config in PHASE1_SPACE.items():
        values = [t.params.get(param_name) for t in top_trials if param_name in t.params]
        if not values:
            narrowed[param_name] = config
            continue

        if config["type"] == "categorical":
            counter = Counter(values)
            threshold = max(1, len(top_trials) // 5)
            frequent = [v for v, c in counter.items() if c >= threshold]
            if len(frequent) < 2:
                frequent = list(counter.keys())[:3]
            narrowed[param_name] = {"type": "categorical", "choices": sorted(set(frequent))}

        elif config["type"] == "loguniform":
            min_val = min(values)
            max_val = max(values)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            log_margin = (log_max - log_min) * 0.3
            orig_lo, orig_hi = ORIG_BOUNDS.get(param_name, (config["low"], config["high"]))
            new_low = max(config["low"], math.exp(log_min - log_margin))
            new_high = min(config["high"], math.exp(log_max + log_margin))
            if new_low >= new_high:
                new_low, new_high = config["low"], config["high"]
            narrowed[param_name] = {"type": "loguniform", "low": new_low, "high": new_high}

        elif config["type"] == "uniform":
            min_val = min(values)
            max_val = max(values)
            margin = (max_val - min_val) * 0.3
            new_low = max(config["low"], min_val - margin)
            new_high = min(config["high"], max_val + margin)
            if new_low >= new_high:
                new_low, new_high = config["low"], config["high"]
            narrowed[param_name] = {"type": "uniform", "low": new_low, "high": new_high}

        else:
            narrowed[param_name] = config

        if narrowed[param_name] != config:
            logger.info(f"  {param_name}: {config} -> {narrowed[param_name]}")

    return narrowed


def run_phase2(n_trials=30):
    """Phase 2: 3-fold, 200 epochs, 30 trials, 缩窄空间"""
    import optuna

    logger = setup_logging(2)
    logger.info("=" * 60)
    logger.info("PHASE 2: 细筛 (3-fold, 200 epochs)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)

    storage = f"sqlite:///{DB_PATH}"

    try:
        phase1_study = optuna.load_study(
            study_name=f"{STUDY_PREFIX}_phase1",
            storage=storage,
        )
    except Exception:
        print("ERROR: Phase 1 study not found. Run --phase 1 first.")
        sys.exit(1)

    narrowed_space = narrow_space_from_study(phase1_study, top_k=15, min_acc=0.30)

    study_name = f"{STUDY_PREFIX}_phase2"
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5,
        seed=123,
        multivariate=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Seed: Phase 1 的 top 5 参数
    if len(study.trials) == 0:
        completed = [t for t in phase1_study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=True)

        for t in completed[:5]:
            seed_params = {}
            for k, v in t.params.items():
                if k in narrowed_space:
                    space_cfg = narrowed_space[k]
                    if space_cfg["type"] == "categorical":
                        if v in space_cfg["choices"]:
                            seed_params[k] = v
                        else:
                            seed_params[k] = space_cfg["choices"][0]
                    elif space_cfg["type"] in ["loguniform", "uniform"]:
                        seed_params[k] = max(space_cfg["low"], min(space_cfg["high"], v))
                    else:
                        seed_params[k] = v
                else:
                    seed_params[k] = v
            study.enqueue_trial(seed_params)

        logger.info("Enqueued top 5 params from Phase 1 as seeds")

    objective = make_objective(
        phase=2,
        space=narrowed_space,
        num_epochs=200,
        eval_folds=[1, 2, 4],  # 3-fold
    )

    remaining = n_trials - len(study.trials)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    save_phase_results(study, phase=2)
    print_all_trials_table(study, phase=2)
    if study.best_trial:
        logger.info(f"Phase 2 完成。最佳准确率: {study.best_value:.4f}")
        logger.info(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

    return study


# ============================================================
# Phase 3: 最终验证 (Top 3)
# ============================================================

def run_phase3():
    """Phase 3: 5-fold x 3-train, 500 epochs, Top 3"""
    import optuna
    import torch

    N_TOP = 3
    N_TRAINS = 3
    NUM_EPOCHS = 500

    logger = setup_logging(3)
    logger.info("=" * 60)
    logger.info(f"PHASE 3: 最终验证 (5-fold x {N_TRAINS}-train, {NUM_EPOCHS} epochs, Top {N_TOP})")
    logger.info("=" * 60)

    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=f"{STUDY_PREFIX}_phase2",
            storage=storage,
        )
        source = "Phase 2"
    except Exception:
        try:
            study = optuna.load_study(
                study_name=f"{STUDY_PREFIX}_phase1",
                storage=storage,
            )
            source = "Phase 1"
        except Exception:
            print("ERROR: No Phase 1 or Phase 2 study found. Run earlier phases first.")
            sys.exit(1)

    logger.info(f"Loading top params from {source}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:N_TOP]

    if len(top_trials) < N_TOP:
        logger.warning(f"Only {len(top_trials)} completed trials available (need {N_TOP})")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from geo_tmain_gigtransformer_copy2 import train_geogig

    results = []
    for rank, trial in enumerate(top_trials):
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating Rank #{rank+1} (source acc: {trial.value:.4f})")
        logger.info(f"Params: {json.dumps(trial.params, indent=2)}")
        logger.info(f"{'='*50}")

        args = create_args_from_params(trial.params, num_epochs=NUM_EPOCHS)

        gig_first_input = args.gig_input_transform_dim + args.gene_num_top_feature * args.gene_output_dim
        logger.info(f"Architecture: gene[{args.gene_input_dim}->{args.gene_hidden_dim}->{args.gene_output_dim}] "
                     f"K={args.gene_num_top_feature} "
                     f"gig[{gig_first_input}->{args.gig_hidden_dim}->{args.gig_output_dim}]")

        fold_results = {}
        for fold_n in range(1, 6):
            fold_accs = []
            for nth_train in range(1, N_TRAINS + 1):
                logger.info(f"  Fold {fold_n}, Train {nth_train}...")
                args.run_name = f"hpo_phase3_rank{rank+1}_fold{fold_n}_train{nth_train}"
                try:
                    with suppress_stdout(suppress=True):
                        acc = train_geogig(args, fold_n, nth_train, device)
                    fold_accs.append(acc)
                    logger.info(f"    -> acc = {acc:.4f}")
                except Exception as e:
                    logger.error(f"    -> FAILED: {e}")
                    fold_accs.append(0.0)
                torch.cuda.empty_cache()

            fold_best = max(fold_accs) if fold_accs else 0.0
            fold_results[f"fold_{fold_n}"] = {
                "all_accs": [float(a) for a in fold_accs],
                "best": float(fold_best),
                "mean": float(np.mean(fold_accs)),
            }
            logger.info(f"  Fold {fold_n} best: {fold_best:.4f}, mean: {np.mean(fold_accs):.4f}")

        fold_bests = [fold_results[f"fold_{i}"]["best"] for i in range(1, 6)]
        overall_mean = float(np.mean(fold_bests))
        overall_std = float(np.std(fold_bests))

        result = {
            "rank": rank + 1,
            "source": source,
            "source_trial": trial.number,
            "source_acc": float(trial.value),
            "params": {k: (float(v) if isinstance(v, float) else v) for k, v in trial.params.items()},
            "fold_results": fold_results,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
        }
        results.append(result)

        logger.info(f"\nRank #{rank+1} 总结:")
        logger.info(f"  5-fold 各 fold 最佳: {[f'{x:.4f}' for x in fold_bests]}")
        logger.info(f"  平均准确率: {overall_mean:.4f} +/- {overall_std:.4f}")

    # 保存最终结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULT_DIR, f"transformer_c2_phase3_final_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 3 最终结果 (Top {N_TOP}):")
    logger.info(f"{'='*60}")
    for r in results:
        logger.info(f"  Rank #{r['rank']}: {r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%  "
                     f"({r['source']} Trial #{r['source_trial']}, acc={r['source_acc']:.4f})")
    logger.info(f"\n结果已保存到: {result_file}")

    # 同时输出到 stdout
    print(f"\n{'='*60}")
    print(f"PHASE 3 FINAL RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  Rank #{r['rank']}: {r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
        print(f"    Params: lr={r['params'].get('lr', '?')}, wd={r['params'].get('weight_decay', '?')}")
    best = max(results, key=lambda x: x['overall_mean'])
    print(f"\nBEST: Rank #{best['rank']}  accuracy={best['overall_mean']:.4f} +/- {best['overall_std']:.4f}")
    print(f"Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
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

    logger.info(f"\n{'='*120}")
    logger.info(f"Phase {phase} 全部 Trials 排名 (完成: {len(completed)}, 剪枝: {len(pruned)})")
    logger.info(f"{'='*120}")

    header = (f"{'Rank':>4} {'Trial':>5} {'Acc':>7} | {'lr':>9} {'drop':>5} "
              f"{'g_hid':>5} {'K':>3} {'gig_in':>6} {'gig_h':>5} "
              f"{'cls_w':>5} {'cls_f':>5} {'wd':>9} {'orth':>5} {'link':>5} {'ent':>5} "
              f"{'g':>4} {'ms':>3} {'clip':>4}")
    logger.info(header)
    logger.info("-" * len(header))

    for rank, t in enumerate(completed):
        p = t.params
        line = (
            f"{rank+1:>4} "
            f"T{t.number:>4} "
            f"{t.value:.4f} | "
            f"{p.get('lr', 0):.6f} "
            f"{p.get('dropout', 0):.3f} "
            f"{p.get('gene_hidden_dim', 0):>5} "
            f"{p.get('gene_num_top_feature', 0):>3} "
            f"{p.get('gig_input_transform_dim', 0):>6} "
            f"{p.get('gig_hidden_dim', 0):>5} "
            f"{p.get('class_weight', 0):.3f} "
            f"{p.get('class_weight_fine', 0):.3f} "
            f"{p.get('weight_decay', 0):.2e} "
            f"{p.get('ortho_weight', 0):.3f} "
            f"{p.get('link_weight', 0):.3f} "
            f"{p.get('ent_weight', 0):.3f} "
            f"{p.get('gamma', 0):.2f} "
            f"{p.get('milestone_step', 0):>3} "
            f"{p.get('clip', 0):.1f}"
        )
        logger.info(line)

    # 同时输出到 stdout (终端)
    print(f"\n{'='*80}")
    print(f"Phase {phase} Results (completed: {len(completed)}, pruned: {len(pruned)})")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Trial':>5} {'Acc':>7} | {'lr':>9} {'drop':>5} {'g_hid':>5} {'K':>3} {'gig_in':>6} {'gig_h':>5}")
    print("-" * 80)
    for rank, t in enumerate(completed):
        p = t.params
        marker = " *" if rank == 0 else ""
        print(
            f"{rank+1:>4} "
            f"T{t.number:>4} "
            f"{t.value:.4f} | "
            f"{p.get('lr', 0):.6f} "
            f"{p.get('dropout', 0):.3f} "
            f"{p.get('gene_hidden_dim', 0):>5} "
            f"{p.get('gene_num_top_feature', 0):>3} "
            f"{p.get('gig_input_transform_dim', 0):>6} "
            f"{p.get('gig_hidden_dim', 0):>5}"
            f"{marker}"
        )
    if pruned:
        print(f"\n  ({len(pruned)} trials pruned by MedianPruner)")

    # 保存到 CSV
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULT_DIR, f"transformer_c2_phase{phase}_all_trials_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write("rank,trial,accuracy,status," + ",".join(TABLE_PARAMS) + "\n")
        for rank, t in enumerate(completed):
            p = t.params
            values = [str(p.get(k, "")) for k in TABLE_PARAMS]
            f.write(f"{rank+1},{t.number},{t.value:.6f},completed," + ",".join(values) + "\n")
        for t in pruned:
            p = t.params
            values = [str(p.get(k, "")) for k in TABLE_PARAMS]
            f.write(f",,0,pruned," + ",".join(values) + "\n")

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
        "top_10": [
            {"trial_number": t.number, "value": t.value, "params": t.params}
            for t in completed[:10]
        ],
        "timestamp": timestamp,
    }

    result_file = os.path.join(RESULT_DIR, f"transformer_c2_phase{phase}_{timestamp}.json")
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
    print("GiG-Transformer (copy2) 超参数优化结果汇总")
    print("=" * 70)

    for phase in [1, 2]:
        study_name = f"{STUDY_PREFIX}_phase{phase}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            print(f"\nPhase {phase}: 未找到数据")
            continue

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None]
        pruned = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.PRUNED]

        if not completed:
            print(f"\nPhase {phase}: 无完成的 trials")
            continue

        completed.sort(key=lambda t: t.value, reverse=True)

        print(f"\n{'='*60}")
        print(f"Phase {phase}: {len(completed)} completed, {len(pruned)} pruned")
        print(f"{'='*60}")
        print(f"  Best accuracy: {study.best_value:.4f}")
        print(f"  Best params:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")

        # 排名表
        print(f"\n  {'Rank':>4} {'Trial':>5} {'Acc':>7} | {'lr':>9} {'drop':>5} {'g_hid':>5} {'K':>3} {'gig_in':>6} {'gig_h':>5} {'cls_w':>5} {'wd':>9}")
        print("  " + "-" * 90)
        for rank, t in enumerate(completed):
            p = t.params
            marker = " *" if rank == 0 else ""
            print(
                f"  {rank+1:>4} "
                f"T{t.number:>4} "
                f"{t.value:.4f} | "
                f"{p.get('lr', 0):.6f} "
                f"{p.get('dropout', 0):.3f} "
                f"{p.get('gene_hidden_dim', 0):>5} "
                f"{p.get('gene_num_top_feature', 0):>3} "
                f"{p.get('gig_input_transform_dim', 0):>6} "
                f"{p.get('gig_hidden_dim', 0):>5} "
                f"{p.get('class_weight', 0):.3f} "
                f"{p.get('weight_decay', 0):.2e}"
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

    # Phase 3 结果
    if os.path.exists(RESULT_DIR):
        phase3_files = [f for f in os.listdir(RESULT_DIR) if f.startswith("transformer_c2_phase3_final")]
        if phase3_files:
            latest = sorted(phase3_files)[-1]
            with open(os.path.join(RESULT_DIR, latest)) as f:
                results = json.load(f)

            print(f"\n{'='*60}")
            print(f"Phase 3 (最终验证 - Top {len(results)}):")
            print(f"{'='*60}")
            for r in results:
                print(f"  Rank #{r['rank']}: {r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
                for fold_key in sorted(r.get('fold_results', {}).keys()):
                    fr = r['fold_results'][fold_key]
                    print(f"    {fold_key}: best={fr['best']:.4f}, mean={fr['mean']:.4f}, runs={fr.get('all_accs', [])}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GiG-Transformer (copy2) Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法:
  python optuna_gigtransformer.py --phase 1 --n_trials 100   # 2-fold 初筛
  python optuna_gigtransformer.py --phase 2 --n_trials 30    # 3-fold 细筛
  python optuna_gigtransformer.py --phase 3                  # 5-fold 最终验证
  python optuna_gigtransformer.py --analyze                  # 汇总分析
        """
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Phase to run (1=初筛, 2=细筛, 3=最终验证)")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--analyze", action="store_true", help="Analyze all results")

    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.analyze:
        analyze_results()
    elif args.phase == 1:
        run_phase1(n_trials=args.n_trials)
    elif args.phase == 2:
        run_phase2(n_trials=args.n_trials)
    elif args.phase == 3:
        run_phase3()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
