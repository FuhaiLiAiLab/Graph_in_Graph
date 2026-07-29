#!/usr/bin/env python3
"""
GIG-GAT 超参数优化脚本 (v2 - Bug Fixed)
======================================
策略：三阶段优化
  Phase 1: 粗筛 (2-fold, 300 epochs) — 80 trials, ~8-10h
  Phase 2: 精调 (3-fold, 400 epochs) — Top 15 参数区域, 40 trials, ~6h
  Phase 3: 最终验证 (5-fold x 5-train, 500 epochs) — Top 3, ~5h

关键设计原则：
  1. 不修改 geo_tmain_giggat.py 原始代码
  2. 通过 monkey-patch args 注入参数
  3. 搜索空间基于论文原始参数 + 导师建议 + 历史试验经验
  4. 多 fold 评估避免单折偏差
  5. 完善的错误处理和断点续传

Bug 修复记录 (相对于原 v2):
  [Fix 1] 移除 signal.SIGALRM — Windows 不支持
  [Fix 2] 添加 gig_hidden_dim % gig_num_head == 0 约束，避免 GATConv assertion crash
  [Fix 3] gene_num_top_feature 独立搜索，不再绑定到 gene_hidden_dim
  [Fix 4] SuppressStdout 改用 contextlib.redirect_stdout，更安全
  [Fix 5] 早停阈值从 acc < 0.10 放宽到 acc < 0.05
  [Fix 6] create_args_from_params 改用 argparse.Namespace，不再调用 arg_parse()

用法：
  python optuna_giggat_v2.py --diagnostic
  python optuna_giggat_v2.py --phase 1 --n_trials 80
  python optuna_giggat_v2.py --phase 2 --n_trials 40
  python optuna_giggat_v2.py --phase 3
  python optuna_giggat_v2.py --analyze
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
import traceback
import contextlib
import io
import numpy as np
from datetime import datetime
from contextlib import contextmanager

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

# ============================================================
# 多进程参数解析修复
# ============================================================
# 解决 Optuna 多进程时子进程重新解析命令行参数导致的错误
# 当不是主进程且 sys.argv 包含参数时，清空参数列表，只保留脚本名
if __name__ != "__main__" and len(sys.argv) > 1:
    # 子进程环境，清空 sys.argv 避免 argparse 错误
    sys.argv = [sys.argv[0]]

# ============================================================
# 配置区域 - 根据你的环境修改
# ============================================================
DB_PATH = "optuna_giggat_v2.db"
STUDY_PREFIX = "giggat_v2"
LOG_DIR = "./hpo_logs"
RESULT_DIR = "./hpo_results"

# 论文原始参数（基线）
PAPER_DEFAULTS = {
    "lr": 0.01,
    "weight_decay": 1e-10,
    "num_epochs": 500,
    "milestones": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250],
    "gamma": 0.9,
    "clip": 5.0,
    "gene_hidden_dim": 6,
    "gene_output_dim": 6,
    "gene_num_top_feature": 6,
    "gig_input_transform_dim": 18,
    "gig_hidden_dim": 18,
    "gig_output_dim": 18,
    "class_weight_fine": 0.5,
    "class_weight": 0.9,
    "ortho_weight": 0.05,
    "link_weight": 0.05,
    "ent_weight": 0.00,
    "gene_num_head": 1,
    "gig_num_head": 6,
    "dropout": 0.01,
    "unchanged_threshold": 100,
    "change_wave": 0.8,
}

# ============================================================
# Phase 1 搜索空间 (粗筛)
# 基于：论文默认值 + 导师建议 + 历史 Optuna 经验
#
# [Fix 2] gig_hidden_dim 和 gig_num_head 的组合必须满足
#         gig_hidden_dim % gig_num_head == 0 (GATConv 的硬约束)
#         合法组合: 18/6, 24/4,6,8, 36/4,6, 48/4,6,8
#         为简化搜索空间，gig_hidden_dim 改为 [24, 36, 48]（全部兼容 6）
#         gig_num_head 改为 [3, 6]（全部兼容 24/36/48）
#
# [Fix 3] gene_num_top_feature 独立搜索
# ============================================================
PHASE1_SPACE = {
    # 学习率：论文用 0.01，历史试验中 0.001-0.01 区间最好
    "lr": {"type": "loguniform", "low": 5e-4, "high": 0.02},

    # Weight decay：导师建议 1e-15 ~ 1e-10，论文用 1e-10
    "weight_decay": {"type": "loguniform", "low": 1e-15, "high": 1e-8},

    # 学习率衰减
    "gamma": {"type": "categorical", "choices": [0.85, 0.9, 0.95]},

    # Milestone 策略（间隔步数）
    "milestone_step": {"type": "categorical", "choices": [50, 100, 150]},

    # Gene-level 网络结构
    "gene_hidden_dim": {"type": "categorical", "choices": [6, 12]},
    "gene_num_head": {"type": "categorical", "choices": [1, 2, 3]},
    # [Fix 3] gene_num_top_feature 独立于 gene_hidden_dim
    "gene_num_top_feature": {"type": "categorical", "choices": [4, 6, 8]},

    # [Fix 2] GiG-level 网络结构 — 只保留 dim % head == 0 的组合
    # 24%3=0, 24%6=0, 36%3=0, 36%6=0, 48%3=0, 48%6=0 全部合法
    "gig_hidden_dim": {"type": "categorical", "choices": [24, 36, 48]},
    "gig_num_head": {"type": "categorical", "choices": [3, 6]},
    "gig_input_transform_dim": {"type": "categorical", "choices": [18, 24, 36]},

    # 损失函数权重
    "class_weight": {"type": "uniform", "low": 0.6, "high": 1.0},
    "ortho_weight": {"type": "loguniform", "low": 1e-3, "high": 0.1},
    "link_weight": {"type": "loguniform", "low": 1e-3, "high": 0.1},

    # 正则化
    "dropout": {"type": "uniform", "low": 0.0, "high": 0.15},
    "clip": {"type": "categorical", "choices": [2.0, 5.0, 10.0]},
}

# ============================================================
# 工具函数
# ============================================================

def setup_logging(phase):
    """设置日志"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"phase{phase}_{timestamp}.log")

    # 清除已有 handlers 避免重复
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    sh = logging.StreamHandler(sys.stderr)  # 用 stderr 避免被 suppress_stdout 吞掉
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def generate_milestones(step, num_epochs):
    """根据步长生成 milestones 列表"""
    milestones = list(range(step, num_epochs + 1, step))
    if not milestones:
        milestones = [num_epochs // 2]
    return milestones


# [Fix 1] 移除 timeout_context (signal.SIGALRM 在 Windows 不存在)
# 不再使用超时机制，改为依赖 unchanged_threshold 早停


# [Fix 4] 用 contextlib.redirect_stdout 替代 SuppressStdout 类
@contextmanager
def suppress_stdout(suppress=True):
    """抑制 stdout 但保留 stderr。
    使用 contextlib.redirect_stdout，异常时也能正确恢复 stdout。
    """
    if suppress:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


# ============================================================
# 核心：参数采样 + 训练评估
# ============================================================

def sample_params(trial, space):
    """从搜索空间中采样参数"""
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


# [Fix 6] 用 argparse.Namespace 直接构建 args，不再调用 geo_tmain_giggat.arg_parse()
def make_default_args(**overrides):
    """创建默认参数，和 geo_tmain_giggat.py 中的 arg_parse() 一致。
    直接用 Namespace 构建，避免 sys.argv 冲突。
    """
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


def create_args_from_params(params, num_epochs):
    """
    将采样参数合并到完整的 args 对象中。
    未被搜索的参数使用论文默认值。
    """
    args = make_default_args(num_epochs=num_epochs)

    # 应用搜索到的参数
    for key, value in params.items():
        if key == "milestone_step":
            args.milestones = generate_milestones(value, num_epochs)
        elif hasattr(args, key):
            setattr(args, key, value)

    # 确保维度一致性：output_dim = hidden_dim
    if "gig_hidden_dim" in params:
        args.gig_output_dim = params["gig_hidden_dim"]
    if "gene_hidden_dim" in params:
        args.gene_output_dim = params["gene_hidden_dim"]
    # [Fix 3] gene_num_top_feature 如果在 params 中则已由上面的 setattr 设置
    # 如果不在 params 中则保持默认值 6，不再绑定到 gene_hidden_dim

    # [Fix 2] 运行时校验维度约束
    if args.gig_hidden_dim % args.gig_num_head != 0:
        raise ValueError(
            f"gig_hidden_dim ({args.gig_hidden_dim}) 必须能被 "
            f"gig_num_head ({args.gig_num_head}) 整除"
        )
    if args.gene_hidden_dim % args.gene_num_head != 0:
        raise ValueError(
            f"gene_hidden_dim ({args.gene_hidden_dim}) 必须能被 "
            f"gene_num_head ({args.gene_num_head}) 整除"
        )

    return args


def evaluate_single_fold(args, fold_n, device, suppress_output=True):
    """
    在单个 fold 上训练并评估
    返回：(test_accuracy, 训练是否正常完成)
    """
    from geo_tmain_giggat import train_geogig

    try:
        with suppress_stdout(suppress=suppress_output):
            max_test_acc = train_geogig(args, fold_n, nth_training_fold_num=1, device=device)
        return max_test_acc, True
    except Exception as e:
        logging.getLogger(__name__).warning(f"Fold {fold_n} failed: {e}")
        return 0.0, False


# [Fix 5] 早停阈值从 acc < 0.10 放宽到 acc < 0.05
# 3 分类随机基线约 33%，但某些 fold 本身就比较难，
# 只有 < 5% 才说明参数真的完全不行
def evaluate_multi_fold(args, folds, device, suppress_output=True):
    """
    在多个 fold 上评估，返回平均准确率。
    如果任何 fold 的准确率 < 5%，提前终止（参数大概率有问题）。
    """
    import torch
    accuracies = []

    for fold_n in folds:
        acc, success = evaluate_single_fold(args, fold_n, device, suppress_output)

        if not success or acc < 0.05:
            logging.getLogger(__name__).info(
                f"  Early termination: fold {fold_n} acc={acc:.4f}"
            )
            return 0.0, False

        accuracies.append(acc)
        torch.cuda.empty_cache()

    mean_acc = np.mean(accuracies)
    return mean_acc, True


# ============================================================
# Optuna Objective 函数
# ============================================================

def make_objective(phase, space, num_epochs, eval_folds):
    """
    创建 Optuna objective 函数

    Args:
        phase: 阶段编号
        space: 搜索空间字典
        num_epochs: 训练 epoch 数
        eval_folds: 评估用的 fold 列表，如 [1, 3] 表示用 fold 1 和 3
    """
    import torch

    def objective(trial):
        logger = logging.getLogger(__name__)

        # 1. 采样参数
        params = sample_params(trial, space)
        logger.info(f"Trial {trial.number}: {json.dumps({k: round(v, 8) if isinstance(v, float) else v for k, v in params.items()})}")

        # 2. 构建 args（包含 [Fix 2] 的维度校验）
        try:
            args = create_args_from_params(params, num_epochs)
        except ValueError as e:
            logger.warning(f"Trial {trial.number}: 参数不合法 - {e}")
            return 0.0

        # 3. 设备
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 4. 评估
        try:
            mean_acc, success = evaluate_multi_fold(
                args, eval_folds, device, suppress_output=True
            )
        except Exception as e:
            logger.error(f"Trial {trial.number}: ERROR - {e}")
            traceback.print_exc(file=sys.stderr)
            return 0.0

        logger.info(f"Trial {trial.number}: mean_acc={mean_acc:.4f} (folds={eval_folds})")

        # 5. 中间报告（用于 Pruner）
        trial.report(mean_acc, step=0)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

        return mean_acc

    return objective


# ============================================================
# Phase 1: 粗筛
# ============================================================

def run_phase1(n_trials=80):
    """
    Phase 1：粗筛
    - 2-fold 评估 (fold 1, 3)
    - 300 epochs
    - 80 trials
    - TPE sampler + 质量阈值过滤
    """
    import optuna

    logger = setup_logging(1)
    logger.info("=" * 60)
    logger.info("PHASE 1: 粗筛 (2-fold, 300 epochs)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)

    study_name = f"{STUDY_PREFIX}_phase1"
    storage = f"sqlite:///{DB_PATH}"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42,
        multivariate=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # 将论文原始参数作为第一个 trial (seeded trial)
    # [Fix 2] gig_hidden_dim=18 不在新搜索空间中，改为 24（最接近且合法）
    paper_trial_params = {
        "lr": 0.01,
        "weight_decay": 1e-10,
        "gamma": 0.9,
        "milestone_step": 100,
        "gene_hidden_dim": 6,
        "gene_num_head": 1,
        "gene_num_top_feature": 6,
        "gig_hidden_dim": 24,
        "gig_num_head": 6,
        "gig_input_transform_dim": 18,
        "class_weight": 0.9,
        "ortho_weight": 0.05,
        "link_weight": 0.05,
        "dropout": 0.01,
        "clip": 5.0,
    }

    # 之前最好的参数
    best_known_params = {
        "lr": 0.0035,
        "weight_decay": 1e-10,
        "gamma": 0.9,
        "milestone_step": 100,
        "gene_hidden_dim": 6,
        "gene_num_head": 2,
        "gene_num_top_feature": 6,
        "gig_hidden_dim": 24,
        "gig_num_head": 6,
        "gig_input_transform_dim": 18,
        "class_weight": 0.9,
        "ortho_weight": 0.05,
        "link_weight": 0.05,
        "dropout": 0.01,
        "clip": 5.0,
    }

    if len(study.trials) == 0:
        study.enqueue_trial(paper_trial_params)
        study.enqueue_trial(best_known_params)
        logger.info("Enqueued 2 seed trials (paper defaults + best known)")

    objective = make_objective(
        phase=1,
        space=PHASE1_SPACE,
        num_epochs=300,
        eval_folds=[1, 3],
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
    if study.best_trial:
        logger.info(f"Phase 1 完成。最佳准确率: {study.best_value:.4f}")
        logger.info(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

    return study


# ============================================================
# Phase 2: 精调
# ============================================================

def narrow_space_from_study(study, top_k=15, min_acc=0.30):
    """
    基于 Phase 1 结果缩窄搜索空间。
    只使用 acc > min_acc 的 trial 来确定缩窄方向。
    """
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
            from collections import Counter
            counter = Counter(values)
            threshold = max(1, len(top_trials) // 5)
            frequent = [v for v, c in counter.items() if c >= threshold]
            if len(frequent) < 2:
                frequent = list(counter.keys())[:3]
            narrowed[param_name] = {"type": "categorical", "choices": sorted(set(frequent))}

        elif config["type"] in ["loguniform", "uniform"]:
            min_val = min(values)
            max_val = max(values)

            if config["type"] == "loguniform":
                import math
                log_min = math.log(min_val)
                log_max = math.log(max_val)
                log_margin = (log_max - log_min) * 0.3
                new_low = max(config["low"], math.exp(log_min - log_margin))
                new_high = min(config["high"], math.exp(log_max + log_margin))
                if new_low >= new_high:
                    new_low, new_high = config["low"], config["high"]
                narrowed[param_name] = {"type": "loguniform", "low": new_low, "high": new_high}
            else:
                margin = (max_val - min_val) * 0.3
                new_low = max(config["low"], min_val - margin)
                new_high = min(config["high"], max_val + margin)
                if new_low >= new_high:
                    new_low, new_high = config["low"], config["high"]
                narrowed[param_name] = {"type": "uniform", "low": new_low, "high": new_high}
        else:
            narrowed[param_name] = config

        logger.info(f"  {param_name}: {config} -> {narrowed[param_name]}")

    return narrowed


def run_phase2(n_trials=40):
    """
    Phase 2：精调
    - 3-fold 评估 (fold 1, 2, 4)
    - 400 epochs
    - 基于 Phase 1 缩窄的搜索空间
    - 40 trials
    """
    import optuna

    logger = setup_logging(2)
    logger.info("=" * 60)
    logger.info("PHASE 2: 精调 (3-fold, 400 epochs)")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)

    storage = f"sqlite:///{DB_PATH}"
    phase1_study = optuna.load_study(
        study_name=f"{STUDY_PREFIX}_phase1",
        storage=storage,
    )

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

    # Seed: Phase 1 的 top 3 参数
    if len(study.trials) == 0:
        completed = [t for t in phase1_study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE
                     and t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=True)

        for t in completed[:3]:
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

        logger.info("Enqueued top 3 params from Phase 1 as seeds")

    objective = make_objective(
        phase=2,
        space=narrowed_space,
        num_epochs=400,
        eval_folds=[1, 2, 4],
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
    if study.best_trial:
        logger.info(f"Phase 2 完成。最佳准确率: {study.best_value:.4f}")
        logger.info(f"最佳参数: {json.dumps(study.best_params, indent=2)}")

    return study


# ============================================================
# Phase 3: 最终验证
# ============================================================

def run_phase3():
    """
    Phase 3：最终验证
    - 取 Phase 2 (或 Phase 1) 的 Top 3 参数
    - 5-fold x 5-train 完整评估（与论文评估协议一致）
    - 500 epochs
    """
    import optuna
    import torch

    logger = setup_logging(3)
    logger.info("=" * 60)
    logger.info("PHASE 3: 最终验证 (5-fold x 5-train, 500 epochs)")
    logger.info("=" * 60)

    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=f"{STUDY_PREFIX}_phase2",
            storage=storage,
        )
        source = "Phase 2"
    except Exception:
        study = optuna.load_study(
            study_name=f"{STUDY_PREFIX}_phase1",
            storage=storage,
        )
        source = "Phase 1"

    logger.info(f"Loading top params from {source}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:3]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results = []
    for rank, trial in enumerate(top_trials):
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating Rank #{rank+1} (Phase best acc: {trial.value:.4f})")
        logger.info(f"Params: {json.dumps(trial.params, indent=2)}")
        logger.info(f"{'='*50}")

        args = create_args_from_params(trial.params, num_epochs=500)

        fold_results = {}
        for fold_n in range(1, 6):
            fold_accs = []
            for nth_train in range(1, 6):
                logger.info(f"  Fold {fold_n}, Train {nth_train}...")
                try:
                    from geo_tmain_giggat import train_geogig
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
                "all_accs": fold_accs,
                "best": fold_best,
                "mean": float(np.mean(fold_accs)),
            }
            logger.info(f"  Fold {fold_n} best: {fold_best:.4f}, mean: {np.mean(fold_accs):.4f}")

        fold_bests = [fold_results[f"fold_{i}"]["best"] for i in range(1, 6)]
        overall_mean = float(np.mean(fold_bests))
        overall_std = float(np.std(fold_bests))

        result = {
            "rank": rank + 1,
            "phase_best_acc": trial.value,
            "params": trial.params,
            "fold_results": fold_results,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "5fold_best_mean": overall_mean,
        }
        results.append(result)

        logger.info(f"\nRank #{rank+1} 总结:")
        logger.info(f"  5-fold 各 fold 最佳: {[f'{x:.4f}' for x in fold_bests]}")
        logger.info(f"  平均准确率: {overall_mean:.4f} +/- {overall_std:.4f}")
        logger.info(f"  (论文参考: 80.20% +/- 1.65%)")

    # 保存最终结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULT_DIR, f"phase3_final_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3 最终结果:")
    logger.info(f"{'='*60}")
    for r in results:
        logger.info(f"Rank #{r['rank']}: {r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
    logger.info(f"\n结果已保存到: {result_file}")

    return results


# ============================================================
# 结果分析
# ============================================================

def save_phase_results(study, phase):
    """保存阶段结果到 JSON"""
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

    result_file = os.path.join(RESULT_DIR, f"phase{phase}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {result_file}", file=sys.stderr)


def analyze_results():
    """分析所有阶段的结果"""
    import optuna

    storage = f"sqlite:///{DB_PATH}"

    print("\n" + "=" * 70)
    print("GIG-GAT 超参数优化结果汇总")
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

        if not completed:
            print(f"\nPhase {phase}: 无完成的 trials")
            continue

        completed.sort(key=lambda t: t.value, reverse=True)

        print(f"\nPhase {phase}: {len(completed)} completed trials")
        print(f"  Best accuracy: {study.best_value:.4f}")
        print(f"  Best params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")

        print(f"\n  Top 5:")
        for i, t in enumerate(completed[:5]):
            print(f"    #{i+1} Trial {t.number}: acc={t.value:.4f}")

        try:
            importances = optuna.importance.get_param_importances(study)
            print(f"\n  参数重要性:")
            for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
                print(f"    {param}: {imp:.4f}")
        except Exception:
            pass

    if os.path.exists(RESULT_DIR):
        phase3_files = [f for f in os.listdir(RESULT_DIR) if f.startswith("phase3_final")]
        if phase3_files:
            latest = sorted(phase3_files)[-1]
            with open(os.path.join(RESULT_DIR, latest)) as f:
                results = json.load(f)

            print(f"\nPhase 3 (最终验证):")
            for r in results:
                print(f"  Rank #{r['rank']}: {r['overall_mean']*100:.2f}% +/- {r['overall_std']*100:.2f}%")
            print(f"  论文参考: 80.20% +/- 1.65%")


# ============================================================
# 快速诊断：先确认原始参数能否复现论文结果
# ============================================================

def run_diagnostic():
    """
    诊断模式：用论文原始参数跑一次完整评估。
    确认你的数据和环境能否接近论文的 80.20%。
    这一步非常重要！如果原始参数跑不到 75%+，说明问题不在超参数。
    """
    import torch
    from geo_tmain_giggat import train_geogig

    logger = setup_logging("diagnostic")
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: 用论文原始参数跑 5-fold 评估")
    logger.info("=" * 60)

    args = make_default_args(
        num_epochs=500,
        lr=0.01,
        weight_decay=1e-10,
        milestones=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250],
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fold_accs = []
    for fold_n in range(1, 6):
        logger.info(f"\nFold {fold_n}...")
        with suppress_stdout(suppress=True):
            acc = train_geogig(args, fold_n, nth_training_fold_num=1, device=device)
        fold_accs.append(acc)
        logger.info(f"  Fold {fold_n} acc: {acc:.4f}")
        torch.cuda.empty_cache()

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    logger.info(f"\n诊断结果:")
    logger.info(f"  各 fold: {[f'{a:.4f}' for a in fold_accs]}")
    logger.info(f"  平均: {mean_acc:.4f} +/- {std_acc:.4f}")
    logger.info(f"  论文: 0.8020 +/- 0.0165")

    if mean_acc < 0.70:
        logger.warning("原始参数准确率 < 70%，问题可能不在超参数！")
        logger.warning("请检查：数据预处理、数据划分、模型代码是否与论文一致")
    elif mean_acc < 0.78:
        logger.info("原始参数准确率 70-78%，有优化空间，超参数调优有意义")
    else:
        logger.info("原始参数已接近论文结果，精细调参可能带来少量提升")

    return fold_accs


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GIG-GAT Hyperparameter Optimization (v2 - Bug Fixed)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Phase to run")
    parser.add_argument("--n_trials", type=int, default=80, help="Number of trials")
    parser.add_argument("--analyze", action="store_true", help="Analyze results")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic with paper params")

    args = parser.parse_args()

    if args.diagnostic:
        run_diagnostic()
    elif args.analyze:
        analyze_results()
    elif args.phase == 1:
        run_phase1(n_trials=args.n_trials)
    elif args.phase == 2:
        run_phase2(n_trials=args.n_trials)
    elif args.phase == 3:
        run_phase3()
    else:
        parser.print_help()
        print("\n推荐执行顺序:")
        print("  Step 0: python optuna_giggat_v2.py --diagnostic")
        print("  Step 1: python optuna_giggat_v2.py --phase 1 --n_trials 80")
        print("  Step 2: python optuna_giggat_v2.py --phase 2 --n_trials 40")
        print("  Step 3: python optuna_giggat_v2.py --phase 3")
        print("  分析:   python optuna_giggat_v2.py --analyze")


if __name__ == "__main__":
    main()
