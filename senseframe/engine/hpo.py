"""
超参搜索模块：基于 Optuna 的 HPO 集成。

设计理念：
- ExperimentConfig.hpo 控制 HPO 行为（enabled/n_trials/sampler/pruner/metric/direction）
- 搜索空间来自场景容器的 get_search_space()，引擎不感知领域参数
- 目标函数可注入（objective_fn），默认实现调用 run_experiment
- 失败 trial 不中断整体搜索，记录错误后继续
- 返回结构化 HPOOutput，包含最佳参数与完整 trial 历史

使用方式：
    from senseframe.engine import ExperimentConfig
    from senseframe.engine.hpo import run_hpo

    cfg = ExperimentConfig.from_dict(yaml_dict)
    cfg.hpo.enabled = True
    result = run_hpo(cfg)
    print(result.best_params, result.best_value)
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import optuna
    from optuna.samplers import (
        CmaEsSampler,
        RandomSampler,
        TPESampler,
    )
    from optuna.pruners import (
        HyperbandPruner,
        MedianPruner,
        NopPruner,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..scenes import get_scene
from ..scenes.base import SearchSpace
from .config import ExperimentConfig, HPOConfig

logger = logging.getLogger(__name__)


# ============================================================
# TrainerConfig 字段集合（用于判断采样参数去向）
# Phase 14.1.3：从 TrainerConfig dataclass 自动派生，不再硬编码
# ============================================================
from dataclasses import fields as _dc_fields
from .config import TrainerConfig as _TrainerConfig
_TRAINER_FIELDS = frozenset(f.name for f in _dc_fields(_TrainerConfig))

# Phase 12.1：TaskSpec 相关字段（写入 scene.params 由场景容器透传到 TaskSpec）
_TASK_FIELDS = frozenset({
    "loss",                    # loss 名称（@register_loss 注册）
    "task_type",               # 任务类型（字符串，通过注册表管理）
    "output_activation",       # 输出激活（softmax/sigmoid/tanh/relu）
    "loss_kwargs",             # loss 构造 kwargs（dict）
})


# ============================================================
# 输出数据结构
# ============================================================
@dataclass
class TrialResult:
    """单次 HPO trial 结果。"""
    trial_number: int
    params: Dict[str, Any]
    metric_value: Optional[float]
    status: str                           # "complete" / "pruned" / "fail"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "params": self.params,
            "metric_value": self.metric_value,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class HPOOutput:
    """HPO 整体输出。"""
    best_params: Dict[str, Any]
    best_value: Optional[float]
    best_trial: Optional[int]
    n_trials: int
    n_complete: int
    n_pruned: int
    n_failed: int
    trials: List[TrialResult] = field(default_factory=list)
    direction: str = "minimize"
    metric: str = "val_loss"
    tracker: Optional["ExplorationTracker"] = None  # P2: 统一探索视图

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_trial": self.best_trial,
            "n_trials": self.n_trials,
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "trials": [t.to_dict() for t in self.trials],
            "direction": self.direction,
            "metric": self.metric,
        }

    def summary(self, top_n: int = 5) -> Dict[str, Any]:
        """
        Phase 4.2：生成 HPO 结果统计摘要。

        无需可视化依赖，纯数据摘要，便于日志输出与下游分析：
        - top_n trials（按 metric 值排序）
        - 参数值分布统计（min/max/mean for float params）
        - 完成率与失败率

        Args:
            top_n: 返回 top-N trial 的数量

        Returns:
            摘要字典
        """
        completed = [t for t in self.trials if t.status == "complete" and t.metric_value is not None]
        # top-N：minimize 升序，maximize 降序
        reverse = (self.direction == "maximize")
        sorted_trials = sorted(completed, key=lambda t: t.metric_value, reverse=reverse)
        top_trials = sorted_trials[:top_n]

        # 参数值分布统计（仅 float 参数）
        param_values: Dict[str, List[float]] = {}
        for t in completed:
            for k, v in t.params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    param_values.setdefault(k, []).append(float(v))

        param_stats = {}
        for k, vals in param_values.items():
            if vals:
                param_stats[k] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": sum(vals) / len(vals),
                    "count": len(vals),
                }

        total = max(self.n_trials, 1)
        return {
            "metric": self.metric,
            "direction": self.direction,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "best_trial": self.best_trial,
            "completion_rate": round(self.n_complete / total, 4),
            "failure_rate": round(self.n_failed / total, 4),
            "prune_rate": round(self.n_pruned / total, 4),
            "top_trials": [
                {
                    "trial_number": t.trial_number,
                    "metric_value": t.metric_value,
                    "params": t.params,
                }
                for t in top_trials
            ],
            "param_distribution": param_stats,
        }


# ============================================================
# 目标函数类型
# ============================================================
ObjectiveFn = Callable[[Any, ExperimentConfig, SearchSpace], float]
"""
目标函数签名：
    def objective(trial, config, search_space) -> float:
        # trial: optuna.trial.Trial
        # config: ExperimentConfig（原始，未被修改）
        # search_space: SearchSpace
        # 返回: metric 值（越大/越小越好，取决于 direction）
"""


# ============================================================
# 搜索空间采样
# ============================================================
def sample_params(trial, search_space: SearchSpace) -> Dict[str, Any]:
    """
    从搜索空间采样参数。

    Args:
        trial: optuna.trial.Trial 实例
        search_space: 场景容器提供的搜索空间

    Returns:
        采样后的参数字典

    Raises:
        ValueError: 未知的参数类型
    """
    params: Dict[str, Any] = {}
    for name, spec in search_space.params.items():
        ptype = spec.get("type")
        if ptype == "float":
            params[name] = trial.suggest_float(
                name,
                low=spec["low"],
                high=spec["high"],
                log=spec.get("log", False),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                low=spec["low"],
                high=spec["high"],
                log=spec.get("log", False),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(
                name,
                choices=spec["values"],
            )
        else:
            raise ValueError(
                f"Unknown param type '{ptype}' for param '{name}'. "
                f"Supported: float / int / categorical"
            )
    return params


# ============================================================
# 参数应用
# ============================================================
def apply_params(config: ExperimentConfig,
                 params: Dict[str, Any]) -> ExperimentConfig:
    """
    创建修改后的 config 副本，将采样参数应用到对应字段。

    规则（Phase 12.1 扩展）：
    - 参数名在 _TRAINER_FIELDS 中 → 覆盖 trainer 同名字段
    - 参数名在 _TASK_FIELDS 中    → 写入 scene.params（场景容器构造 TaskSpec 时透传）
    - 其余参数                    → 写入 scene.params（场景特定参数）

    Args:
        config: 原始配置（不会被修改）
        params: 采样参数

    Returns:
        深拷贝后的新 ExperimentConfig
    """
    new_config = copy.deepcopy(config)
    for name, value in params.items():
        if name in _TRAINER_FIELDS:
            setattr(new_config.trainer, name, value)
        else:
            # trainer / task 字段均通过 scene.params 透传
            # 场景容器在 build_model_for_dataset 阶段将 task 字段提升到 TaskSpec
            new_config.scene.params[name] = value
    return new_config


def get_task_search_space_extension() -> Dict[str, Dict[str, Any]]:
    """
    Phase 12.1：返回 task_spec / loss 搜索空间的预定义 spec。

    场景容器可在 get_search_space() 中合并这些 spec，快速启用 HPO 搜索：

    .. code-block:: python

        def get_search_space(self, model_id, dataset_name, **kwargs):
            space = SearchSpace()
            space.params.update(get_task_search_space_extension())
            space.params["loss"] = {
                "type": "categorical",
                "values": ["cross_entropy", "focal", "mse"],
            }
            return space
    """
    from ..core import list_losses, list_task_types
    return {
        "task_type": {
            "type": "categorical",
            "values": list_task_types(),
        },
        "loss": {
            "type": "categorical",
            "values": list_losses(),
        },
        "output_activation": {
            "type": "categorical",
            "values": ["none", "softmax", "sigmoid", "tanh"],
        },
    }


# ============================================================
# 指标提取
# ============================================================
def extract_metric(result, metric: str) -> float:
    """
    从 TrainOutput 提取指标值。

    查找顺序：
    1. result.final_eval[metric]
    2. result.training[metric]
    3. 特殊别名：val_loss → training.best_val_loss

    Args:
        result: TrainOutput 实例
        metric: 指标名

    Raises:
        ValueError: 指标未找到
    """
    # 特殊别名
    if metric == "val_loss":
        val = result.training.get("best_val_loss")
        if val is not None:
            return float(val)

    # final_eval 优先
    if metric in result.final_eval:
        return float(result.final_eval[metric])

    # training 字典
    if metric in result.training:
        return float(result.training[metric])

    raise ValueError(
        f"Metric '{metric}' not found in TrainOutput. "
        f"Available final_eval: {list(result.final_eval.keys())}, "
        f"training: {list(result.training.keys())}"
    )


# ============================================================
# 默认目标函数
# ============================================================
def _default_objective(trial, config: ExperimentConfig,
                       search_space: SearchSpace,
                       tracker=None) -> float:
    """
    默认目标函数：采样参数 → 修改 config → run_pipeline → 提取 metric。

    注意：此函数会触发完整训练。HPO 时建议在 config.trainer 中设置较少 epochs。

    P2: 若 tracker 不为 None，将每次 trial 的策略和结果写入 ExplorationTracker，
    统一 HPO 数值超参搜索与策略空间搜索的视图。
    """
    from ..runner import run_pipeline

    # 1. 采样
    params = sample_params(trial, search_space)

    # 2. 应用参数
    modified_config = apply_params(config, params)

    # 3. 训练
    result = run_pipeline(modified_config)

    if result.status != "success":
        # P2: 失败也记录到 tracker（feedback=numerical_instability）
        if tracker is not None:
            tracker.add_trial(
                strategy=params,
                result=None,
                feedback={"status": "numerical_instability",
                          "diagnosis": "HPO trial failed (pruned)",
                          "suggestions": []},
            )
        raise optuna.TrialPruned()

    # 4. 提取指标
    metric_value = extract_metric(result, config.hpo.metric)

    # P2: 成功 trial 写入 tracker
    if tracker is not None:
        # 从 result 提取 feedback（若 pipeline 已生成）
        feedback = None
        if hasattr(result, 'training') and result.training:
            # HPO 路径走 run_pipeline，feedback 存储在 PipelineContext.extra 中
            # 此处用 success 状态作为默认 feedback（方案 2 将 feedback 提升为 TrainOutput 字段后可直读）
            feedback = {"status": "success", "diagnosis": "HPO trial succeeded", "suggestions": []}
        tracker.add_trial(
            strategy=params,
            result={"val_accuracy": result.final_eval.get("val_accuracy"),
                    "val_loss": result.final_eval.get("val_loss"),
                    config.hpo.metric: metric_value},
            feedback=feedback,
        )

    return metric_value


# ============================================================
# Sampler / Pruner 工厂
# ============================================================
def _build_sampler(hpo_config: HPOConfig):
    """根据 HPOConfig 创建 Optuna sampler。"""
    name = hpo_config.sampler
    if name == "tpe":
        return TPESampler(seed=42)
    elif name == "random":
        return RandomSampler(seed=42)
    elif name == "cmaes":
        return CmaEsSampler(seed=42)
    else:
        raise ValueError(
            f"Unknown sampler '{name}'. Supported: tpe / random / cmaes"
        )


def _build_pruner(hpo_config: HPOConfig):
    """根据 HPOConfig 创建 Optuna pruner。"""
    name = hpo_config.pruner
    if name == "median":
        return MedianPruner()
    elif name == "none":
        return NopPruner()
    elif name == "hyperband":
        return HyperbandPruner()
    else:
        raise ValueError(
            f"Unknown pruner '{name}'. Supported: median / none / hyperband"
        )


# ============================================================
# 主入口
# ============================================================
def run_hpo(config: ExperimentConfig,
            objective_fn: Optional[ObjectiveFn] = None,
            tracker: Optional["ExplorationTracker"] = None,
            n_jobs: int = 1) -> HPOOutput:
    """
    执行超参搜索。

    Args:
        config: ExperimentConfig 实例（hpo 字段必须 enabled=True）
        objective_fn: 自定义目标函数。若为 None，使用默认实现
                      （调用 run_experiment 进行完整训练）
        tracker: ExplorationTracker 实例。P2: 若不为 None，HPO trial 会写入
                 ExplorationTracker，统一 HPO 数值超参搜索与策略空间搜索的视图。
                 若 tracker 为 None，自动创建一个临时 ExplorationTracker。
        n_jobs: 并行执行的 trial 数。n_jobs=1 时串行执行（默认）。
                s1: n_jobs > 1 时用 ThreadPoolExecutor 并行执行 trial。
                ExplorationTracker 已线程安全（s1 加锁）。

    Returns:
        HPOOutput: 包含最佳参数与完整 trial 历史

    Raises:
        ImportError: Optuna 未安装
        ValueError: HPO 未启用 / 场景未注册 / 搜索空间为空
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Install with: pip install optuna>=3.5.0"
        )

    # 1. 校验 HPO 启用
    if not config.hpo.enabled:
        raise ValueError(
            "HPO is not enabled. Set config.hpo.enabled = True."
        )
    config.hpo.validate()

    # P2: 初始化 ExplorationTracker（自动创建或使用传入的）
    if tracker is None:
        from ..exploration import ExplorationTracker
        tracker = ExplorationTracker()

    # 2. 校验场景并获取搜索空间
    scene = get_scene(config.scene.name)
    search_space = scene.get_search_space(
        config.scene.model_id, config.scene.dataset
    )
    if search_space.is_empty():
        raise ValueError(
            f"Scene '{config.scene.name}' returned empty search space for "
            f"model='{config.scene.model_id}', dataset='{config.scene.dataset}'. "
            f"HPO requires a non-empty search space."
        )

    # 3. 确定目标函数
    if objective_fn is None:
        objective_fn = _default_objective

    # 4. 创建 study（Phase 4.1：支持持久化与断点续搜）
    sampler = _build_sampler(config.hpo)
    pruner = _build_pruner(config.hpo)
    study_kwargs = {
        "direction": config.hpo.direction,
        "sampler": sampler,
        "pruner": pruner,
    }
    # Phase 4.1：持久化后端
    if config.hpo.storage is not None:
        study_kwargs["storage"] = config.hpo.storage
    # Phase 4.1：study 名称（配合 load_if_exists 实现断点续搜）
    if config.hpo.study_name is not None:
        study_kwargs["study_name"] = config.hpo.study_name
    # Phase 4.1：断点续搜
    if config.hpo.load_if_exists:
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)

    # Phase 4.1：断点续搜时，计算剩余 trial 数
    existing_trials = len(study.trials)
    remaining_trials = max(0, config.hpo.n_trials - existing_trials)
    if existing_trials > 0:
        logger.info(
            f"HPO resumed: study='{config.hpo.study_name}', "
            f"existing_trials={existing_trials}, remaining_trials={remaining_trials}"
        )

    logger.info(
        f"HPO started: n_trials={config.hpo.n_trials} (remaining={remaining_trials}), "
        f"sampler={config.hpo.sampler}, pruner={config.hpo.pruner}, "
        f"metric={config.hpo.metric}, direction={config.hpo.direction}, "
        f"storage={'persistent' if config.hpo.storage else 'in-memory'}"
    )

    # 5. 执行 trials（Phase 4.1：仅执行剩余 trial 数）
    trials: List[TrialResult] = []
    n_complete = 0
    n_pruned = 0
    n_failed = 0

    # Phase 4.1：从已有 study 恢复 trial 历史（断点续搜时）
    if existing_trials > 0:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                n_complete += 1
                status = "complete"
                metric_val = t.value
            elif t.state == optuna.trial.TrialState.PRUNED:
                n_pruned += 1
                status = "pruned"
                metric_val = t.value if t.value is not None else None
            elif t.state == optuna.trial.TrialState.FAIL:
                n_failed += 1
                status = "fail"
                metric_val = None
            else:
                continue  # RUNNING / WAITING
            trials.append(TrialResult(
                trial_number=t.number,
                params=dict(t.params),
                metric_value=metric_val,
                status=status,
            ))

    # Phase 4.1：超时控制起始时间
    import time
    start_time = time.time()

    if n_jobs > 1:
        # s1: 并行执行 trial（ThreadPoolExecutor）
        import concurrent.futures

        def _run_single_trial(trial_idx):
            trial = study.ask()
            try:
                value = objective_fn(trial, config, search_space, tracker)
                return trial, value, None, None
            except optuna.TrialPruned:
                return trial, None, "pruned", None
            except Exception as e:
                return trial, None, "fail", str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_run_single_trial, i) for i in range(remaining_trials)]
            for future in concurrent.futures.as_completed(futures):
                trial, value, error_type, error_msg = future.result()
                trial_result = TrialResult(
                    trial_number=trial.number,
                    params={},
                    metric_value=None,
                    status="fail",
                )
                if value is not None:
                    study.tell(trial, value)
                    trial_result.params = dict(trial.params)
                    trial_result.metric_value = value
                    trial_result.status = "complete"
                    n_complete += 1
                elif error_type == "pruned":
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    trial_result.params = dict(trial.params)
                    trial_result.status = "pruned"
                    n_pruned += 1
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    trial_result.params = dict(trial.params)
                    trial_result.status = "fail"
                    trial_result.error = error_msg
                    n_failed += 1
                    logger.warning(
                        f"Trial {trial.number} failed: {error_msg}"
                    )
                trials.append(trial_result)
    else:
        # 原有串行逻辑（含 Phase 4.1 超时控制）
        for trial_idx in range(remaining_trials):
            # Phase 4.1：超时检查
            if config.hpo.timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= config.hpo.timeout:
                    logger.info(
                        f"HPO timeout reached: {elapsed:.1f}s >= {config.hpo.timeout}s, "
                        f"stopping after {len(trials)} trials"
                    )
                    break

            trial = study.ask()
            trial_result = TrialResult(
                trial_number=trial.number,
                params={},
                metric_value=None,
                status="fail",
            )

            try:
                value = objective_fn(trial, config, search_space, tracker)
                study.tell(trial, value)
                trial_result.params = dict(trial.params)
                trial_result.metric_value = value
                trial_result.status = "complete"
                n_complete += 1
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                trial_result.params = dict(trial.params)
                trial_result.status = "pruned"
                n_pruned += 1
            except Exception as e:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                trial_result.params = dict(trial.params)
                trial_result.status = "fail"
                trial_result.error = str(e)
                n_failed += 1
                logger.warning(
                    f"Trial {trial.number} failed: {e}"
                )

            trials.append(trial_result)

    # 6. 提取最佳结果
    best_params = {}
    best_value = None
    best_trial = None
    try:
        best = study.best_trial
        best_params = dict(best.params)
        best_value = best.value
        best_trial = best.number
    except ValueError:
        # 所有 trial 都被 prune 或失败时，study 无 best_trial
        logger.warning(
            "No completed trials — best_params unavailable "
            f"(complete={n_complete}, pruned={n_pruned}, failed={n_failed})"
        )

    output = HPOOutput(
        best_params=best_params,
        best_value=best_value,
        best_trial=best_trial,
        n_trials=len(trials),
        n_complete=n_complete,
        n_pruned=n_pruned,
        n_failed=n_failed,
        trials=trials,
        direction=config.hpo.direction,
        metric=config.hpo.metric,
    )

    logger.info(
        f"HPO completed: best_value={best_value}, "
        f"best_params={best_params}, "
        f"complete={n_complete}, pruned={n_pruned}, failed={n_failed}"
    )

    # Phase 4.1/4.2：结果导出到 JSON（含 summary 摘要）
    if config.hpo.export_path is not None:
        import json
        from pathlib import Path
        export_path = Path(config.hpo.export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_data = output.to_dict()
        # Phase 4.2：附加统计摘要
        export_data["summary"] = output.summary()
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"HPO result exported to: {export_path}")

    # P2: 将 tracker 赋值到 output，统一探索视图
    output.tracker = tracker
    return output


__all__ = [
    "TrialResult",
    "HPOOutput",
    "ObjectiveFn",
    "run_hpo",
    "sample_params",
    "apply_params",
    "extract_metric",
    "OPTUNA_AVAILABLE",
]
