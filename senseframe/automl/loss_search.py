"""RFC-003 ε1：损失函数搜索 — SP 协议的最小应用。

通过 SP（search_protocol）的 ask/tell 驱动损失函数搜索，验证协议栈可行性。

定位：
- 搜索空间 = 损失注册表（list_losses()）+ label_smoothing 浮点
- 采样策略 = SP Sampler（RandomSampler / GridSampler）
- 评估 = run_pipeline（每次评估是一次完整训练）
- 不走 Optuna 路径（HPO 模块已覆盖 Optuna 路径）

与 HPO 的关系：
- HPO（Optuna 路径）：数值超参搜索，场景容器提供搜索空间
- ε1（SP 路径）：损失函数组合搜索，验证 SP ask/tell 可驱动搜索
- 两者并存，不互相替换
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core import list_losses
from ..engine.config import ExperimentConfig
from ..engine.hpo import apply_params, extract_metric
from ..engine.runner.pipeline import run_pipeline
from ..search_protocol import (
    ParameterSpec,
    SearchSpace,
    StudyManager,
    TrialResult,
    get_study_manager,
)

logger = logging.getLogger(__name__)


# ============================================================
# 搜索空间构造
# ============================================================
def build_loss_search_space(
    include_label_smoothing: bool = True,
    extra_losses: Optional[List[str]] = None,
) -> SearchSpace:
    """从 list_losses() 动态构造 SP SearchSpace（ε1）。

    Args:
        include_label_smoothing: 是否包含 label_smoothing 浮点参数
        extra_losses: 额外追加的 loss 名称（如自定义注册的 loss）

    Returns:
        SearchSpace: 含 loss（categorical）+ 可选 label_smoothing（float）的搜索空间

    验证：
        - choices 来自 list_losses()，动态反映注册表状态
        - label_smoothing 范围 0.0-0.3（常见值域）
    """
    losses = list_losses()
    if extra_losses:
        losses = list(set(losses) | set(extra_losses))

    parameters = [
        ParameterSpec(
            name="loss",
            type="categorical",
            choices=losses,
        ),
    ]
    if include_label_smoothing:
        parameters.append(
            ParameterSpec(
                name="label_smoothing",
                type="float",
                low=0.0,
                high=0.3,
            )
        )
    return SearchSpace(parameters=parameters)


# ============================================================
# 输出数据结构
# ============================================================
@dataclass
class LossSearchResult:
    """ε1 损失搜索结果（SP 路径）。

    与 HPOOutput 区别：
    - 不含 Optuna 特定字段（trial_number / n_pruned）
    - trials 使用 SP 的 TrialResult（schema_version + 自省友好）
    - 保留 study_id 供后续查询（如 list_trials / best_trial）
    """
    study_id: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: Optional[float] = None
    n_trials: int = 0
    n_completed: int = 0
    n_failed: int = 0
    trials: List[TrialResult] = field(default_factory=list)
    direction: str = "maximize"
    metric: str = "val_accuracy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "trials": [t.to_dict() for t in self.trials],
            "direction": self.direction,
            "metric": self.metric,
        }


# ============================================================
# 参数应用（扩展 apply_params 支持 label_smoothing → loss_kwargs）
# ============================================================
def _apply_loss_params(
    config: ExperimentConfig,
    params: Dict[str, Any],
) -> ExperimentConfig:
    """将 SP 采样的损失参数应用到 config（ε1）。

    在 apply_params 基础上扩展：
    - loss → scene.params["loss"]
    - label_smoothing → scene.params["loss_kwargs"]["label_smoothing"]
      （合并已有 loss_kwargs，不覆盖其他 key）

    Args:
        config: 原始配置（不会被修改）
        params: SP 采样参数（含 loss / label_smoothing）

    Returns:
        深拷贝后的新 ExperimentConfig
    """
    new_config = copy.deepcopy(config)

    # 复用 apply_params 处理 loss（写入 scene.params["loss"]）
    # 但 label_smoothing 需要特殊处理，先剥离
    params_for_apply = {k: v for k, v in params.items() if k != "label_smoothing"}
    new_config = apply_params(new_config, params_for_apply)

    # label_smoothing → loss_kwargs.label_smoothing
    if "label_smoothing" in params:
        existing_kwargs = dict(new_config.scene.params.get("loss_kwargs", {}) or {})
        existing_kwargs["label_smoothing"] = params["label_smoothing"]
        new_config.scene.params["loss_kwargs"] = existing_kwargs

    return new_config


# ============================================================
# 主入口
# ============================================================
def run_loss_search(
    config: ExperimentConfig,
    n_trials: int = 10,
    direction: str = "maximize",
    metric: str = "val_accuracy",
    sampler: str = "random",
    include_label_smoothing: bool = True,
    study_manager: Optional[StudyManager] = None,
) -> LossSearchResult:
    """通过 SP ask/tell 驱动损失函数搜索（ε1）。

    流程：
    1. 创建 SP Study（search_space = build_loss_search_space()）
    2. for _ in range(n_trials):
         trial = sm.ask(study_id)
         modified = _apply_loss_params(config, trial.params)
         result = run_pipeline(modified)
         sm.tell(trial.trial_id, value, state)
    3. 返回 LossSearchResult（含 best_params / trials / study_id）

    Args:
        config: ExperimentConfig 实例（不会被修改）
        n_trials: 试验次数
        direction: "maximize" / "minimize"
        metric: 评估指标名（如 "val_accuracy" / "val_loss"）
        sampler: SP Sampler 名（"random" / "grid"）
        include_label_smoothing: 搜索空间是否含 label_smoothing
        study_manager: 可选的 StudyManager（None 时用全局单例）

    Returns:
        LossSearchResult
    """
    sm = study_manager or get_study_manager()

    # 1. 创建 Study
    search_space = build_loss_search_space(
        include_label_smoothing=include_label_smoothing
    )
    study_id = sm.create_study(
        name="loss_search",
        direction=direction,
        search_space=search_space,
        sampler=sampler,
    )

    logger.info(
        "ε1 loss search started: n_trials=%d, sampler=%s, metric=%s, direction=%s, "
        "losses=%s",
        n_trials, sampler, metric, direction,
        search_space.parameters[0].choices,
    )

    n_completed = 0
    n_failed = 0

    # 2. SP ask/tell 循环
    for trial_idx in range(n_trials):
        trial = sm.ask(study_id)

        try:
            modified_config = _apply_loss_params(config, trial.params)
            result = run_pipeline(modified_config)

            if result.status != "success":
                sm.tell(
                    trial.trial_id,
                    value=0.0,
                    state="failed",
                    feedback={"error": result.error, "error_code": result.error_code},
                )
                n_failed += 1
                logger.warning(
                    "trial %d failed: loss=%s, error=%s",
                    trial_idx, trial.params.get("loss"), result.error,
                )
                continue

            value = extract_metric(result, metric)
            sm.tell(
                trial.trial_id,
                value=value,
                state="completed",
                feedback={
                    "model_path": result.model_path,
                    "output_dir": result.output_dir,
                    "final_eval": result.final_eval,
                },
            )
            n_completed += 1
            logger.info(
                "trial %d completed: loss=%s, %s=%.4f",
                trial_idx, trial.params.get("loss"), metric, value,
            )

        except Exception as e:
            sm.tell(
                trial.trial_id,
                value=0.0,
                state="failed",
                feedback={"error": str(e)},
            )
            n_failed += 1
            logger.warning(
                "trial %d exception: loss=%s, error=%s",
                trial_idx, trial.params.get("loss"), e,
            )

    # 3. 提取最佳结果
    trials = sm.list_trials(study_id)
    best = sm.best_trial(study_id)

    best_params = best.params if best else {}
    best_value = best.value if best else None

    logger.info(
        "ε1 loss search completed: best_value=%s, best_params=%s, "
        "completed=%d, failed=%d",
        best_value, best_params, n_completed, n_failed,
    )

    return LossSearchResult(
        study_id=study_id,
        best_params=best_params,
        best_value=best_value,
        n_trials=n_trials,
        n_completed=n_completed,
        n_failed=n_failed,
        trials=trials,
        direction=direction,
        metric=metric,
    )


__all__ = [
    "build_loss_search_space",
    "run_loss_search",
    "LossSearchResult",
]
