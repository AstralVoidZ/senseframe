"""P3 阶段 8：PEFT 微调策略搜索 — SP 协议的新应用。

通过 SP ask/tell 驱动 PEFT 微调策略搜索，验证"搜索微调策略而非架构"的新 AutoML 范式。

定位：
- 搜索空间 = PEFT 配置（method + rank + alpha + dropout + target + lr + adapter_bottleneck + prompt_length + freeze_backbone）
- 采样策略 = SP Sampler（RandomSampler / GridSampler）
- 评估 = run_pipeline（每次评估是一次完整微调训练）
- 与 loss_search.py 对称：loss_search 搜损失，peft_search 搜微调策略

与 NAS 的关系：
- NAS（DARTS）：搜索架构（cell_type / n_layers / hidden_dim）
- PEFT 搜索：搜索微调策略（peft_method / rank / alpha）
- 两者并存，不互相替换
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..engine.config import ExperimentConfig
from ..engine.hpo import extract_metric
from ..engine.runner.pipeline import run_pipeline
from ..search_protocol import (
    ParameterSpec,
    SearchSpace,
    StudyManager,
    TrialResult,
    get_study_manager,
)
from .peft_builder import PEFTBuilder

logger = logging.getLogger(__name__)


# ============================================================
# 搜索空间构造
# ============================================================
_DEFAULT_PEFT_METHODS = ["lora", "adapter", "prefix_tuning", "prompt_tuning", "full"]


def build_peft_search_space(
    include_methods: Optional[List[str]] = None,
) -> SearchSpace:
    """构造 PEFT 微调策略搜索空间。

    Args:
        include_methods: 包含的 PEFT 方法子集（None 时用全部 5 种）

    Returns:
        SearchSpace: 含 9 个 PEFT 参数的搜索空间
    """
    methods = include_methods if include_methods is not None else list(_DEFAULT_PEFT_METHODS)

    parameters = [
        ParameterSpec(
            name="peft_method",
            type="categorical",
            choices=methods,
        ),
        ParameterSpec(
            name="peft_rank",
            type="categorical",
            choices=[4, 8, 16, 32, 64],
        ),
        ParameterSpec(
            name="peft_alpha",
            type="categorical",
            choices=[1, 2, 4],
        ),
        ParameterSpec(
            name="peft_dropout",
            type="float",
            low=0.0,
            high=0.3,
            step=0.05,
        ),
        ParameterSpec(
            name="peft_target_modules",
            type="categorical",
            choices=["query", "value", "query_value", "all"],
        ),
        ParameterSpec(
            name="learning_rate",
            type="float",
            low=1e-5,
            high=1e-3,
            log=True,
        ),
        ParameterSpec(
            name="adapter_bottleneck",
            type="categorical",
            choices=[64, 128, 256],
        ),
        ParameterSpec(
            name="prompt_length",
            type="categorical",
            choices=[5, 10, 20, 50],
        ),
        ParameterSpec(
            name="freeze_backbone",
            type="categorical",
            choices=[True, False],
        ),
    ]
    return SearchSpace(parameters=parameters)


# ============================================================
# 输出数据结构
# ============================================================
@dataclass
class PEFTSearchResult:
    """PEFT 微调策略搜索结果（SP 路径）。

    与 LossSearchResult 对称：同样的字段结构，便于统一探索视图。
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
# 参数应用
# ============================================================
def _apply_peft_params(
    config: ExperimentConfig,
    foundation_model: nn.Module,
    params: Dict[str, Any],
) -> ExperimentConfig:
    """将 SP 采样的 PEFT 参数应用到 config。

    - 在 foundation_model 的深拷贝上构建 PEFT 模块
    - PEFT 参数写入 scene.params（场景容器可见）
    - learning_rate 写入 trainer
    - 通过 module_factory 注入 PEFT 模型（覆盖场景默认模型）

    Args:
        config: 原始配置（不会被修改）
        foundation_model: 基础模型（不会被修改，深拷贝后构建 PEFT）
        params: SP 采样参数

    Returns:
        深拷贝后的新 ExperimentConfig
    """
    new_config = copy.deepcopy(config)

    # 在 foundation_model 的深拷贝上构建 PEFT（避免污染原模型）
    # P3-P2-5 修复：大模型（如 MAE 预训练 CSI 基础模型）deepcopy 可能耗时数秒，
    # 应 log 起止便于性能诊断与卡顿定位。
    import time as _time
    _t0 = _time.perf_counter()
    foundation_copy = copy.deepcopy(foundation_model)
    _deepcopy_ms = (_time.perf_counter() - _t0) * 1000
    logger.debug(
        "P3 peft search: foundation_model deepcopy took %.2f ms "
        "(model_id=%s, params=%d)",
        _deepcopy_ms,
        getattr(foundation_model, "model_id", "<unknown>"),
        sum(p.numel() for p in foundation_model.parameters()),
    )
    peft_model = PEFTBuilder.build(foundation_copy, params)

    # PEFT 参数写入 scene.params（场景容器可见 + 可追溯）
    if new_config.scene.params is None:
        from ..core.params import SceneParams
        new_config.scene.params = SceneParams()
    for k, v in params.items():
        new_config.scene.params[k] = v

    # learning_rate 写入 trainer
    if "learning_rate" in params:
        lr = params["learning_rate"]
        try:
            new_config.trainer.learning_rate = float(lr)
        except (TypeError, ValueError):
            pass

    # 通过 module_factory 注入 PEFT 模型
    original_factory = new_config.module_factory

    def peft_module_factory(model, **kwargs):
        # 忽略场景默认 model，使用 PEFT 模型
        if original_factory is not None:
            return original_factory(peft_model, **kwargs)
        raise RuntimeError(
            "module_factory not configured: PEFT 模型注入需要 config.module_factory "
            "提供 LightningModule 包装器"
        )

    new_config.module_factory = peft_module_factory
    return new_config


# ============================================================
# 主入口
# ============================================================
def run_peft_search(
    config: ExperimentConfig,
    foundation_model: nn.Module,
    n_trials: int = 10,
    direction: str = "maximize",
    metric: str = "val_accuracy",
    sampler: str = "random",
    study_manager: Optional[StudyManager] = None,
) -> PEFTSearchResult:
    """通过 SP ask/tell 驱动 PEFT 微调策略搜索。

    流程：
    1. 创建 SP Study（search_space = build_peft_search_space()）
    2. for _ in range(n_trials):
         trial = sm.ask(study_id)
         modified = _apply_peft_params(config, foundation_model, trial.params)
         result = run_pipeline(modified)
         sm.tell(trial.trial_id, value, state)
    3. 返回 PEFTSearchResult

    Args:
        config: ExperimentConfig 实例（不会被修改）
        foundation_model: 基础模型（nn.Module）。若实现 SensingFoundationModel
                         Protocol，取其 .model_id 用作 study name
        n_trials: 试验次数
        direction: "maximize" / "minimize"
        metric: 评估指标名
        sampler: SP Sampler 名
        study_manager: 可选的 StudyManager（None 时用全局单例）

    Returns:
        PEFTSearchResult
    """
    sm = study_manager or get_study_manager()

    # 1. 创建 Study
    search_space = build_peft_search_space()
    model_id = getattr(foundation_model, "model_id", None)
    study_name = f"peft_search_{model_id}" if model_id else "peft_search"

    study_id = sm.create_study(
        name=study_name,
        direction=direction,
        search_space=search_space,
        sampler=sampler,
    )

    logger.info(
        "P3 stage8 peft search started: n_trials=%d, sampler=%s, metric=%s, "
        "direction=%s, study_name=%s",
        n_trials, sampler, metric, direction, study_name,
    )

    n_completed = 0
    n_failed = 0

    # P3-1 修复：try/finally 包裹整个 ask/tell 循环，确保异常时
    # 仍能 stop_study 释放 SP Study 资源（trial 队列 / sampler state）。
    # 原实现中 sm.ask 异常未捕获，study_id 资源会泄露。
    try:
        # 2. SP ask/tell 循环
        for trial_idx in range(n_trials):
            trial = sm.ask(study_id)

            try:
                modified_config = _apply_peft_params(
                    config, foundation_model, trial.params
                )
                result = run_pipeline(modified_config)

                if result.status != "success":
                    sm.tell(
                        trial.trial_id,
                        value=0.0,
                        state="failed",
                        feedback={
                            "error": result.error,
                            "error_code": getattr(result, "error_code", None),
                        },
                    )
                    n_failed += 1
                    logger.warning(
                        "trial %d failed: peft_method=%s, error=%s",
                        trial_idx, trial.params.get("peft_method"), result.error,
                    )
                    continue

                # P3-P2-6 修复：extract_metric 返回 None 时用 0.0 兜底，
                # 避免 sm.tell(value=None) 在内部比较时崩溃。
                value = extract_metric(result, metric)
                if value is None:
                    logger.warning(
                        "trial %d: metric '%s' not found in result, using 0.0",
                        trial_idx, metric,
                    )
                    value = 0.0
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
                    "trial %d completed: peft_method=%s, %s=%.4f",
                    trial_idx, trial.params.get("peft_method"), metric, value,
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
                    "trial %d exception: peft_method=%s, error=%s",
                    trial_idx, trial.params.get("peft_method"), e,
                )
    finally:
        # P3-1 修复：无论循环正常结束还是异常，都尝试停止 study 释放资源。
        # stop_study 可能不存在（旧版 SP），用 getattr 防御。
        try:
            if hasattr(sm, "stop_study"):
                sm.stop_study(study_id)
        except Exception as cleanup_err:
            logger.warning(
                "P3 peft search: stop_study failed for %s: %s",
                study_id, cleanup_err,
            )

    # 3. 提取最佳结果
    trials = sm.list_trials(study_id)
    best = sm.best_trial(study_id)

    best_params = best.params if best else {}
    best_value = best.value if best else None

    logger.info(
        "P3 stage8 peft search completed: best_value=%s, best_params=%s, "
        "completed=%d, failed=%d",
        best_value, best_params, n_completed, n_failed,
    )

    return PEFTSearchResult(
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
    "build_peft_search_space",
    "PEFTSearchResult",
    "run_peft_search",
]
