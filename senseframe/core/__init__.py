"""
Phase 11 — 任务类型与配置正交化（RFC Phase A：开放策略空间）。

将分类专属的硬编码概念抽象为通用 TaskType、LossFactory、FeatureSpec、
SceneParams，让分类之外的检测/分割/回归场景能直接接入 senseframe。

RFC Phase A：TaskType 从封闭枚举改为开放注册表，
Agent 可运行时注册新任务类型。
"""

from .task import (
    TaskType,
    TaskSpec,
    DEFAULT_LOSS,
    DEFAULT_METRICS,
    register_task_type,
    list_task_types,
    has_task_type,
    get_task_type_default_loss,
    get_task_type_default_metrics,
)
from .losses import (
    register_loss,
    get_loss,
    has_loss,
    list_losses,
    list_supervised_losses,
    SELF_SUPERVISED_LOSSES,
    LossConfig,
    loss_from_spec,
)
from .metrics import (
    register_metric,
    get_metric,
    has_metric,
    list_metrics,
)
from .features import FeatureSpec
from .params import SceneParams
from .profiler import DataProfiler, DataProfile
from .validators import (
    ValidationResult,
    Validator,
    shape_validator,
    numerical_stability_validator,
    signature_validator,
    performance_validator,
    transform_pipeline_validator,
    compose,
    run_validation,
)
from .foundation_model import (
    SensingFoundationModel,
    PretrainConfig,
    PEFTConfig,
)

__all__ = [
    "TaskType",
    "TaskSpec",
    "DEFAULT_LOSS",
    "DEFAULT_METRICS",
    "register_task_type",
    "list_task_types",
    "has_task_type",
    "get_task_type_default_loss",
    "get_task_type_default_metrics",
    "register_loss",
    "get_loss",
    "has_loss",
    "list_losses",
    "list_supervised_losses",
    "SELF_SUPERVISED_LOSSES",
    "LossConfig",
    "loss_from_spec",
    "register_metric",
    "get_metric",
    "has_metric",
    "list_metrics",
    "FeatureSpec",
    "SceneParams",
    "DataProfiler",
    "DataProfile",
    "ValidationResult",
    "Validator",
    "shape_validator",
    "numerical_stability_validator",
    "signature_validator",
    "performance_validator",
    "transform_pipeline_validator",
    "compose",
    "run_validation",
    # P3 阶段 8: 感知基础模型抽象
    "SensingFoundationModel",
    "PretrainConfig",
    "PEFTConfig",
]
