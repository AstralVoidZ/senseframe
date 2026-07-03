"""
Phase 11.1 — TaskType 抽象（RFC Phase A：开放策略空间）。

将"分类专属"概念抽象为通用 TaskType，使框架可承载分类之外的
回归/检测/分割等任务。

RFC Phase A 重构要点：
- TaskType 从封闭枚举改为开放注册表：Agent 可运行时注册新任务类型
- TaskSpec.task_type 字段类型从 TaskType 改为 str（兼容枚举值与自定义字符串）
- DEFAULT_LOSS / DEFAULT_METRICS 改为从注册表派生
- TaskSpec.__post_init__ 不再强制枚举转换（三重封锁消除）
- 向后兼容：TaskType 枚举保留作为内置 4 个任务类型的别名

设计原则：
- 预设策略是快车道，不是唯一道路（RFC 原则 3）
- 内置 4 个任务类型仍以原名称注册，YAML 配置无需修改
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .validators import Validator, run_validation


# ============================================================
# RFC Phase A：任务类型注册表（开放策略空间）
# ============================================================
# 任务类型元数据：default_loss + default_metrics + 描述 + 验证器
_TASK_TYPE_REGISTRY: Dict[str, "_TaskTypeMeta"] = {}


@dataclass
class _TaskTypeMeta:
    """任务类型元数据（内部使用）。"""
    name: str
    default_loss: str
    default_metrics: List[str]
    description: str = ""
    validator: Optional[Validator] = None


def register_task_type(
    name: str,
    default_loss: str,
    default_metrics: List[str],
    *,
    description: str = "",
    overwrite: bool = True,
    validator: Optional[Validator] = None,
) -> None:
    """注册任务类型（开放策略空间）。

    RFC-002 阶段 G：新增 validator 参数，注册时自动验证。

    Agent 可运行时注册新任务类型，不被封闭枚举拦截。

    Args:
        name: 任务类型名称（如 "ranking"、"generation"）
        default_loss: 默认 loss 名称（需已通过 @register_loss 注册）
        default_metrics: 默认 metrics 列表（需已通过 @register_metric 注册）
        description: 任务类型描述
        overwrite: True 时覆盖已注册的同名任务类型；False 时已存在则 raise
        validator: 验证器，注册时自动验证

    Examples:
        >>> register_task_type("ranking", "listnet", ["ndcg@5", "map"])
        >>> TaskSpec(task_type="ranking")
    """
    if not overwrite and name in _TASK_TYPE_REGISTRY:
        raise ValueError(f"Task type '{name}' already registered")
    # 验证
    result = run_validation(validator, {"default_loss": default_loss, "default_metrics": default_metrics})
    if not result.passed:
        raise ValueError(
            f"Task type '{name}' 注册验证失败: {'; '.join(result.errors)}"
        )
    _TASK_TYPE_REGISTRY[name] = _TaskTypeMeta(
        name=name,
        default_loss=default_loss,
        default_metrics=list(default_metrics),
        description=description,
        validator=validator,
    )


def list_task_types() -> List[str]:
    """列出所有已注册的任务类型名称。"""
    return list(_TASK_TYPE_REGISTRY.keys())


def has_task_type(name: str) -> bool:
    """检查任务类型是否已注册。"""
    return name in _TASK_TYPE_REGISTRY


def get_task_type_default_loss(name: str) -> str:
    """获取任务类型的默认 loss。未注册则回退 cross_entropy。"""
    meta = _TASK_TYPE_REGISTRY.get(name)
    return meta.default_loss if meta else "cross_entropy"


def get_task_type_default_metrics(name: str) -> List[str]:
    """获取任务类型的默认 metrics。未注册则回退 ["accuracy"]。"""
    meta = _TASK_TYPE_REGISTRY.get(name)
    return list(meta.default_metrics) if meta else ["accuracy"]


# ============================================================
# 内置任务类型（注册为默认，向后兼容 TaskType 枚举）
# ============================================================
def _register_builtin_task_types() -> None:
    """注册内置 4 个任务类型。"""
    register_task_type("classification", "cross_entropy", ["accuracy", "macro_f1"],
                       description="分类任务")
    register_task_type("regression", "mse", ["mse", "mae"],
                       description="回归任务")
    register_task_type("detection", "bce_with_logits", ["map"],
                       description="目标检测任务")
    register_task_type("segmentation", "cross_entropy", ["iou", "dice"],
                       description="分割任务")


_register_builtin_task_types()


# ============================================================
# TaskType 枚举（向后兼容别名，不再用于校验）
# ============================================================
class TaskType(str, Enum):
    """内置任务类型枚举（向后兼容别名）。

    RFC Phase A：此枚举仅作为内置 4 个任务类型的便捷引用，
    不再用于校验。Agent 可注册不在枚举中的新任务类型。

    新代码建议直接使用字符串字面量（如 "classification"），
    或通过 register_task_type() 注册新类型。
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


# 向后兼容：DEFAULT_LOSS / DEFAULT_METRICS 改为从注册表派生
# 但保持原 API 形态（Dict[TaskType, str] / Dict[TaskType, List[str]]）
def _build_default_loss_map() -> Dict[TaskType, str]:
    """从注册表派生 DEFAULT_LOSS（向后兼容）。"""
    return {
        TaskType.CLASSIFICATION: get_task_type_default_loss("classification"),
        TaskType.REGRESSION: get_task_type_default_loss("regression"),
        TaskType.DETECTION: get_task_type_default_loss("detection"),
        TaskType.SEGMENTATION: get_task_type_default_loss("segmentation"),
    }


def _build_default_metrics_map() -> Dict[TaskType, List[str]]:
    """从注册表派生 DEFAULT_METRICS（向后兼容）。"""
    return {
        TaskType.CLASSIFICATION: get_task_type_default_metrics("classification"),
        TaskType.REGRESSION: get_task_type_default_metrics("regression"),
        TaskType.DETECTION: get_task_type_default_metrics("detection"),
        TaskType.SEGMENTATION: get_task_type_default_metrics("segmentation"),
    }


# 模块级常量（向后兼容，但值从注册表派生）
DEFAULT_LOSS: Dict[TaskType, str] = _build_default_loss_map()
DEFAULT_METRICS: Dict[TaskType, List[str]] = _build_default_metrics_map()


@dataclass
class TaskSpec:
    """任务规格：决定 loss / metrics / 输出激活。

    RFC Phase A：task_type 字段类型从 TaskType 改为 str，
    兼容内置枚举值与 Agent 注册的自定义任务类型。

    字段：
    - task_type: 任务类型名称（字符串，如 "classification" 或 Agent 注册的自定义类型）
    - num_classes: 类别数（CLASSIFICATION / DETECTION 必填，REGRESSION 不适用）
    - loss: 显式 loss 名称（None 则按 task_type 取默认）
    - metrics: 显式 metrics 列表（None 则按 task_type 取默认）
    - output_activation: 输出激活函数（None / softmax / sigmoid / tanh / relu）
    - loss_kwargs: loss 构造参数（如 FocalLoss 的 alpha/gamma）
    - extra: 任务特定扩展参数（如 bbox 格式、mask 类型等）
    """

    task_type: str = "classification"
    num_classes: Optional[int] = None
    loss: Optional[str] = None
    metrics: Optional[List[str]] = None
    output_activation: Optional[str] = None
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # RFC Phase A：不再强制枚举转换
        # 兼容传入 TaskType 枚举实例（取 .value）
        if isinstance(self.task_type, TaskType):
            self.task_type = self.task_type.value
        elif isinstance(self.task_type, str):
            # 保持字符串原样（可能是内置类型或 Agent 注册的自定义类型）
            pass

    @property
    def effective_loss(self) -> str:
        # RFC Phase A：从注册表派生，未注册回退 cross_entropy
        return self.loss or get_task_type_default_loss(self.task_type)

    @property
    def effective_metrics(self) -> List[str]:
        # RFC Phase A：从注册表派生，未注册回退 ["accuracy"]
        return self.metrics if self.metrics is not None else get_task_type_default_metrics(self.task_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "loss": self.loss,
            "metrics": self.metrics,
            "output_activation": self.output_activation,
            "loss_kwargs": self.loss_kwargs,
            "extra": self.extra,
        }

    def validate(self) -> None:
        """校验 TaskSpec 字段合法性。

        RFC Phase A：不再强制 task_type 在枚举内，
        仅校验 loss（若显式指定）已注册。
        """
        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError(f"task_spec.num_classes 必须 >= 1，实际: {self.num_classes}")
        if self.loss is not None:
            from .losses import has_loss
            if not has_loss(self.loss):
                raise ValueError(
                    f"task_spec.loss '{self.loss}' 未注册。"
                    f"请使用 @register_loss 装饰器先注册。"
                )
        if self.output_activation is not None and self.output_activation != "none":
            valid_acts = {"none", "softmax", "sigmoid", "tanh", "relu"}
            if self.output_activation not in valid_acts:
                raise ValueError(
                    f"task_spec.output_activation '{self.output_activation}' 不支持，"
                    f"可选: {sorted(valid_acts)}"
                )

    @classmethod
    def classification(cls, num_classes: int, **kwargs) -> "TaskSpec":
        """便捷构造：分类任务。"""
        return cls(task_type="classification", num_classes=num_classes, **kwargs)

    @classmethod
    def regression(cls, **kwargs) -> "TaskSpec":
        """便捷构造：回归任务。"""
        return cls(task_type="regression", num_classes=None, **kwargs)

    @classmethod
    def detection(cls, num_classes: int, **kwargs) -> "TaskSpec":
        """便捷构造：检测任务。"""
        return cls(task_type="detection", num_classes=num_classes, **kwargs)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TaskSpec":
        if d is None:
            return cls()
        # RFC Phase A：不再强制枚举转换，task_type 保持字符串
        return cls(**d)
