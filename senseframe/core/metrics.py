"""
指标注册机制：对称 core/losses.py 的 @register_loss。

通过 @register_metric 装饰器注册指标工厂，
新增指标无需修改 engine/module.py 的 METRIC_MAP。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional
import torch.nn as nn

from .validators import Validator, run_validation


_METRIC_REGISTRY: Dict[str, Callable] = {}
_METRIC_VALIDATORS: Dict[str, Optional[Validator]] = {}


def register_metric(name: str, *, overwrite: bool = True, validator: Optional[Validator] = None):
    """装饰器：注册指标工厂。

    RFC-002 阶段 G：新增 validator 参数，注册时自动验证。

    工厂签名：factory(num_classes: int) -> nn.Module | None
    返回 None 表示该指标在当前环境不可用（如缺少 torchmetrics.detection）。

    Args:
        name: 指标名称（如 "accuracy", "iou"）
        overwrite: True 时覆盖已注册的同名 metric；False 时已存在则 raise
        validator: 验证器，注册时自动验证 factory
    """
    def decorator(factory: Callable):
        if not overwrite and name in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' already registered")
        result = run_validation(validator, factory)
        if not result.passed:
            raise ValueError(
                f"Metric '{name}' 注册验证失败: {'; '.join(result.errors)}"
            )
        _METRIC_REGISTRY[name] = factory
        _METRIC_VALIDATORS[name] = validator
        return factory
    return decorator


def get_metric(name: str, num_classes: int) -> Optional[nn.Module]:
    """构造已注册的指标实例。

    Args:
        name: 指标名称
        num_classes: 类别数（某些指标不需要，工厂自行忽略）

    Returns:
        指标实例，或 None（指标不可用时）

    Raises:
        KeyError: 指标未注册
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Metric '{name}' not registered. "
            f"Available: {sorted(_METRIC_REGISTRY.keys())}. "
            f"Use @register_metric to register new metrics."
        )
    return _METRIC_REGISTRY[name](num_classes)


def has_metric(name: str) -> bool:
    """检查指标是否已注册。"""
    return name in _METRIC_REGISTRY


def list_metrics() -> list:
    """列出所有已注册的指标名。"""
    return sorted(_METRIC_REGISTRY.keys())


# ============================================================
# 内置指标注册
# ============================================================
def _register_builtin_metrics():
    """注册内置指标（延迟导入 torchmetrics，避免无 torchmetrics 时报错）。"""
    try:
        from torchmetrics import (
            Accuracy, F1Score, Precision, Recall,
            MeanSquaredError, MeanAbsoluteError,
        )
    except ImportError:
        return

    @register_metric("accuracy")
    def _accuracy(nc: int):
        return Accuracy(task="multiclass", num_classes=nc)

    @register_metric("macro_f1")
    def _macro_f1(nc: int):
        return F1Score(task="multiclass", num_classes=nc, average="macro")

    @register_metric("micro_f1")
    def _micro_f1(nc: int):
        return F1Score(task="multiclass", num_classes=nc, average="micro")

    @register_metric("weighted_f1")
    def _weighted_f1(nc: int):
        return F1Score(task="multiclass", num_classes=nc, average="weighted")

    @register_metric("macro_precision")
    def _macro_precision(nc: int):
        return Precision(task="multiclass", num_classes=nc, average="macro")

    @register_metric("macro_recall")
    def _macro_recall(nc: int):
        return Recall(task="multiclass", num_classes=nc, average="macro")

    @register_metric("mse")
    def _mse(nc: int):
        return MeanSquaredError()

    @register_metric("mae")
    def _mae(nc: int):
        return MeanAbsoluteError()

    @register_metric("rmse")
    def _rmse(nc: int):
        return MeanSquaredError(squared=False)

    @register_metric("map")
    def _map(nc: int):
        try:
            from torchmetrics.detection import MeanAveragePrecision
            return MeanAveragePrecision()
        except ImportError:
            return None


# 模块加载时自动注册
_register_builtin_metrics()
