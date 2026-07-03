"""
策略兼容性矩阵：声明 task_type / loss / metric / model 的有效组合。

设计理念（RFC-002 原则 5）：
- 探索状态可见，Agent 看见完整搜索空间
- 兼容性矩阵避免 Agent 探索无效组合
- 声明式矩阵，Agent 可查询

使用方式：
    from senseframe.core.compatibility import is_compatible, get_compatible_losses

    # 查询 task_type 与 loss 是否兼容
    is_compatible(task_type="classification", loss="cross_entropy")  # True
    is_compatible(task_type="regression", loss="cross_entropy")      # False

    # 获取兼容的 loss 列表
    get_compatible_losses("classification")  # ["cross_entropy", "focal", ...]
"""

from __future__ import annotations

from typing import Dict, List, Set


# ============================================================
# 兼容性矩阵
# ============================================================
# task_type → 兼容的 loss 列表
_TASK_LOSS_COMPAT: Dict[str, Set[str]] = {
    "classification": {"cross_entropy", "cross_entropy_weighted", "focal", "bce_with_logits"},
    "regression": {"mse", "mae", "smooth_l1"},
    "detection": {"cross_entropy", "focal"},
    "segmentation": {"cross_entropy", "focal"},
    "self_supervised": {"ent_loss", "mse"},
}

# task_type → 兼容的 metric 列表
_TASK_METRIC_COMPAT: Dict[str, Set[str]] = {
    "classification": {"accuracy", "macro_f1", "micro_f1", "weighted_f1", "macro_precision", "macro_recall"},
    "regression": {"mse", "mae", "rmse"},
    "detection": {"map"},
    "segmentation": {"accuracy", "macro_f1"},
    "self_supervised": set(),
}

# task_type → 兼容的 output_activation
_TASK_ACTIVATION_COMPAT: Dict[str, Set[str]] = {
    "classification": {"softmax", "none"},
    "regression": {"none", "relu", "tanh"},
    "detection": {"sigmoid", "softmax"},
    "segmentation": {"softmax"},
    "self_supervised": {"none"},
}


def is_compatible(
    task_type: str = None,
    loss: str = None,
    metric: str = None,
    output_activation: str = None,
) -> bool:
    """查询策略组合是否兼容。

    任意参数为 None 时跳过该项检查。
    未知 task_type 默认兼容（不限制 Agent 探索新任务类型）。

    Args:
        task_type: 任务类型
        loss: loss 名称
        metric: metric 名称
        output_activation: 输出激活函数

    Returns:
        True 兼容，False 不兼容
    """
    if task_type is not None and loss is not None:
        compatible = _TASK_LOSS_COMPAT.get(task_type)
        if compatible is not None and loss not in compatible:
            return False

    if task_type is not None and metric is not None:
        compatible = _TASK_METRIC_COMPAT.get(task_type)
        if compatible is not None and metric not in compatible:
            return False

    if task_type is not None and output_activation is not None:
        compatible = _TASK_ACTIVATION_COMPAT.get(task_type)
        if compatible is not None and output_activation not in compatible:
            return False

    return True


def get_compatible_losses(task_type: str) -> List[str]:
    """获取与 task_type 兼容的 loss 列表。"""
    return sorted(_TASK_LOSS_COMPAT.get(task_type, set()))


def get_compatible_metrics(task_type: str) -> List[str]:
    """获取与 task_type 兼容的 metric 列表。"""
    return sorted(_TASK_METRIC_COMPAT.get(task_type, set()))


def get_compatible_activations(task_type: str) -> List[str]:
    """获取与 task_type 兼容的 output_activation 列表。"""
    return sorted(_TASK_ACTIVATION_COMPAT.get(task_type, set()))


def register_compatibility(
    task_type: str,
    losses: List[str] = None,
    metrics: List[str] = None,
    activations: List[str] = None,
) -> None:
    """注册自定义兼容性规则。

    Agent 注册新 task_type 时可声明其兼容的 loss/metric/activation。

    Args:
        task_type: 任务类型
        losses: 兼容的 loss 列表
        metrics: 兼容的 metric 列表
        activations: 兼容的 output_activation 列表
    """
    if losses is not None:
        existing = _TASK_LOSS_COMPAT.get(task_type, set())
        existing.update(losses)
        _TASK_LOSS_COMPAT[task_type] = existing
    if metrics is not None:
        existing = _TASK_METRIC_COMPAT.get(task_type, set())
        existing.update(metrics)
        _TASK_METRIC_COMPAT[task_type] = existing
    if activations is not None:
        existing = _TASK_ACTIVATION_COMPAT.get(task_type, set())
        existing.update(activations)
        _TASK_ACTIVATION_COMPAT[task_type] = existing


__all__ = [
    "is_compatible",
    "get_compatible_losses",
    "get_compatible_metrics",
    "get_compatible_activations",
    "register_compatibility",
]
