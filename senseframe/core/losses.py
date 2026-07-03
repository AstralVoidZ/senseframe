"""
Phase 11.2 — Loss 工厂可配置。

将硬编码的 F.cross_entropy 替换为可注册的 loss 工厂。
用户可通过 @register_loss 装饰器或 register_loss() 函数注入新 loss。

内置 loss：
- cross_entropy / cross_entropy_weighted
- mse / mae / smooth_l1
- bce_with_logits / focal
- ent_loss（自监督两阶段 Phase 1 用）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .validators import Validator, run_validation


LossFactory = Callable[..., nn.Module]


# 注册项：factory + validator
_LOSS_REGISTRY: Dict[str, LossFactory] = {}
_LOSS_VALIDATORS: Dict[str, Optional[Validator]] = {}


def register_loss(name: str, *, overwrite: bool = True, validator: Optional[Validator] = None):
    """装饰器：注册 loss 工厂。

    RFC-002 阶段 G：新增 validator 参数，注册时自动验证。
    验证失败则拒绝注册，回滚。

    Args:
        name: loss 名称（如 "cross_entropy"）
        overwrite: True 时覆盖已注册的同名 loss；False 时已存在则 raise
        validator: 验证器，注册时自动验证 factory
    """
    def decorator(factory: LossFactory) -> LossFactory:
        if not overwrite and name in _LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' already registered")
        # 验证
        result = run_validation(validator, factory)
        if not result.passed:
            raise ValueError(
                f"Loss '{name}' 注册验证失败: {'; '.join(result.errors)}"
            )
        _LOSS_REGISTRY[name] = factory
        _LOSS_VALIDATORS[name] = validator
        return factory
    return decorator


def get_loss(name: str, **kwargs) -> nn.Module:
    """按名称实例化 loss 模块。"""
    if name not in _LOSS_REGISTRY:
        raise KeyError(
            f"Loss '{name}' not registered. Available: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


def has_loss(name: str) -> bool:
    return name in _LOSS_REGISTRY


def list_losses() -> list:
    return list(_LOSS_REGISTRY.keys())


# ============================================================
# 内置 loss
# ============================================================
@register_loss("cross_entropy")
def _cross_entropy(**kwargs) -> nn.Module:
    return nn.CrossEntropyLoss(**kwargs)


@register_loss("cross_entropy_weighted")
def _cross_entropy_weighted(weights=None, **kwargs) -> nn.Module:
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights, **kwargs)


@register_loss("mse")
def _mse(**kwargs) -> nn.Module:
    return nn.MSELoss(**kwargs)


@register_loss("mae")
def _mae(**kwargs) -> nn.Module:
    return nn.L1Loss(**kwargs)


@register_loss("smooth_l1")
def _smooth_l1(**kwargs) -> nn.Module:
    return nn.SmoothL1Loss(**kwargs)


@register_loss("bce_with_logits")
def _bce_with_logits(**kwargs) -> nn.Module:
    return nn.BCEWithLogitsLoss(**kwargs)


@register_loss("focal")
def _focal(alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> nn.Module:
    """Focal Loss 用于类别不平衡场景。"""

    class FocalLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            loss = self.alpha * (1 - pt) ** self.gamma * ce
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss

    return FocalLoss()


@register_loss("ent_loss")
def _ent_loss(tau: float = 1.0, eps: float = 0.15, lam1: float = 1.0, lam2: float = 1.0) -> nn.Module:
    """自监督 EntLoss（KL + EH + HE + KDE），从 core.ent_loss 导入。"""
    from .ent_loss import EntLoss
    return EntLoss(tau=tau, eps=eps, lam1=lam1, lam2=lam2)


# ============================================================
# 工具：从 TaskSpec 创建 loss
# ============================================================
@dataclass
class LossConfig:
    """loss 配置：从字符串名称 + 可选 kwargs 解析。"""

    name: str
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

    def build(self) -> nn.Module:
        return get_loss(self.name, **self.kwargs)


def loss_from_spec(loss_name: str, loss_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
    """便捷函数：从 (name, kwargs) 构造 loss。"""
    return LossConfig(name=loss_name, kwargs=loss_kwargs or {}).build()
