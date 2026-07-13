"""{SCENE_NAME} 场景的变换原语（自动生成模板）。"""
import numpy as np
from typing import Callable, Dict, List, Optional

from ...common.transforms import ComposedTransform

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def get_transform(name: str) -> Callable:
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name]


def list_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


def compose_transforms(names: List[str], seed: Optional[int] = None, **kwargs) -> Callable:
    """组合多个变换原语。

    Args:
        names: 原语名列表
        seed: 可选的随机种子（P3 上策，详见 wifi_csi.transforms.compose_transforms）
        **kwargs: 传递给每个原语的参数（按原语名分组）

    Returns:
        ComposedTransform 实例（callable，可 pickle 供 DataLoader multi-worker 使用）
    """
    transforms = [(name, get_transform(name)) for name in names]
    return ComposedTransform(transforms, kwargs, seed=seed)
