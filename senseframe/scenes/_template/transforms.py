"""{SCENE_NAME} 场景的变换原语（自动生成模板）。"""
import numpy as np
from typing import Callable, Dict, List

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def get_transform(name: str) -> Callable:
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name]


def list_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


def compose_transforms(names: List[str], **kwargs) -> Callable:
    """组合多个变换原语。"""
    fns = [get_transform(n) for n in names]
    def _composed(x, y):
        for fn in fns:
            x, y = fn(x, y)
        return x, y
    return _composed
