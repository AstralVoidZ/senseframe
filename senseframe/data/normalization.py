"""
归一化工具函数。

Phase R4（架构重构）：从 data.py 拆出，仅包含 normalize / Normalize 工具。
"""

from typing import Any

import numpy as np

from ..registry import get_normalization


def normalize(x: np.ndarray, dataset_name: str) -> np.ndarray:
    """按数据集注册的策略归一化。

    通过 registry.get_normalization() 取得策略（默认 IdentityStrategy）。
    """
    return get_normalization(dataset_name).apply(x)


class Normalize:
    """callable 归一化变换：与容器 get_transforms() 返回值一致。

    用法::

        transform = Normalize("NTU-Fi_HAR")
        x_norm = transform(x, y)  # 返回 (x_norm, y)
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._strategy = get_normalization(dataset_name)

    def __call__(self, x: np.ndarray, y: Any = None):
        return self._strategy.apply(x), y

    def __repr__(self) -> str:
        return f"Normalize(dataset={self.dataset_name!r}, strategy={self._strategy!r})"
