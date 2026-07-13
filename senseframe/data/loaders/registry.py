"""
数据加载器注册表：loader_type → DatasetLoader 实例。
"""

from __future__ import annotations

import logging
from typing import Dict

from .base import DatasetLoader

_logger = logging.getLogger(__name__)

_LOADERS: Dict[str, DatasetLoader] = {}


def register_loader(
    loader_type: str, loader: DatasetLoader, overwrite: bool = False
) -> None:
    """注册数据加载器。

    Args:
        loader_type: 加载器类型标识（如 "tensor", "csi_mat", "csv_folder"）
        loader: DatasetLoader 实例
        overwrite: 是否覆盖已注册的同名加载器（默认 False，重复注册 warning 并跳过）
    """
    if loader_type in _LOADERS and not overwrite:
        _logger.warning(
            "Loader '%s' already registered, skipping (use overwrite=True to replace).",
            loader_type,
        )
        return
    _LOADERS[loader_type] = loader


def get_loader(loader_type: str) -> DatasetLoader:
    """获取数据加载器。未注册则 raise KeyError。"""
    if loader_type not in _LOADERS:
        raise KeyError(
            f"No loader registered for type '{loader_type}'. "
            f"Available: {list(_LOADERS.keys())}"
        )
    return _LOADERS[loader_type]


def has_loader(loader_type: str) -> bool:
    return loader_type in _LOADERS


def _reset_for_test() -> None:
    _LOADERS.clear()
