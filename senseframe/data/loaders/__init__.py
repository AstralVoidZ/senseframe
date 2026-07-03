"""
数据加载器子包：按 loader_type 分派数据集加载逻辑。

替代 CSIDataModule 中的 if-elif 硬编码，每种数据格式对应一个 DatasetLoader 实现。
新数据集只需注册新的 loader_type，无需修改 CSIDataModule。
"""

from .base import DatasetLoader, DatasetSplits
from .registry import register_loader, get_loader, has_loader

__all__ = [
    "DatasetLoader",
    "DatasetSplits",
    "register_loader",
    "get_loader",
    "has_loader",
]
