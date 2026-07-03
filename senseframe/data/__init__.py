"""
senseframe.data：数据层子包。

Phase R4（架构重构）：从顶层 data.py 拆分为子包。
- normalization.py：normalize / Normalize 工具函数
- legacy.py：旧版 CSIDataModule + _CSIDataset + _WidarDataset
- manifest.py：声明式 manifest 加载
- loaders/：数据加载器子包（R5：按 loader_type 分派）

向后兼容：from senseframe.data import normalize, Normalize, CSIDataModule, ...
"""

from .normalization import normalize, Normalize
from .legacy import (
    CSIDataModule,
    _CSIDataset,
    _WidarDataset,
    _resolve_data_path,
)
from .loaders import (
    DatasetLoader,
    DatasetSplits,
    register_loader,
    get_loader,
    has_loader,
)
from ..registry import NORMALIZATION_CONSTANTS

__all__ = [
    "normalize",
    "Normalize",
    "CSIDataModule",
    "NORMALIZATION_CONSTANTS",
    "DatasetLoader",
    "DatasetSplits",
    "register_loader",
    "get_loader",
    "has_loader",
]
