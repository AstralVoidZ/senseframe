"""
内置数据加载器自注册：将 WiFi CSI 场景的 3 种加载器注册到全局注册表。

s3：新增 HDF5 / Parquet loader，扩展数据格式支持。
"""

from .base import DatasetLoader
from .registry import register_loader
from .tensor_loader import TensorLoader
from .csi_mat_loader import CSIMatLoader
from .csv_folder_loader import CSVFolderLoader
from .hdf5_loader import HDF5Loader
from .parquet_loader import ParquetLoader

_builtin_registered = False


def register_builtin_loaders() -> None:
    """注册内置数据加载器（幂等）。"""
    global _builtin_registered
    if _builtin_registered:
        return
    register_loader("tensor", TensorLoader())
    register_loader("csi_mat", CSIMatLoader())
    register_loader("csv_folder", CSVFolderLoader())
    # s3：注册 HDF5 和 Parquet loader
    register_loader("hdf5", HDF5Loader())
    register_loader("parquet", ParquetLoader())
    _builtin_registered = True
