"""
内置数据加载器自注册：将 WiFi CSI 场景的 3 种加载器注册到全局注册表。

s3：新增 HDF5 / Parquet loader，扩展数据格式支持。
"""

import logging

from .base import DatasetLoader
from .registry import register_loader
from .tensor_loader import TensorLoader
from .csi_mat_loader import CSIMatLoader
from .csv_folder_loader import CSVFolderLoader
from .hdf5_loader import HDF5Loader
from .parquet_loader import ParquetLoader

_logger = logging.getLogger(__name__)

_builtin_registered = False


def register_builtin_loaders() -> None:
    """注册内置数据加载器（幂等）。

    修复（问题 4.10）：逐个 try/except ImportError。
    缺失依赖的 loader warning 后跳过，其他 loader 正常注册。
    异常分级：
    - ImportError（预期可恢复，如 h5py/pyarrow 缺失）→ warning + 跳过
    - TypeError/AttributeError/NameError 等代码 bug → error + exc_info
    """
    global _builtin_registered
    if _builtin_registered:
        return

    # (loader_type, loader_cls) 列表：逐个 try/except 隔离
    _loader_specs = [
        ("tensor", TensorLoader),
        ("csi_mat", CSIMatLoader),
        ("csv_folder", CSVFolderLoader),
        # s3：HDF5 / Parquet loader（依赖 h5py / pandas+pyarrow）
        ("hdf5", HDF5Loader),
        ("parquet", ParquetLoader),
    ]

    for loader_type, loader_cls in _loader_specs:
        try:
            register_loader(loader_type, loader_cls())
        except ImportError as e:
            # 预期可恢复异常：外部依赖缺失，降级跳过该 loader
            _logger.warning(
                f"register_builtin_loaders: skip loader '{loader_type}' "
                f"(missing dependency): {e}"
            )
        except Exception as e:
            # 代码 bug：留 exc_info 痕迹，便于排查
            _logger.error(
                f"register_builtin_loaders: failed to register loader "
                f"'{loader_type}': {e}",
                exc_info=True,
            )

    _builtin_registered = True
