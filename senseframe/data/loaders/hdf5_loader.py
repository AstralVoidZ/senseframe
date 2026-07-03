"""HDF5 数据加载器（s3：扩展数据格式支持）。

从 .h5/.hdf5 文件加载训练/测试划分。约定文件结构：

    /train/x, /train/y   训练集特征与标签
    /test/x, /test/y     测试集特征与标签

其中 x/y 为 HDF5 内部 dataset 的键名，可通过构造参数自定义。
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import DatasetLoader, DatasetSplits


def _find_hdf5_file(root: str, dataset_name: str) -> Path:
    """在 root 下查找 dataset_name 对应的 .h5/.hdf5 文件。"""
    candidates = []
    for ext in (".h5", ".hdf5"):
        candidates.append(Path(root, dataset_name + ext))
        candidates.append(Path(root, dataset_name, dataset_name + ext))
    for c in candidates:
        if c.exists():
            return c
    # 兜底：在 root/dataset_name 目录下扫第一个 .h5/.hdf5
    d = Path(root, dataset_name)
    if d.is_dir():
        for ext in (".h5", ".hdf5"):
            files = sorted(d.glob(f"*{ext}"))
            if files:
                return files[0]
    raise FileNotFoundError(
        f"No .h5/.hdf5 file found for dataset '{dataset_name}' under '{root}'. "
        f"Searched: {[str(c) for c in candidates]}"
    )


class HDF5Loader(DatasetLoader):
    """从 HDF5 文件加载样本（s3：扩展数据格式支持）。

    支持从 .h5/.hdf5 文件加载训练/测试划分。HDF5 文件需包含 train/test
    顶层 group，每个 group 下有特征 dataset 与标签 dataset。

    Args:
        feature_key: HDF5 内部特征 dataset 的键名（默认 "x"）
        label_key: HDF5 内部标签 dataset 的键名（默认 "y"）
    """

    def __init__(self, feature_key: str = "x", label_key: str = "y"):
        self.feature_key = feature_key
        self.label_key = label_key

    @property
    def supported_extensions(self):
        """此 loader 处理的文件扩展名。"""
        return (".h5", ".hdf5")

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                f"h5py not installed: {e}. Install with: pip install h5py"
            ) from e

        path = _find_hdf5_file(root, dataset_name)

        with h5py.File(str(path), "r") as f:
            train_x, train_y = self._read_split(f, "train")
            test_x, test_y = self._read_split(f, "test")

        train_ds = TensorDataset(
            torch.as_tensor(np.asarray(train_x), dtype=torch.float32),
            torch.as_tensor(np.asarray(train_y), dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.as_tensor(np.asarray(test_x), dtype=torch.float32),
            torch.as_tensor(np.asarray(test_y), dtype=torch.long),
        )

        return DatasetSplits(train=train_ds, test=test_ds)

    def _read_split(self, f, split: str) -> Tuple[np.ndarray, np.ndarray]:
        if split not in f:
            available_keys = list(f.keys())
            raise KeyError(
                f"Group '{split}' not found in HDF5 file. "
                f"Available keys: {available_keys}"
            )
        g = f[split]
        if self.feature_key not in g:
            raise KeyError(
                f"Dataset '{self.feature_key}' not found in group '{split}'. "
                f"Available: {list(g.keys())}"
            )
        if self.label_key not in g:
            raise KeyError(
                f"Dataset '{self.label_key}' not found in group '{split}'. "
                f"Available: {list(g.keys())}"
            )
        return g[self.feature_key][:], g[self.label_key][:]


__all__ = ["HDF5Loader"]
