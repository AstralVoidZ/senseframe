"""Parquet 数据加载器（s3：扩展数据格式支持）。

从 .parquet/.pq 文件加载表格数据，适合结构化/表格数据的存储和加载。
约定目录结构：

    <root>/<dataset_name>/train.parquet
    <root>/<dataset_name>/test.parquet

每个 parquet 文件包含特征列与一个标签列（默认列名 "label"）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import DatasetLoader, DatasetSplits


def _find_parquet_file(root: str, dataset_name: str, split: str) -> Path:
    """在 root 下查找 dataset_name 对应 split 的 .parquet/.pq 文件。"""
    candidates = []
    for ext in (".parquet", ".pq"):
        candidates.append(Path(root, dataset_name, f"{split}{ext}"))
        candidates.append(Path(root, dataset_name, dataset_name, f"{split}{ext}"))
    for c in candidates:
        if c.exists():
            return c
    # 兜底：在 root/dataset_name 目录下扫第一个匹配 split 前缀的文件
    d = Path(root, dataset_name)
    if d.is_dir():
        for ext in (".parquet", ".pq"):
            files = sorted(d.glob(f"{split}*{ext}"))
            if files:
                return files[0]
    raise FileNotFoundError(
        f"No {split}.parquet/.pq file found for dataset '{dataset_name}' under '{root}'. "
        f"Searched: {[str(c) for c in candidates]}"
    )


class ParquetLoader(DatasetLoader):
    """从 Parquet 文件加载表格数据（s3：扩展数据格式支持）。

    适合结构化/表格数据的存储和加载。每个 parquet 文件包含特征列与一个
    标签列。

    Args:
        label_col: 标签列名（默认 "label"）
        feature_cols: 特征列名列表；None 时取除标签列外的所有列
    """

    def __init__(self, label_col: str = "label",
                 feature_cols: Optional[List[str]] = None):
        self.label_col = label_col
        self.feature_cols = feature_cols

    @property
    def supported_extensions(self):
        """此 loader 处理的文件扩展名。"""
        return (".parquet", ".pq")

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                f"pandas not installed: {e}. Install with: pip install pandas pyarrow"
            ) from e

        train_path = _find_parquet_file(root, dataset_name, "train")
        test_path = _find_parquet_file(root, dataset_name, "test")

        train_ds = self._build_dataset(pd.read_parquet(str(train_path)))
        test_ds = self._build_dataset(pd.read_parquet(str(test_path)))

        return DatasetSplits(train=train_ds, test=test_ds)

    def _build_dataset(self, df) -> TensorDataset:
        if self.label_col not in df.columns:
            raise KeyError(
                f"Label column '{self.label_col}' not found in parquet file. "
                f"Available columns: {list(df.columns)}"
            )
        y = df[self.label_col].to_numpy()

        if self.feature_cols is None:
            feature_cols = [c for c in df.columns if c != self.label_col]
        else:
            feature_cols = self.feature_cols
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Feature columns {missing} not found in parquet file. "
                    f"Available columns: {list(df.columns)}"
                )
        x = df[feature_cols].to_numpy(dtype=np.float32)

        return TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )


__all__ = ["ParquetLoader"]
