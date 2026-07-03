"""
CSV 文件夹加载器：加载 Widar 风格的 .csv 文件，创建 _WidarDataset。
"""

from __future__ import annotations

from .base import DatasetLoader, DatasetSplits
from ._datasets import WidarDataset, resolve_data_path


class CSVFolderLoader(DatasetLoader):
    """Widar 风格的 CSV 文件夹加载器。"""

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        # Widar 数据集目录名可能是 "Widardata" 或 "Widar"
        train_dir = resolve_data_path(root, "Widardata", "train")
        test_dir = resolve_data_path(root, "Widardata", "test")

        # 回退：尝试 dataset_name 本身
        from pathlib import Path
        if not Path(train_dir).exists():
            train_dir = resolve_data_path(root, dataset_name, "train")
            test_dir = resolve_data_path(root, dataset_name, "test")

        return DatasetSplits(
            train=WidarDataset(train_dir),
            test=WidarDataset(test_dir),
        )
