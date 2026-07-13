"""
CSV 文件夹加载器：加载 Widar 风格的 .csv 文件，创建 _WidarDataset。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import DatasetLoader, DatasetSplits
from ._datasets import WidarDataset, resolve_data_path

_logger = logging.getLogger(__name__)


class CSVFolderLoader(DatasetLoader):
    """Widar 风格的 CSV 文件夹加载器。"""

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        # 候选目录名必须从注册表 DatasetSpec.dir_names 派生（单一数据源），
        # 禁止硬编码 "Widardata" 等目录名回退。未注册则 raise，让调用方修复注册。
        from ...registry import get_dataset_spec
        try:
            spec = get_dataset_spec(dataset_name)
        except KeyError:
            raise KeyError(
                f"CSVFolderLoader: 数据集 '{dataset_name}' 未注册，无法派生 dir_names。"
                f"请先通过场景注册（register_dataset）声明该数据集的 dir_names。"
            )
        if not spec.dir_names:
            raise ValueError(
                f"CSVFolderLoader: 数据集 '{dataset_name}' 的 DatasetSpec.dir_names 为空。"
                f"请在注册时声明 dir_names（如 ('Widardata', 'Widar')）。"
            )
        candidate_dirs = list(spec.dir_names)

        # 按候选顺序探测 train/test 目录，使用第一个存在的候选
        train_dir = None
        test_dir = None
        for dir_name in candidate_dirs:
            t = resolve_data_path(root, dir_name, "train")
            te = resolve_data_path(root, dir_name, "test")
            if Path(t).exists():
                train_dir, test_dir = t, te
                break

        # 全部候选均不存在时，使用最后一个候选路径（让下游抛出清晰错误）
        if train_dir is None:
            train_dir = resolve_data_path(root, candidate_dirs[-1], "train")
            test_dir = resolve_data_path(root, candidate_dirs[-1], "test")

        train_ds = WidarDataset(train_dir, layout=spec.layout)
        test_ds = WidarDataset(test_dir, layout=spec.layout)
        # 修复（5.9）：load_splits 返回前 log 样本数/类别分布
        _logger.info(
            "CSVFolderLoader.load_splits: dataset=%s, learning_mode=%s, "
            "train_dir=%s (samples=%d, classes=%d), "
            "test_dir=%s (samples=%d, classes=%d)",
            dataset_name, learning_mode,
            train_dir, len(train_ds.data_list), len(train_ds.category),
            test_dir, len(test_ds.data_list), len(test_ds.category),
        )
        return DatasetSplits(
            train=train_ds,
            test=test_ds,
        )
