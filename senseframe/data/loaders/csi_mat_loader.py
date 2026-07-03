"""
CSI .mat 文件加载器：加载 NTU-Fi 风格的 .mat 文件，创建 CSIDataset。
"""

from __future__ import annotations

from torch.utils.data import ConcatDataset

from .base import DatasetLoader, DatasetSplits
from ._datasets import CSIDataset, resolve_data_path


class CSIMatLoader(DatasetLoader):
    """NTU-Fi 风格的 .mat CSI 数据加载器。

    支持 NTU-Fi-HumanID 和 NTU-Fi_HAR 数据集。
    自监督模式下，NTU-Fi_HAR 使用全部数据做无监督预训练，
    NTU-Fi-HumanID 做监督微调。
    """

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        if dataset_name == "NTU-Fi-HumanID":
            return self._load_humanid(root, learning_mode)
        elif dataset_name == "NTU-Fi_HAR":
            return self._load_ntu_har(root, learning_mode)
        else:
            # 通用 .mat 加载：假设 train_amp / test_amp 子目录
            return self._load_generic(root, dataset_name, learning_mode)

    def _load_humanid(self, root: str, learning_mode: str) -> DatasetSplits:
        train_dir = resolve_data_path(root, "NTU-Fi-HumanID", "test_amp")
        test_dir = resolve_data_path(root, "NTU-Fi-HumanID", "train_amp")
        return DatasetSplits(
            train=CSIDataset(train_dir),
            test=CSIDataset(test_dir),
        )

    def _load_ntu_har(self, root: str, learning_mode: str) -> DatasetSplits:
        train_dir = resolve_data_path(root, "NTU-Fi_HAR", "train_amp")
        test_dir = resolve_data_path(root, "NTU-Fi_HAR", "test_amp")

        if learning_mode == "self_supervised":
            unsupervised = ConcatDataset([
                CSIDataset(train_dir),
                CSIDataset(test_dir),
            ])
            humanid_dir = resolve_data_path(root, "NTU-Fi-HumanID", "test_amp")
            supervised = CSIDataset(humanid_dir)
            humanid_test_dir = resolve_data_path(root, "NTU-Fi-HumanID", "train_amp")
            test = CSIDataset(humanid_test_dir)
            return DatasetSplits(
                unsupervised=unsupervised,
                supervised=supervised,
                test=test,
            )

        return DatasetSplits(
            train=CSIDataset(train_dir),
            test=CSIDataset(test_dir),
        )

    def _load_generic(self, root: str, dataset_name: str,
                      learning_mode: str) -> DatasetSplits:
        train_dir = resolve_data_path(root, dataset_name, "train_amp")
        test_dir = resolve_data_path(root, dataset_name, "test_amp")
        return DatasetSplits(
            train=CSIDataset(train_dir),
            test=CSIDataset(test_dir),
        )
