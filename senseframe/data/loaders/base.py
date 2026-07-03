"""
数据加载器基类 + DatasetSplits 数据类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class DatasetSplits:
    """数据集划分结果。

    Attributes:
        train: 训练集（监督模式）
        test: 测试集
        unsupervised: 无标签预训练集（自监督模式）
        supervised: 监督微调集（自监督模式）
    """
    train: Optional[Dataset] = None
    test: Optional[Dataset] = None
    unsupervised: Optional[Dataset] = None
    supervised: Optional[Dataset] = None


class DatasetLoader(ABC):
    """数据加载器抽象基类。

    每种数据格式（tensor / csi_mat / csv_folder / ...）对应一个实现。
    通过 loader_type 注册到全局 loader 注册表，CSIDataModule 根据
    DatasetSpec.loader_type 分派到对应实现。
    """

    @abstractmethod
    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        """加载数据集划分。

        Args:
            root: 数据根目录
            dataset_name: 数据集注册名（如 "NTU-Fi_HAR"）
            learning_mode: "supervised" / "self_supervised"

        Returns:
            DatasetSplits 包含各划分的 Dataset 对象
        """
        ...
