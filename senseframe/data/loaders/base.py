"""
数据加载器基类 + DatasetSplits 数据类。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import Dataset, random_split

_logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    """数据集划分结果。

    Attributes:
        train: 训练集（监督模式）
        test: 测试集
        val: 验证集（可选，与 DatasetBundle.val 对齐）
        unsupervised: 无标签预训练集（自监督模式）
        supervised: 监督微调集（自监督模式）
    """
    train: Optional[Dataset] = None
    test: Optional[Dataset] = None
    # 根因修复：补齐 val 字段，与 scenes 层 DatasetBundle.val 对齐，
    # 消除 loaders 层与 scenes 层数据结构契约不对齐。
    val: Optional[Dataset] = None
    unsupervised: Optional[Dataset] = None
    supervised: Optional[Dataset] = None

    def validate_filling(self, learning_mode: str = "supervised") -> List[str]:
        """校验填充是否符合契约（与 DatasetBundle.validate_filling 对齐）。

        根因修复：loaders 层此前无校验方法，构造出的 splits 可能缺关键字段
        而静默通过，导致下游 stage_build 取到 None 才报错。现统一在入口校验。

        Args:
            learning_mode: "supervised" / "self_supervised"

        Returns:
            错误列表，空列表表示通过
        """
        errors = []
        if learning_mode == "self_supervised":
            required = ["unsupervised", "supervised", "test"]
        else:
            required = ["train", "test"]
        # val 始终可选
        for field_name in required:
            if getattr(self, field_name) is None:
                errors.append(
                    f"Field '{field_name}' is required for "
                    f"learning_mode='{learning_mode}' but is None"
                )
        return errors


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

    @staticmethod
    def _auto_split_val(splits: DatasetSplits, val_ratio: float,
                        seed: int = 42) -> DatasetSplits:
        """P2-3 修复：从 train 自动划分 val split。

        当数据集无原生 val 且 DatasetSpec.val_split_ratio 非 None 时调用。
        使用 torch.utils.data.random_split + 固定 seed 保证可复现。

        Args:
            splits: 原始 DatasetSplits（val=None）
            val_ratio: val 划分比例（如 0.1 = 10%）
            seed: 随机种子（从 TrainerConfig.seed 派生）

        Returns:
            新的 DatasetSplits，train 缩小，val 已填充
        """
        if splits.train is None or val_ratio <= 0 or val_ratio >= 1:
            return splits
        if splits.val is not None:
            # 已有 val，无需划分
            return splits

        n_total = len(splits.train)
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(
            splits.train, [n_train, n_val], generator=generator,
        )

        _logger.info(
            "_auto_split_val: train %d → train=%d + val=%d (ratio=%.2f, seed=%d)",
            n_total, n_train, n_val, val_ratio, seed,
        )

        return DatasetSplits(
            train=train_subset,
            val=val_subset,
            test=splits.test,
            unsupervised=splits.unsupervised,
            supervised=splits.supervised,
        )
