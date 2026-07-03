"""
旧版数据加载实现（CSIDataModule）。

R-fix：Dataset 类和路径解析已迁移到 data/loaders/_datasets.py。
_load_ut_har 已迁移到 data/loaders/tensor_loader.py。
本模块保留 CSIDataModule + 向后兼容 re-export。
"""

import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from ..registry import DATASET_INFO, NORMALIZATION_CONSTANTS, get_dataset_spec

# R-fix：向后兼容 re-export（新代码应直接从 loaders._datasets 导入）
from .loaders._datasets import resolve_data_path as _resolve_data_path
from .loaders._datasets import CSIDataset as _CSIDataset
from .loaders._datasets import WidarDataset as _WidarDataset


# ============================================================
# LightningDataModule
# ============================================================
class CSIDataModule(pl.LightningDataModule):
    """WiFi CSI 数据模块，通过 loader 注册表分派加载。"""

    def __init__(self, dataset_name: str, root: str,
                 batch_size: int = 64, num_workers: int = 0,
                 learning_mode: str = "supervised",
                 pin_memory: bool = False,
                 persistent_workers: bool = False):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_mode = learning_mode
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        info = DATASET_INFO.get(dataset_name)
        if info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")
        self.num_classes = info["num_classes"]

        self.train_dataset = None
        self.test_dataset = None
        self.unsupervised_dataset = None
        self.supervised_dataset = None

    def setup(self, stage: Optional[str] = None):
        from .loaders import get_loader, has_loader
        from .loaders._builtin import register_builtin_loaders

        register_builtin_loaders()

        spec = get_dataset_spec(self.dataset_name) if self.dataset_name in DATASET_INFO else None
        loader_type = spec.loader_type if spec else None

        if loader_type and has_loader(loader_type):
            splits = get_loader(loader_type).load_splits(
                self.root, self.dataset_name, self.learning_mode,
            )
            self.train_dataset = splits.train
            self.test_dataset = splits.test
            self.unsupervised_dataset = splits.unsupervised
            self.supervised_dataset = splits.supervised
        else:
            raise ValueError(
                f"No loader registered for dataset '{self.dataset_name}' "
                f"(loader_type='{loader_type}'). "
                f"请通过 register_loader 注册加载器。"
            )

        if stage in (None, "fit", "validate", "test"):
            self._validate_splits()

    def _validate_splits(self):
        """校验各数据集划分的完整性。"""
        splits = {"train": self.train_dataset, "test": self.test_dataset}
        if self.supervised_dataset is not None:
            splits["supervised"] = self.supervised_dataset
        for name, ds in splits.items():
            if ds is None:
                continue
            if len(ds) == 0:
                raise ValueError(f"[{self.dataset_name}] {name} split is empty")
            x0, y0 = ds[0]
            x0 = torch.as_tensor(x0)
            if torch.isnan(x0).any():
                raise ValueError(f"[{self.dataset_name}] {name} first sample contains NaN")
            if not torch.isfinite(x0).all():
                raise ValueError(f"[{self.dataset_name}] {name} first sample contains Inf")
            y0_val = y0.item() if isinstance(y0, torch.Tensor) else int(y0)
            if not (0 <= y0_val < self.num_classes):
                raise ValueError(
                    f"[{self.dataset_name}] {name} label {y0_val} out of range [0,{self.num_classes})"
                )

    def train_dataloader(self):
        if self.learning_mode == "self_supervised" and self.unsupervised_dataset:
            return DataLoader(
                self.unsupervised_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def supervised_dataloader(self):
        """自监督模式下监督微调阶段使用的数据加载器。"""
        if self.supervised_dataset is None:
            raise RuntimeError("supervised_dataloader only available in self_supervised mode")
        return DataLoader(
            self.supervised_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
