"""
UT_HAR tensor 加载器：加载 .npy 文件，创建 TensorDataset。
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import DatasetLoader, DatasetSplits
from ._datasets import resolve_data_path


def _load_ut_har(root_dir: str) -> Dict[str, torch.Tensor]:
    """加载 UT_HAR 数据集。

    RFC Phase C：归一化走 registry 策略注册表，消除硬编码 min-max。
    Agent 可通过 register_normalization("UT_HAR_data", strategy) 覆盖。
    未注册时回退到 IdentityStrategy（不做归一化），由后续 transform 处理。
    """
    from ...registry import get_normalization

    data_dir = resolve_data_path(root_dir, "UT_HAR", "data")
    label_dir = resolve_data_path(root_dir, "UT_HAR", "label")
    data_list = glob.glob(str(Path(data_dir) / "*.csv"))
    label_list = glob.glob(str(Path(label_dir) / "*.csv"))

    # RFC Phase C：通过 registry 策略注册表获取归一化
    # 默认未注册 UT_HAR_data，回退 IdentityStrategy（不归一化）
    # Agent 可通过 register_normalization("UT_HAR_data", ZScoreStrategy(...)) 注入
    norm_strategy = get_normalization("UT_HAR_data")

    wifi_data = {}
    for data_dir in data_list:
        data_name = Path(data_dir).stem
        with open(data_dir, "rb") as f:
            data = np.load(f)
            data = data.reshape(len(data), 1, 250, 90)
            # RFC Phase C：走策略注册表，不再硬编码 min-max
            data_norm = norm_strategy.apply(data)
        wifi_data[data_name] = torch.Tensor(data_norm)

    for label_dir in label_list:
        label_name = Path(label_dir).stem
        with open(label_dir, "rb") as f:
            label = np.load(f)
        wifi_data[label_name] = torch.Tensor(label)

    return wifi_data


class TensorLoader(DatasetLoader):
    """UT_HAR 风格的 .npy tensor 加载器。"""

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        data = _load_ut_har(root)
        train_set = TensorDataset(data["X_train"], data["y_train"])
        test_set = TensorDataset(
            torch.cat((data["X_val"], data["X_test"]), 0),
            torch.cat((data["y_val"], data["y_test"]), 0),
        )
        return DatasetSplits(train=train_set, test=test_set)
