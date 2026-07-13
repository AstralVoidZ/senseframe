"""
UT_HAR tensor 加载器：加载 .npy 文件，创建 TensorDataset。
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .base import DatasetLoader, DatasetSplits
from ._datasets import resolve_data_path

# 修复（5.9）：数据加载器零日志，加模块级 logger
_logger = logging.getLogger(__name__)


def _load_ut_har(root_dir: str) -> Dict[str, torch.Tensor]:
    """加载 UT_HAR 数据集。

    RFC Phase C：归一化走 registry 策略注册表，消除硬编码 min-max。
    Agent 可通过 register_normalization("UT_HAR_data", strategy) 覆盖。
    未注册时回退到 IdentityStrategy（不做归一化），由后续 transform 处理。

    reshape 形状从 DatasetSpec.input_shape 派生（单一数据源），禁止硬编码 (1, 250, 90)。
    """
    from ...registry import get_dataset_spec, get_normalization

    # input_shape 与 file_format 必须从注册表派生，未注册则 raise（框架不猜测）
    try:
        spec = get_dataset_spec("UT_HAR_data")
    except KeyError:
        raise KeyError(
            "TensorLoader: 数据集 'UT_HAR_data' 未注册，无法派生 input_shape/file_format。"
            "请先通过场景注册声明该数据集的 input_shape 与 file_format。"
        )
    if not spec.input_shape:
        raise ValueError(
            "TensorLoader: 数据集 'UT_HAR_data' 的 DatasetSpec.input_shape 为空。"
            "请在注册时声明 input_shape（如 (1, 250, 90)）。"
        )
    input_shape = tuple(spec.input_shape)
    # P0-1.6: 扩展名从 DatasetSpec.file_format 派生，禁止 .npy/.csv fallback 探测
    ext_map = {"npy": ".npy", "csv": ".csv", "mat": ".mat", "image": ".jpg"}
    ext = ext_map.get(spec.file_format)
    if ext is None:
        raise ValueError(
            f"TensorLoader: DatasetSpec.file_format='{spec.file_format}' 不支持。"
            f"UT_HAR_data 应注册为 'npy'（或 'csv' 若扩展名确实为 .csv）。"
        )

    # P5 P2-2: 目录名从 spec.dir_names 派生（单一数据源），禁止硬编码 "UT_HAR"
    if not spec.dir_names:
        raise ValueError(
            "TensorLoader: 数据集 'UT_HAR_data' 的 DatasetSpec.dir_names 为空。"
            "请在注册时声明 dir_names（如 ('UT_HAR',)）。"
        )
    dir_name = spec.dir_names[0]
    data_dir = resolve_data_path(root_dir, dir_name, "data")
    label_dir = resolve_data_path(root_dir, dir_name, "label")
    # L3 修复：glob 前 resolve，避免相对路径含 .. 匹配逃逸目录
    data_dir_resolved = Path(data_dir).resolve()
    label_dir_resolved = Path(label_dir).resolve()
    # P0-1.6: 只 glob 声明的扩展名，不探测 fallback
    data_list = glob.glob(str(data_dir_resolved / f"*{ext}"))
    if not data_list:
        raise FileNotFoundError(
            f"TensorLoader: no '*{ext}' files under {data_dir_resolved}. "
            f"若数据集扩展名不同，请在 DatasetSpec.file_format 声明实际扩展名。"
        )
    label_list = glob.glob(str(label_dir_resolved / f"*{ext}"))
    if not label_list:
        raise FileNotFoundError(
            f"TensorLoader: no '*{ext}' files under {label_dir_resolved}. "
            f"若数据集扩展名不同，请在 DatasetSpec.file_format 声明实际扩展名。"
        )

    # RFC Phase C：通过 registry 策略注册表获取归一化
    # P1 修复：UT_HAR_data 已在 _register.py 注册 ZScoreStrategy(17.6529, 5.9034)
    norm_strategy = get_normalization("UT_HAR_data")

    wifi_data = {}
    for data_dir in data_list:
        data_name = Path(data_dir).stem
        with open(data_dir, "rb") as f:
            data = np.load(f)
            # reshape 形状从 DatasetSpec.input_shape 派生，禁止硬编码 (1, 250, 90)
            data = data.reshape(len(data), *input_shape)
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
        # P2-3 修复：保留原始独立 val split（不再拼入 test）
        # 修复前：test = torch.cat((X_val, X_test))，val 被丢弃，early_stopping 监控 test
        # 修复后：val/test 独立，early_stopping 监控真实 val，test 恢复为独立评估集
        val_set = TensorDataset(data["X_val"], data["y_val"])
        test_set = TensorDataset(data["X_test"], data["y_test"])
        # 修复（5.9）：load_splits 返回前 log 样本数/形状/类别分布
        train_y_np = data["y_train"].numpy()
        val_y_np = data["y_val"].numpy()
        test_y_np = data["y_test"].numpy()
        _logger.info(
            "TensorLoader.load_splits: dataset=%s, learning_mode=%s, "
            "train_samples=%d (shape=%s), val_samples=%d, test_samples=%d (shape=%s), "
            "train_classes=%d, val_classes=%d, test_classes=%d",
            dataset_name, learning_mode,
            len(train_set), tuple(train_set[0][0].shape) if len(train_set) > 0 else (),
            len(val_set),
            len(test_set), tuple(test_set[0][0].shape) if len(test_set) > 0 else (),
            len(np.unique(train_y_np)), len(np.unique(val_y_np)), len(np.unique(test_y_np)),
        )
        return DatasetSplits(train=train_set, val=val_set, test=test_set)
