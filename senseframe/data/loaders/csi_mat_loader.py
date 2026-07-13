"""
CSI .mat 文件加载器：加载 NTU-Fi 风格的 .mat 文件，创建 CSIDataset。
"""

from __future__ import annotations

import logging

from torch.utils.data import ConcatDataset

from .base import DatasetLoader, DatasetSplits
from ._datasets import CSIDataset, resolve_data_path

# 修复（5.9）：数据加载器零日志，加模块级 logger
_logger = logging.getLogger(__name__)


class CSIMatLoader(DatasetLoader):
    """NTU-Fi 风格的 .mat CSI 数据加载器。

    支持 NTU-Fi-HumanID 和 NTU-Fi_HAR 数据集。
    自监督模式下，NTU-Fi_HAR 使用全部数据做无监督预训练，
    NTU-Fi-HumanID 做监督微调。
    """

    def load_splits(self, root: str, dataset_name: str,
                    learning_mode: str = "supervised") -> DatasetSplits:
        # P0-1.7: 从 DatasetSpec.layout 读取目录结构声明，传给 CSIDataset
        # P5 P2-2: 目录名从 spec.dir_names 派生（单一数据源），禁止硬编码
        # 框架不猜测 layout/dir_names，数据集必须已注册，未注册则 raise
        from ...registry import get_dataset_spec
        try:
            spec = get_dataset_spec(dataset_name)
        except KeyError:
            raise KeyError(
                f"CSIMatLoader: 数据集 '{dataset_name}' 未注册，无法派生 layout/dir_names。"
                f"请先通过场景注册声明该数据集的 layout 和 dir_names。"
            )
        if not spec.dir_names:
            raise ValueError(
                f"CSIMatLoader: 数据集 '{dataset_name}' 的 DatasetSpec.dir_names 为空。"
                f"请在注册时声明 dir_names。"
            )
        layout = spec.layout
        dir_name = spec.dir_names[0]
        if dataset_name == "NTU-Fi-HumanID":
            splits = self._load_humanid(root, layout, dir_name)
        elif dataset_name == "NTU-Fi_HAR":
            splits = self._load_ntu_har(root, learning_mode, layout, dir_name, spec)
        else:
            # 通用 .mat 加载：假设 train_amp / test_amp 子目录
            splits = self._load_generic(root, layout, dir_name)
        # 修复（5.9）：load_splits 返回前 log 样本数/形状/类别分布
        self._log_splits(dataset_name, learning_mode, splits)
        return splits

    def _log_splits(self, dataset_name: str, learning_mode: str,
                    splits: DatasetSplits) -> None:
        """记录 splits 摘要日志（样本数 / 类别分布 / 首 sample 形状）。"""
        train_n = len(splits.train) if splits.train is not None else 0
        test_n = len(splits.test) if splits.test is not None else 0
        unsup_n = len(splits.unsupervised) if splits.unsupervised is not None else 0
        sup_n = len(splits.supervised) if splits.supervised is not None else 0
        # P1-3: 探测首 sample 形状，验证数据通路（自监督模式下 train 可能为 None，
        # 优先用 train，回退到 unsupervised）
        train_shape = None
        probe_ds = splits.train if splits.train is not None else splits.unsupervised
        if probe_ds is not None and train_n + unsup_n > 0:
            try:
                first_sample = probe_ds[0]
                # first_sample 可能是 (x, y) 或 (x, y, extra)
                if isinstance(first_sample, (list, tuple)) and len(first_sample) >= 1:
                    x = first_sample[0]
                    if hasattr(x, "shape"):
                        train_shape = tuple(x.shape)
            except Exception as e:
                _logger.debug("CSIMatLoader: failed to probe first sample shape: %s", e)
        _logger.info(
            "CSIMatLoader.load_splits: dataset=%s, learning_mode=%s, "
            "train_samples=%d (shape=%s), test_samples=%d, "
            "unsupervised_samples=%d, supervised_samples=%d",
            dataset_name, learning_mode,
            train_n, train_shape, test_n, unsup_n, sup_n,
        )

    def _load_humanid(self, root: str, layout: str, dir_name: str) -> DatasetSplits:
        # P5 P2-2: 目录名从 spec.dir_names 派生，禁止硬编码 "NTU-Fi-HumanID"
        # 注意：NTU-Fi-HumanID 的 test_amp 目录实际存训练数据，train_amp 存测试数据
        # （数据集原始约定，非 bug，不改）
        train_dir = resolve_data_path(root, dir_name, "test_amp")
        test_dir = resolve_data_path(root, dir_name, "train_amp")
        return DatasetSplits(
            train=CSIDataset(train_dir, layout=layout),
            test=CSIDataset(test_dir, layout=layout),
        )

    def _load_ntu_har(self, root: str, learning_mode: str, layout: str,
                      dir_name: str, spec) -> DatasetSplits:
        # P5 P2-2: 目录名从 spec.dir_names 派生，禁止硬编码 "NTU-Fi_HAR"
        train_dir = resolve_data_path(root, dir_name, "train_amp")
        test_dir = resolve_data_path(root, dir_name, "test_amp")

        if learning_mode == "self_supervised":
            unsupervised = ConcatDataset([
                CSIDataset(train_dir, layout=layout),
                CSIDataset(test_dir, layout=layout),
            ])
            # P5 P2-2: 监督数据集从 spec.supervised_source 派生（单一数据源），
            # 禁止硬编码 "NTU-Fi-HumanID"
            if not spec.supervised_source:
                raise ValueError(
                    f"CSIMatLoader: 数据集 '{spec.name}' 声明 self_supervised 模式"
                    f"但 supervised_source 为空。请在注册时声明 supervised_source。"
                )
            from ...registry import get_dataset_spec
            try:
                supervised_spec = get_dataset_spec(spec.supervised_source)
            except KeyError:
                raise KeyError(
                    f"CSIMatLoader: supervised_source '{spec.supervised_source}' 未注册。"
                    f"请先注册该数据集的 spec（含 dir_names 和 layout）。"
                )
            if not supervised_spec.dir_names:
                raise ValueError(
                    f"CSIMatLoader: supervised_source '{spec.supervised_source}'"
                    f"的 dir_names 为空。"
                )
            supervised_dir_name = supervised_spec.dir_names[0]
            supervised_layout = supervised_spec.layout
            # 修复连带 bug：HumanID 应使用自己的 layout，而非沿用 NTU-Fi_HAR 的
            # NTU-Fi_HAR=nested, NTU-Fi-HumanID=flat，两者目录结构不同
            supervised_dir = resolve_data_path(root, supervised_dir_name, "test_amp")
            supervised = CSIDataset(supervised_dir, layout=supervised_layout)
            supervised_test_dir = resolve_data_path(root, supervised_dir_name, "train_amp")
            test = CSIDataset(supervised_test_dir, layout=supervised_layout)
            return DatasetSplits(
                unsupervised=unsupervised,
                supervised=supervised,
                test=test,
            )

        return DatasetSplits(
            train=CSIDataset(train_dir, layout=layout),
            test=CSIDataset(test_dir, layout=layout),
        )

    def _load_generic(self, root: str, layout: str, dir_name: str) -> DatasetSplits:
        # P5 P2-2: 目录名从 spec.dir_names 派生，禁止硬编码 dataset_name
        train_dir = resolve_data_path(root, dir_name, "train_amp")
        test_dir = resolve_data_path(root, dir_name, "test_amp")
        return DatasetSplits(
            train=CSIDataset(train_dir, layout=layout),
            test=CSIDataset(test_dir, layout=layout),
        )
