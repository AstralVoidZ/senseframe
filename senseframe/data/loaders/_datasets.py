"""
共享 Dataset 类和路径解析工具。

从 data/legacy.py 迁移到 loaders 内部，消除 loaders → legacy 的反向依赖。

P0-1.7 修复：CSIDataset / WidarDataset 按 DatasetSpec.layout 声明工作，
禁止探测 fallback（先 */*.ext 再 *.ext）。layout 由注册表单一数据源决定。
"""

import glob
import logging
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


def resolve_data_path(root: str, dataset_name: str, *subdirs: str) -> str:
    """解析数据路径，处理嵌套目录结构。"""
    direct = Path(root, dataset_name, *subdirs)
    if direct.exists():
        return str(direct)

    nested = Path(root, dataset_name, dataset_name, *subdirs)
    if nested.exists():
        return str(nested)

    return str(direct)


class CSIDataset(Dataset):
    """CSI .mat 数据集。

    P0-1.7: 目录结构由 layout 参数声明，禁止探测 fallback。
    - layout="nested"（默认）: <root>/<class_dir>/<sample>.mat
    - layout="flat":           <root>/<sample>.mat
    """

    def __init__(self, root_dir: str, modal: str = "CSIamp", layout: str = "nested"):
        self.root_dir = root_dir
        self.modal = modal
        # L3 修复：resolve 后 glob，避免 .. 逃逸
        root_resolved = Path(root_dir).resolve()
        self.layout = layout

        if layout == "nested":
            # 类别子目录结构 <class_dir>/<sample>.mat
            # 根因修复（P2）：glob/iterdir 返回顺序由文件系统决定，跨运行不一致，
            # 导致 category 类别索引映射漂移（class_to_idx 不稳定）。
            # sorted 保证样本顺序与类别索引跨运行确定性。
            self.data_list = sorted(glob.glob(str(root_resolved / "*" / "*.mat")))
            if not self.data_list:
                raise FileNotFoundError(
                    f"CSIDataset: layout='nested' but no '*/*.mat' under {root_resolved}. "
                    f"若数据集实际为扁平结构，请在 DatasetSpec.layout='flat' 声明。"
                )
            folders = sorted(
                [p for p in root_resolved.iterdir() if p.is_dir()],
                key=lambda p: p.name,
            )
            self.category = {folders[i].name: i for i in range(len(folders))}
        elif layout == "flat":
            # 扁平结构 *.mat
            self.data_list = sorted(glob.glob(str(root_resolved / "*.mat")))
            if not self.data_list:
                raise FileNotFoundError(
                    f"CSIDataset: layout='flat' but no '*.mat' under {root_resolved}."
                )
            # 扁平结构无子目录类别，使用根目录名作为单一类别
            self.category = {root_resolved.name: 0}
        else:
            raise ValueError(
                f"CSIDataset: layout='{layout}' 不支持，可选: 'nested' / 'flat'"
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[Path(sample_dir).parent.name]
        x = sio.loadmat(sample_dir)[self.modal]
        # 根因修复（P1）：原实现返回 raw numpy，违反 Dataset 契约
        # （DataModule transforms 与默认 collate 期望 tensor）。
        # ascontiguousarray 保证数组连续，from_numpy 零拷贝共享内存。
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x, y


class WidarDataset(Dataset):
    """Widar .csv 数据集。

    P0-1.7: 目录结构由 layout 参数声明，禁止探测 fallback。
    - layout="nested"（默认）: <root>/<class_dir>/<sample>.csv
    - layout="flat":           <root>/<sample>.csv
    """

    def __init__(self, root_dir: str, layout: str = "nested"):
        self.root_dir = root_dir
        # L3 修复：resolve 后 glob，避免 .. 逃逸
        root_resolved = Path(root_dir).resolve()
        self.layout = layout

        if layout == "nested":
            # 根因修复（P2）：glob/iterdir 返回顺序由文件系统决定，跨运行不一致，
            # 导致 category 类别索引映射漂移（class_to_idx 不稳定）。
            # sorted 保证样本顺序与类别索引跨运行确定性。
            self.data_list = sorted(glob.glob(str(root_resolved / "*" / "*.csv")))
            if not self.data_list:
                raise FileNotFoundError(
                    f"WidarDataset: layout='nested' but no '*/*.csv' under {root_resolved}. "
                    f"若数据集实际为扁平结构，请在 DatasetSpec.layout='flat' 声明。"
                )
            folders = sorted(
                [p for p in root_resolved.iterdir() if p.is_dir()],
                key=lambda p: p.name,
            )
            self.category = {folders[i].name: i for i in range(len(folders))}
        elif layout == "flat":
            self.data_list = sorted(glob.glob(str(root_resolved / "*.csv")))
            if not self.data_list:
                raise FileNotFoundError(
                    f"WidarDataset: layout='flat' but no '*.csv' under {root_resolved}."
                )
            self.category = {root_resolved.name: 0}
        else:
            raise ValueError(
                f"WidarDataset: layout='{layout}' 不支持，可选: 'nested' / 'flat'"
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[Path(sample_dir).parent.name]
        x = np.genfromtxt(sample_dir, delimiter=",")
        # 根因修复（P1）：原实现返回 raw numpy，违反 Dataset 契约
        # （DataModule transforms 与默认 collate 期望 tensor）。
        # ascontiguousarray 保证数组连续，from_numpy 零拷贝共享内存。
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x, y
