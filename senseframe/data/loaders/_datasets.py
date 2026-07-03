"""
共享 Dataset 类和路径解析工具。

从 data/legacy.py 迁移到 loaders 内部，消除 loaders → legacy 的反向依赖。
"""

import glob
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


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
    """CSI .mat 数据集。"""

    def __init__(self, root_dir: str, modal: str = "CSIamp"):
        self.root_dir = root_dir
        self.modal = modal
        self.data_list = glob.glob(str(Path(root_dir) / "*" / "*.mat"))
        folders = [p for p in Path(root_dir).iterdir() if p.is_dir()]
        self.category = {folders[i].name: i for i in range(len(folders))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[Path(sample_dir).parent.name]
        x = sio.loadmat(sample_dir)[self.modal]
        return x, y


class WidarDataset(Dataset):
    """Widar .csv 数据集。"""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data_list = glob.glob(str(Path(root_dir) / "*" / "*.csv"))
        folders = [p for p in Path(root_dir).iterdir() if p.is_dir()]
        self.category = {folders[i].name: i for i in range(len(folders))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[Path(sample_dir).parent.name]
        x = np.genfromtxt(sample_dir, delimiter=",")
        return x, y
