"""Radio 场景数据集：RadioML 2016A / 2018 加载器。

P1.2 落地：stub 实现 + RadioML 2018 真实加载器（HDF5）。

RadioML 数据集规格：
- RadioML 2016A：11 调制方式 × 20 SNR = 220k 样本，每样本 (2, 128) IQ 数据
- RadioML 2018：24 调制方式 × 26 SNR = 2.5M 样本，每样本 (2, 1024) IQ 数据

RadioML 2018 真实加载：见 `RadioML2018Dataset`，使用 h5py 流式读取，
避免一次性载入 19GB 数据。Stub 仅在数据文件缺失时启用。
真实 RadioML 数据集需从 https://www.deepsig.ai/ 下载并放置到 data_root。
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


# ============================================================
# 数据集元数据
# ============================================================
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "RadioML2016A": {
        "name": "RadioML2016A",
        "num_classes": 11,
        "classes": [
            "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
            "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
        ],
        "input_shape": (2, 128),
        "modality": "iq",
        "modulations": 11,
        "snr_range": (-20, 18, 2),  # -20dB 到 18dB，步长 2
        "file_format": "pkl",
    },
    "RadioML2018": {
        "name": "RadioML2018",
        "num_classes": 24,
        "classes": [
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM",
            "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC",
            "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
            "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "CPFSK",
            "BFPM", "PAM4",
        ],
        "input_shape": (2, 1024),
        "modality": "iq",
        "modulations": 24,
        "snr_range": (-20, 30, 2),
        "file_format": "h5",
    },
}


# ============================================================
# Stub 数据集
# ============================================================
class StubRadioDataset(Dataset):
    """Radio 数据集 stub（无外部依赖）。

    用于契约验证：在没有真实 RadioML 数据文件时，
    返回与真实数据集相同形状的随机样本。

    真实实现应从 {root}/{dataset_name}.{file_format} 加载 IQ 数据。
    """
    def __init__(self, dataset_name: str, n_samples: int = 256, seed: int = 42):
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown radio dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset_name]
        self.info = info
        self.n_samples = n_samples

        # 生成随机 IQ 数据（复用种子保证可复现）
        rng = np.random.default_rng(seed)
        self.x = torch.from_numpy(
            rng.standard_normal((n_samples, *info["input_shape"])).astype(np.float32)
        )
        self.y = torch.from_numpy(
            rng.integers(0, info["num_classes"], (n_samples,)).astype(np.int64)
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ============================================================
# RadioML 2018 真实加载器（HDF5）
# ============================================================
class RadioML2018Dataset(Dataset):
    """RadioML 2018.01A 真实数据集加载器（HDF5）。

    使用 h5py 流式读取，避免一次性载入 19GB X 数据到内存；
    Y（调制标签）/ Z（SNR）相对小（N×8B ≈ 20MB）全部载入内存做过滤。

    Args:
        root: 数据根目录（如 data/radio/radioml_2018_01a），
              内含 `2018_01_OS_3.h5` 或任意 `.h5` 文件
        max_samples: 最大加载样本数（None=全部，int=前 N 个），用于快速测试
        snr_filter: 只加载特定 SNR（dB）的样本（None=全部 SNR），
                    例如 [0, 10] 表示只加载 SNR=0 或 SNR=10 的样本
        modulation_filter: 只加载特定调制方式的样本（None=全部），
                            可传调制名（如 ["BPSK", "QPSK"]）或类别索引（如 [16, 17]）
        chunk_size: 预留参数：h5py 流式读取的 chunk 大小（当前 __getitem__ 单样本读取，
                    该参数仅用于将来可能的批量预取优化）

    HDF5 内部结构：
        /X    : (N, 1024, 2) float32/complex — IQ 样本
        /Y    : (N,) int — 调制类别索引 0-23
        /Z    : (N, 1) int — SNR dB
        /modulation_classes : (24,) str — 24 种调制名
        /snr_classes        : (26,) int — SNR dB 列表

    输出：
        x: torch.float32, shape=(2, 1024)  # 已 transpose 自 (1024, 2)
        y: torch.long, scalar              # 调制类别 0-23
    """

    def __init__(self, root: Union[str, Path],
                 max_samples: Optional[int] = None,
                 snr_filter: Optional[List[int]] = None,
                 modulation_filter: Optional[List[Union[str, int]]] = None,
                 chunk_size: int = 10000):
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(
                f"RadioML2018 数据根目录不存在或非目录: {root_path}"
            )

        # 用 glob 兼容多种命名（2018_01_OS_3.h5 / RadioML2018.h5 等）
        h5_candidates = sorted(root_path.glob("*.h5"))
        if not h5_candidates:
            raise FileNotFoundError(
                f"RadioML2018 数据目录中未找到 .h5 文件: {root_path}"
            )
        self.h5_path = h5_candidates[0]
        self.chunk_size = chunk_size

        # 只读模式打开 h5py，X 保持 lazy 引用，Y/Z 全部载入内存做过滤
        self.f = h5py.File(self.h5_path, "r")

        # 校验必需的 key
        required_keys = {"X", "Y", "Z", "modulation_classes", "snr_classes"}
        missing = required_keys - set(self.f.keys())
        if missing:
            self.f.close()
            raise ValueError(
                f"HDF5 文件缺少必需字段 {missing}: {self.h5_path}. "
                f"请确认是 RadioML 2018.01A 格式。"
            )

        # 读 Y / Z（小数据，全载入内存）
        y_arr = np.asarray(self.f["Y"])
        z_arr = np.asarray(self.f["Z"]).reshape(-1)
        self.modulation_classes = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in np.asarray(self.f["modulation_classes"]).tolist()
        ]
        self.snr_classes = np.asarray(self.f["snr_classes"]).reshape(-1).tolist()
        self._y_all = y_arr
        self._z_all = z_arr

        # 构造过滤 mask
        n_total = len(y_arr)
        mask = np.ones(n_total, dtype=bool)

        # max_samples：截取前 N 个
        if max_samples is not None and max_samples > 0:
            mask &= np.arange(n_total) < max_samples

        # snr_filter：只保留指定 SNR dB
        if snr_filter is not None:
            snr_set = set(int(s) for s in snr_filter)
            mask &= np.isin(z_arr, list(snr_set))

        # modulation_filter：调制名 → 索引，与类别索引混合处理
        if modulation_filter is not None:
            target_idxs: List[int] = []
            name_to_idx = {name: i for i, name in enumerate(self.modulation_classes)}
            for item in modulation_filter:
                if isinstance(item, str):
                    if item not in name_to_idx:
                        self.f.close()
                        raise ValueError(
                            f"未知调制方式: {item}. "
                            f"可用: {self.modulation_classes}"
                        )
                    target_idxs.append(name_to_idx[item])
                else:
                    idx = int(item)
                    if not 0 <= idx < len(self.modulation_classes):
                        self.f.close()
                        raise ValueError(
                            f"调制类别索引越界: {idx}, "
                            f"合法范围 [0, {len(self.modulation_classes)})"
                        )
                    target_idxs.append(idx)
            mask &= np.isin(y_arr, target_idxs)

        # indices 数组：过滤后的样本 idx → 原始 idx
        self.indices = np.nonzero(mask)[0]

        # 关联元数据（供上层使用）
        self.info = DATASET_INFO["RadioML2018"]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(f"索引越界: {idx}, len={len(self)}")
        real_idx = int(self.indices[idx])
        x = self.f["X"][real_idx]  # numpy array, shape (1024, 2)
        x = torch.from_numpy(np.asarray(x)).float().permute(1, 0)  # (2, 1024)
        y = torch.tensor(int(self.f["Y"][real_idx]), dtype=torch.long)
        return x, y

    def __del__(self):
        # 关闭 h5py 文件句柄（避免文件锁/资源泄漏）
        # 注意：random_split 创建的 Subset 共享底层 Dataset 实例，
        # 仅当原 Dataset 被释放时 __del__ 才触发，由 GC 保证。
        try:
            if hasattr(self, "f") and self.f is not None:
                self.f.close()
        except Exception:
            pass


# ============================================================
# 数据集加载函数
# ============================================================
def load_radioml_dataset(dataset_name: str, root: str,
                         learning_mode: str = "supervised") -> Dict[str, Any]:
    """加载 RadioML 数据集。

    Args:
        dataset_name: "RadioML2016A" 或 "RadioML2018"
        root: 数据根目录（含 RadioML 数据文件）
        learning_mode: 仅支持 "supervised"（RadioML 是监督学习数据集）

    Returns:
        dict: {
            "info": 数据集元数据,
            "train": 训练集 Dataset,
            "val": 验证集 Dataset,
            "test": 测试集 Dataset,
        }
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Unknown radio dataset: {dataset_name}. "
            f"Available: {list(DATASET_INFO.keys())}"
        )
    if learning_mode != "supervised":
        raise ValueError(
            f"RadioML 数据集不支持 learning_mode='{learning_mode}'，"
            f"仅支持 'supervised'"
        )

    info = DATASET_INFO[dataset_name]
    root_path = Path(root)

    # 检测真实数据文件是否存在：
    # - RadioML2018：root 下任意 .h5 文件（用户可能重命名为 RadioML2018.h5 / 2018_01_OS_3.h5）
    # - RadioML2016A：root 下固定 RadioML2016A.pkl（暂未实现真实加载器）
    real_file_exists = False
    if dataset_name == "RadioML2018":
        real_file_exists = bool(list(root_path.glob("*.h5"))) if root_path.exists() else False
    elif dataset_name == "RadioML2016A":
        real_file_exists = (root_path / f"{dataset_name}.{info['file_format']}").exists()

    if not real_file_exists:
        # Stub 模式：数据文件不存在时返回随机样本（供契约验证）
        full_ds = StubRadioDataset(dataset_name, n_samples=512, seed=42)
        # 8:1:1 划分 train/val/test
        n = len(full_ds)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n - n_train - n_val],
            generator=torch.Generator().manual_seed(42),
        )
    elif dataset_name == "RadioML2018":
        # 真实 RadioML 2018 加载（HDF5）：lazy h5py，Y/Z 载入内存
        full_ds = RadioML2018Dataset(root=root_path)
        # 8:1:1 划分 train/val/test
        # random_split 返回 Subset，Subset 通过 __getitem__ 间接调用底层
        # RadioML2018Dataset；同一 Dataset 实例（同一 h5py 句柄）在 Subset 间共享。
        n = len(full_ds)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        # RadioML2016A pickle 格式暂未实现真实加载器
        pkl_path = root_path / f"{dataset_name}.{info['file_format']}"
        raise NotImplementedError(
            f"RadioML 2016A 真实加载尚未实现（pickle 格式），"
            f"请使用 stub 模式（删除 {pkl_path}）"
        )

    return {
        "info": info,
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }
