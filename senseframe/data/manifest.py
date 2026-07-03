"""
声明式数据集 manifest：零代码注册自定义数据集。

设计目标：
- 用户通过 JSON/YAML 文件描述数据集，无需继承 SceneContainer 即可训练
- 支持任意文件格式（.npy/.npz/.pt/.mat），由 manifest 声明加载方式
- 归一化策略可配置：None / "auto" / 显式 mean+std / minmax
- 数据集分割可配置：显式 split 或自动按比例分割
- 与 CustomContainer 协同，提供"声明式即用"工作流

manifest JSON 格式示例：
{
    "name": "my_csi_dataset",
    "description": "自定义 WiFi CSI 动作识别数据集",
    "num_classes": 5,
    "input_shape": [3, 114, 500],
    "label_map": {"0": "walk", "1": "run", "2": "sit", "3": "stand", "4": "fall"},
    "file_format": "npy",
    "normalization": "auto",
    "samples": [
        {"path": "data/train/walk_001.npy", "label": 0, "split": "train"},
        {"path": "data/train/walk_002.npy", "label": 0, "split": "train"},
        {"path": "data/test/walk_001.npy", "label": 0, "split": "test"}
    ]
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# 支持的文件格式
# ============================================================
# s3：扩展支持的数据格式（新增 h5/hdf5/parquet/pq）
SUPPORTED_FORMATS = ("npy", "npz", "pt", "mat", "h5", "hdf5", "parquet", "pq")

# 支持的归一化策略
SUPPORTED_NORMALIZATION = (None, "auto", "zscore", "minmax", "none")


# ============================================================
# Manifest 数据结构
# ============================================================
@dataclass
class SampleEntry:
    """单条样本声明。"""
    path: str
    label: int
    split: str = "train"  # train / test / val
    # 可选元数据（如 subject / session，便于跨域分析）
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "label": self.label, "split": self.split, **self.meta}


@dataclass
class DatasetManifest:
    """
    声明式数据集描述。

    零代码注册新数据集：用户只需写一个 JSON/YAML 文件，无需改源码。
    """
    name: str
    samples: List[SampleEntry]
    num_classes: int
    input_shape: List[int]
    label_map: Dict[int, str] = field(default_factory=dict)
    description: str = ""
    file_format: str = "npy"  # npy / npz / pt / mat
    # 归一化策略：
    #   None / "none"  → 不归一化
    #   "auto"         → 自动从训练集计算 z-score 常量
    #   "zscore"       → 使用 normalization_stats 的 mean/std
    #   "minmax"       → 自动从训练集计算 min/max
    normalization: Optional[str] = None
    normalization_stats: Optional[Dict[str, List[float]]] = None
    # 数据集根目录（样本 path 相对此根，若 path 为绝对路径则忽略）
    data_root: Optional[str] = None
    # .mat 格式时的键名（如 "CSIamp"）
    mat_key: Optional[str] = None

    def __post_init__(self):
        if self.file_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"file_format '{self.file_format}' 不支持，可选: {SUPPORTED_FORMATS}"
            )
        if self.normalization not in SUPPORTED_NORMALIZATION:
            raise ValueError(
                f"normalization '{self.normalization}' 不支持，可选: {SUPPORTED_NORMALIZATION}"
            )
        if not self.samples:
            raise ValueError("samples 不能为空")
        if self.num_classes < 2:
            raise ValueError(f"num_classes 必须 >= 2，实际: {self.num_classes}")
        # 校验 label 范围
        for s in self.samples:
            if not (0 <= s.label < self.num_classes):
                raise ValueError(
                    f"样本 {s.path} 的 label={s.label} 超出 [0, {self.num_classes})"
                )

    def get_samples_by_split(self, split: str) -> List[SampleEntry]:
        """按 split 过滤样本。"""
        return [s for s in self.samples if s.split == split]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "num_classes": self.num_classes,
            "input_shape": list(self.input_shape),
            "label_map": {str(k): v for k, v in self.label_map.items()},
            "file_format": self.file_format,
            "normalization": self.normalization,
            "normalization_stats": self.normalization_stats,
            "data_root": self.data_root,
            "mat_key": self.mat_key,
            "samples": [s.to_dict() for s in self.samples],
        }


# ============================================================
# Manifest 加载
# ============================================================
def load_manifest(path: Union[str, Path]) -> DatasetManifest:
    """
    从 JSON 或 YAML 文件加载 manifest。

    支持扩展名：.json / .yaml / .yml

    Args:
        path: manifest 文件路径

    Returns:
        DatasetManifest 实例

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式错误或字段校验失败
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(
                "YAML manifest 需要 PyYAML 包，请 `pip install pyyaml`"
            )
        data = yaml.safe_load(text)
    else:
        # 尝试 JSON 解析作为兜底
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"不支持的 manifest 扩展名 '{suffix}'，支持 .json/.yaml/.yml"
            )

    return manifest_from_dict(data, default_data_root=str(p.parent))


def manifest_from_dict(
    data: Dict[str, Any],
    default_data_root: Optional[str] = None,
) -> DatasetManifest:
    """
    从 dict 构造 DatasetManifest。

    Args:
        data: manifest 字典
        default_data_root: 默认数据根目录（manifest 未指定 data_root 时使用）

    Returns:
        DatasetManifest 实例
    """
    if not isinstance(data, dict):
        raise ValueError(f"manifest 顶层必须是 dict，实际: {type(data)}")

    # 必需字段
    for required in ("name", "samples", "num_classes", "input_shape"):
        if required not in data:
            raise ValueError(f"manifest 缺少必需字段: '{required}'")

    # 解析 samples
    raw_samples = data["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("samples 必须是非空 list")

    samples: List[SampleEntry] = []
    for i, s in enumerate(raw_samples):
        if not isinstance(s, dict):
            raise ValueError(f"samples[{i}] 必须是 dict，实际: {type(s)}")
        if "path" not in s:
            raise ValueError(f"samples[{i}] 缺少 'path' 字段")
        if "label" not in s:
            raise ValueError(f"samples[{i}] 缺少 'label' 字段")
        split = s.get("split", "train")
        meta = {k: v for k, v in s.items() if k not in ("path", "label", "split")}
        samples.append(SampleEntry(
            path=str(s["path"]),
            label=int(s["label"]),
            split=split,
            meta=meta,
        ))

    # label_map：JSON 的 key 是字符串，转为 int
    raw_label_map = data.get("label_map", {})
    label_map = {int(k): str(v) for k, v in raw_label_map.items()}

    # data_root：manifest 未指定则用 default_data_root
    data_root = data.get("data_root", default_data_root)

    return DatasetManifest(
        name=str(data["name"]),
        samples=samples,
        num_classes=int(data["num_classes"]),
        input_shape=[int(x) for x in data["input_shape"]],
        label_map=label_map,
        description=str(data.get("description", "")),
        file_format=str(data.get("file_format", "npy")),
        normalization=data.get("normalization"),
        normalization_stats=data.get("normalization_stats"),
        data_root=data_root,
        mat_key=data.get("mat_key"),
    )


# ============================================================
# 文件加载器
# ============================================================
def _load_sample_file(
    path: Path,
    file_format: str,
    mat_key: Optional[str] = None,
) -> np.ndarray:
    """
    根据文件格式加载单个样本。

    Args:
        path: 文件路径
        file_format: npy / npz / pt / mat
        mat_key: .mat 格式时的键名（必填）

    Returns:
        numpy array

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式不支持或 mat_key 缺失
    """
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    if file_format == "npy":
        return np.load(path)

    if file_format == "npz":
        data = np.load(path)
        # npz 是多键压缩格式，取第一个键的数据
        if hasattr(data, "files"):
            if len(data.files) == 0:
                raise ValueError(f"Empty npz file: {path}")
            return data[data.files[0]]
        return data

    if file_format == "pt":
        t = torch.load(path, weights_only=False)
        if isinstance(t, torch.Tensor):
            return t.numpy()
        # 字典形式：取第一个 tensor
        if isinstance(t, dict):
            for v in t.values():
                if isinstance(v, torch.Tensor):
                    return v.numpy()
        raise ValueError(f"Cannot extract array from .pt file: {path}")

    if file_format == "mat":
        if mat_key is None:
            raise ValueError(
                f"file_format='mat' 需要指定 mat_key，"
                f"如 \"CSIamp\"。manifest 中通过 mat_key 字段声明"
            )
        import scipy.io as sio
        return sio.loadmat(str(path))[mat_key]

    raise ValueError(f"Unsupported file_format: {file_format}")


# ============================================================
# 归一化
# ============================================================
def auto_compute_normalization(
    samples: List[SampleEntry],
    data_root: Optional[str],
    file_format: str,
    mat_key: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    从训练集自动计算 z-score 归一化常量（mean/std）。

    逐样本累加，避免一次性全量加载占用内存。

    Args:
        samples: 训练集样本列表
        data_root: 数据根目录
        file_format: 文件格式
        mat_key: .mat 格式时的键名

    Returns:
        {"mean": float, "std": float}（标量广播，适用于任意形状）
    """
    if not samples:
        raise ValueError("无法计算归一化常量：训练集为空")

    running_sum = 0.0
    running_sq_sum = 0.0
    n_elements = 0

    for s in samples:
        path = _resolve_path(s.path, data_root)
        x = _load_sample_file(path, file_format, mat_key)
        x = x.astype(np.float64)
        running_sum += float(x.sum())
        running_sq_sum += float((x ** 2).sum())
        n_elements += x.size

    mean = running_sum / n_elements
    var = running_sq_sum / n_elements - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))

    return {"mean": [mean], "std": [std]}


def auto_compute_minmax(
    samples: List[SampleEntry],
    data_root: Optional[str],
    file_format: str,
    mat_key: Optional[str] = None,
) -> Dict[str, List[float]]:
    """从训练集自动计算 min/max（用于 minmax 归一化）。"""
    if not samples:
        raise ValueError("无法计算 min/max：训练集为空")

    global_min = float("inf")
    global_max = float("-inf")

    for s in samples:
        path = _resolve_path(s.path, data_root)
        x = _load_sample_file(path, file_format, mat_key)
        x = x.astype(np.float64)
        global_min = min(global_min, float(x.min()))
        global_max = max(global_max, float(x.max()))

    return {"min": [global_min], "max": [global_max]}


def _resolve_path(path: str, data_root: Optional[str]) -> Path:
    """解析样本路径：绝对路径直接用，相对路径拼接 data_root。"""
    p = Path(path)
    if p.is_absolute():
        return p
    if data_root is None:
        return p
    return Path(data_root) / path


# ============================================================
# ManifestDataset
# ============================================================
class ManifestDataset(Dataset):
    """
    基于 manifest 的通用数据集。

    根据 DatasetManifest 加载样本文件，应用归一化，返回 (x_tensor, y_long)。

    特性：
    - 支持 .npy/.npz/.pt/.mat 任意格式
    - 归一化在 __getitem__ 中逐样本应用（避免全量加载）
    - 返回 torch.FloatTensor + torch.long 标签
    - 仅负责 I/O + 归一化，不做 reshape/采样（由 TransformConfig 处理）
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        split: str = "train",
        normalization_stats: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Args:
            manifest: 数据集 manifest
            split: 加载哪个 split（train/test/val）
            normalization_stats: 归一化常量（None 时按 manifest 策略处理）
        """
        self.manifest = manifest
        self.split = split
        self.samples = manifest.get_samples_by_split(split)
        if not self.samples:
            raise ValueError(
                f"Manifest '{manifest.name}' 的 split='{split}' 为空"
            )

        # 解析归一化
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._norm_strategy = manifest.normalization or "none"

        if normalization_stats is not None:
            # 外部传入（如训练集计算后传给测试集）
            self._set_normalization(normalization_stats)
        elif manifest.normalization_stats is not None:
            # manifest 显式声明
            self._set_normalization(manifest.normalization_stats)

    def _set_normalization(self, stats: Dict[str, List[float]]) -> None:
        """设置归一化常量。"""
        if "mean" in stats and "std" in stats:
            self._mean = float(stats["mean"][0]) if isinstance(stats["mean"], list) else float(stats["mean"])
            self._std = float(stats["std"][0]) if isinstance(stats["std"], list) else float(stats["std"])
        if "min" in stats and "max" in stats:
            self._min = float(stats["min"][0]) if isinstance(stats["min"], list) else float(stats["min"])
            self._max = float(stats["max"][0]) if isinstance(stats["max"], list) else float(stats["max"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.samples[idx]
        path = _resolve_path(entry.path, self.manifest.data_root)
        x = _load_sample_file(path, self.manifest.file_format, self.manifest.mat_key)

        # 应用归一化
        x = x.astype(np.float32)
        if self._norm_strategy in ("zscore", "auto") and self._mean is not None:
            x = (x - self._mean) / self._std
        elif self._norm_strategy == "minmax" and self._min is not None:
            x = (x - self._min) / max(self._max - self._min, 1e-12)

        return torch.from_numpy(x), torch.tensor(entry.label, dtype=torch.long)


# ============================================================
# 便捷函数：从 manifest 构造训练/测试 Dataset
# ============================================================
def build_datasets_from_manifest(
    manifest: DatasetManifest,
) -> Tuple[ManifestDataset, ManifestDataset, Optional[Dict[str, Any]]]:
    """
    从 manifest 构造训练集和测试集 Dataset。

    若 manifest.normalization == "auto"，从训练集计算 z-score 常量，
    并应用到测试集。

    Args:
        manifest: 数据集 manifest

    Returns:
        (train_dataset, test_dataset, normalization_stats)
        normalization_stats 为 None 表示不归一化
    """
    # 处理 auto 归一化：从训练集计算
    norm_stats = None
    if manifest.normalization == "auto":
        train_samples = manifest.get_samples_by_split("train")
        norm_stats = auto_compute_normalization(
            train_samples,
            manifest.data_root,
            manifest.file_format,
            manifest.mat_key,
        )
    elif manifest.normalization == "minmax":
        train_samples = manifest.get_samples_by_split("train")
        norm_stats = auto_compute_minmax(
            train_samples,
            manifest.data_root,
            manifest.file_format,
            manifest.mat_key,
        )
    elif manifest.normalization_stats is not None:
        norm_stats = manifest.normalization_stats

    train_ds = ManifestDataset(
        manifest, split="train", normalization_stats=norm_stats,
    )
    test_ds = ManifestDataset(
        manifest, split="test", normalization_stats=norm_stats,
    )

    return train_ds, test_ds, norm_stats


__all__ = [
    "SampleEntry",
    "DatasetManifest",
    "ManifestDataset",
    "SUPPORTED_FORMATS",
    "SUPPORTED_NORMALIZATION",
    "load_manifest",
    "manifest_from_dict",
    "auto_compute_normalization",
    "auto_compute_minmax",
    "build_datasets_from_manifest",
]
