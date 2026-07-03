"""
RFC Phase B：数据驱动决策 — DataProfiler 原语。

DataProfiler 探查数据特征，输出数据画像（DataProfile），
作为 Agent 策略决策的输入（task_type/loss/metric/model 推荐）。

设计原则（RFC 原则 2）：
- 策略选择必须基于数据特征，而非预设常量
- 框架"看数据"而非"查注册表挑候选"

DataProfile 包含：
- 基础统计：样本量、特征维度、类别数、类别分布
- 分布特征：缺失率、数值范围、均值/方差
- 结构特征：空间性（图像）、时序性（序列）、模态
- 推荐策略：基于画像推荐 task_type/loss/metric/model（非强制，可被 Agent 覆盖）

缓存：结果可缓存到 {output_dir}/data_profile.json，避免重复计算。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _count_dtypes(dtypes: Dict[str, str]) -> Dict[str, int]:
    """统计 dtype 分布。"""
    from collections import Counter
    return dict(Counter(dtypes.values()))


@dataclass
class DataProfile:
    """数据画像：DataProfiler 的输出。

    Agent 基于此画像决策 task_type/loss/metric/model。
    """
    # 基础统计
    n_samples: int = 0
    input_shape: Tuple[int, ...] = ()
    n_features: int = 0
    n_classes: Optional[int] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)

    # 分布特征
    missing_rate: float = 0.0
    value_range: Tuple[float, float] = (0.0, 1.0)
    mean: float = 0.0
    std: float = 1.0

    # 结构特征
    is_spatial: bool = False  # 图像/空间数据（input_shape len >= 3，含 H/W）
    is_temporal: bool = False  # 时序数据（input_shape len >= 2，含 sequence_length）
    modality: str = "unknown"  # csi/image/text/tabular/sequence/audio

    # 推荐策略（非强制，Agent 可覆盖）
    recommended_task_type: str = "classification"
    recommended_loss: str = "cross_entropy"
    recommended_metrics: List[str] = field(default_factory=lambda: ["accuracy", "macro_f1"])
    recommended_normalization: str = "zscore"  # none/zscore/minmax

    # 元信息
    dataset_name: str = ""
    # RFC-003 DSP-4：结构深化
    dtypes: Dict[str, str] = field(default_factory=dict)           # 特征名 → dtype 字符串
    feature_names: List[str] = field(default_factory=list)          # 特征名列表（有序）
    nullable: Dict[str, bool] = field(default_factory=dict)         # 特征名 → 是否允许缺失
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict) # 特征名 → 形状
    profile_source: str = ""  # "train" / "test" / "full"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["value_range"] = list(self.value_range)
        # RFC-003 DSP-4：shapes 的 tuple 值转 list
        d["shapes"] = {k: list(v) for k, v in self.shapes.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataProfile":
        d = {**d}
        d["input_shape"] = tuple(d.get("input_shape", ()))
        d["value_range"] = tuple(d.get("value_range", (0.0, 1.0)))
        # RFC-003 DSP-4：兼容旧数据（无新字段时用默认值）
        d["shapes"] = {k: tuple(v) for k, v in d.get("shapes", {}).items()}
        return cls(**d)

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """返回字段结构（RFC-003 DSP-4）。"""
        return {
            "schema_version": "1.0.0",
            "fields": [
                {"name": "dtypes", "type": "Dict[str, str]"},
                {"name": "feature_names", "type": "List[str]"},
                {"name": "nullable", "type": "Dict[str, bool]"},
                {"name": "shapes", "type": "Dict[str, Tuple[int, ...]]"},
            ],
        }

    def describe(self) -> Dict[str, Any]:
        """返回当前画像摘要（RFC-003 DSP-4）。"""
        return {
            "n_features": len(self.feature_names),
            "dtype_distribution": _count_dtypes(self.dtypes),
            "nullable_ratio": sum(1 for v in self.nullable.values() if v) / max(len(self.nullable), 1),
        }

    def save(self, path: str | Path) -> None:
        """缓存数据画像到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "DataProfile":
        """从 JSON 文件加载缓存的数据画像。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class DataProfiler:
    """数据探查器：探查 Dataset/DatasetBundle，输出 DataProfile。

    用法：
        profiler = DataProfiler()
        profile = profiler.profile_dataset(train_dataset, dataset_name="my_data")
        profile.save("output/data_profile.json")

        # Agent 基于 profile 决策
        task_type = profile.recommended_task_type
    """

    def __init__(self, max_samples: int = 1000):
        """
        Args:
            max_samples: 探查时最多采样多少样本（避免全量遍历大数据集）
        """
        self.max_samples = max_samples

    def profile_dataset(
        self,
        dataset,
        dataset_name: str = "",
        profile_source: str = "train",
    ) -> DataProfile:
        """探查 torch Dataset，输出数据画像。

        Args:
            dataset: torch Dataset（支持 __getitem__ 返回 (x, y) 或 (x,)）
            dataset_name: 数据集名称（用于元信息）
            profile_source: 数据来源标记

        Returns:
            DataProfile 数据画像
        """
        n_total = len(dataset) if hasattr(dataset, "__len__") else 0
        n_sample = min(n_total, self.max_samples) if n_total > 0 else 0

        if n_sample == 0:
            return DataProfile(dataset_name=dataset_name, profile_source=profile_source)

        # 采样
        samples_x: List[np.ndarray] = []
        samples_y: List[Any] = []
        indices = np.linspace(0, max(n_total - 1, 0), n_sample, dtype=int)
        for idx in indices:
            try:
                item = dataset[int(idx)]
                if isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        samples_x.append(self._to_numpy(item[0]))
                        samples_y.append(item[1])
                    elif len(item) == 1:
                        samples_x.append(self._to_numpy(item[0]))
                else:
                    samples_x.append(self._to_numpy(item))
            except Exception:
                continue

        if not samples_x:
            return DataProfile(dataset_name=dataset_name, profile_source=profile_source)

        # 基础统计
        first_x = samples_x[0]
        input_shape = first_x.shape
        n_features = int(np.prod(input_shape)) if input_shape else 0

        # 类别统计
        n_classes = None
        class_dist: Dict[str, int] = {}
        if samples_y:
            try:
                y_arr = np.array(samples_y)
                unique, counts = np.unique(y_arr, return_counts=True)
                n_classes = len(unique)
                class_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}
            except Exception:
                pass

        # 分布特征
        all_x = np.stack([x.flatten() for x in samples_x]) if samples_x else np.array([])
        missing_rate = float(np.isnan(all_x).mean()) if all_x.size > 0 else 0.0
        finite_mask = np.isfinite(all_x) if all_x.size > 0 else np.array([])
        finite_x = all_x[finite_mask] if finite_mask.any() else all_x
        value_min = float(finite_x.min()) if finite_x.size > 0 else 0.0
        value_max = float(finite_x.max()) if finite_x.size > 0 else 1.0
        mean = float(finite_x.mean()) if finite_x.size > 0 else 0.0
        std = float(finite_x.std()) if finite_x.size > 0 else 1.0

        # 结构特征推断
        is_spatial = len(input_shape) >= 3  # (C, H, W) 或更高
        is_temporal = len(input_shape) >= 2 and not is_spatial  # (C, T) 或 (T, F)
        modality = self._infer_modality(input_shape, is_spatial, is_temporal)

        # 推荐策略
        rec_task, rec_loss, rec_metrics, rec_norm = self._recommend(
            n_classes=n_classes,
            input_shape=input_shape,
            is_spatial=is_spatial,
            is_temporal=is_temporal,
            missing_rate=missing_rate,
            class_dist=class_dist,
            value_range=(value_min, value_max),
        )

        # RFC-003 DSP-4：结构深化 — 计算 dtypes/feature_names/nullable/shapes
        if input_shape:
            n_feat = int(np.prod(input_shape)) if input_shape else 0
            feature_names_list = [f"feature_{i}" for i in range(n_feat)]
            first_dtype = str(first_x.dtype) if hasattr(first_x, 'dtype') else "unknown"
            dtypes_dict = {name: first_dtype for name in feature_names_list}
            nullable_dict = {name: bool(missing_rate > 0) for name in feature_names_list}
            shapes_dict = {name: tuple(input_shape) for name in feature_names_list}
        else:
            feature_names_list = []
            dtypes_dict = {}
            nullable_dict = {}
            shapes_dict = {}

        return DataProfile(
            n_samples=n_total,
            input_shape=input_shape,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=class_dist,
            missing_rate=missing_rate,
            value_range=(value_min, value_max),
            mean=mean,
            std=std,
            is_spatial=is_spatial,
            is_temporal=is_temporal,
            modality=modality,
            recommended_task_type=rec_task,
            recommended_loss=rec_loss,
            recommended_metrics=rec_metrics,
            recommended_normalization=rec_norm,
            dataset_name=dataset_name,
            dtypes=dtypes_dict,
            feature_names=feature_names_list,
            nullable=nullable_dict,
            shapes=shapes_dict,
            profile_source=profile_source,
        )

    def profile_bundle(
        self,
        bundle,
        dataset_name: str = "",
    ) -> DataProfile:
        """探查 DatasetBundle，优先用 train 集。

        Args:
            bundle: DatasetBundle（含 train/test/val 等属性）
            dataset_name: 数据集名称

        Returns:
            DataProfile 数据画像
        """
        ds = getattr(bundle, "train", None) or getattr(bundle, "test", None)
        if ds is None:
            return DataProfile(dataset_name=dataset_name)
        return self.profile_dataset(ds, dataset_name=dataset_name, profile_source="train")

    def _to_numpy(self, x) -> np.ndarray:
        """转换为 numpy array。"""
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _infer_modality(
        self,
        input_shape: Tuple[int, ...],
        is_spatial: bool,
        is_temporal: bool,
    ) -> str:
        """推断数据模态。"""
        if is_spatial:
            return "image"
        if is_temporal:
            return "sequence"
        if len(input_shape) <= 1:
            return "tabular"
        return "unknown"

    def _recommend(
        self,
        n_classes: Optional[int],
        input_shape: Tuple[int, ...],
        is_spatial: bool,
        is_temporal: bool,
        missing_rate: float,
        class_dist: Dict[str, int],
        value_range: Tuple[float, float],
    ) -> Tuple[str, str, List[str], str]:
        """基于数据画像推荐策略（非强制，Agent 可覆盖）。

        推荐优先从注册表查询可用策略，回退到内置默认。

        Returns:
            (task_type, loss, metrics, normalization)
        """
        from .task import has_task_type, get_task_type_default_loss, get_task_type_default_metrics
        from .losses import has_loss

        # 任务类型：有类别数 → 分类；否则 → 回归
        if n_classes is not None and n_classes > 1:
            task_type = "classification"
        else:
            task_type = "regression"

        # 从注册表获取默认 loss/metrics
        if has_task_type(task_type):
            loss = get_task_type_default_loss(task_type)
            metrics = get_task_type_default_metrics(task_type)
        else:
            loss = "cross_entropy"
            metrics = ["accuracy", "macro_f1"]

        # 归一化推荐：值域不在 [0,1] 且 std > 0 → zscore
        vmin, vmax = value_range
        if std := (vmax - vmin):
            if vmin < -0.1 or vmax > 1.1 or abs(vmax - vmin) > 10:
                norm = "zscore"
            else:
                norm = "minmax"
        else:
            norm = "none"

        # 类别不平衡 → 推荐 focal loss（Agent 可覆盖）
        if n_classes is not None and n_classes > 1 and class_dist:
            counts = list(class_dist.values())
            if counts:
                ratio = max(counts) / max(min(counts), 1)
                if ratio > 5 and has_loss("focal"):
                    loss = "focal"

        return task_type, loss, metrics, norm
