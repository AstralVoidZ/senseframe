"""
Phase 11.3 — FeatureSpec 驱动模型构建。

将"模型需要知道的数据形状"从散落的参数（num_classes / input_dim /
input_shape / data_root）抽象为统一的 FeatureSpec，让 build_model_for_dataset
的契约更清晰，同时为分类之外的回归/检测/分割任务铺路。

设计要点：
- FeatureSpec 描述输入数据的物理/几何特征，与下游任务解耦
- SceneContainer.get_feature_spec(dataset_name, **kwargs) 可选覆写，
  返回该数据集的特征规格
- 默认从 get_dataset_info 自动派生（input_shape + n_features）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class FeatureSpec:
    """输入特征规格。

    字段：
    - input_shape: 完整输入形状（含 batch 维或不含均可，约定不含）
    - num_channels: 通道数（图像/CSI 矩阵的 C 维）
    - sequence_length: 时序长度（NLP/RNN/Transformer 友好）
    - feature_dim: 展平后特征维度（MLP 入口友好）
    - dtype: 输入 dtype（默认 float32）
    - modality: 输入模态标签（"wifi_csi" / "image" / "tabular" / "text" / ...）
    - extra: 场景特定扩展（如天线对数、采样点数）
    """

    input_shape: Optional[Tuple[int, ...]] = None
    num_channels: Optional[int] = None
    sequence_length: Optional[int] = None
    feature_dim: Optional[int] = None
    dtype: str = "float32"
    modality: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.input_shape is not None and not isinstance(self.input_shape, tuple):
            self.input_shape = tuple(self.input_shape)
        # 从 input_shape 派生 num_channels / sequence_length / feature_dim
        if self.input_shape is not None:
            if self.num_channels is None and len(self.input_shape) >= 1:
                self.num_channels = self.input_shape[0]
            if self.sequence_length is None:
                seq = 1
                for dim in self.input_shape[1:]:
                    seq *= dim
                self.sequence_length = seq
            if self.feature_dim is None:
                # feature_dim = 展平后总维度 = prod(input_shape)
                total = 1
                for dim in self.input_shape:
                    total *= dim
                self.feature_dim = total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "num_channels": self.num_channels,
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "dtype": self.dtype,
            "modality": self.modality,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FeatureSpec":
        if d is None:
            return cls()
        if isinstance(d, FeatureSpec):
            return d
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "input_shape" in kwargs and kwargs["input_shape"] is not None:
            kwargs["input_shape"] = tuple(kwargs["input_shape"])
        return cls(**kwargs)

    @classmethod
    def from_dataset_info(cls, info: Dict[str, Any], modality: str = None) -> "FeatureSpec":
        """从 get_dataset_info 字典派生 FeatureSpec（向后兼容）。"""
        input_shape = info.get("input_shape")
        if input_shape is not None and not isinstance(input_shape, tuple):
            input_shape = tuple(input_shape)
        n_features = info.get("n_features")
        num_channels = info.get("num_channels")
        sequence_length = info.get("sequence_length")
        if input_shape is not None:
            if num_channels is None and len(input_shape) >= 1:
                num_channels = input_shape[0]
            if sequence_length is None:
                seq = 1
                for dim in input_shape[1:]:
                    seq *= dim
                sequence_length = seq
        # feature_dim 优先用 n_features，否则从 input_shape 展平
        if n_features is None and input_shape is not None:
            total = 1
            for dim in input_shape:
                total *= dim
            n_features = total
        return cls(
            input_shape=input_shape,
            num_channels=num_channels,
            sequence_length=sequence_length,
            feature_dim=n_features,
            modality=modality or info.get("modality"),
        )

    def with_overrides(self, **kwargs) -> "FeatureSpec":
        """返回覆盖指定字段的副本。"""
        d = self.to_dict()
        for k, v in kwargs.items():
            if k in self.__dataclass_fields__:
                d[k] = v
            else:
                d["extra"][k] = v
        return FeatureSpec.from_dict(d)
