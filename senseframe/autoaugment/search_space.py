"""RFC-003 ε3 AutoAugment：增强搜索空间（P3.1.1）。

定义数据增强搜索空间的数据结构（DSP 合规）：
- AugmentationParameterSpec：单个增强参数规格（op/magnitude/probability）
- AugmentationSearchSpace：完整搜索空间，含 to_sp_search_space() 转换

与 SP SearchSpace 的关系：
- AutoAugment SearchSpace 是 SP SearchSpace 的特化（约束 op 取值范围 + magnitude/probability 范围）
- 通过 to_sp_search_space() 转换为标准 SP SearchSpace，注册到 StudyManager 后可被
  任意 SP Sampler（random / grid / evolutionary / autoaugment / ...）采样

P3 支持的增强算子（候选池）：
- time_jitter：时序抖动（WiFi CSI 时序信号）
- freq_masking：频域掩码（频谱增强）
- noise：高斯噪声
- cutout：随机遮挡（时序片段置零）
- mixup：样本混合（batch 级，P3 暂不通过 search_space 搜索，走 GenericDataModule.collate_fn）

设计原则（RFC-003 原则 4）：
- 搜索对象即 SP SearchSpace（不另起炉灶）
- 增强策略是 SP Sampler 的应用（统一注册）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..search_protocol import ParameterSpec, SearchSpace


# P3 支持的增强算子候选池
SUPPORTED_AUGMENT_OPS: List[str] = [
    "time_jitter",
    "freq_masking",
    "noise",
    "cutout",
    "none",  # no-op，允许搜索"不增强"
]

# 默认搜索空间配置
DEFAULT_N_OPS_RANGE: Tuple[int, int] = (1, 3)
DEFAULT_MAGNITUDE_RANGE: Tuple[float, float] = (0.0, 1.0)
DEFAULT_PROBABILITY_RANGE: Tuple[float, float] = (0.0, 1.0)


@dataclass
class AugmentationParameterSpec:
    """单个增强参数规格（DSP 合规）。

    与 SP ParameterSpec 的区别：
    - SP ParameterSpec 用于通用超参（lr / batch_size / ...）
    - AugmentationParameterSpec 专用于增强参数（op_i / magnitude_i / probability_i）
    - 通过 to_sp_param() 转换为 SP ParameterSpec，复用 SP 采样基础设施

    Attributes:
        name: 参数名（如 "op_0" / "magnitude_0" / "probability_0"）
        type: 参数类型（"categorical" / "float"）
        choices: categorical 类型的可选值列表（op_i 用）
        low: float 类型的下界（magnitude_i / probability_i 用）
        high: float 类型的上界
        default: 默认值（用于 AutoAugmentSampler 初始化）
    """
    name: str
    type: str  # "categorical" / "float"
    choices: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    default: Optional[Any] = None

    def to_sp_param(self) -> ParameterSpec:
        """转换为 SP ParameterSpec。"""
        return ParameterSpec(
            name=self.name,
            type=self.type,
            choices=self.choices,
            low=self.low,
            high=self.high,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "choices": self.choices,
            "low": self.low,
            "high": self.high,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugmentationParameterSpec":
        return cls(
            name=d["name"],
            type=d["type"],
            choices=d.get("choices"),
            low=d.get("low"),
            high=d.get("high"),
            default=d.get("default"),
        )


@dataclass
class AugmentationSearchSpace:
    """数据增强搜索空间（DSP 合规）。

    搜索对象：N 个增强算子的组合 + 每个算子的 magnitude/probability。
    对齐 Google AutoAugment + RandAugment 设计。

    P3 实现策略：
    - 固定 N 个 op 槽位（n_ops），每个槽位搜索 (op, magnitude, probability)
    - op_i: categorical（从 SUPPORTED_AUGMENT_OPS 中选）
    - magnitude_i: float（控制增强强度）
    - probability_i: float（控制增强应用概率）

    Attributes:
        schema_version: schema 版本
        ops: 候选算子池（默认 SUPPORTED_AUGMENT_OPS）
        n_ops: 采样算子数（每次策略含 N 个 op 槽位）
        magnitude_range: magnitude 范围 (low, high)
        probability_range: probability 范围 (low, high)
    """
    schema_version: str = "1.0.0"
    ops: List[str] = field(default_factory=lambda: list(SUPPORTED_AUGMENT_OPS))
    n_ops: int = 2
    magnitude_range: Tuple[float, float] = DEFAULT_MAGNITUDE_RANGE
    probability_range: Tuple[float, float] = DEFAULT_PROBABILITY_RANGE

    def __post_init__(self):
        if self.n_ops < 1:
            raise ValueError(f"n_ops must be >= 1, got {self.n_ops}")
        if self.n_ops > 5:
            raise ValueError(f"n_ops must be <= 5, got {self.n_ops}")
        if not self.ops:
            raise ValueError("ops must not be empty")
        if self.magnitude_range[0] < 0 or self.magnitude_range[1] > 1:
            raise ValueError(
                f"magnitude_range must be in [0, 1], got {self.magnitude_range}"
            )
        if self.magnitude_range[0] > self.magnitude_range[1]:
            raise ValueError(
                f"magnitude_range low > high: {self.magnitude_range}"
            )
        if self.probability_range[0] < 0 or self.probability_range[1] > 1:
            raise ValueError(
                f"probability_range must be in [0, 1], got {self.probability_range}"
            )
        if self.probability_range[0] > self.probability_range[1]:
            raise ValueError(
                f"probability_range low > high: {self.probability_range}"
            )

    def to_sp_search_space(self) -> SearchSpace:
        """转换为 SP SearchSpace。

        每个槽位 i 生成 3 个参数：
        - op_i: categorical（从 self.ops 选）
        - magnitude_i: float（self.magnitude_range）
        - probability_i: float（self.probability_range）

        Returns:
            SP SearchSpace，含 n_ops * 3 个 ParameterSpec
        """
        params: List[ParameterSpec] = []
        for i in range(self.n_ops):
            params.append(ParameterSpec(
                name=f"op_{i}",
                type="categorical",
                choices=list(self.ops),
            ))
            params.append(ParameterSpec(
                name=f"magnitude_{i}",
                type="float",
                low=self.magnitude_range[0],
                high=self.magnitude_range[1],
            ))
            params.append(ParameterSpec(
                name=f"probability_{i}",
                type="float",
                low=self.probability_range[0],
                high=self.probability_range[1],
            ))
        return SearchSpace(parameters=params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ops": list(self.ops),
            "n_ops": self.n_ops,
            "magnitude_range": list(self.magnitude_range),
            "probability_range": list(self.probability_range),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AugmentationSearchSpace":
        return cls(
            schema_version=d.get("schema_version", "1.0.0"),
            ops=list(d.get("ops", SUPPORTED_AUGMENT_OPS)),
            n_ops=int(d.get("n_ops", 2)),
            magnitude_range=tuple(d.get("magnitude_range", DEFAULT_MAGNITUDE_RANGE)),
            probability_range=tuple(d.get("probability_range", DEFAULT_PROBABILITY_RANGE)),
        )

    def describe(self) -> str:
        """人类可读描述（DSP 合规）。"""
        return (
            f"AugmentationSearchSpace(n_ops={self.n_ops}, "
            f"ops={self.ops}, "
            f"magnitude_range={self.magnitude_range}, "
            f"probability_range={self.probability_range})"
        )

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """验证采样参数是否符合搜索空间约束。

        Returns:
            错误消息列表（空列表表示无错误）
        """
        errors: List[str] = []
        for i in range(self.n_ops):
            op_key = f"op_{i}"
            mag_key = f"magnitude_{i}"
            prob_key = f"probability_{i}"

            if op_key not in params:
                errors.append(f"missing {op_key}")
            elif params[op_key] not in self.ops:
                errors.append(
                    f"{op_key}={params[op_key]!r} not in ops={self.ops}"
                )

            if mag_key not in params:
                errors.append(f"missing {mag_key}")
            else:
                mag = params[mag_key]
                if not isinstance(mag, (int, float)):
                    errors.append(f"{mag_key} must be float, got {type(mag).__name__}")
                elif not self.magnitude_range[0] <= mag <= self.magnitude_range[1]:
                    errors.append(
                        f"{mag_key}={mag} out of range {self.magnitude_range}"
                    )

            if prob_key not in params:
                errors.append(f"missing {prob_key}")
            else:
                prob = params[prob_key]
                if not isinstance(prob, (int, float)):
                    errors.append(f"{prob_key} must be float, got {type(prob).__name__}")
                elif not self.probability_range[0] <= prob <= self.probability_range[1]:
                    errors.append(
                        f"{prob_key}={prob} out of range {self.probability_range}"
                    )
        return errors


def build_default_search_space(n_ops: int = 2) -> AugmentationSearchSpace:
    """构造默认增强搜索空间（便捷工厂）。"""
    return AugmentationSearchSpace(n_ops=n_ops)


__all__ = [
    "AugmentationParameterSpec",
    "AugmentationSearchSpace",
    "SUPPORTED_AUGMENT_OPS",
    "DEFAULT_N_OPS_RANGE",
    "DEFAULT_MAGNITUDE_RANGE",
    "DEFAULT_PROBABILITY_RANGE",
    "build_default_search_space",
]
