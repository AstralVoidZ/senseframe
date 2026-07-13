"""RFC-003 ε2 NAS：架构搜索空间（P2.6）。

定义 NAS 搜索空间的数据结构（DSP 合规）：
- ArchitectureParameterSpec：单个架构参数规格（cell_type / n_layers / hidden_dim / ...）
- ArchitectureSearchSpace：完整搜索空间，含 to_sp_search_space() 转换

与 SP SearchSpace 的关系：
- NAS SearchSpace 是 SP SearchSpace 的特化（约束 cell_type 取值范围 + NAS 特化参数）
- 通过 to_sp_search_space() 转换为标准 SP SearchSpace，注册到 StudyManager 后可被
 任意 SP Sampler（random / grid / evolutionary / ...）采样

P2 支持的 cell_type：
- conv1d：1D 卷积网络（WiFi CSI 时序信号）
- rnn：循环神经网络（LSTM / GRU）

P3 推迟的 cell_type：
- attention：Transformer 风格架构
- hybrid：conv1d + rnn 混合
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional

# P2 支持的 cell_type（attention / hybrid 推迟到 P3）
SUPPORTED_CELL_TYPES: List[str] = ["conv1d", "rnn"]
SUPPORTED_ACTIVATIONS: List[str] = ["relu", "gelu", "tanh", "elu"]
SUPPORTED_RNN_TYPES: List[str] = ["lstm", "gru"]


@dataclass
class ArchitectureParameterSpec:
    """单个架构参数规格（DSP 合规）。

    与 SP ParameterSpec 的区别：
    - SP ParameterSpec 用于通用超参（lr / batch_size / ...），type 为 float/int/categorical
    - ArchitectureParameterSpec 专用于 NAS 架构参数（cell_type / n_layers / hidden_dim / ...）
    - 通过 to_sp_param() 转换为 SP ParameterSpec，复用 SP 采样基础设施

    Attributes:
        name: 参数名（如 "cell_type" / "n_layers" / "hidden_dim"）
        type: 参数类型（"categorical" / "int" / "float"）
        choices: categorical 类型的可选值列表
        low: int/float 类型的下界
        high: int/float 类型的上界
        log: int/float 类型是否对数采样
        step: int 类型的步长（None 时随机整数）
        default: 默认值（用于 EvolutionarySampler 初始化）
    """
    name: str
    type: str  # "categorical" / "int" / "float"
    choices: Optional[List[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    default: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureParameterSpec":
        return cls(**d)

    def to_sp_param(self) -> Dict[str, Any]:
        """转换为 SP ParameterSpec 的 dict 表示（用于 to_sp_search_space）。"""
        return {
            "name": self.name,
            "type": self.type,
            "choices": self.choices,
            "low": self.low,
            "high": self.high,
            "log": self.log,
            "step": self.step,
        }


def _default_conv1d_params() -> List[ArchitectureParameterSpec]:
    """conv1d cell_type 的默认搜索参数。"""
    return [
        ArchitectureParameterSpec(
            name="cell_type", type="categorical",
            choices=["conv1d"], default="conv1d",
        ),
        ArchitectureParameterSpec(
            name="n_layers", type="int", low=1, high=8, default=3,
        ),
        ArchitectureParameterSpec(
            name="hidden_dim", type="int", low=16, high=512, log=True, default=64,
        ),
        ArchitectureParameterSpec(
            name="activation", type="categorical",
            choices=SUPPORTED_ACTIVATIONS, default="relu",
        ),
        ArchitectureParameterSpec(
            name="kernel_size", type="categorical",
            choices=[3, 5, 7], default=3,
        ),
        ArchitectureParameterSpec(
            name="dropout", type="float", low=0.0, high=0.5, default=0.1,
        ),
    ]


def _default_rnn_params() -> List[ArchitectureParameterSpec]:
    """rnn cell_type 的默认搜索参数。"""
    return [
        ArchitectureParameterSpec(
            name="cell_type", type="categorical",
            choices=["rnn"], default="rnn",
        ),
        ArchitectureParameterSpec(
            name="n_layers", type="int", low=1, high=8, default=2,
        ),
        ArchitectureParameterSpec(
            name="hidden_dim", type="int", low=16, high=512, log=True, default=128,
        ),
        ArchitectureParameterSpec(
            name="activation", type="categorical",
            choices=["tanh", "relu"], default="tanh",
        ),
        ArchitectureParameterSpec(
            name="rnn_type", type="categorical",
            choices=SUPPORTED_RNN_TYPES, default="lstm",
        ),
        ArchitectureParameterSpec(
            name="bidirectional", type="categorical",
            choices=[True, False], default=False,
        ),
        ArchitectureParameterSpec(
            name="dropout", type="float", low=0.0, high=0.5, default=0.1,
        ),
    ]


def _default_hybrid_params() -> List[ArchitectureParameterSpec]:
    """hybrid cell_type 的默认搜索参数（conv1d + rnn）。"""
    params = _default_conv1d_params()
    # 覆盖 cell_type choices
    params[0] = ArchitectureParameterSpec(
        name="cell_type", type="categorical",
        choices=["hybrid"], default="hybrid",
    )
    # 追加 RNN 特化参数
    params.append(ArchitectureParameterSpec(
        name="rnn_type", type="categorical",
        choices=SUPPORTED_RNN_TYPES, default="lstm",
    ))
    params.append(ArchitectureParameterSpec(
        name="bidirectional", type="categorical",
        choices=[True, False], default=False,
    ))
    return params


@dataclass
class ArchitectureSearchSpace:
    """NAS 架构搜索空间（DSP 合规）。

    P2 支持 cell_type ∈ {"conv1d", "rnn", "hybrid"}（attention 推迟到 P3）。
    每种 cell_type 有对应的默认参数集，可通过 custom_params 覆盖。

    DSP 合规：
    - schema_version: 版本化数据结构
    - schema(): 返回 JSON Schema（自省，反射 fields 构造）
    - describe(): 返回运行时状态摘要
    - to_dict / from_dict: 序列化/反序列化

    Attributes:
        schema_version: 数据结构版本
        cell_types: 允许的 cell_type 列表（默认 ["conv1d", "rnn"]）
        parameters: 完整参数规格列表（按 cell_type 分组拼接）
        custom_params: 用户自定义参数覆盖（按 name 匹配替换默认参数）
    """
    schema_version: str = "1.0.0"
    cell_types: List[str] = field(default_factory=lambda: ["conv1d", "rnn"])
    parameters: List[ArchitectureParameterSpec] = field(default_factory=list)
    custom_params: Dict[str, ArchitectureParameterSpec] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理：若 parameters 为空，按 cell_types 生成默认参数。"""
        if not self.parameters:
            self._build_default_parameters()

    def _build_default_parameters(self) -> None:
        """根据 cell_types 生成默认参数集。"""
        # 验证 cell_types
        for ct in self.cell_types:
            if ct not in SUPPORTED_CELL_TYPES and ct != "hybrid":
                raise ValueError(
                    f"Unsupported cell_type '{ct}' in P2 "
                    f"(supported: {SUPPORTED_CELL_TYPES + ['hybrid']}; "
                    "attention 推迟到 P3)"
                )

        all_params: Dict[str, ArchitectureParameterSpec] = {}

        if "conv1d" in self.cell_types:
            for p in _default_conv1d_params():
                all_params[p.name] = p

        if "rnn" in self.cell_types:
            for p in _default_rnn_params():
                # 若 conv1d 已有同名参数（如 n_layers），合并 choices
                if p.name in all_params:
                    existing = all_params[p.name]
                    if existing.type == "categorical" and existing.choices:
                        merged = list(set(existing.choices + (p.choices or [])))
                        all_params[p.name] = ArchitectureParameterSpec(
                            name=p.name, type="categorical",
                            choices=merged, default=existing.default,
                        )
                else:
                    all_params[p.name] = p

        if "hybrid" in self.cell_types:
            for p in _default_hybrid_params():
                if p.name not in all_params:
                    all_params[p.name] = p

        # 应用用户自定义覆盖
        for name, custom in self.custom_params.items():
            if name in all_params:
                all_params[name] = custom
            else:
                all_params[name] = custom  # 允许追加新参数

        # 特殊处理 cell_type：合并所有 cell_types 的 choices
        if "cell_type" in all_params:
            all_params["cell_type"] = ArchitectureParameterSpec(
                name="cell_type", type="categorical",
                choices=list(self.cell_types), default=self.cell_types[0],
            )

        self.parameters = list(all_params.values())

    # ============================================================
    # DSP 自省协议
    # ============================================================
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """返回 JSON Schema（DSP 自省，反射 fields 构造）。"""
        props: Dict[str, Any] = {}
        for f in fields(cls):
            ftype = f.type
            type_str = str(ftype) if not isinstance(ftype, type) else ftype.__name__

            if f.name == "schema_version":
                props[f.name] = {"type": "string", "default": "1.0.0"}
            elif "List" in type_str or "list" in type_str:
                props[f.name] = {"type": "array"}
            elif "Dict" in type_str or "dict" in type_str:
                props[f.name] = {"type": "object"}
            else:
                props[f.name] = {"type": "string"}

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "ArchitectureSearchSpace",
            "type": "object",
            "schema_version": "1.0.0",
            "properties": props,
        }

    def describe(self) -> Dict[str, Any]:
        """返回运行时状态摘要。"""
        return {
            "schema_version": self.schema_version,
            "cell_types": self.cell_types,
            "n_parameters": len(self.parameters),
            "parameter_names": [p.name for p in self.parameters],
        }

    def to_dict(self) -> Dict[str, Any]:
        """完整序列化。"""
        return {
            "schema_version": self.schema_version,
            "cell_types": list(self.cell_types),
            "parameters": [p.to_dict() for p in self.parameters],
            "custom_params": {k: v.to_dict() for k, v in self.custom_params.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureSearchSpace":
        """从 dict 反序列化。"""
        params = [ArchitectureParameterSpec.from_dict(p) for p in d.get("parameters", [])]
        custom = {
            k: ArchitectureParameterSpec.from_dict(v)
            for k, v in d.get("custom_params", {}).items()
        }
        return cls(
            schema_version=d.get("schema_version", "1.0.0"),
            cell_types=list(d.get("cell_types", ["conv1d", "rnn"])),
            parameters=params,
            custom_params=custom,
        )

    # ============================================================
    # SP 转换
    # ============================================================
    def to_sp_search_space(self):
        """转换为 SP 标准 SearchSpace（用于 SP ask/tell 注册）。

        Returns:
            senseframe.search_protocol.SearchSpace 实例
        """
        from ..search_protocol import ParameterSpec, SearchSpace

        sp_params = []
        for p in self.parameters:
            sp_params.append(ParameterSpec(**p.to_sp_param()))
        return SearchSpace(parameters=sp_params)

    # ============================================================
    # 工具方法
    # ============================================================
    def get_param(self, name: str) -> Optional[ArchitectureParameterSpec]:
        """按 name 查找参数规格。"""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """验证采样参数是否在搜索空间内（返回错误列表，空列表表示通过）。"""
        errors: List[str] = []
        for p in self.parameters:
            if p.name not in params:
                if p.default is None:
                    errors.append(f"missing required parameter: {p.name}")
                continue
            v = params[p.name]
            if p.type == "categorical" and p.choices:
                if v not in p.choices:
                    errors.append(
                        f"{p.name}={v!r} not in choices {p.choices}"
                    )
            elif p.type == "int":
                if not isinstance(v, int):
                    errors.append(f"{p.name}={v!r} is not int")
                elif p.low is not None and v < p.low:
                    errors.append(f"{p.name}={v} < low {p.low}")
                elif p.high is not None and v > p.high:
                    errors.append(f"{p.name}={v} > high {p.high}")
            elif p.type == "float":
                if not isinstance(v, (int, float)):
                    errors.append(f"{p.name}={v!r} is not float")
                elif p.low is not None and v < p.low:
                    errors.append(f"{p.name}={v} < low {p.low}")
                elif p.high is not None and v > p.high:
                    errors.append(f"{p.name}={v} > high {p.high}")
        return errors


__all__ = [
    "ArchitectureParameterSpec",
    "ArchitectureSearchSpace",
    "SUPPORTED_CELL_TYPES",
    "SUPPORTED_ACTIVATIONS",
    "SUPPORTED_RNN_TYPES",
]
