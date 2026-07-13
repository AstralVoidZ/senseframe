"""RFC-003 ε6 experiment 模块类型定义（DSP 合规）。

定义对比实验的核心数据结构，全部满足 DSP（数据结构协议）：
- schema_version: 版本化数据结构
- schema(): 返回 JSON Schema（自省）
- describe(): 返回运行时状态（自省）

与 search_protocol.TrialResult 的区别：
- SP TrialResult：SP-2 Tell 的结果（trial_id + value + intermediate_values）
- experiment TrialResult：对比实验的单次试验结果（experiment_id + group + metrics + 效率 + 状态）
- 两者通过 trial_id 关联（experiment TrialResult 可引用 SP TrialResult 的 trial_id）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Enum
# ============================================================
class TrialGroup(str, Enum):
    """试验组别。"""
    METHOD = "method"                  # Agent + SP 驱动的搜索组
    BASELINE_PAPER = "baseline_paper"  # 论文报告的 baseline
    BASELINE_REPRO = "baseline_repro"  # 官方代码复现的 baseline


class TrialStatus(str, Enum):
    """试验状态。"""
    SUCCESS = "success"
    FAILED = "failed"
    PRUNED = "pruned"  # Multi-fidelity 早停（P2）


# ============================================================
# TrialResult（DSP 合规）
# ============================================================
@dataclass
class TrialResult:
    """单次对比试验结果（DSP 合规：schema_version + 自省）。

    与 SP TrialResult 的关系：
    - SP TrialResult 是搜索协议层的结果（trial_id + value + intermediate_values）
    - experiment TrialResult 是应用层的结果，聚合了性能/效率/人工成本/配置溯源
    - 通过 sp_trial_id 关联（可选，Baseline 组无 SP trial）
    """
    schema_version: str = "1.0.0"

    # 标识
    experiment_id: str = ""
    group: TrialGroup = TrialGroup.METHOD
    method_name: str = ""
    dataset: str = ""
    model_id: str = ""
    run_index: int = 0

    # SP 关联（Method 组有，Baseline 组无）
    sp_trial_id: Optional[str] = None

    # 性能
    metrics: Dict[str, float] = field(default_factory=dict)
    best_model_path: Optional[str] = None

    # 效率（过渡形态：从 TrainOutput 提取，P2 改从 OBP 查询）
    wall_time_s: float = 0.0
    n_epochs_trained: int = 0

    # 人工成本
    manual_tunes: Optional[int] = None   # Baseline 的人工调参次数
    agent_decisions: int = 0             # Method 的 SP ask 次数

    # 配置溯源
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    artifact_manifest_path: Optional[str] = None

    # 状态
    status: TrialStatus = TrialStatus.SUCCESS
    error_msg: Optional[str] = None

    # ============================================================
    # DSP 自省协议
    # ============================================================
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """返回 JSON Schema（DSP-5 自省）。

        用 dataclasses.fields 反射构造，不硬编码字段列表。
        """
        props: Dict[str, Any] = {}
        for f in fields(cls):
            ftype = f.type
            # 类型名简化（去掉 typing 包装）
            if isinstance(ftype, type):
                type_name = ftype.__name__
            else:
                type_str = str(ftype)
                # Optional[X] -> X
                type_str = type_str.replace("typing.Optional[", "").replace("]", "")
                type_str = type_str.replace("typing.Union[", "").replace(", NoneType", "")
                type_name = type_str

            if f.name == "schema_version":
                props[f.name] = {"type": "string", "default": "1.0.0"}
            elif f.name == "group":
                props[f.name] = {
                    "type": "string",
                    "enum": [g.value for g in TrialGroup],
                }
            elif f.name == "status":
                props[f.name] = {
                    "type": "string",
                    "enum": [s.value for s in TrialStatus],
                }
            elif "Dict" in type_name or "dict" in type_name:
                props[f.name] = {"type": "object"}
            elif "List" in type_name or "list" in type_name:
                props[f.name] = {"type": "array"}
            elif "int" in type_name:
                props[f.name] = {"type": "integer"}
            elif "float" in type_name:
                props[f.name] = {"type": "number"}
            elif "str" in type_name:
                props[f.name] = {"type": "string"}
            else:
                props[f.name] = {"type": "object"}

            if f.default is not field and f.default is not None:
                props[f.name]["default"] = (
                    f.default.value if isinstance(f.default, Enum) else f.default
                )

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "TrialResult",
            "type": "object",
            "schema_version": "1.0.0",
            "properties": props,
        }

    def describe(self) -> Dict[str, Any]:
        """返回运行时状态（DSP-5 自省）。

        与 to_dict 的区别：describe 侧重运行时状态摘要（用于日志/监控），
        to_dict 侧重完整序列化（用于持久化）。
        """
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "group": self.group.value,
            "method_name": self.method_name,
            "dataset": self.dataset,
            "model_id": self.model_id,
            "run_index": self.run_index,
            "status": self.status.value,
            "n_metrics": len(self.metrics),
            "wall_time_s": self.wall_time_s,
            "n_epochs_trained": self.n_epochs_trained,
            "agent_decisions": self.agent_decisions,
            "manual_tunes": self.manual_tunes,
            "has_model": self.best_model_path is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """完整序列化（含 Enum 转 value）。"""
        d = asdict(self)
        # Enum → value
        d["group"] = self.group.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialResult":
        """从 dict 反序列化（Enum 自动转换）。"""
        d = dict(d)
        if "group" in d and isinstance(d["group"], str):
            d["group"] = TrialGroup(d["group"])
        if "status" in d and isinstance(d["status"], str):
            d["status"] = TrialStatus(d["status"])
        return cls(**d)


__all__ = [
    "TrialGroup",
    "TrialStatus",
    "TrialResult",
]
