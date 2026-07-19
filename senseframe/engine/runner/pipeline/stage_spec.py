"""RFC-003 DSP-3：Stage IO 声明。

包含：
- FieldSpec：字段规格（stage 读取/写入的字段名与类型）
- StageSpec：Stage IO 规格（name / reads / writes / description）
- stage：Stage IO 声明装饰器
- StageFn：Stage 函数类型别名
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import PipelineContext


# Stage 函数类型：接收 context，返回更新后的 context
StageFn = Callable[["PipelineContext"], "PipelineContext"]


@dataclass
class FieldSpec:
    """字段规格（RFC-003 DSP-3）。

    声明 stage 读取/写入的字段名与类型，作为 stage IO 契约的最小单位。
    """
    name: str
    type: str = "Any"
    required: bool = True
    description: str = ""


@dataclass
class StageSpec:
    """Stage IO 规格（RFC-003 DSP-3）。

    声明 stage 的 name / reads / writes / description，
    由 @stage 装饰器附加到 stage 函数的 _stage_spec 属性。
    """
    name: str
    reads: List[FieldSpec] = field(default_factory=list)
    writes: List[FieldSpec] = field(default_factory=list)
    description: str = ""


def stage(name: str, reads: List[str], writes: List[str], description: str = ""):
    """Stage IO 声明装饰器（RFC-003 DSP-3）。

    将 StageSpec 附加为函数属性 `_stage_spec`，不改变函数运行时行为。

    Args:
        name: stage 名（如 "validate"）
        reads: 该 stage 读取的 PipelineContext 字段名列表
        writes: 该 stage 写入的 PipelineContext 字段名列表
        description: stage 用途说明

    Returns:
        装饰器函数，将被装饰函数原样返回（仅附加 _stage_spec 属性）
    """
    def decorator(fn: StageFn) -> StageFn:
        fn._stage_spec = StageSpec(
            name=name,
            reads=[FieldSpec(name=n) for n in reads],
            writes=[FieldSpec(name=n) for n in writes],
            description=description,
        )
        return fn
    return decorator
