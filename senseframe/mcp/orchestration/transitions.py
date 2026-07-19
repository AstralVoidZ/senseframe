"""HATEOAS _transitions 计算（设计文档 0.6 节 L5 OPP）。

为给定状态返回合法的转换列表（含 suggested_tool 提示 + 数据前置条件）。

设计原则：
- advisory 优于 enforced：_transitions 是建议性的，不阻断 Agent 决策
- suggested_tool 提示下一个工具，但不强制
- prerequisites 列出执行该转换的数据前置条件（advisory）
- 幂等动作（如 `complete` on `Succeeded`）不出现在 _transitions 中（已达成终态）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from senseframe.mcp.models.pipeline_run import (
    PipelineRun,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_PENDING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)
from senseframe.mcp.views.pipeline import TransitionView

__all__ = ["TransitionDef", "get_transitions", "_TRANSITIONS_BY_STATE"]


@dataclass(frozen=True)
class TransitionDef:
    """静态转换定义（不携带运行时信息）。

    Attributes:
        action: 动作名（start/complete/fail/pause/resume/retry/skip）。
        target_state: 目标状态字符串。
        suggested_tool: 建议调用的下一个 tool（None 表示无建议）。
    """

    action: str
    target_state: str
    suggested_tool: Optional[str] = None


# 状态 × 合法动作映射（设计文档 0.6 节 5 状态 + 7 转换）。
# 终态（Succeeded）无转换，幂等动作（如 complete on Succeeded）不在此处声明。
_TRANSITIONS_BY_STATE: dict[str, list[TransitionDef]] = {
    STATE_PENDING: [
        TransitionDef(
            action="start",
            target_state=STATE_RUNNING,
            suggested_tool="senseframe_pipeline_advance",
        ),
        TransitionDef(
            action="skip",
            target_state=STATE_FAILED,
            suggested_tool=None,
        ),
    ],
    STATE_RUNNING: [
        TransitionDef(
            action="complete",
            target_state=STATE_SUCCEEDED,
            suggested_tool="senseframe_pipeline_advance",
        ),
        TransitionDef(
            action="fail",
            target_state=STATE_FAILED,
            suggested_tool=None,
        ),
        TransitionDef(
            action="pause",
            target_state=STATE_PAUSED,
            suggested_tool="senseframe_pipeline_pause",
        ),
    ],
    STATE_PAUSED: [
        TransitionDef(
            action="resume",
            target_state=STATE_RUNNING,
            suggested_tool="senseframe_pipeline_resume",
        ),
    ],
    STATE_SUCCEEDED: [],  # 终态，无转换
    STATE_FAILED: [
        TransitionDef(
            action="retry",
            target_state=STATE_RUNNING,
            suggested_tool="senseframe_pipeline_advance",
        ),
    ],
}


def get_transitions(
    state: str, run: PipelineRun | None = None
) -> list[TransitionView]:
    """返回当前状态下合法的转换列表（含 suggested_tool 提示）。

    advisory 模式：不强制 Agent 执行 suggested_tool，只提供建议。
    数据前置条件（prerequisites）由 run 的 completed_stages 推导。

    Args:
        state: 当前状态字符串（Pending/Running/Paused/Succeeded/Failed）。
        run: PipelineRun 实例（可选，用于推导 prerequisites）。

    Returns:
        TransitionView 列表（可能为空，如 Succeeded 终态）。
    """
    defs = _TRANSITIONS_BY_STATE.get(state, [])
    views: list[TransitionView] = []
    for d in defs:
        prerequisites = _derive_prerequisites(d.action, run)
        views.append(
            TransitionView(
                action=d.action,
                target_state=d.target_state,
                suggested_tool=d.suggested_tool,
                prerequisites=prerequisites,
            )
        )
    return views


def _derive_prerequisites(
    action: str, run: PipelineRun | None
) -> list[str]:
    """推导执行该动作的数据前置条件（advisory）。

    简化策略：
    - start：如果 run 有 stages，要求前置 stage 已完成（advisory）
    - complete：要求至少有一个 stage 已完成
    - retry：清空前置，直接允许
    - 其他动作：无前置

    Args:
        action: 动作名。
        run: PipelineRun 实例（None 时无前置）。

    Returns:
        前置条件描述字符串列表。
    """
    if run is None:
        return []
    prerequisites: list[str] = []
    if action == "start":
        # advisory：start 不需要前置 stage，但建议 completed_stages 为空
        # （Pending 状态下尚未开始任何 stage）
        if run.completed_stages:
            prerequisites.append(
                "completed_stages should be empty for start (advisory)"
            )
    elif action == "complete":
        # advisory：complete 要求至少有一个 stage 已完成
        if not run.completed_stages and run.stages:
            prerequisites.append(
                f"at least one stage must be completed before complete (e.g. '{run.stages[-1]}')"
            )
    elif action == "fail":
        # fail 不需要前置
        pass
    elif action == "pause":
        # pause 是 Running 状态的合法动作，无额外前置
        pass
    elif action == "resume":
        # resume 是 Paused 状态的合法动作，无额外前置
        pass
    elif action == "retry":
        # retry 清空前置
        pass
    elif action == "skip":
        # skip 是 Pending 状态的合法动作，标记失败但不执行
        pass
    return prerequisites
