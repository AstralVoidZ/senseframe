"""AutoML Pipeline HATEOAS _transitions 计算（L4 SP 搜索协议）。

为 AutoML Pipeline 状态返回合法的转换列表（含 suggested_tool 提示）。

设计原则（advisory 优于 enforced）：
- _transitions 是建议性的，不阻断 Agent 决策
- suggested_tool 提示下一个工具，但不强制

AutoML Pipeline 状态：
- Pending: [start]
- Running: [complete, fail, pause, get]
- Paused:  [resume, get]
- Succeeded: [get]（终态）
- Failed:  [retry, get]
"""

from __future__ import annotations

from senseframe.mcp.orchestration.automl_orchestrator import (
    AUTOML_STATE_FAILED,
    AUTOML_STATE_PAUSED,
    AUTOML_STATE_PENDING,
    AUTOML_STATE_RUNNING,
    AUTOML_STATE_SUCCEEDED,
)
from senseframe.mcp.views.pipeline import TransitionView

__all__ = [
    "get_automl_transitions",
    "AUTOML_TRANSITIONS_BY_STATE",
]

# AutoML Pipeline 状态 × 合法动作映射（advisory，不强制）
AUTOML_TRANSITIONS_BY_STATE: dict[str, list[dict[str, str | None]]] = {
    AUTOML_STATE_PENDING: [
        {
            "action": "start",
            "target_state": AUTOML_STATE_RUNNING,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "get",
            "target_state": AUTOML_STATE_PENDING,
            "suggested_tool": "senseframe_automl_get",
        },
    ],
    AUTOML_STATE_RUNNING: [
        {
            "action": "complete",
            "target_state": AUTOML_STATE_RUNNING,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "fail",
            "target_state": AUTOML_STATE_FAILED,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "pause",
            "target_state": AUTOML_STATE_PAUSED,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "get",
            "target_state": AUTOML_STATE_RUNNING,
            "suggested_tool": "senseframe_automl_get",
        },
    ],
    AUTOML_STATE_PAUSED: [
        {
            "action": "resume",
            "target_state": AUTOML_STATE_RUNNING,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "get",
            "target_state": AUTOML_STATE_PAUSED,
            "suggested_tool": "senseframe_automl_get",
        },
    ],
    AUTOML_STATE_SUCCEEDED: [
        {
            "action": "get",
            "target_state": AUTOML_STATE_SUCCEEDED,
            "suggested_tool": "senseframe_automl_get",
        },
    ],
    AUTOML_STATE_FAILED: [
        {
            "action": "retry",
            "target_state": AUTOML_STATE_RUNNING,
            "suggested_tool": "senseframe_automl_advance",
        },
        {
            "action": "get",
            "target_state": AUTOML_STATE_FAILED,
            "suggested_tool": "senseframe_automl_get",
        },
    ],
}


def get_automl_transitions(state: str) -> list[TransitionView]:
    """返回 AutoML Pipeline 当前状态下合法的转换列表（advisory）。

    Args:
        state: 状态字符串（Pending/Running/Paused/Succeeded/Failed）。

    Returns:
        TransitionView 列表（advisory，可能为空）。
    """
    defs = AUTOML_TRANSITIONS_BY_STATE.get(state, [])
    return [
        TransitionView(
            action=d["action"],  # type: ignore[arg-type]
            target_state=d["target_state"],  # type: ignore[arg-type]
            suggested_tool=d.get("suggested_tool"),
            prerequisites=[],
        )
        for d in defs
    ]
