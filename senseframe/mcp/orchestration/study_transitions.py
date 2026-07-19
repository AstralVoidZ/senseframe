"""Study HATEOAS _transitions 计算（L4 SP 搜索协议）。

为 Study 状态返回合法的转换列表（含 suggested_tool 提示）。

设计原则（advisory 优于 enforced）：
- _transitions 是建议性的，不阻断 Agent 决策
- suggested_tool 提示下一个工具，但不强制
- 终态（stopped）无 ask 转换

Study 状态：
- running: [ask, stop, get, compare]
- stopped: [get, compare]（终态，无 ask）
"""

from __future__ import annotations

from senseframe.mcp.views.pipeline import TransitionView

__all__ = [
    "STUDY_STATE_RUNNING",
    "STUDY_STATE_STOPPED",
    "get_study_transitions",
    "STUDY_TRANSITIONS_BY_STATE",
]

STUDY_STATE_RUNNING = "running"
STUDY_STATE_STOPPED = "stopped"

# Study 状态 × 合法动作映射（advisory，不强制）
# running: 可 ask（采样下一个 trial）/ stop（停止）/ get（查询）/ compare（对比）
# stopped: 终态，仅可 get / compare
STUDY_TRANSITIONS_BY_STATE: dict[str, list[dict[str, str | None]]] = {
    STUDY_STATE_RUNNING: [
        {
            "action": "ask",
            "target_state": STUDY_STATE_RUNNING,
            "suggested_tool": "senseframe_study_ask",
        },
        {
            "action": "stop",
            "target_state": STUDY_STATE_STOPPED,
            "suggested_tool": "senseframe_study_stop",
        },
        {
            "action": "get",
            "target_state": STUDY_STATE_RUNNING,
            "suggested_tool": "senseframe_study_get",
        },
        {
            "action": "compare",
            "target_state": STUDY_STATE_RUNNING,
            "suggested_tool": "senseframe_study_compare",
        },
    ],
    STUDY_STATE_STOPPED: [
        {
            "action": "get",
            "target_state": STUDY_STATE_STOPPED,
            "suggested_tool": "senseframe_study_get",
        },
        {
            "action": "compare",
            "target_state": STUDY_STATE_STOPPED,
            "suggested_tool": "senseframe_study_compare",
        },
    ],
}


def get_study_transitions(state: str) -> list[TransitionView]:
    """返回 Study 当前状态下合法的转换列表（advisory）。

    Args:
        state: Study 状态字符串（running / stopped）。

    Returns:
        TransitionView 列表（advisory，可能为空）。
    """
    defs = STUDY_TRANSITIONS_BY_STATE.get(state, [])
    return [
        TransitionView(
            action=d["action"],  # type: ignore[arg-type]
            target_state=d["target_state"],  # type: ignore[arg-type]
            suggested_tool=d.get("suggested_tool"),
            prerequisites=[],
        )
        for d in defs
    ]
