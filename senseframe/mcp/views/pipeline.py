"""PipelineRun 系列 view model（L2 View，FrozenModel 子类）。

设计文档 0.6 节 + 0.4 节 HATEOAS：
- TransitionView：HATEOAS 转换建议（advisory，不强制 Agent）
- StageView：stage 状态视图
- PipelineRunView：run 状态视图，附加 _transitions
- PipelineRunListView：cursor 分页列表视图
- PipelineCreateResponse：创建响应视图
- PipelineAdvanceResponse：状态转移响应视图

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
- PipelineRunView.from_domain 接收 transitions 参数（由 tool 层从 orchestration 获取后传入）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from senseframe.mcp.models.pipeline_run import (
    PipelineRun,
    STATE_FAILED,
    STATE_PENDING,
    STATE_RUNNING,
)
from senseframe.mcp.views._base import FrozenModel, ViewError, _safe_get

__all__ = [
    "TransitionView",
    "StageView",
    "PipelineRunView",
    "PipelineRunListView",
    "PipelineCreateResponse",
    "PipelineAdvanceResponse",
]


class TransitionView(FrozenModel):
    """HATEOAS 转换建议（advisory）。

    Agent 收到 _transitions 后可程序化判断下一步动作 + suggested_tool，
    但不强制执行。

    Attributes:
        action: 合法动作名（start/complete/fail/pause/resume/retry/skip）。
        target_state: 该动作的目标状态（Running/Succeeded/Failed/Paused）。
        suggested_tool: 建议调用的下一个 tool（HATEOAS 提示，不强制）。
        prerequisites: 数据前置条件列表（advisory，如 "stage 'build' must be completed"）。
    """

    action: str
    target_state: str
    suggested_tool: Optional[str] = None
    prerequisites: list[str] = Field(default_factory=list)


class StageView(FrozenModel):
    """stage 状态视图（run 内的每个 stage）。

    Attributes:
        name: stage 名。
        state: stage 状态（pending/running/succeeded/failed/skipped）。
    """

    name: str
    state: str = "pending"


class PipelineRunView(FrozenModel):
    """PipelineRun 公共 JSON 契约（含 _transitions HATEOAS）。

    Attributes:
        run_id: PipelineRun ID。
        state: 当前状态（Pending/Running/Paused/Succeeded/Failed）。
        stages: stage 视图列表。
        created_at: ISO8601 创建时间。
        updated_at: ISO8601 最后更新时间。
        completed_stages: 已完成的 stage 名列表。
        failed_stage: 失败的 stage 名（None 表示未失败）。
        error_message: 失败原因（None 表示未失败）。
        transitions: HATEOAS 转换建议列表（advisory，对应 _transitions 字段）。
    """

    run_id: str
    state: str
    stages: list[StageView]
    created_at: str
    updated_at: str
    completed_stages: list[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
    trial_id: Optional[str] = None
    transitions: list[TransitionView] = Field(default_factory=list)

    @classmethod
    def from_domain(
        cls,
        run: PipelineRun,
        transitions: list[TransitionView] | None = None,
    ) -> PipelineRunView:
        """从 PipelineRun 域对象投影到 view，附加 _transitions。

        分层契约：调用方（tool 层）从 orchestration.get_transitions 获取
        transitions 列表后传入，view 层不直接调用 orchestration。

        Args:
            run: PipelineRun 域对象。
            transitions: HATEOAS 转换建议列表（可选，None 表示空）。

        Returns:
            PipelineRunView 实例。

        Raises:
            ViewError: 输入不是 PipelineRun 或字段缺失。
        """
        try:
            # 推导每个 stage 的状态
            stages_view: list[StageView] = []
            for stage_name in _safe_get(run, "stages", []) or []:
                stage_state = _derive_stage_state(
                    stage_name=stage_name,
                    run_state=_safe_get(run, "state", STATE_PENDING),
                    completed_stages=_safe_get(run, "completed_stages", []) or [],
                    failed_stage=_safe_get(run, "failed_stage", None),
                )
                stages_view.append(StageView(name=stage_name, state=stage_state))

            return cls(
                run_id=_safe_get(run, "run_id"),
                state=_safe_get(run, "state"),
                stages=stages_view,
                created_at=_safe_get(run, "created_at"),
                updated_at=_safe_get(run, "updated_at"),
                completed_stages=list(_safe_get(run, "completed_stages", []) or []),
                failed_stage=_safe_get(run, "failed_stage", None),
                error_message=_safe_get(run, "error_message", None),
                trial_id=_safe_get(run, "trial_id", None),
                transitions=list(transitions or []),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(
                f"PipelineRunView.from_domain: invalid input: {exc}"
            ) from exc


class PipelineRunListView(FrozenModel):
    """PipelineRun 列表视图（cursor 分页）。

    Attributes:
        items: PipelineRunView 列表。
        next_cursor: 下一页 cursor（None 表示无更多）。
        total_count: 总数（不受分页影响）。
        limit: 钳制后的页大小。
    """

    items: list[PipelineRunView]
    next_cursor: Optional[str] = None
    total_count: int = 0
    limit: int = 50

    @classmethod
    def from_page(
        cls,
        page: Any,
        transitions_by_run: dict[str, list[TransitionView]] | None = None,
    ) -> PipelineRunListView:
        """从 Page[PipelineRun] 投影到 view。

        Args:
            page: Page[PipelineRun] 实例（items/next_cursor/total_count/limit）。
            transitions_by_run: 每个 run_id 对应的 transitions 列表（由 tool 层注入）。

        Returns:
            PipelineRunListView 实例。
        """
        try:
            transitions_by_run = transitions_by_run or {}
            items = [
                PipelineRunView.from_domain(
                    run,
                    transitions=transitions_by_run.get(
                        _safe_get(run, "run_id"), []
                    ),
                )
                for run in (_safe_get(page, "items", []) or [])
            ]
            return cls(
                items=items,
                next_cursor=_safe_get(page, "next_cursor"),
                total_count=_safe_get(page, "total_count", 0),
                limit=_safe_get(page, "limit", 50),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(
                f"PipelineRunListView.from_page: invalid input: {exc}"
            ) from exc


class PipelineCreateResponse(FrozenModel):
    """``senseframe_pipeline_create`` 响应视图。

    Attributes:
        run_id: 新建的 PipelineRun ID。
        state: 初始状态（Pending）。
        created_at: ISO8601 创建时间。
        transitions: 当前状态下合法的转换建议。
    """

    run_id: str
    state: str
    created_at: str
    transitions: list[TransitionView] = Field(default_factory=list)


class PipelineAdvanceResponse(FrozenModel):
    """``senseframe_pipeline_advance`` 响应视图。

    Attributes:
        run_id: PipelineRun ID。
        previous_state: 转换前的状态。
        new_state: 转换后的状态（幂等短路时 = previous_state）。
        action: 执行的动作。
        transitions: 新状态下合法的转换建议。
    """

    run_id: str
    previous_state: str
    new_state: str
    action: str
    transitions: list[TransitionView] = Field(default_factory=list)


# ============================================================
# 辅助函数
# ============================================================


def _derive_stage_state(
    stage_name: str,
    run_state: str,
    completed_stages: list[str],
    failed_stage: str | None,
) -> str:
    """推导 stage 的运行时状态（advisory）。

    简化策略：
    - stage 在 completed_stages 中 → "succeeded"
    - stage == failed_stage → "failed"
    - run 已 Succeeded → 所有未在 completed_stages 的 stage 视为 "succeeded"
    - run 已 Failed → 所有未失败 stage 视为 "skipped"（advisory）
    - run 在 Pending → 所有 stage "pending"
    - run 在 Running → 第一个未完成的 stage 视为 "running"，其余 "pending"
    """
    if stage_name in completed_stages:
        return "succeeded"
    if failed_stage is not None and stage_name == failed_stage:
        return "failed"
    if run_state == STATE_PENDING:
        return "pending"
    if run_state == STATE_FAILED:
        return "skipped"
    if run_state == STATE_RUNNING:
        # 第一个未完成的 stage 视为 running（advisory，简化模型）
        return "running"
    return "pending"
