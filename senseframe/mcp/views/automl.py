"""AutoML 系列 view model（L4 SP 搜索协议，FrozenModel 子类）。

设计文档 0.7.3 节 + 0.4 节 HATEOAS：
- AutoMLStageView：stage 状态视图
- AutoMLPipelineView：pipeline 状态视图，附加 _transitions
- AutoMLPipelineListView：cursor 分页列表视图
- AutoMLCreateResponse：创建响应视图
- AutoMLAdvanceResponse：状态转移响应视图

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
- AutoMLPipelineView.from_domain 接收 transitions 参数（由 tool 层注入）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from senseframe.mcp.views._base import FrozenModel, ViewError, _safe_get
from senseframe.mcp.views.pipeline import TransitionView

__all__ = [
    "AutoMLStageView",
    "AutoMLPipelineView",
    "AutoMLPipelineListView",
    "AutoMLCreateResponse",
    "AutoMLAdvanceResponse",
]


class AutoMLStageView(FrozenModel):
    """stage 状态视图（pipeline 内的每个 stage）。

    Attributes:
        name: stage 名（nas / hpo / autoaugment）。
        state: stage 状态（pending / running / completed / failed）。
        study_id: 该 stage 对应的 Study ID（None 表示未创建）。
    """

    name: str
    state: str = "pending"
    study_id: Optional[str] = None


class AutoMLPipelineView(FrozenModel):
    """AutoML Pipeline 公共 JSON 契约（含 _transitions HATEOAS）。

    Attributes:
        pipeline_id: Pipeline ID。
        state: 当前状态（Pending/Running/Paused/Succeeded/Failed）。
        stages: stage 视图列表。
        current_stage_index: 当前 stage 索引（-1 表示未启动）。
        created_at: ISO8601 创建时间。
        updated_at: ISO8601 最后更新时间。
        completed_stages: 已完成的 stage 名列表。
        failed_stage: 失败的 stage 名（None 表示未失败）。
        error_message: 失败原因（None 表示未失败）。
        transitions: HATEOAS 转换建议列表（advisory）。
    """

    pipeline_id: str
    state: str
    stages: list[AutoMLStageView]
    current_stage_index: int = -1
    created_at: str
    updated_at: str
    completed_stages: list[str] = Field(default_factory=list)
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
    transitions: list[TransitionView] = Field(default_factory=list)

    @classmethod
    def from_domain(
        cls,
        pipeline: Any,
        transitions: list[TransitionView] | None = None,
    ) -> AutoMLPipelineView:
        """从 AutoMLPipeline 域对象投影到 view。

        分层契约：调用方（tool 层）从 automl_transitions.get_automl_transitions
        获取 transitions 列表后传入，view 层不直接调用 orchestration。

        Args:
            pipeline: AutoMLPipeline 域对象。
            transitions: HATEOAS 转换建议列表（可选，None 表示空）。

        Returns:
            AutoMLPipelineView 实例。
        """
        try:
            stages_list = _safe_get(pipeline, "stages", []) or []
            completed_stages = list(_safe_get(pipeline, "completed_stages", []) or [])
            failed_stage = _safe_get(pipeline, "failed_stage", None)
            current_idx = _safe_get(pipeline, "current_stage_index", -1)
            study_ids = list(_safe_get(pipeline, "study_ids", []) or [])

            stage_views: list[AutoMLStageView] = []
            for i, name in enumerate(stages_list):
                if i < current_idx:
                    state = "completed"
                elif i == current_idx:
                    state = "running"
                else:
                    state = "pending"
                # 若 stage 在 completed_stages 中，标记为 completed
                if name in completed_stages:
                    state = "completed"
                if failed_stage == name:
                    state = "failed"
                study_id = study_ids[i] if i < len(study_ids) else None
                stage_views.append(
                    AutoMLStageView(name=name, state=state, study_id=study_id)
                )

            return cls(
                pipeline_id=_safe_get(pipeline, "pipeline_id"),
                state=_safe_get(pipeline, "state", "Pending") or "Pending",
                stages=stage_views,
                current_stage_index=current_idx,
                created_at=_safe_get(pipeline, "created_at", "") or "",
                updated_at=_safe_get(pipeline, "updated_at", "") or "",
                completed_stages=completed_stages,
                failed_stage=failed_stage,
                error_message=_safe_get(pipeline, "error_message", None),
                transitions=list(transitions or []),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(f"AutoMLPipelineView.from_domain: invalid input: {exc}") from exc


class AutoMLPipelineListView(FrozenModel):
    """AutoML Pipeline 列表视图（cursor 分页）。

    Attributes:
        items: AutoMLPipelineView 列表。
        next_cursor: 下一页 cursor（None 表示无更多）。
        total_count: 总数（不受分页影响）。
        limit: 钳制后的页大小。
    """

    items: list[AutoMLPipelineView]
    next_cursor: Optional[str] = None
    total_count: int = 0
    limit: int = 50


class AutoMLCreateResponse(FrozenModel):
    """``senseframe_automl_create`` 响应视图。

    Attributes:
        pipeline_id: 新建的 Pipeline ID。
        stages: stage 名列表。
        state: 初始状态（Pending）。
        created_at: ISO8601 创建时间。
        transitions: 当前状态下合法的转换建议（advisory）。
    """

    pipeline_id: str
    stages: list[str]
    state: str
    created_at: str
    transitions: list[TransitionView] = Field(default_factory=list)


class AutoMLAdvanceResponse(FrozenModel):
    """``senseframe_automl_advance`` 响应视图。

    Attributes:
        pipeline_id: Pipeline ID。
        state: 新状态。
        current_stage_index: 当前 stage 索引。
        completed_stages: 已完成的 stage 名列表。
        transitions: 下一步合法的转换建议（advisory）。
    """

    pipeline_id: str
    state: str
    current_stage_index: int
    completed_stages: list[str] = Field(default_factory=list)
    transitions: list[TransitionView] = Field(default_factory=list)
