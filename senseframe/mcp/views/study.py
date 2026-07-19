"""Study 系列 view model（L4 SP 搜索协议，FrozenModel 子类）。

设计文档 0.3 节 L4 SP + 0.4 节 HATEOAS：
- TrialView：Trial 状态视图（含 params / state / value / intermediate_values）
- StudyView：Study 状态视图（含 _transitions HATEOAS）
- StudyCreateResponse / StudyAskResponse / StudyTellResponse：tool 响应视图
- StudyListView：cursor 分页列表视图
- StudyCompareView：多 Study 对比视图

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
- StudyView.from_domain 接收 transitions 参数（由 tool 层注入）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from senseframe.mcp.views._base import FrozenModel, ViewError, _safe_get
from senseframe.mcp.views.pipeline import TransitionView

__all__ = [
    "TrialView",
    "StudyView",
    "StudyCreateResponse",
    "StudyAskResponse",
    "StudyTellResponse",
    "StudyListView",
    "StudyCompareView",
]


class TrialView(FrozenModel):
    """Trial 公共 JSON 契约（SP-2 Ask/Tell 结果）。

    Attributes:
        trial_id: Trial ID。
        study_id: 所属 Study ID。
        params: 采样参数（如 {"lr": 0.001, "batch_size": 32}）。
        state: Trial 状态（pending/running/completed/pruned/failed）。
        value: Trial 目标值（None 表示未完成）。
        intermediate_values: 中间值（epoch → metric，用于 ASHA/Hyperband 早停）。
        datetime_start: ISO8601 开始时间。
        datetime_complete: ISO8601 完成时间（None 表示未完成）。
        feedback: 结构化反馈（如 {"status": "overfitting", ...}）。
    """

    trial_id: str
    study_id: str
    params: dict[str, Any]
    state: str
    value: Optional[float] = None
    intermediate_values: dict[str, float] = Field(default_factory=dict)
    datetime_start: str = ""
    datetime_complete: Optional[str] = None
    feedback: Optional[dict[str, Any]] = None

    @classmethod
    def from_domain(cls, trial: Any) -> TrialView:
        """从 TrialSpec / TrialResult 域对象投影到 view。

        Args:
            trial: TrialSpec（Ask 结果）或 TrialResult（Tell / get_trial 结果）。

        Returns:
            TrialView 实例。

        Raises:
            ViewError: 输入不是 TrialSpec / TrialResult 或字段缺失。
        """
        try:
            params = dict(_safe_get(trial, "params", {}) or {})
            state = _safe_get(trial, "state", "running") or "running"
            # TrialSpec 没有 value / intermediate_values / datetime_complete / feedback
            value = _safe_get(trial, "value", None)
            iv_raw = _safe_get(trial, "intermediate_values", {}) or {}
            # intermediate_values 的 key 可能是 int（TrialResult）或 str（dict）
            # 统一转为 str key 以满足 FrozenModel schema
            intermediate_values: dict[str, float] = {}
            if isinstance(iv_raw, dict):
                for k, v in iv_raw.items():
                    intermediate_values[str(k)] = float(v)
            datetime_start = _safe_get(trial, "datetime_start", "") or ""
            datetime_complete = _safe_get(trial, "datetime_complete", None)
            feedback = _safe_get(trial, "feedback", None)

            return cls(
                trial_id=_safe_get(trial, "trial_id"),
                study_id=_safe_get(trial, "study_id"),
                params=params,
                state=state,
                value=value,
                intermediate_values=intermediate_values,
                datetime_start=datetime_start,
                datetime_complete=datetime_complete,
                feedback=feedback,
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(f"TrialView.from_domain: invalid input: {exc}") from exc


class StudyView(FrozenModel):
    """Study 公共 JSON 契约（含 _transitions HATEOAS）。

    Attributes:
        study_id: Study ID。
        name: Study 名称。
        direction: 优化方向（maximize / minimize）。
        sampler: 采样器名（random / grid / asha / hyperband）。
        status: Study 状态（running / stopped）。
        created_at: ISO8601 创建时间。
        completed_at: ISO8601 完成时间（None 表示未停止）。
        n_trials: 总 trial 数。
        n_completed: 已完成 trial 数。
        best_value: 最佳 trial 的目标值（None 表示无已完成 trial）。
        best_trial_id: 最佳 trial ID（None 表示无已完成 trial）。
        transitions: HATEOAS 转换建议列表（advisory）。
    """

    study_id: str
    name: str
    direction: str
    sampler: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    n_trials: int = 0
    n_completed: int = 0
    best_value: Optional[float] = None
    best_trial_id: Optional[str] = None
    transitions: list[TransitionView] = Field(default_factory=list)

    @classmethod
    def from_domain(
        cls,
        study: Any,
        manager: Any,
        transitions: list[TransitionView] | None = None,
    ) -> StudyView:
        """从 StudySpec 域对象 + StudyManager 投影到 view。

        分层契约：调用方（tool 层）从 study_transitions.get_study_transitions
        获取 transitions 列表后传入，view 层不直接调用 orchestration。

        Args:
            study: StudySpec 域对象。
            manager: StudyManager 实例（用于查询 n_trials / best_trial）。
            transitions: HATEOAS 转换建议列表（可选，None 表示空）。

        Returns:
            StudyView 实例。
        """
        try:
            study_id = _safe_get(study, "study_id")
            status = _safe_get(study, "status", "running") or "running"
            # 通过 manager 查询 trial 统计
            n_trials = 0
            n_completed = 0
            best_value: Optional[float] = None
            best_trial_id: Optional[str] = None
            if manager is not None:
                try:
                    trials = manager.list_trials(study_id)
                    n_trials = len(trials)
                    n_completed = sum(
                        1 for t in trials if t.state in ("completed",)
                    )
                except Exception:
                    trials = []
                try:
                    best = manager.best_trial(study_id)
                    if best is not None:
                        best_value = best.value
                        best_trial_id = best.trial_id
                except Exception:
                    pass

            return cls(
                study_id=study_id,
                name=_safe_get(study, "name", "") or "",
                direction=_safe_get(study, "direction", "maximize") or "maximize",
                sampler=_safe_get(study, "sampler", "random") or "random",
                status=status,
                created_at=_safe_get(study, "created_at", "") or "",
                completed_at=_safe_get(study, "completed_at", None) or None,
                n_trials=n_trials,
                n_completed=n_completed,
                best_value=best_value,
                best_trial_id=best_trial_id,
                transitions=list(transitions or []),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(f"StudyView.from_domain: invalid input: {exc}") from exc


class StudyCreateResponse(FrozenModel):
    """``senseframe_study_create`` 响应视图。

    Attributes:
        study_id: 新建的 Study ID。
        name: Study 名称。
        direction: 优化方向。
        sampler: 采样器名。
        created_at: ISO8601 创建时间。
        transitions: 当前状态下合法的转换建议（advisory）。
    """

    study_id: str
    name: str
    direction: str
    sampler: str
    created_at: str
    transitions: list[TransitionView] = Field(default_factory=list)


class StudyAskResponse(FrozenModel):
    """``senseframe_study_ask`` 响应视图。

    Attributes:
        trial_id: 新采样的 Trial ID。
        study_id: 所属 Study ID。
        params: 采样参数。
        datetime_start: ISO8601 开始时间。
        transitions: 下一步合法的转换建议（advisory）。
    """

    trial_id: str
    study_id: str
    params: dict[str, Any]
    datetime_start: str
    transitions: list[TransitionView] = Field(default_factory=list)


class StudyTellResponse(FrozenModel):
    """``senseframe_study_tell`` 响应视图。

    Attributes:
        trial_id: 上报结果的 Trial ID。
        study_id: 所属 Study ID。
        state: Trial 新状态（completed / failed / pruned）。
        value: 上报的目标值（None 表示失败 / 剪枝）。
        transitions: 下一步合法的转换建议（advisory）。
    """

    trial_id: str
    study_id: str
    state: str
    value: Optional[float] = None
    transitions: list[TransitionView] = Field(default_factory=list)


class StudyListView(FrozenModel):
    """Study 列表视图（cursor 分页）。

    Attributes:
        items: StudyView 列表。
        next_cursor: 下一页 cursor（None 表示无更多）。
        total_count: 总数（不受分页影响）。
        limit: 钳制后的页大小。
    """

    items: list[StudyView]
    next_cursor: Optional[str] = None
    total_count: int = 0
    limit: int = 50


class StudyCompareView(FrozenModel):
    """多 Study 对比视图（结构化对比表）。

    Attributes:
        studies: 参与对比的 StudyView 列表。
        comparison_table: 结构化对比表（每个 entry 含 study_id / best_value /
            n_trials / n_completed / status 等字段）。
        best_study_id: 最佳 Study ID（None 表示无可用数据）。
    """

    studies: list[StudyView]
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)
    best_study_id: Optional[str] = None
