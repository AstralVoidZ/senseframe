"""``senseframe_study_*`` 工具组（L4 SP 搜索协议）。

设计文档 0.3 节定义 7 个 study tool：
- senseframe_study_create  — 创建搜索 study（name + direction + search_space + sampler）
- senseframe_study_ask     — 采样下一个 trial（返回 TrialSpec）
- senseframe_study_tell    — 上报 trial 结果（value + intermediate_values + state + feedback）
- senseframe_study_get     — 查询 study 状态 + 最佳 trial
- senseframe_study_list    — 列出所有 study（cursor 分页）
- senseframe_study_compare — 多 study 对比（结构化对比表）
- senseframe_study_stop    — 停止 study

每个 tool 是 async 函数，签名参考 tools/pipeline.py：
- 使用 MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...)) 包装
- 异常通过 to_tool_error(exc) 转换为 ToolError
- 返回值是 FrozenModel 子类（在 views/study.py 中定义）

ToolAnnotations 矩阵（设计文档 0.4 节）：
- create:  false/false/false/true
- ask:     false/false/false/true
- tell:    false/false/true/true
- get:     true/false/true/false
- list:    true/false/true/false
- compare: true/false/true/false
- stop:    false/false/true/true

关键设计：包装现有 StudyManager（senseframe.search_protocol.StudyManager），
不重新实现 SP 协议。
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from senseframe.mcp.orchestration.study_manager import (
    get_default_manager as _get_default_manager,
    set_default_manager as _set_default_manager,
)
from senseframe.mcp.orchestration.study_transitions import (
    STUDY_STATE_RUNNING,
    STUDY_STATE_STOPPED,
    get_study_transitions,
)
from senseframe.mcp.pagination.cursor import (
    assert_fingerprint_matches,
    encode_cursor,
)
from senseframe.mcp.pagination.page import build_page, clamp_limit
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.pipeline import TransitionView
from senseframe.mcp.views.study import (
    StudyAskResponse,
    StudyCompareView,
    StudyCreateResponse,
    StudyListView,
    StudyTellResponse,
    StudyView,
    TrialView,
)
from senseframe.search_protocol import ParameterSpec, SearchSpace

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_study_create",
    "senseframe_study_ask",
    "senseframe_study_tell",
    "senseframe_study_get",
    "senseframe_study_list",
    "senseframe_study_compare",
    "senseframe_study_stop",
    "get_study_manager",
    "set_study_manager",
    "_study_stack",
]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_study_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


def get_study_manager():
    """返回进程级 StudyManager 单例。

    委托给 orchestration.study_manager.get_default_manager()，确保 tools/
    与 resources/ 共享同一个 StudyManager 实例。
    """
    return _get_default_manager()


def set_study_manager(mgr) -> None:
    """测试注入用：重置 StudyManager 单例。"""
    _set_default_manager(mgr)


# ============================================================
# 搜索空间转换辅助
# ============================================================


def _build_search_space(search_space_spec: dict | list | None) -> SearchSpace:
    """从 dict / list 构造 SearchSpace 域对象。

    支持的输入形式：
    - None → 空 SearchSpace
    - list[dict]：每个 dict 是 ParameterSpec 的字段
    - dict 含 "parameters" key：调用 SearchSpace.from_dict
    - dict 不含 "parameters" key：视为单参数 spec 包成 list

    Args:
        search_space_spec: 搜索空间规格。

    Returns:
        SearchSpace 实例。
    """
    if search_space_spec is None:
        return SearchSpace()
    if isinstance(search_space_spec, list):
        params = [ParameterSpec(**p) for p in search_space_spec]
        return SearchSpace(parameters=params)
    if isinstance(search_space_spec, dict):
        if "parameters" in search_space_spec:
            return SearchSpace.from_dict(search_space_spec)
        # 视为单参数 spec
        return SearchSpace(parameters=[ParameterSpec(**search_space_spec)])
    raise TypeError(
        f"search_space must be dict / list / None, got {type(search_space_spec).__name__}"
    )


# ============================================================
# Tool handlers
# ============================================================


async def senseframe_study_create(
    name: str,
    direction: str = "maximize",
    search_space: dict | list | None = None,
    sampler: str = "random",
    seed: int | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyCreateResponse:
    """创建搜索 Study（包装 StudyManager.create_study）。

    Args:
        name: Study 名称。
        direction: 优化方向（maximize / minimize）。
        search_space: 搜索空间（dict / list / None）。
        sampler: 采样器名（random / grid / asha / hyperband）。
        seed: Study 级种子（None 时使用全局随机）。
        ctx: MCP Context。

    Returns:
        StudyCreateResponse（含 study_id + transitions=[ask, stop]）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_study_create name={name} direction={direction} sampler={sampler}"
        )
    try:
        async with _study_stack.instrument("senseframe_study_create", ctx):
            manager = get_study_manager()
            ss = _build_search_space(search_space)
            study_id = manager.create_study(
                name=name,
                direction=direction,
                search_space=ss,
                sampler=sampler,
                seed=seed,
            )
            study = manager.get_study(study_id)
            transitions = get_study_transitions(STUDY_STATE_RUNNING)
            return StudyCreateResponse(
                study_id=study_id,
                name=study.name,
                direction=study.direction,
                sampler=study.sampler,
                created_at=study.created_at,
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_create failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_ask(
    study_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyAskResponse:
    """采样下一个 Trial（包装 StudyManager.ask）。

    Args:
        study_id: Study ID。
        ctx: MCP Context。

    Returns:
        StudyAskResponse（含 trial_id + params + transitions=[tell, ask]）。

    Raises:
        KeyError: study_id 不存在（→ ToolError study category）。
        RuntimeError: study 已停止。
    """
    if ctx:
        await ctx.info(f"senseframe_study_ask study_id={study_id}")
    try:
        async with _study_stack.instrument("senseframe_study_ask", ctx):
            manager = get_study_manager()
            trial = manager.ask(study_id)
            transitions = get_study_transitions(STUDY_STATE_RUNNING)
            return StudyAskResponse(
                trial_id=trial.trial_id,
                study_id=trial.study_id,
                params=dict(trial.params),
                datetime_start=trial.datetime_start,
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_ask failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_tell(
    trial_id: str,
    value: float,
    intermediate_values: dict[int, float] | None = None,
    state: str = "completed",
    feedback: dict[str, Any] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyTellResponse:
    """上报 Trial 结果（包装 StudyManager.tell）。

    Args:
        trial_id: Trial ID。
        value: Trial 目标值。
        intermediate_values: 中间值（epoch → metric，用于 ASHA/Hyperband 早停）。
        state: Trial 新状态（completed / failed / pruned）。
        feedback: 结构化反馈（如 {"status": "overfitting", ...}）。
        ctx: MCP Context。

    Returns:
        StudyTellResponse（含 trial_id + state + transitions=[ask, stop, get]）。

    Raises:
        KeyError: trial_id 不存在（→ ToolError study category）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_study_tell trial_id={trial_id} value={value} state={state}"
        )
    try:
        async with _study_stack.instrument("senseframe_study_tell", ctx):
            manager = get_study_manager()
            # tell 内部会从 _pending_trials 查找 trial 并获取 study_id
            # 提前查找以获取 study_id（用于响应）
            pending = manager._pending_trials.get(trial_id)  # type: ignore[attr-defined]
            study_id = pending.study_id if pending is not None else ""
            manager.tell(
                trial_id=trial_id,
                value=value,
                intermediate_values=intermediate_values,
                state=state,
                feedback=feedback,
            )
            transitions = get_study_transitions(STUDY_STATE_RUNNING)
            return StudyTellResponse(
                trial_id=trial_id,
                study_id=study_id,
                state=state,
                value=value,
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_tell failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_get(
    study_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyView:
    """查询 Study 状态 + 最佳 trial（包装 StudyManager.get_study / best_trial）。

    Args:
        study_id: Study ID。
        ctx: MCP Context。

    Returns:
        StudyView（含 n_trials / n_completed / best_value + _transitions）。

    Raises:
        KeyError: study_id 不存在（→ ToolError study category）。
    """
    if ctx:
        await ctx.info(f"senseframe_study_get study_id={study_id}")
    try:
        async with _study_stack.instrument("senseframe_study_get", ctx):
            manager = get_study_manager()
            study = manager.get_study(study_id)
            if study is None:
                raise KeyError(f"Study '{study_id}' not found")
            transitions = get_study_transitions(study.status)
            return StudyView.from_domain(study, manager, transitions=transitions)
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_get failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_list(
    cursor: str | None = None,
    limit: int = 50,
    filter_dict: dict[str, Any] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyListView:
    """列出所有 Study（cursor 分页）。

    Args:
        cursor: 不透明 cursor（来自上一次 list 的 next_cursor），None 表示首页。
        limit: 页大小（钳制到 [1, 200]）。
        filter_dict: 过滤字典（支持 {"status": "running"} 等值过滤）。
        ctx: MCP Context。

    Returns:
        StudyListView（含 items + next_cursor + total_count + limit）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_study_list cursor={'set' if cursor else 'none'} limit={limit}"
        )
    try:
        async with _study_stack.instrument("senseframe_study_list", ctx):
            manager = get_study_manager()
            clamped_limit = clamp_limit(limit)
            last_id = assert_fingerprint_matches(cursor, filter_dict)
            all_studies = manager.list_studies()
            # 应用 filter
            filtered = [s for s in all_studies if _matches_study_filter(s, filter_dict)]
            total_count = len(filtered)
            # 按 study_id 字典序排序
            filtered.sort(key=lambda s: s.study_id)
            # 应用 cursor
            if last_id is not None:
                filtered = [s for s in filtered if s.study_id > last_id]
            # limit+1 技巧
            peek = filtered[: clamped_limit + 1]
            has_more = len(peek) > clamped_limit
            items_studies = peek[:clamped_limit]
            # 构建 transitions 索引
            items_views: list[StudyView] = []
            for s in items_studies:
                transitions = get_study_transitions(s.status)
                items_views.append(
                    StudyView.from_domain(s, manager, transitions=transitions)
                )
            # 构建 page
            next_cursor: str | None = None
            if has_more and items_studies:
                next_cursor = encode_cursor(items_studies[-1].study_id, filter_dict)
            return StudyListView(
                items=items_views,
                next_cursor=next_cursor,
                total_count=total_count,
                limit=clamped_limit,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_list failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_compare(
    study_ids: list[str],
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyCompareView:
    """多 Study 对比（结构化对比表）。

    Args:
        study_ids: 参与对比的 Study ID 列表（至少 2 个）。
        ctx: MCP Context。

    Returns:
        StudyCompareView（含 studies + comparison_table + best_study_id）。
    """
    if ctx:
        await ctx.info(f"senseframe_study_compare study_ids={study_ids}")
    try:
        async with _study_stack.instrument("senseframe_study_compare", ctx):
            if not isinstance(study_ids, list) or len(study_ids) < 2:
                raise ValueError(
                    f"study_ids must be a list of at least 2 ids, got: {study_ids}"
                )
            manager = get_study_manager()
            study_views: list[StudyView] = []
            comparison_table: list[dict[str, Any]] = []
            best_study_id: str | None = None
            best_value: float | None = None
            for sid in study_ids:
                study = manager.get_study(sid)
                if study is None:
                    raise KeyError(f"Study '{sid}' not found")
                transitions = get_study_transitions(study.status)
                view = StudyView.from_domain(study, manager, transitions=transitions)
                study_views.append(view)
                comparison_table.append(
                    {
                        "study_id": view.study_id,
                        "name": view.name,
                        "direction": view.direction,
                        "sampler": view.sampler,
                        "status": view.status,
                        "n_trials": view.n_trials,
                        "n_completed": view.n_completed,
                        "best_value": view.best_value,
                        "best_trial_id": view.best_trial_id,
                    }
                )
                # 推导最佳 Study（方向感知）
                if view.best_value is not None:
                    if best_value is None:
                        best_value = view.best_value
                        best_study_id = view.study_id
                    else:
                        if view.direction == "maximize" and view.best_value > best_value:
                            best_value = view.best_value
                            best_study_id = view.study_id
                        elif view.direction == "minimize" and view.best_value < best_value:
                            best_value = view.best_value
                            best_study_id = view.study_id
            return StudyCompareView(
                studies=study_views,
                comparison_table=comparison_table,
                best_study_id=best_study_id,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_compare failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_study_stop(
    study_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyView:
    """停止 Study（包装 StudyManager.stop_study）。

    Args:
        study_id: Study ID。
        ctx: MCP Context。

    Returns:
        StudyView（status=stopped + _transitions=[get, compare]）。

    Raises:
        KeyError: study_id 不存在。
    """
    if ctx:
        await ctx.info(f"senseframe_study_stop study_id={study_id}")
    try:
        async with _study_stack.instrument("senseframe_study_stop", ctx):
            manager = get_study_manager()
            study = manager.get_study(study_id)
            if study is None:
                raise KeyError(f"Study '{study_id}' not found")
            manager.stop_study(study_id)
            study = manager.get_study(study_id)
            transitions = get_study_transitions(STUDY_STATE_STOPPED)
            return StudyView.from_domain(study, manager, transitions=transitions)
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_study_stop failed: {exc}")
        raise to_tool_error(exc)


# ============================================================
# 辅助函数
# ============================================================


def _matches_study_filter(study: Any, filter_dict: dict[str, Any] | None) -> bool:
    """简单等值过滤：filter_dict 中的所有 (k, v) 必须在 study 上匹配。

    支持的 filter key：
    - status: study.status == v
    - direction: study.direction == v
    - sampler: study.sampler == v
    - name: study.name == v
    """
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        if key == "status" and study.status != value:
            return False
        elif key == "direction" and study.direction != value:
            return False
        elif key == "sampler" and study.sampler != value:
            return False
        elif key == "name" and study.name != value:
            return False
    return True
