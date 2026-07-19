"""``senseframe_automl_*`` 工具组（L4 SP 搜索协议 + AutoMLOrchestrator）。

设计文档 0.7.3 节定义 3 个 automl tool：
- senseframe_automl_create  — 创建 AutoML 流水线（config + stages）
- senseframe_automl_advance — 推进流水线（action=start/complete/fail/pause/resume/retry）
- senseframe_automl_get     — 查询流水线状态（含 _transitions HATEOAS）

每个 tool 是 async 函数，签名参考 tools/pipeline.py：
- 使用 MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...)) 包装
- 异常通过 to_tool_error(exc) 转换为 ToolError
- 返回值是 FrozenModel 子类（在 views/automl.py 中定义）

ToolAnnotations 矩阵（设计文档 0.4 节）：
- create:  false/false/false/true
- advance: false/false/true/true
- get:     true/false/true/false
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
from senseframe.mcp.orchestration.automl_orchestrator import (
    AutoMLOrchestrator,
    get_default_orchestrator as _get_default_orchestrator,
    set_default_orchestrator as _set_default_orchestrator,
)
from senseframe.mcp.orchestration.automl_transitions import get_automl_transitions
from senseframe.mcp.pagination.cursor import (
    assert_fingerprint_matches,
    encode_cursor,
)
from senseframe.mcp.pagination.page import clamp_limit
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.automl import (
    AutoMLAdvanceResponse,
    AutoMLCreateResponse,
    AutoMLPipelineListView,
    AutoMLPipelineView,
)
from senseframe.mcp.views.pipeline import TransitionView

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_automl_create",
    "senseframe_automl_advance",
    "senseframe_automl_get",
    "senseframe_automl_list",
    "get_automl_orchestrator",
    "set_automl_orchestrator",
    "_automl_stack",
]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_automl_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


def get_automl_orchestrator() -> AutoMLOrchestrator:
    """返回进程级 AutoMLOrchestrator 单例。"""
    return _get_default_orchestrator()


def set_automl_orchestrator(orch: AutoMLOrchestrator | None) -> None:
    """测试注入用：重置 AutoMLOrchestrator 单例。"""
    _set_default_orchestrator(orch)


# ============================================================
# Tool handlers
# ============================================================


async def senseframe_automl_create(
    config: dict[str, Any],
    stages: list[str],
    ctx: Context[Any, Any, Any] | None = None,
) -> AutoMLCreateResponse:
    """创建 AutoML 流水线（声明式：接受 config + stages）。

    Args:
        config: ExperimentConfig.model_dump() 的 dict。
        stages: stage 名列表，元素必须是 "nas" / "hpo" / "autoaugment"。
            顺序决定执行流（如 ["nas", "hpo", "autoaugment"]）。
        ctx: MCP Context。

    Returns:
        AutoMLCreateResponse（含 pipeline_id + state=Pending + transitions=[start, get]）。

    Raises:
        ValueError: stages 为空或含非法 stage 名。
    """
    if ctx:
        await ctx.info(
            f"senseframe_automl_create stages={stages}"
        )
    try:
        async with _automl_stack.instrument("senseframe_automl_create", ctx):
            orch = _get_default_orchestrator()
            pipeline_id = orch.create_pipeline(
                config=config,
                stages=stages,
            )
            pipeline = orch.get(pipeline_id)
            transitions = get_automl_transitions(pipeline.state)
            return AutoMLCreateResponse(
                pipeline_id=pipeline.pipeline_id,
                stages=list(pipeline.stages),
                state=pipeline.state,
                created_at=pipeline.created_at,
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_automl_create failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_automl_advance(
    pipeline_id: str,
    action: str,
    study_id: str | None = None,
    error_message: str | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> AutoMLAdvanceResponse:
    """推进 AutoML 流水线状态（单一状态变更入口，幂等）。

    Args:
        pipeline_id: Pipeline ID。
        action: 动作名（start/complete/fail/pause/resume/retry）。
        study_id: complete 时可附带当前 stage 对应的 Study ID（记录到 study_ids）。
        error_message: fail 时可附带失败原因。
        ctx: MCP Context。

    Returns:
        AutoMLAdvanceResponse（含 pipeline_id + state + current_stage_index + transitions）。

    Raises:
        KeyError: pipeline_id 不存在（→ ToolError study category）。
        IllegalTransition: (state, action) 不是合法转换（→ ToolError pipeline category）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_automl_advance pipeline_id={pipeline_id} action={action}"
        )
    try:
        async with _automl_stack.instrument("senseframe_automl_advance", ctx):
            orch = _get_default_orchestrator()
            pipeline = orch.advance(
                pipeline_id=pipeline_id,
                action=action,
                study_id=study_id,
                error_message=error_message,
            )
            transitions = get_automl_transitions(pipeline.state)
            return AutoMLAdvanceResponse(
                pipeline_id=pipeline.pipeline_id,
                state=pipeline.state,
                current_stage_index=pipeline.current_stage_index,
                completed_stages=list(pipeline.completed_stages),
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_automl_advance failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_automl_get(
    pipeline_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> AutoMLPipelineView:
    """查询 AutoML 流水线状态（含 _transitions HATEOAS）。

    Args:
        pipeline_id: Pipeline ID。
        ctx: MCP Context。

    Returns:
        AutoMLPipelineView（含 stages + current_stage_index + _transitions）。

    Raises:
        KeyError: pipeline_id 不存在。
    """
    if ctx:
        await ctx.info(f"senseframe_automl_get pipeline_id={pipeline_id}")
    try:
        async with _automl_stack.instrument("senseframe_automl_get", ctx):
            orch = _get_default_orchestrator()
            pipeline = orch.get(pipeline_id)
            transitions = get_automl_transitions(pipeline.state)
            return AutoMLPipelineView.from_domain(pipeline, transitions=transitions)
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_automl_get failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_automl_list(
    cursor: str | None = None,
    limit: int = 50,
    filter_dict: dict[str, Any] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> AutoMLPipelineListView:
    """列出所有 AutoML 流水线（cursor 分页）。

    Args:
        cursor: 不透明 cursor，None 表示首页。
        limit: 页大小（钳制到 [1, 200]）。
        filter_dict: 过滤字典（支持 {"state": "Running"} 等值过滤）。
        ctx: MCP Context。

    Returns:
        AutoMLPipelineListView（含 items + next_cursor + total_count + limit）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_automl_list cursor={'set' if cursor else 'none'} limit={limit}"
        )
    try:
        async with _automl_stack.instrument("senseframe_automl_list", ctx):
            orch = _get_default_orchestrator()
            clamped_limit = clamp_limit(limit)
            last_id = assert_fingerprint_matches(cursor, filter_dict)
            all_pipelines = orch.list_pipelines()
            # 应用 filter
            filtered = [
                p for p in all_pipelines
                if _matches_automl_filter(p, filter_dict)
            ]
            total_count = len(filtered)
            # 按 pipeline_id 字典序排序
            filtered.sort(key=lambda p: p.pipeline_id)
            # 应用 cursor
            if last_id is not None:
                filtered = [p for p in filtered if p.pipeline_id > last_id]
            # limit+1 技巧
            peek = filtered[: clamped_limit + 1]
            has_more = len(peek) > clamped_limit
            items_pipelines = peek[:clamped_limit]
            # 构建 view 列表
            items_views: list[AutoMLPipelineView] = []
            for p in items_pipelines:
                transitions = get_automl_transitions(p.state)
                items_views.append(
                    AutoMLPipelineView.from_domain(p, transitions=transitions)
                )
            # 构建 next_cursor
            next_cursor: str | None = None
            if has_more and items_pipelines:
                next_cursor = encode_cursor(
                    items_pipelines[-1].pipeline_id, filter_dict
                )
            return AutoMLPipelineListView(
                items=items_views,
                next_cursor=next_cursor,
                total_count=total_count,
                limit=clamped_limit,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_automl_list failed: {exc}")
        raise to_tool_error(exc)


# ============================================================
# 辅助函数
# ============================================================


def _matches_automl_filter(
    pipeline: Any, filter_dict: dict[str, Any] | None
) -> bool:
    """简单等值过滤：filter_dict 中的所有 (k, v) 必须在 pipeline 上匹配。

    支持的 filter key：
    - state: pipeline.state == v
    - failed_stage: pipeline.failed_stage == v
    """
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        if key == "state" and pipeline.state != value:
            return False
        elif key == "failed_stage" and pipeline.failed_stage != value:
            return False
    return True
