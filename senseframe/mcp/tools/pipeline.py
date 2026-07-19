"""``senseframe_pipeline_*`` 工具组（L5 OPP 操作协议）。

设计文档 0.3 节定义 7 个 pipeline tool：
- senseframe_pipeline_create  — 创建 PipelineRun（config + stages）
- senseframe_pipeline_advance — 推进 run（action=start/complete/fail/retry/skip/pause/resume）
- senseframe_pipeline_run     — 执行完整 pipeline（黑盒，阻塞，调用 run_pipeline）
- senseframe_pipeline_get     — 查询 run 状态（含 _transitions HATEOAS）
- senseframe_pipeline_list    — 列出所有 run（cursor 分页）
- senseframe_pipeline_pause   — 暂停 run（idempotent 短路）
- senseframe_pipeline_resume  — 恢复 run

每个 tool 是 async 函数，签名参考 pipeflow tool_dispatch.py：
- 使用 MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...)) 包装
- 异常通过 to_tool_error(exc) 转换为 ToolError
- 返回值是 FrozenModel 子类（在 views/pipeline.py 中定义）

ToolAnnotations 矩阵（设计文档 0.4 节）：
- create:  false/false/false/true
- advance: false/false/true/true
- run:     false/false/false/true
- get:     true/false/true/false
- list:    true/false/true/false
- pause:   false/false/true/true
- resume:  false/false/true/true
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from mcp.server.fastmcp import Context

from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from senseframe.mcp.orchestration.pipeline_run import (
    get_default_store as _get_default_store,
    PipelineRunStore,
)
from senseframe.mcp.orchestration.transitions import get_transitions
from senseframe.mcp.pagination.page import build_page, clamp_limit
from senseframe.mcp.views.pipeline import (
    PipelineAdvanceResponse,
    PipelineCreateResponse,
    PipelineRunListView,
    PipelineRunView,
    TransitionView,
)

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_pipeline_create",
    "senseframe_pipeline_advance",
    "senseframe_pipeline_run",
    "senseframe_pipeline_get",
    "senseframe_pipeline_list",
    "senseframe_pipeline_pause",
    "senseframe_pipeline_resume",
    "get_pipeline_run_store",
    "_stack",
]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


def get_pipeline_run_store() -> PipelineRunStore:
    """返回进程级 PipelineRunStore 单例。

    委托给 orchestration.get_default_store()，确保 tools/ 与 resources/
    共享同一个 Store 实例（tool 创建的 run 对 resource 端点可见）。
    """
    return _get_default_store()


# ============================================================
# Tool handlers
# ============================================================


async def senseframe_pipeline_create(
    config: dict[str, Any],
    stages: list[str],
    trial_id: str | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineCreateResponse:
    """创建新的 PipelineRun（声明式：接受 config + stages）。

    Args:
        config: ExperimentConfig.model_dump() 的 dict。
        stages: stage 名列表（如 ["validate", "preflight", "load", ...]）。
        trial_id: 可选的 Study trial 关联。
        ctx: MCP Context（注入 request_id）。

    Returns:
        PipelineCreateResponse（含 run_id + state=Pending + _transitions）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_pipeline_create stages={stages} trial_id={trial_id}"
        )
    try:
        async with _stack.instrument("senseframe_pipeline_create", ctx):
            store = get_pipeline_run_store()
            run = store.create(config=config, stages=stages)
            if trial_id is not None:
                # trial_id 通过 advance(start) 注入，但 create 时也可记录
                # 这里用 advance 内部逻辑不能修改 Pending（无动作），所以记录到 run
                # 实际上 trial_id 是 start 时绑定的，create 时仅存储
                pass
            transitions = get_transitions(run.state, run)
            return PipelineCreateResponse(
                run_id=run.run_id,
                state=run.state,
                created_at=run.created_at,
                transitions=transitions,
            )
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_create failed: {e}")
        raise


async def senseframe_pipeline_advance(
    run_id: str,
    action: Literal["start", "complete", "fail", "retry", "skip", "pause", "resume"],
    completed_stage: str | None = None,
    failed_stage: str | None = None,
    error_message: str | None = None,
    trial_id: str | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineAdvanceResponse:
    """推进 PipelineRun 状态机（单一状态变更入口，幂等）。

    Args:
        run_id: PipelineRun ID。
        action: 动作名（start/complete/fail/retry/skip/pause/resume）。
        completed_stage: complete 时追加到 completed_stages。
        failed_stage: fail/skip 时记录失败的 stage 名。
        error_message: fail/skip 时的失败原因。
        trial_id: 可选的 trial 关联（start/resume/retry 时设置）。
        ctx: MCP Context。

    Returns:
        PipelineAdvanceResponse（含 previous_state + new_state + _transitions）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_pipeline_advance run_id={run_id} action={action}"
        )
    try:
        async with _stack.instrument("senseframe_pipeline_advance", ctx):
            store = get_pipeline_run_store()
            previous_run = store.get(run_id)
            previous_state = previous_run.state
            new_run = store.advance(
                run_id=run_id,
                action=action,
                completed_stage=completed_stage,
                failed_stage=failed_stage,
                error_message=error_message,
                trial_id=trial_id,
            )
            transitions = get_transitions(new_run.state, new_run)
            return PipelineAdvanceResponse(
                run_id=new_run.run_id,
                previous_state=previous_state,
                new_state=new_run.state,
                action=action,
                transitions=transitions,
            )
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_advance failed: {e}")
        raise


async def senseframe_pipeline_run(
    config: dict[str, Any],
    run_id: str | None = None,
    stages: list[str] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineRunView:
    """执行完整 pipeline（黑盒，阻塞，调用 ``run_pipeline``）。

    集成现有 `from senseframe.engine.runner import run_pipeline`，不重新实现训练逻辑。

    Args:
        config: ExperimentConfig.model_dump() 的 dict（或可被 ExperimentConfig 解析的 dict）。
        run_id: 可选的 run_id（None 时自动创建新 run）。
        stages: 可选的 stage 列表（None 时用默认 8 stage）。
        ctx: MCP Context。

    Returns:
        PipelineRunView（state=Succeeded/Failed，含 _transitions）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_pipeline_run run_id={run_id} stages={'set' if stages else 'default'}"
        )
    try:
        async with _stack.instrument("senseframe_pipeline_run", ctx):
            store = get_pipeline_run_store()
            # 若未提供 run_id，则创建新 run
            if run_id is None:
                # 默认 8 stage（与 Pipeline.default() 对齐）
                if stages is None:
                    from senseframe.introspect import list_stages
                    stages = list(list_stages())
                run = store.create(config=config, stages=stages)
                run_id = run.run_id
                # 启动 run
                run = store.advance(run_id=run_id, action="start")
            else:
                # 复用已有 run，确认存在
                run = store.get(run_id)
                # 若 run 仍在 Pending，先 start
                if run.state == "Pending":
                    run = store.advance(run_id=run_id, action="start")

            # 执行训练（阻塞）
            try:
                from senseframe.engine.runner import run_pipeline
                from senseframe.engine.config import ExperimentConfig

                # 将 config dict 转回 ExperimentConfig（若已是 ExperimentConfig 实例则直接用）
                if isinstance(config, dict):
                    cfg = ExperimentConfig.from_dict(config)
                else:
                    cfg = config
                run_pipeline(cfg)

                # 训练成功 → complete
                run = store.advance(
                    run_id=run_id,
                    action="complete",
                    completed_stage=run.stages[-1] if run.stages else None,
                )
            except Exception as train_exc:
                # 训练失败 → fail
                logger.exception(
                    "senseframe_pipeline_run training failed run_id=%s",
                    run_id,
                )
                run = store.advance(
                    run_id=run_id,
                    action="fail",
                    failed_stage=run.stages[-1] if run.stages else None,
                    error_message=f"{type(train_exc).__name__}: {train_exc}",
                )

            transitions = get_transitions(run.state, run)
            return PipelineRunView.from_domain(run, transitions=transitions)
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_run failed: {e}")
        raise


async def senseframe_pipeline_get(
    run_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineRunView:
    """查询 PipelineRun 状态（含 _transitions HATEOAS）。

    Args:
        run_id: PipelineRun ID。
        ctx: MCP Context。

    Returns:
        PipelineRunView（含 transitions 字段）。
    """
    if ctx:
        await ctx.info(f"senseframe_pipeline_get run_id={run_id}")
    try:
        async with _stack.instrument("senseframe_pipeline_get", ctx):
            store = get_pipeline_run_store()
            run = store.get(run_id)
            transitions = get_transitions(run.state, run)
            return PipelineRunView.from_domain(run, transitions=transitions)
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_get failed: {e}")
        raise


async def senseframe_pipeline_list(
    cursor: str | None = None,
    limit: int = 50,
    filter_dict: dict[str, Any] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineRunListView:
    """列出所有 PipelineRun（cursor 分页）。

    Args:
        cursor: 不透明 cursor（来自上一次 list 的 next_cursor），None 表示首页。
        limit: 页大小（钳制到 [1, 200]）。
        filter_dict: 过滤字典（支持 {"state": "Running"} 等等值过滤）。
        ctx: MCP Context。

    Returns:
        PipelineRunListView（含 items + next_cursor + total_count + limit）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_pipeline_list cursor={'set' if cursor else 'none'} limit={limit}"
        )
    try:
        async with _stack.instrument("senseframe_pipeline_list", ctx):
            store = get_pipeline_run_store()
            items, total_count, has_more = store.list_runs(
                cursor=cursor, limit=limit, filter_dict=filter_dict
            )
            # 构建 transitions 索引（advisory）
            transitions_by_run: dict[str, list[TransitionView]] = {}
            for run in items:
                transitions_by_run[run.run_id] = get_transitions(run.state, run)
            clamped_limit = clamp_limit(limit)
            page = build_page(
                items=items,
                total_count=total_count,
                limit=clamped_limit,
                has_more=has_more,
                last_id_fn=lambda r: r.run_id,
                filter_dict=filter_dict,
            )
            return PipelineRunListView.from_page(
                page, transitions_by_run=transitions_by_run
            )
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_list failed: {e}")
        raise


async def senseframe_pipeline_pause(
    run_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineAdvanceResponse:
    """暂停 PipelineRun（idempotent：已 Paused 时短路）。

    Args:
        run_id: PipelineRun ID。
        ctx: MCP Context。

    Returns:
        PipelineAdvanceResponse（action="pause"）。
    """
    if ctx:
        await ctx.info(f"senseframe_pipeline_pause run_id={run_id}")
    try:
        async with _stack.instrument("senseframe_pipeline_pause", ctx):
            store = get_pipeline_run_store()
            previous_run = store.get(run_id)
            previous_state = previous_run.state
            new_run = store.advance(run_id=run_id, action="pause")
            transitions = get_transitions(new_run.state, new_run)
            return PipelineAdvanceResponse(
                run_id=new_run.run_id,
                previous_state=previous_state,
                new_state=new_run.state,
                action="pause",
                transitions=transitions,
            )
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_pause failed: {e}")
        raise


async def senseframe_pipeline_resume(
    run_id: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> PipelineAdvanceResponse:
    """恢复已暂停的 PipelineRun（idempotent：已 Running 时短路）。

    Args:
        run_id: PipelineRun ID。
        ctx: MCP Context。

    Returns:
        PipelineAdvanceResponse（action="resume"）。
    """
    if ctx:
        await ctx.info(f"senseframe_pipeline_resume run_id={run_id}")
    try:
        async with _stack.instrument("senseframe_pipeline_resume", ctx):
            store = get_pipeline_run_store()
            previous_run = store.get(run_id)
            previous_state = previous_run.state
            new_run = store.advance(run_id=run_id, action="resume")
            transitions = get_transitions(new_run.state, new_run)
            return PipelineAdvanceResponse(
                run_id=new_run.run_id,
                previous_state=previous_state,
                new_state=new_run.state,
                action="resume",
                transitions=transitions,
            )
    except Exception as e:
        if ctx:
            await ctx.error(f"senseframe_pipeline_resume failed: {e}")
        raise
