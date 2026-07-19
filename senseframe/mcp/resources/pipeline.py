"""ISP-7 + ISP-9：``senseframe://pipeline/{run_id}/graph`` + ``readiness``。

2 个 Resource 端点：
- ISP-7: senseframe://pipeline/{run_id}/graph     — stage 数据流图（field → producer/consumers）
- ISP-9: senseframe://pipeline/{run_id}/readiness — 运行时数据就绪度（advisory）

依赖：
- senseframe.introspect.pipeline_graph（stage 数据流图）
- senseframe.mcp.orchestration.pipeline_run.PipelineRunStore（读取 run 状态）

注意：advisory 模式 — readiness 字段就绪度只是建议性的，不阻断 Agent。

分层不变量（AST 守卫测试钉死）：
- resources/ 不得 import tools
- resources/ 可 import orchestration（单向依赖允许）
"""

from __future__ import annotations

from typing import Any

__all__ = ["pipeline_graph", "pipeline_readiness"]


def _get_default_store() -> Any:
    """获取默认 PipelineRunStore 单例。

    委托给 orchestration.get_default_store()，与 tools/pipeline.py 共享
    同一个 Store 实例（tool 创建的 run 对 resource 端点可见）。
    """
    from senseframe.mcp.orchestration.pipeline_run import get_default_store

    return get_default_store()


async def pipeline_graph(run_id: str) -> dict[str, Any]:
    """ISP-7：stage 数据流图（field → producer/consumers）。

    Args:
        run_id: PipelineRun ID（路径参数）。

    Returns:
        含 run_id + graph 字段（field → {producers, consumers} 映射）。
    """
    from senseframe.introspect import pipeline_graph as _build_graph

    # 验证 run 存在（不存在会抛 PipelineNotFound）
    store = _get_default_store()
    run = store.get(run_id)
    # 获取静态 graph（来自 Pipeline.default() 的 stages）
    graph = _build_graph()
    return {
        "run_id": run_id,
        "state": run.state,
        "graph": graph,
    }


async def pipeline_readiness(run_id: str) -> dict[str, Any]:
    """ISP-9：运行时数据就绪度（advisory）。

    Args:
        run_id: PipelineRun ID（路径参数）。

    Returns:
        含 run_id + state + completed_stages + readiness 字段。

    Note:
        readiness 是 advisory 的：True 表示字段已就绪，False 表示未就绪。
        Agent 可参考但不应被阻断 — _transitions 的 prerequisites 才是建议。
    """
    from senseframe.introspect import pipeline_graph

    store = _get_default_store()
    run = store.get(run_id)

    # 从静态 graph 推导每个 stage 的 readiness
    graph = pipeline_graph()
    fields = graph.get("fields", {})

    # 简化的 readiness 推导：
    # - 若 stage 在 completed_stages 中 → 所有 writes 字段 ready
    # - 若 stage 已失败 → writes 字段 not ready
    # - 否则 pending（advisory）
    stage_readiness: list[dict[str, Any]] = []
    for stage_name in run.stages:
        completed = stage_name in run.completed_stages
        failed = run.failed_stage == stage_name
        # 该 stage 的 writes 字段
        writes: list[str] = []
        for field_name, info in fields.items():
            if stage_name in info.get("producers", []):
                writes.append(field_name)
        if completed:
            field_status = {w: True for w in writes}
            stage_state = "succeeded"
        elif failed:
            field_status = {w: False for w in writes}
            stage_state = "failed"
        else:
            field_status = {w: False for w in writes}
            stage_state = "pending"
        stage_readiness.append({
            "stage": stage_name,
            "state": stage_state,
            "writes": writes,
            "fields_ready": field_status,
        })

    return {
        "run_id": run_id,
        "state": run.state,
        "completed_stages": list(run.completed_stages),
        "failed_stage": run.failed_stage,
        "readiness": stage_readiness,
        "advisory": True,  # 标记为 advisory
    }
