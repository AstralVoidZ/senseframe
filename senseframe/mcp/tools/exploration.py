"""``senseframe_exploration_recommend`` 工具（L4 SP 搜索协议）。

设计文档 0.8 节阶段 3.3：集成 ExplorationTracker 作为 SP 存储后端。

StudyManager 已为每个 study 内部持有一个 ExplorationTracker（manager._trackers[study_id]），
SP 的 ask/tell 已桥接到 tracker.add_trial/update_trial。本 tool 暴露
ExplorationTracker.recommend_next 能力，让 Agent 看到 feedback-aware 的下一步
策略推荐（RFC-002 阶段 R：闭合探索-反馈回路）。

Agent 使用模式：
1. senseframe_study_ask → trial_id + params
2. senseframe_pipeline_run(apply_params(config, params)) → run_id
3. senseframe_study_tell(trial_id, value, feedback={"status": "overfitting", ...})
4. senseframe_exploration_recommend(study_id) → 结构化推荐列表
5. Agent 据推荐决定下一步策略，继续 ask-tell 循环

ToolAnnotations: true/false/true/false（只读 + 幂等）。
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
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.exploration import (
    ExplorationRecommendationItem,
    ExplorationRecommendationView,
)

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_exploration_recommend",
    "_exploration_stack",
]

# MiddlewareStack：与 study_* 一致
_exploration_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


async def senseframe_exploration_recommend(
    study_id: str,
    task_type: str | None = None,
    top_k: int = 5,
    ctx: Context[Any, Any, Any] | None = None,
) -> ExplorationRecommendationView:
    """基于当前 study 的 feedback 推荐下一策略。

    流程：
    1. 从 StudyManager 获取 study 的 tracker（manager._trackers[study_id]）
    2. 调用 tracker.recommend_next(task_type=task_type, top_k=top_k) 获取推荐
    3. 返回结构化推荐（含 strategy + reason + priority + recommendation_id）

    Args:
        study_id: Study ID。
        task_type: 任务类型（用于查询兼容性矩阵，None 时跳过兼容性推荐）。
        top_k: 返回前 K 个推荐（默认 5）。
        ctx: MCP Context。

    Returns:
        ExplorationRecommendationView（含 study_id + recommendations + n_recommendations）。

    Raises:
        KeyError: study_id 不存在（→ ToolError study category）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_exploration_recommend study_id={study_id} "
            f"task_type={task_type} top_k={top_k}"
        )
    try:
        async with _exploration_stack.instrument(
            "senseframe_exploration_recommend", ctx
        ):
            manager = _get_default_manager()
            # StudyManager 持有 _trackers: Dict[str, ExplorationTracker]
            # 通过私有属性访问（无公共 API），带 type: ignore 注释
            study = manager.get_study(study_id)
            if study is None:
                raise KeyError(f"Study '{study_id}' not found")
            tracker = manager._trackers.get(study_id)  # type: ignore[attr-defined]
            if tracker is None:
                # 理论上不会发生：create_study 时已初始化 tracker
                raise KeyError(f"Tracker for study '{study_id}' not found")
            # 钳制 top_k 到合理范围
            k = max(1, min(int(top_k), 50))
            recommendations = tracker.recommend_next(
                task_type=task_type,
                top_k=k,
            )
            # 把 dict 列表转换为强类型 view
            items: list[ExplorationRecommendationItem] = []
            for rec in recommendations:
                items.append(
                    ExplorationRecommendationItem(
                        strategy=dict(rec.get("strategy", {}) or {}),
                        reason=str(rec.get("reason", "") or ""),
                        priority=str(rec.get("priority", "normal") or "normal"),
                        recommendation_id=str(
                            rec.get("recommendation_id", "") or ""
                        ),
                    )
                )
            return ExplorationRecommendationView(
                study_id=study_id,
                recommendations=items,
                n_recommendations=len(items),
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_exploration_recommend failed: {exc}")
        raise to_tool_error(exc)
