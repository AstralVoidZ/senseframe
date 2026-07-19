"""Exploration 系列 view model（L4 SP 搜索协议，FrozenModel 子类）。

设计文档 0.8 节阶段 3.3：暴露 ExplorationTracker.recommend_next 的结果视图。
- ExplorationRecommendationItem：单条推荐（含 strategy + reason + priority）
- ExplorationRecommendationView：tool 响应视图（含 study_id + recommendations 列表）

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from senseframe.mcp.views._base import FrozenModel

__all__ = [
    "ExplorationRecommendationItem",
    "ExplorationRecommendationView",
]


class ExplorationRecommendationItem(FrozenModel):
    """单条探索推荐（RFC-002 阶段 R）。

    Attributes:
        strategy: 推荐的策略组合（如 {"loss": "focal", "lr_scale": 0.1}）。
        reason: 推荐理由（如 "feedback: 数值不稳定 → 推荐稳定 loss + 降低 lr"）。
        priority: 优先级（high / medium / low / normal）。
        recommendation_id: 推荐项 ID（用于 log_adoption 闭环追溯）。
    """

    strategy: dict[str, Any]
    reason: str = ""
    priority: str = "normal"
    recommendation_id: str = ""


class ExplorationRecommendationView(FrozenModel):
    """``senseframe_exploration_recommend`` 响应视图。

    Attributes:
        study_id: 关联的 Study ID。
        recommendations: 推荐项列表（按 priority 排序，最相关的在前）。
        n_recommendations: 推荐项数量。
    """

    study_id: str
    recommendations: list[ExplorationRecommendationItem] = Field(default_factory=list)
    n_recommendations: int = 0
