"""V017: I16 search category 注释标注预留状态。

Anchor: bug 编号 V017 + 修复 commit 6be8b80。
原始问题: 7 类 category 中 "search" 缺少预留说明，开发者误以为 search
  category 已废弃可删除，导致后续 HPO tool 落地时 category 集合不一致。
修复方式: tool_error.py:52-54 添加 search category 预留注释，
  说明待 HPO tool 落地后将 HPO 异常映射到 search。

如果此测试失败，说明 V017 修复被回退（search category 被移除或集合不再是 7 类）。
"""
from __future__ import annotations

from typing import get_args

import pytest


@pytest.mark.l4_regression
class TestV017SearchCategoryReserved:
    """锁定 V017 修复：7 类 category 集合含 search（预留状态）。"""

    def test_error_categories_are_exactly_seven_with_search(self):
        """V017 anchor: CategoryT 含 7 类，且 "search" 在集合中（预留状态）。

        如果此断言失败，V017 修复被回退。
        """
        from senseframe.mcp.views.tool_error import CategoryT

        categories = set(get_args(CategoryT))
        expected = {
            "pipeline",
            "study",
            "scene",
            "artifact",
            "config",
            "search",
            "internal",
        }

        # V017 关键断言 1：严格 7 类
        assert len(categories) == 7, (
            f"如果此断言失败，V017 修复被回退：category 应为 7 类，"
            f"实际 {len(categories)}: {sorted(categories)}"
        )
        # V017 关键断言 2：集合匹配（含 search 预留）
        assert categories == expected, (
            f"如果此断言失败，V017 修复被回退：category 集合不匹配: "
            f"expected={sorted(expected)} got={sorted(categories)}"
        )
        # V017 关键断言 3：search category 存在（预留状态）
        assert "search" in categories, (
            "如果此断言失败，V017 修复被回退：search category 应在 7 类集合中（预留状态）"
        )
