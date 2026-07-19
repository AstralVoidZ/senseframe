"""Page[T] generic page wrapper + builder + limit clamping（对齐 pipeflow）。

响应格式（所有 list tool 统一）::

    {
        "items": [T, ...],
        "next_cursor": str | None,
        "total_count": int,
        "limit": int,
    }

`build_page` 使用 limit+1 技巧：DAO 返回最多 `limit+1` 行，多出的那行
意味着 `has_more=True`，会被截断。`next_cursor` 从**保留的最后一行**
的 id 编码（不是被丢弃的 peek 行）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from senseframe.mcp.pagination.cursor import encode_cursor

__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "MIN_LIMIT", "Page", "build_page", "clamp_limit"]

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MIN_LIMIT = 1

T = TypeVar("T")


@dataclass(frozen=True)
class Page[T]:
    """Generic page wrapper for cursor-paginated list responses."""

    items: list[T]
    next_cursor: str | None
    total_count: int
    limit: int

    def to_dict(self) -> dict[str, Any]:
        """序列化为 wire 格式。items 原样透传。

        每个 list tool 的调用方负责将自己的模型对象转换为 plain dict，
        tool 层无需感知 dataclass 内部。
        """
        return {
            "items": self.items,
            "next_cursor": self.next_cursor,
            "total_count": self.total_count,
            "limit": self.limit,
        }


def clamp_limit(limit: int) -> int:
    """将 `limit` 钳制到 `[MIN_LIMIT, MAX_LIMIT]`。

    - 负数或零 → MIN_LIMIT
    - 超过 MAX_LIMIT → MAX_LIMIT
    """
    if limit < MIN_LIMIT:
        return MIN_LIMIT
    if limit > MAX_LIMIT:
        return MAX_LIMIT
    return int(limit)


def build_page[T](
    items: list[T],
    total_count: int,
    limit: int,
    has_more: bool,
    last_id_fn: Callable[[T], str],
    filter_dict: dict[str, Any] | None = None,
) -> Page[T]:
    """从 DAO 行 + has_more + id 提取器构造 `Page[T]`。

    Args:
        items: 已保留的行（limit+1 技巧中已截断的 limit 行）。
        total_count: 总数（不受分页影响）。
        limit: 钳制后的 limit 值。
        has_more: 是否还有更多行（DAO 检测到 limit+1 行存在）。
        last_id_fn: 从行提取单调 id 的函数（SenseFrame 一律 str）。
        filter_dict: 当前请求的 filter，用于嵌入 fingerprint。

    Returns:
        `Page[T]` 实例，`next_cursor` 仅在 `has_more=True` 且 `items` 非空时编码。
    """
    next_cursor: str | None = None
    if has_more and items:
        next_cursor = encode_cursor(last_id_fn(items[-1]), filter_dict)
    return Page(
        items=items,
        next_cursor=next_cursor,
        total_count=total_count,
        limit=limit,
    )
