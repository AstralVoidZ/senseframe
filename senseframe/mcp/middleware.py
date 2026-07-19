"""Tool 级中间件：MiddlewareStack + RequestId + RateLimit。

FastMCP stdio 传输无原生中间件钩子（仅 HTTP 传输通过 Starlette 提供），
本模块提供基于 async context manager 的轻量替代。

用法（在 server.py 的 tool wrapper 中）::

    _stack = MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware())

    @mcp.tool(...)
    async def _pipeline_create(..., ctx=None) -> CreateResponse:
        async with _stack.instrument("senseframe_pipeline_create", ctx):
            return pipeline_tools.create(...)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

__all__ = [
    "ToolMiddleware",
    "MiddlewareStack",
    "RequestIdMiddleware",
    "RateLimitMiddleware",
    "RateLimiter",
    "TokenBucketLimiter",
    "get_request_id",
]

# Per-asyncio-task context variable：当前 request_id。
# 由 RequestIdMiddleware.before 设置，.after 清空。
_request_id_ctx: ContextVar[str] = ContextVar("senseframe_request_id", default="-")


def get_request_id() -> str:
    """返回当前 request_id，无活跃中间件时返回 '-'。"""
    return _request_id_ctx.get()


@runtime_checkable
class ToolMiddleware(Protocol):
    """Tool 级中间件协议。

    ``before`` 在核心 tool 调用前执行（context 注入）。
    ``after`` 在成功或失败后执行（cleanup），由 stack 的 finally 块保证。
    """

    async def before(self, ctx: Context[Any, Any, Any] | None, tool_name: str) -> None: ...

    async def after(
        self, ctx: Context[Any, Any, Any] | None, tool_name: str, error: Exception | None
    ) -> None: ...


class MiddlewareStack:
    """有序中间件链，洋葱模式调用。

    ``before`` 钩子按注册顺序执行。
    ``after`` 钩子按反向顺序执行（经典洋葱 unwrap）。
    ``after`` 由 ``finally`` 块保证执行（即使 before 或 tool body 抛异常）。
    """

    def __init__(self, *middlewares: ToolMiddleware) -> None:
        self._middlewares = list(middlewares)

    @asynccontextmanager
    async def instrument(
        self, tool_name: str, ctx: Context[Any, Any, Any] | None
    ) -> AsyncIterator[None]:
        """Async context manager：包裹一次 tool 调用。

        用法::

            async with _stack.instrument("senseframe_pipeline_create", ctx):
                return pipeline_tools.create(...)
        """
        # before 钩子：注册顺序
        for mw in self._middlewares:
            await mw.before(ctx, tool_name)

        error: Exception | None = None
        try:
            yield
        except Exception as exc:
            error = exc
            raise
        finally:
            # after 钩子：反向顺序（洋葱 unwrap）
            for mw in reversed(self._middlewares):
                await mw.after(ctx, tool_name, error)


# ── 限流 ────────────────────────────────────────────────────────────────


@runtime_checkable
class RateLimiter(Protocol):
    """Per-key 限流协议。

    实现方在拒绝时抛 ``RateLimitExceeded``，成功时静默返回。
    ``acquire`` 故意为同步：token-bucket 计算是纯内存算术（无 I/O），
    async 反而增加开销。同时允许在 sync / async 上下文中使用。

    ``key`` 是与领域无关的 bucket 标识符。Per-tool 限流时调用方传 tool 名。
    """

    def acquire(self, key: str) -> None: ...


class TokenBucketLimiter:
    """Token-bucket 限流器：per-key bucket。

    仅依赖 stdlib ``time.monotonic()``，零外部依赖。
    ``calls_per_minute=0`` 完全禁用限流。
    """

    def __init__(self, calls_per_minute: float) -> None:
        self._rate = calls_per_minute
        self._buckets: dict[str, tuple[float, float]] = {}

    def acquire(self, key: str) -> None:
        if self._rate <= 0:
            return
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._rate, now))
        # 按经过时间补充 token
        elapsed = now - last
        tokens = min(self._rate, tokens + elapsed * self._rate / 60.0)
        if tokens < 1.0:
            from senseframe.mcp.errors import RateLimitExceeded

            raise RateLimitExceeded(
                f"rate limit {int(self._rate)}/min exceeded for tool {key!r}"
            )
        self._buckets[key] = (tokens - 1.0, now)


class RateLimitMiddleware:
    """Token-bucket 限流中间件：per-tool bucket，env var 控制。

    委托给 ``RateLimiter`` 实现（默认 ``TokenBucketLimiter``）。
    ``SENSEFRAME_RATE_LIMIT=0`` 完全禁用限流。
    默认 60 calls/min/tool。
    """

    def __init__(
        self,
        calls_per_minute: int = 60,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._limiter: RateLimiter = limiter or TokenBucketLimiter(calls_per_minute)

    async def before(self, ctx: Context[Any, Any, Any] | None, tool_name: str) -> None:
        self._limiter.acquire(tool_name)

    async def after(
        self, ctx: Context[Any, Any, Any] | None, tool_name: str, error: Exception | None
    ) -> None:
        pass  # 错误不退还 token


class RequestIdMiddleware:
    """注入 ``ctx.request_id`` 到模块级 ContextVar。

    tool 调用期间所有层（DAO / orchestrator / FSM）的日志行可通过
    ``get_request_id()`` 读取此值。tool 调用完成后清空。

    ``ctx.request_id`` 是 ``fastmcp.Context`` 的保证属性（mcp>=1.27 验证）。
    无需防御性 ``getattr``。
    """

    async def before(self, ctx: Context[Any, Any, Any] | None, tool_name: str) -> None:
        _request_id_ctx.set(ctx.request_id if ctx is not None else "-")

    async def after(
        self, ctx: Context[Any, Any, Any] | None, tool_name: str, error: Exception | None
    ) -> None:
        _request_id_ctx.set("-")
