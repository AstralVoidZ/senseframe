"""Tool 层错误桥接：业务异常 → MCP ToolError（含 ToolErrorResponse 信封 JSON）。

所有 tool wrapper 捕获业务异常后调用 `to_tool_error`，构造统一信封
`(code, message, category)` 并 surface 给 MCP 客户端。原始异常的完整
堆栈通过 stderr 结构化日志保留，供运维排查。
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp.exceptions import ToolError

from senseframe.mcp.views import ToolErrorResponse

logger = logging.getLogger(__name__)

__all__ = ["ToolError", "to_tool_error"]


def to_tool_error(exc: BaseException) -> ToolError:
    """业务异常 → ToolError（payload 是 ToolErrorResponse 的 JSON）。

    流程：
    1. ToolErrorResponse.envelope_from(exc) 路由异常到 7 类 category
    2. logger.exception 记录完整堆栈到 stderr（运维诊断用）
    3. 返回 ToolError，其 message 是信封 JSON（客户端可程序化解析 category）

    Args:
        exc: 捕获的业务异常实例。

    Returns:
        ToolError 实例，message 为 ToolErrorResponse 的 model_dump_json()。
    """
    envelope = ToolErrorResponse.envelope_from(exc)
    logger.exception("%s envelope=%s", type(exc).__name__, envelope.model_dump_json())
    return ToolError(envelope.model_dump_json())
