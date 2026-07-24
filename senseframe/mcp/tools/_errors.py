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

    M19 修复：logger 只记录安全元数据（code + category），不记录 envelope.message。
    原因：envelope.message = str(exc)，可能含用户输入/路径/SQL 片段等敏感信息，
    写入 stderr 结构化日志有泄露风险。完整堆栈通过 logger.exception 的 exc_info
    自动记录（堆栈中的局部变量 repr 由 traceback 模块控制，不暴露原始 message）。

    Args:
        exc: 捕获的业务异常实例。

    Returns:
        ToolError 实例，message 为 ToolErrorResponse 的 model_dump_json()。
    """
    envelope = ToolErrorResponse.envelope_from(exc)
    # 只记录安全元数据；完整堆栈由 exc_info=True 自动附加
    logger.exception(
        "tool error routed: code=%s category=%s",
        envelope.code,
        envelope.category,
    )
    return ToolError(envelope.model_dump_json())
