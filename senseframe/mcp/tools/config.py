"""senseframe_config_parse MCP tool 实装。

v2 次要差距修复：替代 tool_dispatch.py 的 _not_implemented stub。
LOW 8 修复：迁移到 FrozenModel 强类型响应 + to_tool_error 错误桥接。
I14 修复：接入 MiddlewareStack，与其他 tool 对齐。
"""
from __future__ import annotations

import logging
from typing import Any

import yaml
from mcp.server.fastmcp import Context

from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from ...engine.config import ExperimentConfig
from ...engine.runner.errors import ConfigValidationError
from ..views.config import ConfigParseResponse
from ._errors import to_tool_error

logger = logging.getLogger(__name__)

__all__ = ["senseframe_config_parse", "_config_stack"]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_config_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


async def senseframe_config_parse(
    config_yaml: str,
    ctx: Context[Any, Any, Any] | None = None,
) -> ConfigParseResponse:
    """解析 YAML 配置字符串为 ExperimentConfig。

    Args:
        config_yaml: YAML 格式的配置字符串
        ctx: MCP Context（注入 request_id）。

    Returns:
        ConfigParseResponse（含解析后的 config dict）

    Raises:
        ToolError: YAML 语法错误或配置校验失败
    """
    if ctx:
        await ctx.info("senseframe_config_parse")
    try:
        async with _config_stack.instrument("senseframe_config_parse", ctx):
            # 1. YAML 语法解析
            try:
                config_dict = yaml.safe_load(config_yaml)
            except yaml.YAMLError as e:
                raise ConfigValidationError(f"YAML 语法错误: {e}") from e

            if not isinstance(config_dict, dict):
                raise ConfigValidationError(
                    f"YAML 顶层应为 dict，实际: {type(config_dict).__name__}"
                )

            # 2. ExperimentConfig 解析（含 extra='forbid' 校验）
            config = ExperimentConfig.from_dict(config_dict)

            # 3. 返回强类型响应
            return ConfigParseResponse(config=config.to_dict())
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_config_parse failed: {exc}")
        raise to_tool_error(exc)
