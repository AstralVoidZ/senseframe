"""senseframe_config_parse MCP tool 实装。

v2 次要差距修复：替代 tool_dispatch.py 的 _not_implemented stub。
LOW 8 修复：迁移到 FrozenModel 强类型响应 + to_tool_error 错误桥接。
"""
from __future__ import annotations

import yaml

from ...engine.config import ExperimentConfig
from ...engine.runner.errors import ConfigValidationError
from ..views.config import ConfigParseResponse
from ._errors import to_tool_error


async def senseframe_config_parse(
    config_yaml: str,
) -> ConfigParseResponse:
    """解析 YAML 配置字符串为 ExperimentConfig。

    Args:
        config_yaml: YAML 格式的配置字符串

    Returns:
        ConfigParseResponse（含解析后的 config dict）

    Raises:
        ToolError: YAML 语法错误或配置校验失败
    """
    try:
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
        config.validate()

        # 3. 返回强类型响应
        return ConfigParseResponse(config=config.to_dict())
    except Exception as exc:
        raise to_tool_error(exc)
