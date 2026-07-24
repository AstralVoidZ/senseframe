"""senseframe_config_parse MCP tool 实装。

v2 次要差距修复：替代 tool_dispatch.py 的 _not_implemented stub。
让 Agent 可通过 MCP 解析 YAML 配置为 ExperimentConfig（含 extra='forbid' 校验）。
"""
from __future__ import annotations

from typing import Any, Dict

import yaml

from ...engine.config import ExperimentConfig


async def senseframe_config_parse(
    config_yaml: str,
) -> Dict[str, Any]:
    """解析 YAML 配置字符串为 ExperimentConfig。

    Args:
        config_yaml: YAML 格式的配置字符串

    Returns:
        dict 含：
        - status: "ok" | "error"
        - config: ExperimentConfig.to_dict()（status=ok 时）
        - error: 错误信息（status=error 时）
        - error_code: 错误码（status=error 时）
    """
    # 1. YAML 语法解析
    try:
        config_dict = yaml.safe_load(config_yaml)
    except yaml.YAMLError as e:
        return {
            "status": "error",
            "error": f"YAML 语法错误: {e}",
            "error_code": "CONFIG_YAML_PARSE_ERROR",
        }

    if not isinstance(config_dict, dict):
        return {
            "status": "error",
            "error": f"YAML 顶层应为 dict，实际: {type(config_dict).__name__}",
            "error_code": "CONFIG_VALIDATION_ERROR",
        }

    # 2. ExperimentConfig 解析（含 extra='forbid' 校验）
    # ConfigValidationError 继承 ValueError，pydantic ValidationError 也是 ValueError 子类
    try:
        config = ExperimentConfig.from_dict(config_dict)
        config.validate()
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
            "error_code": "CONFIG_VALIDATION_ERROR",
        }

    # 3. 返回解析后的配置
    return {
        "status": "ok",
        "config": config.to_dict(),
    }
