"""config_parse MCP tool 的响应模型。

LOW 8 修复：与其他 MCP tool 对齐，用 FrozenModel 强类型响应。
"""
from __future__ import annotations

from typing import Any

from ._base import FrozenModel


class ConfigParseResponse(FrozenModel):
    """senseframe_config_parse 的成功响应。

    Attributes:
        config: 解析后的 ExperimentConfig.to_dict() 输出
    """

    config: dict[str, Any]


__all__ = ["ConfigParseResponse"]
