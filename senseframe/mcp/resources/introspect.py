"""ISP-0：``senseframe://introspect`` 聚合索引 Resource。

返回所有 Resource URI 列表 + 服务器版本 + 能力清单。

设计文档 0.3 节 L4 ISP 表的第 1 项，是 Agent 建立 SenseFrame 心智模型的
入口。Agent 调用此 Resource 后可知：
- 服务器的协议版本
- 服务器版本
- 所有可用的 Resource URI（含描述）
- 当前实现的 tool 列表（来自 EXPECTED_TOOLS）
"""

from __future__ import annotations

from typing import Any

__all__ = ["introspect"]

_PROTOCOL_VERSION = "1.0.0"
_SERVER_VERSION = "0.1.0"

# 12 个 Resource URI（设计文档 0.3 节 L4 ISP 表）。
# 在 resources/__init__.py 维护 _RESOURCE_REGISTRY，这里通过列表展示。
_ISP_RESOURCES: list[dict[str, str]] = [
    {
        "uri": "senseframe://introspect",
        "description": "Aggregate index — all Resource endpoints + version + capabilities",
    },
    {
        "uri": "senseframe://schemas/pipeline",
        "description": "PipelineContext schema (calls introspect.context_schema)",
    },
    {
        "uri": "senseframe://schemas/stage",
        "description": "StageSpec schema (calls introspect.stage_io)",
    },
    {
        "uri": "senseframe://schemas/config",
        "description": "ExperimentConfig schema (calls model_json_schema)",
    },
    {
        "uri": "senseframe://schemas/errors",
        "description": "Error code × recovery strategy mapping (based on ERROR_CODES)",
    },
    {
        "uri": "senseframe://pipeline/{run_id}/graph",
        "description": "Stage data-flow graph (field → producer/consumers)",
    },
    {
        "uri": "senseframe://pipeline/{run_id}/readiness",
        "description": "Runtime data readiness (advisory)",
    },
    {
        "uri": "senseframe://scenes",
        "description": "Scene catalog (calls list_scenes)",
    },
    {
        "uri": "senseframe://scenes/{name}/capabilities",
        "description": "Scene capabilities (SceneMeta)",
    },
    {
        "uri": "senseframe://search-space/{scene}/{model_id}",
        "description": "Search space (ParameterSpec list)",
    },
    {
        "uri": "senseframe://protocols",
        "description": "Registered Protocol types",
    },
    {
        "uri": "senseframe://tools/output-schemas",
        "description": "Read tool outputSchema index",
    },
]


async def introspect() -> dict[str, Any]:
    """ISP-0：服务器自省 — 版本 + 能力清单 + Resource URI 列表。

    Returns:
        含 server / protocol_version / server_version / resources / tools 字段。
    """
    # 延迟导入避免循环依赖
    from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS

    return {
        "server": "senseframe-mcp",
        "protocol_version": _PROTOCOL_VERSION,
        "server_version": _SERVER_VERSION,
        "resources": _ISP_RESOURCES,
        "tools": list(EXPECTED_TOOLS),
        "capabilities": {
            "introspection": True,
            "pipeline_lifecycle": True,
            "hateoas_transitions": True,
            "cursor_pagination": True,
            "advisory_only": True,  # HATEOAS _transitions 是 advisory
        },
    }
