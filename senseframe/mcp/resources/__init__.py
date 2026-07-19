"""L4 ISP：12 个自省 Resource 端点（设计文档 0.3 节）。

公开 API：
- introspect.introspect（ISP-0：聚合索引）
- schemas.schema_pipeline / schema_stage / schema_config / schema_errors
  / tools_output_schemas（ISP-1..4 + ISP-11）
- pipeline.pipeline_graph / pipeline_readiness（ISP-7 + ISP-9）
- scenes.scenes / scene_capabilities / search_space（ISP-8 + 9 + 10）
- protocols.protocols（ISP-10 alt）

_RESOURCE_REGISTRY 维护 (uri, name, description, handler) 4 元组列表，
server.py 通过 ``mcp.resource(uri, name=..., description=...)(handler)`` 注册。

分层不变量（AST 守卫测试钉死）：
- resources/ 不得 import tools
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from senseframe.mcp.resources import introspect, pipeline, protocols, schemas, scenes

__all__ = [
    "_RESOURCE_REGISTRY",
    # 模块导出
    "introspect",
    "schemas",
    "pipeline",
    "scenes",
    "protocols",
]

# 12 个 Resource 端点的注册表（设计文档 0.3 节 L4 ISP 表）。
# 格式：(uri, name, description, handler)
_RESOURCE_REGISTRY: list[tuple[str, str, str, Callable[..., Any]]] = [
    # ISP-0: 聚合索引
    (
        "senseframe://introspect",
        "isp-introspect",
        "Aggregate index — all Resource endpoints + version + capabilities",
        introspect.introspect,
    ),
    # ISP-1..4 + ISP-11: schemas
    (
        "senseframe://schemas/pipeline",
        "isp-schema-pipeline",
        "PipelineContext schema (calls introspect.context_schema)",
        schemas.schema_pipeline,
    ),
    (
        "senseframe://schemas/stage",
        "isp-schema-stage",
        "StageSpec schema (calls introspect.stage_io)",
        schemas.schema_stage,
    ),
    (
        "senseframe://schemas/config",
        "isp-schema-config",
        "ExperimentConfig schema (calls model_json_schema)",
        schemas.schema_config,
    ),
    (
        "senseframe://schemas/errors",
        "isp-schema-errors",
        "Error code × recovery strategy mapping (based on ERROR_CODES)",
        schemas.schema_errors,
    ),
    (
        "senseframe://tools/output-schemas",
        "isp-tools-output-schemas",
        "Read tool outputSchema index",
        schemas.tools_output_schemas,
    ),
    # ISP-7 + ISP-9: pipeline graph + readiness
    (
        "senseframe://pipeline/{run_id}/graph",
        "isp-pipeline-graph",
        "Stage data-flow graph (field → producer/consumers)",
        pipeline.pipeline_graph,
    ),
    (
        "senseframe://pipeline/{run_id}/readiness",
        "isp-pipeline-readiness",
        "Runtime data readiness (advisory)",
        pipeline.pipeline_readiness,
    ),
    # ISP-8 + 9 + 10: scenes + capabilities + search-space
    (
        "senseframe://scenes",
        "isp-scenes",
        "Scene catalog (calls list_scenes)",
        scenes.scenes,
    ),
    (
        "senseframe://scenes/{name}/capabilities",
        "isp-scene-capabilities",
        "Scene capabilities (SceneMeta)",
        scenes.scene_capabilities,
    ),
    (
        "senseframe://search-space/{scene}/{model_id}",
        "isp-search-space",
        "Search space (ParameterSpec list)",
        scenes.search_space,
    ),
    # ISP-10 alt: protocols
    (
        "senseframe://protocols",
        "isp-protocols",
        "Registered Protocol types",
        protocols.protocols,
    ),
]
