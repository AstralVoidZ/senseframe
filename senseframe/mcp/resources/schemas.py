"""ISP-1..4 + ISP-11：``senseframe://schemas/*`` + ``senseframe://tools/output-schemas``。

5 个 Resource 端点：
- ISP-1:  senseframe://schemas/pipeline  — PipelineContext schema
- ISP-2:  senseframe://schemas/stage     — StageSpec schema（reads/writes/executor）
- ISP-3:  senseframe://schemas/config    — ExperimentConfig schema
- ISP-4:  senseframe://schemas/errors    — 错误码 × 恢复策略映射
- ISP-11: senseframe://tools/output-schemas — 所有读 tool 的 outputSchema 索引

依赖：
- senseframe.introspect（context_schema / stage_io）
- senseframe.engine.config.ExperimentConfig.model_json_schema()
- senseframe.schemas.ERROR_CODES
- senseframe.mcp.views（outputSchema 自动从 FrozenModel 子类生成）
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "schema_pipeline",
    "schema_stage",
    "schema_config",
    "schema_errors",
    "tools_output_schemas",
]


async def schema_pipeline() -> dict[str, Any]:
    """ISP-1：PipelineContext schema（调用 introspect.context_schema）。

    Returns:
        PipelineContext 字段契约（schema_version + fields 列表）。
    """
    from senseframe.introspect import context_schema

    return context_schema()


async def schema_stage() -> dict[str, Any]:
    """ISP-2：StageSpec schema（调用 introspect.stage_io）。

    Returns:
        所有 stage 的 IO Spec（reads/writes/description）。
    """
    from senseframe.introspect import stage_io

    return stage_io()


async def schema_config() -> dict[str, Any]:
    """ISP-3：ExperimentConfig schema（pydantic v2 model_json_schema）。

    Returns:
        ExperimentConfig 的 JSON Schema（pydantic v2 标准）。
    """
    from senseframe.engine.config import ExperimentConfig

    return ExperimentConfig.model_json_schema()


async def schema_errors() -> dict[str, Any]:
    """ISP-4：错误码 × 恢复策略映射（基于 ERROR_CODES）。

    Agent 收到错误后可程序化查询恢复路径：
    - 每个错误码对应的可恢复性（recoverable）
    - 推荐的下一步动作（advisory）

    Returns:
        含 schema_version + error_codes 列表（每条含 code + description + recoverable + suggested_action）。
    """
    from senseframe.schemas import ERROR_CODES

    # 错误码 × 恢复策略映射（advisory，可内省）。
    # recoverable=True 表示 Agent 可程序化恢复，False 表示需要人工介入。
    _RECOVERY_STRATEGY: dict[str, dict[str, Any]] = {
        "OK": {"recoverable": True, "suggested_action": None},
        "CONFIG_VALIDATION_ERROR": {
            "recoverable": True,
            "suggested_action": "fix config and re-run senseframe_config_parse",
        },
        "CONFIG_PARSE_ERROR": {
            "recoverable": True,
            "suggested_action": "fix YAML syntax and re-run senseframe_config_parse",
        },
        "CONFIG_NOT_FOUND": {
            "recoverable": True,
            "suggested_action": "provide --config path",
        },
        "MISSING_CONFIG": {
            "recoverable": True,
            "suggested_action": "provide --config path",
        },
        "INVALID_CONFIG_FORMAT": {
            "recoverable": True,
            "suggested_action": "ensure YAML top-level is mapping",
        },
        "UNSUPPORTED_FORMAT": {
            "recoverable": True,
            "suggested_action": "use one of supported export_formats",
        },
        "SCENE_NOT_FOUND": {
            "recoverable": True,
            "suggested_action": "register scene via senseframe.scenes.register_scene",
        },
        "DATASET_NOT_SUPPORTED": {
            "recoverable": True,
            "suggested_action": "use a dataset listed in scene.capabilities.supported_datasets",
        },
        "MODEL_NOT_SUPPORTED": {
            "recoverable": True,
            "suggested_action": "use a model listed in scene.capabilities.supported_models",
        },
        "DATA_NOT_FOUND": {
            "recoverable": True,
            "suggested_action": "verify dataset path or use is_dynamic_dataset scene",
        },
        "DATA_LOAD_ERROR": {
            "recoverable": False,
            "suggested_action": "check data integrity (corrupted file)",
        },
        "MODEL_BUILD_ERROR": {
            "recoverable": False,
            "suggested_action": "inspect model factory for code bug",
        },
        "TRAINING_ERROR": {
            "recoverable": False,
            "suggested_action": "check training log for runtime exception",
        },
        "OOM_ERROR": {
            "recoverable": True,
            "suggested_action": "reduce batch_size or use smaller model",
        },
        "CHECKPOINT_ERROR": {
            "recoverable": True,
            "suggested_action": "remove corrupted checkpoint and re-train",
        },
        "SAVE_ERROR": {
            "recoverable": True,
            "suggested_action": "check disk space and permissions",
        },
        "PREFLIGHT_ERROR": {
            "recoverable": True,
            "suggested_action": "free GPU/CPU resources and retry",
        },
        "METADATA_NOT_FOUND": {
            "recoverable": True,
            "suggested_action": "re-run training to generate metadata.json",
        },
        "METADATA_VERSION_ERROR": {
            "recoverable": False,
            "suggested_action": "upgrade SenseFrame or downgrade metadata.json schema_version",
        },
        "UNKNOWN_ERROR": {
            "recoverable": False,
            "suggested_action": "inspect server logs for full exception",
        },
    }

    error_codes: list[dict[str, Any]] = []
    for code, description in ERROR_CODES.items():
        strategy = _RECOVERY_STRATEGY.get(
            code,
            {"recoverable": False, "suggested_action": None},
        )
        error_codes.append({
            "code": code,
            "description": description,
            "recoverable": strategy["recoverable"],
            "suggested_action": strategy["suggested_action"],
        })

    return {
        "schema_version": "1.0.0",
        "error_codes": error_codes,
    }


async def tools_output_schemas() -> dict[str, Any]:
    """ISP-11：所有读 tool 的 outputSchema 索引。

    从 senseframe.mcp.views 中的 FrozenModel 子类自动生成 outputSchema。
    每个视图类的 ``model_json_schema()`` 返回标准 JSON Schema。

    Returns:
        含 schema_version + tools 列表（每条含 name + output_schema）。
    """
    from senseframe.mcp.views import (
        PipelineAdvanceResponse,
        PipelineCreateResponse,
        PipelineRunListView,
        PipelineRunView,
        ToolErrorResponse,
    )

    # 读 tool → 其返回的 view class
    _READ_TOOLS: list[tuple[str, type]] = [
        ("senseframe_pipeline_get", PipelineRunView),
        ("senseframe_pipeline_list", PipelineRunListView),
        # 错误信封（所有 tool 失败时都返回此 schema）
        ("__error_envelope__", ToolErrorResponse),
    ]
    # 写 tool 的响应 schema
    _WRITE_TOOLS: list[tuple[str, type]] = [
        ("senseframe_pipeline_create", PipelineCreateResponse),
        ("senseframe_pipeline_advance", PipelineAdvanceResponse),
        ("senseframe_pipeline_pause", PipelineAdvanceResponse),
        ("senseframe_pipeline_resume", PipelineAdvanceResponse),
        ("senseframe_pipeline_run", PipelineRunView),
    ]

    tools: list[dict[str, Any]] = []
    for name, view_cls in _READ_TOOLS + _WRITE_TOOLS:
        tools.append({
            "name": name,
            "output_schema": view_cls.model_json_schema(),
        })
    return {
        "schema_version": "1.0.0",
        "tools": tools,
    }
