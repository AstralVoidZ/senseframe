"""RFC-003 DSP-5：自省模块。

让 AI Agent 可程序化查询 SenseFrame 的数据结构契约。

所有函数返回 JSON 可序列化的 dict，供 LLM 工具调用与跨进程传输。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def context_schema() -> Dict[str, Any]:
    """返回 PipelineContext 完整字段契约（RFC-003 DSP-5）。

    Returns:
        含 schema_version + fields 列表（每字段含 name/type/fill_stage/has_default）
    """
    from .engine.runner.pipeline import PipelineContext
    return PipelineContext.schema() if hasattr(PipelineContext, "schema") else _fallback_context_schema()


def context_describe(ctx) -> Dict[str, Any]:
    """返回某个 ctx 实例的运行时状态（RFC-003 DSP-5）。

    Args:
        ctx: PipelineContext 实例

    Returns:
        含 completed_fields / extra_keys / trial_id / completed_stages
    """
    if hasattr(ctx, "describe"):
        return ctx.describe()
    return {
        "completed_fields": ctx.completed_fields() if hasattr(ctx, "completed_fields") else [],
        "extra_keys": list(ctx.extra.keys()) if hasattr(ctx, "extra") else [],
        "trial_id": getattr(ctx, "trial_id", ""),
        "completed_stages": list(getattr(ctx, "completed_stages", [])),
    }


def stage_io(name: Optional[str] = None) -> Dict[str, Any]:
    """返回 stage 的 IO Spec（RFC-003 DSP-5）。

    Args:
        name: stage 名（如 "validate"）。None 时返回全部 stage 的 Spec 列表。

    Returns:
        单个 stage 时返回该 stage 的 spec dict；
        name=None 时返回 {"stages": [spec_dict, ...]}
    """
    from .engine.runner.pipeline import Pipeline
    specs = Pipeline.default().stages_with_spec()

    def _spec_to_dict(spec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "reads": [{"name": f.name, "type": f.type, "required": f.required, "description": f.description} for f in spec.reads],
            "writes": [{"name": f.name, "type": f.type, "required": f.required, "description": f.description} for f in spec.writes],
            "description": spec.description,
        }

    if name is not None:
        for spec in specs:
            if spec.name == name:
                return _spec_to_dict(spec)
        return {"error": f"Stage '{name}' not found", "available": [s.name for s in specs]}

    return {"stages": [_spec_to_dict(s) for s in specs]}


def list_stages() -> List[str]:
    """返回 stage 名列表（RFC-003 DSP-5）。"""
    from .engine.runner.pipeline import Pipeline
    return [spec.name for spec in Pipeline.default().stages_with_spec()]


def pipeline_graph() -> Dict[str, Any]:
    """返回完整执行图（RFC-003 DSP-5）。

    field → producer（哪个 stage 写入） / consumer（哪个 stage 读取）的映射。

    Returns:
        {
            "fields": {
                "config": {"producers": [], "consumers": ["validate", "preflight", ...]},
                "scene": {"producers": ["validate"], "consumers": ["preflight", "resolve", ...]},
                ...
            }
        }
    """
    specs = stage_io()["stages"]
    fields: Dict[str, Dict[str, List[str]]] = {}

    for spec in specs:
        stage_name = spec["name"]
        for f in spec["reads"]:
            fname = f["name"]
            if fname not in fields:
                fields[fname] = {"producers": [], "consumers": []}
            fields[fname]["consumers"].append(stage_name)
        for f in spec["writes"]:
            fname = f["name"]
            if fname not in fields:
                fields[fname] = {"producers": [], "consumers": []}
            fields[fname]["producers"].append(stage_name)

    return {"fields": fields}


def data_bundle_schema() -> Dict[str, Any]:
    """返回 DatasetBundle 结构 + 填充规则（RFC-003 DSP-5）。"""
    from .scenes.base import DatasetBundle
    return DatasetBundle.schema()


def data_bundle_describe(bundle, learning_mode: str = "supervised") -> Dict[str, Any]:
    """返回某个 bundle 实例的运行时状态（RFC-003 DSP-5）。

    Args:
        bundle: DatasetBundle 实例
        learning_mode: 学习模式（用于填充规则校验）
    """
    if hasattr(bundle, "describe"):
        return bundle.describe(learning_mode)
    return {
        "filled_fields": bundle.filled_fields() if hasattr(bundle, "filled_fields") else [],
        "learning_mode": learning_mode,
        "validation_errors": bundle.validate_filling(learning_mode) if hasattr(bundle, "validate_filling") else [],
    }


def data_profile_schema() -> Dict[str, Any]:
    """返回 DataProfile 结构（RFC-003 DSP-5）。"""
    from .core.profiler import DataProfile
    return DataProfile.schema() if hasattr(DataProfile, "schema") else {
        "schema_version": "1.0.0",
        "fields": [
            {"name": "dtypes", "type": "Dict[str, str]"},
            {"name": "feature_names", "type": "List[str]"},
            {"name": "nullable", "type": "Dict[str, bool]"},
            {"name": "shapes", "type": "Dict[str, Tuple[int, ...]]"},
        ],
    }


def data_profile_describe(profile) -> Dict[str, Any]:
    """返回某个 profile 实例的运行时状态（RFC-003 DSP-5）。

    Args:
        profile: DataProfile 实例
    """
    if hasattr(profile, "describe"):
        return profile.describe()
    return {
        "n_features": len(getattr(profile, "feature_names", [])),
        "dtype_distribution": {},
        "nullable_ratio": 0.0,
    }


def _fallback_context_schema() -> Dict[str, Any]:
    """PipelineContext.schema() 不可用时的兜底。"""
    return {
        "schema_version": "1.0.0",
        "fields": [],
        "note": "PipelineContext.schema() not available",
    }


__all__ = [
    "context_schema",
    "context_describe",
    "stage_io",
    "list_stages",
    "pipeline_graph",
    "data_bundle_schema",
    "data_bundle_describe",
    "data_profile_schema",
    "data_profile_describe",
]
