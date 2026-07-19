"""Stage 1: 校验配置 schema + 场景注册。"""
from __future__ import annotations

from ..context import PipelineContext
from ..stage_spec import stage
from ...errors import SceneNotRegisteredError, DatasetNotSupportedError, ModelNotSupportedError
from .....scenes import get_scene, has_scene, list_scenes


@stage(
    name="validate",
    reads=["config"],
    writes=["scene", "meta", "model_id", "dataset", "learning_mode"],
    description="Stage 1: 校验配置 schema + 场景注册",
)
def stage_validate(ctx: PipelineContext) -> PipelineContext:
    """Stage 1: 校验配置 schema + 场景注册。"""
    ctx.config.validate()

    if not has_scene(ctx.config.scene.name):
        raise SceneNotRegisteredError(
            f"Scene '{ctx.config.scene.name}' not registered. "
            f"Available: {[k for k in list_scenes().keys() if k != '_unavailable']}"
        )

    ctx.scene = get_scene(ctx.config.scene.name)
    ctx.meta = ctx.scene.meta()
    ctx.model_id = ctx.config.scene.model_id
    ctx.dataset = ctx.config.scene.dataset
    ctx.learning_mode = ctx.config.scene.learning_mode

    if not ctx.meta.is_dynamic_dataset:
        if ctx.dataset not in ctx.meta.supported_datasets:
            raise DatasetNotSupportedError(
                f"Dataset '{ctx.dataset}' not supported by scene '{ctx.config.scene.name}'. "
                f"Supported: {ctx.meta.supported_datasets}"
            )
    if ctx.model_id not in ctx.meta.supported_models:
        raise ModelNotSupportedError(
            f"Model '{ctx.model_id}' not supported by scene '{ctx.config.scene.name}'. "
            f"Supported: {ctx.meta.supported_models}"
        )

    return ctx
