"""Stage 4: 解析 TaskSpec / FeatureSpec / 最终配置。"""
from __future__ import annotations

from ..context import PipelineContext, _logger
from ..stage_spec import stage
from .....routing import ResourceRouter
from ...preflight import preflight_check
from ...resolver import (
    experiment_config_to_dict,
    resolve_task_spec,
    resolve_feature_spec,
    validate_scene_capabilities,
)
from ...errors import ConfigValidationError


@stage(
    name="resolve",
    reads=["config", "scene", "dataset", "num_classes", "data_profile",
           "scene_kwargs", "report", "route_config", "meta", "model_id"],
    writes=["scene_info", "num_classes", "task_spec", "feature_spec",
            "resolved", "lightning_params", "distributed_kwargs"],
    description="Stage 4: 解析 TaskSpec / FeatureSpec / 最终配置",
)
def stage_resolve(ctx: PipelineContext) -> PipelineContext:
    """Stage 4: 解析 TaskSpec / FeatureSpec / 最终配置。"""
    # scene_kwargs 由 stage_load 前置填充
    # 获取 num_classes
    ctx.scene_info = ctx.scene.get_dataset_info(ctx.dataset, **ctx.scene_kwargs)
    ctx.num_classes = ctx.scene_info["num_classes"]

    # 自监督模式特殊处理
    is_self_supervised = (ctx.learning_mode == "self_supervised")
    if is_self_supervised:
        from .....registry import get_dataset_spec, is_dataset_registered
        spec = get_dataset_spec(ctx.dataset) if is_dataset_registered(ctx.dataset) else None
        supervised_source = spec.supervised_source if spec else ""
        if not supervised_source:
            if ctx.dataset != "NTU-Fi_HAR":
                raise ConfigValidationError(
                    f"Self-supervised mode requires dataset with supervised_source, "
                    f"got '{ctx.dataset}' (no supervised_source defined)."
                )
            supervised_source = "NTU-Fi-HumanID"
        from .....registry import get_dataset_spec as _gds
        src_spec = _gds(supervised_source)
        ctx.num_classes = src_spec.num_classes

    # 解析 TaskSpec（支持数据画像推断）
    ctx.task_spec = resolve_task_spec(
        ctx.config, ctx.scene, ctx.dataset, ctx.model_id,
        ctx.num_classes, scene_kwargs=ctx.scene_kwargs,
        data_profile=ctx.data_profile,
    )

    # P3-6：class_weight 自动注入桥接。
    # DataProfile 检测 imbalance_ratio>5 时自动计算 inverse frequency 权重，
    # 此处注入到 TaskSpec.loss_kwargs["weights"]，并将 loss 切换为 cross_entropy_weighted。
    # 设计原则：与 recommended_loss → resolver 自动注入是同一架构模式。
    # 用户显式指定 loss（config.scene.task_spec.loss 非 None）时不覆盖 loss，
    # 但仍注入 weights（供 cross_entropy_weighted 或 focal 的 alpha 使用）。
    if (ctx.data_profile is not None
            and ctx.data_profile.recommended_class_weights is not None
            and ctx.task_spec is not None):
        weights = ctx.data_profile.recommended_class_weights
        current_loss = ctx.task_spec.effective_loss
        # 仅当当前 loss 为普通 cross_entropy 时自动升级为 cross_entropy_weighted
        if current_loss == "cross_entropy":
            from .....core.losses import has_loss
            if has_loss("cross_entropy_weighted"):
                ctx.task_spec.loss = "cross_entropy_weighted"
                _logger.info(
                    "P3-6 auto-inject class_weight: ratio=%.2f, loss %s→cross_entropy_weighted, "
                    "weights=%s",
                    ctx.data_profile.imbalance_ratio, current_loss, weights,
                )
        # 注入 weights 到 loss_kwargs（合并已有 kwargs，不覆盖其他 key）
        existing_kwargs = dict(ctx.task_spec.loss_kwargs or {})
        existing_kwargs["weights"] = weights
        ctx.task_spec.loss_kwargs = existing_kwargs

    # 解析 FeatureSpec
    ctx.feature_spec = resolve_feature_spec(
        ctx.config, ctx.scene, ctx.dataset, scene_kwargs=ctx.scene_kwargs,
    )

    # 校验场景能力
    validate_scene_capabilities(ctx.meta, ctx.task_spec, ctx.learning_mode, ctx.config.scene.name)

    # 解析最终配置
    config_dict = experiment_config_to_dict(ctx.config)
    model_info = ctx.scene.get_model_info(ctx.model_id)
    preflight_check(
        config_dict, model_info, ctx.report, ctx.dataset,
        scene_name=ctx.config.scene.name,
        scene_params=ctx.config.scene.params,
    )
    ctx.resolved = ResourceRouter.resolve_config(config_dict, ctx.route_config, model_info, ctx.report)
    ctx.resolved["deterministic"] = ctx.config.trainer.deterministic

    # Lightning Trainer 参数
    ctx.lightning_params = ResourceRouter.to_lightning_params(ctx.resolved)

    # 分布式训练参数
    ctx.distributed_kwargs = {}
    if "strategy" in ctx.lightning_params:
        ctx.distributed_kwargs["strategy"] = ctx.lightning_params["strategy"]
    if "num_nodes" in ctx.lightning_params:
        ctx.distributed_kwargs["num_nodes"] = ctx.lightning_params["num_nodes"]
    if "sync_batchnorm" in ctx.lightning_params:
        ctx.distributed_kwargs["sync_batchnorm"] = ctx.lightning_params["sync_batchnorm"]

    return ctx
