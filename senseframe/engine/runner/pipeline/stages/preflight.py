"""Stage 2: 资源探测 + 路由 + 预检。"""
from __future__ import annotations

import os

from ..context import PipelineContext
from ..stage_spec import stage
from .....routing import ResourceProbe, ResourceRouter
from .....schemas import TrainOutput
from ...preflight import set_seed


@stage(
    name="preflight",
    reads=["config", "model_id", "dataset", "learning_mode"],  # P2.1: 对齐函数体真实读取
    writes=["report", "route_level", "route_config", "output"],
    description="Stage 2: 资源探测 + 路由 + 预检",
)
def stage_preflight(ctx: PipelineContext) -> PipelineContext:
    """Stage 2: 资源探测 + 路由 + 预检。"""
    # GPU 隔离
    gpu = ctx.config.trainer.gpu
    if gpu is None and ctx.config.scene.params:
        gpu = ctx.config.scene.params.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # 随机种子
    deterministic = ctx.config.trainer.deterministic
    set_seed(ctx.config.trainer.seed, deterministic=deterministic)

    # 资源探测 + 路由
    ctx.report = ResourceProbe.probe()
    ctx.route_level = ResourceRouter.route(ctx.report)
    ctx.route_config = ResourceRouter.get_route_config(ctx.route_level)

    # 初始化 TrainOutput
    ctx.output = TrainOutput(
        status="error",
        model_id=ctx.model_id,
        dataset=ctx.dataset,
        learning_mode=ctx.learning_mode,
    )
    ctx.output.resource = ctx.report.to_dict()
    ctx.output.route_config = {"route_level": ctx.route_level, **ctx.route_config}

    return ctx
