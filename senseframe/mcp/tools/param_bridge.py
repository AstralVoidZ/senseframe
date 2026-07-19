"""``senseframe_apply_params_extended`` 工具（L4 SP 搜索协议）。

设计文档 0.7.3 节 + 0.8 节阶段 3.5：增强 apply_params 支持联合搜索。

包装 senseframe.mcp.orchestration.param_bridge.apply_params_extended，
让 Agent 通过 MCP 调用增强版参数应用：
- 应用采样参数到 config（trainer 字段覆盖 + scene.params 透传）
- 注入 module_factory / datamodule_factory（NAS → HPO 联合搜索）
- 追加 extra_callbacks / trainer_factory

ToolAnnotations: true/false/true/false（只读 + 幂等）。
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from senseframe.mcp.orchestration.param_bridge import apply_params_extended
from senseframe.mcp.tools._errors import to_tool_error

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_apply_params_extended",
    "_param_bridge_stack",
    "ApplyParamsExtendedResponse",
]

# MiddlewareStack：与 study_* 一致
_param_bridge_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


# ============================================================
# 响应视图（轻量级，单独定义避免新增 view 文件）
# ============================================================
from senseframe.mcp.views._base import FrozenModel


class ApplyParamsExtendedResponse(FrozenModel):
    """``senseframe_apply_params_extended`` 响应视图。

    Attributes:
        config: 应用参数 + 注入工厂后的 ExperimentConfig.model_dump() dict。
        applied_params: 应用的采样参数（echo 输入）。
        injected_factories: 注入的工厂字段名列表（如 ["module_factory"]）。
    """

    config: dict[str, Any]
    applied_params: dict[str, Any]
    injected_factories: list[str]


# ============================================================
# Tool handler
# ============================================================


async def senseframe_apply_params_extended(
    config: dict[str, Any],
    params: dict[str, Any],
    module_factory: Any | None = None,
    datamodule_factory: Any | None = None,
    extra_callbacks: list[Any] | None = None,
    trainer_factory: Any | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> ApplyParamsExtendedResponse:
    """应用采样参数 + 注入工厂字段到 ExperimentConfig。

    包装 senseframe.mcp.orchestration.param_bridge.apply_params_extended。

    Args:
        config: ExperimentConfig.model_dump() 的 dict。
        params: 采样参数（来自 SP ask 返回的 trial.params）。
        module_factory: NAS 架构工厂（不可序列化，仅同进程调用有效）。
            MCP 客户端通常传 None；服务端编排器（如 AutoMLOrchestrator）
            可通过进程内调用注入。
        datamodule_factory: DataModule 工厂（同上）。
        extra_callbacks: 额外的 Lightning Callback 列表（同上）。
        trainer_factory: Trainer 工厂（同上）。
        ctx: MCP Context。

    Returns:
        ApplyParamsExtendedResponse（含 config + applied_params + injected_factories）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_apply_params_extended params_keys={list(params.keys())} "
            f"has_module_factory={module_factory is not None} "
            f"has_datamodule_factory={datamodule_factory is not None}"
        )
    try:
        async with _param_bridge_stack.instrument(
            "senseframe_apply_params_extended", ctx
        ):
            from senseframe.engine.config import ExperimentConfig

            # 反序列化 config dict
            if isinstance(config, dict):
                cfg = ExperimentConfig.from_dict(config)
            elif isinstance(config, ExperimentConfig):
                cfg = config
            else:
                raise TypeError(
                    f"config must be dict or ExperimentConfig, "
                    f"got {type(config).__name__}"
                )

            # 调用 apply_params_extended
            new_cfg = apply_params_extended(
                config=cfg,
                params=params,
                module_factory=module_factory,
                datamodule_factory=datamodule_factory,
                extra_callbacks=extra_callbacks,
                trainer_factory=trainer_factory,
            )

            # 跟踪注入的工厂字段
            injected: list[str] = []
            if module_factory is not None:
                injected.append("module_factory")
            if datamodule_factory is not None:
                injected.append("datamodule_factory")
            if extra_callbacks is not None:
                injected.append("extra_callbacks")
            if trainer_factory is not None:
                injected.append("trainer_factory")

            # 序列化回 dict（工厂字段在 runtime 中，会被 exclude=True 排除）
            config_dict = new_cfg.to_dict() if hasattr(new_cfg, "to_dict") else new_cfg.model_dump()

            return ApplyParamsExtendedResponse(
                config=config_dict,
                applied_params=dict(params),
                injected_factories=injected,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_apply_params_extended failed: {exc}")
        raise to_tool_error(exc)
