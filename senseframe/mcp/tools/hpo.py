"""``senseframe_hpo_setup`` 工具（L4 SP 搜索协议）。

设计文档 0.8 节阶段 3.2：拆解 run_hpo 为 Ask-Tell 三步接口。

不修改 engine/hpo.py 的 run_hpo（保持向后兼容），而是在 MCP 层提供
helper tool：把 HPOConfig 转换为 Study 搜索空间，让 Agent 持有循环。

Agent 持有循环（设计文档 0.6 节长任务处理）：
1. senseframe_hpo_setup → study_id
2. senseframe_study_ask(study_id) → trial_id + params
3. senseframe_pipeline_run(config=apply_params(base_config, params)) → run_id
4. senseframe_study_tell(trial_id, value=run_result.val_acc, state="completed")
5. 重复 2-4 直到 n_trials 或 Agent 决定停止
6. senseframe_study_get(study_id) → best_trial

ToolAnnotations: false/false/false/true（创建 Study，openWorld）
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
from senseframe.mcp.orchestration.study_manager import (
    get_default_manager as _get_default_manager,
)
from senseframe.mcp.orchestration.study_transitions import (
    STUDY_STATE_RUNNING,
    get_study_transitions,
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.pipeline import TransitionView
from senseframe.mcp.views.study import StudyCreateResponse
from senseframe.search_protocol import ParameterSpec, SearchSpace

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_hpo_setup",
    "_hpo_stack",
    "_convert_scene_search_space_to_sp",
]

# MiddlewareStack：与 study_* 一致
_hpo_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


def _convert_scene_search_space_to_sp(scene_space: Any) -> SearchSpace:
    """把 scenes.base.SearchSpace 转换为 search_protocol.SearchSpace。

    scenes.base.SearchSpace.params 是 dict[name, spec_dict]，
    spec_dict 含 type / low / high / log / values 等字段。
    search_protocol.SearchSpace.parameters 是 list[ParameterSpec]。

    Args:
        scene_space: scenes.base.SearchSpace 实例（含 .params dict）。

    Returns:
        search_protocol.SearchSpace 实例。
    """
    params: list[ParameterSpec] = []
    if scene_space is None:
        return SearchSpace(parameters=params)
    # scenes.base.SearchSpace 有 .params dict[name, spec_dict]
    raw_params = getattr(scene_space, "params", None) or {}
    for name, spec in raw_params.items():
        spec_dict = dict(spec) if hasattr(spec, "items") else dict(spec.__dict__)
        ptype = spec_dict.get("type")
        if ptype == "categorical":
            # scenes.base 用 "values"，search_protocol 用 "choices"
            choices = spec_dict.get("values") or spec_dict.get("choices")
            params.append(
                ParameterSpec(
                    name=name,
                    type="categorical",
                    choices=list(choices) if choices else [],
                )
            )
        else:
            params.append(
                ParameterSpec(
                    name=name,
                    type=ptype or "float",
                    low=spec_dict.get("low"),
                    high=spec_dict.get("high"),
                    log=spec_dict.get("log", False),
                    step=spec_dict.get("step"),
                )
            )
    return SearchSpace(parameters=params)


async def senseframe_hpo_setup(
    config: dict[str, Any],
    n_trials: int = 20,
    sampler: str = "random",
    direction: str = "minimize",
    ctx: Context[Any, Any, Any] | None = None,
) -> StudyCreateResponse:
    """把 ExperimentConfig 的 HPOConfig 转换为 Study 搜索空间。

    流程：
    1. 从 config.hpo.search_space 或场景的 get_search_space() 获取搜索空间
    2. 调用 StudyManager.create_study(name=f"hpo_{scene.model_id}",
       direction=direction, search_space=SearchSpace(...), sampler=sampler)
    3. 返回 study_id + 搜索空间描述

    Agent 拿到 study_id 后可调用 senseframe_study_ask / tell 循环。

    Args:
        config: ExperimentConfig.model_dump() 的 dict。
        n_trials: 计划 trial 数（仅记录到 Study 名称，不强制）。
        sampler: SP 采样器名（random / grid / asha / hyperband）。
        direction: 优化方向（maximize / minimize）。
        ctx: MCP Context。

    Returns:
        StudyCreateResponse（含 study_id + transitions=[ask, stop]）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_hpo_setup n_trials={n_trials} sampler={sampler} direction={direction}"
        )
    try:
        async with _hpo_stack.instrument("senseframe_hpo_setup", ctx):
            # 1. 解析 ExperimentConfig
            from senseframe.engine.config import ExperimentConfig

            if isinstance(config, dict):
                cfg = ExperimentConfig.from_dict(config)
            elif isinstance(config, ExperimentConfig):
                cfg = config
            else:
                raise TypeError(
                    f"config must be dict or ExperimentConfig, got {type(config).__name__}"
                )

            # 2. 获取搜索空间：优先 config.hpo（无显式 search_space 字段），
            # 回退到场景的 get_search_space()
            search_space: SearchSpace
            scene_name = cfg.scene.name
            model_id = cfg.scene.model_id
            dataset = cfg.scene.dataset
            try:
                from senseframe.scenes import get_scene

                scene = get_scene(scene_name)
                scene_space = scene.get_search_space(model_id, dataset)
                search_space = _convert_scene_search_space_to_sp(scene_space)
            except Exception as exc:
                logger.warning(
                    "senseframe_hpo_setup: scene.get_search_space failed for "
                    "scene=%s model=%s dataset=%s: %s; falling back to empty space",
                    scene_name, model_id, dataset, exc,
                )
                search_space = SearchSpace()

            # 3. 创建 Study
            manager = _get_default_manager()
            study_name = f"hpo_{model_id}"
            study_id = manager.create_study(
                name=study_name,
                direction=direction,
                search_space=search_space,
                sampler=sampler,
            )
            study = manager.get_study(study_id)
            transitions = get_study_transitions(STUDY_STATE_RUNNING)
            return StudyCreateResponse(
                study_id=study_id,
                name=study.name,
                direction=study.direction,
                sampler=study.sampler,
                created_at=study.created_at,
                transitions=transitions,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_hpo_setup failed: {exc}")
        raise to_tool_error(exc)
