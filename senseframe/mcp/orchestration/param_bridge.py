"""参数桥接：apply_params_extended（设计文档 0.7.3 节阶段 3.5）。

增强 engine.hpo.apply_params 支持联合搜索：
- 在原有 trainer 字段 / scene.params 覆盖基础上，支持 module_factory /
  datamodule_factory 注入
- NAS 搜索到的最佳架构可通过 module_factory 注入 HPO Study 的 config
- HPO 搜索到的最佳超参可通过 datamodule_factory 注入 AutoAugment Study

设计原则：
- 不修改 engine/hpo.py 的 apply_params（保持向后兼容）
- 在 mcp 层包装扩展，复用现有逻辑 + 增加工厂注入能力
- 路径安全：pathlib.Path 自动解析绝对路径（项目规范）

使用方式：
    from senseframe.mcp.orchestration.param_bridge import apply_params_extended

    new_config = apply_params_extended(
        config=base_config,
        params=trial.params,
        module_factory=nas_module_factory,  # NAS 搜索结果
    )
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional

from senseframe.engine.config import ExperimentConfig
from senseframe.engine.hpo import apply_params

logger = logging.getLogger(__name__)

__all__ = [
    "apply_params_extended",
]

# 工厂类型别名（与 ExperimentConfig.runtime 字段一致）
ModuleFactory = Callable[..., Any]
DataModuleFactory = Callable[..., Any]


def apply_params_extended(
    config: ExperimentConfig,
    params: dict[str, Any],
    module_factory: Optional[ModuleFactory] = None,
    datamodule_factory: Optional[DataModuleFactory] = None,
    extra_callbacks: Optional[list[Any]] = None,
    trainer_factory: Optional[Callable[..., Any]] = None,
) -> ExperimentConfig:
    """增强版 apply_params：在原有参数覆盖基础上注入工厂字段。

    流程：
    1. 调用 engine.hpo.apply_params(config, params) 应用采样参数
       （覆盖 trainer 字段 / scene.params 透传）
    2. 若 module_factory 非空，注入 new_config.runtime.module_factory
    3. 若 datamodule_factory 非空，注入 new_config.runtime.datamodule_factory
    4. 若 extra_callbacks 非空，追加到 new_config.runtime.extra_callbacks
    5. 若 trainer_factory 非空，注入 new_config.runtime.trainer_factory

    Args:
        config: 原始配置（不会被修改）。
        params: 采样参数（来自 SP ask 返回的 trial.params）。
        module_factory: NAS 架构工厂（如 senseframe.nas.make_nas_module_factory
            返回的函数），None 表示不注入。
        datamodule_factory: DataModule 工厂，None 表示不注入。
        extra_callbacks: 额外的 Lightning Callback 列表，None 表示不追加。
        trainer_factory: Trainer 工厂，None 表示不注入。

    Returns:
        新的 ExperimentConfig 实例（深拷贝 + 参数应用 + 工厂注入）。

    Examples:
        >>> from senseframe.mcp.orchestration.param_bridge import apply_params_extended
        >>> from senseframe.nas import make_nas_module_factory
        >>>
        >>> # NAS 阶段获取最佳架构后，注入 HPO 阶段的 config
        >>> module_factory = make_nas_module_factory(
        ...     arch_params=best_arch_params,
        ...     input_shape=(30, 100),
        ... )
        >>> new_config = apply_params_extended(
        ...     config=base_config,
        ...     params=trial.params,
        ...     module_factory=module_factory,
        ... )
    """
    # 1. 调用原 apply_params 应用采样参数
    new_config = apply_params(config, params)

    # 2. 注入工厂字段（通过 runtime dataclass）
    if module_factory is not None:
        new_config.runtime.module_factory = module_factory
        logger.debug(
            "apply_params_extended: injected module_factory=%s",
            type(module_factory).__name__,
        )
    if datamodule_factory is not None:
        new_config.runtime.datamodule_factory = datamodule_factory
        logger.debug(
            "apply_params_extended: injected datamodule_factory=%s",
            type(datamodule_factory).__name__,
        )
    if extra_callbacks is not None:
        # 追加到现有 callbacks 列表（不替换）
        new_config.runtime.extra_callbacks.extend(extra_callbacks)
        logger.debug(
            "apply_params_extended: appended %d extra_callbacks",
            len(extra_callbacks),
        )
    if trainer_factory is not None:
        new_config.runtime.trainer_factory = trainer_factory
        logger.debug(
            "apply_params_extended: injected trainer_factory=%s",
            type(trainer_factory).__name__,
        )

    return new_config


def extract_best_arch_params(
    study_id: str,
    manager: Any | None = None,
) -> Optional[dict[str, Any]]:
    """从 NAS Study 提取最佳架构参数（用于跨 Study 经验传递）。

    用于 AutoMLOrchestrator 串联 NAS → HPO 时，将 NAS 阶段的最佳架构
    自动注入 HPO 阶段的 module_factory。

    Args:
        study_id: NAS Study ID。
        manager: StudyManager 实例（None 时用进程级单例）。

    Returns:
        最佳 trial 的 params dict，无可用数据时返回 None。
    """
    if manager is None:
        from senseframe.mcp.orchestration.study_manager import (
            get_default_manager,
        )
        manager = get_default_manager()
    best = manager.best_trial(study_id)
    if best is None:
        return None
    return dict(best.params)


def extract_best_hpo_params(
    study_id: str,
    manager: Any | None = None,
) -> Optional[dict[str, Any]]:
    """从 HPO Study 提取最佳超参（用于跨 Study 经验传递）。

    用于 AutoMLOrchestrator 串联 HPO → AutoAugment 时，将 HPO 阶段的
    最佳超参作为 AutoAugment 阶段的初始 config。

    Args:
        study_id: HPO Study ID。
        manager: StudyManager 实例（None 时用进程级单例）。

    Returns:
        最佳 trial 的 params dict，无可用数据时返回 None。
    """
    if manager is None:
        from senseframe.mcp.orchestration.study_manager import (
            get_default_manager,
        )
        manager = get_default_manager()
    best = manager.best_trial(study_id)
    if best is None:
        return None
    return dict(best.params)
