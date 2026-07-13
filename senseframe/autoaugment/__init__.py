"""RFC-003 ε3 AutoAugment：数据增强搜索（P3.1.1-P3.1.3）。

P3 范围（按规划文档第 7 节）：
- AugmentationSearchSpace：增强搜索空间数据结构（DSP 合规）
- AutoAugmentSampler：进化搜索采样器（满足 SP Sampler Protocol）
- AutoAugmentPolicyBuilder：策略参数 → transform 函数
- make_autoaugment_datamodule_factory：增强注入 Pipeline（通过 datamodule_factory）

集成方式（与 NAS 对称）：
- NAS 通过 module_factory 注入架构（替换 scene 默认 model）
- AutoAugment 通过 datamodule_factory 注入增强（替换 scene 默认 transform）
- Pipeline stage_build 已有 datamodule_factory 分支，无需修改 pipeline.py

P4 路线（推迟）：
- RL-based AutoAugment（需训练 RL controller）
- 多数据集增强策略迁移（与 ε4 元学习整合）
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .search_space import (
    AugmentationParameterSpec,
    AugmentationSearchSpace,
    SUPPORTED_AUGMENT_OPS,
    build_default_search_space,
)
from .policy_builder import (
    AutoAugmentPolicyBuilder,
    list_augment_ops,
    get_augment_op,
    make_policy_from_params,
)
from .sampler import AutoAugmentSampler


def make_autoaugment_datamodule_factory(
    policy_params: Dict[str, Any],
    base_train_transform: Optional[Callable] = None,
    base_eval_transform: Optional[Callable] = None,
    builder: Optional[AutoAugmentPolicyBuilder] = None,
    search_space: Optional[AugmentationSearchSpace] = None,
):
    """构造 AutoAugment datamodule_factory（用于 config.datamodule_factory 注入，P3.1.3）。

    Pipeline stage_build 在 ctx.config.datamodule_factory 非空时调用它构造
    DataModule。AutoAugment 通过此函数将增强策略注入 Pipeline：
    1. 用 AutoAugmentPolicyBuilder.build() 构造 train_transform（增强）
    2. eval_transform 不增强（评估阶段不应用增强）
    3. 返回的 datamodule_factory 符合 Pipeline stage_build 契约

    与 NAS make_nas_module_factory 的对称性：
    - NAS：arch_params → ArchitectureBuilder.build() → nn.Module → GenericLightningModule
    - AutoAugment：policy_params → AutoAugmentPolicyBuilder.build() → transform fn → GenericDataModule

    Args:
        policy_params: 增强策略参数 dict（由 SP Sampler 采样得到）
            必含 "op_0" key，其余参数依槽位数而定
        base_train_transform: scene 默认的 train_transform（None 时只用增强 transform）
        base_eval_transform: scene 默认的 eval_transform（None 时用 identity）
        builder: AutoAugmentPolicyBuilder 实例（None 时用默认）
        search_space: AugmentationSearchSpace（用于 builder 验证参数，None 时不验证）

    Returns:
        datamodule_factory 函数：符合 Pipeline stage_build 契约
        签名：(train_dataset, test_dataset, **kwargs) -> GenericDataModule

    Examples:
        >>> from senseframe.autoaugment import make_autoaugment_datamodule_factory
        >>> config.datamodule_factory = make_autoaugment_datamodule_factory(
        ...     trial.params, base_train_transform=cfg.train_transform,
        ... )
        >>> # 现在 run_pipeline(config) 将使用 AutoAugment 增强策略
    """
    from ..engine.datamodule import GenericDataModule

    actual_builder = builder or AutoAugmentPolicyBuilder(search_space=search_space)
    aug_train_transform = actual_builder.build(policy_params)
    # 评估 transform：不增强（用 base_eval_transform 或 identity）
    eval_transform = base_eval_transform if base_eval_transform is not None else (lambda x, y: (x, y))

    # 组合 base_train_transform 与 aug_train_transform
    if base_train_transform is not None:
        def combined_train_transform(x, y):
            x, y = base_train_transform(x, y)
            return aug_train_transform(x, y)
    else:
        combined_train_transform = aug_train_transform

    def datamodule_factory(train_dataset, test_dataset, **kwargs):
        # 从 kwargs 提取 DataModule 参数（stage_build 传入）
        # 必需参数：batch_size, num_workers, pin_memory, persistent_workers, learning_mode
        # 可选参数：val_dataset, unsupervised_dataset, supervised_dataset, streaming, cache_dir, collate_fn
        #
        # 注意：stage_build 会传 train_transform / eval_transform（来自 scene get_transforms），
        # 但 AutoAugment 用自己的增强 transform 替代，因此从 kwargs 移除 scene 的 transform
        # 避免与 combined_train_transform / eval_transform 重复传参。
        kwargs.pop("train_transform", None)
        kwargs.pop("eval_transform", None)
        return GenericDataModule(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_transform=combined_train_transform,
            eval_transform=eval_transform,
            **kwargs,
        )

    return datamodule_factory


__all__ = [
    "AugmentationParameterSpec",
    "AugmentationSearchSpace",
    "SUPPORTED_AUGMENT_OPS",
    "build_default_search_space",
    "AutoAugmentPolicyBuilder",
    "AutoAugmentSampler",
    "list_augment_ops",
    "get_augment_op",
    "make_policy_from_params",
    "make_autoaugment_datamodule_factory",
]
