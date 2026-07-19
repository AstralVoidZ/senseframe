"""RFC-003 ε2 NAS：神经架构搜索（P2.6-P2.9 + P3.3 DARTS/ENAS + P1.3 真实超网）。

P2 范围（按规划文档第 336 行约束）：
- ArchitectureSearchSpace：NAS 搜索空间数据结构（DSP 合规）
- ArchitectureBuilder：架构参数 → nn.Module（conv1d + rnn + hybrid）
- EvolutionarySampler：进化算法搜索（变异 + 交叉 + 选择），满足 Sampler Protocol
- NAS 集成：通过 module_factory 注入 Pipeline（make_nas_module_factory）

P3.3 范围（按规划文档第 429-432 行约束）：
- AttentionNet + ArchitectureBuilder._build_attention：Transformer 风格架构
- DARTSSampler：可微架构搜索（梯度-based，双优化）
- DARTSPipelineRun：DARTS 特殊 PipelineRun（内部双优化）
- ENASSampler：权重共享 NAS（controller-based）
- ArchitectureSearchSpace 扩展 attention cell_type

P1.3 范围（2026-07-19）：
- DARTSSupernet：真实可微超网（所有候选 op 并行 + α softmax 加权）
- DARTSCell：可微 cell（M 个候选 op 并行计算）
- DARTSPipelineRun.use_real_supernet=True 走真实超网双优化路径
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .search_space import ArchitectureSearchSpace, ArchitectureParameterSpec
from .builder import ArchitectureBuilder, AttentionNet
from .sampler import EvolutionarySampler, ENASSampler
from .darts import DARTSSampler, DARTSPipelineRun
from .supernet import DARTSCell, DARTSSupernet, OP_NAMES


def make_nas_module_factory(
    arch_params: Dict[str, Any],
    input_shape: Tuple[int, ...],
    builder: Optional[ArchitectureBuilder] = None,
):
    """构造 NAS module_factory（用于 config.module_factory 注入，P2.9）。

    Pipeline stage_build 在 ctx.config.module_factory 非空时调用它构造
    LightningModule。NAS 通过此函数将架构参数注入 Pipeline：
    1. 用 ArchitectureBuilder.build() 构造 nn.Module（替代 scene 默认模型）
    2. 用 GenericLightningModule 包装（复用训练/验证/指标逻辑）

    Args:
        arch_params: 架构参数 dict（由 SP Sampler 采样得到）
            必含 "cell_type" key，其余参数依 cell_type 而定
        input_shape: 模型输入形状（不含 batch 维），如 (channels, length)
            需与 ctx.scene_info["input_shape"] 一致
        builder: ArchitectureBuilder 实例（None 时用默认）

    Returns:
        module_factory 函数：Callable(model=None, **kwargs) -> GenericLightningModule
        符合 Pipeline stage_build 的 module_factory 契约

    Examples:
        >>> from senseframe.nas import make_nas_module_factory
        >>> config.module_factory = make_nas_module_factory(
        ...     trial.params, input_shape=(30, 100),
        ... )
        >>> # 现在 run_pipeline(config) 将使用 NAS 构建的模型
    """
    from ..engine.module import GenericLightningModule

    actual_builder = builder or ArchitectureBuilder()

    def module_factory(model=None, **kwargs):
        # 忽略 scene 构建的默认 model（NAS 用自己的 arch_params 构造）
        # num_classes 必填：框架不猜测数据集类别数，调用方必须通过 kwargs 传入
        if "num_classes" not in kwargs:
            raise ValueError(
                "make_nas_module_factory: num_classes 必填。"
                "调用方必须通过 kwargs 传入 num_classes（由 DatasetSpec.num_classes 派生）。"
            )
        num_classes = int(kwargs["num_classes"])
        nas_model = actual_builder.build(arch_params, input_shape, num_classes)
        # 用 GenericLightningModule 包装（复用训练逻辑）
        return GenericLightningModule(model=nas_model, **kwargs)

    return module_factory


__all__ = [
    "ArchitectureSearchSpace",
    "ArchitectureParameterSpec",
    "ArchitectureBuilder",
    "AttentionNet",
    "EvolutionarySampler",
    "ENASSampler",
    "DARTSSampler",
    "DARTSPipelineRun",
    "DARTSCell",
    "DARTSSupernet",
    "OP_NAMES",
    "make_nas_module_factory",
]
