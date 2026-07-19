"""P3 阶段 8：感知基础模型抽象。

每个场景可定义自己的基础模型（如 CSI-BERT / Radio-GPT / EEG-Transformer）。
基础模型在大规模无标注感知数据上自监督预训练，在下游任务上微调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn


@dataclass
class PretrainConfig:
    """自监督预训练配置。"""
    method: str = "mae"  # mae / simclr / byol
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    # P3-P2-9 修复：MAE 论文（He et al. 2022）推荐 mask_ratio=0.75，
    # 原默认值 0.5 偏离论文最优配置。75% mask 比例下 MAE 在 ImageNet 上
    # 取得最佳线性探测精度（论文表 5），且强制模型学习全局结构而非局部插值。
    mask_ratio: float = 0.75  # MAE 特有，论文推荐 0.75
    augmentations: List[str] = field(default_factory=list)


@dataclass
class PEFTConfig:
    """PEFT 微调配置（与搜索空间参数对齐）。"""
    peft_method: str = "lora"
    peft_rank: int = 8
    peft_alpha: int = 1
    peft_dropout: float = 0.0
    peft_target_modules: str = "query_value"
    adapter_bottleneck: int = 128
    prompt_length: int = 10
    freeze_backbone: bool = True


@runtime_checkable
class SensingFoundationModel(Protocol):
    """感知基础模型抽象（P3 阶段 8）。

    每个场景可定义自己的基础模型。基础模型在大规模无标注感知数据上
    自监督预训练，在下游任务上通过 PEFT 微调。

    实现者需提供：
    - model_id: 基础模型 ID（如 'csi-bert-base'）
    - modality: 模态（csi / radio / eeg / acoustic）
    - pretrain(): 自监督预训练
    - encode(): 特征提取
    - get_peft_module(): 基于 PEFT 配置构建微调模块
    """

    @property
    def model_id(self) -> str: ...

    @property
    def modality(self) -> str: ...

    def pretrain(self, unlabeled_data: Any, config: PretrainConfig) -> None: ...

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    def get_peft_module(self, peft_config: PEFTConfig) -> nn.Module: ...


__all__ = [
    "PretrainConfig",
    "PEFTConfig",
    "SensingFoundationModel",
]
