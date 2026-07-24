"""CSI 分类头：基础模型 backbone + mean pooling + Linear 分类。

从 scripts/p3_eval_common.py 迁入 senseframe 包（I22 修复），让
senseframe.scenes.wifi_csi 域内自治，tests 不再依赖 scripts 模块。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CSIClassifier(nn.Module):
    """CSI 分类器：backbone + mean pooling + Linear 分类头。

    backbone 可以是：
    - CSIFoundationModel（scratch / full 微调）
    - PEFTModel（LoRA / Adapter / Prefix / Prompt 微调，包装 backbone）

    forward 流程：
    1. backbone(x) → (B, n_patches, d_model) 特征序列
    2. mean pooling → (B, d_model) 全局特征
    3. Linear(d_model, num_classes) → (B, num_classes) logits
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone 返回 (B, n_patches, d_model)
        features = self.backbone(x)
        # mean pooling over patches
        pooled = features.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)  # (B, num_classes)
