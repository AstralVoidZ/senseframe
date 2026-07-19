"""EEG 场景模型：EEGNet / DeepConvNet / TransformerEEG。

P1.2 落地：EEG 信号分类模型，验证 build_model_for_dataset 在 EEG 模态下的可移植性。

所有模型期望输入形状：(B, C, T)
- B: batch size
- C: 通道数（如 22 for BCI Competition IV-2a）
- T: 时间采样点（如 1000 for 4s @ 250Hz）
"""
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# EEGNet：轻量级 EEG 分类网络（Lawhern et al. 2018）
# ============================================================
class EEGNet(nn.Module):
    """EEGNet：紧凑型卷积 EEG 分类器。

    结构：
    - Temporal Conv (F1 filters, kernel=half sample rate)
    - Depthwise Conv (F1 * D filters, per-channel)
    - Separable Conv (F2 filters)
    - Classifier

    输入：(B, C, T)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 22, num_classes: int = 4,
                 f1: int = 8, d: int = 2, f2: int = 16,
                 t_kernel: int = 64, dropout: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        f2 = f1 * d  # depthwise 输出通道数

        # Block 1: Temporal conv + Depthwise conv
        self.conv_temporal = nn.Conv2d(1, f1, (1, t_kernel), padding=(0, t_kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv_depth = nn.Conv2d(f1, f2, (in_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable conv
        self.conv_sep = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # 分类头：AdaptiveAvgPool 把空间维度压到 1x1，输出 (B, f2, 1, 1) → flatten (B, f2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) → (B, 1, C, T) for Conv2d
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Block 1
        x = self.bn1(self.conv_temporal(x))
        x = self.bn2(self.conv_depth(x))
        x = nn.functional.elu(x)
        x = self.drop1(self.pool1(x))
        # Block 2
        x = self.bn3(self.conv_sep(x))
        x = nn.functional.elu(x)
        x = self.drop2(self.pool2(x))
        # 分类
        return self.classifier(x)


# ============================================================
# DeepConvNet：深度卷积 EEG 分类器（Schirrmeister et al. 2017）
# ============================================================
class DeepConvNet(nn.Module):
    """DeepConvNet：4 层卷积 + 池化的深度 EEG 分类器。

    输入：(B, C, T)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 22, num_classes: int = 4,
                 hidden_channels: int = 50, dropout: float = 0.5):
        super().__init__()
        # Block 1: temporal conv
        self.conv1 = nn.Conv2d(1, hidden_channels, (1, 25), padding=(0, 12))
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,
                               (in_channels, 1))
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.drop1 = nn.Dropout(dropout)

        # Block 2-4: spatial-temporal conv
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, (1, 25), padding=(0, 12))
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.drop2 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, (1, 25), padding=(0, 12))
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        self.pool3 = nn.MaxPool2d((1, 2))
        self.drop3 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(nn.functional.elu(self.bn2(self.conv2(
            nn.functional.elu(self.bn1(self.conv1(x))))))))
        x = self.drop2(self.pool2(nn.functional.elu(self.bn3(self.conv3(x)))))
        x = self.drop3(self.pool3(nn.functional.elu(self.bn4(self.conv4(x)))))
        # 通道维度平均后送入分类头
        x = x.mean(dim=-1)  # (B, C, 1)
        return self.classifier(x)


# ============================================================
# TransformerEEG：基于 Transformer Encoder 的 EEG 分类器
# ============================================================
class TransformerEEG(nn.Module):
    """Transformer EEG 分类器。

    结构：通道 embedding → TransformerEncoder → CLS token → Linear
    输入：(B, C, T)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 22, num_classes: int = 4,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 4,
                 patch_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        # Patch embedding：每个通道作为 token 序列
        self.patch_proj = nn.Linear(patch_size, d_model)
        # 通道 embedding（每个通道一个可学习 token）
        self.channel_embed = nn.Parameter(torch.zeros(1, in_channels, d_model))
        nn.init.normal_(self.channel_embed, std=0.02)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) → (B, C, T//P, P) → (B, C, d_model) via patch_proj
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        num_patches = T // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]
        # 平均池化 patch：每个通道作为一个 token
        x = x.reshape(B, C, num_patches, self.patch_size).mean(dim=2)  # (B, C, P)
        x = self.patch_proj(x)  # (B, C, d_model)
        x = x + self.channel_embed
        # 加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, C+1, d_model)
        # Transformer
        x = self.encoder(x)
        return self.classifier(x[:, 0])


# ============================================================
# 自监督预训练模型（SimCLR 风格）
# ============================================================
class EEGLowEncoder(nn.Module):
    """EEG 低层编码器（自监督预训练用）。

    基于 EEGNet 主体（去掉分类头），输出 (B, feature_dim) 表示。
    用于自监督 contrastive learning 预训练。
    """
    def __init__(self, in_channels: int = 22, feature_dim: int = 128,
                 f1: int = 8, d: int = 2, t_kernel: int = 64):
        super().__init__()
        f2 = f1 * d
        # 用 AdaptiveAvgPool 把空间维度压到 1x1，确保 Linear 输入维度确定
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, t_kernel), padding=(0, t_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (in_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f2, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.features(x)


# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    "EEGNet": EEGNet,
    "DeepConvNet": DeepConvNet,
    "TransformerEEG": TransformerEEG,
}

# 自监督模型（仅在 self_supervised 模式下使用）
SELFSUP_MODEL_REGISTRY = {
    "EEGLowEncoder": EEGLowEncoder,
}
