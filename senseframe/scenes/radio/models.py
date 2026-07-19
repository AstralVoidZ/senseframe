"""Radio 场景模型：ResNet1D / CNN1D / Transformer1D。

P1.2 落地：1D 信号模型，验证 build_model_for_dataset 在无线电信号模态下的可移植性。

所有模型期望输入形状：(B, C, L)
- B: batch size
- C: 通道数（IQ=2 / 复数 magnitude=1 / 时频图=2）
- L: 信号长度
"""
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# CNN1D：3 层 1D 卷积 + 全局平均池化
# ============================================================
class CNN1D(nn.Module):
    """简单 1D CNN 调制识别器。

    结构：Conv1d × 3 + BN + ReLU + MaxPool → GlobalAvgPool → Linear
    输入：(B, C, L)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 24,
                 hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 (B, L) 输入，自动 unsqueeze 为 (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# ResNet1D：基于残差块的 1D ResNet
# ============================================================
class _ResBlock1D(nn.Module):
    """1D 残差块。"""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    """1D ResNet 调制识别器。

    结构：Stem Conv → ResBlock × N → GlobalAvgPool → Linear
    输入：(B, C, L)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 24,
                 hidden_channels: int = 64, num_blocks: int = 3, dropout: float = 0.3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # 残差块
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_ResBlock1D(hidden_channels))
        self.blocks = nn.Sequential(*blocks)
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


# ============================================================
# Transformer1D：基于 Transformer Encoder 的 1D 序列模型
# ============================================================
class Transformer1D(nn.Module):
    """Transformer Encoder 调制识别器。

    结构：Patch embedding → TransformerEncoder → CLS token → Linear
    输入：(B, C, L)
    输出：(B, num_classes)
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 24,
                 d_model: int = 128, n_heads: int = 4, num_layers: int = 4,
                 patch_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        # Patch embedding：将 (C, L) 切分为 (L/P, C*P) 然后线性映射到 d_model
        self.patch_proj = nn.Linear(in_channels * patch_size, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # 位置编码（最大长度 1024 patches）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1025, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, C, L = x.shape
        # 切分 patch: (B, C, L) → (B, num_patches, C*P)
        num_patches = L // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]  # 截断到 patch 整数倍
        x = x.reshape(B, C, num_patches, self.patch_size)  # (B, C, P, patch)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, C * self.patch_size)
        x = self.patch_proj(x)  # (B, P, d_model)
        # 加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, P+1, d_model)
        x = x + self.pos_embed[:, :x.size(1)]
        # Transformer
        x = self.encoder(x)
        # 取 CLS token 输出
        return self.classifier(x[:, 0])


# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    "CNN1D": CNN1D,
    "ResNet1D": ResNet1D,
    "Transformer1D": Transformer1D,
}
