"""RFC-003 ε2 NAS：架构构建器（P2.7，P3.3.1 扩展 attention）。

将架构参数 dict 翻译为 nn.Module。

P2 支持：
- conv1d：1D 卷积网络（Conv1d + BatchNorm + Activation + Dropout 堆叠）
- rnn：循环神经网络（LSTM / GRU，可选双向）
- hybrid：conv1d + rnn 级联

P3.3.1 新增：
- attention：Transformer 风格架构（Multi-head attention + FFN 堆叠 + 分类头）

输入约定：
- conv1d / hybrid：input_shape = (channels, length)，模型 flatten 后接 Linear(num_classes)
- rnn：input_shape = (channels, length)，按 length 转置为 (length, channels) 输入 RNN
- attention：input_shape = (channels, length)，channels 作为 embedding_dim，length 作为序列长度
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .search_space import SUPPORTED_ACTIVATIONS, SUPPORTED_RNN_TYPES


def _activation_module(name: str) -> nn.Module:
    """根据激活函数名构造激活模块。"""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(
        f"Unsupported activation '{name}' (supported: {SUPPORTED_ACTIVATIONS})"
    )


def _rnn_cell(
    rnn_type: str,
    input_size: int,
    hidden_dim: int,
    n_layers: int,
    bidirectional: bool,
    dropout: float,
) -> nn.RNNBase:
    """构造 RNN 单元（LSTM / GRU）。"""
    rnn_type = (rnn_type or "lstm").lower()
    if rnn_type not in SUPPORTED_RNN_TYPES:
        raise ValueError(
            f"Unsupported rnn_type '{rnn_type}' (supported: {SUPPORTED_RNN_TYPES})"
        )
    rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
    return rnn_cls(
        input_size=input_size,
        hidden_size=hidden_dim,
        num_layers=n_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=dropout if n_layers > 1 else 0.0,
    )


class Conv1dNet(nn.Module):
    """1D 卷积网络：堆叠 n_layers 个 (Conv1d + BN + Activation + Dropout)。

    输入：(*, channels, length)
    输出：(*, num_classes)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_layers: int = 3,
        hidden_dim: int = 64,
        activation: str = "relu",
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(input_shape) < 1:
            raise ValueError(f"input_shape too short for Conv1dNet: {input_shape}")
        # 输入 channel = input_shape[0]（如 CSI 的 subcarrier 数）
        in_channels = int(input_shape[0])

        layers: List[nn.Module] = []
        prev_ch = in_channels
        pad = kernel_size // 2
        for i in range(n_layers):
            layers.append(nn.Conv1d(
                in_channels=prev_ch,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            ))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(_activation_module(activation))
            layers.append(nn.Dropout(dropout))
            prev_ch = hidden_dim

        self.features = nn.Sequential(*layers)
        # 自适应池化 + 分类头（避免依赖具体 length）
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入可能是 (*, channels, length) 或 (*, length, channels)，统一转成 (*, channels, length)
        if x.dim() >= 2:
            # 若最后一维看起来是 channels（小数值），尝试转置
            # 简化：假设输入已是 (*, channels, length)
            pass
        feat = self.features(x)
        pooled = self.pool(feat).squeeze(-1)  # (*, hidden_dim)
        return self.classifier(pooled)


class RNNNet(nn.Module):
    """RNN 网络（LSTM / GRU，可选双向）。

    输入：(*, channels, length) → 转置为 (*, length, channels) 输入 RNN
    输出：(*, num_classes)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_layers: int = 2,
        hidden_dim: int = 128,
        activation: str = "tanh",
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(input_shape) < 1:
            raise ValueError(f"input_shape too short for RNNNet: {input_shape}")
        input_size = int(input_shape[0])

        self.rnn = _rnn_cell(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        # 注意：LSTM/GRU 的 activation 由内部 gate 决定，外部 activation 仅用于分类头
        self.act = _activation_module(activation)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (*, channels, length) → (*, length, channels)
        if x.dim() >= 3:
            x = x.transpose(-1, -2)
        out, _ = self.rnn(x)
        # 取最后时刻 hidden state
        last = out[:, -1, :] if out.dim() >= 3 else out[-1]
        return self.classifier(self.act(last))


class HybridNet(nn.Module):
    """Hybrid 网络：Conv1d 特征提取 + RNN 时序建模。

    输入：(*, channels, length)
    输出：(*, num_classes)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_layers: int = 3,
        hidden_dim: int = 64,
        activation: str = "relu",
        kernel_size: int = 3,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(input_shape) < 1:
            raise ValueError(f"input_shape too short for HybridNet: {input_shape}")
        in_channels = int(input_shape[0])
        pad = kernel_size // 2

        conv_layers: List[nn.Module] = []
        prev_ch = in_channels
        for _ in range(n_layers):
            conv_layers.append(nn.Conv1d(
                in_channels=prev_ch,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            ))
            conv_layers.append(nn.BatchNorm1d(hidden_dim))
            conv_layers.append(_activation_module(activation))
            conv_layers.append(nn.Dropout(dropout))
            prev_ch = hidden_dim
        self.features = nn.Sequential(*conv_layers)

        # RNN 输入 size = hidden_dim
        self.rnn = _rnn_cell(
            rnn_type=rnn_type,
            input_size=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=max(1, n_layers // 2),
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.act = _activation_module(activation)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (*, channels, length)
        feat = self.features(x)  # (*, hidden_dim, length')
        # 转置为 (*, length', hidden_dim) 输入 RNN
        feat = feat.transpose(-1, -2)
        out, _ = self.rnn(feat)
        last = out[:, -1, :] if out.dim() >= 3 else out[-1]
        return self.classifier(self.act(last))


class AttentionNet(nn.Module):
    """Transformer 风格架构（P3.3.1 新增）。

    Multi-head attention + FFN 堆叠 + 分类头。

    输入：(*, channels, length) → 转置为 (*, length, channels) 作为 token 序列
    输出：(*, num_classes)

    Args:
        input_shape: (channels, length)，channels 作为 embedding_dim，length 作为序列长度
        num_classes: 输出类别数
        n_layers: Transformer encoder 层数
        d_model: embedding 维度（应等于 channels 或通过线性投影）
        n_heads: 多头注意力头数
        dropout: dropout 比率
        activation: FFN 激活函数（gelu/relu）
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_layers: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if len(input_shape) < 1:
            raise ValueError(f"input_shape too short for AttentionNet: {input_shape}")
        in_channels = int(input_shape[0])
        # 若 in_channels != d_model，添加投影层
        self.input_proj = (
            nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, channels, length) → (*, length, channels)
        x = x.transpose(-1, -2)
        x = self.input_proj(x)
        x = self.encoder(x)
        # 取 mean pooling 作为序列表示
        x = x.mean(dim=1)
        return self.classifier(x)


class ArchitectureBuilder:
    """架构构建器：将 arch_params 翻译为 nn.Module（P2.7，P3.3.1 扩展 attention）。

    根据 cell_type 分发到具体实现：
    - conv1d → Conv1dNet
    - rnn → RNNNet
    - hybrid → HybridNet
    - attention → AttentionNet（P3.3.1 新增）

    Args:
        arch_params: 架构参数 dict（由 SP Sampler 采样得到）
            必含 "cell_type" key，其余参数依 cell_type 而定
        input_shape: 输入张量形状（不含 batch 维），如 (channels, length)
        num_classes: 输出类别数

    Returns:
        nn.Module 实例（forward(x) → logits）

    Examples:
        >>> builder = ArchitectureBuilder()
        >>> model = builder.build(
        ...     {"cell_type": "conv1d", "n_layers": 3, "hidden_dim": 64,
        ...      "activation": "relu", "kernel_size": 3, "dropout": 0.1},
        ...     input_shape=(30, 100), num_classes=7,
        ... )
        >>> isinstance(model, nn.Module)
        True
    """

    def build(
        self,
        arch_params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> nn.Module:
        """根据 arch_params 构建 nn.Module。"""
        if "cell_type" not in arch_params:
            raise ValueError(
                "arch_params must contain 'cell_type' key, got: "
                f"{list(arch_params.keys())}"
            )

        cell_type = arch_params["cell_type"]
        if cell_type == "conv1d":
            return self._build_conv1d(arch_params, input_shape, num_classes)
        elif cell_type == "rnn":
            return self._build_rnn(arch_params, input_shape, num_classes)
        elif cell_type == "hybrid":
            return self._build_hybrid(arch_params, input_shape, num_classes)
        elif cell_type == "attention":
            return self._build_attention(arch_params, input_shape, num_classes)
        else:
            raise ValueError(
                f"Unsupported cell_type '{cell_type}' "
                f"(supported: conv1d, rnn, hybrid, attention)"
            )

    def _build_conv1d(
        self,
        arch_params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> Conv1dNet:
        """构建 Conv1dNet。"""
        return Conv1dNet(
            input_shape=input_shape,
            num_classes=num_classes,
            n_layers=int(arch_params.get("n_layers", 3)),
            hidden_dim=int(arch_params.get("hidden_dim", 64)),
            activation=arch_params.get("activation", "relu"),
            kernel_size=int(arch_params.get("kernel_size", 3)),
            dropout=float(arch_params.get("dropout", 0.1)),
        )

    def _build_rnn(
        self,
        arch_params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> RNNNet:
        """构建 RNNNet。"""
        return RNNNet(
            input_shape=input_shape,
            num_classes=num_classes,
            n_layers=int(arch_params.get("n_layers", 2)),
            hidden_dim=int(arch_params.get("hidden_dim", 128)),
            activation=arch_params.get("activation", "tanh"),
            rnn_type=arch_params.get("rnn_type", "lstm"),
            bidirectional=bool(arch_params.get("bidirectional", False)),
            dropout=float(arch_params.get("dropout", 0.1)),
        )

    def _build_hybrid(
        self,
        arch_params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> HybridNet:
        """构建 HybridNet。"""
        return HybridNet(
            input_shape=input_shape,
            num_classes=num_classes,
            n_layers=int(arch_params.get("n_layers", 3)),
            hidden_dim=int(arch_params.get("hidden_dim", 64)),
            activation=arch_params.get("activation", "relu"),
            kernel_size=int(arch_params.get("kernel_size", 3)),
            rnn_type=arch_params.get("rnn_type", "lstm"),
            bidirectional=bool(arch_params.get("bidirectional", False)),
            dropout=float(arch_params.get("dropout", 0.1)),
        )

    def _build_attention(
        self,
        arch_params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        num_classes: int,
    ) -> AttentionNet:
        """构建 AttentionNet（P3.3.1 新增）。"""
        return AttentionNet(
            input_shape=input_shape,
            num_classes=num_classes,
            n_layers=int(arch_params.get("n_layers", 4)),
            d_model=int(arch_params.get("d_model", 64)),
            n_heads=int(arch_params.get("n_heads", 4)),
            dropout=float(arch_params.get("dropout", 0.1)),
            activation=arch_params.get("activation", "gelu"),
        )


__all__ = [
    "ArchitectureBuilder",
    "Conv1dNet",
    "RNNNet",
    "HybridNet",
    "AttentionNet",
]
