"""Stub 模型：DANN / MAE 模型的可预测替身。

Stub 与 Fake 的区别：
- Fake（FakeTrainer 等）：有状态，实现协议语义，用于编排测试
- Stub（StubDannModel 等）：无状态或最小状态，返回固定可预测值，用于算法行为测试

协议来源：
- StubDannModel: senseframe.scenes.wifi_csi.dann.DANNCrossModalModel.forward
- StubMaeModel: senseframe.scenes.wifi_csi.foundation_model.CSIFoundationModel.mae_reconstruct
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class StubDannModel(nn.Module):
    """Stub DANN 模型：forward 返回可预测的 (logits, disc_loss)。

    替代 tests/test_stage_train_dann_loop.py 中的 _DummyDannModel，
    增加梯度反转行为模拟（disc_loss 可微）。

    协议（DANNCrossModalModel.forward）：
        forward(x_eeg, x_csi=None, lambda_=0.0) -> (logits, disc_loss)
        - logits: (B, num_classes) 分类 logits
        - disc_loss: 训练模式 + 提供 x_csi 时返回可微标量；eval/无 x_csi 时返回 None

    Attributes:
        fc: 线性分类头（参数可训练，用于梯度流测试）
        decoder: 模拟 decoder 参数（用于 decoder freeze 测试，默认 requires_grad=True）
    """

    def __init__(self, in_features: int = 10, num_classes: int = 7) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        # 模拟 decoder 参数（DANN decoder freeze 测试依赖）
        self.decoder = nn.Linear(in_features, in_features)
        self.decoder_norm = nn.LayerNorm(in_features)
        self.mask_token = nn.Parameter(torch.zeros(in_features))

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_csi: Optional[torch.Tensor] = None,
        lambda_: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """DANN 前向：返回 (logits, disc_loss)。

        Args:
            x_eeg: (B, in_features) EEG 特征
            x_csi: (B, in_features) CSI 特征（可选，训练时提供）
            lambda_: GRL 对抗强度（Stub 忽略，真实模型用于梯度反转）

        Returns:
            (logits, disc_loss) — logits: (B, num_classes)
            disc_loss: 训练模式 + x_csi 提供时返回可微标量 0.0；否则 None
        """
        logits = self.fc(x_eeg)
        if self.training and x_csi is not None:
            disc_loss = torch.tensor(0.0, requires_grad=True)
            return logits, disc_loss
        return logits, None

    def _get_inner_backbone(self) -> "StubDannModel":
        """模拟 PEFTModel._get_inner_backbone()，返回自身（用于 decoder freeze 测试）。"""
        return self


class StubMaeModel(nn.Module):
    """Stub MAE 模型：mae_reconstruct 返回可预测的 (recon, target, mask)。

    替代 tests/test_psnr_callback_wiring.py 中的 MockMaeModel，
    消除 MagicMock 依赖，提供可预测的张量输出。

    协议（CSIFoundationModel.mae_reconstruct）：
        mae_reconstruct(x, mask_ratio) -> (recon, target, mask)
        - recon: (B, n_patches, patch_dim) 重建张量
        - target: (B, n_patches, patch_dim) 原始 patches
        - mask: (B, n_patches) float，1 = masked / 0 = visible

    Attributes:
        _mask_ratio: 默认 mask_ratio（He 2022 论文默认 0.75）
    """

    def __init__(self, mask_ratio: float = 0.75, n_patches: int = 10, patch_dim: int = 8) -> None:
        super().__init__()
        self._mask_ratio: float = mask_ratio
        self._n_patches = n_patches
        self._patch_dim = patch_dim
        # 可训练参数（用于梯度流测试）
        self.proj = nn.Linear(patch_dim, patch_dim)

    def mae_reconstruct(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE 重建（返回可预测的固定张量）。

        Args:
            x: (B, C, L) 输入信号（Stub 忽略内容，只用 batch size）
            mask_ratio: mask 比例（Stub 按此比例生成 mask）

        Returns:
            (recon, target, mask) 三元组：
            - recon: (B, n_patches, patch_dim) 重建张量
            - target: (B, n_patches, patch_dim) 原始 patches
            - mask: (B, n_patches) float，1 = masked / 0 = visible
        """
        batch_size = x.shape[0] if x.dim() > 0 else 1
        # 生成可预测的 target 和 recon
        target = torch.randn(batch_size, self._n_patches, self._patch_dim)
        recon = self.proj(target)  # 通过可训练层，支持梯度流
        # 生成 mask：按 mask_ratio 比例标记 masked
        n_masked = int(self._n_patches * mask_ratio)
        mask = torch.zeros(batch_size, self._n_patches)
        mask[:, :n_masked] = 1.0  # 前 n_masked 个 patch 为 masked
        return recon, target, mask

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        flag: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MAE 模型前向（返回二元组供 ce_criterion 消费）。

        Args:
            x1: 主输入
            x2: 辅助输入（可选）
            flag: 前向标志（可选）

        Returns:
            (logits, aux_logits) 二元组
        """
        batch_size = x1.shape[0] if x1.dim() > 0 else 1
        logits = torch.randn(batch_size, 7)
        aux_logits = torch.randn(batch_size, 7)
        return logits, aux_logits
