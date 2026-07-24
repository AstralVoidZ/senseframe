"""DANN 跨模态对齐模块（Domain Adaptation by Backpropagation）。

实现 DANN 三组件：
1. GradientReversalLayer (GRL)：前向恒等，反向乘以 -λ
2. ModalityDiscriminator：判别特征来自 CSI 还是 EEG（Task 6 追加）
3. DANNCrossModalModel：封装共享 backbone + 任务头 + 判别器（Task 9 追加）

参考：Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation", ICML 2015.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """梯度反转 autograd Function。

    前向恒等（output = input），反向梯度乘以 -λ。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 返回 (grad_output * -λ, None) —— None 对应 lambda_ 参数（不需要梯度）
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """GRL 函数式接口。

    Args:
        x: 输入 tensor
        lambda_: 对抗强度（0=无反转，1=满反转）

    Returns:
        与 x 形状相同的 tensor（前向恒等，反向反转）
    """
    return GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """GRL 的 nn.Module 包装，便于作为模型组件嵌入。

    用法：
        grl = GradientReversalLayer(lambda_=1.0)
        y = grl(x)  # 前向恒等，反向乘以 -λ

    与 grad_reverse 函数式接口等价，提供 Module 接口方便 nn.Sequential 组合。
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """动态更新 λ（DANN 训练中按 epoch 调度）。"""
        self.lambda_ = lambda_


class ModalityDiscriminator(nn.Module):
    """模态判别器：判别特征来自 CSI 还是 EEG（二分类）。

    输入：(B, n_patches, d_model) 特征序列
    输出：(B, 2) logits [CSI=0, EEG=1]

    设计要点：
    - 3D 输入自动 mean pooling 到 2D 再判别
    - 2 层 MLP + ReLU + Dropout，避免判别器过强
    - 判别器过强会导致 encoder 学不到任务特征，需控制容量
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 2 类：CSI=0, EEG=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3D → 2D mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


def dann_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """DANN 原论文 λ 调度：从 0 渐增到 1。

    λ = 2 / (1 + exp(-10 * p)) - 1
    其中 p = epoch / total_epochs

    论文引用：Ganin et al., "Unsupervised Domain Adaptation by
    Backpropagation", ICML 2015.

    Args:
        epoch: 当前 epoch（从 0 开始）
        total_epochs: 总 epoch 数

    Returns:
        λ 值（0.0 ~ 1.0）
    """
    if total_epochs <= 0:
        return 0.0
    p = epoch / total_epochs
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


class DANNCrossModalModel(nn.Module):
    """DANN 跨模态迁移模型。

    组成：
    - backbone: PEFTModel（EEG 模态，已 replace_patch_embedder）
    - task_head: CSIClassifier（EEG 分类头，含 backbone + classifier）
    - discriminator: ModalityDiscriminator（模态判别器）
    - csi_patch_embedder + csi_pos_embed: CSI 模态特定层（跨模态对齐用）

    双模态特征提取（关键设计）：
    - EEG 路径：backbone.encode_features(x_eeg) → eeg_feat
      backbone 内部 patch_embedder + pos_embed 是 EEG 模态（replace 后）
    - CSI 路径：csi_patch_embedder(x_csi) + csi_pos_embed → backbone.encoder → csi_feat
      CSI 模态特定层独立，但共享 backbone.encoder（modality-agnostic）
    - 对抗：discriminator 对 eeg_feat 和 csi_feat 做模态判别

    为什么不能共享 patch_embedder：
    - replace_patch_embedder 后 backbone 只能处理 EEG 输入
    - CSI 输入形状 (342, 2000) 无法通过 EEG patch_embedder (64, 480)
    - 必须保留 CSI 模态的 patch_embedder + pos_embed 用于 CSI 特征提取

    CSI 模态特定层始终 freeze（预训练好的，不应被对抗训练破坏）。

    λ 调度（照搬 DANN 原论文）：
    lambda_ = 2 / (1 + exp(-10 * p)) - 1
    其中 p = current_epoch / total_epochs
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_head: nn.Module,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        csi_patch_embedder: Optional[nn.Module] = None,
        csi_pos_embed: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.task_head = task_head
        self.discriminator = ModalityDiscriminator(d_model, hidden_dim, dropout)

        # DANN fine-tune 只用 encoder 特征（encode_features），decoder 是 MAE pretrain
        # 专用组件，不参与 fine-tune。freeze backbone decoder 相关参数，避免 optimizer
        # 为无梯度参数维护无用 Adam momentum/variance 显存浪费。
        # 覆盖：decoder.* / decoder_embed / decoder_norm / decoder_proj / decoder_pos_embed / mask_token
        # 穿透 PEFTModel 取内部 CSIFoundationModel，确保无论是否 PEFT 包装都能匹配裸参数名
        inner_backbone = self._get_inner_backbone()
        for name, param in inner_backbone.named_parameters():
            if name.startswith("decoder") or name == "mask_token":
                param.requires_grad = False

        # CSI 模态特定层（跨模态对齐用，始终 freeze）
        if csi_patch_embedder is not None:
            self.csi_patch_embedder = csi_patch_embedder
            for p in self.csi_patch_embedder.parameters():
                p.requires_grad = False
        else:
            self.csi_patch_embedder = None

        if csi_pos_embed is not None:
            # 包装为 Parameter 并 freeze
            self.csi_pos_embed = nn.Parameter(csi_pos_embed.detach().clone())
            self.csi_pos_embed.requires_grad = False
        else:
            self.csi_pos_embed = None

    def _get_inner_backbone(self) -> nn.Module:
        """穿透 PEFTModel 获取内部 CSIFoundationModel（访问 encoder/encoder_norm 用）。

        PEFTModel.backbone 是 CSIFoundationModel，PEFTModel 包装后 LoRA 注入到
        backbone.encoder 的 Linear 层（in-place）。所以 inner_backbone.encoder
        已包含 LoRA，CSI 路径通过它自动应用 LoRA。
        """
        if hasattr(self.backbone, "backbone"):
            return self.backbone.backbone  # PEFTModel.backbone
        return self.backbone  # 非 PEFT 包装（如 scratch/full）

    def _encode_csi_features(self, x_csi: torch.Tensor) -> torch.Tensor:
        """CSI 特征提取：csi_patch_embedder + csi_pos_embed → backbone.encoder。

        共享 backbone.encoder（modality-agnostic，含 LoRA），但 patch_embedder/pos_embed
        独立（CSI 模态特定，freeze）。

        若未提供 csi_patch_embedder（同模态场景，EEG 和 CSI 输入形状相同），
        fallback 到 backbone.encode_features(x_csi)，与旧测试兼容。

        Args:
            x_csi: (B, C_csi, L_csi) CSI 信号

        Returns:
            (B, n_patches_csi, d_model) CSI 特征序列
        """
        if self.csi_patch_embedder is None or self.csi_pos_embed is None:
            # 同模态 fallback：CSI 输入与 EEG 同形状，直接用 backbone 提取
            return self.backbone.encode_features(x_csi)
        inner = self._get_inner_backbone()
        # CSI patch embedding + pos_embed
        patches = self.csi_patch_embedder(x_csi) + self.csi_pos_embed
        # 共享 encoder（含 LoRA，modality-agnostic）
        enc_out = inner.encoder(patches)
        if hasattr(inner, "encoder_norm"):
            enc_out = inner.encoder_norm(enc_out)
        return enc_out

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_csi: Optional[torch.Tensor] = None,
        lambda_: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播。

        Args:
            x_eeg: (B, C_eeg, L_eeg) EEG 信号
            x_csi: (B, C_csi, L_csi) CSI 信号（仅训练时提供）
            lambda_: GRL 对抗强度（0=无对抗，1=满对抗）

        Returns:
            logits: (B, num_classes) EEG 分类 logits
            disc_loss: 模态判别损失（训练时返回，eval 时为 None）
        """
        # EEG 路径：任务头分类（只调一次 backbone）
        feat_eeg = self.backbone.encode_features(x_eeg)  # (B, n_patches_eeg, d_model)
        pooled_eeg = feat_eeg.mean(dim=1)  # (B, d_model)
        logits = self.task_head.classifier(pooled_eeg)

        # 对抗路径（仅训练时 + 提供了 CSI 时）
        disc_loss = None
        if self.training and x_csi is not None:
            # CSI 特征：独立 patch_embedder + 共享 encoder
            feat_csi = self._encode_csi_features(x_csi)  # (B, n_patches_csi, d_model)
            # GRL 反转梯度
            feat_eeg_rev = grad_reverse(feat_eeg, lambda_)
            feat_csi_rev = grad_reverse(feat_csi, lambda_)
            # 判别器
            disc_eeg = self.discriminator(feat_eeg_rev)  # (B, 2)
            disc_csi = self.discriminator(feat_csi_rev)  # (B, 2)
            # 标签：CSI=0, EEG=1
            disc_labels = torch.cat([
                torch.zeros(disc_csi.size(0), dtype=torch.long, device=disc_csi.device),
                torch.ones(disc_eeg.size(0), dtype=torch.long, device=disc_eeg.device),
            ])
            disc_logits = torch.cat([disc_csi, disc_eeg], dim=0)
            disc_loss = F.cross_entropy(disc_logits, disc_labels)

        return logits, disc_loss
