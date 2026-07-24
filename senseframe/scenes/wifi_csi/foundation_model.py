"""P3 阶段 9：CSI 基础模型（MAE 自监督预训练）。

在大规模无标注 CSI 数据上预训练，在下游 HAR/ID 任务上通过 PEFT 微调。

设计要点：
- CSIPatchEmbedder: (B, C, L) -> (B, n_patches, d_model)，沿 L 切 patch
- CSIAttention: 含显式 query/key/value Linear 命名，便于 PEFTBuilder
  通过 peft_target_modules='query_value' 注入 LoRA/Adapter
- MAE 流程：patch embed + pos -> random mask -> encoder(visible) ->
  decoder(visible + mask_token, 还原顺序) -> 重建 masked patches
- loss: MSE on masked patches only
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Tuple

import torch
import torch.nn as nn

from ...automl.peft_builder import PEFTBuilder
from ...core.foundation_model import PEFTConfig, PretrainConfig

logger = logging.getLogger(__name__)


# ============================================================
# 基础层
# ============================================================
class CSIAttention(nn.Module):
    """Multi-head self-attention，显式 query/key/value 投影。

    模块命名为 query/value，使 PEFTBuilder._should_inject_lora 在
    peft_target_modules='query_value' 时能正确匹配并注入 LoRA/Adapter。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        # P3-4 修复：n_heads=0 会触发 ZeroDivisionError（d_model % n_heads）
        # 显式校验阻断，给出清晰错误信息。
        if n_heads < 1:
            raise ValueError(
                f"n_heads must be >= 1, got {n_heads}. "
                f"CSIAttention requires at least one attention head."
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.query(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.dropout(self.out(out))


class CSITransformerEncoderLayer(nn.Module):
    """Pre-LN transformer encoder layer。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CSIAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CSITransformerDecoderLayer(nn.Module):
    """Pre-LN transformer decoder layer（仅 self-attention，无 cross-attention）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CSIAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CSIPatchEmbedder(nn.Module):
    """CSI patch embedder。

    输入 (B, C, L) -> reshape (B, L, C) -> 切分为 (B, n_patches, patch_len, C)
    -> 展平 (B, n_patches, patch_len * C) -> Linear 投影 (B, n_patches, d_model)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        patch_len: int,
        d_model: int,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(
                f"input_shape must be (C, L) tuple, got shape={input_shape}"
            )
        C, L = input_shape
        if L % patch_len != 0:
            raise ValueError(
                f"L={L} must be divisible by patch_len={patch_len}"
            )
        self.C = C
        self.L = L
        self.patch_len = patch_len
        self.n_patches = L // patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len * C, d_model)

    def to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) -> (B, n_patches, patch_len * C)，不投影。"""
        B, C, L = x.shape
        x = x.permute(0, 2, 1).contiguous()  # (B, L, C)
        x = x.view(B, self.n_patches, self.patch_len, C)
        return x.reshape(B, self.n_patches, self.patch_len * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.to_patches(x))


# ============================================================
# CSIFoundationModel
# ============================================================
class CSIFoundationModel(nn.Module):
    """CSI 基础模型（MAE 自监督预训练）。

    实现 SensingFoundationModel Protocol。

    两阶段：
    - pretrain(): MAE 自监督预训练（mask + reconstruct）
    - encode(): 提取特征（用于下游任务）
    - get_peft_module(): 基于 PEFT 配置构建微调模块（深拷贝避免污染预训练权重）
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        patch_len: int = 16,
        decoder_dim: int = 64,
    ):
        super().__init__()
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self._model_id = "csi-mae-base"
        self._modality = "csi"
        self.input_shape = tuple(input_shape)
        self.d_model = d_model
        self.decoder_dim = decoder_dim
        self.patch_len = patch_len

        self.patch_embedder = CSIPatchEmbedder(input_shape, patch_len, d_model)
        n_patches = self.patch_embedder.n_patches

        # P3-P2-10 修复：原实现先 zeros 再 trunc_normal_，zeros 是冗余赋值。
        # 改用 torch.empty 跳过 zeros 写入，直接 trunc_normal_ 初始化。
        # 行为等价（trunc_normal_ 完全覆盖 zeros），但避免一次无意义 memset。
        self.pos_embed = nn.Parameter(torch.empty(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.Sequential(
            *[
                CSITransformerEncoderLayer(d_model, n_heads)
                for _ in range(n_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_embed = nn.Linear(d_model, decoder_dim)
        # P3-P2-10 修复：同 pos_embed，empty 替代冗余 zeros
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # P3-P2-10 修复：同 pos_embed，empty 替代冗余 zeros
        self.decoder_pos_embed = nn.Parameter(
            torch.empty(1, n_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder = nn.Sequential(
            *[
                CSITransformerDecoderLayer(decoder_dim, n_heads)
                for _ in range(n_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        C = self.patch_embedder.C
        pl = self.patch_embedder.patch_len
        self.decoder_proj = nn.Linear(decoder_dim, pl * C)

    # ---------- Protocol 属性 ----------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def n_patches(self) -> int:
        return self.patch_embedder.n_patches

    # ---------- MAE 核心 ----------
    def random_masking(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机 mask patches，返回 (x_visible, mask, ids_restore)。

        x: (B, n_patches, d_model)
        x_visible: (B, len_keep, d_model)，len_keep = int(N * (1 - mask_ratio))
        mask: (B, n_patches) float，1 = masked / 0 = visible，按原始顺序
        ids_restore: (B, n_patches)，用于 decoder 还原顺序
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore

    def _forward_encoder(self, x_visible: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x_visible)
        return self.encoder_norm(x)

    def _forward_decoder(
        self, x_visible: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """重建全部 patches (B, n_patches, patch_len * C)。"""
        B, L, _ = x_visible.shape
        N = ids_restore.shape[1]
        x = self.decoder_embed(x_visible)
        mask_tokens = self.mask_token.expand(B, N - L, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]),
        )
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_norm(x)
        return self.decoder_proj(x)

    def _mae_forward_loss(
        self, x: torch.Tensor, mask_ratio: float
    ) -> torch.Tensor:
        """MAE forward + MSE loss，仅在 masked patches 上计算。

        P3-P2-7 修复：to_patches(x) 与 patch_embedder(x) 共享 reshape 计算，
        原实现重复调用。现缓存 to_patches 结果作为重建 target。
        """
        # 缓存 to_patches 结果（重建 target，不含投影 + pos_embed）
        target = self.patch_embedder.to_patches(x)
        # patches 含投影 + pos_embed（进入 encoder）
        patches = self.patch_embedder.proj(target) + self.pos_embed
        x_visible, mask, ids_restore = self.random_masking(patches, mask_ratio)
        enc_out = self._forward_encoder(x_visible)
        recon = self._forward_decoder(enc_out, ids_restore)
        loss = (recon - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    # ---------- Protocol 方法 ----------
    def pretrain(self, unlabeled_data: Any, config: PretrainConfig) -> None:
        """MAE 自监督预训练。

        unlabeled_data: DataLoader 或可迭代对象，每个 batch 是 (x,) / (x, y) / x
        config: PretrainConfig（用 epochs / learning_rate / mask_ratio）

        P3-P2-8 修复：原实现无任何 epoch/batch loss 日志，训练过程不可观测。
        现按 epoch 打印 mean loss，便于调试与早停判断。
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=config.learning_rate
        )
        self.train()
        # P3-P2-8 修复：累积 epoch loss 用于日志
        for epoch in range(config.epochs):
            epoch_losses = []
            for batch in unlabeled_data:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                loss = self._mae_forward_loss(x, config.mask_ratio)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    "CSIFoundationModel pretrain epoch %d/%d: "
                    "mean_loss=%.6f, n_batches=%d, mask_ratio=%.2f",
                    epoch + 1, config.epochs, mean_loss,
                    len(epoch_losses), config.mask_ratio,
                )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征：(B, C, L) -> (B, n_patches, d_model)。"""
        patches = self.patch_embedder(x) + self.pos_embed
        return self._forward_encoder(patches)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征序列（供 DANN 等下游模块用，避免重复 forward）。

        与 encode() 的区别：
        - encode() 返回 (B, n_patches, d_model)，是 backbone 完整 forward
        - encode_features() 是 encode() 的别名，语义上强调"供下游用"

        Args:
            x: (B, C, L) 输入信号

        Returns:
            (B, n_patches, d_model) 特征序列
        """
        return self.encode(x)

    def get_peft_module(self, peft_config: PEFTConfig) -> nn.Module:
        """基于 PEFT 配置构建微调模块（深拷贝避免污染预训练权重）。"""
        foundation_copy = copy.deepcopy(self)
        params = (
            peft_config.__dict__
            if hasattr(peft_config, "__dict__")
            else dict(peft_config)
        )
        return PEFTBuilder.build(foundation_copy, params)

    def replace_patch_embedder(
        self,
        new_input_shape: Tuple[int, int],
        new_patch_len: int,
    ) -> None:
        """替换 patch_embedder + 相关 modality-specific 参数（跨模态迁移用）。

        用于 B5/B6 跨场景迁移：在 CSI 数据上 MAE 预训练后，替换 patch_embedder
        为目标模态（如 EEG）的维度，保留 transformer encoder + decoder 主体
        （modality-agnostic 部分）。

        重新初始化的模块（modality-specific，维度依赖 input_shape / patch_len）：
        - patch_embedder：proj 权重形状 (d_model, patch_len * C) 改变
        - pos_embed：n_patches 改变（L // patch_len）
        - decoder_pos_embed：n_patches 改变
        - decoder_proj：输出维度 (patch_len * C) 改变

        保留的模块（modality-agnostic，仅依赖 d_model / decoder_dim / n_heads）：
        - encoder / encoder_norm
        - decoder_embed / decoder / decoder_norm
        - mask_token

        Args:
            new_input_shape: 新模态的 (C, L)
            new_patch_len: 新模态的 patch_len

        Raises:
            ValueError: new_input_shape 不是 (C, L) 元组或 L 不能被 patch_len 整除
        """
        # 构建新 patch_embedder（CSIPatchEmbedder.__init__ 内部会校验 input_shape / patch_len）
        new_patch_embedder = CSIPatchEmbedder(
            new_input_shape, new_patch_len, self.d_model
        )
        new_n_patches = new_patch_embedder.n_patches

        # 替换 modality-specific 模块
        self.patch_embedder = new_patch_embedder
        self.input_shape = tuple(new_input_shape)
        self.patch_len = new_patch_len

        # 重新初始化 pos_embed（n_patches 改变）
        # 对齐 __init__ 的 P3-P2-10 优化：torch.empty + trunc_normal_，跳过冗余 zeros
        self.pos_embed = nn.Parameter(torch.empty(1, new_n_patches, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 重新初始化 decoder_pos_embed（n_patches 改变）
        self.decoder_pos_embed = nn.Parameter(
            torch.empty(1, new_n_patches, self.decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # 重新初始化 decoder_proj（输出维度 patch_len * C 改变）
        new_C = new_patch_embedder.C
        new_pl = new_patch_embedder.patch_len
        self.decoder_proj = nn.Linear(self.decoder_dim, new_pl * new_C)

        # encoder / encoder_norm / decoder_embed / mask_token / decoder / decoder_norm
        # 保持不变（modality-agnostic，跨模态迁移的核心价值所在）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward 调用 encode（供 PEFTBuilder 注入 LoRA 等模块时使用）。"""
        return self.encode(x)


__all__ = [
    "CSIFoundationModel",
    "CSIPatchEmbedder",
    "CSIAttention",
    "CSITransformerEncoderLayer",
    "CSITransformerDecoderLayer",
]
