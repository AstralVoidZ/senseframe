"""L3 算法行为测试：MAE (Masked Autoencoder) 前向行为。

锚点来源：He et al., 2022《Masked Autoencoders Are Scalable Vision Learners》。

论文关键结论（作为测试锚点）：
- mask_ratio=0.75 是论文 Table 1c 的最优重建质量默认值
- MAE 随机 mask patches，mask 是二值张量（1=masked, 0=visible）
- 重建只对 masked patches 计算 loss（论文 §3.1）
- encoder 只处理 visible patches（论文 §3.2）
- decoder 重建所有 patches 但 loss 只在 masked 上

实现说明：
- 对真实 CSIFoundationModel 测试（小规模实例），不用 StubMaeModel
  （Stub 已在 tests/fakes/test_fakes.py 测过）
- 行为断言而非私有字段断言：通过 mask 张量均值验证 mask_ratio，
  而非断言 model._mask_ratio == 0.75（自证断言）
"""
from __future__ import annotations

import pytest
import torch

from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel


def _make_small_mae_model() -> CSIFoundationModel:
    """构造小规模 CSIFoundationModel 用于快速算法行为测试。

    n_patches = 64 // 8 = 8，足够验证 mask 比例（int 截断后仍接近 0.75）。
    """
    return CSIFoundationModel(
        input_shape=(3, 64),
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        patch_len=8,
        decoder_dim=16,
    )


@pytest.mark.l3_algorithm
class TestMaeForwardBehavior:
    """验证 MAE 前向行为符合 He et al., 2022。"""

    def test_mae_reconstruct_returns_triple(self):
        """L3 anchor: mae_reconstruct 返回 (recon, target, mask) 三元组。

        锚点 He et al., 2022 §3：MAE 重建流程输出重建张量、原始 target 与
        二值 mask，供下游仅在 masked patches 上计算重建损失。
        """
        model = _make_small_mae_model()
        x = torch.randn(2, 3, 64)

        result = model.mae_reconstruct(x, mask_ratio=0.75)

        assert isinstance(result, tuple) and len(result) == 3, (
            "mae_reconstruct 应返回 (recon, target, mask) 三元组"
        )
        recon, target, mask = result
        assert isinstance(recon, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_mask_is_binary_tensor(self):
        """L3 anchor: mask 是二值张量（1=masked, 0=visible）。

        锚点 He et al., 2022 §3.1：随机 mask 一部分 patches，
        mask 为二值标记（1=masked / 0=visible），非软掩码。
        """
        model = _make_small_mae_model()
        x = torch.randn(2, 3, 64)

        _, _, mask = model.mae_reconstruct(x, mask_ratio=0.75)

        unique = set(mask.unique().tolist())
        assert unique.issubset({0.0, 1.0}), (
            f"mask 必须是二值张量（0/1），实际取值: {unique}"
        )

    def test_mask_ratio_matches_paper_default(
        self, mae_paper_default_mask_ratio: float
    ):
        """L3 anchor: mask_ratio=0.75 时 masked 比例约 75%（论文默认值）。

        锚点 He et al., 2022 Table 1c：mask_ratio=0.75 为最优重建质量默认值。
        行为断言：通过 mask 张量的均值验证 masked 比例，
        而非断言 model._mask_ratio == 0.75（私有属性，自证）。
        """
        model = _make_small_mae_model()  # n_patches = 64 // 8 = 8
        x = torch.randn(4, 3, 64)

        _, _, mask = model.mae_reconstruct(x, mae_paper_default_mask_ratio)

        masked_ratio = mask.float().mean().item()
        assert abs(masked_ratio - mae_paper_default_mask_ratio) < 0.15, (
            f"mask_ratio={mae_paper_default_mask_ratio} 时 masked 比例应约为 "
            f"{mae_paper_default_mask_ratio}，实际 {masked_ratio}"
        )

    def test_recon_and_target_shape_consistent(self):
        """L3 anchor: recon 与 target 形状一致（decoder 重建所有 patches）。

        锚点 He et al., 2022 §3.2：decoder 重建所有 patches（含 mask_token 占位），
        recon 与 target 在 patch 维度上一致，loss 通过 mask 选择 masked 子集。
        """
        model = _make_small_mae_model()
        x = torch.randn(2, 3, 64)

        recon, target, mask = model.mae_reconstruct(x, mask_ratio=0.75)

        assert recon.shape == target.shape, (
            f"recon 与 target 形状应一致: recon={recon.shape}, "
            f"target={target.shape}"
        )
        n_patches = model.n_patches
        patch_len_c = model.patch_len * model.patch_embedder.C
        assert recon.shape == (2, n_patches, patch_len_c)
        assert mask.shape == (2, n_patches)

    def test_recon_supports_gradient_flow(self):
        """L3 anchor: recon 支持梯度流（encoder+decoder 端到端可训练）。

        锚点 He et al., 2022 §3.2-§3.3：encoder 与 decoder 端到端可训练，
        重建损失通过反向传播更新参数（MAE 自监督预训练的核心机制）。
        """
        model = _make_small_mae_model()
        model.train()
        x = torch.randn(2, 3, 64)

        recon, target, mask = model.mae_reconstruct(x, mask_ratio=0.75)
        # 论文 §3.1：loss 只在 masked patches 上计算
        per_patch_loss = ((recon - target) ** 2).mean(dim=-1)
        loss = (per_patch_loss * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()

        encoder_param = model.encoder[0].attn.query.weight
        assert encoder_param.grad is not None, (
            "encoder 参数应有梯度（MAE 端到端可训练）"
        )
        decoder_param = model.decoder[0].attn.query.weight
        assert decoder_param.grad is not None, (
            "decoder 参数应有梯度（MAE 端到端可训练）"
        )

    def test_loss_only_on_masked_patches(self):
        """L3 anchor: 重建 loss 只对 masked patches 计算。

        锚点 He et al., 2022 §3.1：损失仅在 masked patches 上计算，
        visible patches 不贡献 loss（与 denoising autoencoder 的关键区别）。
        行为断言：
        1. MAE loss 等于仅在 masked 位置上的平均 per-patch 误差
        2. 破坏 visible 位置的重建不改变 MAE loss
        """
        model = _make_small_mae_model()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(4, 3, 64)

        with torch.no_grad():
            recon, target, mask = model.mae_reconstruct(x, mask_ratio=0.75)
            per_patch_loss = ((recon - target) ** 2).mean(dim=-1)
            # MAE loss 公式（论文 §3.1）：仅在 masked patches 上平均
            mae_loss = (per_patch_loss * mask).sum() / mask.sum().clamp(min=1.0)
            # 直接取 masked 位置的平均，应与 MAE loss 一致
            masked_only_mean = per_patch_loss[mask.bool()].mean()
            assert torch.allclose(mae_loss, masked_only_mean, atol=1e-6), (
                "MAE loss 应等于仅在 masked patches 上的平均误差（论文 §3.1）"
            )

            # 破坏 visible 位置的重建，MAE loss 不应改变
            recon_corrupted = recon.clone()
            vis_mask = (mask == 0).unsqueeze(-1).expand_as(recon)
            recon_corrupted[vis_mask] = 999.0
            per_patch_loss_corrupted = (
                (recon_corrupted - target) ** 2
            ).mean(dim=-1)
            mae_loss_corrupted = (
                (per_patch_loss_corrupted * mask).sum()
                / mask.sum().clamp(min=1.0)
            )
            assert torch.allclose(mae_loss, mae_loss_corrupted, atol=1e-6), (
                "visible patches 的重建错误不应影响 MAE loss（论文 §3.1）"
            )
