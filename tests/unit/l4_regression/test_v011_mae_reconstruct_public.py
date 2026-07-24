"""V011: mae_reconstruct 公共方法。

Anchor: bug 编号 I11 + 修复 commit 43387a1。
原始问题: self_supervised.py 从类外调用 _forward_encoder/_forward_decoder 私有方法
         破坏封装，MAE 重建流程在三处重复（self_supervised.py /
         _mae_forward_loss / p0_pretrain_with_psnr.py）。
修复方式: 暴露 mae_reconstruct 公共方法统一 MAE 重建流程，
         返回 (recon, target, mask) 三个张量。

如果此测试失败，说明 V011 修复被回退。
"""
from __future__ import annotations

import torch

import pytest

from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel


@pytest.mark.l4_regression
class TestV011MaeReconstructPublic:
    """锁定 V011 修复：mae_reconstruct 返回 (recon, target, mask) 三张量。"""

    def test_mae_reconstruct_returns_valid_tensors(self):
        """V011 anchor: mae_reconstruct(x, mask_ratio) 返回 (recon, target, mask) 三张量。"""
        model = CSIFoundationModel(
            input_shape=(3, 32),
            d_model=16,
            n_heads=2,
            patch_len=16,
            decoder_dim=8,
        )
        x = torch.randn(2, 3, 32)

        recon, target, mask = model.mae_reconstruct(x, mask_ratio=0.75)

        # 验证 shape
        n_patches = model.n_patches  # 32 // 16 = 2
        patch_len_C = 16 * 3  # 48
        assert recon.shape == (2, n_patches, patch_len_C), (
            "如果此断言失败，V011 修复被回退：recon shape 不符"
        )
        assert target.shape == (2, n_patches, patch_len_C), (
            "如果此断言失败，V011 修复被回退：target shape 不符"
        )
        assert mask.shape == (2, n_patches), (
            "如果此断言失败，V011 修复被回退：mask shape 不符"
        )
        # mask 应含 0 和 1
        assert mask.min() >= 0, (
            "如果此断言失败，V011 修复被回退：mask 最小值应 >= 0"
        )
        assert mask.max() <= 1, (
            "如果此断言失败，V011 修复被回退：mask 最大值应 <= 1"
        )
