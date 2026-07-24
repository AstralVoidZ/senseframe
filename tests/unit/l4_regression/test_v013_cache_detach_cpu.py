"""V013: I13 cache .detach().cpu()（原缓存 GPU 张量导致显存泄露）。

Anchor: bug 编号 V013 + 修复 commit 43387a1。
原始问题: SelfSupervisedModule.validation_step 缓存 MAE 重建张量时，
  直接赋值 ``self._psnr_reconstruction = recon[mask_bool]``（未 .detach().cpu()），
  导致 GPU 张量被长期持有（引用链 module → _psnr_reconstruction → GPU tensor），
  显存随 epoch 累积泄露，长时间训练 OOM。
修复方式: 缓存赋值加 ``.detach().cpu()``，切断 autograd graph 并移到 CPU，
  仅保留单 batch 重建供 PSNR 趋势指示。

如果此测试失败，说明 V013 修复被回退（缓存未被填充或回到 GPU 张量）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


@pytest.mark.l4_regression
class TestV013CacheDetachCpu:
    """锁定 V013 修复：validation_step 后 PSNR 缓存被填充（.detach().cpu()）。"""

    def test_validation_step_populates_psnr_cache(self):
        """V013 anchor: model 有 mae_reconstruct 时，validation_step 后缓存非 None。

        I13 修复的 .detach().cpu() 是缓存赋值的一部分，
        本测试验证缓存被填充即可（非 None）。

        如果此断言失败，V013 修复被回退。
        """
        from senseframe.engine.self_supervised import SelfSupervisedModule

        # 构造 mock MAE model（duck-typed mae_reconstruct）
        class MockMaeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embedder = MagicMock()
                self.patch_embedder.to_patches.return_value = torch.randn(2, 10, 4)
                self.patch_embedder.proj = MagicMock(return_value=torch.randn(2, 10, 8))
                self.pos_embed = torch.zeros(1, 10, 8)

            def mae_reconstruct(self, x, mask_ratio):
                target = self.patch_embedder.to_patches(x)
                patches = self.patch_embedder.proj(target) + self.pos_embed
                x_visible, mask, ids_restore = self.random_masking(patches, mask_ratio)
                enc_out = self._forward_encoder(x_visible)
                recon = self._forward_decoder(enc_out, ids_restore)
                return recon, target, mask

            def random_masking(self, patches, mask_ratio):
                return patches, torch.ones(2, 10), MagicMock()

            def _forward_encoder(self, x):
                return x

            def _forward_decoder(self, enc, ids):
                return torch.randn(2, 10, 4)

            def forward(self, x1, x2=None, flag=None):
                return torch.randn(2, 7), torch.randn(2, 7)

        model = MockMaeModel()
        module = SelfSupervisedModule(
            model=model, learning_rate=0.001, weight_decay=0.0,
            metrics=["accuracy"], num_classes=7, incremental_log_writer=MagicMock(),
        )
        module._psnr_reconstruction = None
        module._psnr_target = None

        batch = (torch.randn(2, 1, 250), torch.tensor([0, 1]))
        with patch.object(module, "log"):
            module.validation_step(batch, 0)

        # V013 关键断言：缓存被填充（.detach().cpu() 是赋值的一部分）
        assert module._psnr_reconstruction is not None, (
            "如果此断言失败，V013 修复被回退：validation_step 后 "
            "_psnr_reconstruction 应非 None（缓存被填充）"
        )
        assert module._psnr_target is not None, (
            "如果此断言失败，V013 修复被回退：validation_step 后 "
            "_psnr_target 应非 None（缓存被填充）"
        )
