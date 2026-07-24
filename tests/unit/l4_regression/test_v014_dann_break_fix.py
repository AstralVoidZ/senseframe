"""V014: I4 DANN break 修复 + decoder freeze 真实 bug。

Anchor: bug 编号 V014 + 修复 commit 7285324。
原始问题: DANNCrossModalModel 初始化时未 freeze backbone decoder 参数，
  导致 DANN fine-tune 时 optimizer 为无梯度参数维护无用 Adam momentum/variance，
  显存浪费；且 decoder 参数意外参与对抗训练，破坏 MAE 预训练特征。
修复方式: __init__ 中遍历 inner_backbone 参数，将 decoder 相关参数
  （decoder.* / decoder_embed / decoder_norm / decoder_proj /
  decoder_pos_embed / mask_token）requires_grad=False。

如果此测试失败，说明 V014 修复被回退（decoder 未 freeze 或 encoder 无梯度）。
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.l4_regression
class TestV014DannBreakFix:
    """锁定 V014 修复：decoder freeze + encoder 梯度流。"""

    @staticmethod
    def _make_small_dann():
        """构造小规模 DANN 模型用于测试（复用 test_dann_cross_modal 构造方式）。"""
        from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
        from senseframe.scenes.wifi_csi.classifier import CSIClassifier
        from senseframe.scenes.wifi_csi.dann import DANNCrossModalModel

        backbone = CSIFoundationModel(
            input_shape=(3, 64), d_model=32, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=1,
            patch_len=8, decoder_dim=16,
        )
        task_head = CSIClassifier(backbone, d_model=32, num_classes=2)
        model = DANNCrossModalModel(
            backbone=backbone, task_head=task_head,
            d_model=32, hidden_dim=16, dropout=0.0,
        )
        return model

    def test_decoder_frozen_and_encoder_has_grad(self):
        """V014 anchor: decoder 参数 requires_grad=False，encoder 参数有梯度。

        如果此断言失败，V014 修复被回退。
        """
        model = self._make_small_dann()
        model.train()
        x_eeg = torch.randn(4, 3, 64)
        x_csi = torch.randn(4, 3, 64)
        logits, disc_loss = model(x_eeg, x_csi, lambda_=1.0)
        disc_loss.backward()

        # V014 关键断言 1：decoder 相关参数应被 freeze（requires_grad=False）
        # 覆盖：decoder.* / decoder_embed / decoder_norm / decoder_proj / decoder_pos_embed / mask_token
        decoder_not_frozen = [
            name for name, param in model.backbone.named_parameters()
            if (name.startswith("decoder") or name == "mask_token")
            and param.requires_grad
        ]
        assert not decoder_not_frozen, (
            f"如果此断言失败，V014 修复被回退：decoder params should be frozen "
            f"(requires_grad=False) but got requires_grad=True: {decoder_not_frozen}"
        )

        # V014 关键断言 2：encoder 相关参数应有梯度（requires_grad=True 且 grad is not None）
        encoder_missing = [
            name for name, param in model.backbone.named_parameters()
            if param.requires_grad and param.grad is None
            and not name.startswith("decoder")
            and name != "mask_token"
        ]
        assert not encoder_missing, (
            f"如果此断言失败，V014 修复被回退：encoder params without grad: "
            f"{encoder_missing}"
        )
