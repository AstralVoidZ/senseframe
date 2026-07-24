"""DANN 跨模态对齐单元测试。

测试原则：
- 全部用合成数据，快速运行（< 30 秒）
- 真实 forward pass / 真实梯度回传
- GRL 前向恒等 + 反向反转验证

参考：Ganin & Lempitsky, "Unsupervised Domain Adaptation by
Backpropagation", ICML 2015.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from senseframe.scenes.wifi_csi.dann import (
    GradientReversalLayer,
    grad_reverse,
    ModalityDiscriminator,
    DANNCrossModalModel,
    dann_lambda_schedule,
)


class TestGradientReversalLayer:
    """GRL 前向恒等 + 反向反转。"""

    def test_grl_forward_identity(self):
        """前向：输出 = 输入。"""
        x = torch.randn(4, 32, requires_grad=True)
        y = grad_reverse(x, lambda_=1.0)
        assert torch.allclose(y, x)

    def test_grl_backward_negate(self):
        """反向：梯度乘以 -λ。"""
        x = torch.randn(4, 32, requires_grad=True)
        y = grad_reverse(x, lambda_=1.0)
        y.sum().backward()
        # 反向梯度应为 -1.0（因为 sum 的梯度是 1，GRL 乘 -λ=-1）
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_grl_lambda_zero(self):
        """λ=0 时梯度为 0（无对抗）。"""
        x = torch.randn(4, 32, requires_grad=True)
        y = grad_reverse(x, lambda_=0.0)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_grl_lambda_half(self):
        """λ=0.5 时梯度乘以 -0.5。"""
        x = torch.randn(4, 32, requires_grad=True)
        y = grad_reverse(x, lambda_=0.5)
        y.sum().backward()
        assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


class TestModalityDiscriminator:
    """判别器输出形状 + 容量控制。"""

    def test_discriminator_output_shape_3d(self):
        """3D 输入 (B, n_patches, d_model) → (B, 2)。"""
        disc = ModalityDiscriminator(d_model=32, hidden_dim=16)
        x = torch.randn(4, 10, 32)  # (B, n_patches, d_model)
        out = disc(x)
        assert out.shape == (4, 2)

    def test_discriminator_mean_pooling_3d(self):
        """3D 输入自动 mean pooling 到 2D。"""
        # dropout=0.0 保证两次 forward 的随机性一致，可验证 mean pooling 等价
        disc = ModalityDiscriminator(d_model=32, hidden_dim=16, dropout=0.0)
        disc.eval()  # 双保险关闭 dropout
        x_3d = torch.randn(4, 10, 32)
        x_2d = x_3d.mean(dim=1)
        out_3d = disc(x_3d)
        out_2d = disc(x_2d)
        assert torch.allclose(out_3d, out_2d)

    def test_discriminator_2d_input(self):
        """2D 输入 (B, d_model) 直接判别（不 pooling）。"""
        disc = ModalityDiscriminator(d_model=32, hidden_dim=16)
        x = torch.randn(4, 32)
        out = disc(x)
        assert out.shape == (4, 2)


class TestDannLambdaSchedule:
    """DANN λ 调度（照搬原论文 2/(1+exp(-10*p))-1）。"""

    def test_lambda_at_epoch_zero(self):
        """epoch=0 时 λ=0（无对抗，专注任务学习）。"""
        assert dann_lambda_schedule(0, 100) == 0.0

    def test_lambda_at_final_epoch(self):
        """epoch=total 时 λ=1（满对抗）。"""
        lam = dann_lambda_schedule(100, 100)
        assert abs(lam - 1.0) < 0.01  # sigmoid 接近 1 但不等于 1

    def test_lambda_monotonic_increasing(self):
        """λ 随 epoch 单调递增。"""
        lambdas = [dann_lambda_schedule(e, 100) for e in range(0, 101, 10)]
        for i in range(1, len(lambdas)):
            assert lambdas[i] > lambdas[i-1]

    def test_lambda_at_half_epoch(self):
        """epoch=total/2 时 λ≈0.99（接近满对抗）。"""
        lam = dann_lambda_schedule(50, 100)
        assert 0.95 < lam < 1.0


class TestDANNCrossModalModel:
    """DANN 模型前向 + 训练流程。"""

    def _make_small_dann(self):
        """构造小规模 DANN 模型用于测试。"""
        from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
        from senseframe.scenes.wifi_csi.classifier import CSIClassifier
        # CSI backbone（小规模）
        backbone = CSIFoundationModel(
            input_shape=(3, 64), d_model=32, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=1,
            patch_len=8, decoder_dim=16,
        )
        # task_head：backbone + 分类头（DANNCrossModalModel 会绕过 task_head.forward，
        # 直接调 task_head.classifier，因为 backbone 已在 encode_features 中调过）
        task_head = CSIClassifier(backbone, d_model=32, num_classes=2)
        model = DANNCrossModalModel(
            backbone=backbone, task_head=task_head,
            d_model=32, hidden_dim=16, dropout=0.0,
        )
        return model

    def test_dann_forward_eeg_only_eval(self):
        """eval 模式下只传 EEG，返回 (logits, None)。"""
        model = self._make_small_dann()
        model.eval()
        x_eeg = torch.randn(4, 3, 64)
        logits, disc_loss = model(x_eeg)
        assert logits.shape == (4, 2)
        assert disc_loss is None

    def test_dann_forward_with_csi_train(self):
        """train 模式下传 EEG + CSI，返回 (logits, disc_loss)。"""
        model = self._make_small_dann()
        model.train()
        x_eeg = torch.randn(4, 3, 64)
        x_csi = torch.randn(4, 3, 64)
        logits, disc_loss = model(x_eeg, x_csi, lambda_=1.0)
        assert logits.shape == (4, 2)
        assert disc_loss is not None
        assert disc_loss.dim() == 0  # 标量

    def test_dann_eval_no_csi(self):
        """eval 模式下不传 CSI 也能正常 forward。"""
        model = self._make_small_dann()
        model.eval()
        x_eeg = torch.randn(2, 3, 64)
        logits, disc_loss = model(x_eeg, x_csi=None, lambda_=0.5)
        assert logits.shape == (2, 2)
        assert disc_loss is None


class TestDANNCrossModalEndToEnd:
    """端到端小规模验证。"""

    def test_dann_train_one_epoch(self):
        """合成数据跑通 1 epoch DANN 训练（3 个 batch）。"""
        import torch.nn.functional as F
        from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
        from senseframe.scenes.wifi_csi.classifier import CSIClassifier

        backbone = CSIFoundationModel(
            input_shape=(3, 64), d_model=32, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=1,
            patch_len=8, decoder_dim=16,
        )
        task_head = CSIClassifier(backbone, d_model=32, num_classes=2)
        model = DANNCrossModalModel(backbone, task_head, d_model=32, hidden_dim=16)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        total_loss = None
        for _ in range(3):  # 3 个 batch
            x_eeg = torch.randn(4, 3, 64)
            y_eeg = torch.zeros(4, dtype=torch.long)
            x_csi = torch.randn(4, 3, 64)
            logits, disc_loss = model(x_eeg, x_csi, lambda_=0.5)
            task_loss = F.cross_entropy(logits, y_eeg)
            total_loss = task_loss + disc_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        assert torch.isfinite(total_loss), "total_loss should be finite"

    def test_dann_eval_mode_no_disc(self):
        """eval 模式下不计算 disc_loss。"""
        from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
        from senseframe.scenes.wifi_csi.classifier import CSIClassifier

        backbone = CSIFoundationModel(
            input_shape=(3, 64), d_model=32, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=1,
            patch_len=8, decoder_dim=16,
        )
        task_head = CSIClassifier(backbone, d_model=32, num_classes=2)
        model = DANNCrossModalModel(backbone, task_head, d_model=32, hidden_dim=16)

        model.eval()
        with torch.no_grad():
            x_eeg = torch.randn(2, 3, 64)
            logits, disc_loss = model(x_eeg)
        assert disc_loss is None
        assert logits.shape == (2, 2)
