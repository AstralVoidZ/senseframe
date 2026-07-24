"""L3 算法行为测试：DANN 梯度反转行为。

锚点来源：Ganin & Lempitsky, 2015《Unsupervised Domain Adaptation by
Backpropagation》。

论文关键结论（作为测试锚点）：
- DANN 使用梯度反转层（GRL），λ 按训练进度从 0 → 1 增加（论文 §3.3）
- 训练时 task_loss + disc_loss 双 loss，eval 时只输出 logits（无 disc_loss）
- encoder 参数通过 task_loss 和 disc_loss 都更新（GRL 反传到 encoder）
- decoder 参数在 DANN fine-tune 时应冻结（只用 encoder 特征做域对齐）

实现说明：
- 优先用真实 DANNCrossModalModel 测试（无 SenseFi 依赖，可直接导入）
- 行为断言而非 MagicMock：真实前向 + 真实 backward，验证梯度流
- decoder freeze 断言：遍历 backbone 参数，decoder 相关参数 requires_grad=False
- 梯度流断言：disc_loss.backward() 后 encoder 参数 grad is not None
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from senseframe.scenes.wifi_csi.classifier import CSIClassifier
from senseframe.scenes.wifi_csi.dann import (
    DANNCrossModalModel,
    dann_lambda_schedule,
)
from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel


def _make_small_dann_model() -> DANNCrossModalModel:
    """构造小规模 DANNCrossModalModel 用于快速算法行为测试。

    backbone 与 task_head 共享 CSIFoundationModel（CSI/EEG 同形状，简化测试），
    不传 csi_patch_embedder（fallback 到 backbone.encode_features，同模态场景）。
    """
    backbone = CSIFoundationModel(
        input_shape=(3, 64), d_model=32, n_heads=4,
        n_encoder_layers=2, n_decoder_layers=1,
        patch_len=8, decoder_dim=16,
    )
    task_head = CSIClassifier(backbone, d_model=32, num_classes=2)
    return DANNCrossModalModel(
        backbone=backbone, task_head=task_head,
        d_model=32, hidden_dim=16, dropout=0.0,
    )


@pytest.mark.l3_algorithm
class TestDannGradientReversal:
    """验证 DANN 梯度反转行为符合 Ganin & Lempitsky, 2015。"""

    def test_forward_train_returns_differentiable_disc_loss(self):
        """L3 anchor: 训练模式 forward(x_eeg, x_csi, lambda_) 返回 (logits, disc_loss)，disc_loss 可微。

        锚点 Ganin & Lempitsky, 2015 §2：训练时同时优化任务损失与域判别损失，
        判别损失通过 GRL 反传到 encoder，使 encoder 学到域不变特征。
        行为断言：disc_loss.requires_grad=True（可微，支持反向传播）。
        """
        model = _make_small_dann_model()
        model.train()
        x_eeg = torch.randn(4, 3, 64)
        x_csi = torch.randn(4, 3, 64)

        logits, disc_loss = model(x_eeg, x_csi, lambda_=1.0)

        assert logits.shape == (4, 2), f"logits shape 应为 (4, 2)，实际 {logits.shape}"
        assert disc_loss is not None, "训练模式 + 提供 x_csi 时应返回 disc_loss"
        assert disc_loss.requires_grad, (
            "disc_loss 必须可微（GRL 反传依赖，论文 §2）"
        )
        assert disc_loss.dim() == 0, "disc_loss 应为标量"

    def test_forward_eval_returns_none_disc_loss(self):
        """L3 anchor: forward(x_eeg, None, 0.0) eval 模式返回 (logits, None)。

        锚点 Ganin & Lempitsky, 2015 §3.3：eval 阶段无域对齐，
        只输出任务 logits（无 disc_loss），判别器不参与推理。
        """
        model = _make_small_dann_model()
        model.eval()
        x_eeg = torch.randn(4, 3, 64)

        logits, disc_loss = model(x_eeg, x_csi=None, lambda_=0.0)

        assert logits.shape == (4, 2), f"logits shape 应为 (4, 2)，实际 {logits.shape}"
        assert disc_loss is None, (
            "eval 模式下应返回 (logits, None)，无 disc_loss（论文 §3.3）"
        )

    def test_task_loss_updates_task_head(self):
        """L3 anchor: task_loss.backward() 后 task_head 参数有梯度。

        锚点 Ganin & Lempitsky, 2015 §2：任务损失（task_loss）正常回传，
        更新任务头（classifier）参数，与标准分类训练一致。
        """
        model = _make_small_dann_model()
        model.train()
        x_eeg = torch.randn(4, 3, 64)
        y_eeg = torch.zeros(4, dtype=torch.long)

        logits, _ = model(x_eeg, None, lambda_=0.0)
        task_loss = F.cross_entropy(logits, y_eeg)
        task_loss.backward()

        assert model.task_head.classifier.weight.grad is not None, (
            "task_loss.backward() 后 task_head.classifier 参数应有梯度（论文 §2）"
        )

    def test_disc_loss_updates_encoder_via_gradient_reversal(self):
        """L3 anchor: disc_loss.backward() 后 encoder 参数有梯度（梯度反转）。

        锚点 Ganin & Lempitsky, 2015 §2-§3：GRL 前向恒等、反向乘以 -λ，
        disc_loss 通过 GRL 反传到 encoder，使 encoder 学到域不变特征。
        行为断言：disc_loss.backward() 后 encoder 参数 grad is not None
        （GRL 不阻断梯度，仅反转方向）。
        """
        model = _make_small_dann_model()
        model.train()
        x_eeg = torch.randn(4, 3, 64)
        x_csi = torch.randn(4, 3, 64)

        _, disc_loss = model(x_eeg, x_csi, lambda_=1.0)
        disc_loss.backward()

        encoder_param = model.backbone.encoder[0].attn.query.weight
        assert encoder_param.grad is not None, (
            "disc_loss.backward() 后 encoder 参数应有梯度（GRL 梯度反转，论文 §2）"
        )

    def test_decoder_frozen_during_dann_finetune(self):
        """L3 anchor: decoder 参数 requires_grad=False（DANN fine-tune 冻结 decoder）。

        锚点 Ganin & Lempitsky, 2015：DANN fine-tune 只用 encoder 特征做域对齐，
        decoder 是 MAE 预训练专用组件，不参与 fine-tune。
        行为断言：遍历 backbone 参数，所有 decoder 相关参数
        （decoder.* / decoder_embed / decoder_norm / decoder_proj /
        decoder_pos_embed / mask_token）requires_grad=False。
        """
        model = _make_small_dann_model()

        decoder_not_frozen = [
            name for name, param in model.backbone.named_parameters()
            if (name.startswith("decoder") or name == "mask_token")
            and param.requires_grad
        ]
        assert not decoder_not_frozen, (
            f"decoder 参数应被冻结（requires_grad=False），"
            f"但以下参数仍可训练: {decoder_not_frozen}"
        )

    def test_lambda_schedule_zero_to_one(self):
        """L3 anchor: λ 按训练进度从 0 → 1 增加（论文 §3.3）。

        锚点 Ganin & Lempitsky, 2015 §3.3：λ = 2/(1+exp(-10*p))-1，
        p = epoch/total_epochs，从 0 渐增到 1，逐步增强对抗强度
        （初期专注任务学习，后期强化域对齐）。
        """
        # epoch=0 → λ=0（无对抗，专注任务学习）
        assert dann_lambda_schedule(0, 100) == 0.0, (
            "epoch=0 时 λ 应为 0（论文 §3.3 渐进式对抗起点）"
        )
        # 末尾 epoch → λ≈1（满对抗）
        final_lambda = dann_lambda_schedule(100, 100)
        assert abs(final_lambda - 1.0) < 0.01, (
            f"epoch=total 时 λ 应接近 1，实际 {final_lambda}"
        )
        # 单调递增
        lambdas = [dann_lambda_schedule(e, 100) for e in range(0, 101, 10)]
        for i in range(1, len(lambdas)):
            assert lambdas[i] > lambdas[i - 1], (
                "λ 应随 epoch 单调递增（论文 §3.3 渐进式对抗）"
            )
