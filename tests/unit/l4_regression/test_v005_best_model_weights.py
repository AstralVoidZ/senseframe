"""V005: DANN best model 权重保存/加载。

Anchor: bug 编号 I2 + 修复 commit 83a7ce2。
原始问题: 训练结束后 model 持有末轮权重而非 best epoch 权重，
         导致 final_eval 指标与 model 权重不匹配。
修复方式: best epoch 时保存 state_dict 副本（clone 避免引用共享），
         循环结束后 load_state_dict 加载回 model。

如果此测试失败，说明 V005 修复被回退。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest


class _PerEpochValLoader:
    """每次 __iter__ 返回不同 epoch 的 val 数据。"""

    def __init__(self, epoch_data_list):
        self.epoch_data_list = epoch_data_list
        self.epoch_idx = 0

    def __iter__(self):
        data = self.epoch_data_list[self.epoch_idx % len(self.epoch_data_list)]
        self.epoch_idx += 1
        return iter([data])


class _WeightTrackingModel(nn.Module):
    """预测固定为 class 0（大 bias），但 linear 权重仍被训练更新。"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)

    def forward(self, x, x_csi=None, lambda_=0.0):
        # 缩小 linear 贡献，加大 class 0 bias，确保 argmax 始终为 class 0
        hidden = self.linear(x) * 0.001
        bias = torch.tensor([10.0, 0.0, 0.0], device=x.device)
        logits = hidden + bias
        disc_loss = torch.tensor(0.0, requires_grad=True)
        return logits, disc_loss


@pytest.mark.l4_regression
class TestV005BestModelWeights:
    """锁定 V005 修复：训练结束后 model 持有 best epoch 权重。"""

    def test_model_state_equals_best_state(self):
        """V005 anchor: model.state_dict() == result.best_state（best epoch 权重已加载回 model）。

        构造 2 epoch 训练：epoch 1 val_acc=1.0（best），epoch 2 val_acc=0.0。
        模型用大 bias 固定预测为 class 0，但 linear 权重仍被 optimizer 更新，
        确保 best_state（epoch 1 权重）!= 末轮权重（epoch 2）。
        循环结束后应加载 best_state 回 model。
        """
        from senseframe.engine.runner.pipeline.stages.train import (
            _train_dann_loop,
            DannTrainResult,
        )

        torch.manual_seed(42)
        model = _WeightTrackingModel()

        ctx = MagicMock()
        ctx.model = model
        ctx.datamodule = MagicMock()
        # 训练数据：随机 x，标签为 class 0（与预测一致，产生非零梯度更新 linear）
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(4, 10), torch.tensor([0, 0, 0, 0]))
        ]
        # 验证数据：epoch 1 → 标签 class 0（val_acc=1.0），epoch 2 → 标签 class 1（val_acc=0.0）
        x_val = torch.randn(4, 10)
        ctx.datamodule.val_dataloader.return_value = _PerEpochValLoader([
            (x_val, torch.tensor([0, 0, 0, 0])),  # epoch 1: val_acc=1.0
            (x_val, torch.tensor([1, 1, 1, 1])),  # epoch 2: val_acc=0.0
        ])
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            result = _train_dann_loop(ctx, epochs=2, learning_rate=0.01)

        # 验证返回 DannTrainResult
        assert isinstance(result, DannTrainResult)
        # best_epoch 应是 1（val_acc=1.0 > epoch 2 的 0.0）
        assert result.best_epoch == 1
        # best_state 非 None
        assert result.best_state is not None

        # V005 核心断言：model 当前权重 == best_state（已加载回 best epoch 权重）
        current_state = model.state_dict()
        for k in current_state:
            assert torch.equal(current_state[k], result.best_state[k]), (
                "如果此断言失败，V005 修复被回退：权重未恢复到 best epoch"
            )
