"""V001: DANN 早停 flaky test 修复（val_acc=0 时不设置 best_epoch）。

Anchor: bug 编号 C1 + 修复 commit a393df9。
原始问题: val_acc 始终为 0.0 时早停行为不确定，旧测试用 RNG 产生 val_acc
         导致 flaky（不同 random seed 下 val_acc 可能非 0，早停 epoch 数变化）。
修复方式: 用确定性空 val_loader 消除 RNG flaky（val_acc 确定性为 0.0，
         sum([])/max(0,1)=0/1=0.0），配合 I1 的 best_val_acc=-1.0 初始化
         让早停计数可预测（patience=2 时跑 3 epoch 后 break）。

如果此测试失败，说明 V001 修复被回退。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest


class _DummyDannModel(nn.Module):
    """模拟 DANN 模型：forward(x_eeg, x_csi, lambda_) -> (logits, disc_loss)。"""

    def __init__(self, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x_eeg, x_csi=None, lambda_=0.0):
        logits = self.fc(x_eeg)
        disc_loss = torch.tensor(0.0, requires_grad=True)
        return logits, disc_loss


@pytest.mark.l4_regression
class TestV001PipelineFlakyFix:
    """锁定 V001 修复：确定性空 val_loader 消除早停 flaky。"""

    def test_empty_val_loader_deterministic_early_stop(self):
        """V001 anchor: 空 val_loader 确定性产生 val_acc=0.0，patience=2 时跑 3 epoch 后 break。

        C1: 用确定性空 loader 消除 RNG flaky。
        """
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        # 空 val_loader：val_acc 确定性为 0.0（sum([])/max(0,1)=0/1=0.0）
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": None,
            "gradient_clip_val": None,
            "early_stopping": 2,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger") as mock_logger:
            _train_dann_loop(ctx, epochs=100, learning_rate=0.01)

        # patience=2，val_acc 始终 0.0（空 val_loader 确定性）
        # I1 修复后 best_val_acc 初始 -1.0：
        # epoch 1: val_acc=0.0 > -1.0 → True（best 更新），count=0
        # epoch 2: val_acc=0.0 > 0.0 → False，count=1
        # epoch 3: val_acc=0.0 > 0.0 → False，count=2 >= patience=2, break
        epoch_logs = [c for c in mock_logger.info.call_args_list
                      if "DANN epoch" in str(c)]
        early_stop_logs = [c for c in mock_logger.info.call_args_list
                           if "early stopping" in str(c)]
        assert len(epoch_logs) == 3, (
            "如果此断言失败，V001 修复被回退：空 val_loader 应确定性产生 3 个 epoch 日志"
        )
        assert len(early_stop_logs) == 1, (
            "如果此断言失败，V001 修复被回退：应确定性触发 1 次早停日志"
        )
