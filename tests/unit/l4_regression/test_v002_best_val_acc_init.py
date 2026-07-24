"""V002: best_val_acc 初始值 -1.0（原 0.0 导致首 epoch 误判）。

Anchor: bug 编号 I1 + 修复 commit 83a7ce2。
原始问题: best_val_acc 初始 0.0 与合法值 0.0 冲突，val_acc=0 时
         0.0 > 0.0 为 False，best_epoch 始终 None。
修复方式: best_val_acc 初始化为 -1.0，使 0.0 > -1.0 为 True，首 epoch 即记录 best。

如果此测试失败，说明 V002 修复被回退。
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
class TestV002BestValAccInit:
    """锁定 V002 修复：best_val_acc=-1.0 使 val_acc=0 时 best_epoch 被记录。"""

    def test_best_epoch_recorded_when_val_acc_zero(self):
        """V002 anchor: val_acc=0 时 ctx.best_epoch is not None（0.0 > -1.0 为 True）。

        用空 val_loader 确定性产生 val_acc=0.0（sum([])/max(0,1)=0/1=0.0），
        避免 RNG 导致的非确定性。
        """
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        # 空 val_loader：val_acc 确定性为 0.0
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=2, learning_rate=0.01)

        assert ctx.best_epoch is not None, (
            "如果此断言失败，V002 修复被回退：val_acc=0 时 best_epoch 仍应被记录（best_val_acc=-1.0）"
        )
        assert isinstance(ctx.best_epoch, int)
        assert 1 <= ctx.best_epoch <= 2
