"""_train_dann_loop 搜索空间消费测试。

验证 DANN 路径读取 ctx.resolved 的 optimizer/scheduler/gradient_clip_val（MEDIUM 5 修复）。
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


class TestDannLoopOptimizerConfig:
    """验证 DANN 路径从 ctx.resolved 读取 optimizer 配置。"""

    def test_sgd_optimizer_used(self):
        """ctx.resolved['optimizer']='sgd' 时，DANN 用 SGD 而非 AdamW。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "sgd",
            "weight_decay": 0.01,
            "scheduler": None,
            "gradient_clip_val": None,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.optim.SGD", wraps=torch.optim.SGD) as mock_sgd, \
             patch("torch.optim.AdamW", wraps=torch.optim.AdamW) as mock_adamw:
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        mock_sgd.assert_called_once()
        mock_adamw.assert_not_called()

    def test_gradient_clip_applied(self):
        """ctx.resolved['gradient_clip_val']=1.0 时，梯度被裁剪。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        # 非空 train_dataloader，让训练循环执行
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": None,
            "gradient_clip_val": 1.0,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as mock_clip:
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        mock_clip.assert_called_once()
        # 验证 max_norm 参数 == 1.0
        _, kwargs = mock_clip.call_args
        args = mock_clip.call_args[0]
        # clip_grad_norm_(parameters, max_norm, ...) 第二个位置参数是 max_norm
        assert args[1] == 1.0 or kwargs.get("max_norm") == 1.0

    def test_cosine_scheduler_steps(self):
        """ctx.resolved['scheduler']='cosine' 时，scheduler.step() 每 epoch 调用。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "gradient_clip_val": None,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.optim.lr_scheduler.CosineAnnealingLR") as mock_cosine_cls:
            _train_dann_loop(ctx, epochs=3, learning_rate=0.01)
            # scheduler 被实例化
            mock_cosine_cls.assert_called_once()
            # 拿到 mock 实例，验证 step() 被调用 3 次（每 epoch 一次）
            mock_scheduler = mock_cosine_cls.return_value
            assert mock_scheduler.step.call_count == 3

    def test_early_stopping_breaks_loop(self):
        """ctx.resolved['early_stopping']=2 时，连续 2 epoch 无提升则 break。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
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

        # patience=2，val_acc 始终 0.0（空 val_loader）
        # epoch 1: best_val_acc 0.0→0.0 (no improve, count=0→0? 不，初始 best=0.0，val_acc=0.0 不 > 0.0，所以 count=1)
        # epoch 2: count=2 >= patience=2, break
        # 所以 "DANN epoch" 日志应出现 2 次（epoch 1 + epoch 2），early stopping 日志 1 次
        epoch_logs = [c for c in mock_logger.info.call_args_list
                      if "DANN epoch" in str(c)]
        early_stop_logs = [c for c in mock_logger.info.call_args_list
                           if "early stopping" in str(c)]
        assert len(epoch_logs) == 2  # 只跑了 2 个 epoch 就 break
        assert len(early_stop_logs) == 1  # 早停日志出现 1 次
