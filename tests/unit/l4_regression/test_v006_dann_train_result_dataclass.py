"""V006: DannTrainResult dataclass 替代 ctx 私有属性。

Anchor: bug 编号 I3 + 修复 commit 83a7ce2。
原始问题: _train_dann_loop 通过 ctx._dann_best_val_loss / _dann_best_val_macro_f1
         私有属性跨函数传递结果，违反封装且难以测试。
修复方式: 引入 DannTrainResult dataclass 作为返回值，移除 ctx 私有属性。

如果此测试失败，说明 V006 修复被回退。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace
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
class TestV006DannTrainResultDataclass:
    """锁定 V006 修复：返回 DannTrainResult 而非通过 ctx 私有属性传递。"""

    def test_returns_dataclass_and_no_ctx_private_attrs(self):
        """V006 anchor: 返回 DannTrainResult 实例；ctx 无 _dann_best_* 私有属性。

        用 SimpleNamespace（非 MagicMock）以便 hasattr 正确检测私有属性是否存在。
        """
        from senseframe.engine.runner.pipeline.stages.train import (
            _train_dann_loop,
            DannTrainResult,
        )

        ctx = SimpleNamespace(
            model=_DummyDannModel(),
            datamodule=MagicMock(),
            scene_kwargs={},
            lightning_params={"accelerator": "cpu"},
            training_log=[],
            resolved={
                "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
                "gradient_clip_val": None, "early_stopping": None,
            },
        )
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            result = _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        assert isinstance(result, DannTrainResult), (
            "如果此断言失败，V006 修复被回退：应返回 DannTrainResult 实例"
        )
        assert hasattr(result, "best_score")
        assert hasattr(result, "best_epoch")
        assert hasattr(result, "best_val_loss")
        assert hasattr(result, "best_val_macro_f1")
        assert hasattr(result, "best_state")
        # 验证 ctx 不再有 _dann_best_* 私有属性
        assert not hasattr(ctx, "_dann_best_val_loss"), (
            "如果此断言失败，V006 修复被回退：ctx 不应再有 _dann_best_val_loss 私有属性"
        )
        assert not hasattr(ctx, "_dann_best_val_macro_f1"), (
            "如果此断言失败，V006 修复被回退：ctx 不应再有 _dann_best_val_macro_f1 私有属性"
        )
