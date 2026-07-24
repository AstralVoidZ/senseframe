"""V019: I18 training_log epoch off-by-one（原 epoch+1 导致 1-indexed 与 Lightning 0-indexed 不一致）。

Anchor: bug 编号 V019 + 修复 commit 6102b8c。
原始问题: SelfSupervisedModule.on_validation_epoch_end 构造 epoch_entry 时
  ``epoch = self.current_epoch + 1``，导致日志 epoch 从 2 开始（Lightning 2.x
  on_validation_epoch_end 触发时 current_epoch 已递增），与
  GenericLightningModule（已去掉 +1）不一致，跨阶段对比错位 1 epoch。
修复方式: 去掉 +1，``epoch_entry = {"epoch": self.current_epoch, ...}``，
  与 GenericLightningModule（module.py L623）对齐。

如果此测试失败，说明 V019 修复被回退（epoch 又带回 +1）。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch.nn as nn


@pytest.mark.l4_regression
class TestV019TrainingLogEpochOffByOne:
    """锁定 V019 修复：epoch 字段等于 current_epoch（无 +1）。"""

    @staticmethod
    def _build_module(phase: str = "supervised"):
        """构造 SelfSupervisedModule 实例（supervised 阶段才会写 training_log）。"""
        from senseframe.engine.self_supervised import SelfSupervisedModule

        model = nn.Linear(10, 5)
        module = SelfSupervisedModule(model=model, num_classes=5)
        module.phase = phase
        return module

    @staticmethod
    def _attach_mock_trainer(module, current_epoch: int = 3, callback_metrics=None):
        """绑定 mock trainer（绕过真实 Lightning 训练循环）。"""
        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.current_epoch = current_epoch
        trainer.callback_metrics = callback_metrics or {}
        module._trainer = trainer
        return trainer

    def test_epoch_field_not_plus_one(self):
        """V019 anchor: entry["epoch"] == current_epoch（无 +1）。

        如果此断言失败，V019 修复被回退。
        """
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(module, current_epoch=5)
        module._current_epoch_loss = 0.5
        module._current_epoch_steps = 2

        module.on_validation_epoch_end()

        assert len(module.training_log) == 1
        entry = module.training_log[0]
        # V019 关键断言：epoch == current_epoch（无 +1）
        assert entry["epoch"] == 5, (
            f"如果此断言失败，V019 修复被回退：epoch 应为 5（不带 +1），"
            f"实际 {entry['epoch']}"
        )
