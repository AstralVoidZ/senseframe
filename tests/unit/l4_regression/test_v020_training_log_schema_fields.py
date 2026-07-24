"""V020: I19 training_log schema 字段（补 lr + train_accuracy=None）。

Anchor: bug 编号 V020 + 修复 commit 6102b8c。
原始问题: SelfSupervisedModule.on_validation_epoch_end 的 epoch_entry 缺
  lr 和 train_accuracy 字段，与 GenericLightningModule（module.py L624/L650）
  和 DANN 路径不一致，导致 schemas.TrainingLogEntry 契约违反，
  analyze_training_result 的 train-val gap 检测因缺字段被跳过。
修复方式: epoch_entry 补 lr（从 callback_metrics['learning_rate'] 读取）
  和 train_accuracy=None（SelfSupervisedModule 无 train_metrics，恒为 None）。

如果此测试失败，说明 V020 修复被回退（entry 又缺 lr/train_accuracy 字段）。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch.nn as nn


@pytest.mark.l4_regression
class TestV020TrainingLogSchemaFields:
    """锁定 V020 修复：training_log entry 含 lr + train_accuracy 字段。"""

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

    def test_training_log_entry_contains_lr_and_train_accuracy(self):
        """V020 anchor: entry 含 "lr" 和 "train_accuracy" 字段。

        如果此断言失败，V020 修复被回退。
        """
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(module, current_epoch=2)
        module._current_epoch_loss = 0.5
        module._current_epoch_steps = 2

        module.on_validation_epoch_end()

        entry = module.training_log[0]
        # V020 关键断言：必含 lr + train_accuracy 字段
        assert "lr" in entry, (
            "如果此断言失败，V020 修复被回退：training_log entry 缺 lr 字段"
        )
        assert "train_accuracy" in entry, (
            "如果此断言失败，V020 修复被回退：training_log entry 缺 train_accuracy 字段"
        )
