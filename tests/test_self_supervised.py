"""SelfSupervisedModule training_log schema 对齐测试（批次 3 Group B / I18+I19）。

验证 on_validation_epoch_end 的 epoch_entry：
- epoch 字段不带 +1（与 GenericLightningModule 对齐，I18）
- 含 lr + train_accuracy 字段（与 DANN 路径 + schemas.TrainingLogEntry 对齐，I19）

参考：
- senseframe/engine/self_supervised.py:on_validation_epoch_end
- senseframe/engine/module.py:on_validation_epoch_end（GenericLightningModule 参照）
- senseframe/schemas.py:TrainingLogEntry（字段契约）
"""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn


class TestSelfSupervisedTrainingLogSchema:
    """SelfSupervisedModule on_validation_epoch_end 字段契约（I18+I19）。"""

    def _build_module(self, phase: str = "supervised"):
        """构造 SelfSupervisedModule 实例（supervised 阶段才会写 training_log）。"""
        from senseframe.engine.self_supervised import SelfSupervisedModule

        model = nn.Linear(10, 5)
        module = SelfSupervisedModule(model=model, num_classes=5)
        # 默认 phase='self_supervised'，on_validation_epoch_end 会 early-return；
        # 切换到 supervised 才走训练中验证的 epoch_entry 构造路径。
        module.phase = phase
        return module

    def _attach_mock_trainer(self, module, current_epoch: int = 3,
                             callback_metrics=None):
        """绑定 mock trainer（绕过真实 Lightning 训练循环）。

        LightningModule.trainer / current_epoch 均为 property，
        底层读取 self._trainer.{current_epoch,...}。
        """
        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.current_epoch = current_epoch
        trainer.callback_metrics = callback_metrics or {}
        module._trainer = trainer
        return trainer

    def test_epoch_field_not_plus_one(self):
        """I18：epoch 字段应等于 current_epoch，不带 +1。

        对照 GenericLightningModule（module.py L623）已去掉 +1，
        SelfSupervisedModule 应保持一致以便跨阶段对比。
        """
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(module, current_epoch=5)
        module._current_epoch_loss = 0.5
        module._current_epoch_steps = 2

        module.on_validation_epoch_end()

        assert len(module.training_log) == 1
        entry = module.training_log[0]
        # I18 关键断言：epoch == current_epoch（无 +1）
        assert entry["epoch"] == 5, f"epoch 应为 5（不带 +1），实际 {entry['epoch']}"

    def test_training_log_entry_contains_lr_and_train_accuracy(self):
        """I19：entry 必含 lr + train_accuracy 字段（schemas.TrainingLogEntry 契约）。"""
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(module, current_epoch=2)
        module._current_epoch_loss = 0.5
        module._current_epoch_steps = 2

        module.on_validation_epoch_end()

        entry = module.training_log[0]
        # I19 关键断言：必含 lr + train_accuracy 字段
        assert "lr" in entry, "training_log entry 缺 lr 字段（I19）"
        assert "train_accuracy" in entry, "training_log entry 缺 train_accuracy 字段（I19）"

    def test_lr_read_from_callback_metrics_learning_rate(self):
        """I19：lr 应从 callback_metrics['learning_rate'] 读取。

        on_train_epoch_end 已 log "learning_rate"（self_supervised.py L280），
        on_validation_epoch_end 从 callback_metrics 读取并 round(6)。
        """
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(
            module, current_epoch=1,
            callback_metrics={"learning_rate": 0.001},
        )
        module._current_epoch_loss = 0.3
        module._current_epoch_steps = 1

        module.on_validation_epoch_end()

        entry = module.training_log[0]
        assert entry["lr"] == 0.001, f"lr 应为 0.001，实际 {entry['lr']!r}"

    def test_train_accuracy_is_none_when_no_train_metrics(self):
        """I19：SelfSupervisedModule 无 train_metrics，train_accuracy 应为 None。"""
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(module, current_epoch=0)
        module._current_epoch_loss = 0.3
        module._current_epoch_steps = 1

        module.on_validation_epoch_end()

        entry = module.training_log[0]
        # SelfSupervisedModule 未注册 train_metrics，train_accuracy 必为 None
        assert entry["train_accuracy"] is None, \
            f"train_accuracy 应为 None，实际 {entry['train_accuracy']!r}"

    def test_lr_none_when_callback_metrics_missing_learning_rate(self):
        """I19：callback_metrics 无 'learning_rate' 时 lr 应为 None。"""
        module = self._build_module(phase="supervised")
        self._attach_mock_trainer(
            module, current_epoch=0,
            callback_metrics={},  # 无 learning_rate 键
        )
        module._current_epoch_loss = 0.3
        module._current_epoch_steps = 1

        module.on_validation_epoch_end()

        entry = module.training_log[0]
        assert entry["lr"] is None, f"lr 应为 None，实际 {entry['lr']!r}"
