"""SelfSupervisedModule training_log schema 细节测试。

I18/I19 的修复锁定已迁移至 L4 回归测试：
- tests/unit/l4_regression/test_v019_training_log_epoch_offbyone.py（I18 epoch 无 +1）
- tests/unit/l4_regression/test_v020_training_log_schema_fields.py（I19 字段存在）

本文件保留 I19 的细节行为测试（lr 具体值、train_accuracy 为 None、缺键回退），
这些用例 L4 未完全覆盖（L4 只验证字段存在，不验证具体值）。

参考：
- senseframe/engine/self_supervised.py:on_validation_epoch_end
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
