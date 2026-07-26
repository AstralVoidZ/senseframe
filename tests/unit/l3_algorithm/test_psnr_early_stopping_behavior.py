"""L3 算法行为测试：PSNR EarlyStopping patience 计数与 should_stop 触发。

从 L1 契约层（tests/unit/l1_contract/test_lightning_api_contract.py）迁移而来。
原测试 test_psnr_callback_sets_trainer_should_stop 验证的是 patience 计数逻辑
和 trainer.should_stop 触发行为，属于算法行为测试而非 API 契约测试，
因此迁移至 L3 算法层。

锚点来源：Lightning EarlyStopping 机制 + SenseFrame PSNR 早停策略。
- patience: 连续 N 个 epoch 无提升后触发停止
- min_delta: 最小提升阈值
- trainer.should_stop: Lightning Trainer 早停协议
"""
from __future__ import annotations

import pytest
import torch

from senseframe.engine.callbacks.psnr_early_stopping import (
    PSNREarlyStoppingCallback,
)
from tests.fakes.fake_lightning_module import FakeLightningModule
from tests.fakes.fake_trainer import FakeTrainer


@pytest.mark.l3_algorithm
class TestPSNREarlyStoppingBehavior:
    """验证 PSNREarlyStoppingCallback 的 patience 计数与 should_stop 触发行为。"""

    def test_psnr_callback_sets_trainer_should_stop(self):
        """PSNR 连续 patience 个 epoch 无提升时，Callback 设 trainer.should_stop=True。

        使用 FakeTrainer + FakeLightningModule 验证早停行为（不 mock Lightning）。
        """
        # patience=2: 连续 2 个 epoch 无提升则触发停止
        callback = PSNREarlyStoppingCallback(patience=2, min_delta=0.1)
        trainer = FakeTrainer(sanity_checking=False)

        pl_module = FakeLightningModule()

        # epoch 1: 高 PSNR（完美重建，mse≈0 → 100.0）
        pl_module._psnr_reconstruction = torch.zeros(4)
        pl_module._psnr_target = torch.zeros(4)
        callback.on_validation_epoch_end(trainer, pl_module)
        assert not trainer.should_stop, (
            "首个 epoch 有提升，不应触发 should_stop"
        )

        # epoch 2-3: PSNR 无提升（相同输入），累计 patience
        callback.on_validation_epoch_end(trainer, pl_module)
        callback.on_validation_epoch_end(trainer, pl_module)
        assert trainer.should_stop is True, (
            "连续 patience 个 epoch 无 PSNR 提升，应设 trainer.should_stop=True"
            "（Lightning Trainer.should_stop 早停协议）"
        )
