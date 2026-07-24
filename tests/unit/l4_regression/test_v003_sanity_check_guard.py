"""V003: PSNREarlyStopping sanity_check 守卫（不污染 best_psnr）。

Anchor: bug 编号 I9 + 修复 commit 45dc0ce。
原始问题: Lightning fit 开始跑 sanity check，未训练模型的 PSNR 写入 best_psnr，
         导致后续早停判断错乱（项目其他 on_validation_epoch_end 均有此守卫，
         唯独 PSNR 回调缺失）。
修复方式: on_validation_epoch_end 在 sanity_checking 阶段直接返回，不更新状态机。

如果此测试失败，说明 V003 修复被回退。
"""
from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

from senseframe.engine.callbacks.psnr_early_stopping import PSNREarlyStoppingCallback


@pytest.mark.l4_regression
class TestV003SanityCheckGuard:
    """锁定 V003 修复：sanity_check 阶段不污染 best_psnr/counter/should_stop。"""

    def test_sanity_check_does_not_pollute_state(self):
        """V003 anchor: sanity_check 阶段 best_psnr/counter/should_stop 均未被修改。"""
        callback = PSNREarlyStoppingCallback(patience=3, min_delta=0.0)

        # 模拟 sanity_check 阶段
        trainer = MagicMock()
        trainer.sanity_checking = True
        pl_module = MagicMock()
        # pl_module 有缓存（模拟 validation_step 已写入）
        pl_module._psnr_reconstruction = torch.randn(10)
        pl_module._psnr_target = torch.randn(10)

        # 记录初始状态
        initial_best_psnr = callback.best_psnr

        callback.on_validation_epoch_end(trainer, pl_module)

        assert callback.best_psnr == initial_best_psnr, (
            "如果此断言失败，V003 修复被回退：sanity_check 阶段不应更新 best_psnr"
        )
        assert callback.counter == 0, (
            "如果此断言失败，V003 修复被回退：sanity_check 阶段不应更新 counter"
        )
        assert callback.should_stop is False, (
            "如果此断言失败，V003 修复被回退：sanity_check 阶段不应触发 should_stop"
        )
