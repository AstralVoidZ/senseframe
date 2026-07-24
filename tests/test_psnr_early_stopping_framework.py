"""框架级 PSNREarlyStopping Callback 测试。

验证 PSNREarlyStopping 可作为 Lightning Callback 使用，
且 best_psnr 跟踪正确（v2 差距 3 修复）。
"""
from __future__ import annotations

import math
import types

import torch

from senseframe.engine.callbacks.psnr_early_stopping import (
    PSNREarlyStoppingCallback,
    compute_psnr,
)


class TestComputePsnrFramework:
    """PSNR 计算正确性（框架级，迁移自 scripts/p0_pretrain_with_psnr.py）。"""

    def test_perfect_reconstruction(self):
        x = torch.randn(4, 3, 64)
        assert compute_psnr(x, x) == 100.0

    def test_higher_is_better(self):
        target = torch.zeros(4, 3, 64)
        good = target + 0.01 * torch.randn(4, 3, 64)
        bad = target + 1.0 * torch.randn(4, 3, 64)
        assert compute_psnr(good, target) > compute_psnr(bad, target)

    def test_psnr_absolute_db_value(self):
        """验证 PSNR 绝对 dB 值：target=zeros, reconstructed=ones*0.5, max_value=5.0。

        MSE = 0.25, max_value^2 = 25, PSNR = 10*log10(25/0.25) = 20.0 dB。
        """
        target = torch.zeros(2, 3, 4)
        reconstructed = torch.ones(2, 3, 4) * 0.5
        expected = 10 * math.log10(25.0 / 0.25)
        psnr = compute_psnr(reconstructed, target, max_value=5.0)
        assert abs(psnr - expected) < 0.01


class TestPSNREarlyStoppingCallback:
    """Lightning Callback 行为测试。"""

    def test_is_lightning_callback(self):
        """应是 pytorch_lightning.Callback 子类。"""
        try:
            from pytorch_lightning import Callback
        except ImportError:
            from lightning import Callback
        assert issubclass(PSNREarlyStoppingCallback, Callback)

    def test_initial_state(self):
        cb = PSNREarlyStoppingCallback(patience=3, min_delta=0.1)
        assert cb.should_stop is False
        assert cb.best_psnr is None

    def test_improvement_resets_counter(self):
        cb = PSNREarlyStoppingCallback(patience=3, min_delta=0.1)
        cb._update_psnr(10.0)
        cb._update_psnr(11.0)  # 提升 1.0 > min_delta
        cb._update_psnr(12.0)
        assert cb.should_stop is False
        assert cb.counter == 0
        assert cb.best_psnr == 12.0

    def test_no_improvement_triggers_stop(self):
        cb = PSNREarlyStoppingCallback(patience=2, min_delta=0.1)
        cb._update_psnr(10.0)
        cb._update_psnr(10.05)  # 提升 0.05 < min_delta
        assert cb.counter == 1
        cb._update_psnr(10.05)  # 仍无提升
        assert cb.counter == 2
        assert cb.should_stop is True

    def test_on_validation_epoch_end_noop_without_cache(self):
        """pl_module 无缓存张量时，on_validation_epoch_end 应为 no-op。"""
        cb = PSNREarlyStoppingCallback(patience=1, min_delta=0.1)
        trainer = types.SimpleNamespace(should_stop=False)
        pl_module = types.SimpleNamespace(log=lambda *args, **kwargs: None)
        cb.on_validation_epoch_end(trainer, pl_module)
        assert cb.should_stop is False
        assert cb.best_psnr is None
        assert trainer.should_stop is False

    def test_on_validation_epoch_end_triggers_stop_with_cache(self):
        """pl_module 有缓存张量 + patience=1 + min_delta=0.1：第二次无提升应触发停止。"""
        cb = PSNREarlyStoppingCallback(patience=1, min_delta=0.1)
        trainer = types.SimpleNamespace(should_stop=False)
        log_calls = []
        pl_module = types.SimpleNamespace(
            _psnr_reconstruction=torch.zeros(2, 3, 4),
            _psnr_target=torch.ones(2, 3, 4) * 0.5,
            log=lambda *args, **kwargs: log_calls.append((args, kwargs)),
        )
        # 第一次调用：PSNR=10*log10(25/0.25)=20.0，设置 best_psnr
        cb.on_validation_epoch_end(trainer, pl_module)
        assert cb.best_psnr is not None
        assert cb.counter == 0
        assert cb.should_stop is False
        assert trainer.should_stop is False
        # 第二次调用：PSNR 仍 20.0，无提升，counter=1 >= patience=1 → should_stop=True
        cb.on_validation_epoch_end(trainer, pl_module)
        assert cb.counter == 1
        assert cb.should_stop is True
        assert trainer.should_stop is True
        # 验证 pl_module.log 被调用且记录 "val_psnr"
        assert len(log_calls) == 2
        assert log_calls[0][0][0] == "val_psnr"
        assert log_calls[1][0][0] == "val_psnr"
