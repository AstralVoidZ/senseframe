"""PSNR 计算 + early stopping 逻辑单元测试。

测试原则：
- 全部用合成数据，快速运行（< 5 秒）
- 真实 tensor 运算
- PSNR 物理意义验证：完美重建=100dB，MSE 越小 PSNR 越高
- early stopping 状态机验证：counter 重置 + patience 触发
"""
from __future__ import annotations

import pytest
import torch

from scripts.p0_pretrain_with_psnr import (
    compute_psnr,
    PSNREarlyStopping,
)


# ============================================================
# TestComputePSNR
# ============================================================
class TestComputePSNR:
    """PSNR 计算正确性。"""

    def test_psnr_perfect_reconstruction(self):
        """完全重建时 PSNR 应为 100.0（mse < 1e-10）。"""
        x = torch.randn(4, 3, 64)
        psnr = compute_psnr(x, x)
        assert psnr == 100.0

    def test_psnr_zero_reconstruction(self):
        """target=0, reconstructed=1 时 PSNR 应为 10*log10(25/1)≈13.98 dB。"""
        target = torch.zeros(4, 3, 64)
        reconstructed = torch.ones(4, 3, 64)
        psnr = compute_psnr(reconstructed, target, max_value=5.0)
        expected = 10 * torch.log10(torch.tensor(25.0))
        assert abs(psnr - expected.item()) < 0.01

    def test_psnr_higher_is_better(self):
        """MSE 越小，PSNR 越高。"""
        target = torch.zeros(4, 3, 64)
        good = target + 0.01 * torch.randn(4, 3, 64)  # 接近 target
        bad = target + 1.0 * torch.randn(4, 3, 64)    # 远离 target
        psnr_good = compute_psnr(good, target, max_value=5.0)
        psnr_bad = compute_psnr(bad, target, max_value=5.0)
        assert psnr_good > psnr_bad

    def test_psnr_default_max_value(self):
        """默认 max_value=5.0（CSI 归一化后 5σ 边界）。"""
        target = torch.zeros(2, 3, 64)
        reconstructed = torch.ones(2, 3, 64) * 0.5
        psnr = compute_psnr(reconstructed, target)
        expected = 10 * torch.log10(torch.tensor(25.0 / 0.25))
        assert abs(psnr - expected.item()) < 0.01


# ============================================================
# TestPSNREarlyStopping
# ============================================================
class TestPSNREarlyStopping:
    """early stopping 逻辑测试。"""

    def test_initial_state_no_stop(self):
        """初始状态不应停止。"""
        es = PSNREarlyStopping(patience=3, min_delta=0.1)
        assert not es.should_stop

    def test_improvement_resets_counter(self):
        """PSNR 持续提升时 counter 不增加。"""
        es = PSNREarlyStopping(patience=3, min_delta=0.1)
        es(10.0)  # 初始
        es(11.0)  # 提升 1.0 > min_delta
        es(12.0)  # 提升 1.0 > min_delta
        assert not es.should_stop
        assert es.counter == 0

    def test_no_improvement_increases_counter(self):
        """PSNR 未提升时 counter 增加。"""
        es = PSNREarlyStopping(patience=3, min_delta=0.1)
        es(10.0)
        es(10.05)  # 提升 0.05 < min_delta
        assert es.counter == 1
        es(10.05)  # 仍无提升
        assert es.counter == 2

    def test_patience_exceeded_triggers_stop(self):
        """连续 patience 次无提升触发停止。"""
        es = PSNREarlyStopping(patience=3, min_delta=0.1)
        es(10.0)
        es(10.0)  # 无提升 1
        es(10.0)  # 无提升 2
        es(10.0)  # 无提升 3
        assert es.should_stop

    def test_best_psnr_tracked(self):
        """best_psnr 跟踪历史最高值。"""
        es = PSNREarlyStopping(patience=5, min_delta=0.1)
        es(10.0)
        es(15.0)
        es(12.0)  # 下降
        assert es.best_psnr == 15.0
