"""L1 契约测试：PSNR 标准公式（信号处理标准定义）。

锚点来源：PSNR 标准定义 PSNR = 10·log10(MAX²/MSE)（信号处理标准公式）。
https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

标准公式：PSNR = 10·log10(MAX²/MSE)
- MAX = 信号最大可能值（SenseFrame CSI 归一化后 5σ 边界，默认 5.0）
- MSE = 均方误差（mean of squared error）
- 单位：dB（对数尺度）

完美重建（MSE=0）约定返回 100.0（上限，非 inf）。

本测试用已知输入输出的数学期望值验证公式（如 MSE=1, max=5 → PSNR≈13.98 dB），
不引用源码常量（消除自证断言）。
"""
from __future__ import annotations

import math

import pytest
import torch

from senseframe.engine.callbacks.psnr_early_stopping import compute_psnr


@pytest.mark.l1_contract
class TestPsnrFormulaContract:
    """验证 compute_psnr 实现标准 PSNR 公式 10·log10(MAX²/MSE)。"""

    def test_standard_formula_mse_one_max_five(self):
        """L1 anchor: PSNR=10·log10(MAX²/MSE)，MSE=1,max=5 → ≈13.98 dB，锚点标准公式。"""
        # PSNR = 10·log10(5²/1) = 10·log10(25) ≈ 13.9794 dB
        reconstructed = torch.tensor([0.0])
        target = torch.tensor([1.0])  # MSE = mean((0-1)²) = 1.0
        psnr = compute_psnr(reconstructed, target, max_value=5.0)
        expected = 10 * math.log10(25.0 / 1.0)
        assert abs(psnr - expected) < 0.01, f"PSNR 应为 {expected:.4f}，实际 {psnr}"

    def test_standard_formula_mse_four_max_five(self):
        """L1 anchor: PSNR=10·log10(MAX²/MSE)，MSE=4,max=5 → ≈7.96 dB，锚点标准公式。"""
        # PSNR = 10·log10(25/4) = 10·log10(6.25) ≈ 7.9588 dB
        reconstructed = torch.tensor([0.0])
        target = torch.tensor([2.0])  # MSE = mean((0-2)²) = 4.0
        psnr = compute_psnr(reconstructed, target, max_value=5.0)
        expected = 10 * math.log10(25.0 / 4.0)
        assert abs(psnr - expected) < 0.01, f"PSNR 应为 {expected:.4f}，实际 {psnr}"

    def test_perfect_reconstruction_returns_cap(self):
        """L1 anchor: 完美重建（MSE=0）返回 100.0（约定上限，非 inf），锚点标准公式约定。"""
        reconstructed = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])  # MSE = 0
        psnr = compute_psnr(reconstructed, target, max_value=5.0)
        assert psnr == 100.0, f"完美重建应返回 100.0（约定上限），实际 {psnr}"

    def test_default_max_value_is_five(self):
        """L1 anchor: max_value 默认 5.0（CSI 归一化 5σ 边界），锚点 CSI 归一化约定。"""
        reconstructed = torch.tensor([0.0])
        target = torch.tensor([1.0])  # MSE = 1.0
        psnr_default = compute_psnr(reconstructed, target)
        # CSI 归一化约定：max_value = 5.0 → PSNR = 10·log10(25/1) ≈ 13.98
        expected_with_max5 = 10 * math.log10(25.0)
        assert abs(psnr_default - expected_with_max5) < 0.01, \
            f"默认 max_value 应为 5.0（PSNR 应 ≈ {expected_with_max5:.4f}），实际 {psnr_default}"

    def test_logarithmic_scale_mse_quadruple(self):
        """L1 anchor: PSNR 为对数尺度，MSE 4x → PSNR 降 10·log10(4)≈6.02 dB，锚点标准公式。"""
        recon = torch.tensor([0.0])
        target_mse1 = torch.tensor([1.0])   # MSE = 1.0
        target_mse4 = torch.tensor([2.0])   # MSE = 4.0
        psnr1 = compute_psnr(recon, target_mse1, max_value=5.0)
        psnr4 = compute_psnr(recon, target_mse4, max_value=5.0)
        # MSE 4 倍 → PSNR 降 10·log10(4) ≈ 6.0206 dB（对数尺度验证）
        expected_drop = 10 * math.log10(4.0)
        actual_drop = psnr1 - psnr4
        assert abs(actual_drop - expected_drop) < 0.01, \
            f"PSNR 降幅应为 {expected_drop:.4f}，实际 {actual_drop}"

    def test_higher_mse_lower_psnr(self):
        """L1 anchor: MSE 越大 PSNR 越低（反比关系），锚点标准公式 10·log10(MAX²/MSE)。"""
        recon = torch.tensor([0.0])
        target_small = torch.tensor([0.5])  # MSE = 0.25
        target_large = torch.tensor([2.0])  # MSE = 4.0
        psnr_small = compute_psnr(recon, target_small, max_value=5.0)
        psnr_large = compute_psnr(recon, target_large, max_value=5.0)
        assert psnr_small > psnr_large, "MSE 小则 PSNR 高（标准公式反比关系）"

    def test_psnr_scales_with_max_value(self):
        """L1 anchor: PSNR 随 max_value 增大而增大，锚点标准公式 10·log10(MAX²/MSE)。"""
        recon = torch.tensor([0.0])
        target = torch.tensor([1.0])  # MSE = 1.0
        psnr_max5 = compute_psnr(recon, target, max_value=5.0)
        psnr_max10 = compute_psnr(recon, target, max_value=10.0)
        # 10·log10(100/1) - 10·log10(25/1) = 10·log10(4) ≈ 6.02
        assert psnr_max10 > psnr_max5, "max_value 大则 PSNR 高（标准公式）"
        expected_diff = 10 * math.log10(100.0 / 25.0)
        assert abs((psnr_max10 - psnr_max5) - expected_diff) < 0.01

    def test_multielement_tensor_averages_mse(self):
        """L1 anchor: MSE 为多元素均值，锚点标准公式 MSE = mean(Δ²)。"""
        # MSE = mean((1-0)², (0-0)², (0-0)², (0-0)²) = 0.25
        recon = torch.tensor([0.0, 0.0, 0.0, 0.0])
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        psnr = compute_psnr(recon, target, max_value=5.0)
        expected = 10 * math.log10(25.0 / 0.25)
        assert abs(psnr - expected) < 0.01, f"多元素 MSE 应取均值，PSNR 应为 {expected:.4f}，实际 {psnr}"
