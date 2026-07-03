"""senseframe.scenes.wifi_csi.transforms 模块测试。

覆盖 13 个 transform 原语 + 注册表 + compose_transforms。
重点验证 S1 修复的 5 个数值稳定性 bug。
"""

import numpy as np
import pytest

from senseframe.scenes.wifi_csi.transforms import (
    hampel_filter,
    moving_average,
    phase_unwrap,
    linear_phase_calibration,
    stft_doppler,
    cwt_transform,
    fft_features,
    select_subcarriers,
    differential_features,
    bvp_estimate,
    time_jitter,
    freq_masking,
    amplitude_rotation,
    TRANSFORM_REGISTRY,
    get_transform,
    list_transforms,
    compose_transforms,
)


class TestHampelFilter:
    """Hampel 滤波器：1D/2D 输入，离群点被替换。"""

    def test_hampel_1d(self):
        # 用有方差的随机数据作为基底，保证窗口内 MAD > 0
        rng = np.random.RandomState(0)
        x = rng.randn(100)
        x[50] = 100.0  # 离群点
        result = hampel_filter(x, window=5, threshold=3.0)
        assert result[50] != 100.0
        assert abs(result[50]) < 10.0  # 被中位数替换，接近 0

    def test_hampel_2d(self):
        rng = np.random.RandomState(0)
        x = rng.randn(3, 100)
        x[1, 50] = 100.0
        result = hampel_filter(x, window=5, threshold=3.0)
        assert result[1, 50] != 100.0
        assert abs(result[1, 50]) < 10.0


class TestMovingAverage:
    """滑动平均：window=3 平滑、window=0 抛错（S1）、window=1 原样。"""

    def test_window3_smoothing(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(x, window=3)
        assert result.shape == x.shape
        # 中间点应为窗口均值
        np.testing.assert_allclose(result[1], np.mean([1.0, 2.0, 3.0]))

    def test_window0_raises(self):
        # S1 修复：window < 1 抛 ValueError
        with pytest.raises(ValueError):
            moving_average(np.array([1.0, 2.0, 3.0]), window=0)

    def test_window1_identity(self):
        # 边界：window=1 原样返回
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = moving_average(x, window=1)
        np.testing.assert_allclose(result, x)


class TestPhaseUnwrap:
    """相位解卷绕：跳变相位被展开。"""

    def test_unwrap_jumps(self):
        phase = np.array([0.0, 1.5, 3.0, -3.0, -1.5, 0.0])
        result = phase_unwrap(phase)
        # 解卷绕后无大于 pi 的跳变
        diffs = np.abs(np.diff(result))
        assert np.all(diffs < np.pi)
        # 整体应单调递增
        assert result[-1] > result[0]


class TestLinearPhaseCalibration:
    """线性相位校准：线性趋势被去除。"""

    def test_linear_trend_removed(self):
        t = np.arange(100)
        linear = 2.0 * t + 5.0
        noise = 0.1 * np.random.randn(100)
        phase = linear + noise
        result = linear_phase_calibration(phase)
        # 去除线性趋势后均值接近 0
        assert abs(np.mean(result)) < 1.0


class TestSTFTDoppler:
    """STFT 多普勒：输出 shape (n_fft//2+1, frames)。"""

    def test_output_shape(self):
        x = np.random.randn(128)
        n_fft = 64
        result = stft_doppler(x, n_fft=n_fft)
        assert result.shape[0] == n_fft // 2 + 1
        assert result.shape[1] >= 1


class TestCWTTransform:
    """CWT 小波变换：无 NaN（S1 修复复数卷积）、shape (len(scales), T)。"""

    def test_no_nan(self):
        # S1 修复：保留复数卷积，输出无 NaN
        x = np.random.randn(64)
        scales = np.arange(1, 9)
        result = cwt_transform(x, scales=scales)
        assert not np.any(np.isnan(result))

    def test_output_shape(self):
        x = np.random.randn(64)
        scales = np.arange(1, 9)
        result = cwt_transform(x, scales=scales)
        assert result.shape == (len(scales), 64)


class TestFFTFeatures:
    """FFT 特征：输出 shape (n_fft//2+1,)。"""

    def test_output_shape(self):
        x = np.random.randn(64)
        result = fft_features(x, n_fft=64)
        assert result.shape == (64 // 2 + 1,)


class TestSelectSubcarriers:
    """子载波选择：按索引选择。"""

    def test_select(self):
        x = np.random.randn(10, 50)
        indices = np.array([0, 2, 4])
        result = select_subcarriers(x, indices)
        assert result.shape == (3, 50)
        np.testing.assert_allclose(result[0], x[0])
        np.testing.assert_allclose(result[1], x[2])


class TestDifferentialFeatures:
    """差分特征：长度减 1。"""

    def test_diff_length(self):
        x = np.random.randn(50)
        result = differential_features(x)
        assert result.shape == (49,)


class TestBVPEstimate:
    """BVP 估计：正常输出、空数组返回空（S1）、fs=0 抛错（S1）。"""

    def test_normal_output(self):
        x = np.random.randn(256)
        result = bvp_estimate(x, fs=1000.0)
        assert result.size > 0

    def test_empty_array_returns_empty(self):
        # S1 修复：空数组输入返回空数组
        x = np.array([], dtype=np.float64)
        result = bvp_estimate(x)
        assert result.size == 0

    def test_fs_zero_raises(self):
        # S1 修复：fs <= 0 抛 ValueError
        with pytest.raises(ValueError):
            bvp_estimate(np.random.randn(256), fs=0)


class TestTimeJitter:
    """时域抖动：shape 不变，值有扰动。"""

    def test_shape_and_perturbation(self):
        x = np.random.randn(50)
        result = time_jitter(x, sigma=0.1)
        assert result.shape == x.shape
        assert not np.allclose(result, x)


class TestFreqMasking:
    """频域掩码：mask_ratio=0 返回原值（S1）、正常比例部分屏蔽。"""

    def test_mask_ratio_zero_returns_original(self):
        # S1 修复：mask_ratio <= 0 返回原值
        x = np.random.randn(50)
        result = freq_masking(x, mask_ratio=0.0)
        np.testing.assert_array_equal(result, x)

    def test_mask_ratio_normal(self):
        x = np.random.randn(50) + 10.0  # 偏移避免恰好为 0
        result = freq_masking(x, mask_ratio=0.1)
        assert np.any(result == 0)  # 部分被屏蔽
        assert not np.allclose(result, x)


class TestAmplitudeRotation:
    """幅度旋转：2D 二维旋转正确（S1）、1D len>=2 走旋转分支。"""

    def test_2d_rotation_preserves_norm(self):
        # S1 修复：统一逻辑，2D 输入二维旋转正确
        x = np.random.randn(3, 2)
        result = amplitude_rotation(x, angle_range=5.0)
        assert result.shape == x.shape
        # 旋转保持每行 2-向量范数
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=-1),
            np.linalg.norm(x, axis=-1),
            atol=1e-10,
        )

    def test_1d_rotation_branch(self):
        # 1D 输入长度 >= 2 走旋转分支
        x = np.random.randn(5)
        result = amplitude_rotation(x, angle_range=5.0)
        assert result.shape == x.shape
        # 仅前两个元素被旋转，其余不变
        np.testing.assert_allclose(result[2:], x[2:])


class TestComposeTransforms:
    """compose_transforms：组合原语无 NaN、未知原语抛错。"""

    def test_compose_no_nan(self):
        pytest.importorskip("torch")
        composed = compose_transforms(["hampel", "moving_average"])
        x = np.random.randn(4, 30).astype(np.float32)
        out, y = composed(x, None)
        assert not np.any(np.isnan(out))

    def test_unknown_primitive_raises(self):
        with pytest.raises(ValueError):
            compose_transforms(["unknown_name"])


class TestTransformRegistry:
    """TRANSFORM_REGISTRY：13 个原语，get/list 正常。"""

    def test_registry_has_13_primitives(self):
        assert len(TRANSFORM_REGISTRY) == 13

    def test_get_transform(self):
        assert get_transform("hampel") is hampel_filter
        assert get_transform("nonexistent") is None

    def test_list_transforms_sorted(self):
        names = list_transforms()
        assert len(names) == 13
        assert "hampel" in names
        assert names == sorted(names)
