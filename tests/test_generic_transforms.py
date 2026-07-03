"""senseframe.scenes.generic.transforms 与 catalog 模块测试。

覆盖 RFC-002 阶段 U 新增的 generic 场景 8 个 transform 原语
（4 特征工程 + 4 数据增强）+ 注册表 + compose_transforms + 技术目录。
"""

import numpy as np
import pytest

from senseframe.scenes.generic.transforms import (
    rolling_stats,
    fft_features,
    wavelet_decomp,
    seasonal_decompose,
    jitter,
    scaling,
    window_warp,
    magnitude_warp,
    TRANSFORM_REGISTRY,
    get_transform,
    list_transforms,
    compose_transforms,
)
from senseframe.scenes.generic.catalog import (
    list_techniques,
    list_categories,
    get_applicable_techniques,
    is_augment,
    suggest_pipeline,
    suggest_augment,
)


# ============================================================
# 特征工程原语
# ============================================================
class TestRollingStats:
    """rolling_stats：滑动窗口统计。"""

    def test_mean_smoothing(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_stats(x, window=3, stat="mean")
        assert result.shape == x.shape
        # 中间点应为窗口均值
        np.testing.assert_allclose(result[2], np.mean([2.0, 3.0, 4.0]))

    def test_std_non_negative(self):
        rng = np.random.RandomState(0)
        x = rng.randn(50)
        result = rolling_stats(x, window=5, stat="std")
        assert result.shape == x.shape
        assert np.all(result >= 0.0)

    def test_window_zero_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            rolling_stats(x, window=0, stat="mean")

    def test_empty_array(self):
        x = np.array([])
        result = rolling_stats(x, window=5, stat="mean")
        assert result.size == 0


class TestFFTFeatures:
    """fft_features：FFT 幅值谱。"""

    def test_output_shape(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 64))
        result = fft_features(x)
        # n_fft=None → n_fft=T=64，输出 (64//2+1,) = (33,)
        assert result.shape == (33,)

    def test_output_shape_with_n_fft(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 64))
        result = fft_features(x, n_fft=128)
        assert result.shape == (65,)


class TestWaveletDecomp:
    """wavelet_decomp：haar 小波分解。"""

    def test_haar_decomposition(self):
        rng = np.random.RandomState(0)
        x = rng.randn(64)
        result = wavelet_decomp(x, wavelet="haar", level=2)
        assert result.ndim == 1
        assert result.shape[0] > 0
        assert not np.any(np.isnan(result))

    def test_level_zero_raises(self):
        x = np.ones(32)
        with pytest.raises(ValueError):
            wavelet_decomp(x, wavelet="haar", level=0)


class TestSeasonalDecompose:
    """seasonal_decompose：季节性分解。"""

    def test_period12_decompose(self):
        t = np.arange(60)
        x = np.sin(2 * np.pi * t / 12.0) + 0.1 * t  # 周期 + 趋势
        result = seasonal_decompose(x, period=12)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))

    def test_period_zero_raises(self):
        x = np.ones(24)
        with pytest.raises(ValueError):
            seasonal_decompose(x, period=0)


# ============================================================
# 数据增强原语
# ============================================================
class TestJitter:
    """jitter：时域抖动。"""

    def test_shape_unchanged_and_perturbed(self):
        rng = np.random.RandomState(0)
        x = rng.randn(4, 50)
        result = jitter(x, sigma=0.05)
        assert result.shape == x.shape
        assert not np.allclose(result, x)


class TestScaling:
    """scaling：幅度缩放。"""

    def test_shape_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.randn(4, 50)
        result = scaling(x, sigma=0.1)
        assert result.shape == x.shape


class TestWindowWarp:
    """window_warp：窗口切片。"""

    def test_shape_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.randn(4, 50)
        result = window_warp(x, ratio=0.3)
        assert result.shape == x.shape


class TestMagnitudeWarp:
    """magnitude_warp：幅度扭曲。"""

    def test_shape_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.randn(4, 50)
        result = magnitude_warp(x, sigma=0.1, knot=4)
        assert result.shape == x.shape


# ============================================================
# 组合与注册表
# ============================================================
class TestComposeTransforms:
    """compose_transforms：组合原语。"""

    def test_compose_no_nan(self):
        pytest.importorskip("torch")
        rng = np.random.RandomState(0)
        x = rng.randn(4, 50).astype(np.float32)
        composed = compose_transforms(["rolling_stats", "jitter"])
        out, y = composed(x, None)
        assert out.shape == x.shape
        assert not np.any(np.isnan(out))

    def test_unknown_primitive_raises(self):
        with pytest.raises(ValueError):
            compose_transforms(["unknown_name"])


class TestTransformRegistry:
    """TRANSFORM_REGISTRY：8 个原语。"""

    def test_registry_has_8_primitives(self):
        assert len(TRANSFORM_REGISTRY) == 8

    def test_get_transform(self):
        assert get_transform("rolling_stats") is rolling_stats
        assert get_transform("nonexistent") is None

    def test_list_transforms_sorted(self):
        names = list_transforms()
        assert len(names) == 8
        assert names == sorted(names)


# ============================================================
# 技术目录
# ============================================================
class TestGenericCatalog:
    """generic 场景技术目录。"""

    def test_list_techniques_count(self):
        assert len(list_techniques()) == 8

    def test_list_categories(self):
        assert list_categories() == ["augmentation", "feature_engineering"]

    def test_get_applicable_techniques_all(self):
        # applicable=["*"]，任意数据集都返回全部
        result = get_applicable_techniques("any_dataset")
        assert len(result) == 8

    def test_is_augment(self):
        assert is_augment("jitter") is True
        assert is_augment("rolling_stats") is False

    def test_suggest_pipeline_excludes_augment(self):
        pipeline = suggest_pipeline("any")
        assert len(pipeline) > 0
        for name in pipeline:
            assert is_augment(name) is False

    def test_suggest_augment_only_augment(self):
        augment = suggest_augment("any")
        assert len(augment) > 0
        for name in augment:
            assert is_augment(name) is True
