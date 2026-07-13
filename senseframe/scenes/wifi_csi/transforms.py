"""
WiFi CSI 信号处理原语库。

设计理念（RFC-002 原则 4）：
- 场景包提供领域深度，框架核心保持轻量
- 每个原语是独立可组合的 transform 函数
- Agent 可通过 get_transforms 的 pipeline 配置组合多个原语
- 也可通过 load_extension 生成自定义原语

原语分类：
- 去噪：hampel_filter, moving_average
- 相位处理：phase_unwrap, linear_phase_calibration
- 时频变换：stft_doppler, cwt_transform, fft_features
- 特征工程：select_subcarriers, differential_features, bvp_estimate
- 数据增强：time_jitter, freq_masking, amplitude_rotation

所有原语签名：fn(x: np.ndarray, *args, **kwargs) -> np.ndarray
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...common.transforms import ComposedTransform


# ============================================================
# 去噪原语
# ============================================================
def hampel_filter(x: np.ndarray, window: int = 5, threshold: float = 3.0) -> np.ndarray:
    """Hampel 滤波器：基于滑动窗口中位数检测并替换离群点。

    Args:
        x: 输入 CSI 数据，shape (..., T) 或 (..., C, T)
        window: 滑动窗口半径（总窗口 = 2*window+1）
        threshold: 离群点判定阈值（单位为 MAD）

    Returns:
        去噪后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    # 对最后一维（时间维）做 Hampel
    if x.ndim == 1:
        result = _hampel_1d(x, window, threshold)
    else:
        # 展平到 (N, T)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        for i in range(x_flat.shape[0]):
            x_flat[i] = _hampel_1d(x_flat[i], window, threshold)
        result = x_flat.reshape(orig_shape)
    return result


def _hampel_1d(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """1D Hampel 滤波。"""
    n = len(x)
    result = x.copy()
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        window_data = x[lo:hi]
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))
        if mad > 0 and np.abs(x[i] - median) > threshold * 1.4826 * mad:
            result[i] = median
    return result


def moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    """滑动平均平滑。"""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="same")
    # 多维：对最后一维做卷积
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    kernel = np.ones(window) / window
    for i in range(x_flat.shape[0]):
        x_flat[i] = np.convolve(x_flat[i], kernel, mode="same")
    return x_flat.reshape(orig_shape)


# ============================================================
# 相位处理原语
# ============================================================
def phase_unwrap(phase: np.ndarray) -> np.ndarray:
    """相位解卷绕。

    Args:
        phase: 相位数据（弧度），shape 任意

    Returns:
        解卷绕后的连续相位
    """
    phase = np.asarray(phase, dtype=np.float64)
    return np.unwrap(phase)


def linear_phase_calibration(phase: np.ndarray) -> np.ndarray:
    """线性相位校准：去除载波频率偏移（CFO）引起的线性相位偏移。

    对每个子载波的相位做线性拟合，去除线性趋势。

    Args:
        phase: 相位数据，shape (..., T) 或 (..., C, T)

    Returns:
        校准后的相位
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim == 1:
        return _detrend_linear(phase)
    orig_shape = phase.shape
    x_flat = phase.reshape(-1, orig_shape[-1])
    for i in range(x_flat.shape[0]):
        x_flat[i] = _detrend_linear(x_flat[i])
    return x_flat.reshape(orig_shape)


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    """去除线性趋势。"""
    n = len(x)
    t = np.arange(n)
    # 最小二乘拟合
    coeffs = np.polyfit(t, x, 1)
    trend = np.polyval(coeffs, t)
    return x - trend


# ============================================================
# 时频变换原语
# ============================================================
def stft_doppler(
    x: np.ndarray,
    n_fft: int = 64,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """短时傅里叶变换提取多普勒频谱。

    Args:
        x: 输入 CSI 数据，shape (..., T)
        n_fft: FFT 窗口大小
        hop_length: 跳跃长度（None 时 = n_fft // 2）

    Returns:
        STFT 幅值谱，shape (..., n_fft//2 + 1, frames)
    """
    x = np.asarray(x, dtype=np.float64)
    if hop_length is None:
        hop_length = n_fft // 2
    window = np.hanning(n_fft)

    def _stft_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        n_frames = max(1, (n - n_fft) // hop_length + 1)
        result = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * hop_length
            frame = sig[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frame = frame * window
            spectrum = np.fft.rfft(frame)
            result[:, i] = np.abs(spectrum)
        return result

    if x.ndim == 1:
        return _stft_1d(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    results = [_stft_1d(row) for row in x_flat]
    # 合并：每个 row 变为 (freq, frames)
    return np.stack(results, axis=0).reshape(*orig_shape[:-1], results[0].shape[0], results[0].shape[1])


def cwt_transform(
    x: np.ndarray,
    scales: Optional[np.ndarray] = None,
    wavelet: str = "morlet",
) -> np.ndarray:
    """连续小波变换。

    Args:
        x: 输入信号，shape (..., T)
        scales: 尺度参数（None 时自动选择）
        wavelet: 小波类型

    Returns:
        CWT 系数幅值，shape (..., len(scales), T)
    """
    x = np.asarray(x, dtype=np.float64)
    if scales is None:
        scales = np.arange(1, 33)

    def _cwt_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        result = np.zeros((len(scales), n))
        for i, scale in enumerate(scales):
            # 简化 Morlet 小波
            t = np.arange(-scale * 3, scale * 3 + 1) / scale
            wavelet_func = np.exp(1j * 2 * np.pi * t) * np.exp(-t ** 2 / 2)
            # 卷积（保留复数卷积以保留相位信息）
            conv = np.convolve(sig, wavelet_func, mode="same")
            # np.convolve mode="same" 返回 max(len(sig), len(wavelet_func))，裁剪到信号长度
            if len(conv) > n:
                start = (len(conv) - n) // 2
                conv = conv[start:start + n]
            result[i] = np.abs(conv)
        return result

    if x.ndim == 1:
        return _cwt_1d(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    results = [_cwt_1d(row) for row in x_flat]
    return np.stack(results, axis=0).reshape(*orig_shape[:-1], len(scales), orig_shape[-1])


def fft_features(x: np.ndarray, n_fft: Optional[int] = None) -> np.ndarray:
    """FFT 频域特征提取。

    Args:
        x: 输入 CSI 数据，shape (..., T)
        n_fft: FFT 点数（None 时 = T）

    Returns:
        FFT 幅值谱，shape (..., n_fft//2 + 1)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        spectrum = np.fft.rfft(x, n=n_fft)
        return np.abs(spectrum)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    if n_fft is None:
        n_fft = orig_shape[-1]
    results = [np.abs(np.fft.rfft(row, n=n_fft)) for row in x_flat]
    return np.stack(results, axis=0).reshape(*orig_shape[:-1], n_fft // 2 + 1)


# ============================================================
# 特征工程原语
# ============================================================
def select_subcarriers(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """子载波选择。

    Args:
        x: 输入 CSI 数据，shape (..., C, T) 或 (C, ...)
        indices: 要选择的子载波索引

    Returns:
        选择后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)
    # 假设 C 在倒数第二维
    return x[..., indices, :]


def differential_features(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """差分特征：一阶差分。

    Args:
        x: 输入数据
        axis: 差分轴（默认最后一维，时间维）

    Returns:
        差分后的数据（长度减 1）
    """
    x = np.asarray(x, dtype=np.float64)
    return np.diff(x, axis=axis)


def bvp_estimate(
    x: np.ndarray,
    freq_range: Tuple[float, float] = (0.5, 5.0),
    fs: float = 1000.0,
) -> np.ndarray:
    """体速度剖面（BVP）估计。

    基于菲涅尔区理论，从 CSI 时序变化估计人体运动速度分布。

    Args:
        x: 输入 CSI 数据，shape (..., T)
        freq_range: 感兴趣的频率范围 (low, high) Hz
        fs: 采样率 Hz

    Returns:
        BVP 特征
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    n_fft = max(256, x.shape[-1] if x.ndim > 0 else 256)

    def _bvp_1d(sig: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(sig, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        # 选择频率范围内的功率
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        power = np.abs(spectrum) ** 2
        return power[mask]

    if x.ndim == 1:
        return _bvp_1d(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    results = [_bvp_1d(row) for row in x_flat]
    max_len = max(len(r) for r in results)
    padded = [np.pad(r, (0, max_len - len(r))) for r in results]
    return np.stack(padded, axis=0).reshape(*orig_shape[:-1], max_len)


# ============================================================
# 数据增强原语
# ============================================================
def time_jitter(x: np.ndarray, sigma: float = 0.01, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """时域抖动增强：添加高斯噪声。

    Args:
        x: 输入数据
        sigma: 噪声标准差
        rng: 可选的独立随机数生成器。P3 上策：消除对全局 np.random 状态的依赖，
            确保多 worker 间随机性独立且可复现。None 时回退到全局 np.random（向后兼容）。

    Returns:
        增强后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    r = rng if rng is not None else np.random.default_rng()
    noise = r.normal(0, sigma, x.shape)
    return x + noise


def freq_masking(x: np.ndarray, mask_ratio: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """频域掩码增强：随机屏蔽部分频率分量。

    Args:
        x: 输入数据
        mask_ratio: 屏蔽比例
        rng: 可选的独立随机数生成器（P3 上策，详见 time_jitter）

    Returns:
        增强后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    if mask_ratio <= 0 or x.size == 0:
        return result
    r = rng if rng is not None else np.random.default_rng()
    if x.ndim >= 1:
        n = x.shape[-1]
        n_mask = max(1, int(n * mask_ratio))
        if n_mask >= n:
            n_mask = max(1, n - 1)
        # r.integers 上界 exclusive，np.random.randint 也是 exclusive，语义一致
        mask_start = r.integers(0, max(1, n - n_mask + 1))
        result[..., mask_start:mask_start + n_mask] = 0
    return result


def amplitude_rotation(x: np.ndarray, angle_range: float = 5.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """幅度旋转增强：对 CSI 幅值做随机旋转。

    Args:
        x: 输入数据
        angle_range: 旋转角度范围（度）
        rng: 可选的独立随机数生成器（P3 上策，详见 time_jitter）

    Returns:
        增强后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    r = rng if rng is not None else np.random.default_rng()
    angle = r.uniform(-angle_range, angle_range)
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # 对最后一维做旋转：按最后一维长度判断
    if x.shape[-1] >= 2:
        result = x.copy()
        result[..., 0] = x[..., 0] * cos_a - x[..., 1] * sin_a
        result[..., 1] = x[..., 0] * sin_a + x[..., 1] * cos_a
        return result
    # 最后一维长度 < 2，只能做标量旋转
    return x * cos_a


# ============================================================
# 原语注册表（供 catalog.py 和 get_transforms 使用）
# ============================================================
TRANSFORM_REGISTRY = {
    "hampel": hampel_filter,
    "moving_average": moving_average,
    "phase_unwrap": phase_unwrap,
    "linear_phase_calibration": linear_phase_calibration,
    "stft": stft_doppler,
    "cwt": cwt_transform,
    "fft": fft_features,
    "select_subcarriers": select_subcarriers,
    "differential": differential_features,
    "bvp": bvp_estimate,
    "time_jitter": time_jitter,
    "freq_masking": freq_masking,
    "amplitude_rotation": amplitude_rotation,
}


def get_transform(name: str):
    """按名获取 transform 原语。"""
    return TRANSFORM_REGISTRY.get(name)


def list_transforms() -> list:
    """列出所有已注册的 transform 原语名。"""
    return sorted(TRANSFORM_REGISTRY.keys())


def compose_transforms(names: list, seed: Optional[int] = None, **kwargs) -> callable:
    """组合多个 transform 原语为单一函数。

    Args:
        names: 原语名列表，如 ["hampel", "phase_unwrap", "stft"]
        seed: 可选的随机种子。P3 上策：为 ComposedTransform 创建独立 np.random.Generator，
            消除对全局 np.random 状态的依赖。None 时不创建 Generator（原语回退到全局 np.random）。
        **kwargs: 传递给每个原语的参数（按原语名分组）

    Returns:
        ComposedTransform 实例（callable，可 pickle 供 DataLoader multi-worker 使用）
    """
    transforms = []
    for name in names:
        fn = TRANSFORM_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown transform: {name}. Available: {list_transforms()}")
        transforms.append((name, fn))

    return ComposedTransform(transforms, kwargs, seed=seed)


__all__ = [
    "hampel_filter",
    "moving_average",
    "phase_unwrap",
    "linear_phase_calibration",
    "stft_doppler",
    "cwt_transform",
    "fft_features",
    "select_subcarriers",
    "differential_features",
    "bvp_estimate",
    "time_jitter",
    "freq_masking",
    "amplitude_rotation",
    "TRANSFORM_REGISTRY",
    "get_transform",
    "list_transforms",
    "compose_transforms",
]
