"""
通用时序数据变换原语库。

设计理念（RFC-002 阶段 U：场景包深度扩展）：
- 为 generic 场景提供时序特征工程 + 通用数据增强原语
- 每个原语是独立可组合的 transform 函数
- Agent 可通过 get_transforms 的 pipeline 配置组合多个原语
- 也可通过 load_extension 生成自定义原语

原语分类：
- 特征工程：rolling_stats, fft_features, wavelet_decomp, seasonal_decompose
- 数据增强：jitter, scaling, window_warp, magnitude_warp

所有原语签名：fn(x: np.ndarray, *args, **kwargs) -> np.ndarray
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ============================================================
# 特征工程原语
# ============================================================
def rolling_stats(x: np.ndarray, window: int = 5, stat: str = "mean") -> np.ndarray:
    """滑动窗口统计特征。

    对最后一维（时间维）做滚动窗口统计，输出与输入等长的序列
    （边界用 edge padding 补齐，保证长度不变）。

    Args:
        x: 输入数据，shape (..., T)
        window: 滑动窗口大小（必须 >= 1）
        stat: 统计类型，"mean" / "std" / "min" / "max"

    Returns:
        滚动统计序列，shape 与输入一致

    Raises:
        ValueError: window < 1 或 stat 不合法
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    valid_stats = {"mean", "std", "min", "max"}
    if stat not in valid_stats:
        raise ValueError(f"stat must be one of {valid_stats}, got '{stat}'")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    def _roll_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        if n == 0:
            return sig.copy()
        half = window // 2
        # edge padding 保证输出长度不变
        padded = np.pad(sig, (half, half), mode="edge")
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            w = padded[i:i + window]
            if stat == "mean":
                out[i] = np.mean(w)
            elif stat == "std":
                s = np.std(w)
                out[i] = 0.0 if not np.isfinite(s) else s
            elif stat == "min":
                out[i] = np.min(w)
            else:  # max
                out[i] = np.max(w)
        return out

    if x.ndim == 1:
        return _roll_1d(x)
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    result = np.empty_like(x_flat)
    for i in range(x_flat.shape[0]):
        result[i] = _roll_1d(x_flat[i])
    return result.reshape(orig_shape)


def fft_features(x: np.ndarray, n_fft: Optional[int] = None) -> np.ndarray:
    """FFT 频域特征提取。

    对最后一维做实数 FFT，返回幅值谱（单边）。

    Args:
        x: 输入数据，shape (..., T)
        n_fft: FFT 点数（None 时 = T）

    Returns:
        FFT 幅值谱，shape (..., n_fft//2 + 1)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    if x.ndim == 1:
        spectrum = np.fft.rfft(x, n=n_fft)
        return np.abs(spectrum)

    orig_shape = x.shape
    if n_fft is None:
        n_fft = orig_shape[-1]
    x_flat = x.reshape(-1, orig_shape[-1])
    results = [np.abs(np.fft.rfft(row, n=n_fft)) for row in x_flat]
    return np.stack(results, axis=0).reshape(*orig_shape[:-1], n_fft // 2 + 1)


def wavelet_decomp(
    x: np.ndarray,
    wavelet: str = "haar",
    level: int = 2,
) -> np.ndarray:
    """简化小波分解（haar 小波，逐层平均/差分）。

    对最后一维做多层 haar 小波分解，返回拼接的近似 + 细节系数。
    每层将信号分为低频（平均）和高频（差分）两部分，长度减半。

    Args:
        x: 输入数据，shape (..., T)
        wavelet: 小波类型（当前仅支持 "haar"）
        level: 分解层数（必须 >= 1）

    Returns:
        小波系数序列，最后一维长度 = ceil(T / 2^level) + 各层细节系数之和

    Raises:
        ValueError: level < 1 或 wavelet 不支持
    """
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    if wavelet != "haar":
        raise ValueError(f"Only 'haar' wavelet is supported, got '{wavelet}'")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    def _haar_decomp_1d(sig: np.ndarray, lv: int) -> np.ndarray:
        coeffs = []
        current = sig
        for _ in range(lv):
            n = len(current)
            if n < 2:
                break
            # haar：相邻两元素平均（低频）与差分（高频）
            even = current[0::2]
            odd = current[1::2]
            min_len = min(len(even), len(odd))
            approx = (even[:min_len] + odd[:min_len]) / np.sqrt(2.0)
            detail = (even[:min_len] - odd[:min_len]) / np.sqrt(2.0)
            coeffs.append(detail)  # 细节系数放前面
            current = approx
        coeffs.append(current)  # 最后一层近似系数
        # 拼接：[detail_L, detail_{L-1}, ..., detail_1, approx]
        coeffs.reverse()
        return np.concatenate(coeffs)

    if x.ndim == 1:
        return _haar_decomp_1d(x, level)

    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    results = [_haar_decomp_1d(row, level) for row in x_flat]
    # 各行长度可能不同，取最大长度补零对齐
    max_len = max(len(r) for r in results)
    padded = [np.pad(r, (0, max_len - len(r))) for r in results]
    return np.stack(padded, axis=0).reshape(*orig_shape[:-1], max_len)


def seasonal_decompose(x: np.ndarray, period: int = 12) -> np.ndarray:
    """简单季节性分解。

    用移动平均提取趋势分量，原值减去趋势得到残差（去趋势序列）。
    适用于周期性时序数据，去除趋势后保留周期 + 残差信息。

    Args:
        x: 输入数据，shape (..., T)
        period: 季节周期长度（用于移动平均窗口，必须 >= 1）

    Returns:
        去趋势后的残差序列，shape 与输入一致

    Raises:
        ValueError: period < 1
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    def _decompose_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        if n == 0:
            return sig.copy()
        win = min(period, n)
        # 居中移动平均提取趋势
        half = win // 2
        padded = np.pad(sig, (half, half), mode="edge")
        trend = np.empty(n, dtype=np.float64)
        for i in range(n):
            trend[i] = np.mean(padded[i:i + win])
        residual = sig - trend
        return residual

    if x.ndim == 1:
        return _decompose_1d(x)

    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    result = np.empty_like(x_flat)
    for i in range(x_flat.shape[0]):
        result[i] = _decompose_1d(x_flat[i])
    return result.reshape(orig_shape)


# ============================================================
# 数据增强原语
# ============================================================
def jitter(x: np.ndarray, sigma: float = 0.01, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """时域抖动增强：添加高斯噪声。

    Args:
        x: 输入数据
        sigma: 噪声标准差
        rng: 可选的独立随机数生成器（P3 上策，详见 wifi_csi.transforms.time_jitter）

    Returns:
        增强后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    r = rng if rng is not None else np.random
    noise = r.normal(0, sigma, x.shape)
    return x + noise


def scaling(x: np.ndarray, sigma: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """幅度缩放增强：乘以 (1 + 高斯噪声)。

    对整个序列乘以一个随机标量因子，改变整体幅度。

    Args:
        x: 输入数据
        sigma: 缩放因子的噪声标准差
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        增强后的数据
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    r = rng if rng is not None else np.random
    factor = 1.0 + r.normal(0, sigma)
    return x * factor


def window_warp(x: np.ndarray, ratio: float = 0.1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """窗口切片增强：随机裁剪一段，用插值放缩回原长度。

    随机选取序列中一段子窗口，通过线性插值将其拉伸/压缩回原序列长度。

    Args:
        x: 输入数据，shape (..., T)
        ratio: 裁剪比例（0 < ratio <= 1），裁剪长度 = int(T * ratio)
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        增强后的数据，shape 与输入一致
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    r = rng if rng is not None else np.random

    def _warp_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        if n < 2:
            return sig.copy()
        crop_len = max(2, int(n * ratio))
        crop_len = min(crop_len, n)
        start = r.integers(0, max(1, n - crop_len + 1))
        window = sig[start:start + crop_len]
        # 线性插值放缩回原长度
        if len(window) < 2:
            return sig.copy()
        x_old = np.linspace(0, 1, len(window))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, window)

    if x.ndim == 1:
        return _warp_1d(x)

    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    result = np.empty_like(x_flat)
    for i in range(x_flat.shape[0]):
        result[i] = _warp_1d(x_flat[i])
    return result.reshape(orig_shape)


def magnitude_warp(x: np.ndarray, sigma: float = 0.1, knot: int = 4, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """幅度扭曲增强：用样条插值生成非线性扭曲曲线。

    生成一条由 knot 个控制点构成的平滑随机曲线，乘到原序列上，
    实现非线性幅度扭曲（比 scaling 更复杂的幅度变换）。

    Args:
        x: 输入数据，shape (..., T)
        sigma: 控制点噪声标准差
        knot: 样条控制点数量（必须 >= 2）
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        增强后的数据，shape 与输入一致

    Raises:
        ValueError: knot < 2
    """
    if knot < 2:
        raise ValueError(f"knot must be >= 2, got {knot}")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    r = rng if rng is not None else np.random

    def _warp_1d(sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        if n < 2:
            return sig.copy()
        # 生成 knot 个控制点（1 + 高斯噪声），用线性插值扩展到 n
        ctrl_x = np.linspace(0, 1, knot)
        ctrl_y = 1.0 + r.normal(0, sigma, size=knot)
        warp_curve = np.interp(np.linspace(0, 1, n), ctrl_x, ctrl_y)
        return sig * warp_curve

    if x.ndim == 1:
        return _warp_1d(x)

    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    result = np.empty_like(x_flat)
    for i in range(x_flat.shape[0]):
        result[i] = _warp_1d(x_flat[i])
    return result.reshape(orig_shape)


# ============================================================
# 原语注册表（供 catalog.py 和 get_transforms 使用）
# ============================================================
TRANSFORM_REGISTRY = {
    "rolling_stats": rolling_stats,
    "fft_features": fft_features,
    "wavelet_decomp": wavelet_decomp,
    "seasonal_decompose": seasonal_decompose,
    "jitter": jitter,
    "scaling": scaling,
    "window_warp": window_warp,
    "magnitude_warp": magnitude_warp,
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
        names: 原语名列表，如 ["rolling_stats", "jitter"]
        seed: 可选的随机种子（P3 上策，详见 wifi_csi.transforms.compose_transforms）
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


class ComposedTransform:
    """组合多个 transform 原语的 callable 类（可 pickle）。

    替代旧 composed 闭包，确保 DataLoader num_workers>0 时序列化不失败。
    P3 上策：持有独立 np.random.Generator，在 __call__ 中注入到原语，
    消除对全局 np.random 状态的依赖。
    """

    def __init__(self, transforms, kwargs, seed: Optional[int] = None):
        import inspect
        self.transforms = list(transforms)
        self.kwargs = dict(kwargs)
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self._accepts_rng = [
            'rng' in inspect.signature(fn).parameters for _, fn in self.transforms
        ]

    def __call__(self, x, y=None):
        import torch
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        for (name, fn), accepts_rng in zip(self.transforms, self._accepts_rng):
            params = self.kwargs.get(name, {})
            if accepts_rng:
                x_np = fn(x_np, rng=self.rng, **params)
            else:
                x_np = fn(x_np, **params)
        x_out = torch.from_numpy(x_np).float() if isinstance(x, torch.Tensor) else x_np
        return x_out, y


__all__ = [
    "rolling_stats",
    "fft_features",
    "wavelet_decomp",
    "seasonal_decompose",
    "jitter",
    "scaling",
    "window_warp",
    "magnitude_warp",
    "TRANSFORM_REGISTRY",
    "get_transform",
    "list_transforms",
    "compose_transforms",
]
