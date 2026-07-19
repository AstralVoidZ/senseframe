"""EEG 场景数据变换：CSP / 时频分析 / 通道标准化。

P1.2 落地：验证 SceneContainer.get_transforms 在 EEG 模态下的可移植性，
特别是自监督模式下 supervised_transform 的填充契约。

支持原语：
- normalize_eeg: 按通道标准化（沿时间轴计算 mean/std）
- bandpass_filter: 4 阶巴特沃斯带通滤波（默认 8-30Hz μ/β 频段）
- csp_features: 共空间模式特征提取（监督分类常用）
- time_freq: 短时傅里叶变换时频表示
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ...common.transforms import ComposedTransform


# ============================================================
# 变换原语注册表
# ============================================================
TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def _register(name: str):
    """原语注册装饰器。"""
    def decorator(fn: Callable) -> Callable:
        TRANSFORM_REGISTRY[name] = fn
        return fn
    return decorator


def get_transform(name: str) -> Callable:
    """获取变换原语类（调用方需实例化）。"""
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown eeg transform: {name}. "
            f"Available: {list(TRANSFORM_REGISTRY.keys())}"
        )
    return TRANSFORM_REGISTRY[name]


def list_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


# ============================================================
# EEG 数据变换原语
# ============================================================
# EEG 原始数据形状：(C, T) — C 通道数（如 22），T 时间采样点

@_register("normalize_eeg")
class NormalizeEEG:
    """EEG 通道标准化（沿时间轴）。

    对每个通道独立标准化到 0 均值单位方差。
    (C, T) → (C, T)

    注意：ComposedTransform 期望原语返回 numpy array，最终由 ComposedTransform
    统一转回 tensor。原语内部用 torch 计算后转回 numpy 返回。
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        self.eps = eps

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # 沿时间轴（最后一维）标准化，每个通道独立
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return x.numpy(), y


@_register("bandpass_filter")
class BandpassFilter:
    """巴特沃斯带通滤波（简化版）。

    实际生产实现需 scipy.signal.butter + filtfilt；
    此处用 1D 卷积近似带通效果（仅用于契约验证 stub）。
    (C, T) → (C, T)
    """
    def __init__(self, low: float = 8.0, high: float = 30.0,
                 fs: float = 250.0, kernel_size: int = 15, **kwargs):
        self.low = low
        self.high = high
        self.fs = fs
        self.kernel_size = kernel_size
        # 简化：用平均滤波近似（实际应实现完整巴特沃斯）
        self._kernel = torch.ones(kernel_size) / kernel_size

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # 对每个通道做 1D 卷积（用 conv1d 近似带通）
        # x: (C, T) → (1, C, T) for conv1d
        x_ = x.unsqueeze(0)
        k = self._kernel.to(x.device).to(x.dtype)
        # (1, 1, K) → (1, 1, T) per channel
        k = k.view(1, 1, -1).expand(x.shape[0], -1, -1)
        # 需要分组卷积：每组一个通道
        from torch.nn.functional import conv1d
        pad = self.kernel_size // 2
        x_filtered = conv1d(x_, k, padding=pad, groups=x.shape[0])
        return x_filtered.squeeze(0).numpy(), y


@_register("csp_features")
class CSPFeatures:
    """共空间模式（CSP）特征提取 stub。

    生产实现需拟合 CSP 投影矩阵（基于类间/类内方差比），
    此处用线性投影 stub 验证契约：将 EEG 通道数压缩到固定维度。
    (C, T) → (n_components, T)
    """
    def __init__(self, n_components: int = 6, n_channels: int = 22, **kwargs):
        self.n_components = n_components
        # 随机投影矩阵（实际应拟合 CSP）
        self._proj = torch.randn(n_components, n_channels) * 0.1

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # x: (C, T) → (n_components, T)
        proj = self._proj.to(x.device).to(x.dtype)
        x = proj @ x  # (n_comp, C) @ (C, T) → (n_comp, T)
        return x.numpy(), y


@_register("time_freq")
class TimeFreq:
    """短时傅里叶变换时频表示。

    (C, T) → (C, F, T_frames)
    """
    def __init__(self, n_fft: int = 64, hop_length: int = 16, **kwargs):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # 对每个通道做 STFT，取 magnitude
        # x: (C, T)
        C, T = x.shape
        specs = []
        for c in range(C):
            spec = torch.stft(
                x[c], n_fft=self.n_fft, hop_length=self.hop_length,
                return_complex=True,
            )
            specs.append(spec.abs())  # (F, T_frames)
        x = torch.stack(specs, dim=0)  # (C, F, T_frames)
        return x.numpy(), y


# ============================================================
# 组合变换
# ============================================================
def compose_transforms(names: List[str], seed: Optional[int] = None, **kwargs) -> Callable:
    """组合多个变换原语（对齐 wifi_csi.transforms.compose_transforms）。

    原语类在组合时实例化，符合 ComposedTransform 对 callable 的期望。
    """
    transforms = []
    for name in names:
        cls = get_transform(name)
        params = kwargs.get(name, {})
        instance = cls(**params) if params else cls()
        transforms.append((name, instance))
    return ComposedTransform(transforms, {}, seed=seed)
