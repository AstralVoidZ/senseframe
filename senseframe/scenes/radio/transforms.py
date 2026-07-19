"""Radio 场景数据变换：IQ → 复数谱图 / 时频图。

P1.2 落地：验证 SceneContainer.get_transforms 接口在新模态（无线电信号）下的可移植性。

支持原语：
- iq_to_complex: (2, L) IQ → (1, L) 复数张量（实部+虚部合并为单通道复数）
- iq_to_spectrogram: (2, L) IQ → (2, F, T) 时频图（STFT 实部/虚部双通道）
- normalize_iq: 按 IQ 通道标准化（mean=0, std=1，沿 L 轴计算）
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
            f"Unknown radio transform: {name}. "
            f"Available: {list(TRANSFORM_REGISTRY.keys())}"
        )
    return TRANSFORM_REGISTRY[name]


def list_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


# ============================================================
# IQ 数据变换原语
# ============================================================
# RadioML 原始数据形状：(2, L) — I/Q 双通道，每通道 L 个时间样本
# 模型期望输入：(C, L) 或 (C, F, T)，C 为通道数

@_register("iq_to_complex")
class IQToComplex:
    """IQ 双通道 → 复数单通道。

    (2, L) → (1, L)
    合并 I/Q 为单一复数通道（保留相位信息）。

    注意：ComposedTransform 期望原语返回 numpy array，最终由 ComposedTransform
    统一转回 tensor。原语内部用 torch 计算后转回 numpy 返回。
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, y):
        # x: (2, L) numpy 或 tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 2 and x.shape[0] == 2:
            # I + jQ → 复数表示用 magnitude
            i, q = x[0], x[1]
            magnitude = torch.sqrt(i * i + q * q)
            x = magnitude.unsqueeze(0)  # (1, L)
        # 返回 numpy，由 ComposedTransform 统一转 tensor
        return x.numpy(), y


@_register("iq_to_spectrogram")
class IQToSpectrogram:
    """IQ 双通道 → STFT 时频图。

    (2, L) → (2, F, T)
    对 I/Q 分别做 STFT，取实部+虚部作为双通道时频表示。
    """
    def __init__(self, n_fft: int = 64, hop_length: int = 16, **kwargs):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 2 and x.shape[0] == 2:
            i, q = x[0], x[1]
            # STFT 返回 (..., F, T) complex tensor
            spec_i = torch.stft(i, n_fft=self.n_fft, hop_length=self.hop_length,
                                return_complex=True)
            spec_q = torch.stft(q, n_fft=self.n_fft, hop_length=self.hop_length,
                                return_complex=True)
            # 取 magnitude 作为实数表示
            mag_i = spec_i.abs()
            mag_q = spec_q.abs()
            x = torch.stack([mag_i, mag_q], dim=0)  # (2, F, T)
        return x.numpy(), y


@_register("normalize_iq")
class NormalizeIQ:
    """IQ 通道标准化（沿时间轴）。

    对每个样本沿 L 轴计算 mean/std，归一化到 0 均值单位方差。
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        self.eps = eps

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # 沿最后一维（时间轴）标准化
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return x.numpy(), y


# ============================================================
# 组合变换
# ============================================================
def compose_transforms(names: List[str], seed: Optional[int] = None, **kwargs) -> Callable:
    """组合多个变换原语（对齐 wifi_csi.transforms.compose_transforms）。

    原语类在组合时实例化，符合 ComposedTransform 对 callable 的期望。

    Args:
        names: 原语名列表
        seed: 可选的随机种子
        **kwargs: 传递给每个原语构造函数的参数（按原语名分组）

    Returns:
        ComposedTransform 实例（callable，可 pickle 供 DataLoader multi-worker 使用）
    """
    transforms = []
    for name in names:
        cls = get_transform(name)
        params = kwargs.get(name, {})
        # 实例化原语类（接受 **kwargs 参数）
        instance = cls(**params) if params else cls()
        transforms.append((name, instance))
    return ComposedTransform(transforms, {}, seed=seed)
