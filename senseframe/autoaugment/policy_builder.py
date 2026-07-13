"""RFC-003 ε3 AutoAugment：策略构建器（P3.1.2）。

将增强策略参数 dict 翻译为 transform 函数 fn(x, y) -> (x, y)。

策略参数格式（由 AugmentationSearchSpace.to_sp_search_space 采样得到）：
    {
        "op_0": "time_jitter", "magnitude_0": 0.3, "probability_0": 0.5,
        "op_1": "noise",       "magnitude_1": 0.1, "probability_1": 0.8,
        ...
    }

构建逻辑：
- 每个槽位 i 的 (op_i, magnitude_i, probability_i) 构造一个增强原语
- 多个原语按顺序组合（pipeline），每个原语按 probability_i 概率应用
- op_i == "none" 时跳过该槽位

增强原语（WiFi CSI 时序信号适配）：
- time_jitter：时序抖动（在时间轴上随机偏移）
- freq_masking：频域掩码（随机掩码部分频率通道）
- noise：高斯噪声
- cutout：随机遮挡（时序片段置零）
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .search_space import AugmentationSearchSpace, SUPPORTED_AUGMENT_OPS

logger = logging.getLogger(__name__)


# ============================================================
# 增强原语实现
# ============================================================

def _time_jitter(x: np.ndarray, magnitude: float) -> np.ndarray:
    """时序抖动：在时间轴上随机偏移。

    Args:
        x: 输入数据，形状 (channels, length) 或 (length,)
        magnitude: 偏移幅度（0-1，相对于 length 的比例）
    """
    if x.ndim < 1:
        return x
    length = x.shape[-1]
    max_shift = max(1, int(length * magnitude * 0.1))
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return x
    return np.roll(x, shift, axis=-1).copy()


def _freq_masking(x: np.ndarray, magnitude: float) -> np.ndarray:
    """频域掩码：随机掩码部分频率通道。

    假设输入是时频表示（channels=frequency bins, length=time）。
    对一般时序输入，沿 channels 轴掩码（若 channels >= 2）。

    Args:
        x: 输入数据，形状 (channels, length) 或 (length,)
        magnitude: 掩码比例（0-1，相对于 channels 的比例）
    """
    if x.ndim < 2:
        return x
    n_channels = x.shape[0]
    if n_channels < 2:
        return x
    n_mask = max(1, int(n_channels * magnitude * 0.3))
    mask_start = random.randint(0, max(0, n_channels - n_mask))
    result = x.copy()
    result[mask_start:mask_start + n_mask] = 0
    return result


def _noise(x: np.ndarray, magnitude: float) -> np.ndarray:
    """高斯噪声：添加零均值高斯噪声。

    Args:
        x: 输入数据
        magnitude: 噪声标准差（0-1，相对于数据 std 的比例）
    """
    data_std = float(np.std(x)) if x.size > 0 else 1.0
    noise_std = magnitude * 0.1 * (data_std + 1e-8)
    noise = np.random.normal(0, noise_std, size=x.shape).astype(x.dtype)
    return x + noise


def _cutout(x: np.ndarray, magnitude: float) -> np.ndarray:
    """随机遮挡：时序片段置零。

    Args:
        x: 输入数据，形状 (channels, length) 或 (length,)
        magnitude: 遮挡长度（0-1，相对于 length 的比例）
    """
    if x.ndim < 1:
        return x
    length = x.shape[-1]
    cutout_len = max(1, int(length * magnitude * 0.2))
    start = random.randint(0, max(0, length - cutout_len))
    result = x.copy()
    result[..., start:start + cutout_len] = 0
    return result


# 增强算子注册表
_AUGMENT_OPS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "time_jitter": _time_jitter,
    "freq_masking": _freq_masking,
    "noise": _noise,
    "cutout": _cutout,
    "none": lambda x, m: x,  # no-op
}


def list_augment_ops() -> list:
    """列出所有增强算子名。"""
    return list(_AUGMENT_OPS.keys())


def get_augment_op(name: str) -> Optional[Callable[[np.ndarray, float], np.ndarray]]:
    """获取增强算子函数。"""
    return _AUGMENT_OPS.get(name)


# ============================================================
# P5 P1-A：module-level callable 类（可 pickle，替代闭包）
# ============================================================

class IdentityTransform:
    """Identity transform（module-level callable 类，可 pickle）。

    替代旧代码中的 ``lambda x, y: (x, y)`` 闭包。
    """

    def __call__(self, x, y=None):
        return x, y


class AutoAugmentTransform:
    """AutoAugment transform（module-level callable 类，可 pickle）。

    P5 P1-A：替代旧闭包实现，确保 DataLoader num_workers>0 时可序列化。
    旧代码在 build() 中返回 lambda/嵌套函数，导致 PicklingError。

    Args:
        ops_chain: [(op_name, magnitude, probability), ...] 增强算子链
        rng: 随机数生成器（None 时创建独立实例）
    """

    def __init__(self, ops_chain: list, rng: Optional[np.random.Generator] = None):
        self.ops_chain = list(ops_chain)
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, x, y=None):
        if not self.ops_chain:
            return x, y

        # 转为 numpy（若输入是 torch.Tensor）
        is_tensor = hasattr(x, "numpy")
        if is_tensor:
            x_np = x.numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            x_np = np.asarray(x)

        # 按顺序应用增强原语
        for op_name, magnitude, probability in self.ops_chain:
            if self._rng.random() > probability:
                continue
            op_fn = _AUGMENT_OPS.get(op_name)
            if op_fn is None:
                logger.warning(f"Unknown augment op: {op_name}, skip")
                continue
            x_np = op_fn(x_np, magnitude)

        # 转回原类型
        if is_tensor:
            import torch
            result_x = torch.from_numpy(x_np)
        else:
            result_x = x_np
        return result_x, y


# ============================================================
# 策略构建器
# ============================================================

class AutoAugmentPolicyBuilder:
    """增强策略构建器（P3.1.2）。

    将策略参数 dict 翻译为 transform 函数 fn(x, y) -> (x, y)。

    策略参数格式：
        {
            "op_0": "time_jitter", "magnitude_0": 0.3, "probability_0": 0.5,
            "op_1": "noise",       "magnitude_1": 0.1, "probability_1": 0.8,
        }

    build() 返回的 transform 函数：
        - 输入：x（np.ndarray 或 torch.Tensor），y（label）
        - 输出：(x, y)（增强后的样本）
        - 内部按顺序应用每个槽位的增强原语，每个原语按 probability_i 概率应用
    """

    def __init__(self, search_space: Optional[AugmentationSearchSpace] = None):
        """初始化策略构建器。

        Args:
            search_space: 增强搜索空间（用于验证参数范围，None 时不验证）
        """
        self.search_space = search_space

    def build(
        self,
        policy_params: Dict[str, Any],
    ) -> Callable[[Any, Any], Tuple[Any, Any]]:
        """构建 transform 函数。

        Args:
            policy_params: 策略参数 dict（由 SP Sampler 采样得到）

        Returns:
            transform 函数 fn(x, y) -> (x, y)
        """
        # 验证参数（若提供 search_space）
        if self.search_space is not None:
            errors = self.search_space.validate_params(policy_params)
            if errors:
                raise ValueError(
                    f"Invalid policy params: {errors}"
                )

        # 解析策略：提取每个槽位的 (op, magnitude, probability)
        ops_chain: list = []
        i = 0
        while f"op_{i}" in policy_params:
            op_name = policy_params[f"op_{i}"]
            magnitude = float(policy_params.get(f"magnitude_{i}", 0.0))
            probability = float(policy_params.get(f"probability_{i}", 1.0))
            ops_chain.append((op_name, magnitude, probability))
            i += 1

        if not ops_chain:
            # 空策略：返回 identity（module-level callable 类，可 pickle）
            return IdentityTransform()

        # P5 P1-A：返回 AutoAugmentTransform 实例（module-level callable 类，可 pickle）
        # 旧代码返回闭包函数，DataLoader num_workers>0 时 PicklingError
        return AutoAugmentTransform(ops_chain)

    def build_eval_transform(
        self,
        policy_params: Dict[str, Any],
    ) -> Callable[[Any, Any], Tuple[Any, Any]]:
        """构建评估 transform（无增强，仅 identity）。

        评估阶段不应应用增强，返回 identity 函数。

        Args:
            policy_params: 策略参数（忽略，仅用于接口对称）

        Returns:
            identity transform fn(x, y) -> (x, y)
        """
        return IdentityTransform()


def make_policy_from_params(
    policy_params: Dict[str, Any],
    search_space: Optional[AugmentationSearchSpace] = None,
) -> Callable[[Any, Any], Tuple[Any, Any]]:
    """便捷工厂：从参数构造 transform 函数。"""
    builder = AutoAugmentPolicyBuilder(search_space=search_space)
    return builder.build(policy_params)


__all__ = [
    "AutoAugmentPolicyBuilder",
    "make_policy_from_params",
    "list_augment_ops",
    "get_augment_op",
]
