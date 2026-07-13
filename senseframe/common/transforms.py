"""通用 Transform 工具（跨场景共享）。

提供可 pickle 的 callable 类，替代闭包函数，确保 DataLoader num_workers>0 时
序列化不失败（闭包不可 pickle，会触发 PicklingError）。
"""

import inspect
from typing import Callable, Dict, List, Optional

import numpy as np


class ChainedTransform:
    """依次应用多个 transform 的 callable 类（可 pickle）。

    替代容器中的 _combined / _train_with_aug 闭包。
    每个 transform 签名：fn(x, y) -> (x, y)

    Args:
        transforms: transform 列表，按顺序应用
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x, y=None):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class ComposedTransform:
    """组合多个 transform 原语的 callable 类（可 pickle）。

    替代旧 composed 闭包，确保 DataLoader num_workers>0 时序列化不失败。
    依次对输入应用 (name, fn) 列表中的原语，处理 torch.Tensor ↔ numpy 转换。

    通过 inspect.signature 自动判断每个原语是否接受 rng 和 y 参数：
    - 接受 y 的原语（如 detection 场景的 mixup）会同时变换 x 和 y
    - 接受 rng 的原语会注入独立 Generator

    P3 上策：持有独立 np.random.Generator，消除对全局 np.random 状态的依赖。
    Generator 可 pickle，随对象传递到 worker。
    """

    def __init__(self, transforms, kwargs, seed: Optional[int] = None):
        # transforms: List[Tuple[str, Callable]]；kwargs: Dict[str, dict]
        self.transforms = list(transforms)
        self.kwargs = dict(kwargs)
        # 独立 Generator：相同 seed 产生相同序列，不同 ComposedTransform 实例独立
        self.rng = np.random.default_rng(seed) if seed is not None else None
        # 预计算每个原语是否接受 rng / y 参数，避免 try/except 误吞原语内部 TypeError
        self._accepts_rng = [
            'rng' in inspect.signature(fn).parameters for _, fn in self.transforms
        ]
        self._accepts_y = [
            'y' in inspect.signature(fn).parameters for _, fn in self.transforms
        ]

    def __call__(self, x, y=None):
        import torch
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        # 仅当存在接受 y 的原语时才转换 y（保持无 y 变换场景的 y 透传行为）
        process_y = any(self._accepts_y) and y is not None
        if process_y:
            y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        else:
            y_np = None
        for (name, fn), accepts_rng, accepts_y in zip(
            self.transforms, self._accepts_rng, self._accepts_y
        ):
            params = self.kwargs.get(name, {})
            rng_kw = {'rng': self.rng} if accepts_rng else {}
            if accepts_y and y_np is not None:
                x_np, y_np = fn(x_np, y_np, **rng_kw, **params)
            else:
                x_np = fn(x_np, **rng_kw, **params)
        x_out = torch.from_numpy(x_np).float() if isinstance(x, torch.Tensor) else x_np
        if process_y:
            y_out = (
                torch.from_numpy(y_np).float()
                if isinstance(y, torch.Tensor)
                else y_np
            )
            return x_out, y_out
        return x_out, y
