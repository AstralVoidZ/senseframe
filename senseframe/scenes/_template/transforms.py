"""{SCENE_NAME} 场景的变换原语（自动生成模板）。"""
import numpy as np
from typing import Callable, Dict, List, Optional

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def get_transform(name: str) -> Callable:
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {name}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    return TRANSFORM_REGISTRY[name]


def list_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


def compose_transforms(names: List[str], seed: Optional[int] = None, **kwargs) -> Callable:
    """组合多个变换原语。

    Args:
        names: 原语名列表
        seed: 可选的随机种子（P3 上策，详见 wifi_csi.transforms.compose_transforms）
        **kwargs: 传递给每个原语的参数
    """
    fns = [get_transform(n) for n in names]
    return ComposedTransform(fns, seed=seed)


class ComposedTransform:
    """组合多个 transform 原语的 callable 类（可 pickle）。

    替代旧 _composed 闭包，确保 DataLoader num_workers>0 时序列化不失败。
    P3 上策：持有独立 np.random.Generator，消除对全局 np.random 状态的依赖。
    """

    def __init__(self, fns, seed: Optional[int] = None):
        import inspect
        self.fns = list(fns)
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self._accepts_rng = [
            'rng' in inspect.signature(fn).parameters for fn in self.fns
        ]

    def __call__(self, x, y):
        for fn, accepts_rng in zip(self.fns, self._accepts_rng):
            if accepts_rng:
                x, y = fn(x, y, rng=self.rng)
            else:
                x, y = fn(x, y)
        return x, y
