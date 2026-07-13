"""通用 Transform 工具（跨场景共享）。

提供可 pickle 的 callable 类，替代闭包函数，确保 DataLoader num_workers>0 时
序列化不失败（闭包不可 pickle，会触发 PicklingError）。
"""

from typing import Callable, List


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
