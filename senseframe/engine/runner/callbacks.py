"""Callback 基类与生命周期声明机制。

P0-1：解决 stage_eval 复用 stage_train trainer 时 callback 跨 stage 触发问题。
设计：声明式 active_stages 属性 + Pipeline.run stage 边界设置 _active 标志。
"""
from typing import Set

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl


class StageAwareCallback(pl.Callback):
    """声明 active_stages 的 Callback 基类。

    active_stages 为空集合时在所有 stage 激活（默认行为，向后兼容）。
    Pipeline.run 在 stage 边界调用 set_active(stage_name) 设置 _active 标志。

    子类通过声明 active_stages 类属性约束自身生命周期：
        class IntermediateMetricLogger(StageAwareCallback):
            active_stages = {"train"}  # 只在 stage_train 激活

    在 hook 入口处调用 self.is_active() 判断是否跳过。
    """

    active_stages: Set[str] = set()  # 空集合 = 所有 stage 激活

    def set_active(self, stage: str) -> None:
        """由 Pipeline.run 在 stage 边界调用，设置当前激活状态。"""
        # 空集合表示所有 stage 激活（向后兼容）
        if not self.active_stages:
            self._active = True
        else:
            self._active = stage in self.active_stages

    def is_active(self) -> bool:
        """当前是否处于激活 stage。默认 True（未设置时保持激活）。"""
        return getattr(self, "_active", True)


class FrozenDict(dict):
    """stage_train 后 intermediate_values 冻结，stage_eval 写入直接报错。

    防御性兜底：即使 callback active_stages 配置错误，也能在 stage_eval
    写入 intermediate_values 时立即抛错，避免污染数据。
    """

    def __setitem__(self, key, value):
        raise RuntimeError(
            f"intermediate_values frozen after stage_train, "
            f"cannot write key={key!r}. "
            f"Likely cause: stage_eval trainer.validate() triggered "
            f"IntermediateMetricLogger callback. "
            f"Check callback active_stages configuration."
        )

    def __repr__(self) -> str:
        return f"FrozenDict({super().__repr__()})"
