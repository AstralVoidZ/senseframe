"""FakePruner：SP Pruner Protocol 的测试替身。

实现 SenseFrame 自有 Pruner Protocol（非 Optuna BasePruner）。
协议来源：senseframe.search_protocol.Pruner（@runtime_checkable Protocol）。

关键接口（SP Protocol）：
- name: str 类属性 — 早停策略名
- should_prune(trial_id, intermediate_values, rung) -> bool — 早停判断

注意：本项目 Pruner Protocol 未声明 prune 方法（那是 Optuna 协议）。
FakePruner 只实现 SP Protocol。

调用上下文：IntermediateMetricLogger.on_validation_epoch_end 在
trainer.sanity_checking=False 时调用 pruner.should_prune(...)；
返回 True 则设 trainer.should_stop=True。
"""
from __future__ import annotations

from typing import Dict, Optional


class FakePruner:
    """SP Pruner Protocol 的测试替身。

    should_prune() 返回预设的布尔值，供测试断言早停编排行为。
    通过 isinstance(x, Pruner) runtime_checkable 检查。

    Attributes:
        name: 早停策略名（默认 "fake"）
        should_prune_value: 预设返回值（默认 False）
        call_count: should_prune 被调用次数
        last_trial_id: 最后一次调用的 trial_id
        last_rung: 最后一次调用的 rung
    """

    name: str = "fake"

    def __init__(self, should_prune: bool = False) -> None:
        self.should_prune_value: bool = should_prune
        self.call_count: int = 0
        self.last_trial_id: Optional[str] = None
        self.last_rung: Optional[int] = None
        self.last_intermediate_values: Optional[Dict[int, float]] = None

    def should_prune(
        self,
        trial_id: str,
        intermediate_values: Dict[int, float],
        rung: int,
    ) -> bool:
        """早停判断（返回预设值，记录调用参数）。

        Args:
            trial_id: 试验 ID
            intermediate_values: 中间指标字典 {epoch: value}
            rung: 当前 rung（1-indexed）

        Returns:
            预设的 should_prune_value
        """
        self.call_count += 1
        self.last_trial_id = trial_id
        self.last_rung = rung
        self.last_intermediate_values = dict(intermediate_values)
        return self.should_prune_value
