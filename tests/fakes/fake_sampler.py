"""FakeSampler：SP Sampler Protocol 的测试替身。

实现 SenseFrame 自有 Sampler Protocol（非 Optuna BaseSampler）。
协议来源：senseframe.search_protocol.Sampler（@runtime_checkable Protocol）。

关键接口（SP Protocol）：
- name: str 类属性 — 采样策略名
- __init__(seed: Optional[int] = None) — StudyManager 通过 sampler_cls(seed=...) 实例化
- sample(search_space, history) -> Dict[str, Any] — 采样下一组参数

注意：本项目 Sampler Protocol 未声明 sample_independent / sample_relative
（那是 Optuna 协议）。FakeSampler 只实现 SP Protocol。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeSampler:
    """SP Sampler Protocol 的测试替身。

    sample() 返回搜索空间中每个参数的下界（可预测），供测试断言。
    通过 isinstance(x, Sampler) runtime_checkable 检查。

    Attributes:
        name: 采样策略名（默认 "fake"）
        seed: 随机种子（StudyManager 实例化时传入）
        sample_count: sample() 被调用次数
    """

    name: str = "fake"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.sample_count: int = 0
        self._warm_started: bool = False

    def sample(self, search_space: Any, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """采样下一组参数（返回搜索空间下界）。

        Args:
            search_space: SearchSpace 对象，支持两种接口：
                - search_protocol.SearchSpace: parameters: List[ParameterSpec]
                - scenes.base.SearchSpace: params: Dict[str, Dict]
            history: 历史试验列表

        Returns:
            参数字典（每个参数取下界）
        """
        self.sample_count += 1
        result: Dict[str, Any] = {}
        # 审查修复：支持 search_protocol.SearchSpace（parameters: List[ParameterSpec]）
        # 和 scenes.base.SearchSpace（params: Dict）两种接口
        if hasattr(search_space, "parameters"):
            for spec in search_space.parameters:
                result[spec.name] = getattr(spec, "low", 0)
        elif hasattr(search_space, "params"):
            for name, spec in search_space.params.items():
                result[name] = spec.get("low", 0) if isinstance(spec, dict) else getattr(spec, "low", 0)
        return result

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """元学习预热（可选方法，Protocol 声明但不强制）。"""
        self._warm_started = True
