"""RFC-003 ε2 NAS：进化算法采样器（P2.8）。

EvolutionarySampler 满足 SP Sampler Protocol（@runtime_checkable）。

算法（简化进化策略）：
1. 初始化阶段（population 未满）：随机生成个体
2. 进化阶段（population 已满）：
   a. 锦标赛选择（tournament_select）：从 population 中选 k 个，取最优
   b. 变异（mutate）：对父代参数进行随机扰动
   c. 返回子代参数

fitness 来源：从 history 中提取每个 trial 的 value（SP Tell 上报的指标）。

注册：
- 注册到 SP Sampler 注册表（register_sampler("evolutionary", EvolutionarySampler)）
- 不注册为 Pruner（EvolutionarySampler 不是早停策略）
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..search_protocol import (
    Sampler,
    SearchSpace,
    register_sampler,
)

logger = logging.getLogger(__name__)


class EvolutionarySampler:
    """进化算法采样器（P2.8）。

    满足 SP Sampler Protocol（name + sample 方法）。
    通过 register_sampler("evolutionary", EvolutionarySampler) 注册到 SP。

    Args:
        population_size: 种群大小（达到后开始进化）
        mutation_rate: 变异概率（每个参数被扰动的概率）
        tournament_size: 锦标赛选择大小
        direction: 优化方向，"maximize" 或 "minimize"
        seed: 随机种子（None 时不固定）
    """
    name = "evolutionary"

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        direction: str = "maximize",
        seed: Optional[int] = None,
    ):
        if population_size < 2:
            raise ValueError(
                f"population_size must be >= 2, got {population_size}"
            )
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(
                f"mutation_rate must be in [0, 1], got {mutation_rate}"
            )
        if tournament_size < 1:
            raise ValueError(
                f"tournament_size must be >= 1, got {tournament_size}"
            )
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.direction = direction  # "maximize" / "minimize"
        # population: list of (params, fitness)；fitness=None 表示尚未评估
        self._population: List[Tuple[Dict[str, Any], Optional[float]]] = []
        self._rng = random.Random(seed)

    # ============================================================
    # SP Sampler Protocol 实现
    # ============================================================
    def sample(
        self,
        search_space: SearchSpace,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """采样：初始化阶段随机，进化阶段变异。

        Args:
            search_space: SP SearchSpace（含参数规格）
            history: 已完成 trial 的历史（list of dict，含 "params" 和 "result"）
                result 字段含 "value"（fitness）

        Returns:
            采样参数 dict
        """
        # 同步 population：从 history 中补充已评估的个体
        self._sync_population_from_history(history)

        # 初始化阶段：population 未满，随机生成
        if len(self._population) < self.population_size:
            params = self._random_init(search_space)
            self._population.append((params, None))
            return params

        # 进化阶段：锦标赛选择 + 变异
        parent = self._tournament_select()
        child = self._mutate(parent, search_space)
        self._population.append((child, None))
        return child

    # ============================================================
    # 内部方法
    # ============================================================
    def _sync_population_from_history(self, history: List[Dict[str, Any]]) -> None:
        """从 SP history 同步已评估个体的 fitness。

        SP history entry 结构：
        - entry["params"] 或 entry["strategy"]：参数 dict
        - entry["result"]["value"]：fitness 值
        - entry["status"]：trial 状态
        """
        # 用 trial_id / params 哈希去重（避免重复添加）
        # 简化：用 params 的 frozenset 作为标识
        seen = set()
        for params, fitness in self._population:
            if fitness is not None:
                key = self._params_key(params)
                seen.add(key)

        for entry in history:
            params = entry.get("params") or entry.get("strategy") or {}
            if not params:
                continue
            key = self._params_key(params)
            if key in seen:
                # 已在 population 中且已评估，更新 fitness
                for i, (p, f) in enumerate(self._population):
                    if self._params_key(p) == key and f is None:
                        result = entry.get("result") or {}
                        value = result.get("value") if isinstance(result, dict) else None
                        if value is not None:
                            self._population[i] = (p, float(value))
                        break
                continue

            # 新个体（可能由其他 Sampler 生成），加入 population
            result = entry.get("result") or {}
            value = result.get("value") if isinstance(result, dict) else None
            fitness = float(value) if value is not None else None
            self._population.append((params, fitness))
            seen.add(key)

        # 修剪：保留最近 population_size * 2 个个体（避免无限增长）
        if len(self._population) > self.population_size * 2:
            self._population = self._population[-self.population_size * 2:]

    @staticmethod
    def _params_key(params: Dict[str, Any]) -> str:
        """生成 params 的稳定 key（用于去重）。"""
        try:
            return repr(sorted(params.items()))
        except Exception:
            return str(params)

    def _random_init(self, search_space: SearchSpace) -> Dict[str, Any]:
        """随机初始化个体（与 RandomSampler 等价）。"""
        params: Dict[str, Any] = {}
        for p in search_space.parameters:
            if p.type == "float":
                if p.log and p.low is not None and p.high is not None:
                    params[p.name] = math.exp(
                        self._rng.uniform(math.log(p.low), math.log(p.high))
                    )
                else:
                    params[p.name] = self._rng.uniform(p.low or 0.0, p.high or 1.0)
            elif p.type == "int":
                params[p.name] = self._rng.randint(int(p.low or 0), int(p.high or 100))
            elif p.type == "categorical" and p.choices:
                params[p.name] = self._rng.choice(p.choices)
        return params

    def _tournament_select(self) -> Dict[str, Any]:
        """锦标赛选择：从已评估个体中选 tournament_size 个，返回最优的 params。"""
        evaluated = [(p, f) for p, f in self._population if f is not None]
        if not evaluated:
            # 无已评估个体，随机选一个
            idx = self._rng.randint(0, len(self._population) - 1)
            return self._population[idx][0]

        # 随机选 tournament_size 个（不重复）
        k = min(self.tournament_size, len(evaluated))
        contenders = self._rng.sample(evaluated, k)

        # 选最优（maximize: 最大值；minimize: 最小值）
        if self.direction == "maximize":
            best = max(contenders, key=lambda x: x[1])
        else:
            best = min(contenders, key=lambda x: x[1])
        return best[0]

    def _mutate(
        self,
        parent: Dict[str, Any],
        search_space: SearchSpace,
    ) -> Dict[str, Any]:
        """对父代参数进行变异。

        每个参数以 mutation_rate 概率被扰动：
        - categorical：随机选另一个 choice
        - int：在 [low, high] 内随机扰动（±10% 范围，至少 ±1）
        - float：在 [low, high] 内随机扰动（±10% 范围）
        """
        child = dict(parent)
        param_specs = {p.name: p for p in search_space.parameters}

        for name, value in parent.items():
            if name not in param_specs:
                # 未知参数，原样保留
                continue
            if self._rng.random() > self.mutation_rate:
                continue  # 不变异

            spec = param_specs[name]
            if spec.type == "categorical" and spec.choices:
                # 随机选另一个 choice（若 choices 数量 >= 2）
                if len(spec.choices) >= 2:
                    other_choices = [c for c in spec.choices if c != value]
                    child[name] = self._rng.choice(other_choices)
            elif spec.type == "int":
                low = int(spec.low or 0)
                high = int(spec.high or 100)
                # 扰动幅度：max(1, 10% 范围)
                delta = max(1, (high - low) // 10)
                new_val = value + self._rng.randint(-delta, delta)
                child[name] = max(low, min(high, new_val))
            elif spec.type == "float":
                low = float(spec.low or 0.0)
                high = float(spec.high or 1.0)
                delta = (high - low) * 0.1
                new_val = value + self._rng.uniform(-delta, delta)
                child[name] = max(low, min(high, new_val))

        return child

    # ============================================================
    # 工具方法（用于测试 / 调试）
    # ============================================================
    def population_size_actual(self) -> int:
        """返回当前 population 实际大小。"""
        return len(self._population)

    def evaluated_count(self) -> int:
        """返回已评估个体数。"""
        return sum(1 for _, f in self._population if f is not None)


# 注册到 SP Sampler 注册表
register_sampler("evolutionary", EvolutionarySampler)


__all__ = ["EvolutionarySampler"]
