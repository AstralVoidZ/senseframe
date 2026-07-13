"""RFC-003 SP：搜索协议 — Ask-Tell 标准化搜索接口。

对齐 Optuna Study/Trial/Sampler 范式 + Google Vizier RPC 思想。
让 SenseFrame 可作为搜索服务被 AutoML 主控调用。
"""
from __future__ import annotations

import logging
import threading
import uuid

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from .exploration import ExplorationTracker

if TYPE_CHECKING:
    # P3.2.4：避免循环导入，HistoryStore 类型注解用 TYPE_CHECKING
    from .exploration import HistoryStore  # noqa: F401

# 修复（5.12 / 5.4）：模块级 logger + OTel 埋点常量导入
# 旧逻辑：StudyManager ask/tell/create_study/best_trial 全部零日志；
# ML_SEARCH_COVERAGE_RATIO 定义但从未被 record_training_metric 调用。
logger = logging.getLogger(__name__)
try:
    from .observability_otel import record_training_metric, ML_SEARCH_COVERAGE_RATIO
except ImportError:
    # OTel 未安装时降级为 no-op
    def record_training_metric(*args, **kwargs):
        pass

    ML_SEARCH_COVERAGE_RATIO = "ml.search.coverage_ratio"


@dataclass
class ParameterSpec:
    """参数规格（SP-1）。"""
    name: str
    type: str  # "float" / "int" / "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "type": self.type,
            "low": self.low, "high": self.high,
            "choices": self.choices, "log": self.log, "step": self.step,
        }


@dataclass
class SearchSpace:
    """搜索空间（SP-1）。"""
    parameters: List[ParameterSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"parameters": [p.to_dict() for p in self.parameters]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchSpace":
        params = [ParameterSpec(**p) for p in d.get("parameters", [])]
        return cls(parameters=params)


@dataclass
class StudySpec:
    """Study 规格（SP-1）。"""
    study_id: str
    name: str
    direction: str  # "maximize" / "minimize"
    search_space: SearchSpace
    sampler: str = "random"
    status: str = "running"  # running / completed / stopped
    created_at: str = ""
    completed_at: str = ""
    # P3-1：study 级种子，None 时使用全局随机。
    # 每个 trial 的实际种子 = seed + trial_index，既可复现又不重复。
    # 彻底消除 sampler 对全局 random 模块的依赖（避免被 set_seed 重置）。
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id, "name": self.name,
            "direction": self.direction,
            "search_space": self.search_space.to_dict(),
            "sampler": self.sampler, "status": self.status,
            "created_at": self.created_at, "completed_at": self.completed_at,
            "seed": self.seed,
        }


@dataclass
class TrialSpec:
    """Trial 规格 — Ask 结果（SP-2）。"""
    trial_id: str
    study_id: str
    params: Dict[str, Any]
    state: str = "running"  # running / completed / failed / pruned
    datetime_start: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id, "study_id": self.study_id,
            "params": self.params, "state": self.state,
            "datetime_start": self.datetime_start,
        }


@dataclass
class TrialResult:
    """Trial 结果 — Tell 结果（SP-2）。"""
    trial_id: str
    study_id: str
    params: Dict[str, Any]
    value: float
    intermediate_values: Dict[int, float] = field(default_factory=dict)
    state: str = "completed"  # completed / failed / pruned
    datetime_complete: str = ""
    feedback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id, "study_id": self.study_id,
            "params": self.params, "value": self.value,
            "intermediate_values": self.intermediate_values,
            "state": self.state, "datetime_complete": self.datetime_complete,
            "feedback": self.feedback,
        }


# ============================================================
# SP-3: Sampler 注册表 + 内置 Random/Grid
# ============================================================

@runtime_checkable
class Sampler(Protocol):
    """Sampler 契约（SP-3，P0.5 新增；P3.2.1 扩展 warm_start）。

    所有 Sampler 必须满足此 Protocol：拥有 `name` 类属性与 `sample` 方法。
    `@runtime_checkable` 使 `isinstance(sampler, Sampler)` 可用，
    仅校验属性/方法存在性，不校验签名。

    P3.2.1 新增：可选 `warm_start` 方法（非强制，不实现则 no-op）。
    由于 `@runtime_checkable` Protocol 不强制实现所有方法，现有
    RandomSampler/GridSampler/EvolutionarySampler/AutoAugmentSampler/
    ASHASampler/HyperbandSampler 无需修改即可继续通过
    `isinstance(x, Sampler)` 检查。EvolutionarySampler 和
    AutoAugmentSampler 实现 warm_start 作为元学习受益示例（用源数据集
    成功策略作为初始 population 的种子）。
    """
    name: str  # 类属性（采样策略名）

    def sample(
        self,
        search_space: "SearchSpace",
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]: ...

    # P3.2.1 新增：可选 warm_start 方法（非强制，不实现则 no-op）
    def warm_start(
        self,
        source_history: List[Dict[str, Any]],
    ) -> None: ...


_SAMPLERS: Dict[str, type] = {}


def register_sampler(name: str, sampler_cls: type) -> None:
    """注册 Sampler（SP-3）。"""
    _SAMPLERS[name] = sampler_cls


def get_sampler(name: str) -> Optional[type]:
    """获取 Sampler 类。"""
    return _SAMPLERS.get(name)


def list_samplers() -> List[str]:
    """列出已注册的 Sampler。"""
    return list(_SAMPLERS.keys())


class RandomSampler:
    """随机采样器（SP-3 内置）— 持有独立 RNG，不受全局 set_seed 影响。

    P3-1 修复：旧代码使用全局 random 模块，被 stage_preflight 的 set_seed(42)
    周期性重置，导致 HPO 多 trial 产出相同参数。现在 __init__ 创建独立
    np.random.Generator，与训练 pipeline 的 RNG 完全隔离。

    P3-2 修复：旧代码忽略 ParameterSpec.step 字段（random.randint 不考虑 step），
    导致 batch_size=9 在 step=8 约束下被错误产出。现在统一消费 step。
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def sample(self, search_space: SearchSpace, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        params = {}
        for p in search_space.parameters:
            if p.type == "float":
                if p.log and p.low and p.high:
                    # log-uniform 采样
                    # numpy 2.x 的 Generator 无 loguniform 方法，用 exp(uniform(log)) 等价实现
                    params[p.name] = float(
                        np.exp(self._rng.uniform(np.log(p.low), np.log(p.high)))
                    )
                else:
                    params[p.name] = float(self._rng.uniform(p.low or 0, p.high or 1))
            elif p.type == "int":
                low, high = int(p.low or 0), int(p.high or 100)
                if p.step:
                    # P3-2：消费 step 字段——采样值必须是 low + k*step 的形式
                    step_int = int(p.step)
                    n_steps = (high - low) // step_int + 1
                    params[p.name] = low + int(self._rng.integers(0, n_steps)) * step_int
                else:
                    params[p.name] = int(self._rng.integers(low, high + 1))
            elif p.type == "categorical" and p.choices:
                params[p.name] = str(self._rng.choice(p.choices))
        return params

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """P3.2.1 ε4 元学习：no-op（RandomSampler 不从 warm-start 受益）。

        保留此方法仅为满足 Python 3.12+ @runtime_checkable Protocol
        的 isinstance 检查（要求所有 Protocol 方法存在）。RandomSampler
        是无状态采样器，不从源数据集历史偏向中受益。
        """
        return None


class GridSampler:
    """网格采样器（SP-3 内置）— 确定性采样，seed 参数被接受但忽略。"""
    name = "grid"

    def __init__(self, seed: Optional[int] = None):
        # GridSampler 按网格顺序采样，不需要 RNG。接受 seed 仅为满足
        # StudyManager.ask() 的统一调用接口（sampler_cls(seed=...)）。
        self._grid_index: Dict[str, int] = {}  # study_id -> index

    def sample(self, search_space: SearchSpace, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 生成网格点
        import itertools
        grids = []
        for p in search_space.parameters:
            if p.type == "float":
                step = p.step or (p.high - p.low) / 5 if p.low and p.high else 0.1
                grids.append([p.low + i * step for i in range(int((p.high - p.low) / step) + 1)] if p.low and p.high else [0])
            elif p.type == "int":
                low, high = int(p.low or 0), int(p.high or 10)
                if p.step:
                    # P3-2：消费 step 字段
                    grids.append(list(range(low, high + 1, int(p.step))))
                else:
                    grids.append(list(range(low, high + 1)))
            elif p.type == "categorical" and p.choices:
                grids.append(p.choices)
            else:
                grids.append([None])

        all_combos = list(itertools.product(*grids))
        idx = len(history) % len(all_combos)
        combo = all_combos[idx]
        return {p.name: v for p, v in zip(search_space.parameters, combo)}

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """P3.2.1 ε4 元学习：no-op（GridSampler 按 grid 顺序采样，不受 history 影响）。

        保留此方法仅为满足 Python 3.12+ @runtime_checkable Protocol
        的 isinstance 检查。
        """
        return None


# 注册内置 Sampler
register_sampler("random", RandomSampler)
register_sampler("grid", GridSampler)
# P1.3: TPE 诚实降级——不伪装为 TPE，显式标注为 random_fallback
# 当 Optuna 可用时 StudyManager 会优先使用 Optuna TPE；此处为纯 Python fallback
register_sampler("tpe", RandomSampler)  # ⚠️ random_fallback: 真正 TPE 需 Optuna，此处退化为随机搜索
register_sampler("random_fallback", RandomSampler)  # 显式别名，避免 tpe 名称误导


# ============================================================
# SP-3 扩展: Pruner 注册表（P2.1 新增，ε5 Multi-fidelity 早停）
# ============================================================

@runtime_checkable
class Pruner(Protocol):
    """Pruner 契约（SP-3 扩展，P2.1）。

    所有 Pruner 必须满足此 Protocol：拥有 `name` 类属性与 `should_prune` 方法。
    `@runtime_checkable` 使 `isinstance(pruner, Pruner)` 可用，
    仅校验属性/方法存在性，不校验签名。

    契约语义：
    - `name`：Pruner 策略名（用于注册表 key + StudySpec 持久化）
    - `should_prune`：基于已收集的 intermediate_values 与当前 rung，
      返回 True 表示丢弃该 trial（早停），False 表示继续训练到下一 rung。

    `intermediate_values` 为 Dict[int, float]，key 为 epoch/rung 序号，
    value 为该 epoch 的目标指标（如 val_accuracy）。
    `rung` 为当前所处的 rung 序号（从 0 开始）。
    """
    name: str  # 类属性（早停策略名）

    def should_prune(
        self,
        trial_id: str,
        intermediate_values: Dict[int, float],
        rung: int,
    ) -> bool: ...


_PRUNERS: Dict[str, type] = {}


def register_pruner(name: str, pruner_cls: type) -> None:
    """注册 Pruner（SP-3 扩展，P2.1）。

    Args:
        name: Pruner 策略名（如 "hyperband" / "asha" / "median"）
        pruner_cls: Pruner 类（必须满足 Pruner Protocol）
    """
    _PRUNERS[name] = pruner_cls


def get_pruner(name: str) -> Optional[type]:
    """获取 Pruner 类。

    Args:
        name: Pruner 策略名

    Returns:
        Pruner 类；未注册返回 None
    """
    return _PRUNERS.get(name)


def list_pruners() -> List[str]:
    """列出已注册的 Pruner 策略名。

    Returns:
        Pruner 策略名列表（按注册顺序）
    """
    return list(_PRUNERS.keys())


# ============================================================
# SP-3 扩展: 内置 Pruner 实现（P2.2，ε5 Multi-fidelity）
# ============================================================
# HyperbandSampler / ASHASampler 同时满足 Sampler 和 Pruner Protocol。
# 采样策略与 RandomSampler 相同（随机），区别在 should_prune 逻辑。
# 注册到 sampler 与 pruner 两个注册表，使 StudyManager 可通过 sampler="asha"
# 同时获得采样与早停能力。
#
# 算法参考：
# - ASHA: Asynchronous Successive Halving (Li et al., 2018)
# - Hyperband: Multi-bracket SHA (Li et al., 2017)


class ASHASampler:
    """ASHA（Asynchronous Successive Halving）Sampler + Pruner（P2.2）。

    ASHA 是 SHA 的异步版本，用于 Multi-fidelity 早停。

    算法：
    - max_resource (R): 最大资源量（epoch 数）
    - eta (η): 降比率（每 rung 保留 1/η 的 trial）
    - 在每个 rung，将 trial 按 intermediate value 排序，保留 top 1/η，剪枝其余

    同时满足 Sampler Protocol（sample 方法）与 Pruner Protocol（should_prune 方法）。

    Args:
        max_resource: 最大资源量（epoch 数），默认 81
        eta: 降比率，默认 3
        direction: 优化方向，"maximize" 或 "minimize"
    """
    name = "asha"

    def __init__(self, max_resource: int = 81, eta: int = 3, direction: str = "maximize"):
        self.max_resource = max_resource
        self.eta = eta
        self.direction = direction  # "maximize" / "minimize"
        # rung_index -> [(value, trial_id)]
        self._rungs: Dict[int, List[Tuple[float, str]]] = {}

    def sample(self, search_space: SearchSpace, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """随机采样（与 RandomSampler 相同，ASHA 的区别仅在 should_prune）。"""
        import random
        params = {}
        for p in search_space.parameters:
            if p.type == "float":
                if p.log and p.low and p.high:
                    import math
                    params[p.name] = math.exp(random.uniform(math.log(p.low), math.log(p.high)))
                else:
                    params[p.name] = random.uniform(p.low or 0, p.high or 1)
            elif p.type == "int":
                params[p.name] = random.randint(int(p.low or 0), int(p.high or 100))
            elif p.type == "categorical" and p.choices:
                params[p.name] = random.choice(p.choices)
        return params

    def should_prune(self, trial_id: str, intermediate_values: Dict[int, float], rung: int) -> bool:
        """ASHA 早停判断。

        在 rung 处将 trial 按 intermediate value 排序，保留 top 1/η，剪枝其余。
        若 rung 处 trial 数不足 η，则不剪枝（数据不足）。

        Args:
            trial_id: Trial 标识
            intermediate_values: {epoch: value} 已收集的中间值（1-indexed epoch）
            rung: 当前 rung 序号（1-indexed epoch，由 max(intermediate_values.keys()) 得来）

        Returns:
            True 表示应剪枝该 trial，False 表示继续训练
        """
        if rung not in intermediate_values:
            return False
        value = intermediate_values[rung]

        # 记录 trial 在该 rung 的值（避免重复记录同一 trial）
        if rung not in self._rungs:
            self._rungs[rung] = []
        if not any(tid == trial_id for _, tid in self._rungs[rung]):
            self._rungs[rung].append((value, trial_id))

        # 数据点不足，不剪枝
        if len(self._rungs[rung]) < self.eta:
            return False

        # 排序：maximize 时降序（高值在前），minimize 时升序（低值在前）
        reverse = (self.direction == "maximize")
        sorted_vals = sorted(self._rungs[rung], key=lambda x: x[0], reverse=reverse)
        # 保留 top 1/η
        n_keep = max(1, len(self._rungs[rung]) // self.eta)
        kept_ids = set(tid for _, tid in sorted_vals[:n_keep])

        return trial_id not in kept_ids

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """P3.2.1 ε4 元学习：no-op（ASHA 采样与 RandomSampler 等价，区别仅在 should_prune）。

        保留此方法仅为满足 Python 3.12+ @runtime_checkable Protocol
        的 isinstance 检查。
        """
        return None


class HyperbandSampler:
    """Hyperband Sampler + Pruner（P2.2）。

    Hyperband 是多 bracket 的 SHA，通过不同 bracket 探索不同资源分配策略。

    算法：
    - 在 ASHA 基础上增加 brackets 维度
    - bracket 数: floor(log_η(R)) + 1
    - 每个 bracket 有独立的 rung 跟踪
    - trial 通过 trial_id hash 分配到 bracket（确定性 + 均匀分布）

    同时满足 Sampler Protocol 与 Pruner Protocol。

    Args:
        max_resource: 最大资源量（epoch 数），默认 81
        eta: 降比率，默认 3
        direction: 优化方向，"maximize" 或 "minimize"
    """
    name = "hyperband"

    def __init__(self, max_resource: int = 81, eta: int = 3, direction: str = "maximize"):
        self.max_resource = max_resource
        self.eta = eta
        self.direction = direction
        # bracket 数: floor(log_η(R)) + 1，至少为 1
        import math
        if max_resource > 1 and eta > 1:
            self.n_brackets = max(1, int(math.log(max_resource) / math.log(eta)) + 1)
        else:
            self.n_brackets = 1
        # 每个 bracket 独立的 rung 跟踪
        self._brackets: List[Dict[int, List[Tuple[float, str]]]] = [
            {} for _ in range(self.n_brackets)
        ]
        # trial_id -> bracket index（确定性分配）
        self._trial_bracket: Dict[str, int] = {}

    def _get_bracket(self, trial_id: str) -> int:
        """获取 trial 所属的 bracket（基于 trial_id hash 确定性分配）。"""
        if trial_id not in self._trial_bracket:
            import hashlib
            h = int(hashlib.md5(trial_id.encode()).hexdigest(), 16)
            self._trial_bracket[trial_id] = h % self.n_brackets
        return self._trial_bracket[trial_id]

    def sample(self, search_space: SearchSpace, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """随机采样（与 RandomSampler 相同，Hyperband 的区别仅在 should_prune）。"""
        import random
        params = {}
        for p in search_space.parameters:
            if p.type == "float":
                if p.log and p.low and p.high:
                    import math
                    params[p.name] = math.exp(random.uniform(math.log(p.low), math.log(p.high)))
                else:
                    params[p.name] = random.uniform(p.low or 0, p.high or 1)
            elif p.type == "int":
                params[p.name] = random.randint(int(p.low or 0), int(p.high or 100))
            elif p.type == "categorical" and p.choices:
                params[p.name] = random.choice(p.choices)
        return params

    def should_prune(self, trial_id: str, intermediate_values: Dict[int, float], rung: int) -> bool:
        """Hyperband 早停判断（基于 bracket 内的 ASHA 逻辑）。

        每个 bracket 独立跟踪 rung，trial 在所属 bracket 内与同 rung 的其他 trial 比较。
        """
        if rung not in intermediate_values:
            return False
        value = intermediate_values[rung]

        bracket_idx = self._get_bracket(trial_id)
        bracket_rungs = self._brackets[bracket_idx]

        # 记录 trial 在该 rung 的值（避免重复记录）
        if rung not in bracket_rungs:
            bracket_rungs[rung] = []
        if not any(tid == trial_id for _, tid in bracket_rungs[rung]):
            bracket_rungs[rung].append((value, trial_id))

        # 数据点不足，不剪枝
        if len(bracket_rungs[rung]) < self.eta:
            return False

        # 排序：maximize 时降序，minimize 时升序
        reverse = (self.direction == "maximize")
        sorted_vals = sorted(bracket_rungs[rung], key=lambda x: x[0], reverse=reverse)
        n_keep = max(1, len(bracket_rungs[rung]) // self.eta)
        kept_ids = set(tid for _, tid in sorted_vals[:n_keep])

        return trial_id not in kept_ids

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """P3.2.1 ε4 元学习：no-op（Hyperband 采样与 RandomSampler 等价，区别仅在 should_prune）。

        保留此方法仅为满足 Python 3.12+ @runtime_checkable Protocol
        的 isinstance 检查。
        """
        return None


# 注册为 Sampler + Pruner（同时具备采样与早停能力）
register_sampler("asha", ASHASampler)
register_sampler("hyperband", HyperbandSampler)
register_pruner("asha", ASHASampler)
register_pruner("hyperband", HyperbandSampler)


# ============================================================
# SP 核心：StudyManager（Ask-Tell + ExplorationTracker 桥接）
# ============================================================
class StudyManager:
    """Study 管理器（SP-1~3 + SP-5）。

    管理 Study 生命周期，桥接 ExplorationTracker 作为存储后端。
    实现 Ask-Tell 标准化搜索接口。
    """

    def __init__(self):
        self._studies: Dict[str, StudySpec] = {}
        self._trackers: Dict[str, ExplorationTracker] = {}  # study_id -> tracker
        self._pending_trials: Dict[str, TrialSpec] = {}  # trial_id -> TrialSpec (ask 后 tell 前)
        self._lock = threading.Lock()

    def create_study(
        self,
        name: str,
        direction: str = "maximize",
        search_space: Optional[SearchSpace] = None,
        sampler: str = "random",
        warm_start_from: Optional[str] = None,
        history_store: Optional["HistoryStore"] = None,
        seed: Optional[int] = None,
    ) -> str:
        """创建 Study（SP-1；P3.2.4 扩展 warm-start；P3-1 扩展 seed）。

        Args:
            name: Study 名称
            direction: 优化方向，"maximize" 或 "minimize"
            search_space: 搜索空间（None 时使用空 SearchSpace）
            sampler: 采样器名（注册到 SP Sampler 注册表）
            warm_start_from: P3.2.4 新增——源数据集名（如 "UT_HAR_data"），
                用于从该数据集历史 warm-start 当前 study。None 表示不 warm-start。
            history_store: P3.2.4 新增——HistoryStore 实例，用于加载源数据集历史。
                与 warm_start_from 配合使用；若 warm_start_from 为 None 则忽略。
            seed: P3-1 新增——study 级种子。None 时使用全局随机（不推荐，可能被
                set_seed 重置）；指定时每个 trial 的种子 = seed + trial_index，
                确保可复现且 trial 间参数有多样性。

        Returns:
            study_id
        """
        # 修复（5.12）：create_study 入口 INFO 日志
        n_params = len(search_space.parameters) if search_space else 0
        logger.info(
            "SP create_study: name=%s, direction=%s, sampler=%s, n_params=%d, "
            "warm_start_from=%s, seed=%s",
            name, direction, sampler, n_params, warm_start_from, seed,
        )
        study_id = f"study_{uuid.uuid4().hex[:8]}"
        study = StudySpec(
            study_id=study_id, name=name, direction=direction,
            search_space=search_space or SearchSpace(),
            sampler=sampler, created_at=datetime.now().isoformat(),
            seed=seed,
        )
        with self._lock:
            self._studies[study_id] = study
            self._trackers[study_id] = ExplorationTracker()

        # P3.2.4: warm-start 注入（延迟导入 MetaLearner 避免循环）
        if warm_start_from and history_store is not None:
            from .automl.meta_learner import MetaLearner
            meta_learner = MetaLearner(self, history_store)
            meta_learner.warm_start(study_id, warm_start_from)

        # 修复（5.12）：create_study 出口 INFO 日志
        logger.info(
            "SP create_study done: study_id=%s, name=%s",
            study_id, name,
        )
        return study_id

    def get_study(self, study_id: str) -> Optional[StudySpec]:
        return self._studies.get(study_id)

    def list_studies(self) -> List[StudySpec]:
        return list(self._studies.values())

    def stop_study(self, study_id: str) -> None:
        if study_id in self._studies:
            self._studies[study_id].status = "stopped"
            self._studies[study_id].completed_at = datetime.now().isoformat()

    # ---- SP-2: Ask-Tell ----

    def ask(self, study_id: str) -> TrialSpec:
        """请求下一组参数（SP-2 Ask）。"""
        # 修复（5.12）：ask 入口 INFO 日志
        logger.info("SP ask: study_id=%s", study_id)
        study = self._studies.get(study_id)
        if study is None:
            raise KeyError(f"Study '{study_id}' not found")
        if study.status != "running":
            raise RuntimeError(f"Study '{study_id}' is not running (status={study.status})")

        tracker = self._trackers[study_id]
        sampler_cls = get_sampler(study.sampler) or RandomSampler
        # P0.4：改用 ExplorationTracker.get_history 公共 API，不直接访问 tracker.history
        history = tracker.get_history()
        # P3-1：每个 trial 使用 study_seed + trial_index 作为独立种子，
        # 彻底消除 sampler 对全局 random 模块的依赖。
        # 相同 study_seed → 相同参数序列（可复现）；不同 study_seed → 不同参数。
        trial_index = len(history)
        study_seed = study.seed
        sampler_seed = (study_seed + trial_index) if study_seed is not None else None
        sampler = sampler_cls(seed=sampler_seed)
        params = sampler.sample(study.search_space, history)

        trial_id = f"trial_{uuid.uuid4().hex[:8]}"
        trial = TrialSpec(
            trial_id=trial_id, study_id=study_id, params=params,
            datetime_start=datetime.now().isoformat(),
        )
        with self._lock:
            self._pending_trials[trial_id] = trial
            # 也记录到 tracker（status=pending）
            tracker.add_trial(strategy=params, result=None, trial_id=trial_id)

        # 修复（5.4）：ML_SEARCH_COVERAGE_RATIO OTel 埋点（搜索空间覆盖率）
        # 计算方式：已采样 trial 数 / (n_params * 10) 的归一化比率（启发式）
        # OTel 未初始化时 no-op，可无脑调
        try:
            n_params = len(study.search_space.parameters)
            n_trials = len(history)
            # 覆盖率：粗略估计——已采样 trial 数占预期采样预算的比例
            # 预期采样预算 = max(n_params * 10, 20)
            expected_budget = max(n_params * 10, 20) if n_params > 0 else 20
            coverage_ratio = min(1.0, n_trials / expected_budget)
            record_training_metric(
                ML_SEARCH_COVERAGE_RATIO,
                value=float(coverage_ratio),
                stage="hpo",
            )
        except Exception:
            pass

        # 修复（5.12）：ask 出口 INFO 日志
        logger.info(
            "SP ask done: study_id=%s, trial_id=%s, params=%s",
            study_id, trial_id, params,
        )
        return trial

    def tell(
        self,
        trial_id: str,
        value: float,
        intermediate_values: Optional[Dict[int, float]] = None,
        state: str = "completed",
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """报告试验结果（SP-2 Tell）。"""
        # 修复（5.12）：tell 入口 INFO 日志
        logger.info(
            "SP tell: trial_id=%s, value=%s, state=%s",
            trial_id, value, state,
        )
        trial = self._pending_trials.get(trial_id)
        if trial is None:
            raise KeyError(f"Trial '{trial_id}' not found (not asked or already told)")

        tracker = self._trackers[trial.study_id]
        result = {"value": value}
        if intermediate_values:
            result["intermediate_values"] = intermediate_values

        # P0.4：改用 ExplorationTracker.update_trial 公共 API，
        # 消除 with tracker._lock + tracker.history 直接改写的封装破裂
        tracker.update_trial(
            trial_id=trial_id,
            result=result,
            status=state,
            feedback=feedback,
        )

        # 从 pending 移除
        with self._lock:
            del self._pending_trials[trial_id]

        # 修复（5.12）：tell 出口 INFO 日志
        logger.info(
            "SP tell done: trial_id=%s, study_id=%s, state=%s, value=%s",
            trial_id, trial.study_id, state, value,
        )

    def get_trial(self, trial_id: str) -> Optional[TrialResult]:
        """查询 Trial 结果（SP-2）。"""
        for study_id, tracker in self._trackers.items():
            entry = tracker.get_trial(trial_id)
            if entry:
                result_val = 0.0
                if entry.get("result") and "value" in entry["result"]:
                    result_val = entry["result"]["value"]
                return TrialResult(
                    trial_id=trial_id, study_id=study_id,
                    params=entry.get("strategy", {}),
                    value=result_val,
                    intermediate_values=entry.get("result", {}).get("intermediate_values", {}),
                    state=entry.get("status", "completed"),
                    datetime_complete=entry.get("timestamp", ""),
                    feedback=entry.get("feedback"),
                )
        return None

    def list_trials(self, study_id: str) -> List[TrialResult]:
        """列出 Study 的全部 Trial（SP-2）。"""
        tracker = self._trackers.get(study_id)
        if tracker is None:
            return []
        # P0.4：改用 ExplorationTracker.get_history 公共 API，不直接访问 tracker.history
        history = tracker.get_history()
        results = []
        for entry in history:
            result_val = 0.0
            if entry.get("result") and "value" in entry["result"]:
                result_val = entry["result"]["value"]
            # P2.5: 读取 intermediate_values（与 get_trial 保持一致）
            iv = entry.get("result", {}).get("intermediate_values", {})
            results.append(TrialResult(
                trial_id=entry["trial_id"], study_id=study_id,
                params=entry.get("strategy", {}),
                value=result_val,
                intermediate_values=iv,
                state=entry.get("status", "completed"),
                datetime_complete=entry.get("timestamp", ""),
                feedback=entry.get("feedback"),
            ))
        return results

    def best_trial(self, study_id: str) -> Optional[TrialResult]:
        """获取最优 Trial（SP-2）。"""
        # 修复（5.12）：best_trial 入口 INFO 日志
        logger.info("SP best_trial: study_id=%s", study_id)
        study = self._studies.get(study_id)
        if study is None:
            logger.info("SP best_trial done: study_id=%s not found", study_id)
            return None
        tracker = self._trackers[study_id]
        mode = "max" if study.direction == "maximize" else "min"
        metric = "value"
        best = tracker.best_trial(metric=metric, mode=mode)
        if best is None:
            logger.info(
                "SP best_trial done: study_id=%s, no completed trials yet",
                study_id,
            )
            return None
        result = self.get_trial(best["trial_id"])
        # 修复（5.12）：best_trial 出口 INFO 日志
        logger.info(
            "SP best_trial done: study_id=%s, best_trial_id=%s, value=%s",
            study_id,
            best["trial_id"],
            result.value if result else "N/A",
        )
        return result


# ============================================================
# 全局 StudyManager 单例
# ============================================================
_study_manager: Optional[StudyManager] = None


def get_study_manager() -> StudyManager:
    """获取全局 StudyManager 单例。"""
    global _study_manager
    if _study_manager is None:
        _study_manager = StudyManager()
    return _study_manager


__all__ = [
    "ParameterSpec", "SearchSpace", "StudySpec",
    "TrialSpec", "TrialResult",
    "StudyManager", "get_study_manager",
    "Sampler",  # P0.5：Sampler Protocol
    "register_sampler", "get_sampler", "list_samplers",
    "RandomSampler", "GridSampler",
    "Pruner",  # P2.1：Pruner Protocol（ε5 Multi-fidelity 早停）
    "register_pruner", "get_pruner", "list_pruners",
    "ASHASampler",  # P2.2：ε5 Multi-fidelity 内置实现
    "HyperbandSampler",
]
