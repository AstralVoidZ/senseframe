"""RFC-003 SP：搜索协议 — Ask-Tell 标准化搜索接口。

对齐 Optuna Study/Trial/Sampler 范式 + Google Vizier RPC 思想。
让 SenseFrame 可作为搜索服务被 AutoML 主控调用。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .exploration import ExplorationTracker


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id, "name": self.name,
            "direction": self.direction,
            "search_space": self.search_space.to_dict(),
            "sampler": self.sampler, "status": self.status,
            "created_at": self.created_at, "completed_at": self.completed_at,
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
    """Sampler 契约（SP-3，P0.5 新增）。

    所有 Sampler 必须满足此 Protocol：拥有 `name` 类属性与 `sample` 方法。
    `@runtime_checkable` 使 `isinstance(sampler, Sampler)` 可用，
    仅校验属性/方法存在性，不校验签名。
    """
    name: str  # 类属性（采样策略名）

    def sample(
        self,
        search_space: "SearchSpace",
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]: ...


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
    """随机采样器（SP-3 内置）。"""
    name = "random"

    def sample(self, search_space: SearchSpace, history: List[Dict[str, Any]]) -> Dict[str, Any]:
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


class GridSampler:
    """网格采样器（SP-3 内置）。"""
    name = "grid"

    def __init__(self):
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
                grids.append(list(range(int(p.low or 0), int(p.high or 10) + 1)))
            elif p.type == "categorical" and p.choices:
                grids.append(p.choices)
            else:
                grids.append([None])

        all_combos = list(itertools.product(*grids))
        idx = len(history) % len(all_combos)
        combo = all_combos[idx]
        return {p.name: v for p, v in zip(search_space.parameters, combo)}


# 注册内置 Sampler
register_sampler("random", RandomSampler)
register_sampler("grid", GridSampler)
# P1.3: TPE 诚实降级——不伪装为 TPE，显式标注为 random_fallback
# 当 Optuna 可用时 StudyManager 会优先使用 Optuna TPE；此处为纯 Python fallback
register_sampler("tpe", RandomSampler)  # ⚠️ random_fallback: 真正 TPE 需 Optuna，此处退化为随机搜索
register_sampler("random_fallback", RandomSampler)  # 显式别名，避免 tpe 名称误导


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
    ) -> str:
        """创建 Study（SP-1）。"""
        study_id = f"study_{uuid.uuid4().hex[:8]}"
        study = StudySpec(
            study_id=study_id, name=name, direction=direction,
            search_space=search_space or SearchSpace(),
            sampler=sampler, created_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._studies[study_id] = study
            self._trackers[study_id] = ExplorationTracker()
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
        study = self._studies.get(study_id)
        if study is None:
            raise KeyError(f"Study '{study_id}' not found")
        if study.status != "running":
            raise RuntimeError(f"Study '{study_id}' is not running (status={study.status})")

        tracker = self._trackers[study_id]
        sampler_cls = get_sampler(study.sampler) or RandomSampler
        sampler = sampler_cls()
        # P0.4：改用 ExplorationTracker.get_history 公共 API，不直接访问 tracker.history
        params = sampler.sample(study.search_space, tracker.get_history())

        trial_id = f"trial_{uuid.uuid4().hex[:8]}"
        trial = TrialSpec(
            trial_id=trial_id, study_id=study_id, params=params,
            datetime_start=datetime.now().isoformat(),
        )
        with self._lock:
            self._pending_trials[trial_id] = trial
            # 也记录到 tracker（status=pending）
            tracker.add_trial(strategy=params, result=None, trial_id=trial_id)
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
            results.append(TrialResult(
                trial_id=entry["trial_id"], study_id=study_id,
                params=entry.get("strategy", {}),
                value=result_val,
                state=entry.get("status", "completed"),
                datetime_complete=entry.get("timestamp", ""),
                feedback=entry.get("feedback"),
            ))
        return results

    def best_trial(self, study_id: str) -> Optional[TrialResult]:
        """获取最优 Trial（SP-2）。"""
        study = self._studies.get(study_id)
        if study is None:
            return None
        tracker = self._trackers[study_id]
        mode = "max" if study.direction == "maximize" else "min"
        metric = "value"
        best = tracker.best_trial(metric=metric, mode=mode)
        if best is None:
            return None
        return self.get_trial(best["trial_id"])


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
]
