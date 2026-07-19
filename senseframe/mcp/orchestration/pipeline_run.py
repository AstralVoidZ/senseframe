"""PipelineRun 状态机 + PipelineRunStore（设计文档 0.6 节）。

设计文档 0.6 节定义 5 状态 + 7 转换：

5 状态：
- Pending — 已创建未启动
- Running — 正在执行
- Paused  — 暂停（可恢复）
- Succeeded — 成功完成
- Failed  — 失败

7 转换：
- start:    Pending → Running
- complete: Running → Succeeded
- fail:     Running → Failed
- pause:    Running → Paused
- resume:   Paused  → Running
- retry:    Failed  → Running
- skip:     Pending → Failed（标记失败但不执行）

幂等短路（参考 pipeflow fsm/machine.py 的 _IDEMPOTENT）：
- complete on Succeeded → no-op（返回当前状态）
- fail on Failed → no-op
- pause on Paused → no-op

FSM 实现选择：纯 Python dict，零外部依赖（不引入 `transitions` 库）。

存储选择：内存 dict + threading.RLock（不引入 SQLite）。
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from senseframe.mcp.errors import IllegalTransition, PipelineNotFound
from senseframe.mcp.models.pipeline_run import (
    PipelineRun,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_PENDING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)

__all__ = [
    "PipelineRunStore",
    "get_default_store",
    "set_default_store",
    "trigger",
    "VALID_ACTIONS",
    "VALID_TRANSITIONS",
    "IDEMPOTENT_ACTIONS",
]


# ============================================================
# FSM：5 状态 + 7 转换（纯 Python dict，零外部依赖）
# ============================================================

# 合法状态转换：(current_state, action) → new_state
# 不含幂等短路（幂等动作在 _IDEMPOTENT_ACTIONS 中定义）。
VALID_TRANSITIONS: dict[tuple[str, str], str] = {
    # Pending →
    (STATE_PENDING, "start"): STATE_RUNNING,
    (STATE_PENDING, "skip"): STATE_FAILED,
    # Running →
    (STATE_RUNNING, "complete"): STATE_SUCCEEDED,
    (STATE_RUNNING, "fail"): STATE_FAILED,
    (STATE_RUNNING, "pause"): STATE_PAUSED,
    # Paused →
    (STATE_PAUSED, "resume"): STATE_RUNNING,
    # Failed →
    (STATE_FAILED, "retry"): STATE_RUNNING,
}

# 幂等短路：(state, action) → no-op（返回当前状态）
# 设计文档 0.6 节：complete on Succeeded / fail on Failed / pause on Paused
IDEMPOTENT_ACTIONS: dict[str, frozenset[str]] = {
    STATE_SUCCEEDED: frozenset({"complete"}),
    STATE_FAILED: frozenset({"fail"}),
    STATE_PAUSED: frozenset({"pause"}),
}

# 合法动作集合
VALID_ACTIONS: frozenset[str] = frozenset(
    {transition[1] for transition in VALID_TRANSITIONS}
) | frozenset(
    action for actions in IDEMPOTENT_ACTIONS.values() for action in actions
)


def trigger(current_state: str, action: str) -> str:
    """状态转换：返回新状态。

    Args:
        current_state: 当前状态字符串。
        action: 动作名（start/complete/fail/pause/resume/retry/skip）。

    Returns:
        新状态字符串。

    Raises:
        IllegalTransition: 当前状态不允许该动作（非法转换）。
    """
    # 幂等短路：当前状态已达成该动作的目标状态
    if action in IDEMPOTENT_ACTIONS.get(current_state, frozenset()):
        return current_state
    # 合法转换
    new_state = VALID_TRANSITIONS.get((current_state, action))
    if new_state is None:
        raise IllegalTransition(
            f"action {action!r} not allowed in state {current_state!r}"
        )
    return new_state


# ============================================================
# PipelineRunStore：内存存储 + threading.RLock
# ============================================================


class PipelineRunStore:
    """PipelineRun 内存存储 + 状态机入口。

    线程安全：用 ``threading.RLock`` 保护内部 dict 读写。
    存储介质：内存 dict（``dict[str, PipelineRun]``），不引入 SQLite。

    所有状态变更通过 ``advance`` 集中入口，避免分散的状态修改。
    """

    def __init__(self) -> None:
        # 用 RLock 避免同线程递归死锁
        self._lock = threading.RLock()
        # run_id → PipelineRun
        self._runs: dict[str, PipelineRun] = {}

    # ---- 创建 / 查询 ----

    def create(self, config: dict, stages: list[str]) -> PipelineRun:
        """创建新 PipelineRun。

        Args:
            config: ExperimentConfig.model_dump() 的 dict。
            stages: stage 名列表。

        Returns:
            新创建的 PipelineRun（state=Pending）。
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be dict, got {type(config).__name__}")
        if not isinstance(stages, list) or not all(
            isinstance(s, str) for s in stages
        ):
            raise TypeError("stages must be list[str]")
        run_id = uuid4().hex
        now = _now_iso()
        run = PipelineRun(
            run_id=run_id,
            config=dict(config),  # 防御性 copy
            stages=list(stages),
            state=STATE_PENDING,
            created_at=now,
            updated_at=now,
            completed_stages=[],
            failed_stage=None,
            error_message=None,
            trial_id=None,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> PipelineRun:
        """查询 run 状态。

        Args:
            run_id: PipelineRun ID。

        Returns:
            PipelineRun 实例。

        Raises:
            PipelineNotFound: run_id 不存在。
        """
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise PipelineNotFound(f"PipelineRun {run_id!r} not found")
        return run

    # ---- 状态转移 ----

    def advance(
        self,
        run_id: str,
        action: str,
        *,
        completed_stage: str | None = None,
        failed_stage: str | None = None,
        error_message: str | None = None,
        trial_id: str | None = None,
    ) -> PipelineRun:
        """状态转移：集中入口，幂等 + 非法转换短路。

        幂等短路（参考 pipeflow fsm/machine.py 的 _IDEMPOTENT）：
        - complete on Succeeded → no-op（返回当前 run）
        - fail on Failed → no-op
        - pause on Paused → no-op

        Args:
            run_id: PipelineRun ID。
            action: 动作名（start/complete/fail/pause/resume/retry/skip）。
            completed_stage: complete 时追加到 completed_stages。
            failed_stage: fail/skip 时记录失败的 stage 名。
            error_message: fail/skip 时的失败原因。
            trial_id: 可选的 trial 关联（start/resume/retry 时设置）。

        Returns:
            新的 PipelineRun 实例（旧实例保留 immutable 历史快照）。

        Raises:
            PipelineNotFound: run_id 不存在。
            IllegalTransition: 当前状态不允许该动作。
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise PipelineNotFound(f"PipelineRun {run_id!r} not found")

            # 幂等短路：当前状态已达成目标
            if action in IDEMPOTENT_ACTIONS.get(run.state, frozenset()):
                return run

            new_state = trigger(run.state, action)
            now = _now_iso()

            # 更新 completed_stages / failed_stage / error_message
            new_completed = list(run.completed_stages)
            if action == "complete" and completed_stage is not None:
                if completed_stage not in new_completed:
                    new_completed.append(completed_stage)

            new_failed_stage = run.failed_stage
            new_error_message = run.error_message
            if action in ("fail", "skip"):
                new_failed_stage = failed_stage
                new_error_message = error_message
            elif action == "retry":
                # retry 清空失败标记
                new_failed_stage = None
                new_error_message = None

            new_trial_id = run.trial_id
            if trial_id is not None:
                new_trial_id = trial_id

            new_run = PipelineRun(
                run_id=run.run_id,
                config=run.config,
                stages=run.stages,
                state=new_state,
                created_at=run.created_at,
                updated_at=now,
                completed_stages=new_completed,
                failed_stage=new_failed_stage,
                error_message=new_error_message,
                trial_id=new_trial_id,
            )
            self._runs[run_id] = new_run
            return new_run

    # ---- 列表（cursor 分页） ----

    def list_runs(
        self,
        cursor: str | None = None,
        limit: int = 50,
        filter_dict: dict[str, Any] | None = None,
    ) -> tuple[list[PipelineRun], int, bool]:
        """分页查询所有 run。

        分页基于 run_id 的字典序（uuid4 hex，单调无碰撞）。
        filter_dict 当前支持 ``{"state": "Running"}`` 简单等值过滤。

        Args:
            cursor: 不透明 cursor（来自上一次 list_runs 的 next_cursor），None 表示首次。
            limit: 钳制后的页大小。
            filter_dict: 过滤字典。

        Returns:
            ``(items, total_count, has_more)`` 三元组。
            ``items``: 已截断到 limit 的 PipelineRun 列表。
            ``total_count``: 满足 filter 的总数。
            ``has_more``: 是否还有更多行。
        """
        # 延迟导入：cursor 模块在分页层，本模块已在 orchestration 层
        from senseframe.mcp.pagination.cursor import assert_fingerprint_matches
        from senseframe.mcp.pagination.page import clamp_limit

        clamped_limit = clamp_limit(limit)
        last_id = assert_fingerprint_matches(cursor, filter_dict)

        with self._lock:
            # 按 run_id 字典序排序
            sorted_runs = sorted(self._runs.values(), key=lambda r: r.run_id)
            # 应用 filter
            filtered = [
                r for r in sorted_runs if _matches_filter(r, filter_dict)
            ]
            total_count = len(filtered)
            # 应用 cursor（last_id 之后）
            if last_id is not None:
                filtered = [r for r in filtered if r.run_id > last_id]
            # limit+1 技巧：检查是否还有更多
            peek = filtered[: clamped_limit + 1]
            has_more = len(peek) > clamped_limit
            items = peek[:clamped_limit]
        return items, total_count, has_more

    # ---- 删除 ----

    def delete(self, run_id: str) -> None:
        """删除 run。

        Args:
            run_id: PipelineRun ID。

        Raises:
            PipelineNotFound: run_id 不存在。
        """
        with self._lock:
            if run_id not in self._runs:
                raise PipelineNotFound(f"PipelineRun {run_id!r} not found")
            del self._runs[run_id]


# ============================================================
# 辅助函数
# ============================================================


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(UTC).isoformat()


def _matches_filter(run: PipelineRun, filter_dict: dict[str, Any] | None) -> bool:
    """简单等值过滤：filter_dict 中的所有 (k, v) 必须在 run 上匹配。

    支持的 filter key：
    - state: run.state == v
    - trial_id: run.trial_id == v
    """
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        if key == "state" and run.state != value:
            return False
        elif key == "trial_id" and run.trial_id != value:
            return False
    return True


# ============================================================
# 进程级单例（tools/ 与 resources/ 共享同一个 Store）
# ============================================================
# 用 threading.RLock 保护单例创建，避免并发首调用产生两个实例。
# 测试可通过 set_default_store(None) 重置，或注入 mock。
_DEFAULT_STORE: PipelineRunStore | None = None
_DEFAULT_STORE_LOCK = threading.RLock()


def get_default_store() -> PipelineRunStore:
    """返回进程级 PipelineRunStore 单例（惰性初始化）。

    tools/pipeline.py 与 resources/pipeline.py 共享此单例，
    确保 tool 创建的 run 对 resource 端点可见。
    """
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = PipelineRunStore()
        return _DEFAULT_STORE


def set_default_store(store: PipelineRunStore | None) -> None:
    """注入/重置进程级 PipelineRunStore 单例。

    测试用：注入 mock store 或重置为 None 触发重新初始化。
    """
    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = store
