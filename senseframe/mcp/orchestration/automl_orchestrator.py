"""AutoMLOrchestrator：MCP L4 SP 协议实现载体（设计文档 0.7.3 节）。

协调 HPO → NAS → AutoAugment 流水线，通过 Ask-Tell 协议暴露给 Agent。
每个子搜索是独立的 Study（由 StudyManager 管理），Orchestrator 负责
串联 + 经验传递（如 NAS 最佳架构自动注入 HPO Study 的 module_factory）。

设计原则（advisory 优于 enforced）：
- Orchestrator 不直接执行搜索，仅管理状态机 + 提供推进原语
- Agent 持有循环控制权，决定何时启动/完成每个子搜索
- _transitions 提示下一步动作，但不强制

状态机（5 状态 + 6 转换）：
- Pending — 已创建未启动
- Running — 正在执行某个子搜索
- Paused  — 暂停（可恢复）
- Succeeded — 所有 stage 完成
- Failed  — 失败

转换：
- start:    Pending → Running
- complete: Running → Running（当前 stage 完成，推进到下一个 stage）或 → Succeeded（最后一个 stage）
- fail:     Running → Failed
- pause:    Running → Paused
- resume:   Paused  → Running
- retry:    Failed  → Running

存储：内存 dict + threading.RLock（不引入 SQLite）。
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from senseframe.mcp.errors import IllegalTransition
from senseframe.mcp.orchestration.study_manager import (
    get_default_manager as _get_default_study_manager,
)

__all__ = [
    "AutoMLOrchestrator",
    "AutoMLPipeline",
    "AUTOML_STATE_PENDING",
    "AUTOML_STATE_RUNNING",
    "AUTOML_STATE_PAUSED",
    "AUTOML_STATE_SUCCEEDED",
    "AUTOML_STATE_FAILED",
    "AUTOML_VALID_STAGES",
    "AUTOML_VALID_ACTIONS",
    "AUTOML_VALID_TRANSITIONS",
    "get_default_orchestrator",
    "set_default_orchestrator",
]

# ============================================================
# 状态常量
# ============================================================
AUTOML_STATE_PENDING = "Pending"
AUTOML_STATE_RUNNING = "Running"
AUTOML_STATE_PAUSED = "Paused"
AUTOML_STATE_SUCCEEDED = "Succeeded"
AUTOML_STATE_FAILED = "Failed"

# 合法 stage 类型（按设计文档 0.7.3 节）
AUTOML_VALID_STAGES: frozenset[str] = frozenset({"nas", "hpo", "autoaugment"})

# 合法状态转换：(current_state, action) → new_state
# complete 是特殊动作：在最后一个 stage 完成时 → Succeeded，否则 → Running（推进到下一 stage）
AUTOML_VALID_TRANSITIONS: dict[tuple[str, str], str] = {
    (AUTOML_STATE_PENDING, "start"): AUTOML_STATE_RUNNING,
    (AUTOML_STATE_RUNNING, "complete"): AUTOML_STATE_RUNNING,  # 推进 stage（实际可能 → Succeeded）
    (AUTOML_STATE_RUNNING, "fail"): AUTOML_STATE_FAILED,
    (AUTOML_STATE_RUNNING, "pause"): AUTOML_STATE_PAUSED,
    (AUTOML_STATE_PAUSED, "resume"): AUTOML_STATE_RUNNING,
    (AUTOML_STATE_FAILED, "retry"): AUTOML_STATE_RUNNING,
}

# 幂等短路：(state, action) → no-op
AUTOML_IDEMPOTENT_ACTIONS: dict[str, frozenset[str]] = {
    AUTOML_STATE_SUCCEEDED: frozenset({"complete"}),
    AUTOML_STATE_FAILED: frozenset({"fail"}),
    AUTOML_STATE_PAUSED: frozenset({"pause"}),
}

# 合法动作集合
AUTOML_VALID_ACTIONS: frozenset[str] = frozenset(
    {t[1] for t in AUTOML_VALID_TRANSITIONS}
) | frozenset(
    a for acts in AUTOML_IDEMPOTENT_ACTIONS.values() for a in acts
)


# ============================================================
# AutoMLPipeline 域对象
# ============================================================
class AutoMLPipeline:
    """AutoML 流水线域对象（可变，受 Orchestrator RLock 保护）。

    Attributes:
        pipeline_id: UUID4 hex 字符串。
        config: ExperimentConfig.model_dump() 的 dict 快照。
        stages: stage 名列表（如 ["nas", "hpo", "autoaugment"]）。
        state: 当前状态字符串。
        current_stage_index: 当前执行的 stage 索引（0-based，-1 表示未启动）。
        study_ids: 每个 stage 对应的 Study ID 列表（与 stages 同长，None 表示未创建）。
        created_at: ISO8601 创建时间。
        updated_at: ISO8601 最后更新时间。
        completed_stages: 已完成的 stage 名列表。
        failed_stage: 失败时记录的 stage 名（None 表示未失败）。
        error_message: 失败原因（None 表示未失败）。
    """

    def __init__(
        self,
        pipeline_id: str,
        config: dict[str, Any],
        stages: list[str],
        created_at: str,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.config = config
        self.stages: list[str] = list(stages)
        self.state: str = AUTOML_STATE_PENDING
        self.current_stage_index: int = -1
        self.study_ids: list[str | None] = [None] * len(self.stages)
        self.created_at = created_at
        self.updated_at = created_at
        self.completed_stages: list[str] = []
        self.failed_stage: str | None = None
        self.error_message: str | None = None


# ============================================================
# AutoMLOrchestrator
# ============================================================
class AutoMLOrchestrator:
    """AutoML 流水线编排器（MCP L4 SP 协议实现载体）。

    协调 HPO → NAS → AutoAugment 流水线，通过 Ask-Tell 协议暴露给 Agent。
    每个子搜索是独立的 Study，Orchestrator 负责串联 + 经验传递。
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, AutoMLPipeline] = {}
        self._lock = threading.RLock()

    # ---- CRUD ----

    def create_pipeline(
        self,
        config: dict[str, Any],
        stages: list[str],
    ) -> str:
        """创建 AutoML 流水线（返回 pipeline_id）。

        Args:
            config: ExperimentConfig.model_dump() 的 dict 快照。
            stages: stage 名列表，元素必须是 "nas" / "hpo" / "autoaugment"。
                顺序决定执行流：
                - ["nas", "hpo"]：先搜索架构，再基于最佳架构搜索超参
                - ["hpo", "autoaugment"]：先搜索超参，再搜索增强策略
                - ["nas", "hpo", "autoaugment"]：完整流水线

        Returns:
            pipeline_id（UUID4 hex 字符串）。

        Raises:
            ValueError: stages 为空或含非法 stage 名。
        """
        if not stages:
            raise ValueError("stages must be a non-empty list")
        invalid = [s for s in stages if s not in AUTOML_VALID_STAGES]
        if invalid:
            raise ValueError(
                f"invalid stage names: {invalid}; "
                f"valid stages: {sorted(AUTOML_VALID_STAGES)}"
            )
        pipeline_id = f"automl_{uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        with self._lock:
            pipeline = AutoMLPipeline(
                pipeline_id=pipeline_id,
                config=dict(config),
                stages=list(stages),
                created_at=now,
            )
            self._pipelines[pipeline_id] = pipeline
        return pipeline_id

    def get(self, pipeline_id: str) -> AutoMLPipeline:
        """查询流水线状态。

        Args:
            pipeline_id: Pipeline ID。

        Returns:
            AutoMLPipeline 实例。

        Raises:
            KeyError: pipeline_id 不存在。
        """
        with self._lock:
            pipeline = self._pipelines.get(pipeline_id)
            if pipeline is None:
                raise KeyError(f"AutoML pipeline '{pipeline_id}' not found")
            return pipeline

    def list_pipelines(self) -> list[AutoMLPipeline]:
        """列出所有流水线。"""
        with self._lock:
            return list(self._pipelines.values())

    def delete(self, pipeline_id: str) -> None:
        """删除流水线。

        Args:
            pipeline_id: Pipeline ID。

        Raises:
            KeyError: pipeline_id 不存在。
        """
        with self._lock:
            if pipeline_id not in self._pipelines:
                raise KeyError(f"AutoML pipeline '{pipeline_id}' not found")
            del self._pipelines[pipeline_id]

    # ---- 状态机 ----

    def advance(
        self,
        pipeline_id: str,
        action: str,
        study_id: str | None = None,
        error_message: str | None = None,
    ) -> AutoMLPipeline:
        """推进流水线状态（单一状态变更入口，幂等）。

        Args:
            pipeline_id: Pipeline ID。
            action: 动作名（start/complete/fail/pause/resume/retry）。
            study_id: complete 时可附带当前 stage 对应的 Study ID（记录到 study_ids）。
            error_message: fail 时可附带失败原因。

        Returns:
            更新后的 AutoMLPipeline 实例。

        Raises:
            KeyError: pipeline_id 不存在。
            IllegalTransition: (state, action) 不是合法转换。
        """
        with self._lock:
            pipeline = self._pipelines.get(pipeline_id)
            if pipeline is None:
                raise KeyError(f"AutoML pipeline '{pipeline_id}' not found")
            current_state = pipeline.state

            # 幂等短路
            idempotent = AUTOML_IDEMPOTENT_ACTIONS.get(current_state, frozenset())
            if action in idempotent:
                return pipeline

            # complete 特殊处理：最后一个 stage 完成时 → Succeeded
            if action == "complete" and current_state == AUTOML_STATE_RUNNING:
                # 记录 study_id（若提供）
                if study_id is not None and 0 <= pipeline.current_stage_index < len(pipeline.stages):
                    pipeline.study_ids[pipeline.current_stage_index] = study_id
                # 标记当前 stage 完成
                current_stage = pipeline.stages[pipeline.current_stage_index]
                if current_stage not in pipeline.completed_stages:
                    pipeline.completed_stages.append(current_stage)
                # 推进到下一个 stage
                pipeline.current_stage_index += 1
                if pipeline.current_stage_index >= len(pipeline.stages):
                    pipeline.state = AUTOML_STATE_SUCCEEDED
                else:
                    pipeline.state = AUTOML_STATE_RUNNING
                pipeline.updated_at = datetime.now(UTC).isoformat()
                return pipeline

            # start 特殊处理：进入第一个 stage
            if action == "start" and current_state == AUTOML_STATE_PENDING:
                pipeline.current_stage_index = 0
                pipeline.state = AUTOML_STATE_RUNNING
                pipeline.updated_at = datetime.now(UTC).isoformat()
                return pipeline

            # retry 特殊处理：从 Failed 回到 Running，清理 failed_stage
            if action == "retry" and current_state == AUTOML_STATE_FAILED:
                pipeline.state = AUTOML_STATE_RUNNING
                pipeline.failed_stage = None
                pipeline.error_message = None
                pipeline.updated_at = datetime.now(UTC).isoformat()
                return pipeline

            # fail 特殊处理：记录 failed_stage
            if action == "fail" and current_state == AUTOML_STATE_RUNNING:
                if 0 <= pipeline.current_stage_index < len(pipeline.stages):
                    pipeline.failed_stage = pipeline.stages[pipeline.current_stage_index]
                pipeline.error_message = error_message
                pipeline.state = AUTOML_STATE_FAILED
                pipeline.updated_at = datetime.now(UTC).isoformat()
                return pipeline

            # 通用转换（pause/resume）
            key = (current_state, action)
            if key in AUTOML_VALID_TRANSITIONS:
                pipeline.state = AUTOML_VALID_TRANSITIONS[key]
                pipeline.updated_at = datetime.now(UTC).isoformat()
                return pipeline

            raise IllegalTransition(
                f"illegal transition: state={current_state} action={action} "
                f"pipeline_id={pipeline_id}"
            )

    def set_study_for_stage(
        self,
        pipeline_id: str,
        stage_index: int,
        study_id: str,
    ) -> None:
        """为指定 stage 设置 Study ID（Agent 创建 Study 后调用）。

        Args:
            pipeline_id: Pipeline ID。
            stage_index: stage 索引（0-based）。
            study_id: 该 stage 对应的 Study ID。

        Raises:
            KeyError: pipeline_id 不存在。
            IndexError: stage_index 越界。
        """
        with self._lock:
            pipeline = self._pipelines.get(pipeline_id)
            if pipeline is None:
                raise KeyError(f"AutoML pipeline '{pipeline_id}' not found")
            if not 0 <= stage_index < len(pipeline.stages):
                raise IndexError(
                    f"stage_index {stage_index} out of range "
                    f"(stages len={len(pipeline.stages)})"
                )
            pipeline.study_ids[stage_index] = study_id
            pipeline.updated_at = datetime.now(UTC).isoformat()


# ============================================================
# 进程级单例
# ============================================================
_default_orchestrator: AutoMLOrchestrator | None = None
_default_lock = threading.RLock()


def get_default_orchestrator() -> AutoMLOrchestrator:
    """返回进程级 AutoMLOrchestrator 单例（惰性初始化）。"""
    global _default_orchestrator
    with _default_lock:
        if _default_orchestrator is None:
            _default_orchestrator = AutoMLOrchestrator()
        return _default_orchestrator


def set_default_orchestrator(
    orch: AutoMLOrchestrator | None,
) -> None:
    """注入/重置进程级 AutoMLOrchestrator 单例（测试用）。"""
    global _default_orchestrator
    with _default_lock:
        _default_orchestrator = orch
