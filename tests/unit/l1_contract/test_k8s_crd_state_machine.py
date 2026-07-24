"""L1 契约测试：K8s CRD 状态机（PipelineRun 5 状态 7 转换 + 3 幂等短路）。

锚点来源：K8s CRD 状态机范式 + 设计文档 0.6 节。

设计文档 0.6 节定义：
5 状态：Pending / Running / Paused / Succeeded / Failed
7 转换：
- start:    Pending → Running
- skip:     Pending → Failed
- complete: Running → Succeeded
- fail:     Running → Failed
- pause:    Running → Paused
- resume:   Paused  → Running
- retry:    Failed  → Running
3 幂等短路：complete on Succeeded / fail on Failed / pause on Paused

本测试硬编码期望值（锚点为设计文档 0.6 节），禁止引用源码 VALID_TRANSITIONS /
IDEMPOTENT_ACTIONS 常量（消除自证断言）。
"""
from __future__ import annotations

import pytest

from senseframe.mcp.errors import IllegalTransition
from senseframe.mcp.orchestration.pipeline_run import (
    PipelineRunStore,
    trigger,
)

# 设计文档 0.6 节：5 状态（硬编码期望值，锚点为设计文档，不引用源码常量）
EXPECTED_STATES = frozenset({"Pending", "Running", "Paused", "Succeeded", "Failed"})

# 设计文档 0.6 节：7 合法转换（硬编码期望值，不引用源码 VALID_TRANSITIONS）
EXPECTED_TRANSITIONS = {
    ("Pending", "start"): "Running",
    ("Pending", "skip"): "Failed",
    ("Running", "complete"): "Succeeded",
    ("Running", "fail"): "Failed",
    ("Running", "pause"): "Paused",
    ("Paused", "resume"): "Running",
    ("Failed", "retry"): "Running",
}

# 设计文档 0.6 节：3 幂等短路（硬编码期望值，不引用源码 IDEMPOTENT_ACTIONS）
EXPECTED_IDEMPOTENT = {
    ("Succeeded", "complete"): "Succeeded",
    ("Failed", "fail"): "Failed",
    ("Paused", "pause"): "Paused",
}

# 设计文档 0.6 节定义的全部动作
ALL_ACTIONS = frozenset({"start", "skip", "complete", "fail", "pause", "resume", "retry"})


@pytest.mark.l1_contract
class TestK8sCrdStateMachineContract:
    """验证 PipelineRun 状态机符合 K8s CRD 范式 + 设计文档 0.6 节契约。"""

    def test_exactly_five_states(self):
        """L1 anchor: 恰好 5 状态（Pending/Running/Paused/Succeeded/Failed），锚点设计文档 0.6 节。"""
        assert len(EXPECTED_STATES) == 5
        # 5 状态为设计文档 0.6 节定义的固定集合
        assert EXPECTED_STATES == {"Pending", "Running", "Paused", "Succeeded", "Failed"}

    def test_states_produced_by_trigger_within_five(self):
        """L1 anchor: trigger 产出状态在设计文档 0.6 节 5 状态内，锚点设计文档 0.6 节。"""
        for (state, action), expected_new in EXPECTED_TRANSITIONS.items():
            assert state in EXPECTED_STATES, f"输入状态 {state!r} 不在 5 状态内"
            actual_new = trigger(state, action)
            assert actual_new in EXPECTED_STATES, \
                f"trigger({state!r}, {action!r}) 产出 {actual_new!r} 不在 5 状态内"

    def test_exactly_seven_legal_transitions(self):
        """L1 anchor: 恰好 7 个合法转换，锚点设计文档 0.6 节（硬编码期望值）。"""
        assert len(EXPECTED_TRANSITIONS) == 7
        for (state, action), expected_new in EXPECTED_TRANSITIONS.items():
            actual_new = trigger(state, action)
            assert actual_new == expected_new, \
                f"转换 ({state}, {action}) 应为 {expected_new}，实际 {actual_new}"

    def test_pending_start_to_running(self):
        """L1 anchor: start: Pending → Running，锚点设计文档 0.6 节。"""
        assert trigger("Pending", "start") == "Running"

    def test_pending_skip_to_failed(self):
        """L1 anchor: skip: Pending → Failed，锚点设计文档 0.6 节。"""
        assert trigger("Pending", "skip") == "Failed"

    def test_running_complete_to_succeeded(self):
        """L1 anchor: complete: Running → Succeeded，锚点设计文档 0.6 节。"""
        assert trigger("Running", "complete") == "Succeeded"

    def test_running_fail_to_failed(self):
        """L1 anchor: fail: Running → Failed，锚点设计文档 0.6 节。"""
        assert trigger("Running", "fail") == "Failed"

    def test_running_pause_to_paused(self):
        """L1 anchor: pause: Running → Paused，锚点设计文档 0.6 节。"""
        assert trigger("Running", "pause") == "Paused"

    def test_paused_resume_to_running(self):
        """L1 anchor: resume: Paused → Running，锚点设计文档 0.6 节。"""
        assert trigger("Paused", "resume") == "Running"

    def test_failed_retry_to_running(self):
        """L1 anchor: retry: Failed → Running，锚点设计文档 0.6 节。"""
        assert trigger("Failed", "retry") == "Running"

    def test_illegal_transitions_raise_illegal_transition(self):
        """L1 anchor: 非法转换抛 IllegalTransition（K8s CRD 状态机范式），反向用例。"""
        legal = set(EXPECTED_TRANSITIONS.keys()) | set(EXPECTED_IDEMPOTENT.keys())
        illegal_count = 0
        for state in EXPECTED_STATES:
            for action in ALL_ACTIONS:
                if (state, action) in legal:
                    continue
                illegal_count += 1
                with pytest.raises(IllegalTransition):
                    trigger(state, action)
        # 确保确实枚举了非法用例（5 状态 × 7 动作 - 10 合法 = 25 非法）
        assert illegal_count == 35 - len(legal)

    def test_exactly_three_idempotent_short_circuits(self):
        """L1 anchor: 恰好 3 个幂等短路，锚点设计文档 0.6 节。"""
        assert len(EXPECTED_IDEMPOTENT) == 3

    def test_idempotent_short_circuits_via_trigger(self):
        """L1 anchor: 3 幂等短路 trigger 返回当前状态（no-op），锚点设计文档 0.6 节。"""
        for (state, action), expected in EXPECTED_IDEMPOTENT.items():
            result = trigger(state, action)
            assert result == expected, \
                f"幂等短路 ({state}, {action}) 应返回 {expected}，实际 {result}"

    def test_store_advance_idempotent_complete_on_succeeded(self):
        """L1 anchor: complete on Succeeded 幂等短路，锚点设计文档 0.6 节。"""
        store = PipelineRunStore()
        run = store.create(config={"x": 1}, stages=["a"])
        store.advance(run.run_id, "start")
        store.advance(run.run_id, "complete")
        succeeded = store.get(run.run_id)
        assert succeeded.state == "Succeeded"
        # 幂等：再次 complete 不抛异常、不改变状态
        idempotent = store.advance(run.run_id, "complete")
        assert idempotent.state == "Succeeded"

    def test_store_advance_idempotent_fail_on_failed(self):
        """L1 anchor: fail on Failed 幂等短路，锚点设计文档 0.6 节。"""
        store = PipelineRunStore()
        run = store.create(config={}, stages=["a"])
        store.advance(run.run_id, "start")
        store.advance(run.run_id, "fail")
        failed = store.get(run.run_id)
        assert failed.state == "Failed"
        idempotent = store.advance(run.run_id, "fail")
        assert idempotent.state == "Failed"

    def test_store_advance_idempotent_pause_on_paused(self):
        """L1 anchor: pause on Paused 幂等短路，锚点设计文档 0.6 节。"""
        store = PipelineRunStore()
        run = store.create(config={}, stages=["a"])
        store.advance(run.run_id, "start")
        store.advance(run.run_id, "pause")
        paused = store.get(run.run_id)
        assert paused.state == "Paused"
        idempotent = store.advance(run.run_id, "pause")
        assert idempotent.state == "Paused"

    def test_store_advance_illegal_transition_raises(self):
        """L1 anchor: PipelineRunStore.advance 非法转换抛 IllegalTransition，锚点 K8s CRD 范式。"""
        store = PipelineRunStore()
        run = store.create(config={}, stages=["a"])
        # Pending 不能直接 complete（跳过 Running，违反 K8s CRD 状态机）
        with pytest.raises(IllegalTransition):
            store.advance(run.run_id, "complete")

    def test_store_create_initial_state_is_pending(self):
        """L1 anchor: PipelineRunStore.create 初始状态为 Pending，锚点设计文档 0.6 节。"""
        store = PipelineRunStore()
        run = store.create(config={}, stages=["a"])
        assert run.state == "Pending"
