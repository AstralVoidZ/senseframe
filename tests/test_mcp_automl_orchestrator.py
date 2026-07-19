"""阶段 3.4 MCP AutoMLOrchestrator 测试。

覆盖：
- AutoMLOrchestrator 状态机（5 状态 + 6 转换 + 幂等短路）
- senseframe_automl_create / advance / get / list 全流程
- HATEOAS _transitions（每个状态下合法动作）
- ToolError 信封路由（KeyError → study category，IllegalTransition → pipeline）
- stage 推进逻辑（complete → 推进 stage_index / Succeeded）
- cursor 分页

设计文档 0.7.3 节 AutoMLOrchestrator + 0.8 节阶段 3.4。
"""

from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from senseframe.mcp.orchestration.automl_orchestrator import (
    AUTOML_STATE_FAILED,
    AUTOML_STATE_PAUSED,
    AUTOML_STATE_PENDING,
    AUTOML_STATE_RUNNING,
    AUTOML_STATE_SUCCEEDED,
    AUTOML_VALID_ACTIONS,
    AUTOML_VALID_STAGES,
    AutoMLOrchestrator,
    get_default_orchestrator,
    set_default_orchestrator,
)
from senseframe.mcp.orchestration.automl_transitions import (
    AUTOML_TRANSITIONS_BY_STATE,
    get_automl_transitions,
)
from senseframe.mcp.tools.automl import (
    senseframe_automl_advance,
    senseframe_automl_create,
    senseframe_automl_get,
    senseframe_automl_list,
)
from senseframe.mcp.views.automl import (
    AutoMLAdvanceResponse,
    AutoMLCreateResponse,
    AutoMLPipelineListView,
    AutoMLPipelineView,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fresh_orchestrator():
    """每个测试用独立的 AutoMLOrchestrator，避免进程级单例污染。"""
    set_default_orchestrator(AutoMLOrchestrator())
    orch = get_default_orchestrator()
    yield orch
    set_default_orchestrator(AutoMLOrchestrator())


@pytest.fixture
def simple_config():
    """简单 config dict（仅含 scene 必填字段）。"""
    return {
        "scene": {
            "name": "generic",
            "model_id": "MLP",
            "dataset": "mock_dataset",
        },
        "input_features": [{"name": "x", "shape": [10], "type": "continuous"}],
        "output_features": [{"name": "y", "shape": [3], "type": "categorical"}],
    }


# ============================================================
# 状态机常量
# ============================================================


class TestAutoMLStateConstants:
    """AutoML 状态机常量。"""

    def test_five_states_defined(self):
        """5 状态：Pending/Running/Paused/Succeeded/Failed。"""
        states = {
            AUTOML_STATE_PENDING,
            AUTOML_STATE_RUNNING,
            AUTOML_STATE_PAUSED,
            AUTOML_STATE_SUCCEEDED,
            AUTOML_STATE_FAILED,
        }
        assert len(states) == 5

    def test_valid_stages_are_nas_hpo_autoaugment(self):
        """合法 stage 集合：nas / hpo / autoaugment。"""
        assert AUTOML_VALID_STAGES == frozenset({"nas", "hpo", "autoaugment"})

    def test_valid_actions_include_six(self):
        """合法动作含 6 个：start/complete/fail/pause/resume/retry。"""
        expected = {"start", "complete", "fail", "pause", "resume", "retry"}
        assert expected.issubset(AUTOML_VALID_ACTIONS)


# ============================================================
# create / advance / get / list 流程
# ============================================================


class TestAutoMLCreateAdvanceGet:
    """automl_create / advance / get 流程。"""

    @pytest.mark.asyncio
    async def test_create_returns_pipeline_id_with_pending_state(
        self, fresh_orchestrator, simple_config
    ):
        """create 返回 pipeline_id + state=Pending + transitions=[start, get]。"""
        resp = await senseframe_automl_create(
            config=simple_config,
            stages=["nas", "hpo"],
        )
        assert isinstance(resp, AutoMLCreateResponse)
        assert resp.pipeline_id.startswith("automl_")
        assert resp.state == AUTOML_STATE_PENDING
        assert resp.stages == ["nas", "hpo"]
        # Pending 状态 transitions 应含 start
        actions = {t.action for t in resp.transitions}
        assert "start" in actions
        assert "get" in actions

    @pytest.mark.asyncio
    async def test_create_rejects_empty_stages(self, fresh_orchestrator, simple_config):
        """stages 为空 → ToolError。"""
        with pytest.raises(ToolError):
            await senseframe_automl_create(config=simple_config, stages=[])

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_stage_name(self, fresh_orchestrator, simple_config):
        """stages 含非法名 → ToolError。"""
        with pytest.raises(ToolError):
            await senseframe_automl_create(
                config=simple_config, stages=["nas", "invalid_stage"]
            )

    @pytest.mark.asyncio
    async def test_advance_start_transitions_to_running(
        self, fresh_orchestrator, simple_config
    ):
        """start: Pending → Running（current_stage_index=0）。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas", "hpo"]
        )
        adv_resp = await senseframe_automl_advance(
            pipeline_id=create_resp.pipeline_id, action="start"
        )
        assert isinstance(adv_resp, AutoMLAdvanceResponse)
        assert adv_resp.state == AUTOML_STATE_RUNNING
        assert adv_resp.current_stage_index == 0
        # Running 状态 transitions 应含 complete / fail / pause
        actions = {t.action for t in adv_resp.transitions}
        assert "complete" in actions
        assert "fail" in actions

    @pytest.mark.asyncio
    async def test_advance_complete_advances_stage_index(
        self, fresh_orchestrator, simple_config
    ):
        """complete 推进 stage_index（最后一个 stage 完成时 → Succeeded）。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas", "hpo"]
        )
        pipeline_id = create_resp.pipeline_id
        # start → stage 0 (nas)
        await senseframe_automl_advance(pipeline_id=pipeline_id, action="start")
        # complete nas → stage 1 (hpo)
        adv1 = await senseframe_automl_advance(pipeline_id=pipeline_id, action="complete")
        assert adv1.state == AUTOML_STATE_RUNNING
        assert adv1.current_stage_index == 1
        assert "nas" in adv1.completed_stages
        # complete hpo → Succeeded
        adv2 = await senseframe_automl_advance(pipeline_id=pipeline_id, action="complete")
        assert adv2.state == AUTOML_STATE_SUCCEEDED
        assert "hpo" in adv2.completed_stages

    @pytest.mark.asyncio
    async def test_advance_fail_transitions_to_failed(
        self, fresh_orchestrator, simple_config
    ):
        """fail: Running → Failed。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas"]
        )
        await senseframe_automl_advance(
            pipeline_id=create_resp.pipeline_id, action="start"
        )
        adv_resp = await senseframe_automl_advance(
            pipeline_id=create_resp.pipeline_id,
            action="fail",
            error_message="boom",
        )
        assert adv_resp.state == AUTOML_STATE_FAILED

    @pytest.mark.asyncio
    async def test_advance_pause_resume_roundtrip(
        self, fresh_orchestrator, simple_config
    ):
        """pause: Running → Paused；resume: Paused → Running。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas"]
        )
        pipeline_id = create_resp.pipeline_id
        await senseframe_automl_advance(pipeline_id=pipeline_id, action="start")
        # pause
        adv1 = await senseframe_automl_advance(pipeline_id=pipeline_id, action="pause")
        assert adv1.state == AUTOML_STATE_PAUSED
        # resume
        adv2 = await senseframe_automl_advance(pipeline_id=pipeline_id, action="resume")
        assert adv2.state == AUTOML_STATE_RUNNING

    @pytest.mark.asyncio
    async def test_advance_retry_from_failed(
        self, fresh_orchestrator, simple_config
    ):
        """retry: Failed → Running，清理 failed_stage。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas"]
        )
        pipeline_id = create_resp.pipeline_id
        await senseframe_automl_advance(pipeline_id=pipeline_id, action="start")
        await senseframe_automl_advance(
            pipeline_id=pipeline_id, action="fail", error_message="boom"
        )
        adv_resp = await senseframe_automl_advance(
            pipeline_id=pipeline_id, action="retry"
        )
        assert adv_resp.state == AUTOML_STATE_RUNNING

    @pytest.mark.asyncio
    async def test_advance_unknown_pipeline_raises_tool_error(
        self, fresh_orchestrator
    ):
        """advance 不存在的 pipeline_id → ToolError，category=study（KeyError）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_automl_advance(
                pipeline_id="nonexistent", action="start"
            )
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "study"

    @pytest.mark.asyncio
    async def test_advance_illegal_transition_raises_tool_error(
        self, fresh_orchestrator, simple_config
    ):
        """非法转换（如 Pending 状态下 complete）→ ToolError，category=pipeline。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas"]
        )
        with pytest.raises(ToolError) as exc_info:
            await senseframe_automl_advance(
                pipeline_id=create_resp.pipeline_id, action="complete"
            )
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "pipeline"

    @pytest.mark.asyncio
    async def test_get_returns_pipeline_view_with_stages(
        self, fresh_orchestrator, simple_config
    ):
        """get 返回 AutoMLPipelineView，含 stages 列表。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas", "hpo", "autoaugment"]
        )
        view = await senseframe_automl_get(pipeline_id=create_resp.pipeline_id)
        assert isinstance(view, AutoMLPipelineView)
        assert view.pipeline_id == create_resp.pipeline_id
        assert view.state == AUTOML_STATE_PENDING
        assert len(view.stages) == 3
        # 所有 stage 应是 pending 状态
        for stage in view.stages:
            assert stage.state == "pending"

    @pytest.mark.asyncio
    async def test_get_after_start_shows_first_stage_running(
        self, fresh_orchestrator, simple_config
    ):
        """start 后第一个 stage 状态为 running。"""
        create_resp = await senseframe_automl_create(
            config=simple_config, stages=["nas", "hpo"]
        )
        await senseframe_automl_advance(
            pipeline_id=create_resp.pipeline_id, action="start"
        )
        view = await senseframe_automl_get(pipeline_id=create_resp.pipeline_id)
        assert view.stages[0].state == "running"
        assert view.stages[1].state == "pending"

    @pytest.mark.asyncio
    async def test_list_returns_paginated_pipelines(
        self, fresh_orchestrator, simple_config
    ):
        """list 返回 AutoMLPipelineListView。"""
        for i in range(3):
            await senseframe_automl_create(
                config=simple_config, stages=["nas"]
            )
        view = await senseframe_automl_list(limit=10)
        assert isinstance(view, AutoMLPipelineListView)
        assert view.total_count >= 3
        assert len(view.items) >= 3


# ============================================================
# HATEOAS transitions 完整性
# ============================================================


class TestAutoMLTransitions:
    """AutoML HATEOAS _transitions 完整性。"""

    def test_pending_state_has_start_and_get(self):
        """Pending 状态 transitions=[start, get]。"""
        transitions = get_automl_transitions(AUTOML_STATE_PENDING)
        actions = {t.action for t in transitions}
        assert "start" in actions
        assert "get" in actions

    def test_running_state_has_complete_fail_pause_get(self):
        """Running 状态 transitions=[complete, fail, pause, get]。"""
        transitions = get_automl_transitions(AUTOML_STATE_RUNNING)
        actions = {t.action for t in transitions}
        assert "complete" in actions
        assert "fail" in actions
        assert "pause" in actions
        assert "get" in actions

    def test_paused_state_has_resume_and_get(self):
        """Paused 状态 transitions=[resume, get]。"""
        transitions = get_automl_transitions(AUTOML_STATE_PAUSED)
        actions = {t.action for t in transitions}
        assert "resume" in actions
        assert "get" in actions

    def test_succeeded_state_has_only_get(self):
        """Succeeded 状态（终态）transitions=[get]。"""
        transitions = get_automl_transitions(AUTOML_STATE_SUCCEEDED)
        actions = {t.action for t in transitions}
        assert actions == {"get"}

    def test_failed_state_has_retry_and_get(self):
        """Failed 状态 transitions=[retry, get]。"""
        transitions = get_automl_transitions(AUTOML_STATE_FAILED)
        actions = {t.action for t in transitions}
        assert "retry" in actions
        assert "get" in actions

    def test_all_states_have_get_transition(self):
        """所有状态都应含 get 转换（advisory 查询）。"""
        for state in AUTOML_TRANSITIONS_BY_STATE:
            transitions = get_automl_transitions(state)
            actions = {t.action for t in transitions}
            assert "get" in actions, f"state={state} missing get transition"


# ============================================================
# View 层契约
# ============================================================


class TestAutoMLViews:
    """AutoML view FrozenModel 契约。"""

    def test_automl_pipeline_view_is_frozen(self):
        """AutoMLPipelineView 必须不可变。"""
        from pydantic import ValidationError

        view = AutoMLPipelineView(
            pipeline_id="p1",
            state="Pending",
            stages=[],
            created_at="2026-07-19T00:00:00",
            updated_at="2026-07-19T00:00:00",
        )
        with pytest.raises(ValidationError):
            view.state = "Running"  # type: ignore[misc]

    def test_automl_pipeline_view_rejects_extra_fields(self):
        """AutoMLPipelineView 必须拒绝未知字段。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AutoMLPipelineView(
                pipeline_id="p1",
                state="Pending",
                stages=[],
                created_at="2026-07-19T00:00:00",
                updated_at="2026-07-19T00:00:00",
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================
# 直接测试 Orchestrator（不经 MCP tool 包装）
# ============================================================


class TestAutoMLOrchestratorDirect:
    """直接测试 AutoMLOrchestrator 类（不经 MCP tool 包装）。"""

    def test_create_pipeline_returns_id_with_correct_stages(self):
        """create_pipeline 返回 pipeline_id，stages 正确存储。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={"x": 1}, stages=["nas", "hpo"])
        pipeline = orch.get(pid)
        assert pipeline.pipeline_id == pid
        assert pipeline.stages == ["nas", "hpo"]
        assert pipeline.state == AUTOML_STATE_PENDING
        assert pipeline.current_stage_index == -1
        assert pipeline.study_ids == [None, None]

    def test_set_study_for_stage_records_study_id(self):
        """set_study_for_stage 记录 study_id 到 study_ids 列表。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={}, stages=["nas", "hpo"])
        orch.set_study_for_stage(pid, stage_index=0, study_id="study_abc")
        pipeline = orch.get(pid)
        assert pipeline.study_ids[0] == "study_abc"
        assert pipeline.study_ids[1] is None

    def test_set_study_for_stage_out_of_range_raises_index_error(self):
        """stage_index 越界 → IndexError。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={}, stages=["nas"])
        with pytest.raises(IndexError):
            orch.set_study_for_stage(pid, stage_index=5, study_id="study_abc")

    def test_delete_removes_pipeline(self):
        """delete 删除 pipeline。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={}, stages=["nas"])
        orch.delete(pid)
        with pytest.raises(KeyError):
            orch.get(pid)

    def test_complete_with_study_id_records_to_stage(self):
        """complete 时附带 study_id 记录到当前 stage。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={}, stages=["nas", "hpo"])
        orch.advance(pid, action="start")
        orch.advance(pid, action="complete", study_id="study_nas_1")
        pipeline = orch.get(pid)
        assert pipeline.study_ids[0] == "study_nas_1"

    def test_idempotent_fail_on_failed_state(self):
        """fail on Failed 状态 → 幂等短路（不抛异常）。"""
        orch = AutoMLOrchestrator()
        pid = orch.create_pipeline(config={}, stages=["nas"])
        orch.advance(pid, action="start")
        orch.advance(pid, action="fail", error_message="boom")
        # 再次 fail 应幂等短路
        pipeline = orch.advance(pid, action="fail")
        assert pipeline.state == AUTOML_STATE_FAILED
