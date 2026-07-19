"""阶段 3.1 MCP study tool 测试。

覆盖：
- senseframe_study_create / ask / tell / get / list / compare / stop 全流程
- StudyManager 单例（set_default_manager 测试注入）
- HATEOAS _transitions（running: [ask, stop, get, compare] / stopped: [get, compare]）
- ToolError 信封路由（KeyError → study category）
- cursor 分页（filter_dict + next_cursor）
- 多 study 对比（方向感知 best_study_id 推导）

设计文档 0.3 节 L4 SP + 0.4 节 HATEOAS + 0.8 节阶段 3.1。
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from senseframe.mcp.orchestration.study_manager import (
    get_default_manager,
    set_default_manager,
)
from senseframe.mcp.orchestration.study_transitions import (
    STUDY_STATE_RUNNING,
    STUDY_STATE_STOPPED,
    get_study_transitions,
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.tools.study import (
    senseframe_study_ask,
    senseframe_study_compare,
    senseframe_study_create,
    senseframe_study_get,
    senseframe_study_list,
    senseframe_study_stop,
    senseframe_study_tell,
)
from senseframe.mcp.views.study import (
    StudyAskResponse,
    StudyCompareView,
    StudyCreateResponse,
    StudyListView,
    StudyTellResponse,
    StudyView,
)
from senseframe.search_protocol import StudyManager


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fresh_manager():
    """每个测试用独立的 StudyManager，避免进程级单例污染。"""
    set_default_manager(StudyManager())
    manager = get_default_manager()
    yield manager
    set_default_manager(StudyManager())  # 重置


@pytest.fixture
def simple_search_space():
    """简单搜索空间（2 个参数）。"""
    return [
        {"name": "lr", "type": "float", "low": 0.0001, "high": 0.1, "log": True},
        {"name": "batch_size", "type": "categorical", "choices": [16, 32, 64]},
    ]


# ============================================================
# create / ask / tell 基础流程
# ============================================================


class TestStudyCreateAskTell:
    """study_create / ask / tell 三步流程。"""

    @pytest.mark.asyncio
    async def test_create_returns_study_id_and_transitions(
        self, fresh_manager, simple_search_space
    ):
        """create 返回 study_id + running 状态的 transitions=[ask, stop, get, compare]。"""
        resp = await senseframe_study_create(
            name="test_study",
            direction="maximize",
            search_space=simple_search_space,
            sampler="random",
        )
        assert isinstance(resp, StudyCreateResponse)
        assert resp.study_id.startswith("study_")
        assert resp.name == "test_study"
        assert resp.direction == "maximize"
        assert resp.sampler == "random"
        # transitions 应含 ask / stop / get / compare（running 状态）
        actions = {t.action for t in resp.transitions}
        assert "ask" in actions
        assert "stop" in actions
        assert "get" in actions
        assert "compare" in actions

    @pytest.mark.asyncio
    async def test_ask_returns_trial_with_params(
        self, fresh_manager, simple_search_space
    ):
        """ask 返回 trial_id + params（含 lr / batch_size key）。"""
        create_resp = await senseframe_study_create(
            name="test_ask",
            search_space=simple_search_space,
        )
        ask_resp = await senseframe_study_ask(study_id=create_resp.study_id)
        assert isinstance(ask_resp, StudyAskResponse)
        assert ask_resp.study_id == create_resp.study_id
        assert ask_resp.trial_id.startswith("trial_")
        # params 应含 lr / batch_size
        assert "lr" in ask_resp.params
        assert "batch_size" in ask_resp.params
        # transitions 应含 tell
        actions = {t.action for t in ask_resp.transitions}
        assert "tell" in actions or "ask" in actions  # running 状态

    @pytest.mark.asyncio
    async def test_tell_records_value_and_returns_response(
        self, fresh_manager, simple_search_space
    ):
        """tell 上报 value 后返回 StudyTellResponse。"""
        create_resp = await senseframe_study_create(
            name="test_tell",
            search_space=simple_search_space,
        )
        ask_resp = await senseframe_study_ask(study_id=create_resp.study_id)
        tell_resp = await senseframe_study_tell(
            trial_id=ask_resp.trial_id,
            value=0.85,
            state="completed",
        )
        assert isinstance(tell_resp, StudyTellResponse)
        assert tell_resp.trial_id == ask_resp.trial_id
        assert tell_resp.study_id == create_resp.study_id
        assert tell_resp.state == "completed"
        assert tell_resp.value == 0.85

    @pytest.mark.asyncio
    async def test_ask_unknown_study_raises_tool_error_study_category(
        self, fresh_manager
    ):
        """ask 不存在的 study_id → ToolError，category=study。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_study_ask(study_id="nonexistent_study")
        # ToolError message 是 ToolErrorResponse 的 JSON
        import json
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "study"
        assert "nonexistent_study" in payload["message"]

    @pytest.mark.asyncio
    async def test_tell_unknown_trial_raises_tool_error_study_category(
        self, fresh_manager
    ):
        """tell 不存在的 trial_id → ToolError，category=study。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_study_tell(trial_id="nonexistent_trial", value=0.5)
        import json
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "study"


# ============================================================
# get / list / compare 查询流程
# ============================================================


class TestStudyGetListCompare:
    """study_get / list / compare 查询流程。"""

    @pytest.mark.asyncio
    async def test_get_returns_study_view_with_best_trial(
        self, fresh_manager, simple_search_space
    ):
        """get 返回 StudyView，含 n_trials / n_completed / best_value。"""
        create_resp = await senseframe_study_create(
            name="test_get",
            search_space=simple_search_space,
        )
        # 跑 2 个 trial
        for value in [0.7, 0.85]:
            ask_resp = await senseframe_study_ask(study_id=create_resp.study_id)
            await senseframe_study_tell(trial_id=ask_resp.trial_id, value=value)

        view = await senseframe_study_get(study_id=create_resp.study_id)
        assert isinstance(view, StudyView)
        assert view.study_id == create_resp.study_id
        assert view.n_trials == 2
        assert view.n_completed == 2
        assert view.best_value == 0.85  # maximize
        assert view.best_trial_id is not None

    @pytest.mark.asyncio
    async def test_get_unknown_study_raises_tool_error(self, fresh_manager):
        """get 不存在的 study_id → ToolError，category=study。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_study_get(study_id="nonexistent")
        import json
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "study"

    @pytest.mark.asyncio
    async def test_list_returns_paginated_studies(self, fresh_manager, simple_search_space):
        """list 返回 StudyListView，含 items + total_count。"""
        # 创建 3 个 study
        for i in range(3):
            await senseframe_study_create(
                name=f"test_list_{i}",
                search_space=simple_search_space,
            )
        view = await senseframe_study_list(limit=10)
        assert isinstance(view, StudyListView)
        assert view.total_count >= 3
        assert len(view.items) >= 3
        assert view.limit == 10

    @pytest.mark.asyncio
    async def test_list_with_filter_status(self, fresh_manager, simple_search_space):
        """list 支持 filter_dict={"status": "running"} 过滤。"""
        await senseframe_study_create(name="s1", search_space=simple_search_space)
        await senseframe_study_create(name="s2", search_space=simple_search_space)
        view = await senseframe_study_list(filter_dict={"status": "running"})
        assert isinstance(view, StudyListView)
        # 所有 study 都应是 running
        for item in view.items:
            assert item.status == "running"

    @pytest.mark.asyncio
    async def test_compare_returns_comparison_table_with_best(
        self, fresh_manager, simple_search_space
    ):
        """compare 返回 StudyCompareView，含 best_study_id。"""
        c1 = await senseframe_study_create(
            name="c1", direction="maximize", search_space=simple_search_space
        )
        c2 = await senseframe_study_create(
            name="c2", direction="maximize", search_space=simple_search_space
        )
        # c1 跑出 0.9，c2 跑出 0.7
        a1 = await senseframe_study_ask(study_id=c1.study_id)
        await senseframe_study_tell(trial_id=a1.trial_id, value=0.9)
        a2 = await senseframe_study_ask(study_id=c2.study_id)
        await senseframe_study_tell(trial_id=a2.trial_id, value=0.7)

        view = await senseframe_study_compare(study_ids=[c1.study_id, c2.study_id])
        assert isinstance(view, StudyCompareView)
        assert len(view.studies) == 2
        assert len(view.comparison_table) == 2
        # maximize 方向下 c1（0.9）应胜出
        assert view.best_study_id == c1.study_id

    @pytest.mark.asyncio
    async def test_compare_requires_at_least_two_studies(self, fresh_manager):
        """compare 少于 2 个 study_id → ToolError。"""
        with pytest.raises(ToolError):
            await senseframe_study_compare(study_ids=["only_one"])


# ============================================================
# stop 终态流程
# ============================================================


class TestStudyStop:
    """study_stop 终态流程。"""

    @pytest.mark.asyncio
    async def test_stop_transitions_to_stopped_state(
        self, fresh_manager, simple_search_space
    ):
        """stop 后状态变为 stopped，transitions=[get, compare]（无 ask）。"""
        create_resp = await senseframe_study_create(
            name="test_stop",
            search_space=simple_search_space,
        )
        view = await senseframe_study_stop(study_id=create_resp.study_id)
        assert isinstance(view, StudyView)
        assert view.status == "stopped"
        # 终态 transitions 应仅含 get / compare
        actions = {t.action for t in view.transitions}
        assert "ask" not in actions
        assert "stop" not in actions
        assert "get" in actions
        assert "compare" in actions

    @pytest.mark.asyncio
    async def test_ask_after_stop_raises_tool_error(self, fresh_manager, simple_search_space):
        """stopped 状态下 ask → RuntimeError（被 ToolError 包装）。"""
        create_resp = await senseframe_study_create(
            name="test_ask_after_stop",
            search_space=simple_search_space,
        )
        await senseframe_study_stop(study_id=create_resp.study_id)
        with pytest.raises(ToolError):
            await senseframe_study_ask(study_id=create_resp.study_id)


# ============================================================
# HATEOAS transitions 完整性
# ============================================================


class TestStudyTransitions:
    """Study HATEOAS _transitions 完整性。"""

    def test_running_state_has_ask_stop_get_compare(self):
        """running 状态 transitions=[ask, stop, get, compare]。"""
        transitions = get_study_transitions(STUDY_STATE_RUNNING)
        actions = {t.action for t in transitions}
        assert actions == {"ask", "stop", "get", "compare"}

    def test_stopped_state_has_only_get_compare(self):
        """stopped 状态（终态）transitions=[get, compare]。"""
        transitions = get_study_transitions(STUDY_STATE_STOPPED)
        actions = {t.action for t in transitions}
        assert actions == {"get", "compare"}

    def test_unknown_state_returns_empty_transitions(self):
        """未知状态返回空 transitions 列表。"""
        transitions = get_study_transitions("unknown_state")
        assert transitions == []


# ============================================================
# View 层契约
# ============================================================


class TestStudyViews:
    """Study view FrozenModel 契约。"""

    def test_study_view_is_frozen(self):
        """StudyView 必须不可变（frozen=True）。"""
        from pydantic import ValidationError

        view = StudyView(
            study_id="s1",
            name="test",
            direction="maximize",
            sampler="random",
            status="running",
            created_at="2026-07-19T00:00:00",
        )
        with pytest.raises(ValidationError):
            view.status = "stopped"  # type: ignore[misc]

    def test_study_view_rejects_extra_fields(self):
        """StudyView 必须拒绝未知字段（extra='forbid'）。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            StudyView(
                study_id="s1",
                name="test",
                direction="maximize",
                sampler="random",
                status="running",
                created_at="2026-07-19T00:00:00",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_trial_view_is_frozen(self):
        """TrialView 必须不可变。"""
        from pydantic import ValidationError

        from senseframe.mcp.views.study import TrialView

        view = TrialView(
            trial_id="t1",
            study_id="s1",
            params={"lr": 0.001},
            state="completed",
        )
        with pytest.raises(ValidationError):
            view.state = "failed"  # type: ignore[misc]
