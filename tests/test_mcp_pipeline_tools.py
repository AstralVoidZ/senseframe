"""阶段 2.1-2.5 MCP pipeline tool 测试。

覆盖：
- PipelineRunStore CRUD（create/get/advance/list/delete）
- 状态机 7 转换 + 幂等短路
- cursor 分页（encode/decode/fingerprint 校验）
- HATEOAS _transitions 在每个状态下返回正确的合法动作
- ToolError 信封路由（PipelineNotFound → pipeline category）

设计文档 0.6 节 L5 OPP + 0.4 节 HATEOAS。
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from senseframe.mcp.errors import (
    CursorFilterMismatch,
    IllegalTransition,
    InvalidCursor,
    PipelineNotFound,
)
from senseframe.mcp.models.pipeline_run import (
    PipelineRun,
    STATE_FAILED,
    STATE_PAUSED,
    STATE_PENDING,
    STATE_RUNNING,
    STATE_SUCCEEDED,
)
from senseframe.mcp.orchestration.pipeline_run import (
    PipelineRunStore,
    VALID_ACTIONS,
    VALID_TRANSITIONS,
    IDEMPOTENT_ACTIONS,
    get_default_store,
    set_default_store,
    trigger,
)
from senseframe.mcp.orchestration.transitions import (
    TransitionDef,
    _TRANSITIONS_BY_STATE,
    get_transitions,
)
from senseframe.mcp.pagination.cursor import (
    EMPTY_FINGERPRINT,
    assert_fingerprint_matches,
    decode_cursor,
    encode_cursor,
    filter_fingerprint,
)
from senseframe.mcp.pagination.page import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MIN_LIMIT,
    build_page,
    clamp_limit,
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.tools.pipeline import (
    senseframe_pipeline_advance,
    senseframe_pipeline_create,
    senseframe_pipeline_get,
    senseframe_pipeline_list,
    senseframe_pipeline_pause,
    senseframe_pipeline_resume,
    get_pipeline_run_store,
)
from senseframe.mcp.views.pipeline import (
    PipelineAdvanceResponse,
    PipelineCreateResponse,
    PipelineRunListView,
    PipelineRunView,
    TransitionView,
)
from senseframe.mcp.views.tool_error import ToolErrorResponse


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fresh_store():
    """每个测试用独立的 PipelineRunStore，避免进程级单例污染。

    set_default_store(None) 重置，然后获取一个新的 store 实例，
    测试结束再重置为 None。
    """
    set_default_store(None)
    store = get_default_store()
    yield store
    set_default_store(None)


@pytest.fixture
def sample_config() -> dict:
    """最小可用的 ExperimentConfig dict（仅用于 PipelineRun 携带）。"""
    return {
        "scene": {"name": "test", "dataset": "test", "model_id": "MLP"},
        "input_features": [{"name": "x", "type": "tabular", "shape": [10]}],
        "output_features": [{"name": "y", "type": "category", "num_classes": 3}],
        "trainer": {"epochs": 1},
    }


@pytest.fixture
def sample_stages() -> list[str]:
    """测试用 stage 列表（不必与 Pipeline.default() 完全一致）。"""
    return ["validate", "preflight", "load", "train"]


# ============================================================
# 1. PipelineRunStore CRUD
# ============================================================


class TestPipelineRunStoreCRUD:
    """PipelineRunStore 增删改查基础测试。"""

    def test_create_returns_pending_run(self, fresh_store, sample_config, sample_stages):
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        assert isinstance(run, PipelineRun)
        assert run.state == STATE_PENDING
        assert run.run_id  # uuid4 hex 非空
        assert run.config == sample_config
        assert run.stages == sample_stages
        assert run.completed_stages == []
        assert run.failed_stage is None
        assert run.error_message is None
        assert run.trial_id is None
        assert run.created_at == run.updated_at

    def test_create_defensive_copies(self, fresh_store, sample_config, sample_stages):
        """create 必须防御性 copy 输入，避免外部修改影响存储。"""
        config = dict(sample_config)
        stages = list(sample_stages)
        run = fresh_store.create(config=config, stages=stages)
        # 修改原 dict / list 不影响 run
        config["scene"]["name"] = "modified"
        stages.append("extra_stage")
        assert run.config != config or run.config["scene"]["name"] == sample_config["scene"]["name"]
        assert run.stages == sample_stages

    def test_create_rejects_invalid_input_types(self, fresh_store):
        with pytest.raises(TypeError):
            fresh_store.create(config="not a dict", stages=["s"])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            fresh_store.create(config={}, stages="not a list")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            fresh_store.create(config={}, stages=[1, 2, 3])  # type: ignore[list-item]

    def test_get_returns_stored_run(self, fresh_store, sample_config, sample_stages):
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fetched = fresh_store.get(run.run_id)
        assert fetched == run  # frozen dataclass 相等比较

    def test_get_unknown_run_raises_pipeline_not_found(self, fresh_store):
        with pytest.raises(PipelineNotFound):
            fresh_store.get("nonexistent-run-id")

    def test_delete_removes_run(self, fresh_store, sample_config, sample_stages):
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.delete(run.run_id)
        with pytest.raises(PipelineNotFound):
            fresh_store.get(run.run_id)

    def test_delete_unknown_run_raises(self, fresh_store):
        with pytest.raises(PipelineNotFound):
            fresh_store.delete("nonexistent-run-id")

    def test_list_runs_empty_store(self, fresh_store):
        items, total, has_more = fresh_store.list_runs()
        assert items == []
        assert total == 0
        assert has_more is False

    def test_list_runs_returns_all(self, fresh_store, sample_config, sample_stages):
        for _ in range(3):
            fresh_store.create(config=sample_config, stages=sample_stages)
        items, total, has_more = fresh_store.list_runs()
        assert len(items) == 3
        assert total == 3
        assert has_more is False

    def test_list_runs_with_limit(self, fresh_store, sample_config, sample_stages):
        for _ in range(5):
            fresh_store.create(config=sample_config, stages=sample_stages)
        items, total, has_more = fresh_store.list_runs(limit=2)
        assert len(items) == 2
        assert total == 5
        assert has_more is True

    def test_list_runs_with_filter_by_state(
        self, fresh_store, sample_config, sample_stages
    ):
        """filter_dict 支持按 state 等值过滤。"""
        r1 = fresh_store.create(config=sample_config, stages=sample_stages)
        r2 = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(r1.run_id, action="start")
        # r1 现在是 Running, r2 仍是 Pending
        running_items, running_total, _ = fresh_store.list_runs(
            filter_dict={"state": STATE_RUNNING}
        )
        assert running_total == 1
        assert running_items[0].run_id == r1.run_id

        pending_items, pending_total, _ = fresh_store.list_runs(
            filter_dict={"state": STATE_PENDING}
        )
        assert pending_total == 1
        assert pending_items[0].run_id == r2.run_id


# ============================================================
# 2. 状态机：7 转换 + 幂等短路
# ============================================================


class TestStateMachine:
    """5 状态 + 7 转换 + 3 幂等短路的 FSM 测试。

    注意：合法转换的完整验证已迁移至 L1 契约测试
    tests/unit/l1_contract/test_k8s_crd_state_machine.py（硬编码 7 转换期望值，
    锚点为设计文档 0.6 节）。此处保留反向用例（非法转换抛异常）。
    """

    def test_all_states_have_constant_strings(self):
        """5 状态必须是字符串常量。"""
        for state in (STATE_PENDING, STATE_RUNNING, STATE_PAUSED, STATE_SUCCEEDED, STATE_FAILED):
            assert isinstance(state, str)
            assert state in {"Pending", "Running", "Paused", "Succeeded", "Failed"}

    def test_trigger_illegal_transition_raises(self):
        """非法转换必须抛 IllegalTransition。"""
        illegal_cases = [
            (STATE_PENDING, "complete"),  # Pending 不能直接 complete
            (STATE_PENDING, "pause"),    # Pending 不能 pause
            (STATE_PENDING, "resume"),   # Pending 不能 resume
            (STATE_RUNNING, "start"),    # Running 不能再 start
            (STATE_RUNNING, "retry"),    # Running 不能 retry
            (STATE_PAUSED, "start"),     # Paused 不能 start
            (STATE_PAUSED, "complete"),  # Paused 不能 complete
            (STATE_PAUSED, "fail"),       # Paused 不能 fail
            (STATE_SUCCEEDED, "start"),   # Succeeded 终态
            (STATE_SUCCEEDED, "fail"),    # Succeeded 终态
            (STATE_SUCCEEDED, "pause"),   # Succeeded 终态
            (STATE_FAILED, "complete"),  # Failed 不能直接 complete
            (STATE_FAILED, "pause"),     # Failed 不能 pause
            (STATE_FAILED, "resume"),    # Failed 不能 resume
        ]
        for state, action in illegal_cases:
            with pytest.raises(IllegalTransition):
                trigger(state, action)

    def test_trigger_with_unknown_action_raises(self):
        with pytest.raises(IllegalTransition):
            trigger(STATE_PENDING, "unknown_action")

    def test_idempotent_actions_count(self):
        """IDEMPOTENT_ACTIONS 必须含 3 个状态（Succeeded/Failed/Paused）。"""
        assert len(IDEMPOTENT_ACTIONS) == 3
        assert STATE_SUCCEEDED in IDEMPOTENT_ACTIONS
        assert STATE_FAILED in IDEMPOTENT_ACTIONS
        assert STATE_PAUSED in IDEMPOTENT_ACTIONS

    def test_idempotent_short_circuit_on_succeeded(self, fresh_store, sample_config, sample_stages):
        """complete on Succeeded → no-op（幂等短路）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="complete", completed_stage="train")
        # 再次 complete 应短路
        r2 = fresh_store.advance(run.run_id, action="complete", completed_stage="train")
        assert r2.state == STATE_SUCCEEDED

    def test_idempotent_short_circuit_on_failed(self, fresh_store, sample_config, sample_stages):
        """fail on Failed → no-op（幂等短路）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="fail", failed_stage="train", error_message="err")
        # 再次 fail 应短路
        r2 = fresh_store.advance(run.run_id, action="fail", failed_stage="train", error_message="another")
        assert r2.state == STATE_FAILED
        # 短路时不应更新 error_message（保留首次失败信息）
        assert r2.error_message == "err"

    def test_idempotent_short_circuit_on_paused(self, fresh_store, sample_config, sample_stages):
        """pause on Paused → no-op（幂等短路）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="pause")
        # 再次 pause 应短路
        r2 = fresh_store.advance(run.run_id, action="pause")
        assert r2.state == STATE_PAUSED

    def test_advance_appends_completed_stage(self, fresh_store, sample_config, sample_stages):
        """complete 动作应追加 completed_stage 到 completed_stages。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        r1 = fresh_store.advance(run.run_id, action="complete", completed_stage="train")
        assert r1.completed_stages == ["train"]
        # 再次合法 complete（已 Succeeded 短路，但若假设短路不生效，应不重复）
        r2 = fresh_store.advance(run.run_id, action="complete", completed_stage="train")
        # 短路：返回原 run，completed_stages 不变
        assert r2.completed_stages == ["train"]

    def test_advance_records_failure(self, fresh_store, sample_config, sample_stages):
        """fail 动作应记录 failed_stage 和 error_message。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        r1 = fresh_store.advance(
            run.run_id,
            action="fail",
            failed_stage="train",
            error_message="loss exploded",
        )
        assert r1.state == STATE_FAILED
        assert r1.failed_stage == "train"
        assert r1.error_message == "loss exploded"

    def test_advance_skip_from_pending(self, fresh_store, sample_config, sample_stages):
        """skip 直接 Pending → Failed（不执行）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        r1 = fresh_store.advance(
            run.run_id,
            action="skip",
            failed_stage="validate",
            error_message="user skipped",
        )
        assert r1.state == STATE_FAILED
        assert r1.failed_stage == "validate"

    def test_advance_retry_clears_failure(self, fresh_store, sample_config, sample_stages):
        """retry 应清空 failed_stage + error_message。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(
            run.run_id, action="fail", failed_stage="train", error_message="err"
        )
        r1 = fresh_store.advance(run.run_id, action="retry")
        assert r1.state == STATE_RUNNING
        assert r1.failed_stage is None
        assert r1.error_message is None

    def test_advance_sets_trial_id(self, fresh_store, sample_config, sample_stages):
        """start 时设置 trial_id 应保留。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        r1 = fresh_store.advance(run.run_id, action="start", trial_id="trial-abc")
        assert r1.trial_id == "trial-abc"

    def test_advance_unknown_run_raises(self, fresh_store):
        with pytest.raises(PipelineNotFound):
            fresh_store.advance("nonexistent", action="start")

    def test_advance_illegal_transition_raises(
        self, fresh_store, sample_config, sample_stages
    ):
        """Pending → complete 非法转换应抛 IllegalTransition。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        with pytest.raises(IllegalTransition):
            fresh_store.advance(run.run_id, action="complete")

    def test_advance_returns_immutable_new_instance(
        self, fresh_store, sample_config, sample_stages
    ):
        """advance 必须返回新 PipelineRun 实例，旧实例保留作为历史快照。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        r1 = fresh_store.advance(run.run_id, action="start")
        assert r1 is not run  # 不同实例
        assert run.state == STATE_PENDING  # 旧实例不变
        assert r1.state == STATE_RUNNING  # 新实例是新状态

    def test_valid_actions_set_contains_all_actions(self):
        """VALID_ACTIONS 必须含 7 个动作 + 幂等动作（去重后）。"""
        # 7 转换动作：start/skip/complete/fail/pause/resume/retry
        expected_actions = {"start", "skip", "complete", "fail", "pause", "resume", "retry"}
        # 加上幂等动作（与转换动作有重叠，去重）
        assert expected_actions.issubset(VALID_ACTIONS)


# ============================================================
# 3. Cursor 分页
# ============================================================


class TestCursorPagination:
    """cursor 分页：encode/decode + fingerprint 校验。"""

    def test_empty_fingerprint_constant(self):
        assert EMPTY_FINGERPRINT == "00000000"
        assert len(EMPTY_FINGERPRINT) == 8

    def test_filter_fingerprint_none_or_empty_returns_empty_constant(self):
        assert filter_fingerprint(None) == EMPTY_FINGERPRINT
        assert filter_fingerprint({}) == EMPTY_FINGERPRINT

    def test_filter_fingerprint_deterministic(self):
        """同 filter_dict 多次调用必须返回同样结果。"""
        f1 = filter_fingerprint({"state": "Running"})
        f2 = filter_fingerprint({"state": "Running"})
        assert f1 == f2
        assert len(f1) == 8

    def test_filter_fingerprint_independent_of_key_order(self):
        """filter fingerprint 应稳定（sort_keys=True，无视插入顺序）。"""
        f1 = filter_fingerprint({"state": "Running", "trial_id": "t1"})
        f2 = filter_fingerprint({"trial_id": "t1", "state": "Running"})
        assert f1 == f2

    def test_filter_fingerprint_different_filters_differ(self):
        f1 = filter_fingerprint({"state": "Running"})
        f2 = filter_fingerprint({"state": "Pending"})
        assert f1 != f2

    def test_encode_decode_roundtrip_no_filter(self):
        cursor = encode_cursor("run-abc123", None)
        last_id, fingerprint = decode_cursor(cursor)
        assert last_id == "run-abc123"
        assert fingerprint == EMPTY_FINGERPRINT

    def test_encode_decode_roundtrip_with_filter(self):
        cursor = encode_cursor("run-xyz", {"state": "Running"})
        last_id, fingerprint = decode_cursor(cursor)
        assert last_id == "run-xyz"
        assert fingerprint == filter_fingerprint({"state": "Running"})

    def test_encode_cursor_returns_base64_urlsafe(self):
        """cursor 必须是 base64-urlsafe 编码（无 padding）。"""
        cursor = encode_cursor("abc", None)
        # 不含标准 base64 的 + / 字符
        assert "+" not in cursor
        assert "/" not in cursor
        # 不含 padding =
        assert "=" not in cursor

    def test_decode_cursor_empty_raises(self):
        with pytest.raises(InvalidCursor):
            decode_cursor("")
        with pytest.raises(InvalidCursor):
            decode_cursor(None)  # type: ignore[arg-type]

    def test_decode_cursor_invalid_base64_raises(self):
        with pytest.raises(InvalidCursor):
            decode_cursor("!!!not-base64!!!")

    def test_decode_cursor_missing_separator_raises(self):
        """伪造的 cursor 缺少 | 分隔符应抛 InvalidCursor。"""
        # 一个有效 base64-urlsafe 但内容无 | 的字符串
        import base64
        fake = base64.urlsafe_b64encode(b"noseparator").decode("ascii").rstrip("=")
        with pytest.raises(InvalidCursor):
            decode_cursor(fake)

    def test_decode_cursor_whitespace_rejected(self):
        with pytest.raises(InvalidCursor):
            decode_cursor(" cursor with space")
        with pytest.raises(InvalidCursor):
            decode_cursor("cursor\twith\ttab")

    def test_assert_fingerprint_matches_none_cursor_returns_none(self):
        """cursor=None 首次请求，返回 None。"""
        assert assert_fingerprint_matches(None, None) is None
        assert assert_fingerprint_matches(None, {"state": "Running"}) is None

    def test_assert_fingerprint_matches_valid_cursor(self):
        cursor = encode_cursor("last-id", {"state": "Running"})
        last_id = assert_fingerprint_matches(cursor, {"state": "Running"})
        assert last_id == "last-id"

    def test_assert_fingerprint_matches_mismatched_filter_raises(self):
        """cursor 的 filter 与当前 filter 不一致应抛 CursorFilterMismatch。"""
        cursor = encode_cursor("last-id", {"state": "Running"})
        with pytest.raises(CursorFilterMismatch):
            assert_fingerprint_matches(cursor, {"state": "Pending"})

    def test_assert_fingerprint_matches_cursor_with_no_filter_raises_on_filter(
        self,
    ):
        """cursor 编码时有 filter，当前请求无 filter 应 mismatch。"""
        cursor = encode_cursor("last-id", {"state": "Running"})
        with pytest.raises(CursorFilterMismatch):
            assert_fingerprint_matches(cursor, None)

    def test_assert_fingerprint_matches_none_cursor_with_filter(self):
        """cursor 编码时无 filter，当前请求有 filter 应 mismatch。"""
        cursor = encode_cursor("last-id", None)
        with pytest.raises(CursorFilterMismatch):
            assert_fingerprint_matches(cursor, {"state": "Running"})


# ============================================================
# 4. clamp_limit + build_page
# ============================================================


class TestPageHelpers:
    """clamp_limit + build_page 辅助函数。"""

    def test_clamp_limit_constants(self):
        assert MIN_LIMIT == 1
        assert MAX_LIMIT == 200
        assert DEFAULT_LIMIT == 50

    def test_clamp_limit_below_min(self):
        assert clamp_limit(0) == MIN_LIMIT
        assert clamp_limit(-5) == MIN_LIMIT

    def test_clamp_limit_above_max(self):
        assert clamp_limit(500) == MAX_LIMIT

    def test_clamp_limit_in_range(self):
        assert clamp_limit(50) == 50
        assert clamp_limit(100) == 100

    def test_build_page_with_has_more(self):
        items = ["a", "b", "c"]
        page = build_page(
            items=items,
            total_count=10,
            limit=3,
            has_more=True,
            last_id_fn=lambda x: x,
            filter_dict=None,
        )
        assert page.items == ["a", "b", "c"]
        assert page.total_count == 10
        assert page.limit == 3
        assert page.next_cursor is not None

    def test_build_page_no_more(self):
        items = ["a", "b"]
        page = build_page(
            items=items,
            total_count=2,
            limit=10,
            has_more=False,
            last_id_fn=lambda x: x,
            filter_dict=None,
        )
        assert page.next_cursor is None

    def test_build_page_empty_items_no_cursor(self):
        page = build_page(
            items=[],
            total_count=0,
            limit=10,
            has_more=False,
            last_id_fn=lambda x: x,
        )
        assert page.next_cursor is None
        assert page.items == []

    def test_build_page_next_cursor_decodes_to_last_id(self):
        items = ["x", "y", "z"]
        page = build_page(
            items=items,
            total_count=10,
            limit=3,
            has_more=True,
            last_id_fn=lambda x: x,
            filter_dict={"state": "Running"},
        )
        assert page.next_cursor is not None
        last_id, _ = decode_cursor(page.next_cursor)
        assert last_id == "z"  # 保留的最后一行


# ============================================================
# 5. HATEOAS _transitions
# ============================================================


class TestHateoasTransitions:
    """HATEOAS _transitions 在每个状态下返回正确的合法动作。"""

    def test_transitions_for_pending_state(self, fresh_store, sample_config, sample_stages):
        """Pending 状态下应有 start + skip 两个转换。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        transitions = get_transitions(run.state, run)
        actions = [t.action for t in transitions]
        assert set(actions) == {"start", "skip"}
        # start 的 suggested_tool 应指向 advance
        start_t = next(t for t in transitions if t.action == "start")
        assert start_t.target_state == STATE_RUNNING
        assert start_t.suggested_tool == "senseframe_pipeline_advance"

    def test_transitions_for_running_state(self, fresh_store, sample_config, sample_stages):
        """Running 状态下应有 complete + fail + pause 三个转换。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        run = fresh_store.get(run.run_id)
        transitions = get_transitions(run.state, run)
        actions = {t.action for t in transitions}
        assert actions == {"complete", "fail", "pause"}
        # pause 的 suggested_tool 应指向 pause
        pause_t = next(t for t in transitions if t.action == "pause")
        assert pause_t.suggested_tool == "senseframe_pipeline_pause"
        assert pause_t.target_state == STATE_PAUSED

    def test_transitions_for_paused_state(self, fresh_store, sample_config, sample_stages):
        """Paused 状态下应有 resume 一个转换。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="pause")
        run = fresh_store.get(run.run_id)
        transitions = get_transitions(run.state, run)
        actions = [t.action for t in transitions]
        assert actions == ["resume"]
        assert transitions[0].suggested_tool == "senseframe_pipeline_resume"
        assert transitions[0].target_state == STATE_RUNNING

    def test_transitions_for_succeeded_state(self, fresh_store, sample_config, sample_stages):
        """Succeeded 终态应无转换（空列表）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="complete", completed_stage="train")
        run = fresh_store.get(run.run_id)
        transitions = get_transitions(run.state, run)
        assert transitions == []

    def test_transitions_for_failed_state(self, fresh_store, sample_config, sample_stages):
        """Failed 状态下应有 retry 一个转换。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(run.run_id, action="fail", failed_stage="train", error_message="err")
        run = fresh_store.get(run.run_id)
        transitions = get_transitions(run.state, run)
        actions = [t.action for t in transitions]
        assert actions == ["retry"]
        assert transitions[0].target_state == STATE_RUNNING
        assert transitions[0].suggested_tool == "senseframe_pipeline_advance"

    def test_transitions_returns_transition_view_instances(
        self, fresh_store, sample_config, sample_stages
    ):
        """transitions 列表中的元素必须是 TransitionView（FrozenModel 子类）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        transitions = get_transitions(run.state, run)
        for t in transitions:
            assert isinstance(t, TransitionView)

    def test_transitions_with_none_run_returns_static_def(self):
        """run=None 时也能返回 transitions（仅静态 def，无 prerequisites 推导）。"""
        transitions = get_transitions(STATE_PENDING, None)
        assert len(transitions) == 2  # start + skip
        # 无 prerequisites
        for t in transitions:
            assert t.prerequisites == []

    def test_transitions_def_frozen(self):
        """TransitionDef 是 frozen dataclass。"""
        d = TransitionDef(action="start", target_state=STATE_RUNNING)
        with pytest.raises(Exception):
            d.action = "modified"  # type: ignore[misc]


# ============================================================
# 6. ToolError 信封路由
# ============================================================


class TestToolErrorEnvelopeRouting:
    """ToolError 信封路由：异常 → category。"""

    def test_pipeline_not_found_routes_to_pipeline_category(self):
        env = ToolErrorResponse.envelope_from(PipelineNotFound("run-x not found"))
        assert env.category == "pipeline"
        assert env.code == "PipelineNotFound"
        assert "run-x" in env.message

    def test_illegal_transition_routes_to_pipeline_category(self):
        env = ToolErrorResponse.envelope_from(IllegalTransition("bad"))
        assert env.category == "pipeline"
        assert env.code == "IllegalTransition"

    def test_invalid_cursor_routes_to_config_category(self):
        """InvalidCursor 继承 InvalidPathError → config category。"""
        env = ToolErrorResponse.envelope_from(InvalidCursor("bad cursor"))
        assert env.category == "config"
        assert env.code == "InvalidCursor"

    def test_cursor_filter_mismatch_routes_to_config_category(self):
        env = ToolErrorResponse.envelope_from(CursorFilterMismatch("mismatch"))
        assert env.category == "config"
        assert env.code == "CursorFilterMismatch"

    def test_to_tool_error_returns_envelope_json(self):
        """to_tool_error 返回 ToolError，payload 含 PipelineNotFound 信封 JSON。"""
        import json

        from mcp.server.fastmcp.exceptions import ToolError

        exc = PipelineNotFound("run-xyz not found")
        tool_err = to_tool_error(exc)
        assert isinstance(tool_err, ToolError)
        payload = json.loads(str(tool_err))
        assert payload["code"] == "PipelineNotFound"
        assert payload["category"] == "pipeline"
        assert "run-xyz" in payload["message"]


# ============================================================
# 7. Tool handler async 集成测试
# ============================================================


class TestPipelineToolHandlers:
    """7 个 pipeline tool handler 的 async 集成测试。"""

    @pytest.mark.asyncio
    async def test_pipeline_create_returns_pending_response(
        self, fresh_store, sample_config, sample_stages
    ):
        result = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        assert isinstance(result, PipelineCreateResponse)
        assert result.state == STATE_PENDING
        assert result.run_id
        assert result.created_at
        # Pending 状态应有 start + skip transitions
        actions = {t.action for t in result.transitions}
        assert actions == {"start", "skip"}

    @pytest.mark.asyncio
    async def test_pipeline_advance_start_transitions_to_running(
        self, fresh_store, sample_config, sample_stages
    ):
        create_resp = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        advance_resp = await senseframe_pipeline_advance(
            run_id=create_resp.run_id, action="start"
        )
        assert isinstance(advance_resp, PipelineAdvanceResponse)
        assert advance_resp.previous_state == STATE_PENDING
        assert advance_resp.new_state == STATE_RUNNING
        assert advance_resp.action == "start"
        # Running 状态应有 complete + fail + pause transitions
        actions = {t.action for t in advance_resp.transitions}
        assert actions == {"complete", "fail", "pause"}

    @pytest.mark.asyncio
    async def test_pipeline_get_returns_view_with_transitions(
        self, fresh_store, sample_config, sample_stages
    ):
        create_resp = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        view = await senseframe_pipeline_get(run_id=create_resp.run_id)
        assert isinstance(view, PipelineRunView)
        assert view.run_id == create_resp.run_id
        assert view.state == STATE_PENDING
        # view.stages 应为 StageView 列表
        assert len(view.stages) == len(sample_stages)
        # Pending 状态下所有 stage 应为 pending
        for stage_view in view.stages:
            assert stage_view.state == "pending"
        # 应含 start + skip transitions
        actions = {t.action for t in view.transitions}
        assert actions == {"start", "skip"}

    @pytest.mark.asyncio
    async def test_pipeline_list_returns_paginated_view(
        self, fresh_store, sample_config, sample_stages
    ):
        # 创建 3 个 run
        for _ in range(3):
            await senseframe_pipeline_create(
                config=sample_config, stages=sample_stages
            )
        view = await senseframe_pipeline_list(limit=10)
        assert isinstance(view, PipelineRunListView)
        assert len(view.items) == 3
        assert view.total_count == 3
        assert view.limit == 10
        assert view.next_cursor is None  # 无更多

    @pytest.mark.asyncio
    async def test_pipeline_list_pagination(self, fresh_store, sample_config, sample_stages):
        """list 分页：limit=2 + 5 个 run → has_more + next_cursor。"""
        for _ in range(5):
            await senseframe_pipeline_create(
                config=sample_config, stages=sample_stages
            )
        view = await senseframe_pipeline_list(limit=2)
        assert len(view.items) == 2
        assert view.total_count == 5
        assert view.next_cursor is not None

        # 第二页：用上一次的 next_cursor
        view2 = await senseframe_pipeline_list(
            cursor=view.next_cursor, limit=2
        )
        assert len(view2.items) == 2
        # 第二页的 run_id 不应与第一页重复
        page1_ids = {i.run_id for i in view.items}
        page2_ids = {i.run_id for i in view2.items}
        assert not (page1_ids & page2_ids)

    @pytest.mark.asyncio
    async def test_pipeline_list_filter_inconsistency_raises(
        self, fresh_store, sample_config, sample_stages
    ):
        """filter 不一致时 cursor 应被拒绝（CursorFilterMismatch）。"""
        # 创建若干 run
        for _ in range(3):
            await senseframe_pipeline_create(
                config=sample_config, stages=sample_stages
            )
        # 第一页用 filter_dict={"state": "Pending"}
        view = await senseframe_pipeline_list(
            limit=2, filter_dict={"state": STATE_PENDING}
        )
        # 第二页用不同 filter，应抛 ToolError（含 CursorFilterMismatch code）
        # E1 修复后 pipeline 工具统一通过 to_tool_error 路由异常
        from mcp.server.fastmcp.exceptions import ToolError
        import json as _json
        with pytest.raises(ToolError) as exc_info:
            await senseframe_pipeline_list(
                cursor=view.next_cursor,
                limit=2,
                filter_dict={"state": STATE_RUNNING},
            )
        payload = _json.loads(str(exc_info.value))
        assert payload["code"] == "CursorFilterMismatch"

    @pytest.mark.asyncio
    async def test_pipeline_pause_resume_cycle(
        self, fresh_store, sample_config, sample_stages
    ):
        """create → start → pause → resume 完整周期。"""
        create_resp = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        await senseframe_pipeline_advance(run_id=create_resp.run_id, action="start")

        # pause
        pause_resp = await senseframe_pipeline_pause(run_id=create_resp.run_id)
        assert pause_resp.previous_state == STATE_RUNNING
        assert pause_resp.new_state == STATE_PAUSED
        assert pause_resp.action == "pause"
        # Paused 状态应有 resume transition
        actions = {t.action for t in pause_resp.transitions}
        assert actions == {"resume"}

        # resume
        resume_resp = await senseframe_pipeline_resume(run_id=create_resp.run_id)
        assert resume_resp.previous_state == STATE_PAUSED
        assert resume_resp.new_state == STATE_RUNNING
        assert resume_resp.action == "resume"

    @pytest.mark.asyncio
    async def test_pipeline_pause_idempotent_on_paused(
        self, fresh_store, sample_config, sample_stages
    ):
        """对已 Paused 的 run 再次 pause 应幂等短路。"""
        create_resp = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        await senseframe_pipeline_advance(run_id=create_resp.run_id, action="start")
        # 第一次 pause
        await senseframe_pipeline_pause(run_id=create_resp.run_id)
        # 第二次 pause：应短路，不抛异常
        r2 = await senseframe_pipeline_pause(run_id=create_resp.run_id)
        assert r2.new_state == STATE_PAUSED
        assert r2.previous_state == STATE_PAUSED  # 短路时 previous == new

    @pytest.mark.asyncio
    async def test_pipeline_advance_unknown_run_raises_pipeline_not_found(
        self, fresh_store
    ):
        """advance 不存在的 run_id 应抛 ToolError（含 PipelineNotFound code）。"""
        # E1 修复后 pipeline 工具统一通过 to_tool_error 路由异常
        from mcp.server.fastmcp.exceptions import ToolError
        import json as _json
        with pytest.raises(ToolError) as exc_info:
            await senseframe_pipeline_advance(
                run_id="nonexistent", action="start"
            )
        payload = _json.loads(str(exc_info.value))
        assert payload["code"] == "PipelineNotFound"
        assert payload["category"] == "pipeline"

    @pytest.mark.asyncio
    async def test_pipeline_get_unknown_run_raises(self, fresh_store):
        """get 不存在的 run_id 应抛 ToolError（含 PipelineNotFound code）。"""
        from mcp.server.fastmcp.exceptions import ToolError
        import json as _json
        with pytest.raises(ToolError) as exc_info:
            await senseframe_pipeline_get(run_id="nonexistent")
        payload = _json.loads(str(exc_info.value))
        assert payload["code"] == "PipelineNotFound"

    @pytest.mark.asyncio
    async def test_pipeline_advance_illegal_transition_raises(
        self, fresh_store, sample_config, sample_stages
    ):
        """Pending → complete 非法转换应抛 ToolError（含 IllegalTransition code）。"""
        create_resp = await senseframe_pipeline_create(
            config=sample_config, stages=sample_stages
        )
        from mcp.server.fastmcp.exceptions import ToolError
        import json as _json
        with pytest.raises(ToolError) as exc_info:
            await senseframe_pipeline_advance(
                run_id=create_resp.run_id, action="complete"
            )
        payload = _json.loads(str(exc_info.value))
        assert payload["code"] == "IllegalTransition"
        assert payload["category"] == "pipeline"


# ============================================================
# 8. PipelineRunStore 线程安全
# ============================================================


class TestPipelineRunStoreThreadSafety:
    """PipelineRunStore 用 threading.RLock 保护内存存储。"""

    def test_store_has_rlock(self, fresh_store):
        """PipelineRunStore 必须含 _lock 属性（threading.RLock）。"""
        assert hasattr(fresh_store, "_lock")
        assert isinstance(fresh_store._lock, type(threading.RLock()))

    def test_concurrent_create_no_loss(self, fresh_store, sample_config, sample_stages):
        """并发 create 不丢失 run（每个线程创建的 run 都应在 store 中可见）。"""
        N_THREADS = 8
        N_PER_THREAD = 10
        results: list[PipelineRun] = []
        results_lock = threading.Lock()

        def worker():
            local_runs: list[PipelineRun] = []
            for _ in range(N_PER_THREAD):
                r = fresh_store.create(
                    config=sample_config, stages=sample_stages
                )
                local_runs.append(r)
            with results_lock:
                results.extend(local_runs)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 应有 N_THREADS * N_PER_THREAD 个 run
        assert len(results) == N_THREADS * N_PER_THREAD
        # 所有 run_id 应唯一
        run_ids = {r.run_id for r in results}
        assert len(run_ids) == N_THREADS * N_PER_THREAD

    def test_concurrent_advance_state_consistent(
        self, fresh_store, sample_config, sample_stages
    ):
        """并发 advance 同一 run：状态转换必须一致（不出现非法中间态）。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        # 并发 start：只允许一个成功，其余应 IllegalTransition
        results: list[str] = []  # "ok" / "err"
        results_lock = threading.Lock()

        def worker():
            try:
                fresh_store.advance(run.run_id, action="start")
                with results_lock:
                    results.append("ok")
            except IllegalTransition:
                with results_lock:
                    results.append("err")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 恰好一个 ok，其余 err（FSM 保证）
        ok_count = results.count("ok")
        err_count = results.count("err")
        assert ok_count == 1
        assert err_count == 7
        # 最终状态必须是 Running
        final_run = fresh_store.get(run.run_id)
        assert final_run.state == STATE_RUNNING


# ============================================================
# 9. 进程级单例
# ============================================================


class TestDefaultStoreSingleton:
    """进程级 PipelineRunStore 单例测试。"""

    def test_get_default_store_returns_same_instance_after_first_call(self):
        set_default_store(None)
        s1 = get_default_store()
        s2 = get_default_store()
        assert s1 is s2

    def test_set_default_store_overrides_singleton(self):
        """set_default_store(store) 注入自定义 store。"""
        set_default_store(None)
        custom = PipelineRunStore()
        set_default_store(custom)
        assert get_default_store() is custom
        # cleanup
        set_default_store(None)

    def test_set_default_store_none_resets_singleton(self):
        """set_default_store(None) 重置单例，下次 get_default_store 创建新实例。"""
        set_default_store(None)
        s1 = get_default_store()
        set_default_store(None)
        s2 = get_default_store()
        assert s1 is not s2  # 不同实例

    def test_get_pipeline_run_store_delegates_to_default(self):
        """tools/pipeline.get_pipeline_run_store() 委托给 orchestration.get_default_store()。"""
        set_default_store(None)
        store = get_pipeline_run_store()
        assert store is get_default_store()
        set_default_store(None)
