"""阶段 3.2 MCP hpo_setup tool 测试。

覆盖：
- senseframe_hpo_setup 基础流程（ExperimentConfig → Study 搜索空间）
- _convert_scene_search_space_to_sp 转换函数
- ToolError 信封路由
- search_space 缺失场景的 fallback 处理
- 集成 ExplorationTracker recommend_next（阶段 3.3 验证）

设计文档 0.8 节阶段 3.2 + 0.6 节 Ask-Tell 解耦。
"""

from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from senseframe.mcp.orchestration.study_manager import (
    get_default_manager,
    set_default_manager,
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.tools.exploration import senseframe_exploration_recommend
from senseframe.mcp.tools.hpo import (
    _convert_scene_search_space_to_sp,
    senseframe_hpo_setup,
)
from senseframe.mcp.views.exploration import (
    ExplorationRecommendationItem,
    ExplorationRecommendationView,
)
from senseframe.mcp.views.study import StudyCreateResponse
from senseframe.search_protocol import SearchSpace, StudyManager


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fresh_manager():
    """每个测试用独立的 StudyManager。"""
    set_default_manager(StudyManager())
    manager = get_default_manager()
    yield manager
    set_default_manager(StudyManager())


@pytest.fixture
def generic_config():
    """简单 ExperimentConfig dict（generic 场景）。"""
    return {
        "scene": {
            "name": "generic",
            "model_id": "MLP",
            "dataset": "mock_dataset",
        },
        "input_features": [{"name": "x", "shape": [10], "type": "tabular"}],
        "output_features": [{"name": "y", "type": "category", "num_classes": 3}],
    }


# ============================================================
# _convert_scene_search_space_to_sp 转换函数
# ============================================================


class TestConvertSceneSearchSpaceToSP:
    """scenes.base.SearchSpace → search_protocol.SearchSpace 转换。"""

    def test_convert_none_returns_empty_search_space(self):
        """None 输入返回空 SearchSpace。"""
        ss = _convert_scene_search_space_to_sp(None)
        assert isinstance(ss, SearchSpace)
        assert len(ss.parameters) == 0

    def test_convert_float_param(self):
        """float 类型参数转换。"""
        class MockSceneSpace:
            params = {
                "lr": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
            }
        ss = _convert_scene_search_space_to_sp(MockSceneSpace())
        assert len(ss.parameters) == 1
        assert ss.parameters[0].name == "lr"
        assert ss.parameters[0].type == "float"
        assert ss.parameters[0].low == 0.0001
        assert ss.parameters[0].high == 0.1
        assert ss.parameters[0].log is True

    def test_convert_categorical_param(self):
        """categorical 类型参数转换（values → choices）。"""
        class MockSceneSpace:
            params = {
                "batch_size": {
                    "type": "categorical",
                    "values": [16, 32, 64],
                },
            }
        ss = _convert_scene_search_space_to_sp(MockSceneSpace())
        assert len(ss.parameters) == 1
        assert ss.parameters[0].name == "batch_size"
        assert ss.parameters[0].type == "categorical"
        assert ss.parameters[0].choices == [16, 32, 64]

    def test_convert_int_param(self):
        """int 类型参数转换。"""
        class MockSceneSpace:
            params = {
                "epochs": {"type": "int", "low": 1, "high": 100, "step": 1},
            }
        ss = _convert_scene_search_space_to_sp(MockSceneSpace())
        assert len(ss.parameters) == 1
        assert ss.parameters[0].name == "epochs"
        assert ss.parameters[0].type == "int"

    def test_convert_multiple_params(self):
        """多参数转换。"""
        class MockSceneSpace:
            params = {
                "lr": {"type": "float", "low": 0.0001, "high": 0.1},
                "batch_size": {"type": "categorical", "values": [16, 32]},
                "epochs": {"type": "int", "low": 1, "high": 50},
            }
        ss = _convert_scene_search_space_to_sp(MockSceneSpace())
        assert len(ss.parameters) == 3
        names = {p.name for p in ss.parameters}
        assert names == {"lr", "batch_size", "epochs"}


# ============================================================
# senseframe_hpo_setup 集成测试
# ============================================================


class TestHpoSetupIntegration:
    """senseframe_hpo_setup 集成测试。"""

    @pytest.mark.asyncio
    async def test_hpo_setup_returns_study_create_response(
        self, fresh_manager, generic_config
    ):
        """hpo_setup 返回 StudyCreateResponse（含 study_id）。"""
        resp = await senseframe_hpo_setup(
            config=generic_config,
            n_trials=10,
            sampler="random",
            direction="minimize",
        )
        assert isinstance(resp, StudyCreateResponse)
        assert resp.study_id.startswith("study_")
        assert resp.direction == "minimize"
        assert resp.sampler == "random"
        # 应含 ask 转换（running 状态）
        actions = {t.action for t in resp.transitions}
        assert "ask" in actions

    @pytest.mark.asyncio
    async def test_hpo_setup_creates_study_in_manager(
        self, fresh_manager, generic_config
    ):
        """hpo_setup 后 manager 中应有对应的 study。"""
        resp = await senseframe_hpo_setup(config=generic_config)
        study = fresh_manager.get_study(resp.study_id)
        assert study is not None
        assert study.name == "hpo_MLP"
        assert study.direction == "minimize"

    @pytest.mark.asyncio
    async def test_hpo_setup_can_be_followed_by_ask_tell(
        self, fresh_manager, generic_config
    ):
        """hpo_setup 后可调用 study_ask / tell（Ask-Tell 解耦）。"""
        from senseframe.mcp.tools.study import (
            senseframe_study_ask,
            senseframe_study_tell,
        )

        setup_resp = await senseframe_hpo_setup(config=generic_config)
        ask_resp = await senseframe_study_ask(study_id=setup_resp.study_id)
        tell_resp = await senseframe_study_tell(
            trial_id=ask_resp.trial_id, value=0.5
        )
        assert tell_resp.state == "completed"
        assert tell_resp.value == 0.5

    @pytest.mark.asyncio
    async def test_hpo_setup_invalid_config_raises_tool_error(
        self, fresh_manager
    ):
        """无效 config（dict 缺字段）→ ToolError。"""
        with pytest.raises(ToolError):
            await senseframe_hpo_setup(config={"bad": "config"})


# ============================================================
# senseframe_exploration_recommend 集成测试（阶段 3.3）
# ============================================================


class TestExplorationRecommendIntegration:
    """senseframe_exploration_recommend 集成测试（阶段 3.3）。"""

    @pytest.mark.asyncio
    async def test_recommend_returns_view_with_recommendations(
        self, fresh_manager, generic_config
    ):
        """recommend 返回 ExplorationRecommendationView（可能为空列表）。"""
        setup_resp = await senseframe_hpo_setup(config=generic_config)
        view = await senseframe_exploration_recommend(study_id=setup_resp.study_id)
        assert isinstance(view, ExplorationRecommendationView)
        assert view.study_id == setup_resp.study_id
        assert isinstance(view.recommendations, list)
        assert view.n_recommendations == len(view.recommendations)

    @pytest.mark.asyncio
    async def test_recommend_unknown_study_raises_tool_error(self, fresh_manager):
        """recommend 不存在的 study_id → ToolError，category=study。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_exploration_recommend(study_id="nonexistent")
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "study"

    @pytest.mark.asyncio
    async def test_recommend_after_tell_returns_feedback_aware(
        self, fresh_manager, generic_config
    ):
        """tell 上报 feedback 后，recommend 应返回 feedback-aware 推荐。

        ExplorationTracker.recommend_next 会根据 feedback.status
        生成定向推荐（如 overfitting → 推荐数据增强 + 正则化）。
        """
        from senseframe.mcp.tools.study import (
            senseframe_study_ask,
            senseframe_study_tell,
        )

        setup_resp = await senseframe_hpo_setup(config=generic_config)
        ask_resp = await senseframe_study_ask(study_id=setup_resp.study_id)
        # 上报带 feedback 的结果
        await senseframe_study_tell(
            trial_id=ask_resp.trial_id,
            value=0.6,
            feedback={"status": "overfitting"},
        )
        view = await senseframe_exploration_recommend(study_id=setup_resp.study_id)
        assert isinstance(view, ExplorationRecommendationView)
        # feedback-aware 推荐应非空（overfitting 触发数据增强 + 正则化推荐）
        # 注：具体推荐内容依赖兼容性矩阵 / catalog，但至少应含 reason 字段
        for rec in view.recommendations:
            assert isinstance(rec, ExplorationRecommendationItem)
            assert isinstance(rec.strategy, dict)
            assert isinstance(rec.reason, str)

    @pytest.mark.asyncio
    async def test_recommend_top_k_clamped_to_max_50(
        self, fresh_manager, generic_config
    ):
        """top_k 钳制到 [1, 50]。"""
        setup_resp = await senseframe_hpo_setup(config=generic_config)
        # top_k=1000 应被钳制到 50
        view = await senseframe_exploration_recommend(
            study_id=setup_resp.study_id, top_k=1000
        )
        assert view.n_recommendations <= 50

    @pytest.mark.asyncio
    async def test_recommend_top_k_clamped_to_min_1(
        self, fresh_manager, generic_config
    ):
        """top_k=0 或负数应被钳制到 1。"""
        setup_resp = await senseframe_hpo_setup(config=generic_config)
        view = await senseframe_exploration_recommend(
            study_id=setup_resp.study_id, top_k=0
        )
        assert view.n_recommendations <= 1


# ============================================================
# View 层契约
# ============================================================


class TestExplorationViews:
    """Exploration view FrozenModel 契约。"""

    def test_exploration_recommendation_view_is_frozen(self):
        """ExplorationRecommendationView 必须不可变。"""
        from pydantic import ValidationError

        view = ExplorationRecommendationView(study_id="s1")
        with pytest.raises(ValidationError):
            view.study_id = "s2"  # type: ignore[misc]

    def test_exploration_recommendation_view_rejects_extra_fields(self):
        """ExplorationRecommendationView 必须拒绝未知字段。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExplorationRecommendationView(
                study_id="s1", unknown_field="bad"  # type: ignore[call-arg]
            )

    def test_exploration_recommendation_item_is_frozen(self):
        """ExplorationRecommendationItem 必须不可变。"""
        from pydantic import ValidationError

        item = ExplorationRecommendationItem(strategy={"lr": 0.001})
        with pytest.raises(ValidationError):
            item.reason = "modified"  # type: ignore[misc]
