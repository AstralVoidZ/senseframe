"""阶段 4.3 MCP skill tool 测试。

覆盖：
- senseframe_skill_save：合法技能 / 语法错误 / validated 状态
- senseframe_skill_get：已存在 / 不存在 / 指定版本
- senseframe_skill_search：相关度分数 / 空库 / top_k 钳制
- senseframe_skill_remove：已存在 / 不存在 / 依赖检查 / force=True
- 异常路由：SkillNotFoundError / SkillHasDependentsError / SkillValidationError → category="internal"

设计文档 0.3 节技能库契约 + 0.5 节错误信封。

测试隔离：用 monkeypatch 替换 senseframe.skills._default_library 为独立 SkillLibrary，
避免污染全局技能库（~/.senseframe/skills）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from senseframe.mcp.tools.skill import (
    senseframe_skill_get,
    senseframe_skill_remove,
    senseframe_skill_save,
    senseframe_skill_search,
)
from senseframe.mcp.views.skill import (
    SkillRemoveResponse,
    SkillSaveResponse,
    SkillSearchResponse,
    SkillView,
)
from senseframe.skills import Skill, SkillLibrary


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def isolated_library(monkeypatch, tmp_path):
    """每个测试用独立的 SkillLibrary 实例，避免污染全局。

    通过 monkeypatch 替换 senseframe.skills._default_library，
    使 save_skill / load_skill / search_skills_with_scores / get_skill_library
    都返回此独立实例。
    """
    import senseframe.skills as skills_module

    fake_lib = SkillLibrary(storage_dir=str(tmp_path / "test_skills"))
    monkeypatch.setattr(skills_module, "_default_library", fake_lib)
    return fake_lib


VALID_CODE = '''
def hello():
    """A valid skill."""
    return "hello"
'''

SYNTAX_ERROR_CODE = '''
def hello(
    # missing closing paren
'''


# ============================================================
# senseframe_skill_save 测试
# ============================================================


class TestSkillSave:
    """senseframe_skill_save 测试。"""

    @pytest.mark.asyncio
    async def test_save_valid_skill(self, isolated_library):
        """保存合法技能返回 validated=True, saved=True。"""
        result = await senseframe_skill_save(
            name="test_skill",
            code=VALID_CODE,
            description="A test skill",
            tags=["test"],
        )
        assert isinstance(result, SkillSaveResponse)
        assert result.saved is True
        assert result.validated is True
        assert result.name == "test_skill"
        assert result.version == "1.0.0"
        assert result.validation_errors == []

    @pytest.mark.asyncio
    async def test_save_syntax_error_skill(self, isolated_library):
        """保存语法错误技能返回 validated=False, saved=False。"""
        result = await senseframe_skill_save(
            name="bad_skill",
            code=SYNTAX_ERROR_CODE,
        )
        assert isinstance(result, SkillSaveResponse)
        assert result.saved is False
        assert result.validated is False
        assert len(result.validation_errors) > 0
        # validation_errors 应含 SyntaxError 信息
        assert any("SyntaxError" in e for e in result.validation_errors)

    @pytest.mark.asyncio
    async def test_save_with_version(self, isolated_library):
        """保存指定版本的技能。"""
        result = await senseframe_skill_save(
            name="versioned_skill",
            code=VALID_CODE,
            version="2.1.0",
        )
        assert result.saved is True
        assert result.version == "2.1.0"

    @pytest.mark.asyncio
    async def test_save_with_tags_and_source_path(self, isolated_library):
        """保存含 tags 和 source_path 的技能。"""
        result = await senseframe_skill_save(
            name="full_skill",
            code=VALID_CODE,
            description="full skill",
            tags=["a", "b"],
            source_path="/path/to/source.py",
        )
        assert result.saved is True
        # 通过 get 验证持久化字段
        view = await senseframe_skill_get(name="full_skill")
        assert view.tags == ["a", "b"]
        assert view.source_path == "/path/to/source.py"


# ============================================================
# senseframe_skill_get 测试
# ============================================================


class TestSkillGet:
    """senseframe_skill_get 测试。"""

    @pytest.mark.asyncio
    async def test_get_existing_skill(self, isolated_library):
        """获取已存在的技能。"""
        await senseframe_skill_save(name="get_test", code=VALID_CODE)
        result = await senseframe_skill_get(name="get_test")
        assert isinstance(result, SkillView)
        assert result.name == "get_test"
        assert result.code == VALID_CODE
        assert result.validated is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises_tool_error(self, isolated_library):
        """获取不存在的技能 → ToolError（category=internal）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_skill_get(name="nonexistent")
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "internal"
        assert payload["code"] == "SkillNotFoundError"

    @pytest.mark.asyncio
    async def test_get_specific_version(self, isolated_library):
        """获取指定版本的技能。"""
        await senseframe_skill_save(
            name="versioned", code=VALID_CODE, version="1.0.0"
        )
        await senseframe_skill_save(
            name="versioned",
            code=VALID_CODE + '\n# v2 marker\n',
            version="2.0.0",
        )
        # 获取 v1（应保留在 _versions 历史中）
        result = await senseframe_skill_get(name="versioned", version="1.0.0")
        assert result.version == "1.0.0"
        # 获取 v2（当前版本）
        latest = await senseframe_skill_get(name="versioned")
        assert latest.version == "2.0.0"

    @pytest.mark.asyncio
    async def test_get_nonexistent_version_raises(self, isolated_library):
        """获取不存在的版本 → ToolError。"""
        await senseframe_skill_save(name="v_test", code=VALID_CODE)
        with pytest.raises(ToolError) as exc_info:
            await senseframe_skill_get(name="v_test", version="9.9.9")
        payload = json.loads(str(exc_info.value))
        assert payload["code"] == "SkillNotFoundError"


# ============================================================
# senseframe_skill_search 测试
# ============================================================


class TestSkillSearch:
    """senseframe_skill_search 测试。"""

    @pytest.mark.asyncio
    async def test_search_returns_scored_results(self, isolated_library):
        """检索返回带分数的结果列表。"""
        await senseframe_skill_save(
            name="cnn_classifier",
            code=VALID_CODE,
            description="CNN image classifier for classification tasks",
            tags=["cnn", "classification", "image"],
        )
        await senseframe_skill_save(
            name="rnn_sequence",
            code=VALID_CODE,
            description="RNN sequence model for time series",
            tags=["rnn", "sequence", "time"],
        )
        result = await senseframe_skill_search(query="image classification", top_k=5)
        assert isinstance(result, SkillSearchResponse)
        assert result.query == "image classification"
        assert len(result.items) > 0
        # cnn_classifier 应比 rnn_sequence 更相关（在前）
        assert result.items[0].skill.name == "cnn_classifier"
        assert result.items[0].score > 0
        # 分数降序
        for i in range(len(result.items) - 1):
            assert result.items[i].score >= result.items[i + 1].score
        assert result.total_count == len(result.items)

    @pytest.mark.asyncio
    async def test_search_empty_library(self, isolated_library):
        """空库检索返回空列表。"""
        result = await senseframe_skill_search(query="anything")
        assert isinstance(result, SkillSearchResponse)
        assert len(result.items) == 0
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_search_top_k_clamped_to_max(self, isolated_library):
        """top_k=1000 钳制到 50。"""
        await senseframe_skill_save(
            name="s1", code=VALID_CODE, description="test query match"
        )
        result = await senseframe_skill_search(query="test", top_k=1000)
        assert result.top_k == 50

    @pytest.mark.asyncio
    async def test_search_top_k_clamped_to_min(self, isolated_library):
        """top_k=0 钳制到 1。"""
        await senseframe_skill_save(
            name="s1", code=VALID_CODE, description="test query match"
        )
        result = await senseframe_skill_search(query="test", top_k=0)
        assert result.top_k == 1

    @pytest.mark.asyncio
    async def test_search_returns_skill_views(self, isolated_library):
        """检索结果 items 是 SkillSearchResultView，含 SkillView + score。"""
        from senseframe.mcp.views.skill import SkillSearchResultView

        await senseframe_skill_save(
            name="visible_skill",
            code=VALID_CODE,
            description="a visible skill",
            tags=["visible"],
        )
        result = await senseframe_skill_search(query="visible")
        assert len(result.items) > 0
        item = result.items[0]
        assert isinstance(item, SkillSearchResultView)
        assert isinstance(item.skill, SkillView)
        assert item.skill.name == "visible_skill"
        assert isinstance(item.score, float)


# ============================================================
# senseframe_skill_remove 测试
# ============================================================


class TestSkillRemove:
    """senseframe_skill_remove 测试。"""

    @pytest.mark.asyncio
    async def test_remove_existing_skill(self, isolated_library):
        """删除已存在的技能。"""
        await senseframe_skill_save(name="to_remove", code=VALID_CODE)
        result = await senseframe_skill_remove(name="to_remove")
        assert isinstance(result, SkillRemoveResponse)
        assert result.removed is True
        assert result.name == "to_remove"
        assert result.force is False
        # 再次 get 应失败
        with pytest.raises(ToolError):
            await senseframe_skill_get(name="to_remove")

    @pytest.mark.asyncio
    async def test_remove_nonexistent_raises_tool_error(self, isolated_library):
        """删除不存在的技能 → ToolError。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_skill_remove(name="nonexistent")
        payload = json.loads(str(exc_info.value))
        assert payload["code"] == "SkillNotFoundError"

    @pytest.mark.asyncio
    async def test_remove_with_dependency_raises_tool_error(
        self, isolated_library
    ):
        """有依赖且 force=False → ToolError（SkillHasDependentsError）。"""
        # 保存 base 和 derived 技能
        await senseframe_skill_save(name="base", code=VALID_CODE)
        await senseframe_skill_save(name="derived", code=VALID_CODE)
        # 手动设置 derived 依赖 base（save_skill 不直接支持 depends_on 参数）
        from senseframe.skills import get_skill_library

        lib = get_skill_library()
        derived_skill = lib.get("derived")
        assert derived_skill is not None
        derived_skill.depends_on = ["base"]
        lib.update(derived_skill)

        with pytest.raises(ToolError) as exc_info:
            await senseframe_skill_remove(name="base", force=False)
        payload = json.loads(str(exc_info.value))
        assert payload["code"] == "SkillHasDependentsError"
        assert payload["category"] == "internal"

    @pytest.mark.asyncio
    async def test_remove_force_bypasses_dependency(self, isolated_library):
        """force=True 绕过依赖检查。"""
        await senseframe_skill_save(name="base", code=VALID_CODE)
        await senseframe_skill_save(name="derived", code=VALID_CODE)
        from senseframe.skills import get_skill_library

        lib = get_skill_library()
        derived = lib.get("derived")
        assert derived is not None
        derived.depends_on = ["base"]
        lib.update(derived)

        result = await senseframe_skill_remove(name="base", force=True)
        assert result.removed is True
        assert result.force is True


# ============================================================
# Skill 异常路由测试
# ============================================================


class TestSkillErrorRouting:
    """skill 异常路由到 category='internal' 测试。"""

    def test_skill_not_found_routes_to_internal(self):
        """SkillNotFoundError → category='internal'。"""
        from senseframe.mcp.errors import SkillNotFoundError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            SkillNotFoundError("skill not found")
        )
        assert envelope.category == "internal"
        assert envelope.code == "SkillNotFoundError"

    def test_skill_has_dependents_routes_to_internal(self):
        """SkillHasDependentsError → category='internal'。"""
        from senseframe.mcp.errors import SkillHasDependentsError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            SkillHasDependentsError("skill has dependents")
        )
        assert envelope.category == "internal"
        assert envelope.code == "SkillHasDependentsError"

    def test_skill_validation_error_routes_to_internal(self):
        """SkillValidationError → category='internal'。"""
        from senseframe.mcp.errors import SkillValidationError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            SkillValidationError("validation failed")
        )
        assert envelope.category == "internal"
        assert envelope.code == "SkillValidationError"

    def test_skill_error_base_class_routes_to_internal(self):
        """SkillError 基类 → category='internal'（子类全覆盖）。"""
        from senseframe.mcp.errors import SkillError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            SkillError("generic skill failure")
        )
        assert envelope.category == "internal"
        assert envelope.code == "SkillError"


# ============================================================
# View 层契约
# ============================================================


class TestSkillViews:
    """Skill view FrozenModel 契约。"""

    def test_skill_view_is_frozen(self):
        """SkillView 必须不可变（frozen=True）。"""
        from pydantic import ValidationError

        view = SkillView(
            name="s1",
            description="d",
            code="x = 1",
        )
        with pytest.raises(ValidationError):
            view.name = "modified"  # type: ignore[misc]

    def test_skill_view_rejects_extra_fields(self):
        """SkillView 必须拒绝未知字段（extra='forbid'）。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SkillView(
                name="s1",
                description="d",
                code="x = 1",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_skill_view_from_domain_skill(self):
        """SkillView.from_domain 从 Skill 域对象投影。"""
        skill = Skill(
            name="domain_skill",
            description="a domain skill",
            code="def f(): pass",
            tags=["x", "y"],
            version="3.0.0",
            depends_on=["base"],
        )
        view = SkillView.from_domain(skill)
        assert view.name == "domain_skill"
        assert view.description == "a domain skill"
        assert view.code == "def f(): pass"
        assert view.tags == ["x", "y"]
        assert view.version == "3.0.0"
        assert view.depends_on == ["base"]

    def test_skill_view_from_domain_dict(self):
        """SkillView.from_domain 也支持 dict 输入（_safe_get 兼容）。"""
        d = {
            "name": "dict_skill",
            "description": "from dict",
            "code": "x = 1",
            "tags": ["t"],
        }
        view = SkillView.from_domain(d)
        assert view.name == "dict_skill"
        assert view.tags == ["t"]
