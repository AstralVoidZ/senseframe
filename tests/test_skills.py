"""senseframe.skills 模块测试。

覆盖 Skill dataclass、SkillLibrary（含嵌入检索、版本管理、依赖追踪）、
嵌入函数与全局 API。
"""

import math
import uuid
from datetime import datetime

import pytest

from senseframe.skills import (
    Skill,
    SkillLibrary,
    get_skill_library,
    save_skill,
    load_skill,
    search_skills,
    list_skills,
    _embed_text,
    _cosine_similarity,
)


# ============================================================
# TestSkill
# ============================================================

class TestSkill:
    """Skill dataclass 基本行为。"""

    def test_create_minimal(self):
        s = Skill(name="my_skill", description="desc", code="x = 1")
        assert s.name == "my_skill"
        assert s.description == "desc"
        assert s.code == "x = 1"
        assert s.tags == []
        assert s.version == "1.0.0"
        assert s.validated is False
        assert s.validation_errors == []

    def test_depends_on_default_empty(self):
        s = Skill(name="s", description="", code="x = 1")
        assert s.depends_on == []

    def test_created_at_auto_filled(self):
        s = Skill(name="s", description="", code="x = 1")
        assert s.created_at != ""
        # ISO 格式可解析
        datetime.fromisoformat(s.created_at)

    def test_created_at_preserved_when_provided(self):
        s = Skill(name="s", description="", code="x = 1", created_at="2024-01-01T00:00:00")
        assert s.created_at == "2024-01-01T00:00:00"

    def test_to_dict_from_dict_roundtrip(self):
        s = Skill(
            name="focal_loss",
            description="focal loss for imbalanced",
            code="def focal(): pass",
            tags=["loss", "classification"],
            version="2.1.0",
            depends_on=["base_loss"],
        )
        d = s.to_dict()
        s2 = Skill.from_dict(d)
        assert s2.name == s.name
        assert s2.description == s.description
        assert s2.code == s.code
        assert s2.tags == s.tags
        assert s2.version == s.version
        assert s2.depends_on == s.depends_on
        assert s2.created_at == s.created_at

    def test_from_dict_missing_optional_fields(self):
        d = {"name": "s", "code": "x = 1"}
        s = Skill.from_dict(d)
        assert s.name == "s"
        assert s.description == ""
        assert s.tags == []
        assert s.version == "1.0.0"
        assert s.depends_on == []


# ============================================================
# TestSkillLibrary
# ============================================================

class TestSkillLibrary:
    """SkillLibrary 的注册、检索、版本管理、依赖追踪。"""

    def test_register_success(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        s = Skill(name="my_loss", description="a loss", code="x = 1")
        ok = lib.register(s)
        assert ok is True
        assert s.validated is True
        assert s.validation_errors == []
        assert lib.get("my_loss") is s

    def test_register_syntax_error_returns_false(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        s = Skill(name="bad", description="", code="def f(")
        ok = lib.register(s)
        assert ok is False
        assert s.validated is False
        assert len(s.validation_errors) > 0
        assert lib.get("bad") is None

    def test_get_latest_version(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        s = Skill(name="v", description="", code="x = 1", version="1.0.0")
        lib.register(s)
        got = lib.get("v")
        assert got is not None
        assert got.version == "1.0.0"

    def test_get_specific_version(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        v1 = Skill(name="v", description="", code="x = 1", version="1.0.0")
        v2 = Skill(name="v", description="", code="x = 2", version="2.0.0")
        lib.register(v1)
        lib.register(v2)
        got = lib.get("v", version="1.0.0")
        assert got is not None
        assert got.version == "1.0.0"
        assert got.code == "x = 1"
        # 最新版本是 v2
        latest = lib.get("v")
        assert latest.version == "2.0.0"

    def test_get_nonexistent_version_returns_none(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        s = Skill(name="v", description="", code="x = 1", version="1.0.0")
        lib.register(s)
        assert lib.get("v", version="不存在") is None

    def test_list_skills_sorted(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        for name in ["c_skill", "a_skill", "b_skill"]:
            lib.register(Skill(name=name, description="", code="x = 1"))
        assert lib.list_skills() == ["a_skill", "b_skill", "c_skill"]

    def test_search_returns_relevant(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        lib.register(Skill(
            name="focal_loss",
            description="focal loss for imbalanced classification",
            code="x = 1",
            tags=["loss", "classification"],
        ))
        lib.register(Skill(
            name="data_loader",
            description="load csv data",
            code="x = 1",
            tags=["data"],
        ))
        results = lib.search("focal loss classification")
        assert len(results) > 0
        assert results[0].name == "focal_loss"

    def test_search_top_k(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        for i in range(5):
            lib.register(Skill(
                name=f"loss_{i}",
                description=f"loss function variant {i}",
                code="x = 1",
                tags=["loss"],
            ))
        results = lib.search("loss", top_k=2)
        assert len(results) <= 2

    def test_search_empty_library(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        assert lib.search("anything") == []

    def test_remove(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        lib.register(Skill(name="r", description="", code="x = 1"))
        assert lib.remove("r") is True
        assert lib.get("r") is None

    def test_remove_with_dependency_raises(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        lib.register(Skill(name="base", description="", code="x = 1"))
        lib.register(Skill(
            name="derived",
            description="",
            code="x = 2",
            depends_on=["base"],
        ))
        with pytest.raises(ValueError):
            lib.remove("base", force=False)

    def test_remove_force(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        lib.register(Skill(name="base", description="", code="x = 1"))
        lib.register(Skill(
            name="derived",
            description="",
            code="x = 2",
            depends_on=["base"],
        ))
        assert lib.remove("base", force=True) is True
        assert lib.get("base") is None

    def test_register_same_name_preserves_old_version(self, tmp_path):
        lib = SkillLibrary(storage_dir=str(tmp_path))
        v1 = Skill(name="s", description="", code="x = 1", version="1.0.0")
        lib.register(v1)
        v2 = Skill(name="s", description="", code="x = 2", version="2.0.0")
        lib.register(v2)
        # 旧版本保留在 _versions
        assert "s" in lib._versions
        assert len(lib._versions["s"]) == 1
        assert lib._versions["s"][0].version == "1.0.0"
        # 当前是 v2
        assert lib.get("s").version == "2.0.0"


# ============================================================
# TestEmbedding
# ============================================================

class TestEmbedding:
    """嵌入函数：L2 归一化、余弦相似度、词形变化。"""

    def test_embed_text_l2_normalized(self):
        vec = _embed_text("focal loss classification")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_same_text(self):
        a = _embed_text("some text")
        sim = _cosine_similarity(a, a)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_different_text_less_than_one(self):
        a = _embed_text("focal loss classification")
        b = _embed_text("data loader utility")
        sim = _cosine_similarity(a, b)
        assert sim < 1.0

    def test_word_form_variation_nonzero_similarity(self):
        """hash-based 嵌入对 classify/classification 有非零相似度（共享字符 3-gram）。"""
        a = _embed_text("classify")
        b = _embed_text("classification")
        sim = _cosine_similarity(a, b)
        assert sim > 0.0


# ============================================================
# TestGlobalAPI
# ============================================================

class TestGlobalAPI:
    """全局 API：save_skill / load_skill / search_skills / list_skills / get_skill_library。

    注意：全局 API 使用默认技能库（~/.senseframe/skills），用唯一技能名避免冲突。
    """

    @pytest.fixture
    def unique_name(self):
        return f"test_global_{uuid.uuid4().hex[:8]}"

    def test_save_and_load(self, unique_name):
        code = "def my_func(): return 42"
        ok = save_skill(
            name=unique_name,
            code=code,
            description="global api test",
            tags=["test"],
        )
        assert ok is True
        try:
            loaded = load_skill(unique_name)
            assert loaded is not None
            assert loaded.code == code
        finally:
            get_skill_library().remove(unique_name, force=True)

    def test_search_skills(self, unique_name):
        save_skill(
            name=unique_name,
            code="x = 1",
            description="focal loss for classification",
            tags=["loss", "classification"],
        )
        try:
            results = search_skills("focal loss classification")
            names = [s.name for s in results]
            assert unique_name in names
        finally:
            get_skill_library().remove(unique_name, force=True)

    def test_list_skills_contains(self, unique_name):
        save_skill(name=unique_name, code="x = 1", description="d")
        try:
            assert unique_name in list_skills()
        finally:
            get_skill_library().remove(unique_name, force=True)

    def test_get_skill_library_singleton(self):
        a = get_skill_library()
        b = get_skill_library()
        assert a is b
