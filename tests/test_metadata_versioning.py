"""metadata.json 版本管理契约测试。

P3 演进（2026-07-18）：测试 schema_version 版本协商、迁移链、旧版兼容、未来版本拒绝。

测试维度：
1. 版本常量与迁移注册表契约
2. get_metadata_version 版本识别
3. migrate_metadata 迁移链（旧版升级 / 当前版幂等 / 未来版拒绝 / 无路径拒绝）
4. load_metadata 文件 IO + 迁移
5. MetadataVersionError 类型层级与 classify_error 命中
6. pipeline.stage_export 写入 schema_version 契约
7. 消费者（inference/export/serving）改用 load_metadata 契约
8. ERROR_CODES 与 SKILL.md 错误码表闭环
9. 版本比较工具
10. make_metadata_skeleton 骨架
"""

import json
from pathlib import Path

import pytest

from senseframe.engine.metadata import (
    CURRENT_METADATA_VERSION,
    LEGACY_VERSION,
    MIGRATIONS,
    get_metadata_version,
    migrate_metadata,
    load_metadata,
    make_metadata_skeleton,
    _version_tuple,
    _version_lt,
    _version_gt,
    _find_next_migration,
)
from senseframe.engine.runner.errors import (
    SenseFrameError,
    MetadataVersionError,
    classify_error,
)
from senseframe.schemas import ERROR_CODES


# ============================================================
# 1. 版本常量与迁移注册表契约
# ============================================================
class TestVersionConstants:
    """版本常量与迁移注册表的结构契约。"""

    def test_current_metadata_version_is_2_0_0(self):
        """CURRENT_METADATA_VERSION 固定为 '2.0.0'（首个显式版本）。

        版本号变更需同步更新此测试 + MIGRATIONS + SKILL.md 错误码表。
        """
        assert CURRENT_METADATA_VERSION == "2.0.0"

    def test_legacy_version_is_1_0_0(self):
        """LEGACY_VERSION 固定为 '1.0.0'（无 schema_version 字段的旧版隐式版本）。"""
        assert LEGACY_VERSION == "1.0.0"

    def test_migrations_contains_legacy_to_current(self):
        """MIGRATIONS 必须含 (LEGACY, CURRENT) 迁移路径，确保旧版可升级。"""
        assert (LEGACY_VERSION, CURRENT_METADATA_VERSION) in MIGRATIONS

    def test_migrations_values_are_callable(self):
        """MIGRATIONS 的值必须是可调用函数。"""
        for (src, dst), fn in MIGRATIONS.items():
            assert callable(fn), f"MIGRATIONS[({src}, {dst})] 不是可调用对象"
            assert isinstance(src, str) and isinstance(dst, str)

    def test_current_version_follows_semver(self):
        """CURRENT_METADATA_VERSION 遵循语义化版本 MAJOR.MINOR.PATCH。"""
        parts = CURRENT_METADATA_VERSION.split(".")
        assert len(parts) == 3, f"版本号应为三段式，实际: {CURRENT_METADATA_VERSION}"
        for p in parts:
            int(p)  # 每段必须是整数


# ============================================================
# 2. get_metadata_version 版本识别
# ============================================================
class TestGetMetadataVersion:
    """get_metadata_version 版本识别契约。"""

    def test_no_schema_version_returns_legacy(self):
        """无 schema_version 字段 → LEGACY_VERSION（向后兼容旧版 metadata.json）。"""
        assert get_metadata_version({"model_id": "MLP"}) == LEGACY_VERSION

    def test_empty_string_returns_legacy(self):
        """schema_version 为空字符串 → LEGACY_VERSION（falsy 回退）。"""
        assert get_metadata_version({"schema_version": ""}) == LEGACY_VERSION

    def test_none_value_returns_legacy(self):
        """schema_version 为 None → LEGACY_VERSION（falsy 回退）。"""
        assert get_metadata_version({"schema_version": None}) == LEGACY_VERSION

    def test_explicit_version_returned(self):
        """有 schema_version → 返回该版本字符串。"""
        assert get_metadata_version({"schema_version": "2.0.0"}) == "2.0.0"
        assert get_metadata_version({"schema_version": "3.0.0"}) == "3.0.0"


# ============================================================
# 3. migrate_metadata 迁移链
# ============================================================
class TestMigrateMetadata:
    """migrate_metadata 迁移链契约。"""

    def test_current_version_idempotent(self):
        """当前版 metadata → 幂等返回，不修改 data。"""
        data = {"schema_version": CURRENT_METADATA_VERSION, "model_id": "MLP"}
        result = migrate_metadata(data)
        assert result is data  # 幂等直接返回同一对象
        assert result["schema_version"] == CURRENT_METADATA_VERSION

    def test_legacy_version_upgraded_to_current(self):
        """旧版（无 schema_version）→ 升级到 CURRENT，字段保留。"""
        data = {"model_id": "MLP", "dataset": "UT_HAR", "num_classes": 14}
        result = migrate_metadata(data)
        assert result["schema_version"] == CURRENT_METADATA_VERSION
        # 原有字段保留
        assert result["model_id"] == "MLP"
        assert result["dataset"] == "UT_HAR"
        assert result["num_classes"] == 14

    def test_future_version_rejected(self):
        """高于 CURRENT 的版本 → MetadataVersionError。"""
        data = {"schema_version": "3.0.0"}
        with pytest.raises(MetadataVersionError) as exc_info:
            migrate_metadata(data)
        assert "3.0.0" in str(exc_info.value)
        assert CURRENT_METADATA_VERSION in str(exc_info.value)

    def test_future_minor_version_rejected(self):
        """高于 CURRENT 的次要版本 → MetadataVersionError。"""
        data = {"schema_version": "2.1.0"}
        with pytest.raises(MetadataVersionError):
            migrate_metadata(data)

    def test_no_migration_path_rejected(self):
        """源版本 < CURRENT 但 MIGRATIONS 无路径 → MetadataVersionError。

        模拟方式：临时注入一个 MIGRATIONS 不认识的版本。
        """
        data = {"schema_version": "1.5.0"}  # MIGRATIONS 无 1.5.0 的迁移
        with pytest.raises(MetadataVersionError) as exc_info:
            migrate_metadata(data)
        assert "1.5.0" in str(exc_info.value)

    def test_migration_preserves_all_fields(self):
        """迁移不丢失原有字段（向后兼容契约）。"""
        data = {
            "model_id": "ResNet18",
            "dataset": "Widar",
            "num_classes": 22,
            "input_shape": [1, 250, 90],
            "normalization": {"mean": [0.5], "std": [0.5]},
            "label_map": {"0": "A", "1": "B"},
            "final_eval": {"val_accuracy": 0.85},
        }
        result = migrate_metadata(data)
        for key in data:
            assert result[key] == data[key], f"迁移后字段 {key} 丢失或被修改"
        assert result["schema_version"] == CURRENT_METADATA_VERSION


# ============================================================
# 4. load_metadata 文件 IO + 迁移
# ============================================================
class TestLoadMetadata:
    """load_metadata 文件 IO + 迁移契约。"""

    def test_load_current_version_file(self, tmp_path):
        """加载当前版 metadata.json → 返回 dict 含 CURRENT schema_version。"""
        data = {"schema_version": CURRENT_METADATA_VERSION, "model_id": "MLP"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_metadata(meta_path)
        assert result["schema_version"] == CURRENT_METADATA_VERSION
        assert result["model_id"] == "MLP"

    def test_load_legacy_version_file_upgraded(self, tmp_path):
        """加载旧版 metadata.json（无 schema_version）→ 自动升级到 CURRENT。"""
        data = {"model_id": "MLP", "dataset": "UT_HAR"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_metadata(meta_path)
        assert result["schema_version"] == CURRENT_METADATA_VERSION
        assert result["model_id"] == "MLP"

    def test_load_future_version_file_raises(self, tmp_path):
        """加载未来版 metadata.json → MetadataVersionError。"""
        data = {"schema_version": "99.0.0"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(MetadataVersionError):
            load_metadata(meta_path)

    def test_load_nonexistent_file_raises(self, tmp_path):
        """文件不存在 → FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "nonexistent.json")

    def test_load_corrupted_json_raises(self, tmp_path):
        """加载损坏的 JSON → json.JSONDecodeError。

        Review 补充（2026-07-18）：原文档声明 JSONDecodeError 为可抛异常但无测试覆盖。
        """
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text("not a valid json {{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_metadata(meta_path)

    def test_load_accepts_str_or_path(self, tmp_path):
        """load_metadata 接受 str 或 Path 参数。"""
        data = {"schema_version": CURRENT_METADATA_VERSION}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        # str 参数
        result_str = load_metadata(str(meta_path))
        assert result_str["schema_version"] == CURRENT_METADATA_VERSION

        # Path 参数
        result_path = load_metadata(meta_path)
        assert result_path["schema_version"] == CURRENT_METADATA_VERSION


# ============================================================
# 5. MetadataVersionError 类型层级
# ============================================================
class TestMetadataVersionError:
    """MetadataVersionError 异常类型层级契约。"""

    def test_inherits_senseframe_error(self):
        """MetadataVersionError 继承 SenseFrameError。"""
        assert issubclass(MetadataVersionError, SenseFrameError)

    def test_error_code_attribute(self):
        """error_code 类属性 = 'METADATA_VERSION_ERROR'。"""
        assert MetadataVersionError.error_code == "METADATA_VERSION_ERROR"

    def test_classify_error_hits_senseframe_branch(self):
        """classify_error 对 MetadataVersionError 返回 METADATA_VERSION_ERROR。"""
        exc = MetadataVersionError("test")
        assert classify_error(exc) == "METADATA_VERSION_ERROR"

    def test_classify_error_no_string_matching(self):
        """classify_error 不依赖字符串匹配（空消息也能正确分类）。"""
        exc = MetadataVersionError("")
        assert classify_error(exc) == "METADATA_VERSION_ERROR"


# ============================================================
# 6. pipeline.stage_export 写入 schema_version 契约
# ============================================================
class TestPipelineWritesSchemaVersion:
    """pipeline.stage_export 写入 schema_version 契约。

    通过源码检查验证 metadata dict 含 schema_version 字段，
    避免运行完整 pipeline 的开销。
    """

    @pytest.fixture
    def pipeline_source(self):
        """读取 pipeline 包源码（stage_export 现位于 pipeline/stages/export.py）。

        拆分背景：原 pipeline.py 上帝文件拆分为 pipeline/ 包，
        stage_export 迁移到 pipeline/stages/export.py。
        """
        pipeline_path = (
            Path(__file__).resolve().parents[1]
            / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "export.py"
        )
        return pipeline_path.read_text(encoding="utf-8")

    def test_stage_export_uses_make_metadata_skeleton(self, pipeline_source):
        """stage_export 通过 make_metadata_skeleton() 构造 metadata（schema_version 由骨架函数注入）。

        遗留问题 3 修复（2026-07-19）：pipeline 不再直接引用 CURRENT_METADATA_VERSION，
        改用 make_metadata_skeleton(**fields) 构造 metadata，schema_version 由骨架函数
        统一注入，版本管理职责完全内聚到 metadata 模块。
        """
        assert "make_metadata_skeleton(" in pipeline_source, (
            "pipeline/stages/export.py 未调用 make_metadata_skeleton()。"
            "stage_export 应通过骨架函数构造 metadata，确保 schema_version 始终存在。"
        )

    def test_pipeline_imports_make_metadata_skeleton(self, pipeline_source):
        """pipeline/stages/export.py 导入 make_metadata_skeleton（schema_version 注入入口）。"""
        assert "make_metadata_skeleton" in pipeline_source, (
            "pipeline/stages/export.py 未导入 make_metadata_skeleton。"
            "stage_export 应通过 make_metadata_skeleton() 注入 schema_version，"
            "而非直接引用 CURRENT_METADATA_VERSION（版本管理职责内聚到 metadata 模块）。"
        )

    def test_pipeline_does_not_directly_reference_current_version(self, pipeline_source):
        """pipeline/stages/export.py 不再直接引用 CURRENT_METADATA_VERSION（版本管理职责内聚）。

        遗留问题 3 修复（2026-07-19）：pipeline 不应知道版本号是什么，
        版本字段由 make_metadata_skeleton 统一注入。
        """
        assert "CURRENT_METADATA_VERSION" not in pipeline_source, (
            "pipeline/stages/export.py 仍在直接引用 CURRENT_METADATA_VERSION。"
            "应通过 make_metadata_skeleton() 注入 schema_version，"
            "让版本管理职责完全内聚到 metadata 模块。"
        )

    def test_strict_schema_checks_schema_version_type(self, pipeline_source):
        """strict_schema=True 时校验 schema_version 类型为 str。"""
        assert '("schema_version", metadata["schema_version"], str)' in pipeline_source, (
            "strict_schema 类型检查列表未包含 schema_version。"
        )


# ============================================================
# 7. 消费者改用 load_metadata 契约
# ============================================================
class TestConsumersUseLoadMetadata:
    """消费者（inference/export/serving）改用 load_metadata 契约。

    通过源码检查验证消费者不再直接 json.load metadata.json，
    而是通过 load_metadata 进行版本协商。
    """

    @pytest.fixture
    def consumer_sources(self):
        """读取所有消费者的源码。"""
        root = Path(__file__).resolve().parents[1]
        return {
            "inference": (root / "senseframe" / "inference.py").read_text(encoding="utf-8"),
            "export": (root / "senseframe" / "export.py").read_text(encoding="utf-8"),
            "serving": (root / "senseframe" / "serving.py").read_text(encoding="utf-8"),
        }

    def test_inference_imports_load_metadata(self, consumer_sources):
        """inference.py 导入 load_metadata。"""
        assert "from .engine.metadata import load_metadata" in consumer_sources["inference"]

    def test_inference_calls_load_metadata(self, consumer_sources):
        """inference.py 调用 load_metadata 加载 metadata。"""
        assert "load_metadata(metadata_path)" in consumer_sources["inference"]

    def test_inference_no_direct_json_load_of_metadata(self, consumer_sources):
        """inference.py 不再直接 json.loads metadata_path 内容。"""
        # 排除 json.dumps（输出序列化）和注释中的 json.loads
        src = consumer_sources["inference"]
        # 查找 json.loads(metadata_path... 模式（直接读 metadata 文件）
        assert "json.loads(metadata_path" not in src, (
            "inference.py 仍直接 json.loads(metadata_path...)，应改用 load_metadata。"
        )

    def test_export_imports_load_metadata(self, consumer_sources):
        """export.py 导入 load_metadata。"""
        assert "from .engine.metadata import load_metadata" in consumer_sources["export"]

    def test_export_calls_load_metadata(self, consumer_sources):
        """export.py 调用 load_metadata 加载 metadata。"""
        assert "load_metadata(metadata_path)" in consumer_sources["export"]

    def test_export_no_direct_json_load_of_metadata(self, consumer_sources):
        """export.py 不再直接 json.load(f) 读取 metadata 文件。"""
        src = consumer_sources["export"]
        # export.py 仍有 json.dumps（写 export_manifest），只禁 json.load 读取 metadata
        # 查找 "with open(metadata_path" 模式
        assert 'with open(metadata_path' not in src, (
            "export.py 仍用 open(metadata_path) 直接读取，应改用 load_metadata。"
        )

    def test_serving_imports_load_metadata(self, consumer_sources):
        """serving.py 导入 load_metadata。"""
        assert "from .engine.metadata import load_metadata" in consumer_sources["serving"]

    def test_serving_calls_load_metadata(self, consumer_sources):
        """serving.py 调用 load_metadata 加载 metadata。"""
        assert "load_metadata(meta_path)" in consumer_sources["serving"]

    def test_serving_no_direct_json_load_of_metadata(self, consumer_sources):
        """serving.py 不再直接 json.loads meta_path 内容。"""
        src = consumer_sources["serving"]
        assert "json.loads(meta_path" not in src, (
            "serving.py 仍直接 json.loads(meta_path...)，应改用 load_metadata。"
        )


# ============================================================
# 8. ERROR_CODES 与 SKILL.md 错误码表闭环
# ============================================================
class TestErrorCodesComplete:
    """ERROR_CODES 字典与 SKILL.md 错误码表的闭环契约。"""

    def test_metadata_not_found_in_error_codes(self):
        """METADATA_NOT_FOUND 在 ERROR_CODES 字典中（消除幽灵错误码）。"""
        assert "METADATA_NOT_FOUND" in ERROR_CODES

    def test_metadata_version_error_in_error_codes(self):
        """METADATA_VERSION_ERROR 在 ERROR_CODES 字典中。"""
        assert "METADATA_VERSION_ERROR" in ERROR_CODES

    def test_skill_md_contains_metadata_error_codes(self):
        """SKILL.md 错误码表含 METADATA_NOT_FOUND 与 METADATA_VERSION_ERROR。"""
        skill_path = (
            Path(__file__).resolve().parents[1] / "SKILL.md"
        )
        skill_content = skill_path.read_text(encoding="utf-8")
        assert "METADATA_NOT_FOUND" in skill_content, (
            "SKILL.md 错误码表未包含 METADATA_NOT_FOUND。"
        )
        assert "METADATA_VERSION_ERROR" in skill_content, (
            "SKILL.md 错误码表未包含 METADATA_VERSION_ERROR。"
        )


# ============================================================
# 9. 版本比较工具
# ============================================================
class TestVersionComparison:
    """_version_tuple / _version_lt / _version_gt 版本比较工具。"""

    def test_version_tuple_parses_semver(self):
        """_version_tuple 将 '2.0.0' 转为 (2, 0, 0)。"""
        assert _version_tuple("2.0.0") == (2, 0, 0)
        assert _version_tuple("1.10.3") == (1, 10, 3)

    def test_version_lt_basic(self):
        """_version_lt 基本比较。"""
        assert _version_lt("1.0.0", "2.0.0")
        assert _version_lt("2.0.0", "2.0.1")
        assert _version_lt("2.0.0", "2.1.0")
        assert not _version_lt("2.0.0", "2.0.0")
        assert not _version_lt("3.0.0", "2.0.0")

    def test_version_gt_basic(self):
        """_version_gt 基本比较。"""
        assert _version_gt("2.0.0", "1.0.0")
        assert _version_gt("2.0.1", "2.0.0")
        assert _version_gt("2.1.0", "2.0.0")
        assert not _version_gt("2.0.0", "2.0.0")
        assert not _version_gt("1.0.0", "2.0.0")

    def test_version_comparison_handles_multi_digit(self):
        """版本比较正确处理多位数（如 2.10.0 > 2.9.0）。"""
        assert _version_gt("2.10.0", "2.9.0")
        assert _version_lt("2.9.0", "2.10.0")


# ============================================================
# 9.1 版本字符串格式校验（Review 补充 2026-07-18）
# ============================================================
class TestVersionFormatValidation:
    """非法版本字符串格式校验契约。

    Review 补充（2026-07-18）：
    - 非法版本（如 "abc"/"2.0.x"/"2.0"/"2.0.0.0"）应抛 MetadataVersionError
    - 而非 ValueError（确保 classify_error 命中 METADATA_VERSION_ERROR）
    """

    @pytest.mark.parametrize("invalid_version", [
        "abc",          # 非数字
        "2.0.x",        # 含非数字段
        "2.0",          # 两段（非三段）
        "2.0.0.0",      # 四段
        "2..0.0",       # 空段
        "v2.0.0",       # 含前缀
        "2.0.0-beta",   # 含预发布标识
    ])
    def test_get_metadata_version_rejects_invalid_format(self, invalid_version):
        """get_metadata_version 拒绝非法版本格式 → MetadataVersionError。"""
        with pytest.raises(MetadataVersionError):
            get_metadata_version({"schema_version": invalid_version})

    @pytest.mark.parametrize("invalid_version", ["abc", "2.0.x", "2.0", "2.0.0.0"])
    def test_migrate_metadata_rejects_invalid_format(self, invalid_version):
        """migrate_metadata 拒绝非法版本格式 → MetadataVersionError（非 ValueError）。"""
        with pytest.raises(MetadataVersionError):
            migrate_metadata({"schema_version": invalid_version})

    def test_get_metadata_version_accepts_valid_semver(self):
        """合法 semver 正常返回。"""
        assert get_metadata_version({"schema_version": "2.0.0"}) == "2.0.0"
        assert get_metadata_version({"schema_version": "1.0.0"}) == "1.0.0"
        assert get_metadata_version({"schema_version": "10.20.30"}) == "10.20.30"


# ============================================================
# 9.2 迁移链死循环防护（Review 补充 2026-07-18）
# ============================================================
class TestMigrationLoopProtection:
    """migrate_metadata 迁移链死循环防护契约。

    Review 补充（2026-07-18）：MIGRATIONS 含环时应抛 MetadataVersionError 而非死循环。
    """

    def test_migrate_with_cyclic_migrations_raises(self, monkeypatch):
        """MIGRATIONS 含环时抛 MetadataVersionError（迭代上限触发）。"""
        from senseframe.engine import metadata as metadata_module

        # 构造含环的 MIGRATIONS：1.0.0 → 2.0.0 → 1.0.0（环）
        # 同时提升 CURRENT 到 3.0.0 使环无法到达目标
        cyclic_migrations = {
            ("1.0.0", "2.0.0"): lambda d: d,
            ("2.0.0", "1.0.0"): lambda d: d,  # 环回 1.0.0
        }
        monkeypatch.setattr(metadata_module, "MIGRATIONS", cyclic_migrations)
        monkeypatch.setattr(metadata_module, "CURRENT_METADATA_VERSION", "3.0.0")

        with pytest.raises(MetadataVersionError) as exc_info:
            migrate_metadata({"model_id": "MLP"})  # LEGACY → 1.0.0
        assert "迭代超限" in str(exc_info.value) or "无迁移路径" in str(exc_info.value)


# ============================================================
# 10. make_metadata_skeleton 骨架
# ============================================================
class TestMakeMetadataSkeleton:
    """make_metadata_skeleton 骨架构造契约。"""

    def test_skeleton_contains_schema_version(self):
        """骨架 dict 含 schema_version 字段。"""
        skeleton = make_metadata_skeleton()
        assert "schema_version" in skeleton

    def test_skeleton_version_is_current(self):
        """骨架的 schema_version = CURRENT_METADATA_VERSION。"""
        assert make_metadata_skeleton()["schema_version"] == CURRENT_METADATA_VERSION


# ============================================================
# 11. _find_next_migration 迁移链查找
# ============================================================
class TestFindNextMigration:
    """_find_next_migration 迁移链查找契约。"""

    def test_find_legacy_to_current(self):
        """从 LEGACY_VERSION 出发，找到 CURRENT_METADATA_VERSION。"""
        assert _find_next_migration(LEGACY_VERSION) == CURRENT_METADATA_VERSION

    def test_find_current_returns_none(self):
        """从 CURRENT_METADATA_VERSION 出发，无后继迁移（已是最新）。"""
        assert _find_next_migration(CURRENT_METADATA_VERSION) is None

    def test_find_unknown_version_returns_none(self):
        """未知版本出发，无迁移路径。"""
        assert _find_next_migration("99.0.0") is None


# ============================================================
# 12. make_metadata_skeleton 接受可选字段（遗留问题 3 修复 2026-07-19）
# ============================================================
class TestMakeMetadataSkeletonAcceptsFields:
    """make_metadata_skeleton(**fields) 接受可选字段契约。

    遗留问题 3 修复（2026-07-19）：make_metadata_skeleton 增强为接受 **fields，
    pipeline.stage_export 改用此函数构造 metadata，消除"骨架函数仅测试使用"的死代码状态。
    """

    def test_skeleton_with_no_fields(self):
        """无字段时返回只含 schema_version 的 dict。"""
        skeleton = make_metadata_skeleton()
        assert skeleton == {"schema_version": CURRENT_METADATA_VERSION}

    def test_skeleton_with_string_field(self):
        """传入字符串字段被正确写入。"""
        skeleton = make_metadata_skeleton(model_id="MLP", dataset="UT_HAR")
        assert skeleton["model_id"] == "MLP"
        assert skeleton["dataset"] == "UT_HAR"
        assert skeleton["schema_version"] == CURRENT_METADATA_VERSION

    def test_skeleton_with_complex_field_types(self):
        """传入复杂类型字段（dict/list/None）被正确写入。"""
        skeleton = make_metadata_skeleton(
            config={"epochs": 200, "lr": 0.001},
            metrics=["acc", "f1"],
            manifest=None,
            label_map={"0": "walking", "1": "running"},
        )
        assert skeleton["config"] == {"epochs": 200, "lr": 0.001}
        assert skeleton["metrics"] == ["acc", "f1"]
        assert skeleton["manifest"] is None
        assert skeleton["label_map"] == {"0": "walking", "1": "running"}

    def test_skeleton_schema_version_not_overridable(self):
        """传入 schema_version 字段不覆盖 CURRENT_METADATA_VERSION。

        防御性契约：骨架函数是版本字段唯一真相源，调用方误传 schema_version 也被忽略。
        """
        skeleton = make_metadata_skeleton(schema_version="9.9.9")
        assert skeleton["schema_version"] == CURRENT_METADATA_VERSION

    def test_skeleton_simulates_pipeline_usage(self):
        """模拟 pipeline.stage_export 的实际使用模式（端到端契约）。"""
        # 模拟 pipeline.py stage_export 中的调用模式
        metadata = make_metadata_skeleton(
            model_id="ResNet18",
            dataset="UT_HAR_data",
            num_classes=7,
            learning_mode="supervised",
            config={"epochs": 100, "batch_size": 64},
            final_eval={"val_acc": 0.92, "test_acc": 0.89},
            created_at="2026-07-19T10:00:00",
        )
        # schema_version 自动注入
        assert metadata["schema_version"] == CURRENT_METADATA_VERSION
        # 业务字段正确写入
        assert metadata["model_id"] == "ResNet18"
        assert metadata["dataset"] == "UT_HAR_data"
        assert metadata["num_classes"] == 7
        assert metadata["final_eval"]["test_acc"] == 0.89


# ============================================================
# 13. _find_migration_path BFS 图搜索（遗留问题 2 修复 2026-07-19）
# ============================================================
class TestFindMigrationPath:
    """_find_migration_path BFS 图搜索契约。

    遗留问题 2 修复（2026-07-19）：替代 _find_next_migration 的线性查找，
    支持分支迁移图，找到最短可达路径，避免分支首个 dst 走到死端时的误报。
    visited 集合天然防环，含环迁移图不会死循环。
    """

    def test_path_empty_when_start_equals_target(self):
        """start == target 时返回空列表（无需迁移）。"""
        from senseframe.engine.metadata import _find_migration_path
        assert _find_migration_path("2.0.0", "2.0.0") == []

    def test_path_direct_edge(self):
        """直连边：LEGACY_VERSION → CURRENT_METADATA_VERSION。"""
        from senseframe.engine.metadata import _find_migration_path
        path = _find_migration_path(LEGACY_VERSION, CURRENT_METADATA_VERSION)
        assert path == [(LEGACY_VERSION, CURRENT_METADATA_VERSION)]

    def test_path_branch_dead_end_avoided(self, monkeypatch):
        """分支死端：1.0.0 → 2.0.0（死端）/ 1.0.0 → 1.5.0 → 3.0.0（活路），
        BFS 应选活路而非首先注册的死端路径。

        这正是遗留问题 2 的核心场景：旧线性 _find_next_migration 返回首先注册的
        dst=2.0.0，2.0.0 无后继，走到死端后误报"无迁移路径"——即使存在
        1.0.0 → 1.5.0 → 3.0.0 活路。BFS 同时探索所有分支，找到活路。
        """
        from senseframe.engine import metadata as metadata_module
        from senseframe.engine.metadata import _find_migration_path
        monkeypatch.setattr(metadata_module, "MIGRATIONS", {
            ("1.0.0", "2.0.0"): lambda d: d,  # 死端（2.0.0 无后继）
            ("1.0.0", "1.5.0"): lambda d: d,  # 活路起点
            ("1.5.0", "3.0.0"): lambda d: d,  # 活路终点
        })
        path = _find_migration_path("1.0.0", "3.0.0")
        assert path == [("1.0.0", "1.5.0"), ("1.5.0", "3.0.0")]

    def test_path_cyclic_graph_no_infinite_loop(self, monkeypatch):
        """含环迁移图不导致死循环（visited 集合防环）。"""
        from senseframe.engine import metadata as metadata_module
        from senseframe.engine.metadata import _find_migration_path
        monkeypatch.setattr(metadata_module, "MIGRATIONS", {
            ("1.0.0", "2.0.0"): lambda d: d,
            ("2.0.0", "1.0.0"): lambda d: d,  # 环回 1.0.0
            ("2.0.0", "3.0.0"): lambda d: d,  # 活路出口
        })
        path = _find_migration_path("1.0.0", "3.0.0")
        # BFS 找到 1.0.0 → 2.0.0 → 3.0.0，不进入环
        assert path == [("1.0.0", "2.0.0"), ("2.0.0", "3.0.0")]

    def test_path_unreachable_returns_none(self, monkeypatch):
        """目标不可达时返回 None（而非死循环或异常）。"""
        from senseframe.engine import metadata as metadata_module
        from senseframe.engine.metadata import _find_migration_path
        monkeypatch.setattr(metadata_module, "MIGRATIONS", {
            ("1.0.0", "2.0.0"): lambda d: d,  # 2.0.0 无后继，无法到 3.0.0
        })
        path = _find_migration_path("1.0.0", "3.0.0")
        assert path is None

    def test_path_chooses_shortest(self, monkeypatch):
        """多条可达路径时选最短（BFS 特性）。"""
        from senseframe.engine import metadata as metadata_module
        from senseframe.engine.metadata import _find_migration_path
        monkeypatch.setattr(metadata_module, "MIGRATIONS", {
            # 路径 A：1.0.0 → 2.0.0 → 3.0.0 → 4.0.0（3 步）
            ("1.0.0", "2.0.0"): lambda d: d,
            ("2.0.0", "3.0.0"): lambda d: d,
            ("3.0.0", "4.0.0"): lambda d: d,
            # 路径 B：1.0.0 → 5.0.0 → 4.0.0（2 步，更短）
            ("1.0.0", "5.0.0"): lambda d: d,
            ("5.0.0", "4.0.0"): lambda d: d,
        })
        path = _find_migration_path("1.0.0", "4.0.0")
        # BFS 找到最短路径 1.0.0 → 5.0.0 → 4.0.0
        assert len(path) == 2
        assert path[0] == ("1.0.0", "5.0.0")
        assert path[1] == ("5.0.0", "4.0.0")

    def test_path_unknown_start_returns_none(self, monkeypatch):
        """未知起始版本返回 None（无任何迁移从该版本出发）。"""
        from senseframe.engine import metadata as metadata_module
        from senseframe.engine.metadata import _find_migration_path
        monkeypatch.setattr(metadata_module, "MIGRATIONS", {
            ("1.0.0", "2.0.0"): lambda d: d,
        })
        path = _find_migration_path("99.0.0", "2.0.0")
        assert path is None
