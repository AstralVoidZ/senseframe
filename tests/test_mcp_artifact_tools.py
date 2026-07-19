"""阶段 4.2 MCP artifact tool 测试。

覆盖：
- senseframe_artifact_verify：三重校验（hash + schema + 必填产物）+ recursive 模式
- senseframe_artifact_list：cursor 分页 + filter + run_id
- senseframe_artifact_export：zip / tar / manifest 三种格式 + 指定 artifact_names
- 异常路由：ManifestNotFoundError / UnsupportedExportFormatError → category="artifact"
- cursor filter mismatch → CursorFilterMismatch

设计文档 0.3 节 L4 产物契约 + 0.5 节错误信封。
"""

from __future__ import annotations

import json
import tarfile
import zipfile

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from senseframe.engine.runner.artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    sha256_file,
    sha256_str,
)
from senseframe.mcp.tools.artifact import (
    senseframe_artifact_export,
    senseframe_artifact_list,
    senseframe_artifact_verify,
)
from senseframe.mcp.views.artifact import (
    ArtifactDescriptorView,
    ArtifactExportResponse,
    ArtifactListView,
    ArtifactVerifyResponse,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_output_dir(tmp_path):
    """创建含 manifest.json + 产物文件的示例 output_dir。

    包含 4 个产物（model_weights / metadata / config / training_log），全部通过校验。
    manifest.json 包含所有必填字段（run_id / created_at / senseframe_version /
    pipeline_version / config_hash / data_hash）。
    """
    # 1. 创建产物文件
    (tmp_path / "model.pth").write_bytes(b"fake model weights")
    (tmp_path / "metadata.json").write_text(json.dumps({"run_id": "test"}))
    (tmp_path / "config.yaml").write_text("scene: wifi_csi")
    (tmp_path / "training_log.jsonl").write_text('{"epoch": 1}\n')

    # 2. 创建 manifest.json
    artifacts = []
    for name, path, kind in [
        ("model_weights", "model.pth", "model"),
        ("metadata", "metadata.json", "metadata"),
        ("config", "config.yaml", "config"),
        ("training_log", "training_log.jsonl", "log"),
    ]:
        file_path = tmp_path / path
        artifacts.append(
            ArtifactDescriptor(
                name=name,
                path=path,
                kind=kind,
                producer_stage="stage_export",
                content_hash=sha256_file(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )

    manifest = ArtifactManifest(
        run_id="test-run-001",
        created_at="2026-07-19T00:00:00",
        senseframe_version="0.1.0",
        pipeline_version="1",
        config_hash=sha256_str("config"),
        data_hash=sha256_str("data"),
        artifacts=artifacts,
    )
    manifest.save(tmp_path)
    return tmp_path


# ============================================================
# senseframe_artifact_verify 测试
# ============================================================


class TestArtifactVerify:
    """senseframe_artifact_verify 测试。"""

    @pytest.mark.asyncio
    async def test_verify_returns_correct_structure(self, sample_output_dir):
        """verify 返回 ArtifactVerifyResponse 含 hash_check/schema_missing/missing_artifacts。"""
        result = await senseframe_artifact_verify(str(sample_output_dir))
        assert isinstance(result, ArtifactVerifyResponse)
        assert result.run_id == "test-run-001"
        assert "model_weights" in result.hash_check
        assert "metadata" in result.hash_check
        assert "config" in result.hash_check
        assert "training_log" in result.hash_check
        # 全部产物完整 → overall_ok=True
        assert result.overall_ok is True
        assert result.manifest_schema_missing == []
        assert result.missing_artifacts == []

    @pytest.mark.asyncio
    async def test_verify_detects_hash_mismatch(self, sample_output_dir):
        """产物文件被修改后 hash 校验失败。"""
        (sample_output_dir / "model.pth").write_bytes(b"tampered")
        result = await senseframe_artifact_verify(str(sample_output_dir))
        assert result.hash_check["model_weights"] is False
        assert result.overall_ok is False

    @pytest.mark.asyncio
    async def test_verify_missing_required_artifact(self, sample_output_dir):
        """缺失必填产物（config）时 overall_ok=False。"""
        (sample_output_dir / "config.yaml").unlink()
        result = await senseframe_artifact_verify(str(sample_output_dir))
        # config 缺失会反映在 hash_check 中（文件不存在 → False）
        assert result.hash_check["config"] is False
        assert result.overall_ok is False

    @pytest.mark.asyncio
    async def test_verify_manifest_not_found_raises_tool_error(self, tmp_path):
        """output_dir 不存在时 → ToolError（category=artifact）。

        senseframe_artifact_verify 内部抛 ManifestNotFoundError，
        to_tool_error 桥接为 ToolError，payload category="artifact"。
        """
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_verify(str(tmp_path / "nonexistent"))
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "artifact"
        assert payload["code"] == "ManifestNotFoundError"

    @pytest.mark.asyncio
    async def test_verify_recursive_mode(self, sample_output_dir):
        """recursive=True 时也返回 ArtifactVerifyResponse。"""
        result = await senseframe_artifact_verify(
            str(sample_output_dir), recursive=True
        )
        assert isinstance(result, ArtifactVerifyResponse)
        assert result.run_id == "test-run-001"
        # 根目录 "." 的 hash 校验应通过
        assert "model_weights" in result.hash_check

    @pytest.mark.asyncio
    async def test_verify_output_dir_is_resolved_to_absolute(self, sample_output_dir):
        """返回的 output_dir 是绝对路径。"""
        # 用相对路径调用（cwd 不一定是 tmp_path，所以只用绝对路径验证）
        result = await senseframe_artifact_verify(str(sample_output_dir))
        from pathlib import Path

        assert Path(result.output_dir).is_absolute()
        assert Path(result.output_dir) == sample_output_dir.resolve()


# ============================================================
# senseframe_artifact_list 测试
# ============================================================


class TestArtifactList:
    """senseframe_artifact_list 测试。"""

    @pytest.mark.asyncio
    async def test_list_returns_all_artifacts(self, sample_output_dir):
        """list 返回 manifest 中所有产物。"""
        result = await senseframe_artifact_list(str(sample_output_dir))
        assert isinstance(result, ArtifactListView)
        assert result.total_count == 4
        assert len(result.items) == 4
        assert result.run_id == "test-run-001"
        # items 是 ArtifactDescriptorView 实例
        for item in result.items:
            assert isinstance(item, ArtifactDescriptorView)
            assert item.name
            assert item.path
            assert item.kind

    @pytest.mark.asyncio
    async def test_list_filter_by_kind(self, sample_output_dir):
        """filter_dict 按 kind 过滤。"""
        result = await senseframe_artifact_list(
            str(sample_output_dir), filter_dict={"kind": "model"}
        )
        assert result.total_count == 1
        assert len(result.items) == 1
        assert result.items[0].name == "model_weights"

    @pytest.mark.asyncio
    async def test_list_filter_by_producer_stage(self, sample_output_dir):
        """filter_dict 按 producer_stage 过滤。"""
        result = await senseframe_artifact_list(
            str(sample_output_dir), filter_dict={"producer_stage": "stage_export"}
        )
        assert result.total_count == 4
        assert len(result.items) == 4

    @pytest.mark.asyncio
    async def test_list_cursor_pagination(self, sample_output_dir):
        """cursor 分页：limit=2 时返回前 2 个 + next_cursor。"""
        result = await senseframe_artifact_list(str(sample_output_dir), limit=2)
        assert len(result.items) == 2
        assert result.next_cursor is not None
        assert result.total_count == 4
        assert result.limit == 2

        # 翻第二页
        result2 = await senseframe_artifact_list(
            str(sample_output_dir), cursor=result.next_cursor, limit=2
        )
        assert len(result2.items) == 2
        # 第二页的 name 都应大于第一页最后一个 name（字典序）
        first_page_last = result.items[-1].name
        for item in result2.items:
            assert item.name > first_page_last

        # 第三页应无更多
        assert result2.next_cursor is None or result2.next_cursor

    @pytest.mark.asyncio
    async def test_list_cursor_filter_mismatch_raises_tool_error(
        self, sample_output_dir
    ):
        """cursor 的 filter 与当前 filter 不一致 → ToolError（category=config）。

        CursorFilterMismatch 继承 InvalidPathError，路由到 config category。
        """
        from senseframe.mcp.pagination.cursor import encode_cursor

        # 用 kind=model 的 filter 编码 cursor
        cursor = encode_cursor("model_weights", {"kind": "model"})
        # 用 kind=metadata 的 filter 解码 → CursorFilterMismatch → ToolError
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_list(
                str(sample_output_dir),
                cursor=cursor,
                filter_dict={"kind": "metadata"},
            )
        payload = json.loads(str(exc_info.value))
        assert payload["code"] == "CursorFilterMismatch"

    @pytest.mark.asyncio
    async def test_list_manifest_not_found_raises_tool_error(self, tmp_path):
        """output_dir 不存在 → ToolError（category=artifact）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_list(str(tmp_path / "nonexistent"))
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "artifact"
        assert payload["code"] == "ManifestNotFoundError"

    @pytest.mark.asyncio
    async def test_list_limit_clamped_to_min(self, sample_output_dir):
        """limit=0 被钳制到 MIN_LIMIT=1。"""
        result = await senseframe_artifact_list(str(sample_output_dir), limit=0)
        assert result.limit == 1
        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_list_returns_sorted_by_name(self, sample_output_dir):
        """items 按 name 字典序排列。"""
        result = await senseframe_artifact_list(str(sample_output_dir))
        names = [item.name for item in result.items]
        assert names == sorted(names)


# ============================================================
# senseframe_artifact_export 测试
# ============================================================


class TestArtifactExport:
    """senseframe_artifact_export 测试。"""

    @pytest.mark.asyncio
    async def test_export_zip_format(self, sample_output_dir):
        """zip 格式导出。"""
        result = await senseframe_artifact_export(
            str(sample_output_dir), format="zip"
        )
        assert isinstance(result, ArtifactExportResponse)
        assert result.format == "zip"
        assert result.artifact_count == 4
        from pathlib import Path

        assert Path(result.output_path).exists()
        assert zipfile.is_zipfile(result.output_path)
        # zip 内应含 manifest.json + 4 个产物文件
        with zipfile.ZipFile(result.output_path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "model.pth" in names
            assert "metadata.json" in names
            assert "config.yaml" in names
            assert "training_log.jsonl" in names

    @pytest.mark.asyncio
    async def test_export_tar_format(self, sample_output_dir):
        """tar 格式导出。"""
        result = await senseframe_artifact_export(
            str(sample_output_dir), format="tar"
        )
        assert result.format == "tar"
        from pathlib import Path

        assert Path(result.output_path).exists()
        assert tarfile.is_tarfile(result.output_path)
        # tar 内应含 manifest.json + 4 个产物文件
        with tarfile.open(result.output_path) as tf:
            names = tf.getnames()
            assert "manifest.json" in names
            assert "model.pth" in names

    @pytest.mark.asyncio
    async def test_export_manifest_format(self, sample_output_dir):
        """manifest 格式仅导出 manifest.json（无产物文件）。"""
        result = await senseframe_artifact_export(
            str(sample_output_dir), format="manifest"
        )
        assert result.format == "manifest"
        from pathlib import Path

        assert Path(result.output_path).exists()
        # manifest 格式不包含产物文件，artifact_count 仍记录源产物数
        assert result.artifact_count == 4
        # 导出的文件应是合法 JSON
        data = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
        assert data["run_id"] == "test-run-001"
        assert len(data["artifacts"]) == 4

    @pytest.mark.asyncio
    async def test_export_unsupported_format_raises_tool_error(
        self, sample_output_dir
    ):
        """不支持的格式 → ToolError（category=artifact）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_export(
                str(sample_output_dir), format="rar"
            )
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "artifact"
        assert payload["code"] == "UnsupportedExportFormatError"

    @pytest.mark.asyncio
    async def test_export_specific_artifacts(self, sample_output_dir):
        """指定 artifact_names 只导出部分产物。"""
        result = await senseframe_artifact_export(
            str(sample_output_dir),
            artifact_names=["model_weights", "config"],
            format="zip",
        )
        assert result.artifact_count == 2
        # zip 内应只含 manifest.json + 2 个指定产物文件
        with zipfile.ZipFile(result.output_path) as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "model.pth" in names
            assert "config.yaml" in names
            # 未指定的产物不应在 zip 内
            assert "metadata.json" not in names
            assert "training_log.jsonl" not in names

    @pytest.mark.asyncio
    async def test_export_returns_content_hash(self, sample_output_dir):
        """返回的 content_hash 是导出文件的 SHA256。"""
        from senseframe.engine.runner.artifacts import sha256_file

        result = await senseframe_artifact_export(
            str(sample_output_dir), format="zip"
        )
        expected_hash = sha256_file(result.output_path)
        assert result.content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_export_manifest_not_found_raises_tool_error(self, tmp_path):
        """output_dir 不存在 → ToolError（category=artifact）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_export(str(tmp_path / "nonexistent"))
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "artifact"
        assert payload["code"] == "ManifestNotFoundError"

    @pytest.mark.asyncio
    async def test_export_no_artifacts_after_filter_raises_tool_error(
        self, sample_output_dir
    ):
        """artifact_names 不匹配任何产物 → ToolError（MissingRequiredArtifactError）。"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_artifact_export(
                str(sample_output_dir),
                artifact_names=["non_existent_artifact"],
                format="zip",
            )
        payload = json.loads(str(exc_info.value))
        assert payload["category"] == "artifact"
        assert payload["code"] == "MissingRequiredArtifactError"

    @pytest.mark.asyncio
    async def test_export_total_size_bytes_matches_artifacts(self, sample_output_dir):
        """total_size_bytes 等于所有源产物 size_bytes 之和。"""
        from senseframe.engine.runner.pipeline.artifacts_api import load_manifest

        manifest = load_manifest(sample_output_dir)
        expected_size = sum(a.size_bytes for a in manifest.artifacts)

        result = await senseframe_artifact_export(
            str(sample_output_dir), format="manifest"
        )
        assert result.total_size_bytes == expected_size


# ============================================================
# Artifact 异常路由测试
# ============================================================


class TestArtifactErrorRouting:
    """artifact 异常路由到 category='artifact' 测试。"""

    def test_manifest_not_found_routes_to_artifact(self):
        """ManifestNotFoundError → category='artifact'。"""
        from senseframe.mcp.errors import ManifestNotFoundError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            ManifestNotFoundError("manifest.json not found")
        )
        assert envelope.category == "artifact"
        assert envelope.code == "ManifestNotFoundError"

    def test_unsupported_export_format_routes_to_artifact(self):
        """UnsupportedExportFormatError → category='artifact'。"""
        from senseframe.mcp.errors import UnsupportedExportFormatError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            UnsupportedExportFormatError("format 'rar' not supported")
        )
        assert envelope.category == "artifact"
        assert envelope.code == "UnsupportedExportFormatError"

    def test_artifact_hash_mismatch_routes_to_artifact(self):
        """ArtifactHashMismatchError → category='artifact'。"""
        from senseframe.mcp.errors import ArtifactHashMismatchError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            ArtifactHashMismatchError("hash mismatch on model.pth")
        )
        assert envelope.category == "artifact"

    def test_missing_required_artifact_routes_to_artifact(self):
        """MissingRequiredArtifactError → category='artifact'。"""
        from senseframe.mcp.errors import MissingRequiredArtifactError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            MissingRequiredArtifactError("config artifact missing")
        )
        assert envelope.category == "artifact"

    def test_artifact_path_escape_routes_to_artifact(self):
        """ArtifactPathEscapeError → category='artifact'。"""
        from senseframe.mcp.errors import ArtifactPathEscapeError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            ArtifactPathEscapeError("path escapes output_dir")
        )
        assert envelope.category == "artifact"

    def test_manifest_schema_error_routes_to_artifact(self):
        """ManifestSchemaError → category='artifact'。"""
        from senseframe.mcp.errors import ManifestSchemaError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            ManifestSchemaError("missing required field: run_id")
        )
        assert envelope.category == "artifact"

    def test_artifact_error_base_class_routes_to_artifact(self):
        """ArtifactError 基类 → category='artifact'（子类全覆盖）。"""
        from senseframe.mcp.errors import ArtifactError
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        envelope = ToolErrorResponse.envelope_from(
            ArtifactError("generic artifact failure")
        )
        assert envelope.category == "artifact"
