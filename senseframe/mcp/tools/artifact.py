"""``senseframe_artifact_*`` 工具组（L4 产物校验/列表/导出）。

设计文档 0.3 节定义 3 个 artifact tool：
- senseframe_artifact_verify — 校验产物完整性（hash + schema + 必填产物三重校验）
- senseframe_artifact_list   — 列出 manifest 中的产物（cursor 分页 + filter）
- senseframe_artifact_export — 导出产物为 zip / tar / manifest

每个 tool 是 async 函数，签名参考 tools/study.py：
- 使用 MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...)) 包装
- 异常通过 to_tool_error(exc) 转换为 ToolError
- 返回值是 FrozenModel 子类（在 views/artifact.py 中定义）

ToolAnnotations 矩阵（设计文档 0.4 节）：
- verify: true/false/true/false   （只读、幂等、封闭世界）
- list:   true/false/true/false   （只读、幂等、封闭世界）
- export: false/false/false/true  （写盘、非幂等、开放世界）

关键设计：直接复用 senseframe.engine.runner.pipeline.artifacts_api 中的
薄包装层（load_manifest / verify_artifacts_full / verify_artifacts_recursive）
与 senseframe.engine.runner.artifacts.sha256_file，不重新实现 hash 计算逻辑。
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context

from senseframe.engine.runner.artifacts import sha256_file
from senseframe.engine.runner.pipeline.artifacts_api import (
    load_manifest,
    verify_artifacts_full,
    verify_artifacts_recursive,
)
from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.errors import (
    ManifestNotFoundError,
    MissingRequiredArtifactError,
    UnsupportedExportFormatError,
)
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from senseframe.mcp.pagination.cursor import (
    assert_fingerprint_matches,
    encode_cursor,
)
from senseframe.mcp.pagination.page import clamp_limit
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.artifact import (
    ArtifactDescriptorView,
    ArtifactExportResponse,
    ArtifactListView,
    ArtifactVerifyResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_artifact_verify",
    "senseframe_artifact_list",
    "senseframe_artifact_export",
    "_artifact_stack",
]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_artifact_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)

# 必填产物种类（P2-5：与 artifacts.py 的 _REQUIRED_ARTIFACT_KINDS 保持一致）
# 改用 kind 校验替代 name 校验，避免 "metadata" vs "model_metadata" 误报
_REQUIRED_ARTIFACT_KINDS = frozenset({"config", "metadata", "log"})
# 支持的导出格式
_SUPPORTED_EXPORT_FORMATS = frozenset({"zip", "tar", "manifest"})


# ============================================================
# 辅助函数
# ============================================================


def _matches_artifact_filter(desc: Any, filter_dict: dict[str, Any] | None) -> bool:
    """简单等值过滤：filter_dict 中的所有 (k, v) 必须在 desc 上匹配。

    支持的 filter key（与 ArtifactDescriptor 字段对齐）：
    - kind / producer_stage / name / path 等任意字段名
    """
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        actual = getattr(desc, key, None)
        if actual is None and isinstance(desc, dict):
            actual = desc.get(key)
        if actual != value:
            return False
    return True


def _load_run_id(out_path: Path) -> str:
    """尝试加载 manifest 的 run_id，失败时返回空字符串。

    ArtifactVerifyResponse 在 manifest schema 不完整时仍要返回（不能直接 raise），
    所以 run_id 加载失败需静默降级。
    """
    try:
        manifest = load_manifest(out_path)
        return manifest.run_id
    except Exception:
        return ""


# ============================================================
# Tool handlers
# ============================================================


async def senseframe_artifact_verify(
    output_dir: str,
    recursive: bool = False,
    ctx: Context[Any, Any, Any] | None = None,
) -> ArtifactVerifyResponse:
    """校验产物完整性（hash + schema + 必填产物三重校验）。

    Args:
        output_dir: 训练输出目录（含 manifest.json）。
        recursive: True 时递归校验子目录的 manifest.json（HPO 多 trial 场景）。
            根目录用 "." 表示，子目录用相对路径表示。
        ctx: MCP Context。

    Returns:
        ArtifactVerifyResponse（含 hash_check / manifest_schema_missing /
        missing_artifacts / overall_ok）。

    Raises:
        ManifestNotFoundError: output_dir 不存在。
        ToolError: 其他异常经 to_tool_error 桥接（artifact category）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_artifact_verify output_dir={output_dir} recursive={recursive}"
        )
    try:
        async with _artifact_stack.instrument("senseframe_artifact_verify", ctx):
            out_path = Path(output_dir).resolve()
            if not out_path.exists():
                raise ManifestNotFoundError(f"output_dir not found: {out_path}")

            if recursive:
                # 递归校验：扫描子目录 manifest.json（max_depth=3）
                recursive_result = verify_artifacts_recursive(out_path, max_depth=3)
                # 取根目录 "." 的校验结果；若根目录无 manifest 则用空 dict
                root_check = recursive_result.get(".", {})
                # recursive 模式仅做 hash 校验，schema/missing 用 full 补充
                full = verify_artifacts_full(out_path)
                hash_check = root_check if root_check else full.get("hash_check", {})
            else:
                full = verify_artifacts_full(out_path)
                hash_check = full.get("hash_check", {})

            manifest_schema_missing = full.get("manifest_schema_missing", [])
            missing_artifacts = full.get("missing_artifacts", [])

            run_id = _load_run_id(out_path)

            # 综合判定：所有 hash 通过 + schema 完整 + 必填产物齐全
            overall_ok = (
                (all(hash_check.values()) if hash_check else False)
                and not manifest_schema_missing
                and not missing_artifacts
            )

            return ArtifactVerifyResponse(
                run_id=run_id,
                output_dir=str(out_path),
                hash_check=hash_check,
                manifest_schema_missing=manifest_schema_missing,
                missing_artifacts=missing_artifacts,
                overall_ok=overall_ok,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_artifact_verify failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_artifact_list(
    output_dir: str,
    cursor: str | None = None,
    limit: int = 50,
    filter_dict: dict[str, Any] | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> ArtifactListView:
    """列出指定 manifest 的产物（cursor 分页 + filter）。

    Args:
        output_dir: 训练输出目录（含 manifest.json）。
        cursor: 不透明 cursor（来自上一次 list 的 next_cursor），None 表示首页。
        limit: 页大小（钳制到 [1, 200]）。
        filter_dict: 过滤字典（支持 {"kind": "model"} 等值过滤）。
        ctx: MCP Context。

    Returns:
        ArtifactListView（含 items + next_cursor + total_count + limit + run_id）。

    Raises:
        ManifestNotFoundError: output_dir 不存在或 manifest.json 缺失。
        CursorFilterMismatch: cursor 的 filter 与当前 filter 不一致。
        ToolError: 其他异常经 to_tool_error 桥接。
    """
    if ctx:
        await ctx.info(
            f"senseframe_artifact_list output_dir={output_dir} "
            f"cursor={'set' if cursor else 'none'} limit={limit}"
        )
    try:
        async with _artifact_stack.instrument("senseframe_artifact_list", ctx):
            out_path = Path(output_dir).resolve()
            if not out_path.exists():
                raise ManifestNotFoundError(f"output_dir not found: {out_path}")

            manifest = load_manifest(out_path)
            run_id = manifest.run_id

            # 过滤
            all_artifacts = list(manifest.artifacts or [])
            filtered = [a for a in all_artifacts if _matches_artifact_filter(a, filter_dict)]
            total_count = len(filtered)

            # 排序（按 name 字典序，确保 cursor 稳定）
            filtered.sort(key=lambda a: a.name)

            # cursor 分页
            clamped_limit = clamp_limit(limit)
            last_id = assert_fingerprint_matches(cursor, filter_dict)
            if last_id is not None:
                filtered = [a for a in filtered if a.name > last_id]

            # limit+1 技巧
            peek = filtered[: clamped_limit + 1]
            has_more = len(peek) > clamped_limit
            page_items = peek[:clamped_limit]

            next_cursor: str | None = None
            if has_more and page_items:
                next_cursor = encode_cursor(page_items[-1].name, filter_dict)

            items = [ArtifactDescriptorView.from_domain(a) for a in page_items]

            return ArtifactListView(
                items=items,
                next_cursor=next_cursor,
                total_count=total_count,
                limit=clamped_limit,
                run_id=run_id,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_artifact_list failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_artifact_export(
    output_dir: str,
    artifact_names: list[str] | None = None,
    format: str = "zip",
    ctx: Context[Any, Any, Any] | None = None,
) -> ArtifactExportResponse:
    """导出产物（zip / tar / manifest 三种格式）。

    Args:
        output_dir: 训练输出目录（含 manifest.json）。
        artifact_names: 要导出的产物名列表（None 表示全部）。
        format: 导出格式（zip / tar / manifest）。
            - zip：将 manifest.json + 选中产物文件压缩成 .zip
            - tar：将 manifest.json + 选中产物文件打包成 .tar（无压缩）
            - manifest：仅导出 manifest.json（不含产物文件，artifact_count 仍记录源产物数）
        ctx: MCP Context。

    Returns:
        ArtifactExportResponse（含 output_path / format / artifact_count /
        total_size_bytes / content_hash / run_id）。

    Raises:
        UnsupportedExportFormatError: format 不在 {zip, tar, manifest} 中。
        ManifestNotFoundError: output_dir 不存在或 manifest.json 缺失。
        MissingRequiredArtifactError: 筛选后无任何产物可导出。
        ToolError: 其他异常经 to_tool_error 桥接。
    """
    if ctx:
        await ctx.info(
            f"senseframe_artifact_export output_dir={output_dir} format={format}"
        )
    try:
        async with _artifact_stack.instrument("senseframe_artifact_export", ctx):
            if format not in _SUPPORTED_EXPORT_FORMATS:
                raise UnsupportedExportFormatError(
                    f"Unsupported format: {format}. "
                    f"Supported: {sorted(_SUPPORTED_EXPORT_FORMATS)}"
                )

            out_path = Path(output_dir).resolve()
            if not out_path.exists():
                raise ManifestNotFoundError(f"output_dir not found: {out_path}")

            manifest = load_manifest(out_path)
            run_id = manifest.run_id

            # 筛选要导出的产物
            all_artifacts = list(manifest.artifacts or [])
            if artifact_names:
                wanted = set(artifact_names)
                export_artifacts = [a for a in all_artifacts if a.name in wanted]
            else:
                export_artifacts = list(all_artifacts)

            if not export_artifacts:
                raise MissingRequiredArtifactError(
                    f"No artifacts to export (filter={artifact_names})"
                )

            # 计算源产物总大小（不含导出文件压缩开销）
            total_size = sum(a.size_bytes for a in export_artifacts)

            # 导出路径：output_dir/exports/<filename>
            export_dir = out_path / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            manifest_json = json.dumps(
                manifest.to_dict(), ensure_ascii=False, indent=2, default=str
            )

            if format == "manifest":
                # 仅导出 manifest.json（无产物文件）
                export_path = export_dir / f"manifest_{run_id}.json"
                export_path.write_text(manifest_json, encoding="utf-8")
            elif format == "zip":
                export_path = export_dir / f"artifacts_{run_id}.zip"
                # S1 修复：resolve out_path 一次，用于路径穿越校验
                out_path_resolved = out_path.resolve()
                with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("manifest.json", manifest_json)
                    for a in export_artifacts:
                        file_path = out_path / a.path
                        # S1 修复：校验解析后路径仍在 out_path 之内，防路径穿越
                        try:
                            file_path.resolve().relative_to(out_path_resolved)
                        except ValueError:
                            logger.warning(
                                "artifact_export: skipping path escaping output_dir: %s",
                                a.path,
                            )
                            continue
                        if file_path.exists():
                            # arcname 用相对路径，避免绝对路径前缀
                            zf.write(file_path, arcname=a.path)
            elif format == "tar":
                export_path = export_dir / f"artifacts_{run_id}.tar"
                # S1 修复：resolve out_path 一次，用于路径穿越校验
                out_path_resolved = out_path.resolve()
                with tarfile.open(export_path, "w") as tf:
                    manifest_bytes = manifest_json.encode("utf-8")
                    info = tarfile.TarInfo(name="manifest.json")
                    info.size = len(manifest_bytes)
                    tf.addfile(info, io.BytesIO(manifest_bytes))
                    for a in export_artifacts:
                        file_path = out_path / a.path
                        # S1 修复：校验解析后路径仍在 out_path 之内，防路径穿越
                        try:
                            file_path.resolve().relative_to(out_path_resolved)
                        except ValueError:
                            logger.warning(
                                "artifact_export: skipping path escaping output_dir: %s",
                                a.path,
                            )
                            continue
                        if file_path.exists():
                            tf.add(file_path, arcname=a.path)

            # 计算导出文件 hash
            content_hash = sha256_file(export_path)

            return ArtifactExportResponse(
                output_path=str(export_path),
                format=format,
                artifact_count=len(export_artifacts),
                total_size_bytes=total_size,
                content_hash=content_hash,
                run_id=run_id,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_artifact_export failed: {exc}")
        raise to_tool_error(exc)
