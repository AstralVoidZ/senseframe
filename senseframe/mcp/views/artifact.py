"""Artifact 系列 view model（阶段 4.2 产物校验/列表/导出，FrozenModel 子类）。

设计文档 0.3 节 L4 产物契约 + 0.5 节错误信封：
- ArtifactDescriptorView：单个产物描述符的 JSON 契约（投影 ArtifactDescriptor 域对象）
- ArtifactVerifyResponse：senseframe_artifact_verify 响应（三重校验结果）
- ArtifactListView：senseframe_artifact_list 响应（cursor 分页）
- ArtifactExportResponse：senseframe_artifact_export 响应（zip/tar/manifest 导出元信息）

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
- ArtifactDescriptorView.from_domain 接收 ArtifactDescriptor 域对象，仅做字段投影
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from senseframe.mcp.views._base import FrozenModel, ViewError, _safe_get

__all__ = [
    "ArtifactDescriptorView",
    "ArtifactVerifyResponse",
    "ArtifactListView",
    "ArtifactExportResponse",
]


class ArtifactDescriptorView(FrozenModel):
    """单个产物的公共 JSON 契约。

    Attributes:
        name: 产物逻辑名（如 "model_weights"）。
        path: 相对 output_dir 的路径。
        kind: 产物类型（model/metadata/log/metrics/config/profile/feedback）。
        producer_stage: 生产者 stage 名（如 "stage_export"）。
        content_hash: 文件 SHA256 哈希。
        size_bytes: 文件大小（字节）。
        content_schema: 内容契约（字段名/类型）。
    """

    name: str
    path: str
    kind: str
    producer_stage: str
    content_hash: str
    size_bytes: int
    content_schema: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, desc: Any) -> ArtifactDescriptorView:
        """从 ArtifactDescriptor 域对象投影到 view。

        Args:
            desc: ArtifactDescriptor dataclass 实例（或同形 dict）。

        Returns:
            ArtifactDescriptorView 实例。

        Raises:
            ViewError: 输入不是 ArtifactDescriptor 或字段缺失。
        """
        try:
            return cls(
                name=_safe_get(desc, "name"),
                path=_safe_get(desc, "path"),
                kind=_safe_get(desc, "kind"),
                producer_stage=_safe_get(desc, "producer_stage"),
                content_hash=_safe_get(desc, "content_hash"),
                size_bytes=_safe_get(desc, "size_bytes", 0) or 0,
                content_schema=_safe_get(desc, "content_schema", {}) or {},
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(
                f"ArtifactDescriptorView.from_domain: invalid input: {exc}"
            ) from exc


class ArtifactVerifyResponse(FrozenModel):
    """``senseframe_artifact_verify`` 响应视图。

    三重校验结果（hash + manifest schema + 必填产物）聚合到一个信封。

    Attributes:
        run_id: 来源 manifest 的 run_id（加载失败时为空字符串）。
        output_dir: 解析后的 output_dir 绝对路径。
        hash_check: {产物名: hash 是否匹配}（路径逃逸的产物标记 False）。
        manifest_schema_missing: manifest 缺失的必填字段名列表（空表示完整）。
        missing_artifacts: 缺失的必填产物名列表（空表示齐全）。
        overall_ok: 综合判定（三重校验全通过）。
    """

    run_id: str
    output_dir: str
    hash_check: dict[str, bool] = Field(default_factory=dict)
    manifest_schema_missing: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    overall_ok: bool = False


class ArtifactListView(FrozenModel):
    """``senseframe_artifact_list`` 响应视图（cursor 分页）。

    Attributes:
        items: ArtifactDescriptorView 列表。
        next_cursor: 下一页 cursor（None 表示无更多）。
        total_count: 总数（不受分页影响）。
        limit: 钳制后的页大小。
        run_id: 当前 manifest 的 run_id（空字符串表示加载失败）。
    """

    items: list[ArtifactDescriptorView]
    next_cursor: Optional[str] = None
    total_count: int = 0
    limit: int = 50
    run_id: str = ""


class ArtifactExportResponse(FrozenModel):
    """``senseframe_artifact_export`` 响应视图。

    Attributes:
        output_path: 导出文件的绝对路径。
        format: 导出格式（zip/tar/manifest）。
        artifact_count: 包含的产物数。
        total_size_bytes: 源产物总大小（字节，不含导出文件压缩开销）。
        content_hash: 导出文件的 SHA256。
        run_id: 来源 manifest 的 run_id。
    """

    output_path: str
    format: str
    artifact_count: int
    total_size_bytes: int
    content_hash: str
    run_id: str = ""
