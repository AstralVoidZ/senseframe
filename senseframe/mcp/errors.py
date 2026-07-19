"""MCP 协议层错误 + ML 业务错误映射。

设计文档 0.5 节错误信封分层：
- 协议层错误（9 子类，对齐 pipeflow）：本模块独立定义
- ML 业务错误（继承 SenseFrameError）：从 senseframe.engine.runner.errors 导入后映射到 category

协议层错误覆盖 MCP 服务器自身的状态机/校验/限流等失败，
ML 业务错误覆盖训练/数据/场景等 SenseFrame 领域失败。
两类错误统一通过 ToolErrorResponse.envelope_from 路由到 7 类 category。
"""

from __future__ import annotations

# ============================================================
# 协议层错误（9 子类，对齐 pipeflow 协议栈）
# ============================================================


class MCPProtocolError(Exception):
    """MCP 协议层错误基类。

    所有协议层错误（PipelineRun 状态机 / 校验 / 限流等）继承此类，
    便于 ToolErrorResponse.envelope_from 通过 isinstance 路由。
    """

    error_code: str = "MCP_PROTOCOL_ERROR"


class PipelineNotFound(MCPProtocolError):
    """PipelineRun 不存在（run_id 未注册或已被清除）。"""

    error_code = "PIPELINE_NOT_FOUND"


class StageNotFound(MCPProtocolError):
    """Stage 不存在（stage_id 未在 pipeline 中注册）。"""

    error_code = "STAGE_NOT_FOUND"


class IllegalTransition(MCPProtocolError):
    """PipelineRun 状态机非法转移（如 Pending → Succeeded 跳过 Running）。"""

    error_code = "ILLEGAL_TRANSITION"


class StageOrderViolation(MCPProtocolError):
    """Stage 顺序违规（前置 stage 未完成时推进后续 stage）。"""

    error_code = "STAGE_ORDER_VIOLATION"


class MaxRetriesExceeded(MCPProtocolError):
    """Stage 重试次数超过上限。"""

    error_code = "MAX_RETRIES_EXCEEDED"


class SchemaValidationError(MCPProtocolError):
    """输入/输出 schema 校验失败（extra='forbid' 触发等）。"""

    error_code = "SCHEMA_VALIDATION_ERROR"


class TimeBudgetExceeded(MCPProtocolError):
    """时间预算超限（pipeline/stage 运行时间超过声明上限）。"""

    error_code = "TIME_BUDGET_EXCEEDED"


class InvalidPathError(MCPProtocolError):
    """路径非法（cursor/URI 解析失败或路径形似 CLI flag）。"""

    error_code = "INVALID_PATH_ERROR"


class RateLimitExceeded(MCPProtocolError):
    """令牌桶限流触发（per-tool calls_per_minute 超限）。"""

    error_code = "RATE_LIMIT_EXCEEDED"


# ============================================================
# 分页相关错误（cursor 解析 + filter fingerprint 校验）
# ============================================================
# 对齐 pipeflow CursorFilterMismatch / InvalidCursor（§6.18）。
# 继承 InvalidPathError 以便 ToolErrorResponse._CATEGORY_BY_EXC
# 仍能路由到 config category（InvalidPathError 已在映射中）。


class CursorFilterMismatch(InvalidPathError):
    """Cursor 的 filter_fingerprint 与当前请求的 filter 不一致。

    客户端尝试用旧 filter 的 cursor 配合新 filter 请求，必须重启
    from cursor=None。
    """

    error_code = "CURSOR_FILTER_MISMATCH"


class InvalidCursor(InvalidPathError):
    """Cursor base64 解码失败或缺少必需字段（§6.18）。"""

    error_code = "INVALID_CURSOR"


# ============================================================
# Artifact 产物域错误（阶段 4.2 新增）
# ============================================================
# 7 类 category 中的 artifact 类别映射：所有 artifact 异常继承 ArtifactError，
# 由 ToolErrorResponse._CATEGORY_BY_EXC 统一路由到 category="artifact"。


class ArtifactError(MCPProtocolError):
    """产物校验/操作错误的公共基类。"""

    error_code = "ARTIFACT_ERROR"


class ManifestNotFoundError(ArtifactError):
    """manifest.json 文件不存在。"""

    error_code = "MANIFEST_NOT_FOUND"


class ManifestSchemaError(ArtifactError):
    """manifest schema 校验失败（缺失必填字段）。"""

    error_code = "MANIFEST_SCHEMA_ERROR"


class ArtifactHashMismatchError(ArtifactError):
    """产物 hash 校验失败（文件内容与 manifest 记录不一致）。"""

    error_code = "ARTIFACT_HASH_MISMATCH"


class MissingRequiredArtifactError(ArtifactError):
    """必填产物缺失（config/metadata/training_log）。"""

    error_code = "MISSING_REQUIRED_ARTIFACT"


class ArtifactPathEscapeError(ArtifactError):
    """产物路径逃逸 output_dir（安全校验失败）。"""

    error_code = "ARTIFACT_PATH_ESCAPE"


class UnsupportedExportFormatError(ArtifactError):
    """不支持的导出格式。"""

    error_code = "UNSUPPORTED_EXPORT_FORMAT"


# ============================================================
# Skill 技能域错误（阶段 4.3 新增）
# ============================================================
# 7 类 category 中无 "skill"（严格不可增减），故所有 skill 异常继承 SkillError，
# 由 ToolErrorResponse._CATEGORY_BY_EXC 统一路由到 category="internal"。
# 客户端通过 error_code 字段区分具体 skill 错误类型。


class SkillError(MCPProtocolError):
    """技能库操作的公共基类。"""

    error_code = "SKILL_ERROR"


class SkillNotFoundError(SkillError):
    """技能不存在。"""

    error_code = "SKILL_NOT_FOUND"


class SkillHasDependentsError(SkillError):
    """技能被其他技能依赖，无法删除。"""

    error_code = "SKILL_HAS_DEPENDENTS"


class SkillValidationError(SkillError):
    """技能代码验证失败（SyntaxError）。"""

    error_code = "SKILL_VALIDATION_ERROR"


# ============================================================
# ML 业务错误（从 senseframe.engine.runner.errors 导入，不重复定义）
# ============================================================
# 设计文档 0.5 节：ML 业务错误继承 SenseFrameError，已存在 error_code 属性。
# 此处仅做重新导出，便于 ToolErrorResponse._CATEGORY_BY_EXC 集中引用。

from senseframe.engine.runner.errors import (  # noqa: E402
    CheckpointError,
    ConfigValidationError,
    DataCorruptedError,
    DataNotFoundError,
    DatasetNotSupportedError,
    MetadataVersionError,
    ModelBuildError,
    ModelNotSupportedError,
    OOMError,
    PreflightError,
    SaveError,
    SceneNotRegisteredError,
    SenseFrameError,
    TrainingError,
)

__all__ = [
    # 协议层错误基类
    "MCPProtocolError",
    # 协议层错误 9 子类
    "PipelineNotFound",
    "StageNotFound",
    "IllegalTransition",
    "StageOrderViolation",
    "MaxRetriesExceeded",
    "SchemaValidationError",
    "TimeBudgetExceeded",
    "InvalidPathError",
    "RateLimitExceeded",
    "CursorFilterMismatch",
    "InvalidCursor",
    # Artifact 产物域错误（阶段 4.2）
    "ArtifactError",
    "ManifestNotFoundError",
    "ManifestSchemaError",
    "ArtifactHashMismatchError",
    "MissingRequiredArtifactError",
    "ArtifactPathEscapeError",
    "UnsupportedExportFormatError",
    # Skill 技能域错误（阶段 4.3）
    "SkillError",
    "SkillNotFoundError",
    "SkillHasDependentsError",
    "SkillValidationError",
    # ML 业务错误（re-export）
    "SenseFrameError",
    "SceneNotRegisteredError",
    "DatasetNotSupportedError",
    "ModelNotSupportedError",
    "DataNotFoundError",
    "DataCorruptedError",
    "OOMError",
    "CheckpointError",
    "PreflightError",
    "TrainingError",
    "ModelBuildError",
    "SaveError",
    "ConfigValidationError",
    "MetadataVersionError",
]
