"""统一错误信封 view model。

所有 tool wrapper 通过 `tools/_errors.to_tool_error` 路由异常，
构造 `ToolErrorResponse` 信封 `(code, message, category)` 并以 `ToolError`
形式 surface 给 MCP 客户端。信封形状是 tool 错误的公共契约；
`category` 字段让客户端按类别路由错误（pipeline / study / scene / artifact
/ config / search / internal）。

设计文档 0.5 节定义 7 类 category，严格不可增减。
"""

from __future__ import annotations

from typing import Literal

from senseframe.mcp.errors import (
    ArtifactError,
    CheckpointError,
    ConfigValidationError,
    DataCorruptedError,
    DataNotFoundError,
    DatasetNotSupportedError,
    IllegalTransition,
    InvalidPathError,
    MaxRetriesExceeded,
    MetadataVersionError,
    ModelBuildError,
    ModelNotSupportedError,
    OOMError,
    PipelineNotFound,
    PreflightError,
    RateLimitExceeded,
    SaveError,
    SchemaValidationError,
    SceneNotRegisteredError,
    SkillError,
    SkillHasDependentsError,
    SkillNotFoundError,
    SkillValidationError,
    StageNotFound,
    StageOrderViolation,
    TimeBudgetExceeded,
    TrainingError,
)
from senseframe.mcp.views._base import FrozenModel

__all__ = ["ToolErrorResponse", "CategoryT"]

# 7 类 category，严格按设计文档 0.5 节定义，不可增减。
CategoryT = Literal[
    "pipeline",
    "study",
    "scene",
    "artifact",
    "config",
    "search",
    "internal",
]

# 异常类 → category 映射。顺序敏感（isinstance 按顺序匹配，具体类在前）。
# 协议层错误（9 子类）→ pipeline / config / internal
# ML 业务错误（继承 SenseFrameError）→ scene / config / internal
# study / search 类别已覆盖：KeyError → study（StudyManager.ask/tell
# 在 study_id / trial_id 不存在时抛 KeyError，统一路由到 study category）。
_CATEGORY_BY_EXC: tuple[tuple[type[BaseException], CategoryT], ...] = (
    # --- 协议层错误 ---
    (PipelineNotFound, "pipeline"),
    (StageNotFound, "pipeline"),
    (IllegalTransition, "pipeline"),
    (StageOrderViolation, "pipeline"),
    (MaxRetriesExceeded, "pipeline"),
    (SchemaValidationError, "config"),
    (TimeBudgetExceeded, "pipeline"),
    (InvalidPathError, "config"),
    (RateLimitExceeded, "internal"),  # 限流归入 internal
    # --- artifact 类别（阶段 4.2 新增）---
    # 所有产物域异常继承 ArtifactError，统一路由到 artifact category。
    # 放在 KeyError 之前以确保 artifact 异常优先匹配（不进入 study 兜底）。
    (ArtifactError, "artifact"),
    # --- skill 类别（阶段 4.3 新增）---
    # 7 类 category 中无 "skill"，故所有 skill 异常统一路由到 internal category。
    # 放在 KeyError 之前以确保 skill 异常优先匹配（不进入 study 兜底）。
    # 客户端通过 error_code 字段区分具体 skill 错误类型。
    (SkillNotFoundError, "internal"),
    (SkillHasDependentsError, "internal"),
    (SkillValidationError, "internal"),
    (SkillError, "internal"),  # 兜底：未知 skill 异常也归入 internal
    # --- study 类别（阶段 3 新增）---
    # StudyManager.ask/tell/get_study 在 study_id / trial_id 不存在时抛 KeyError，
    # 统一路由到 study category（早于 internal 兜底匹配）。
    (KeyError, "study"),
    # --- ML 业务错误 ---
    (SceneNotRegisteredError, "scene"),
    (DatasetNotSupportedError, "scene"),
    (ModelNotSupportedError, "scene"),
    (DataNotFoundError, "scene"),
    (DataCorruptedError, "scene"),
    (OOMError, "internal"),
    (CheckpointError, "internal"),
    (PreflightError, "config"),
    (TrainingError, "internal"),
    (ModelBuildError, "internal"),
    (SaveError, "internal"),
    (ConfigValidationError, "config"),
    (MetadataVersionError, "config"),
)


class ToolErrorResponse(FrozenModel):
    """公共错误信封 (code, message, category)。

    Fields:
        code: 异常类名（如 `PipelineNotFound`）。
        message: `str(exc)` — 已由 orchestrator 脱敏（无 SQL 片段/路径/堆栈）。
        category: 7 类 category 之一，客户端按类别路由错误恢复策略。
    """

    code: str
    message: str
    category: CategoryT

    @classmethod
    def envelope_from(cls, exc: BaseException) -> ToolErrorResponse:
        """根据异常类型路由到正确 category。

        - 已知异常类（在 _CATEGORY_BY_EXC 中）→ 对应 category
        - 未知异常类 → category='internal'（脱敏兜底）

        Args:
            exc: 捕获的异常实例。

        Returns:
            ToolErrorResponse 实例，可序列化为 JSON 信封。
        """
        category: CategoryT = "internal"
        for exc_cls, cat in _CATEGORY_BY_EXC:
            if isinstance(exc, exc_cls):
                category = cat
                break
        return cls(
            code=type(exc).__name__,
            message=str(exc),
            category=category,
        )
