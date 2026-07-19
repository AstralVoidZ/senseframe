"""``senseframe_skill_*`` 工具组 view model（阶段 4.3 技能库 CRUD + 语义检索）。

设计文档 0.3 节技能库契约 + 0.5 节错误信封：
- SkillView：单个技能的 JSON 契约（投影 Skill 域对象）
- SkillSaveResponse：senseframe_skill_save 响应（含 validated / validation_errors / saved）
- SkillRemoveResponse：senseframe_skill_remove 响应（含 removed / force）
- SkillSearchResultView：单个检索结果（含 SkillView + 相关度 score）
- SkillSearchResponse：senseframe_skill_search 响应（含 query + items + total_count + top_k）

所有 view 必须继承 FrozenModel（extra='forbid' + frozen=True）。

分层不变量（AST 守卫测试钉死）：
- views/ 不 import orchestration / tools / storage / spec
- SkillView.from_domain 接收 Skill 域对象，仅做字段投影
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from senseframe.mcp.views._base import FrozenModel, ViewError, _safe_get

__all__ = [
    "SkillView",
    "SkillSaveResponse",
    "SkillRemoveResponse",
    "SkillSearchResultView",
    "SkillSearchResponse",
]


class SkillView(FrozenModel):
    """单个技能的公共 JSON 契约。

    Attributes:
        name: 技能名（唯一）。
        description: 技能描述（供检索用）。
        code: 技能代码内容（Python 源码）。
        tags: 标签列表（供检索用）。
        version: 语义版本号（默认 "1.0.0"）。
        created_at: 创建时间（ISO 格式字符串）。
        validated: 是否通过语法验证。
        validation_errors: 验证错误信息列表（空表示无错误）。
        depends_on: 依赖的其他技能名列表。
        source_path: 来源扩展文件路径（便于追溯）。
    """

    name: str
    description: str
    code: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""
    validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    source_path: str = ""

    @classmethod
    def from_domain(cls, skill: Any) -> SkillView:
        """从 Skill 域对象投影到 view。

        Args:
            skill: Skill dataclass 实例（或同形 dict）。

        Returns:
            SkillView 实例。

        Raises:
            ViewError: 输入不是 Skill 或字段缺失。
        """
        try:
            return cls(
                name=_safe_get(skill, "name"),
                description=_safe_get(skill, "description", ""),
                code=_safe_get(skill, "code", ""),
                tags=list(_safe_get(skill, "tags", []) or []),
                version=_safe_get(skill, "version", "1.0.0"),
                created_at=_safe_get(skill, "created_at", ""),
                validated=_safe_get(skill, "validated", False),
                validation_errors=list(
                    _safe_get(skill, "validation_errors", []) or []
                ),
                depends_on=list(_safe_get(skill, "depends_on", []) or []),
                source_path=_safe_get(skill, "source_path", ""),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ViewError(
                f"SkillView.from_domain: invalid input: {exc}"
            ) from exc


class SkillSaveResponse(FrozenModel):
    """``senseframe_skill_save`` 响应视图。

    Attributes:
        name: 技能名。
        version: 保存的版本号。
        validated: 是否通过语法验证。
        validation_errors: 验证错误信息列表（空表示无错误）。
        saved: 是否实际写入（False = 验证失败或重复跳过）。
    """

    name: str
    version: str
    validated: bool
    validation_errors: list[str] = Field(default_factory=list)
    saved: bool = False


class SkillRemoveResponse(FrozenModel):
    """``senseframe_skill_remove`` 响应视图。

    Attributes:
        name: 技能名。
        removed: 是否实际删除（True 表示已删除）。
        force: 是否强制删除（忽略依赖检查）。
    """

    name: str
    removed: bool
    force: bool = False


class SkillSearchResultView(FrozenModel):
    """单个检索结果（含相关度分数）。

    Attributes:
        skill: 技能 view。
        score: 相关度分数（0~1，越大越相关）。
    """

    skill: SkillView
    score: float


class SkillSearchResponse(FrozenModel):
    """``senseframe_skill_search`` 响应视图。

    Attributes:
        query: 查询字符串（原样回显）。
        items: 检索结果列表（按 score 降序）。
        total_count: 结果总数（受 top_k 限制）。
        top_k: 钳制后的 top_k 值（[1, 50]）。
    """

    query: str
    items: list[SkillSearchResultView]
    total_count: int = 0
    top_k: int = 5
