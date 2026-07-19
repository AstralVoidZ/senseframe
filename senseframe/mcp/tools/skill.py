"""``senseframe_skill_*`` 工具组（阶段 4.3 技能库 CRUD + 语义检索）。

设计文档 0.3 节定义 4 个 skill tool：
- senseframe_skill_save   — 保存技能（含代码验证 + 版本管理）
- senseframe_skill_get    — 获取技能详情
- senseframe_skill_search — 语义检索技能（hash-based / sentence-transformers）
- senseframe_skill_remove — 删除技能（含依赖检查）

每个 tool 是 async 函数，签名参考 tools/study.py：
- 使用 MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...)) 包装
- 异常通过 to_tool_error(exc) 转换为 ToolError
- 返回值是 FrozenModel 子类（在 views/skill.py 中定义）

ToolAnnotations 矩阵（设计文档 0.4 节）：
- save:   false/false/false/true   （写库、非幂等、开放世界）
- get:    true/false/true/false    （只读、幂等、封闭世界）
- search: true/false/true/false    （只读、幂等、封闭世界）
- remove: false/true/false/true    （破坏性、非幂等、开放世界）

关键设计：直接复用 senseframe.skills 模块的 SkillLibrary 单例
（save_skill / load_skill / search_skills_with_scores / get_skill_library），
不重新实现技能库逻辑。
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from senseframe.mcp.config import rate_limit as _rate_limit_cfg
from senseframe.mcp.errors import (
    SkillHasDependentsError,
    SkillNotFoundError,
)
from senseframe.mcp.middleware import (
    MiddlewareStack,
    RateLimitMiddleware,
    RequestIdMiddleware,
    TokenBucketLimiter,
)
from senseframe.mcp.tools._errors import to_tool_error
from senseframe.mcp.views.skill import (
    SkillRemoveResponse,
    SkillSaveResponse,
    SkillSearchResponse,
    SkillSearchResultView,
    SkillView,
)
from senseframe.skills import (
    get_skill_library,
    load_skill as _load_skill,
    save_skill as _save_skill,
    search_skills_with_scores,
)

logger = logging.getLogger(__name__)

__all__ = [
    "senseframe_skill_save",
    "senseframe_skill_get",
    "senseframe_skill_search",
    "senseframe_skill_remove",
    "_skill_stack",
]

# MiddlewareStack：每个 tool 调用经过 RequestId + RateLimit 中间件
_skill_stack = MiddlewareStack(
    RequestIdMiddleware(),
    RateLimitMiddleware(limiter=TokenBucketLimiter(_rate_limit_cfg())),
)


# ============================================================
# Tool handlers
# ============================================================


async def senseframe_skill_save(
    name: str,
    code: str,
    description: str = "",
    tags: list[str] | None = None,
    source_path: str = "",
    version: str = "1.0.0",
    ctx: Context[Any, Any, Any] | None = None,
) -> SkillSaveResponse:
    """保存技能（含代码验证 + 版本管理）。

    Args:
        name: 技能名（唯一）。
        code: 技能代码内容（Python 源码，会通过 compile 验证语法）。
        description: 技能描述（供检索用）。
        tags: 标签列表（供检索用）。
        source_path: 来源扩展文件路径（便于追溯）。
        version: 语义版本号（默认 "1.0.0"）。
        ctx: MCP Context。

    Returns:
        SkillSaveResponse（含 validated / validation_errors / saved）。

    Notes:
        - 验证失败时返回 saved=False，不抛异常（让 Agent 通过 validation_errors 修正代码）。
        - 验证成功时 saved=True，并刷新 SkillLibrary 中的技能记录。
    """
    if ctx:
        await ctx.info(
            f"senseframe_skill_save name={name} version={version}"
        )
    try:
        async with _skill_stack.instrument("senseframe_skill_save", ctx):
            # 调用现有 API（返回 bool，False = 验证失败）
            success = _save_skill(
                name=name,
                code=code,
                description=description,
                tags=tags or [],
                source_path=source_path,
                version=version,
            )

            if not success:
                # 验证失败：构造 validation_errors（重新 compile 提取错误信息）
                validation_errors: list[str] = []
                try:
                    compile(code, f"<skill:{name}>", "exec")
                except SyntaxError as se:
                    validation_errors.append(
                        f"SyntaxError: {se.msg} (line {se.lineno})"
                    )
                return SkillSaveResponse(
                    name=name,
                    version=version,
                    validated=False,
                    validation_errors=validation_errors,
                    saved=False,
                )

            # 验证成功：读取实际保存的 skill 获取 validated 状态
            skill = _load_skill(name)
            validated = skill.validated if skill is not None else True

            return SkillSaveResponse(
                name=name,
                version=version,
                validated=validated,
                validation_errors=[],
                saved=True,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_skill_save failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_skill_get(
    name: str,
    version: str | None = None,
    ctx: Context[Any, Any, Any] | None = None,
) -> SkillView:
    """获取技能详情。

    Args:
        name: 技能名。
        version: 指定版本号（None 时返回最新版本）。
        ctx: MCP Context。

    Returns:
        SkillView（含 name / description / code / tags / version / validated 等）。

    Raises:
        SkillNotFoundError: 技能不存在（被 to_tool_error 桥接为 ToolError）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_skill_get name={name} version={version}"
        )
    try:
        async with _skill_stack.instrument("senseframe_skill_get", ctx):
            skill = _load_skill(name, version=version)
            if skill is None:
                version_suffix = f" version={version}" if version else ""
                raise SkillNotFoundError(
                    f"Skill '{name}'{version_suffix} not found"
                )
            return SkillView.from_domain(skill)
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_skill_get failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_skill_search(
    query: str,
    top_k: int = 5,
    ctx: Context[Any, Any, Any] | None = None,
) -> SkillSearchResponse:
    """语义检索技能（hash-based / sentence-transformers）。

    Args:
        query: 查询字符串（描述要找的技能能力）。
        top_k: 返回前 K 个最相关技能（钳制到 [1, 50]）。
        ctx: MCP Context。

    Returns:
        SkillSearchResponse（含 query / items / total_count / top_k）。
        items 按 score 降序，仅含 score > 0 的项。
    """
    if ctx:
        await ctx.info(
            f"senseframe_skill_search query={query!r} top_k={top_k}"
        )
    try:
        async with _skill_stack.instrument("senseframe_skill_search", ctx):
            # 钳制 top_k 到 [1, 50]
            k = max(1, min(int(top_k), 50))
            results = search_skills_with_scores(query, top_k=k)
            items = [
                SkillSearchResultView(
                    skill=SkillView.from_domain(skill),
                    score=score,
                )
                for skill, score in results
            ]
            return SkillSearchResponse(
                query=query,
                items=items,
                total_count=len(items),
                top_k=k,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_skill_search failed: {exc}")
        raise to_tool_error(exc)


async def senseframe_skill_remove(
    name: str,
    force: bool = False,
    ctx: Context[Any, Any, Any] | None = None,
) -> SkillRemoveResponse:
    """删除技能（含依赖检查）。

    Args:
        name: 技能名。
        force: True 时强制删除（忽略依赖），False 时若有依赖则拒绝。
        ctx: MCP Context。

    Returns:
        SkillRemoveResponse（含 name / removed / force）。

    Raises:
        SkillNotFoundError: 技能不存在。
        SkillHasDependentsError: 有依赖且 force=False（被 ValueError 触发）。
    """
    if ctx:
        await ctx.info(
            f"senseframe_skill_remove name={name} force={force}"
        )
    try:
        async with _skill_stack.instrument("senseframe_skill_remove", ctx):
            lib = get_skill_library()
            try:
                success = lib.remove(name, force=force)
            except ValueError as exc:
                # SkillLibrary.remove 在有依赖且 force=False 时抛 ValueError
                # （错误信息含 "depended on by"）
                if "depended on by" in str(exc):
                    raise SkillHasDependentsError(str(exc))
                raise

            if not success:
                # remove 返回 False 表示 name 不存在
                raise SkillNotFoundError(f"Skill '{name}' not found")

            return SkillRemoveResponse(
                name=name,
                removed=True,
                force=force,
            )
    except Exception as exc:
        if ctx:
            await ctx.error(f"senseframe_skill_remove failed: {exc}")
        raise to_tool_error(exc)
