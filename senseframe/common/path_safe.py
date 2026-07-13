"""路径安全工具：统一路径解析 + 穿越防护。

设计原则：
1. 外部路径（manifest/配置/CLI 输入）一律视为不可信
2. 拼接后必须校验解析结果仍在期望基目录内
3. 拒绝存储绝对路径或外部路径到 manifest（避免 pathlib 拼接语义被利用）

pathlib 语义陷阱：
    Path("/a/b") / "/etc/passwd"  → PosixPath('/etc/passwd')
    Path("/a/b") / "../../etc"    → PosixPath('/a/b/../../etc') → resolve → /etc
绝对路径作为右操作数时，pathlib 会丢弃左侧 base，这是路径穿越的根源。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# 标识符白名单：字母/数字/下划线/连字符，首字符必须字母或数字
# 用于 model_id / dataset / scene_name 等构造路径的标识符
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


def resolve_under(base: PathLike, target: PathLike) -> Path:
    """解析 target 相对于 base 的绝对路径，逃逸时抛 ValueError。

    - target 绝对路径：必须仍位于 base 下，否则拒绝
    - target 相对路径：base / target 后 resolve，必须仍位于 base 下
    - 解析后路径不在 base 内：抛 ValueError（路径穿越）

    Args:
        base: 基目录（可信，会 resolve 为绝对路径）
        target: 目标路径（不可信，可能含 .. 或绝对路径）

    Returns:
        解析后的绝对路径（保证在 base 内）

    Raises:
        ValueError: target 解析后逃逸 base
    """
    base_resolved = Path(base).resolve()
    target_path = Path(target)

    if target_path.is_absolute():
        resolved = target_path.resolve()
    else:
        resolved = (base_resolved / target_path).resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"path escapes base directory: target={target!r} "
            f"resolved={resolved} base={base_resolved}"
        )
    return resolved


def safe_relative_path(base: PathLike, target: PathLike) -> str:
    """返回 target 相对 base 的相对路径字符串（用于 manifest 存储）。

    逃逸时抛 ValueError（拒绝存储绝对路径或外部路径）。
    存储相对路径避免 pathlib 拼接时绝对路径右操作数丢弃 base 的陷阱。

    Args:
        base: 基目录（可信）
        target: 目标路径（不可信）

    Returns:
        相对路径字符串（如 "runs/xxx/metrics.csv"）

    Raises:
        ValueError: target 不在 base 内
    """
    resolved = resolve_under(base, target)
    base_resolved = Path(base).resolve()
    return str(resolved.relative_to(base_resolved))


def sanitize_path_component(name: str) -> str:
    """清洗用于构造路径的标识符（如 model_id/dataset/scene_name）。

    拒绝含路径分隔符或 .. 的输入，避免路径注入。

    Args:
        name: 标识符（来自配置或 CLI）

    Returns:
        清洗后的标识符

    Raises:
        ValueError: name 含路径分隔符 / .. / 为空 / 含非法字符
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"name must be non-empty string, got: {name!r}")
    # 显式拒绝路径分隔符与 ..
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"name contains path separator or traversal: {name!r}")
    # 白名单校验
    if not _SAFE_COMPONENT_RE.match(name):
        raise ValueError(
            f"name contains illegal characters (allowed: A-Z a-z 0-9 _ -): {name!r}"
        )
    return name


def is_path_component_safe(name: str) -> bool:
    """检查标识符是否安全（不抛异常，返回 bool）。

    用于配置加载时的非破坏性校验。

    Args:
        name: 标识符

    Returns:
        True 安全，False 不安全
    """
    try:
        sanitize_path_component(name)
        return True
    except ValueError:
        return False


__all__ = [
    "resolve_under",
    "safe_relative_path",
    "sanitize_path_component",
    "is_path_component_safe",
]
