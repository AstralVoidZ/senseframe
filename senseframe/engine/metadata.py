"""
metadata.json 版本管理与迁移。

P3 演进（2026-07-18）：引入 schema_version 字段到 metadata.json，支持版本协商与迁移。

设计目标：
- **版本协商**：加载时自动识别 schema_version，旧版（无字段）走 fallback 升级路径
- **迁移链**：MIGRATIONS 注册表记录版本间迁移函数，未来版本变更只需注册新迁移
- **未来版本拒绝**：加载高于 CURRENT 的版本时抛 MetadataVersionError，提示用户升级 SenseFrame

版本号约定（语义化版本）：
- MAJOR：不兼容的 schema 变更（字段重命名/删除）
- MINOR：向后兼容的字段新增
- PATCH：bug 修复

消费者契约：
- 所有 metadata.json 读取应通过 `load_metadata(path)`，而非直接 `json.load()`
- `load_metadata` 返回的 dict 保证 `schema_version == CURRENT_METADATA_VERSION`
- 写入端（pipeline.stage_export）直接设置 `schema_version = CURRENT_METADATA_VERSION`

历史背景：
- 1.0.0（隐式）：初版 metadata.json，无 schema_version 字段
- 2.0.0（2026-07-18）：首个显式版本，字段结构与 1.0.0 兼容，仅标记版本
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .runner.errors import MetadataVersionError


# ============================================================
# 版本常量
# ============================================================
CURRENT_METADATA_VERSION = "2.0.0"
"""当前 SenseFrame 写出与读取的 metadata.json schema 版本。"""

LEGACY_VERSION = "1.0.0"
"""旧版 metadata.json 的隐式版本（无 schema_version 字段）。"""


# ============================================================
# 迁移函数链
# ============================================================
def _migrate_1_0_0_to_2_0_0(data: Dict[str, Any]) -> Dict[str, Any]:
    """1.0.0 → 2.0.0：首个显式版本标记，无字段迁移。

    旧版 metadata.json 无 schema_version 字段，所有字段与 2.0.0 兼容。
    此迁移为 no-op，仅记录版本号变更，为未来版本迁移建立机制。
    """
    return data


# 迁移注册表：(from_version, to_version) -> migration_fn
# 迁移函数签名：(data: dict) -> dict，返回迁移后的 data（不修改 schema_version 字段）
# migrate_metadata 会在迁移完成后统一设置 schema_version = CURRENT_METADATA_VERSION
MIGRATIONS: Dict[Tuple[str, str], Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    ("1.0.0", "2.0.0"): _migrate_1_0_0_to_2_0_0,
}


# ============================================================
# 版本比较工具
# ============================================================
# Review 修复（2026-07-18）：
# 1. 非法版本字符串（如 "abc"/"2.0.x"）抛 MetadataVersionError 而非 ValueError，
#    确保 classify_error 命中 METADATA_VERSION_ERROR 分支。
# 2. 版本段数归一化：用 zip_longest 填 0 比较，"2.0" 与 "2.0.0" 语义相等。
#    避免短前缀误判（(2,0) < (2,0,0) 的元组比较陷阱）。
import re

_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
"""严格 semver 校验：仅接受 MAJOR.MINOR.PATCH 三段数字。"""


def _validate_version(v: str) -> str:
    """校验版本字符串格式，非法时抛 MetadataVersionError。

    要求严格 semver 格式：MAJOR.MINOR.PATCH（三段数字，点分隔）。
    拒绝 "2.0"（两段）、"2.0.x"（非数字）、"2.0.0.0"（四段）、"" （空）。
    """
    if not isinstance(v, str) or not _SEMVER_PATTERN.match(v):
        raise MetadataVersionError(
            f"metadata schema_version '{v}' 格式非法，要求 MAJOR.MINOR.PATCH 三段数字"
            f"（如 '2.0.0'）。"
        )
    return v


def _version_tuple(v: str) -> Tuple[int, int, int]:
    """将 '2.0.0' 转为 (2, 0, 0)，便于语义比较。

    调用前应先通过 _validate_version 校验格式。
    """
    parts = v.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _version_lt(a: str, b: str) -> bool:
    return _version_tuple(a) < _version_tuple(b)


def _version_gt(a: str, b: str) -> bool:
    return _version_tuple(a) > _version_tuple(b)


# ============================================================
# 公共 API
# ============================================================
def get_metadata_version(data: Dict[str, Any]) -> str:
    """获取 metadata 的 schema_version。

    无 schema_version 字段视为 LEGACY_VERSION（1.0.0），向后兼容旧版 metadata.json。
    有字段时校验格式（非法字符串抛 MetadataVersionError）。
    """
    v = data.get("schema_version")
    if not v:
        return LEGACY_VERSION
    # Review 修复：校验显式版本格式，非法字符串（如 "abc"/"2.0"）抛 MetadataVersionError
    return _validate_version(v)


def migrate_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """应用迁移链，将旧版 metadata 升级到 CURRENT_METADATA_VERSION。

    策略：
    - 源版本 == CURRENT：直接返回（幂等）
    - 源版本 > CURRENT：抛 MetadataVersionError（用户需升级 SenseFrame）
    - 源版本 < CURRENT：逐版本应用 MIGRATIONS 中的迁移函数

    迁移完成后，data["schema_version"] 被设置为 CURRENT_METADATA_VERSION。

    Review 修复（2026-07-18）：
    - 版本格式校验前置（_validate_version），非法字符串抛 MetadataVersionError
    - 添加迭代上限（len(MIGRATIONS) + 1），防止 MIGRATIONS 含环时死循环

    Args:
        data: 原始 metadata dict（可能来自旧版或新版 metadata.json）

    Returns:
        升级到 CURRENT_METADATA_VERSION 的 metadata dict

    Raises:
        MetadataVersionError: 版本格式非法 / 高于 CURRENT / 无迁移路径 / 迁移链含环
    """
    current = get_metadata_version(data)

    if current == CURRENT_METADATA_VERSION:
        return data

    if _version_gt(current, CURRENT_METADATA_VERSION):
        raise MetadataVersionError(
            f"metadata schema_version {current} 高于当前 SenseFrame 支持的版本 "
            f"{CURRENT_METADATA_VERSION}。请升级 SenseFrame，或使用旧版 metadata.json。"
        )

    # Review 修复（2026-07-18）：用 BFS 图搜索找最短迁移路径，解决线性查找的分支死端问题。
    # 原 _find_next_migration 线性查找返回首先注册的 dst，若分支首个 dst 走到死端，
    # 即使存在其他分支可达 CURRENT 也会误报"无迁移路径"。BFS 保证找到最短可达路径。
    path = _find_migration_path(current, CURRENT_METADATA_VERSION)
    if path is None:
        raise MetadataVersionError(
            f"无迁移路径从 schema_version {current} 到 {CURRENT_METADATA_VERSION}。"
            f"可能 metadata.json 损坏或来自不兼容的分支。"
        )
    # 沿路径逐版本应用迁移函数
    for src, dst in path:
        migration_fn = MIGRATIONS[(src, dst)]
        data = migration_fn(data)
    data["schema_version"] = CURRENT_METADATA_VERSION
    return data


def _find_migration_path(
    start: str,
    target: str,
) -> Optional[List[Tuple[str, str]]]:
    """BFS 搜索从 start 到 target 的最短迁移路径。

    Review 修复（2026-07-18）：替代 _find_next_migration 的线性查找，
    支持分支迁移图（同一源版本多个目标版本），返回最短可达路径。

    Args:
        start: 起始版本
        target: 目标版本

    Returns:
        迁移步骤列表 [(src1, dst1), (src2, dst2), ...]，None 表示无路径。
        空列表表示 start == target（无需迁移）。
    """
    if start == target:
        return []
    # BFS：queue 存 (当前版本, 路径)
    from collections import deque
    queue: deque = deque([(start, [])])
    visited: set = {start}
    while queue:
        current, path = queue.popleft()
        # 查找所有从 current 出发的迁移
        for (src, dst), fn in MIGRATIONS.items():
            if src == current and dst not in visited:
                new_path = path + [(src, dst)]
                if dst == target:
                    return new_path
                visited.add(dst)
                queue.append((dst, new_path))
    return None


def _find_next_migration(from_version: str) -> Optional[str]:
    """查找从 from_version 出发的下一个迁移目标版本。

    迁移链是线性的：每个版本最多有一个后继版本。若存在多个后继（分支），
    返回首先注册的那个（MIGRATIONS 字典插入顺序）。
    """
    for (src, dst) in MIGRATIONS.keys():
        if src == from_version:
            return dst
    return None


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """加载 metadata.json + 自动版本协商 + 迁移。

    所有 metadata.json 读取应通过此函数，而非直接 `json.load()`。
    返回的 dict 保证 `schema_version == CURRENT_METADATA_VERSION`。

    Args:
        path: metadata.json 文件路径

    Returns:
        迁移到 CURRENT_METADATA_VERSION 的 metadata dict

    Raises:
        FileNotFoundError: 文件不存在
        MetadataVersionError: 版本不兼容且无迁移路径
        json.JSONDecodeError: JSON 解析失败
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return migrate_metadata(data)


def make_metadata_skeleton(**fields: Any) -> Dict[str, Any]:
    """构造含 schema_version 的 metadata 骨架 dict。

    供 stage_export 写入 metadata.json 时调用，确保版本字段始终存在。
    Review 修复（2026-07-18）：接受可选字段，消除"骨架函数仅测试使用"的死代码状态。
    遗留问题 3 修复（2026-07-19）：schema_version 不可被覆盖——
    骨架函数是版本字段的唯一真相源，调用方误传 schema_version 也被强制覆盖为
    CURRENT_METADATA_VERSION。pipeline.stage_export 已改用此函数构造 metadata。

    Args:
        **fields: 额外的 metadata 字段（如 model_id="MLP", dataset="UT_HAR"）
            schema_version 字段会被忽略（始终为 CURRENT_METADATA_VERSION）

    Returns:
        dict，含 schema_version=CURRENT_METADATA_VERSION + 所有传入字段（除 schema_version）

    Usage:
        # 简单骨架（仅 schema_version）
        skeleton = make_metadata_skeleton()

        # 带字段的骨架（pipeline.stage_export 使用）
        metadata = make_metadata_skeleton(
            model_id="MLP",
            dataset="UT_HAR",
            num_classes=14,
        )
    """
    result: Dict[str, Any] = dict(fields)
    # 强制覆盖 schema_version（骨架函数是版本字段唯一真相源）
    result["schema_version"] = CURRENT_METADATA_VERSION
    return result
