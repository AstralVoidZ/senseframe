"""RFC-004 方案 G：公共溯源 API。

包含：
- load_manifest：加载训练产物清单
- verify_artifacts：校验产物 hash
- verify_artifacts_recursive：递归校验多 trial 目录
- verify_manifest_schema：校验 manifest schema 完整性
- verify_artifacts_full：完整校验（hash + schema + 必填产物）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..artifacts import (
    ArtifactManifest,
    verify_artifacts as _verify_artifacts,
    verify_artifacts_recursive as _verify_artifacts_recursive,
    verify_manifest_schema as _verify_manifest_schema,
    verify_artifacts_full as _verify_artifacts_full,
)


def load_manifest(output_dir) -> ArtifactManifest:
    """加载训练产物清单，供训练后分析。

    Args:
        output_dir: 训练输出目录（含 manifest.json）或 manifest.json 路径

    Returns:
        ArtifactManifest 实例
    """
    return ArtifactManifest.load(Path(output_dir))


def verify_artifacts(output_dir) -> Dict[str, bool]:
    """校验 output_dir 中所有产物的 hash，检测是否被篡改。

    Args:
        output_dir: 训练输出目录（含 manifest.json）

    Returns:
        {产物名: hash 是否匹配}
    """
    return _verify_artifacts(Path(output_dir))


def verify_artifacts_recursive(output_dir, max_depth: int = 3) -> Dict[str, Dict[str, bool]]:
    """递归校验 output_dir 及子目录中所有 manifest.json 的产物 hash（P3-4）。

    用于 HPO 多 trial 场景：output_dir/ 下可能有 trial_0/、trial_1/ 等子目录，
    每个子目录有自己的 manifest.json。单 run 场景请用 verify_artifacts。

    Args:
        output_dir: 根输出目录
        max_depth: 最大递归深度

    Returns:
        {子目录相对路径: {产物名: hash 是否匹配}}，根目录用 "." 表示
    """
    return _verify_artifacts_recursive(Path(output_dir), max_depth=max_depth)


def verify_manifest_schema(manifest_path) -> List[str]:
    """校验 manifest.json schema 完整性，返回缺失字段列表（P5 P3-11）。

    Args:
        manifest_path: manifest.json 路径

    Returns:
        缺失的 manifest 字段名列表（空列表表示完整）
    """
    return _verify_manifest_schema(Path(manifest_path))


def verify_artifacts_full(output_dir) -> Dict[str, Any]:
    """完整校验：hash + schema + 必填产物（P5 P3-11）。

    Args:
        output_dir: 训练输出目录

    Returns:
        {
            "hash_check": {产物名: bool},
            "manifest_schema_missing": [缺失字段],
            "missing_artifacts": [缺失产物名],
        }
    """
    return _verify_artifacts_full(Path(output_dir))
