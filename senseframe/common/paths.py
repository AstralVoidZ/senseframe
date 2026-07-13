"""项目路径常量:单一数据源。

PROJECT_ROOT 仅用于框架自身资源(如内置 configs、模板),不用于数据路径推导。
数据路径由调用者显式提供,框架不猜测、不探测、不 fallback。

提供方式(三选一,优先级从高到低):
1. CLI --data-root 参数
2. YAML scene.data_root 字段
3. SENSEFRAME_DATA_ROOT 环境变量

三者都未提供 → 启动时 raise FileNotFoundError。

注意:scripts/tests 在 import senseframe 之前仍需本地 bootstrap 将项目根加入
sys.path(chicken-and-egg),但 bootstrap 之后的业务代码应统一使用 PROJECT_ROOT。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# senseframe/common/paths.py → parents[2] = 项目根
# 仅用于框架自身资源定位,禁止用于数据路径推导
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_data_root(explicit: Optional[str] = None) -> Path:
    """解析数据根目录,只接受显式提供,不探测、不猜测。

    优先级:
    1. explicit 显式指定的路径(CLI / YAML 配置)
    2. SENSEFRAME_DATA_ROOT 环境变量

    Args:
        explicit: 用户显式指定的路径(最高优先级)

    Returns:
        解析后的绝对路径(不校验存在性,由调用方在加载数据时校验)

    Raises:
        FileNotFoundError: explicit 与环境变量都未提供
    """
    if explicit:
        return Path(explicit).resolve()
    env_root = os.environ.get("SENSEFRAME_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    raise FileNotFoundError(
        "Data root not provided. Provide via one of:\n"
        "  - YAML config: scene.data_root: /path/to/CSI_DATASETS\n"
        "  - CLI arg: --data-root /path/to/CSI_DATASETS\n"
        "  - Env var: SENSEFRAME_DATA_ROOT=/path/to/CSI_DATASETS"
    )


__all__ = ["PROJECT_ROOT", "resolve_data_root"]
