"""跨进程运行时状态持久化（P2-1）。

解决 CLI 多子命令场景下 _WSL2_WARNED / _LOGGING_CONFIGURED 进程内标志
失效的问题：每次子命令都是新进程，进程内 global 变量无去重效果。

设计：用 ~/.senseframe/runtime_state.json 持久化跨进程状态。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

_logger = logging.getLogger(__name__)

# 跨进程状态文件位置（与 skills 存储对齐，使用 ~/.senseframe/）
RUNTIME_STATE_DIR = Path(os.path.expanduser("~/.senseframe"))
RUNTIME_STATE_FILE = RUNTIME_STATE_DIR / "runtime_state.json"


def load_runtime_state() -> Dict[str, Any]:
    """加载跨进程运行时状态。

    Returns:
        状态 dict（不存在或读取失败时返回空 dict）
    """
    try:
        if RUNTIME_STATE_FILE.exists():
            with open(RUNTIME_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        _logger.debug("Failed to load runtime state: %s", e)
    return {}


def save_runtime_state(state: Dict[str, Any]) -> None:
    """保存跨进程运行时状态。

    Args:
        state: 完整的状态 dict（覆盖写入）
    """
    try:
        RUNTIME_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _logger.warning("Failed to save runtime state: %s", e)


def get_state(key: str, default: Any = None) -> Any:
    """读取单个状态键。"""
    return load_runtime_state().get(key, default)


def set_state(key: str, value: Any) -> None:
    """设置单个状态键（read-modify-write）。"""
    state = load_runtime_state()
    state[key] = value
    save_runtime_state(state)


__all__ = [
    "RUNTIME_STATE_DIR",
    "RUNTIME_STATE_FILE",
    "load_runtime_state",
    "save_runtime_state",
    "get_state",
    "set_state",
]
