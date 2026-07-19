"""StudyManager 进程级单例（L4 SP 搜索协议）。

为 MCP tools/ 与 resources/ 提供共享的 StudyManager 实例。
StudyManager（来自 senseframe.search_protocol）已实现 Ask-Tell 接口
与 ExplorationTracker 桥接，本模块仅做进程级单例包装。

测试可通过 set_default_manager(None) 重置，或注入 mock。
"""

from __future__ import annotations

import threading

from senseframe.search_protocol import StudyManager

__all__ = [
    "get_default_manager",
    "set_default_manager",
]

_default_lock = threading.RLock()
_default_instance: StudyManager | None = None


def get_default_manager() -> StudyManager:
    """返回进程级 StudyManager 单例（惰性初始化）。

    tools/study.py 与 resources/ 共享此单例，确保 tool 创建的 study
    对 resource 端点可见。
    """
    global _default_instance
    with _default_lock:
        if _default_instance is None:
            _default_instance = StudyManager()
        return _default_instance


def set_default_manager(mgr: StudyManager | None) -> None:
    """注入/重置进程级 StudyManager 单例。

    测试用：注入 mock manager 或重置为 None 触发重新初始化。
    """
    global _default_instance
    with _default_lock:
        _default_instance = mgr
