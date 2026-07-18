"""SenseFrame 通用工具模块。

只放跨模块复用、无业务逻辑的纯工具。
有业务语义的工具应归对应功能模块，避免 common 成为"垃圾抽屉"。
"""
from .checkpoint import load_checkpoint_flexible
from .path_safe import (
    resolve_under,
    safe_relative_path,
    sanitize_path_component,
    is_path_component_safe,
)
from .paths import PROJECT_ROOT

__all__ = [
    "resolve_under",
    "safe_relative_path",
    "sanitize_path_component",
    "is_path_component_safe",
    "PROJECT_ROOT",
    "load_checkpoint_flexible",
]
