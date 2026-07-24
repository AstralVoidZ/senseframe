"""L1 conftest：外部协议契约测试 fixtures。

L1 测试锚定外部协议/库 API，不 mock 被测代码自身，只验证代码行为
符合外部契约。测试通过直接 import 被测模块（如 _TOOL_REGISTRY、server）
获取契约常量，无需 fixture 注入。
"""
from __future__ import annotations
