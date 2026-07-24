"""L1 conftest：外部协议契约测试 fixtures。

L1 测试锚定外部协议/库 API，不 mock 被测代码自身，只验证代码行为
符合外部契约。Fixture 以"契约常量"和"被测模块导入"为主。
"""
from __future__ import annotations

import pytest


@pytest.fixture
def tool_registry():
    """senseframe.mcp.server._TOOL_REGISTRY（L1 契约常量）。

    L1 测试锚定 MCP tool 协议（name/description/inputSchema），
    通过 inspect.signature 验证工具签名符合 MCP spec。
    """
    from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS
    return EXPECTED_TOOLS


@pytest.fixture
def mcp_server_module():
    """senseframe.mcp.server 模块（L1 契约：lifespan / FastMCP）。

    L1 测试锚定 FastMCP lifespan 协议（startup/shutdown 语义）。
    """
    from senseframe.mcp import server
    return server
