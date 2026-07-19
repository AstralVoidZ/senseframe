"""SenseFrame MCP 服务器包。

5 层单向依赖架构（设计文档 0.2 节）：

    L1 Protocol  →  L2 View  →  L3 Orchestration  →  L4 ISP/L5 OPP  →  models
    (server / tool_dispatch) (views) (orchestration) (resources / tools)

AST 守卫测试（tests/test_mcp_architecture_invariants.py）钉死分层不变量。

本阶段（1.2-1.6）已实现基础设施：
- L1 Protocol: server.py (FastMCP + lifespan + signal handlers)
                tool_dispatch.py (EXPECTED_TOOLS + _TOOL_REGISTRY)
- L2 View: views/_base.py (FrozenModel), views/tool_error.py (ToolErrorResponse)
- 中间件: middleware.py (MiddlewareStack + RequestId + TokenBucket)
- 配置: config.py (validate_config)
- 错误: errors.py (9 协议层错误 + ML 业务错误映射)

后续阶段填充 resources/ tools/ orchestration/ storage/ pagination/ modules/。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
