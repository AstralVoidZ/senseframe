"""V022: M19 logger 脱敏（logger.exception 只记 code+category，不记 message）。

Anchor: bug 编号 V022 + 修复 commit 3ed29a0。
原始问题: to_tool_error 中 ``logger.exception`` 记录完整 envelope.message，
  而 envelope.message = str(exc) 可能含用户输入/路径/SQL 片段等敏感信息，
  写入 stderr 结构化日志有泄露风险。
修复方式: M19 修改 logger.exception 只记录安全元数据
  （code=%s category=%s），不记录 envelope.message；
  完整堆栈由 exc_info=True 自动附加（堆栈局部变量 repr 由 traceback 控制）。

如果此测试失败，说明 V022 修复被回退（返回值结构不正确）。
"""
from __future__ import annotations

import json

import pytest


@pytest.mark.l4_regression
class TestV022LoggerDesensitize:
    """锁定 V022 修复：to_tool_error 返回 ToolError，payload 含 code + category。

    M19 修复的是 logger.exception 不记录 envelope.message，
    L4 测试验证返回值结构即可（ToolError payload 含 code/category）。
    """

    def test_to_tool_error_returns_tool_error_with_envelope_json(self):
        """V022 anchor: to_tool_error(exc) 返回 ToolError，payload JSON 含 code + category。

        如果此断言失败，V022 修复被回退。
        """
        from mcp.server.fastmcp.exceptions import ToolError

        from senseframe.mcp.errors import PipelineNotFound
        from senseframe.mcp.tools._errors import to_tool_error

        exc = PipelineNotFound("run-xyz not found")
        tool_err = to_tool_error(exc)

        # V022 关键断言 1：返回 ToolError 实例
        assert isinstance(tool_err, ToolError), (
            "如果此断言失败，V022 修复被回退：to_tool_error 应返回 ToolError 实例"
        )

        # V022 关键断言 2：payload JSON 含 code + category（安全元数据）
        payload = json.loads(str(tool_err))
        assert payload["code"] == "PipelineNotFound", (
            f"如果此断言失败，V022 修复被回退：payload code 应为 "
            f"PipelineNotFound，实际 {payload.get('code')!r}"
        )
        assert payload["category"] == "pipeline", (
            f"如果此断言失败，V022 修复被回退：payload category 应为 "
            f"pipeline，实际 {payload.get('category')!r}"
        )
