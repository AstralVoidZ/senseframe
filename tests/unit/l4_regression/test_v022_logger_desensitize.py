"""V022: M19 logger 脱敏（logger.exception 只记 code+category，不记 message）。

Anchor: bug 编号 V022 + 修复 commit 3ed29a0。
原始问题: to_tool_error 中 ``logger.exception`` 记录完整 envelope.message，
  而 envelope.message = str(exc) 可能含用户输入/路径/SQL 片段等敏感信息，
  写入 stderr 结构化日志有泄露风险。
修复方式: M19 修改 logger.exception 只记录安全元数据
  （code=%s category=%s），不记录 envelope.message；
  完整堆栈由 exc_info=True 自动附加（堆栈局部变量 repr 由 traceback 控制）。

回归测试策略：mock logger，验证 logger.exception 的调用参数不含
envelope.message（敏感信息），只含 code + category（安全元数据）。
如果修复被回退（logger.exception 重新记录 envelope.message），
mock 断言会捕获到敏感消息出现在日志参数中。
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.mark.l4_regression
class TestV022LoggerDesensitize:
    """锁定 V022 修复：logger.exception 不记录 envelope.message（脱敏）。"""

    def test_logger_exception_does_not_log_envelope_message(self):
        """V022 anchor: logger.exception 的调用参数不含 envelope.message。

        envelope.message = str(exc)，含敏感信息（用户输入/路径/SQL 片段）。
        M19 修复后 logger.exception 只记 code=%s category=%s。

        回滚验证：若修复被回退（logger.exception 重新记录 envelope.message），
        下方断言会因敏感消息出现在 log_args 中而失败。
        """
        from senseframe.mcp.errors import PipelineNotFound
        from senseframe.mcp.tools._errors import to_tool_error

        # 用含明显标记的敏感消息构造异常
        sensitive_message = "run-xyz not found; user=admin; path=/secret/data"
        exc = PipelineNotFound(sensitive_message)

        with patch("senseframe.mcp.tools._errors.logger") as mock_logger:
            tool_err = to_tool_error(exc)

        # V022 核心断言 1：logger.exception 被调用
        mock_logger.exception.assert_called_once()

        # V022 核心断言 2：调用参数不含敏感消息
        # logger.exception("tool error routed: code=%s category=%s", code, category)
        # 若回退为 logger.exception("...%s", envelope.message)，sensitive_message 会出现在 args
        call_args = mock_logger.exception.call_args
        all_args_str = str(call_args)
        assert sensitive_message not in all_args_str, (
            "V022 修复被回退：logger.exception 的调用参数含 envelope.message（敏感信息）。"
            f"实际调用参数: {all_args_str}"
        )

        # V022 核心断言 3：调用参数含安全元数据（code + category）
        assert "PipelineNotFound" in all_args_str or "code" in all_args_str, (
            "logger.exception 应记录 code 安全元数据"
        )
        assert "pipeline" in all_args_str or "category" in all_args_str, (
            "logger.exception 应记录 category 安全元数据"
        )

        # V022 辅助断言：返回值结构正确（ToolError payload 含 code + category）
        from mcp.server.fastmcp.exceptions import ToolError
        assert isinstance(tool_err, ToolError)
        payload = json.loads(str(tool_err))
        assert payload["code"] == "PipelineNotFound"
        assert payload["category"] == "pipeline"
