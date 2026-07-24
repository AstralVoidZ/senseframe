"""V015: I14 MiddlewareStack 接入。

Anchor: bug 编号 V015 + 修复 commit 6be8b80。
原始问题: config_parse 是唯一未接入 MiddlewareStack 的 tool，绕过
  RequestId + RateLimit 中间件，导致请求无法追踪 + 限流失效。
修复方式: tools/config.py 中创建 ``_config_stack = MiddlewareStack(...)``
  并通过 ``async with _config_stack.instrument(...)`` 包装核心逻辑。

如果此测试失败，说明 V015 修复被回退（config tool 未接入 MiddlewareStack）。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.l4_regression
class TestV015MiddlewareStackWiring:
    """锁定 V015 修复：senseframe_config_parse 通过 _config_stack.instrument 注入 request_id。

    行为测试：调用 senseframe_config_parse 时，RequestIdMiddleware 应将 ctx.request_id
    注入模块级 ContextVar（get_request_id() 返回非 "-"）。

    替代 AST 检查的理由：AST 结构检查会因合法重命名（如 _config_stack 改为局部变量）
    误报，行为测试直接验证功能不受重命名影响。
    """

    @staticmethod
    def _valid_yaml() -> str:
        """最小合法 YAML（与 tests/test_mcp_config_parse.py 对齐）。"""
        return """
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: MLP
  data_root: /tmp/data

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 50
  batch_size: 32
"""

    @pytest.mark.asyncio
    async def test_config_parse_wires_request_id_middleware(self):
        """V015 anchor: senseframe_config_parse 调用期间 request_id 被注入 ContextVar。

        通过 spy ExperimentConfig.from_dict（在 async with _config_stack.instrument
        块内被调用）捕获 get_request_id() 的值。如果 _config_stack.instrument 被移除，
        get_request_id() 始终返回 "-"，测试失败。

        如果此断言失败，V015 修复被回退。
        """
        from senseframe.engine.config import ExperimentConfig
        from senseframe.mcp.middleware import get_request_id
        from senseframe.mcp.tools.config import senseframe_config_parse

        mock_ctx = SimpleNamespace(
            request_id="v015-test-req-123",
            info=AsyncMock(),
            error=AsyncMock(),
        )

        captured_request_ids: list[str] = []
        original_from_dict = ExperimentConfig.from_dict

        def spy_from_dict(d):
            # 在 _config_stack.instrument 块内捕获 request_id
            captured_request_ids.append(get_request_id())
            return original_from_dict(d)

        with patch.object(ExperimentConfig, "from_dict", spy_from_dict):
            await senseframe_config_parse(
                config_yaml=self._valid_yaml(), ctx=mock_ctx
            )

        assert captured_request_ids, (
            "ExperimentConfig.from_dict 未被调用，spy 未触发"
        )
        assert captured_request_ids[0] == "v015-test-req-123", (
            "V015 修复被回退：senseframe_config_parse 未通过 _config_stack.instrument "
            f"注入 request_id（实际值: {captured_request_ids[0]}），"
            "RequestIdMiddleware 可能未接入或 instrument 调用被移除"
        )
