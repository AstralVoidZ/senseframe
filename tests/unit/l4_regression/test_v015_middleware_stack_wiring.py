"""V015: I14 MiddlewareStack 接入。

Anchor: bug 编号 V015 + 修复 commit 6be8b80。
原始问题: config_parse 是唯一未接入 MiddlewareStack 的 tool，绕过
  RequestId + RateLimit 中间件，导致请求无法追踪 + 限流失效。
修复方式: tools/config.py 中创建 ``_config_stack = MiddlewareStack(...)``
  并通过 ``async with _config_stack.instrument(...)`` 包装核心逻辑。

如果此测试失败，说明 V015 修复被回退（config tool 未接入 MiddlewareStack）。
"""
from __future__ import annotations

import ast
import pathlib

import pytest


@pytest.mark.l4_regression
class TestV015MiddlewareStackWiring:
    """锁定 V015 修复：tools/config.py 通过 _config_stack.instrument 调用。"""

    @staticmethod
    def _has_call_pattern(tree, func_name: str, method_name: str) -> bool:
        """检查 AST 是否含 ``<func_name>.<method_name>(...)`` 调用。"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == method_name:
                    value = node.value
                    if isinstance(value, ast.Name) and value.id == func_name:
                        return True
        return False

    def test_config_tools_use_middleware_stack(self):
        """V015 anchor: AST 检查 tools/config.py 含 _config_stack.instrument(...) 调用。

        如果此断言失败，V015 修复被回退。
        """
        # tests/unit/l4_regression/test_v015_*.py → SenseFrame/
        # parents[0]=l4_regression, [1]=unit, [2]=tests, [3]=SenseFrame
        senseframe_root = pathlib.Path(__file__).resolve().parents[3]
        py = senseframe_root / "senseframe" / "mcp" / "tools" / "config.py"
        assert py.exists(), f"{py} must exist"

        tree = ast.parse(py.read_text(encoding="utf-8"))
        # V015 关键断言：含 _config_stack.instrument 调用
        assert self._has_call_pattern(tree, "_config_stack", "instrument"), (
            f"如果此断言失败，V015 修复被回退：{py} must call "
            f"_config_stack.instrument(...) to wrap tool logic with MiddlewareStack"
        )
