"""V018: I17 no-op validate 移除。

Anchor: bug 编号 V018 + 修复 commit 6be8b80。
原始问题: tools/config.py 含 ``config.validate()`` no-op 调用，
  该方法不执行任何校验逻辑（空实现），给读者"已校验"的虚假安全感。
修复方式: I17 移除 no-op validate 调用，校验逻辑由
  ExperimentConfig.from_dict + pydantic extra='forbid' 承载。

如果此测试失败，说明 V018 修复被回退（no-op validate 被重新引入）。
"""
from __future__ import annotations

import ast
import pathlib

import pytest


@pytest.mark.l4_regression
class TestV018NoopValidateRemoved:
    """锁定 V018 修复：tools/config.py 不含 config.validate() 调用。"""

    def test_config_py_has_no_config_validate_call(self):
        """V018 anchor: AST 检查 tools/config.py 不含 config.validate() 调用。

        缩小匹配范围：仅匹配 ``config.validate()``（ExperimentConfig 实例的 no-op
        validate），不再误报其他合法的 ``.validate()`` 调用（如 pydantic 模型校验）。

        如果此断言失败，V018 修复被回退（no-op validate 被重新引入）。
        """
        # tests/unit/l4_regression/test_v018_*.py → SenseFrame/
        # parents[0]=l4_regression, [1]=unit, [2]=tests, [3]=SenseFrame
        senseframe_root = pathlib.Path(__file__).resolve().parents[3]
        py = senseframe_root / "senseframe" / "mcp" / "tools" / "config.py"
        assert py.exists(), f"{py} must exist"

        tree = ast.parse(py.read_text(encoding="utf-8"))

        # V018 关键断言：不含 config.validate() 调用（no-op validate 已移除）
        # 仅匹配 Call.func 是 Attribute 且 attr == "validate" 且
        # func.value 是 Name 且 id == "config" 的情况，避免误报其他 .validate() 调用
        config_validate_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "validate"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "config"
                ):
                    config_validate_calls.append(node)
        assert not config_validate_calls, (
            f"如果此断言失败，V018 修复被回退：tools/config.py 不应含 "
            f"config.validate() 调用（no-op validate 已移除），"
            f"发现 {len(config_validate_calls)} 处"
        )
