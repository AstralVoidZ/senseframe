"""V016: I15 pydantic.ValidationError 路由到 config category。

Anchor: bug 编号 V016 + 修复 commit 6be8b80。
原始问题: 子配置（SceneConfig/TrainerConfig 等）field_validator 校验失败时
  pydantic 抛 ValidationError，但 _CATEGORY_BY_EXC 无此映射，
  误路由到 internal category（客户端无法按 config 错误恢复）。
修复方式: tool_error.py 的 _CATEGORY_BY_EXC 添加
  ``(pydantic.ValidationError, "config")`` 映射，
  放在 ConfigValidationError 之后。

如果此测试失败，说明 V016 修复被回退（ValidationError 误路由到 internal）。
"""
from __future__ import annotations

import json

import pytest


@pytest.mark.l4_regression
class TestV016PydanticValidationErrorRoute:
    """锁定 V016 修复：子配置 ValidationError 路由到 config category。"""

    @pytest.mark.asyncio
    async def test_subconfig_validation_error_routes_to_config(self):
        """V016 anchor: trainer.epochs=-1 触发 pydantic ValidationError 时 category=config。

        如果此断言失败，V016 修复被回退。
        """
        from mcp.server.fastmcp.exceptions import ToolError

        from senseframe.mcp.tools.config import senseframe_config_parse

        bad_yaml = """
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
  epochs: -1
"""
        with pytest.raises(ToolError) as exc_info:
            await senseframe_config_parse(config_yaml=bad_yaml)

        envelope = json.loads(str(exc_info.value))
        # V016 关键断言：category == "config"（非 internal）
        assert envelope["category"] == "config", (
            f"如果此断言失败，V016 修复被回退：子配置 ValidationError 应路由到 "
            f"config category，实际: {envelope['category']}"
        )
