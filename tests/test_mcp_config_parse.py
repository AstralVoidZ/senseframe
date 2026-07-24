"""senseframe_config_parse MCP tool 测试。

LOW 8 修复：适配 FrozenModel 响应 + ToolError 错误。
"""
from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError


class TestConfigParse:
    """senseframe_config_parse handler 测试。"""

    @pytest.fixture
    def valid_yaml(self):
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

    @pytest.fixture
    def invalid_yaml_unknown_field(self):
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

unknown_field: should_fail
"""

    @pytest.mark.asyncio
    async def test_parse_valid_yaml(self, valid_yaml):
        """合法 YAML 应返回 ConfigParseResponse。"""
        from senseframe.mcp.tools.config import senseframe_config_parse
        from senseframe.mcp.views.config import ConfigParseResponse

        result = await senseframe_config_parse(config_yaml=valid_yaml)

        assert isinstance(result, ConfigParseResponse)
        assert result.config["scene"]["name"] == "wifi_csi"
        assert result.config["trainer"]["epochs"] == 50

    @pytest.mark.asyncio
    async def test_parse_invalid_yaml_rejected(self, invalid_yaml_unknown_field):
        """含未知字段的 YAML 应抛 ToolError。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        with pytest.raises(ToolError) as exc_info:
            await senseframe_config_parse(config_yaml=invalid_yaml_unknown_field)

        envelope = json.loads(str(exc_info.value))
        assert envelope["category"] == "config"

    @pytest.mark.asyncio
    async def test_parse_malformed_yaml(self):
        """YAML 语法错误应抛 ToolError（category=config）。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        with pytest.raises(ToolError) as exc_info:
            await senseframe_config_parse(config_yaml=": : : invalid yaml")

        envelope = json.loads(str(exc_info.value))
        assert envelope["category"] == "config"

    @pytest.mark.asyncio
    async def test_parse_missing_required_field(self):
        """缺必需字段应抛 ToolError。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        with pytest.raises(ToolError) as exc_info:
            await senseframe_config_parse(config_yaml="scene: {name: wifi_csi}")

        envelope = json.loads(str(exc_info.value))
        assert envelope["category"] == "config"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "yaml_input",
        ["42", "- a\n- b"],
        ids=["scalar", "list"],
    )
    async def test_parse_non_dict_top_level_rejected(self, yaml_input):
        """非 dict 顶层应抛 ToolError（category=config）。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        with pytest.raises(ToolError) as exc_info:
            await senseframe_config_parse(config_yaml=yaml_input)

        envelope = json.loads(str(exc_info.value))
        assert envelope["category"] == "config"

    @pytest.mark.asyncio
    async def test_subconfig_validation_error_routes_to_config(self):
        """I15 修复：子配置 ValidationError 应路由到 config category。

        ``trainer.epochs=-1`` 触发 TrainerConfig._validate_epochs field_validator，
        pydantic 包装为 ValidationError。修复前 _CATEGORY_BY_EXC 无此映射，
        误路由到 internal；修复后正确路由到 config。
        """
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
        assert envelope["category"] == "config", (
            f"子配置 ValidationError 应路由到 config，实际: {envelope['category']}"
        )
