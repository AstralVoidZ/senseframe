"""senseframe_config_parse MCP tool 测试。

验证 YAML 配置可解析为 ExperimentConfig（v2 次要差距修复）。
"""
from __future__ import annotations

import pytest


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
        """合法 YAML 应返回解析后的 ExperimentConfig dict。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        result = await senseframe_config_parse(config_yaml=valid_yaml)

        assert result["status"] == "ok"
        assert result["config"]["scene"]["name"] == "wifi_csi"
        assert result["config"]["trainer"]["epochs"] == 50
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_parse_invalid_yaml_rejected(self, invalid_yaml_unknown_field):
        """含未知字段的 YAML 应被 extra='forbid' 拒绝。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        result = await senseframe_config_parse(config_yaml=invalid_yaml_unknown_field)

        assert result["status"] == "error"
        assert result["error_code"] == "CONFIG_VALIDATION_ERROR"
        assert "unknown_field" in result["error"] or "未知字段" in result["error"]

    @pytest.mark.asyncio
    async def test_parse_malformed_yaml(self):
        """YAML 语法错误应返回 error。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        result = await senseframe_config_parse(config_yaml=": : : invalid yaml")

        assert result["status"] == "error"
        assert result["error_code"] == "CONFIG_YAML_PARSE_ERROR"

    @pytest.mark.asyncio
    async def test_parse_missing_required_field(self):
        """缺必需字段应返回 error。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        result = await senseframe_config_parse(config_yaml="scene: {name: wifi_csi}")

        assert result["status"] == "error"
        assert result["error_code"] == "CONFIG_VALIDATION_ERROR"
        assert "input_features" in result["error"] or "output_features" in result["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config_yaml",
        [
            "42",          # 标量
            "- a\n- b",    # list
        ],
        ids=["scalar", "list"],
    )
    async def test_parse_non_dict_top_level_rejected(self, config_yaml):
        """YAML 顶层非 dict（标量 / list）应返回 error。"""
        from senseframe.mcp.tools.config import senseframe_config_parse

        result = await senseframe_config_parse(config_yaml=config_yaml)

        assert result["status"] == "error"
        assert result["error_code"] == "CONFIG_VALIDATION_ERROR"
