"""L1 契约测试：Pydantic v2 Schema 配置契约。

锚点来源：PyTorch Lightning 官方 API（BaseModel / extra / frozen / field_validator）。
- pydantic v2 文档: https://docs.pydantic.dev/latest/concepts/models/
  - extra='forbid': 拒绝未声明字段（防止 schema drift）
  - frozen=True: 模型不可变（构造后不可修改字段）
  - ConfigDict: pydantic v2 配置方式（替代 v1 的 class Config）
- pydantic v2 迁移指南: https://docs.pydantic.dev/latest/migration/
  - @field_validator 替代 v1 的 @validator
  - @model_validator 替代 v1 的 @root_validator

验证目标：
- FrozenModel 满足 extra='forbid' + frozen=True（pydantic v2 文档契约）
- FrozenModel 子类继承 frozen/extra 配置
- engine/config.py 使用 @field_validator（v2 装饰器），不使用 v1 @validator
"""
from __future__ import annotations

import inspect
import re

import pytest


@pytest.mark.l1_contract
class TestPydanticSchemaContract:
    """验证 FrozenModel 与 pydantic v2 配置符合官方文档契约。"""

    def test_frozen_model_extra_is_forbid(self):
        """L1 anchor: FrozenModel.model_config['extra'] == 'forbid'，锚点：pydantic v2 文档。

        pydantic v2 文档: ConfigDict(extra='forbid') 拒绝未声明字段，
        防止客户端因 schema drift 接收到意外字段。
        断言目标: model_config['extra'] 的值，锚点: pydantic v2 官方文档。
        """
        from senseframe.mcp.views._base import FrozenModel

        assert FrozenModel.model_config["extra"] == "forbid", (
            f"FrozenModel extra 应为 'forbid'（pydantic v2 文档），"
            f"实际 {FrozenModel.model_config['extra']!r}"
        )

    def test_frozen_model_frozen_is_true(self):
        """L1 anchor: FrozenModel.model_config['frozen'] is True，锚点：pydantic v2 文档。

        pydantic v2 文档: ConfigDict(frozen=True) 使模型不可变，
        构造后赋值字段会抛 ValidationError。
        """
        from senseframe.mcp.views._base import FrozenModel

        assert FrozenModel.model_config["frozen"] is True, (
            f"FrozenModel frozen 应为 True（pydantic v2 文档），"
            f"实际 {FrozenModel.model_config['frozen']!r}"
        )

    def test_frozen_model_rejects_extra_fields(self):
        """L1 anchor: extra='forbid' 拒绝未声明字段，锚点：pydantic v2 ValidationError 行为。

        pydantic v2 文档: extra='forbid' 时传入未声明字段抛 ValidationError。
        这是行为级验证（而非仅检查配置值），确保配置实际生效。
        """
        from pydantic import ValidationError

        from senseframe.mcp.views._base import FrozenModel

        class _TestModel(FrozenModel):
            x: int

        with pytest.raises(ValidationError):
            _TestModel(x=1, unknown_field=2)

    def test_frozen_model_is_immutable(self):
        """L1 anchor: frozen=True 使模型不可变，锚点：pydantic v2 ValidationError 行为。

        pydantic v2 文档: frozen=True 时对已构造模型的字段赋值抛 ValidationError。
        这是行为级验证，确保 frozen 配置实际生效。
        """
        from pydantic import ValidationError

        from senseframe.mcp.views._base import FrozenModel

        class _TestModel(FrozenModel):
            x: int

        m = _TestModel(x=1)
        with pytest.raises(ValidationError):
            m.x = 2

    def test_tool_error_response_inherits_frozen_extra_config(self):
        """L1 anchor: ToolErrorResponse 继承 frozen=True + extra='forbid'，锚点：pydantic v2 继承。

        pydantic v2 文档: 子类继承父类的 model_config。FrozenModel 的子类
        (ToolErrorResponse) 必须继承 frozen + extra='forbid' 配置。
        锚点: pydantic v2 官方文档的 model_config 继承行为。
        """
        from senseframe.mcp.views.tool_error import ToolErrorResponse

        assert ToolErrorResponse.model_config["frozen"] is True, (
            "ToolErrorResponse 应继承 FrozenModel 的 frozen=True 配置"
        )
        assert ToolErrorResponse.model_config["extra"] == "forbid", (
            "ToolErrorResponse 应继承 FrozenModel 的 extra='forbid' 配置"
        )

    def test_config_parse_response_inherits_frozen_extra_config(self):
        """L1 anchor: ConfigParseResponse 继承 frozen=True + extra='forbid'，锚点：pydantic v2 继承。

        pydantic v2 文档: 子类继承父类的 model_config。
        ConfigParseResponse 是 FrozenModel 的子类，必须继承配置。
        """
        from senseframe.mcp.views.config import ConfigParseResponse

        assert ConfigParseResponse.model_config["frozen"] is True, (
            "ConfigParseResponse 应继承 FrozenModel 的 frozen=True 配置"
        )
        assert ConfigParseResponse.model_config["extra"] == "forbid", (
            "ConfigParseResponse 应继承 FrozenModel 的 extra='forbid' 配置"
        )

    def test_apply_params_response_inherits_frozen_extra_config(self):
        """L1 anchor: ApplyParamsExtendedResponse 继承 frozen=True + extra='forbid'。

        锚点：pydantic v2 model_config 继承行为。
        ApplyParamsExtendedResponse 定义在 tools/param_bridge.py，是 FrozenModel 子类。
        """
        from senseframe.mcp.tools.param_bridge import ApplyParamsExtendedResponse

        assert ApplyParamsExtendedResponse.model_config["frozen"] is True, (
            "ApplyParamsExtendedResponse 应继承 FrozenModel 的 frozen=True 配置"
        )
        assert ApplyParamsExtendedResponse.model_config["extra"] == "forbid", (
            "ApplyParamsExtendedResponse 应继承 FrozenModel 的 extra='forbid' 配置"
        )

    def test_engine_config_uses_field_validator_not_v1_validator(self):
        """L1 anchor: engine/config.py 使用 pydantic v2 @field_validator，不使用 v1 @validator。

        锚点：pydantic v2 迁移指南
        (https://docs.pydantic.dev/latest/migration/#validator-changes)。
        v2 用 @field_validator 替代 v1 的 @validator。
        断言: 源码含 @field_validator 装饰器，不含 v1 的 @validator（非 field_ 前缀）。
        """
        from senseframe.engine import config as config_module

        source = inspect.getsource(config_module)

        # pydantic v2 契约：使用 @field_validator 装饰器
        assert "@field_validator" in source, (
            "engine/config.py 必须使用 pydantic v2 的 @field_validator 装饰器"
        )
        assert "field_validator" in source, (
            "engine/config.py 必须导入 field_validator"
        )

        # pydantic v1 的 @validator（非 @field_validator）不应出现
        # 用负向先行断言匹配 @validator 但不匹配 @field_validator
        v1_pattern = re.compile(r"(?<!field_)@validator\b")
        v1_matches = v1_pattern.findall(source)
        assert not v1_matches, (
            f"engine/config.py 不应使用 pydantic v1 的 @validator 装饰器，"
            f"应改用 v2 的 @field_validator。发现 v1 模式: {v1_matches}"
        )

    def test_engine_config_uses_configdict_not_class_config(self):
        """L1 anchor: engine/config.py 使用 pydantic v2 ConfigDict，不使用 v1 class Config。

        锚点：pydantic v2 迁移指南
        (https://docs.pydantic.dev/latest/migration/#changes-to-pydanticbasemodel)。
        v2 用 model_config = ConfigDict(...) 替代 v1 的 class Config:。
        """
        from senseframe.engine import config as config_module

        source = inspect.getsource(config_module)

        # pydantic v2 契约：使用 ConfigDict
        assert "ConfigDict" in source, (
            "engine/config.py 应使用 pydantic v2 的 ConfigDict 配置模型"
        )

        # pydantic v1 的 class Config 不应出现（作为模型配置）
        # 注意：class Config 可能出现在其他上下文，但 v2 模型配置应使用 ConfigDict
        v1_config_pattern = re.compile(r"class\s+Config\s*:")
        v1_matches = v1_config_pattern.findall(source)
        assert not v1_matches, (
            f"engine/config.py 不应使用 pydantic v1 的 class Config 配置，"
            f"应改用 v2 的 model_config = ConfigDict(...)。"
        )
