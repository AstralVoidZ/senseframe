"""ISP-10（alt）：``senseframe://protocols`` 已注册 Protocol 类型。

返回 SenseFrame 已注册的 Protocol 类型列表（RFC-003 DSP-1）。

依赖：
- senseframe.engine.runner.pipeline.protocols（8 个 Protocol 类型）

注意：resources/ 不得 import tools/（AST 守卫测试钉死）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["protocols"]


async def protocols() -> dict[str, Any]:
    """ISP-10：已注册 Protocol 类型索引。

    Returns:
        含 schema_version + protocols 列表（每条含 name + methods）。
    """
    from senseframe.engine.runner.pipeline.protocols import (
        DataModuleProtocol,
        FeatureSpecProtocol,
        LoggerProtocol,
        ModelProtocol,
        SceneMetaProtocol,
        SceneProtocol,
        TaskSpecProtocol,
        TrainerProtocol,
    )

    # Protocol 类型注册表（name → Protocol class）
    protocol_classes = [
        ("SceneProtocol", SceneProtocol),
        ("ModelProtocol", ModelProtocol),
        ("DataModuleProtocol", DataModuleProtocol),
        ("TrainerProtocol", TrainerProtocol),
        ("LoggerProtocol", LoggerProtocol),
        ("SceneMetaProtocol", SceneMetaProtocol),
        ("TaskSpecProtocol", TaskSpecProtocol),
        ("FeatureSpecProtocol", FeatureSpecProtocol),
    ]

    protocols_list: list[dict[str, Any]] = []
    for name, cls in protocol_classes:
        # 提取 Protocol 声明的方法 / 属性
        methods: list[str] = []
        attributes: list[str] = []
        # 遍历 __protocol_attrs__（PEP 544）
        if hasattr(cls, "__protocol_attrs__"):
            for attr_name in sorted(cls.__protocol_attrs__):
                attr = getattr(cls, attr_name, None)
                if callable(attr) or isinstance(
                    getattr(cls, attr_name, None), (property,)
                ):
                    methods.append(attr_name)
                else:
                    attributes.append(attr_name)
        protocols_list.append({
            "name": name,
            "methods": methods,
            "attributes": attributes,
            "runtime_checkable": getattr(cls, "_is_runtime_protocol", False),
        })

    return {
        "schema_version": "1.0.0",
        "protocols": protocols_list,
    }
