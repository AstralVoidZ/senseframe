"""View 层基类：FrozenModel + from_domain 辅助工具。

`senseframe.mcp.views` 暴露 Pydantic v2 BaseModel 子类作为 MCP 客户端
消费的公共 JSON 契约（FastMCP 自动从 model_json_schema() 生成 outputSchema）。

分层不变量（AST 守卫测试钉死）：
    tools/  →  views/  →  models/
反向导入禁止：
    views/ 不得 import orchestration / tools / storage / spec
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Frozen BaseModel：extra='forbid' 拒绝未知字段 + frozen=True 不可变。

    设计依据（设计文档 0.5 节错误信封 + pipeflow D5）：
    - view 是公共契约，客户端依赖声明字段集；静默新增字段是 schema drift 信号
    - frozen=True 镜像底层 dataclass（models/*.py 为 @dataclass(frozen=True)），
      views 构造后不可变
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ViewError(TypeError):
    """from_domain 输入不是已知 dataclass 时抛出。

    被 tool 层作为 TypeError 捕获，路由到 to_tool_error →
    ToolErrorResponse(category="internal")。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    """读取 obj.name，容忍 dict / dataclass / BaseModel 三种输入。

    - dict：obj.get(name, default)
    - dataclass / BaseModel / 普通对象：getattr(obj, name, default)
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_value(v: Any) -> str:
    """StrEnum / str → str。

    非 str 输入读取 .value 属性（StrEnum / IntEnum 等枚举值）。
    """
    if isinstance(v, str):
        return v
    result: str = v.value
    return result
