"""L1 契约测试：MCP 工具签名与 ToolAnnotations 协议。

锚点来源：MCP spec（Model Context Protocol）+ FastMCP 工具协议。
- MCP spec: https://modelcontextprotocol.io/docs/concepts/tools
  每个 tool 必须有 name (string) / description (string) / inputSchema (JSON Schema)。
- FastMCP: 工具注册通过 add_tool(fn, name, description, annotations)，
  inputSchema 由函数签名的类型注解自动生成。
- ToolAnnotations 含 readOnlyHint / destructiveHint / idempotentHint / openWorldHint
  四个布尔语义提示（MCP spec 定义）。

与旧测试 test_mcp_architecture_invariants.py 的区别：
- 旧测试 `assert len(EXPECTED_TOOLS) == 29` 引用源码常量 EXPECTED_TOOLS。
- 新测试硬编码 29 个工具名集合（锚点：设计文档 0.4 节 ToolAnnotations 矩阵），
  锚点与被测对象分离，消除自证断言。
"""
from __future__ import annotations

import inspect

import pytest

# ============================================================
# 外部锚点：设计文档 0.4 节定义的 29 个 tool 名称
# （不引用 senseframe.mcp.tool_dispatch.EXPECTED_TOOLS，
#   而是硬编码期望集合，锚点外部化）
# ============================================================
_EXPECTED_TOOL_COUNT = 29

_EXPECTED_TOOL_NAMES: frozenset[str] = frozenset({
    # 声明类（2）
    "senseframe_config_parse",
    "senseframe_pipeline_create",
    # 状态转移类（1）
    "senseframe_pipeline_advance",
    # 执行类（1）
    "senseframe_pipeline_run",
    # 查询类（4）
    "senseframe_pipeline_get",
    "senseframe_pipeline_list",
    "senseframe_pipeline_pause",
    "senseframe_pipeline_resume",
    # Study 类（7）
    "senseframe_study_create",
    "senseframe_study_ask",
    "senseframe_study_tell",
    "senseframe_study_get",
    "senseframe_study_list",
    "senseframe_study_compare",
    "senseframe_study_stop",
    # HPO 类（1）
    "senseframe_hpo_setup",
    # Exploration 类（1）
    "senseframe_exploration_recommend",
    # AutoML 类（4）
    "senseframe_automl_create",
    "senseframe_automl_advance",
    "senseframe_automl_get",
    "senseframe_automl_list",
    # Param Bridge 类（1）
    "senseframe_apply_params_extended",
    # Artifact 类（3）
    "senseframe_artifact_verify",
    "senseframe_artifact_list",
    "senseframe_artifact_export",
    # 技能类（4）
    "senseframe_skill_save",
    "senseframe_skill_get",
    "senseframe_skill_search",
    "senseframe_skill_remove",
})

# MCP spec 定义的 ToolAnnotations 四个布尔语义提示字段
_TOOLANNOTATION_HINTS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)


@pytest.mark.l1_contract
class TestMcpToolSignaturesContract:
    """验证 MCP tool 注册与签名符合 MCP spec + FastMCP 协议契约。"""

    def test_tool_count_matches_design_spec(self):
        """L1 anchor: 注册的 tool 数量必须等于 29，锚点：设计文档 0.4 节 ToolAnnotations 矩阵。

        锚点外部化：29 是硬编码常量（来自设计文档），不引用 EXPECTED_TOOLS。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        assert len(_TOOL_REGISTRY) == _EXPECTED_TOOL_COUNT, (
            f"设计文档定义 {_EXPECTED_TOOL_COUNT} 个 tool，"
            f"实际注册 {len(_TOOL_REGISTRY)} 个"
        )

    def test_tool_names_match_design_spec(self):
        """L1 anchor: 注册的 tool 名称集合必须与设计文档 0.4 节定义完全一致。

        锚点：硬编码的 _EXPECTED_TOOL_NAMES 集合（设计文档 0.4 节），
        被测对象：_TOOL_REGISTRY 中的实际 tool 名称。
        两者分离，改源码常量不会自动通过本测试。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        actual_names = {entry[0] for entry in _TOOL_REGISTRY}
        assert actual_names == _EXPECTED_TOOL_NAMES, (
            f"tool 名称集合不匹配设计文档: "
            f"missing={sorted(_EXPECTED_TOOL_NAMES - actual_names)} "
            f"extra={sorted(actual_names - _EXPECTED_TOOL_NAMES)}"
        )

    def test_each_registry_entry_is_name_desc_handler_triple(self):
        """L1 anchor: 每个 _TOOL_REGISTRY 条目是 (name, description, handler) 三元组。

        锚点：FastMCP add_tool(fn, name, description, annotations) 协议要求
        每个注册条目提供 name / description / handler function。
        MCP spec: tool 必须有 name (string) 和 description (string)。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for entry in _TOOL_REGISTRY:
            assert isinstance(entry, tuple | list), (
                f"registry entry 必须是 tuple/list，实际 {type(entry).__name__}: {entry}"
            )
            assert len(entry) == 3, (
                f"registry entry 必须是 3 元组 (name, description, handler)，"
                f"实际 {len(entry)} 元: {entry}"
            )

    def test_tool_names_are_non_empty_strings(self):
        """L1 anchor: 每个 tool name 是非空 string，锚点：MCP spec name 字段定义。

        MCP spec: tool.name 必须是 string 类型，非空。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, _handler in _TOOL_REGISTRY:
            assert isinstance(name, str), (
                f"tool name 必须是 str，实际 {type(name).__name__}: {name!r}"
            )
            assert len(name) > 0, f"tool name 不能为空字符串"

    def test_tool_descriptions_are_non_empty_strings(self):
        """L1 anchor: 每个 tool description 是非空 string，锚点：MCP spec description 字段定义。

        MCP spec: tool.description 必须是 string 类型，描述工具用途。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, desc, _handler in _TOOL_REGISTRY:
            assert isinstance(desc, str), (
                f"tool {name} description 必须是 str，"
                f"实际 {type(desc).__name__}"
            )
            assert len(desc) > 0, f"tool {name} description 不能为空字符串"

    def test_tool_handlers_are_callable(self):
        """L1 anchor: 每个 tool handler 是 callable，锚点：FastMCP add_tool 协议。

        FastMCP: add_tool 的第一个参数是 callable（函数），MCP server 通过
        该 callable 执行 tool 请求。非 callable 无法注册。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, handler in _TOOL_REGISTRY:
            assert callable(handler), (
                f"tool {name} handler 必须是 callable，"
                f"实际 {type(handler).__name__}"
            )

    def test_tool_handlers_are_async_coroutines(self):
        """L1 anchor: 每个 tool handler 是 async coroutine function，锚点：FastMCP async 协议。

        FastMCP: tool handler 必须是 async def 定义的协程函数，
        MCP server 通过 await 调用 handler。sync 函数无法注册为 FastMCP tool。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, handler in _TOOL_REGISTRY:
            assert inspect.iscoroutinefunction(handler), (
                f"tool {name} handler 必须是 async coroutine function，"
                f"实际 {type(handler).__name__}"
            )

    def test_tool_handlers_have_return_type_annotation(self):
        """L1 anchor: 每个 tool handler 有返回类型注解，锚点：FastMCP outputSchema 生成协议。

        FastMCP: 从 handler 的返回类型注解自动生成 outputSchema (JSON Schema)。
        无返回注解的 handler 无法生成 outputSchema，违反 MCP spec。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, handler in _TOOL_REGISTRY:
            sig = inspect.signature(handler)
            assert sig.return_annotation is not inspect.Signature.empty, (
                f"tool {name} handler 必须有返回类型注解（FastMCP 用其生成 outputSchema）"
            )

    def test_tool_handlers_have_ctx_parameter(self):
        """L1 anchor: 每个 tool handler 有 ctx 参数，锚点：FastMCP Context 注入协议。

        FastMCP: tool handler 的 ctx 参数接收 MCP Context 对象，
        用于注入 request_id / 日志 / 进度报告。ctx 通常是最后一个参数且默认 None。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, handler in _TOOL_REGISTRY:
            sig = inspect.signature(handler)
            assert "ctx" in sig.parameters, (
                f"tool {name} handler 必须有 ctx 参数（FastMCP Context 注入），"
                f"实际参数: {list(sig.parameters.keys())}"
            )

    def test_tool_handlers_have_annotated_input_parameters(self):
        """L1 anchor: 每个 tool handler 的输入参数有类型注解，锚点：FastMCP inputSchema 生成协议。

        FastMCP: 从 handler 参数的类型注解自动生成 inputSchema (JSON Schema)。
        无注解的参数无法生成 inputSchema，违反 MCP spec。
        ctx 参数例外（FastMCP 内部注入，不出现在 inputSchema 中）。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        for name, _desc, handler in _TOOL_REGISTRY:
            sig = inspect.signature(handler)
            for param_name, param in sig.parameters.items():
                if param_name in ("ctx", "self"):
                    continue
                assert param.annotation is not inspect.Parameter.empty, (
                    f"tool {name} 参数 {param_name} 必须有类型注解"
                    f"（FastMCP 用其生成 inputSchema）"
                )

    def test_toolannotations_matrix_covers_all_tools(self):
        """L1 anchor: ToolAnnotations 矩阵覆盖全部 29 个 tool，锚点：MCP spec + 设计文档 0.4 节。

        MCP spec: 每个 tool 可选附带 ToolAnnotations 语义提示。
        设计文档 0.4 节: 全部 29 个 tool 都有 ToolAnnotations 定义。
        """
        from senseframe.mcp.server import _ANNOTATIONS
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        annotated_names = set(_ANNOTATIONS.keys())
        registered_names = {entry[0] for entry in _TOOL_REGISTRY}
        assert annotated_names == registered_names, (
            f"ToolAnnotations 矩阵未覆盖全部 tool: "
            f"missing={sorted(registered_names - annotated_names)} "
            f"extra={sorted(annotated_names - registered_names)}"
        )

    def test_toolannotations_has_four_boolean_hints(self):
        """L1 anchor: 每个 ToolAnnotations 含 4 个布尔语义提示，锚点：MCP spec ToolAnnotations 定义。

        MCP spec: ToolAnnotations 含 readOnlyHint / destructiveHint /
        idempotentHint / openWorldHint 四个布尔字段，描述工具语义。
        """
        from senseframe.mcp.server import _ANNOTATIONS

        for name, annotations in _ANNOTATIONS.items():
            for hint in _TOOLANNOTATION_HINTS:
                assert hasattr(annotations, hint), (
                    f"tool {name} 的 ToolAnnotations 缺少 {hint} 字段"
                )
                value = getattr(annotations, hint)
                assert isinstance(value, bool), (
                    f"tool {name} 的 ToolAnnotations.{hint} 必须是 bool，"
                    f"实际 {type(value).__name__}: {value!r}"
                )

    def test_no_duplicate_tool_names_in_registry(self):
        """L1 anchor: 注册表中无重复 tool 名称，锚点：MCP spec tool.name 唯一性。

        MCP spec: tool.name 在同一 server 内必须唯一，客户端按 name 调用 tool。
        """
        from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

        names = [entry[0] for entry in _TOOL_REGISTRY]
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"注册表中有重复 tool 名称: {sorted(duplicates)}"
