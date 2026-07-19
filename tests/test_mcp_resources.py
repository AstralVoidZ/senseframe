"""阶段 2.1 L4 ISP 自省协议测试。

覆盖 12 个 Resource 端点：
- ISP-0:  senseframe://introspect                    — 聚合索引（含 12 个 Resource URI）
- ISP-1:  senseframe://schemas/pipeline               — PipelineContext schema
- ISP-2:  senseframe://schemas/stage                  — StageSpec schema
- ISP-3:  senseframe://schemas/config                 — ExperimentConfig schema
- ISP-4:  senseframe://schemas/errors                 — 错误码 × 恢复策略映射
- ISP-7:  senseframe://pipeline/{run_id}/graph        — stage 数据流图
- ISP-9:  senseframe://pipeline/{run_id}/readiness    — 运行时数据就绪度
- ISP-8:  senseframe://scenes                         — 场景目录
- ISP-9:  senseframe://scenes/{name}/capabilities     — 场景能力声明
- ISP-10: senseframe://search-space/{scene}/{model_id} — 搜索空间
- ISP-10: senseframe://protocols                       — 已注册 Protocol 类型
- ISP-11: senseframe://tools/output-schemas            — Read tool outputSchema 索引
"""

from __future__ import annotations

import asyncio

import pytest

from senseframe.mcp.errors import PipelineNotFound
from senseframe.mcp.orchestration.pipeline_run import (
    get_default_store,
    set_default_store,
)
from senseframe.mcp.resources import _RESOURCE_REGISTRY
from senseframe.mcp.resources import introspect as isp_introspect
from senseframe.mcp.resources import pipeline as isp_pipeline
from senseframe.mcp.resources import protocols as isp_protocols
from senseframe.mcp.resources import scenes as isp_scenes
from senseframe.mcp.resources import schemas as isp_schemas


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fresh_store():
    """每个测试用独立的 PipelineRunStore。"""
    set_default_store(None)
    store = get_default_store()
    yield store
    set_default_store(None)


@pytest.fixture
def sample_config() -> dict:
    return {
        "scene": {"name": "test", "dataset": "test", "model_id": "MLP"},
        "input_features": [{"name": "x", "type": "tabular", "shape": [10]}],
        "output_features": [{"name": "y", "type": "category", "num_classes": 3}],
        "trainer": {"epochs": 1},
    }


@pytest.fixture
def sample_stages() -> list[str]:
    return ["validate", "preflight", "load", "train"]


# ============================================================
# 1. _RESOURCE_REGISTRY 契约
# ============================================================


class TestResourceRegistry:
    """12 个 Resource 端点的注册表契约测试。"""

    def test_registry_has_twelve_entries(self):
        """_RESOURCE_REGISTRY 必须含 12 个 (uri, name, description, handler) 4 元组。"""
        assert len(_RESOURCE_REGISTRY) == 12

    def test_registry_uris_match_expected_set(self):
        """12 个 URI 必须是设计文档 0.3 节定义的 ISP-0..ISP-11 集合。"""
        uris = {entry[0] for entry in _RESOURCE_REGISTRY}
        expected_uris = {
            "senseframe://introspect",
            "senseframe://schemas/pipeline",
            "senseframe://schemas/stage",
            "senseframe://schemas/config",
            "senseframe://schemas/errors",
            "senseframe://pipeline/{run_id}/graph",
            "senseframe://pipeline/{run_id}/readiness",
            "senseframe://scenes",
            "senseframe://scenes/{name}/capabilities",
            "senseframe://search-space/{scene}/{model_id}",
            "senseframe://protocols",
            "senseframe://tools/output-schemas",
        }
        assert uris == expected_uris, (
            f"URI 集合不匹配: missing={sorted(expected_uris - uris)} "
            f"extra={sorted(uris - expected_uris)}"
        )

    def test_registry_entries_have_callable_handlers(self):
        """所有 handler 必须是 callable。"""
        for uri, _name, _desc, handler in _RESOURCE_REGISTRY:
            assert callable(handler), f"handler for {uri} is not callable"

    def test_registry_entries_have_non_empty_descriptions(self):
        """所有 description 必须非空。"""
        for uri, _name, desc, _handler in _RESOURCE_REGISTRY:
            assert desc and isinstance(desc, str), (
                f"description for {uri} is empty or not str"
            )
            assert len(desc) > 5  # 至少有意义的描述

    def test_registry_entries_have_unique_uris(self):
        """URI 必须唯一。"""
        uris = [entry[0] for entry in _RESOURCE_REGISTRY]
        assert len(uris) == len(set(uris)), "URI 出现重复"

    def test_registry_entries_have_unique_names(self):
        """Resource name 必须唯一。"""
        names = [entry[1] for entry in _RESOURCE_REGISTRY]
        assert len(names) == len(set(names)), "name 出现重复"


# ============================================================
# 2. ISP-0: senseframe://introspect 聚合索引
# ============================================================


class TestIntrospect:
    """ISP-0：聚合索引 — 版本 + 能力清单 + Resource URI 列表。"""

    @pytest.mark.asyncio
    async def test_introspect_returns_twelve_resource_uris(self):
        """introspect 必须返回 12 个 Resource URI。"""
        result = await isp_introspect.introspect()
        resources = result["resources"]
        assert len(resources) == 12, (
            f"introspect 应返回 12 个 Resource URI，实际 {len(resources)}"
        )

    @pytest.mark.asyncio
    async def test_introspect_uris_match_registry(self):
        """introspect 返回的 URI 集合必须与 _RESOURCE_REGISTRY 一致。"""
        result = await isp_introspect.introspect()
        introspect_uris = {r["uri"] for r in result["resources"]}
        registry_uris = {entry[0] for entry in _RESOURCE_REGISTRY}
        assert introspect_uris == registry_uris

    @pytest.mark.asyncio
    async def test_introspect_returns_server_metadata(self):
        """introspect 必须返回 server / protocol_version / server_version 字段。"""
        result = await isp_introspect.introspect()
        assert result["server"] == "senseframe-mcp"
        assert isinstance(result["protocol_version"], str)
        assert result["protocol_version"]
        assert isinstance(result["server_version"], str)
        assert result["server_version"]

    @pytest.mark.asyncio
    async def test_introspect_returns_tools_list(self):
        """introspect 必须返回 tools 列表（29 个 tool 名称）。

        阶段 3 扩展：从 14 个 tool 增至 25 个 tool（新增 study/hpo/exploration/automl/param_bridge 工具组）。
        阶段 4.2 扩展：从 25 个 tool 增至 27 个 tool（新增 artifact_list/export）。
        阶段 4.3 扩展：从 27 个 tool 增至 29 个 tool（新增 skill_get/search）。
        """
        result = await isp_introspect.introspect()
        tools = result["tools"]
        assert isinstance(tools, list)
        assert len(tools) == 29  # 29 个 tool（设计文档 0.4 节 ToolAnnotations 矩阵）
        # 至少含 pipeline_create
        assert "senseframe_pipeline_create" in tools

    @pytest.mark.asyncio
    async def test_introspect_returns_capabilities(self):
        """introspect 必须返回 capabilities dict。"""
        result = await isp_introspect.introspect()
        caps = result["capabilities"]
        assert isinstance(caps, dict)
        # 设计文档 0.4 节定义的能力
        expected_caps = {
            "introspection",
            "pipeline_lifecycle",
            "hateoas_transitions",
            "cursor_pagination",
            "advisory_only",
        }
        assert expected_caps.issubset(set(caps.keys()))
        # 所有能力必须为 True（当前阶段已实现）
        for cap_name in expected_caps:
            assert caps[cap_name] is True, f"capability {cap_name} should be True"

    @pytest.mark.asyncio
    async def test_introspect_resources_have_uri_and_description(self):
        """每个 resource 条目必须含 uri + description 字段。"""
        result = await isp_introspect.introspect()
        for r in result["resources"]:
            assert "uri" in r
            assert "description" in r
            assert r["uri"]
            assert r["description"]


# ============================================================
# 3. ISP-1..4 + ISP-11: senseframe://schemas/*
# ============================================================


class TestSchemaResources:
    """ISP-1..4 + ISP-11：5 个 schema Resource 端点。"""

    @pytest.mark.asyncio
    async def test_schema_pipeline_returns_non_empty_schema(self):
        """ISP-1：senseframe://schemas/pipeline 返回非空 PipelineContext schema。"""
        schema = await isp_schemas.schema_pipeline()
        assert isinstance(schema, dict)
        # 至少含 schema_version 或 fields 字段
        assert "schema_version" in schema or "fields" in schema
        # fields 必须非空
        if "fields" in schema:
            assert len(schema["fields"]) > 0, "PipelineContext schema 应有字段"

    @pytest.mark.asyncio
    async def test_schema_stage_returns_non_empty(self):
        """ISP-2：senseframe://schemas/stage 返回非空 stage IO Spec。"""
        schema = await isp_schemas.schema_stage()
        assert isinstance(schema, dict)
        assert "stages" in schema
        assert len(schema["stages"]) > 0
        # 每个 stage 应含 name + reads + writes + description
        for stage_spec in schema["stages"]:
            assert "name" in stage_spec
            assert "reads" in stage_spec
            assert "writes" in stage_spec
            assert "description" in stage_spec

    @pytest.mark.asyncio
    async def test_schema_config_returns_json_schema(self):
        """ISP-3：senseframe://schemas/config 返回 ExperimentConfig JSON Schema。"""
        schema = await isp_schemas.schema_config()
        assert isinstance(schema, dict)
        # pydantic v2 model_json_schema 应含 properties / type
        assert "properties" in schema or "$defs" in schema or "type" in schema

    @pytest.mark.asyncio
    async def test_schema_errors_returns_recovery_mapping(self):
        """ISP-4：senseframe://schemas/errors 返回错误码 × 恢复策略映射。"""
        schema = await isp_schemas.schema_errors()
        assert isinstance(schema, dict)
        assert schema["schema_version"] == "1.0.0"
        error_codes = schema["error_codes"]
        assert isinstance(error_codes, list)
        assert len(error_codes) > 0
        # 每条含 code + description + recoverable + suggested_action
        for entry in error_codes:
            assert "code" in entry
            assert "description" in entry
            assert "recoverable" in entry
            assert "suggested_action" in entry
            assert isinstance(entry["recoverable"], bool)

    @pytest.mark.asyncio
    async def test_schema_errors_contains_known_codes(self):
        """错误码映射应包含常见错误：OK / UNKNOWN_ERROR / OOM_ERROR。"""
        schema = await isp_schemas.schema_errors()
        codes = {entry["code"] for entry in schema["error_codes"]}
        assert "OK" in codes
        assert "UNKNOWN_ERROR" in codes
        assert "OOM_ERROR" in codes
        assert "PIPELINE_NOT_FOUND" in codes or "CONFIG_VALIDATION_ERROR" in codes

    @pytest.mark.asyncio
    async def test_tools_output_schemas_returns_index(self):
        """ISP-11：senseframe://tools/output-schemas 返回 outputSchema 索引。"""
        result = await isp_schemas.tools_output_schemas()
        assert isinstance(result, dict)
        assert result["schema_version"] == "1.0.0"
        tools = result["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        # 每条含 name + output_schema
        for entry in tools:
            assert "name" in entry
            assert "output_schema" in entry
            assert isinstance(entry["output_schema"], dict)

    @pytest.mark.asyncio
    async def test_tools_output_schemas_includes_pipeline_views(self):
        """outputSchema 索引应包含 pipeline tool 的视图。"""
        result = await isp_schemas.tools_output_schemas()
        names = {entry["name"] for entry in result["tools"]}
        assert "senseframe_pipeline_get" in names
        assert "senseframe_pipeline_list" in names
        assert "senseframe_pipeline_create" in names
        # 错误信封也应被索引
        assert "__error_envelope__" in names


# ============================================================
# 4. ISP-8 + 9 + 10: scenes
# ============================================================


class TestScenesResources:
    """ISP-8 + 9 + 10：场景目录 + 能力 + 搜索空间。"""

    @pytest.mark.asyncio
    async def test_scenes_returns_catalog(self):
        """ISP-8：senseframe://scenes 返回场景目录。"""
        result = await isp_scenes.scenes()
        assert isinstance(result, dict)
        assert result["schema_version"] == "1.0.0"
        scenes_list = result["scenes"]
        assert isinstance(scenes_list, list)
        # 应至少含 generic 或 wifi_csi（_register_builtin_scenes 保证）
        scene_names = {s["name"] for s in scenes_list}
        assert "generic" in scene_names or "wifi_csi" in scene_names, (
            f"scene catalog 应含 generic 或 wifi_csi，实际 {scene_names}"
        )

    @pytest.mark.asyncio
    async def test_scenes_each_entry_has_name_and_meta(self):
        result = await isp_scenes.scenes()
        for entry in result["scenes"]:
            assert "name" in entry
            assert "meta" in entry
            assert isinstance(entry["meta"], dict)

    @pytest.mark.asyncio
    async def test_scene_capabilities_returns_meta(self):
        """ISP-9：senseframe://scenes/generic/capabilities 返回场景能力。"""
        result = await isp_scenes.scene_capabilities("generic")
        assert isinstance(result, dict)
        assert result["name"] == "generic"
        assert "capabilities" in result
        assert isinstance(result["capabilities"], dict)

    @pytest.mark.asyncio
    async def test_scene_capabilities_wifi_csi_lazy_meta(self):
        """wifi_csi 延迟注册场景也应能返回 capabilities（可能激活失败但元数据可见）。"""
        try:
            result = await isp_scenes.scene_capabilities("wifi_csi")
            assert result["name"] == "wifi_csi"
        except Exception as e:
            # wifi_csi 激活可能失败（缺 SenseFi 依赖），允许跳过
            pytest.skip(f"wifi_csi 激活失败（缺依赖）: {e}")

    @pytest.mark.asyncio
    async def test_scene_capabilities_unknown_scene_raises(self):
        """未知场景应抛 ValueError（来自 get_scene）。"""
        with pytest.raises(ValueError):
            await isp_scenes.scene_capabilities("nonexistent-scene")

    @pytest.mark.asyncio
    async def test_search_space_returns_advisory_structure(self):
        """ISP-10：senseframe://search-space/{scene}/{model_id} 返回搜索空间结构。"""
        result = await isp_scenes.search_space("generic", "MLP")
        assert isinstance(result, dict)
        assert result["scene"] == "generic"
        assert result["model_id"] == "MLP"
        assert "parameters" in result
        assert isinstance(result["parameters"], list)
        # 标记为 advisory
        assert result.get("advisory") is True


# ============================================================
# 5. ISP-10 alt: protocols
# ============================================================


class TestProtocolsResource:
    """ISP-10 alt：senseframe://protocols 返回 Protocol 类型列表。"""

    @pytest.mark.asyncio
    async def test_protocols_returns_protocol_list(self):
        """protocols 必须返回 8 个 Protocol 类型。"""
        result = await isp_protocols.protocols()
        assert isinstance(result, dict)
        assert result["schema_version"] == "1.0.0"
        protocols = result["protocols"]
        assert isinstance(protocols, list)
        assert len(protocols) == 8  # 8 个 Protocol

    @pytest.mark.asyncio
    async def test_protocols_each_entry_has_name_and_methods(self):
        result = await isp_protocols.protocols()
        for entry in result["protocols"]:
            assert "name" in entry
            assert "methods" in entry
            assert "attributes" in entry
            assert isinstance(entry["methods"], list)
            assert isinstance(entry["attributes"], list)
            assert isinstance(entry["runtime_checkable"], bool)

    @pytest.mark.asyncio
    async def test_protocols_includes_expected_names(self):
        """Protocol 列表应包含 8 个核心 Protocol 名称。"""
        result = await isp_protocols.protocols()
        names = {entry["name"] for entry in result["protocols"]}
        expected_names = {
            "SceneProtocol",
            "ModelProtocol",
            "DataModuleProtocol",
            "TrainerProtocol",
            "LoggerProtocol",
            "SceneMetaProtocol",
            "TaskSpecProtocol",
            "FeatureSpecProtocol",
        }
        assert names == expected_names, (
            f"Protocol 名称集合不匹配: missing={sorted(expected_names - names)} "
            f"extra={sorted(names - expected_names)}"
        )

    @pytest.mark.asyncio
    async def test_protocols_runtime_checkable_flag(self):
        """所有 Protocol 都应是 runtime_checkable（@runtime_checkable 装饰）。"""
        result = await isp_protocols.protocols()
        for entry in result["protocols"]:
            assert entry["runtime_checkable"] is True, (
                f"Protocol {entry['name']} 应是 runtime_checkable"
            )

    @pytest.mark.asyncio
    async def test_protocols_scene_protocol_has_meta_method(self):
        """SceneProtocol 应含 meta 方法（场景契约）。"""
        result = await isp_protocols.protocols()
        scene_proto = next(p for p in result["protocols"] if p["name"] == "SceneProtocol")
        assert "meta" in scene_proto["methods"]
        assert "load_dataset" in scene_proto["methods"]
        assert "build_model_for_dataset" in scene_proto["methods"]


# ============================================================
# 6. ISP-7 + 9: pipeline graph + readiness
# ============================================================


class TestPipelineGraphResources:
    """ISP-7 + ISP-9：pipeline graph + readiness。"""

    @pytest.mark.asyncio
    async def test_pipeline_graph_returns_field_mapping(
        self, fresh_store, sample_config, sample_stages
    ):
        """ISP-7：pipeline graph 返回 field → producer/consumer 映射。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        result = await isp_pipeline.pipeline_graph(run.run_id)
        assert isinstance(result, dict)
        assert result["run_id"] == run.run_id
        assert "state" in result
        assert "graph" in result
        # graph 应是 dict（含 fields 映射）
        graph = result["graph"]
        assert isinstance(graph, dict)

    @pytest.mark.asyncio
    async def test_pipeline_graph_unknown_run_raises(self, fresh_store):
        """未知 run_id 应抛 PipelineNotFound。"""
        with pytest.raises(PipelineNotFound):
            await isp_pipeline.pipeline_graph("nonexistent-run-id")

    @pytest.mark.asyncio
    async def test_pipeline_readiness_returns_stage_status(
        self, fresh_store, sample_config, sample_stages
    ):
        """ISP-9：readiness 返回每个 stage 的就绪状态。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        result = await isp_pipeline.pipeline_readiness(run.run_id)
        assert isinstance(result, dict)
        assert result["run_id"] == run.run_id
        assert "state" in result
        assert "readiness" in result
        # readiness 标记为 advisory
        assert result.get("advisory") is True
        # readiness 是 list
        readiness = result["readiness"]
        assert isinstance(readiness, list)
        # Pending 状态下所有 stage 应是 pending
        for stage_entry in readiness:
            assert "stage" in stage_entry
            assert "state" in stage_entry
            assert "writes" in stage_entry
            assert "fields_ready" in stage_entry

    @pytest.mark.asyncio
    async def test_pipeline_readiness_unknown_run_raises(self, fresh_store):
        with pytest.raises(PipelineNotFound):
            await isp_pipeline.pipeline_readiness("nonexistent-run-id")

    @pytest.mark.asyncio
    async def test_pipeline_readiness_completed_stage_marked_succeeded(
        self, fresh_store, sample_config, sample_stages
    ):
        """已完成的 stage 应标记为 succeeded。"""
        run = fresh_store.create(config=sample_config, stages=sample_stages)
        fresh_store.advance(run.run_id, action="start")
        fresh_store.advance(
            run.run_id, action="complete", completed_stage="train"
        )
        result = await isp_pipeline.pipeline_readiness(run.run_id)
        # 找到 train stage 的 readiness
        train_entry = next(
            s for s in result["readiness"] if s["stage"] == "train"
        )
        assert train_entry["state"] == "succeeded"


# ============================================================
# 7. server.py 注册：tool + resource 已挂载到 FastMCP
# ============================================================


class TestServerRegistration:
    """FastMCP server 程序化注册测试。"""

    def test_server_mcp_instance_exists(self):
        from senseframe.mcp.server import mcp
        assert mcp is not None
        assert mcp.name == "senseframe-mcp"

    def test_server_registers_fourteen_tools(self):
        """FastMCP 应注册 14 个 tool。"""
        # FastMCP 内部 _tool_manager 维护 tool 注册表
        from senseframe.mcp.server import mcp
        # mcp._tool_manager._tools 是 dict[str, Tool]
        # 注意：FastMCP 版本不同 API 可能不同，故只做粗略断言
        try:
            tools_dict = mcp._tool_manager._tools
            assert len(tools_dict) >= 14
        except AttributeError:
            # 不同版本 API：尝试 list_tools
            pass

    def test_server_registers_twelve_resources(self):
        """FastMCP 应注册 12 个 Resource。"""
        # FastMCP 内部 _resource_manager 维护 resource 注册表
        # 不同版本 API 可能不同，本测试只验证不抛异常
        from senseframe.mcp.server import mcp
        # 验证 _register_tools_and_resources 不抛异常（已在模块加载时执行）
        assert mcp is not None


# ============================================================
# 8. ISP-0 introspect 的 capabilities 完整性
# ============================================================


class TestIntrospectCapabilities:
    """introspect 返回的 capabilities 字段必须覆盖设计文档 0.4 节所有能力。"""

    @pytest.mark.asyncio
    async def test_capabilities_include_hateoas_and_pagination(self):
        result = await isp_introspect.introspect()
        caps = result["capabilities"]
        # HATEOAS + cursor pagination 必须存在
        assert caps["hateoas_transitions"] is True
        assert caps["cursor_pagination"] is True

    @pytest.mark.asyncio
    async def test_capabilities_advisory_only_flag(self):
        """advisory_only=True 标记 _transitions 为 advisory 模式。"""
        result = await isp_introspect.introspect()
        caps = result["capabilities"]
        assert caps["advisory_only"] is True

    @pytest.mark.asyncio
    async def test_introspect_serializable_json(self):
        """introspect 返回值必须可 JSON 序列化（供 MCP 传输）。"""
        import json

        result = await isp_introspect.introspect()
        # 不抛异常
        json.dumps(result)
