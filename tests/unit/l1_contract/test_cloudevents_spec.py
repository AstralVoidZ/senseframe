"""L1 契约测试：CloudEvents 1.0 规范（CNCF）。

锚点来源：CloudEvents 1.0 规范（CNCF）。
https://github.com/cloudevents/spec/blob/v1.0/spec.md

CloudEvents 1.0 REQUIRED 上下文属性（spec §3.1）：
- id / source / type / specversion

CloudEvents 1.0 OPTIONAL 上下文属性（spec §3.2）：
- time / datacontenttype / subject

本测试硬编码 CloudEvents 1.0 规范要求（必填字段集合 / specversion 值 / source URI 格式），
不引用源码常量（消除自证断言）。
"""
from __future__ import annotations

import pytest

from senseframe.orchestration import CloudEvent, make_event

# CloudEvents 1.0 规范：必填上下文属性（spec §3.1）
# 锚点：https://github.com/cloudevents/spec/blob/v1.0/spec.md#required-attributes
CLOUDEVENTS_REQUIRED_FIELDS = frozenset({"id", "source", "type", "specversion"})

# CloudEvents 1.0 规范：可选上下文属性（spec §3.2）
# 锚点：https://github.com/cloudevents/spec/blob/v1.0/spec.md#optional-attributes
CLOUDEVENTS_OPTIONAL_FIELDS = frozenset({"data", "datacontenttype", "subject", "time"})

# CloudEvents 1.0 规范要求 specversion == "1.0"（spec §3.1）
CLOUDEVENTS_SPEC_VERSION = "1.0"


@pytest.mark.l1_contract
class TestCloudEventsSpecContract:
    """验证 SenseFrame CloudEvent 符合 CloudEvents 1.0 规范契约。"""

    def test_required_fields_populated(self):
        """L1 anchor: 必填字段 id/source/type/specversion 构造后非空，锚点 CloudEvents 1.0 spec §3.1。"""
        event = CloudEvent(
            source="/senseframe/pipeline/run_1",
            type="senseframe.pipeline.started",
        )
        for field in CLOUDEVENTS_REQUIRED_FIELDS:
            value = getattr(event, field, None)
            assert value, f"必填字段 {field!r} 不应为空（CloudEvents 1.0 REQUIRED）"

    def test_specversion_is_one_dot_zero(self):
        """L1 anchor: specversion == '1.0'，锚点 CloudEvents 1.0 spec §3.1。"""
        event = CloudEvent(source="/s", type="t")
        assert event.specversion == CLOUDEVENTS_SPEC_VERSION

    def test_time_populated_after_construction(self):
        """L1 anchor: time 字段构造后自动填充，锚点 CloudEvents 1.0 time 属性（SenseFrame 总是填充）。"""
        event = CloudEvent(source="/s", type="t")
        assert event.time, "time 字段构造后应非空"

    def test_optional_fields_present(self):
        """L1 anchor: CloudEvent 支持可选字段 data/datacontenttype/subject，锚点 CloudEvents 1.0 spec §3.2。"""
        event = CloudEvent(source="/s", type="t")
        # CloudEvents 1.0 OPTIONAL 属性应可在 CloudEvent 上访问
        assert hasattr(event, "data"), "CloudEvent 应支持 data 字段（CloudEvents 1.0 OPTIONAL）"
        assert hasattr(event, "datacontenttype"), \
            "CloudEvent 应支持 datacontenttype 字段（CloudEvents 1.0 OPTIONAL）"
        assert hasattr(event, "subject"), \
            "CloudEvent 应支持 subject 字段（CloudEvents 1.0 OPTIONAL）"

    def test_source_is_uri_reference(self):
        """L1 anchor: source 是 URI-reference 格式，锚点 CloudEvents 1.0 spec §3.2。"""
        event = CloudEvent(
            source="/senseframe/pipeline/run_1",
            type="senseframe.pipeline.started",
        )
        # CloudEvents 1.0: source MUST be a non-empty URI-reference
        # SenseFrame 使用绝对路径 URI 格式（/senseframe/pipeline/{run_id}）
        assert event.source, "source 不应为空（CloudEvents 1.0 REQUIRED + URI-reference）"
        assert event.source.startswith("/"), \
            f"source 应为 URI 路径格式，实际 {event.source!r}"

    def test_make_event_source_is_uri(self):
        """L1 anchor: make_event 生成 source 为 URI 格式，锚点 CloudEvents 1.0 spec §3.2。"""
        event = make_event("senseframe.pipeline.started", "run_abc", {"phase": "Running"})
        assert event.source.startswith("/"), \
            f"make_event source 应为 URI，实际 {event.source!r}"

    def test_id_auto_generated_unique(self):
        """L1 anchor: id 构造后自动生成且唯一，锚点 CloudEvents 1.0 spec §3.1 (id REQUIRED)。"""
        e1 = CloudEvent(source="/s", type="t")
        e2 = CloudEvent(source="/s", type="t")
        assert e1.id, "id 应自动生成（CloudEvents 1.0 REQUIRED）"
        assert e2.id, "id 应自动生成（CloudEvents 1.0 REQUIRED）"
        assert e1.id != e2.id, "不同事件 id 应唯一（CloudEvents 1.0 spec）"

    def test_to_dict_includes_required_fields(self):
        """L1 anchor: to_dict 序列化含所有必填字段，锚点 CloudEvents 1.0 spec §3.1。"""
        event = CloudEvent(
            source="/senseframe/pipeline/run_1",
            type="senseframe.pipeline.started",
        )
        d = event.to_dict()
        for field in CLOUDEVENTS_REQUIRED_FIELDS:
            assert field in d, f"to_dict 应包含必填字段 {field!r}"
            assert d[field], f"to_dict 中 {field!r} 不应为空"
        # time 也应序列化（SenseFrame 契约：总是填充 time）
        assert "time" in d and d["time"]

    def test_to_dict_specversion_is_one(self):
        """L1 anchor: to_dict 中 specversion == '1.0'，锚点 CloudEvents 1.0 spec。"""
        d = CloudEvent(source="/s", type="t").to_dict()
        assert d["specversion"] == CLOUDEVENTS_SPEC_VERSION

    def test_to_dict_includes_optional_fields(self):
        """L1 anchor: to_dict 含可选字段 data/datacontenttype/subject，锚点 CloudEvents 1.0 spec §3.2。"""
        event = CloudEvent(source="/s", type="t", data={"k": "v"})
        d = event.to_dict()
        assert "data" in d
        assert "datacontenttype" in d
        assert "subject" in d, "to_dict 应包含 subject（CloudEvents 1.0 OPTIONAL）"

    def test_to_json_is_valid_json_with_required_fields(self):
        """L1 anchor: to_json 可反序列化且含必填字段，锚点 CloudEvents 1.0 spec §3.1。"""
        import json
        event = CloudEvent(
            source="/senseframe/pipeline/run_x",
            type="senseframe.pipeline.started",
            data={"phase": "running"},
        )
        parsed = json.loads(event.to_json())
        for field in CLOUDEVENTS_REQUIRED_FIELDS:
            assert field in parsed, f"to_json 反序列化后应含必填字段 {field!r}"
        assert parsed["specversion"] == CLOUDEVENTS_SPEC_VERSION
