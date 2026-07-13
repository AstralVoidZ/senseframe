"""RFC-003 DSP-5 introspect 模块测试。"""

import json

import pytest

from senseframe.introspect import (
    context_schema,
    context_describe,
    stage_io,
    list_stages,
    pipeline_graph,
    data_bundle_schema,
    data_bundle_describe,
    data_profile_schema,
    data_profile_describe,
)
from senseframe.engine.runner.pipeline import PipelineContext
from senseframe.scenes.base import DatasetBundle
from senseframe.core.profiler import DataProfile
from senseframe.engine.config import ExperimentConfig, SceneConfig, InputFeature, OutputFeature, TrainerConfig


def _make_test_config():
    """构造测试用 ExperimentConfig。"""
    return ExperimentConfig(
        scene=SceneConfig(name="test", dataset="test", model_id="test"),
        input_features=[InputFeature(name="features", type="tabular", shape=[10])],
        output_features=[OutputFeature(name="label", type="category", num_classes=3)],
        trainer=TrainerConfig(epochs=1),
    )


class TestContextSchema:
    def test_returns_json_serializable(self):
        schema = context_schema()
        json.dumps(schema)  # 不抛异常

    def test_has_schema_version(self):
        schema = context_schema()
        assert "schema_version" in schema
        assert schema["schema_version"] == "1.0.0"

    def test_has_fields(self):
        schema = context_schema()
        assert "fields" in schema
        assert len(schema["fields"]) > 0

    def test_fields_have_required_keys(self):
        schema = context_schema()
        for f in schema["fields"]:
            assert "name" in f
            assert "type" in f
            assert "fill_stage" in f
            assert "has_default" in f

    def test_config_field_exists(self):
        schema = context_schema()
        names = [f["name"] for f in schema["fields"]]
        assert "config" in names

    def test_config_fill_stage_is_init(self):
        schema = context_schema()
        for f in schema["fields"]:
            if f["name"] == "config":
                assert f["fill_stage"] == "init"
                break


class TestContextDescribe:
    def test_empty_context(self):
        config = _make_test_config()
        ctx = PipelineContext(config=config)
        desc = context_describe(ctx)
        assert "completed_fields" in desc
        assert "config" in desc["completed_fields"]
        assert "extra_keys" in desc
        assert "trial_id" in desc
        assert "completed_stages" in desc

    def test_json_serializable(self):
        config = _make_test_config()
        ctx = PipelineContext(config=config)
        desc = context_describe(ctx)
        json.dumps(desc)


class TestStageIO:
    def test_list_all_stages(self):
        result = stage_io()
        assert "stages" in result
        assert len(result["stages"]) == 9

    def test_get_single_stage(self):
        result = stage_io("validate")
        assert result["name"] == "validate"
        assert "reads" in result
        assert "writes" in result
        assert "description" in result

    def test_validate_reads_config(self):
        result = stage_io("validate")
        read_names = [f["name"] for f in result["reads"]]
        assert "config" in read_names

    def test_validate_writes_scene(self):
        result = stage_io("validate")
        write_names = [f["name"] for f in result["writes"]]
        assert "scene" in write_names
        assert "meta" in write_names

    def test_nonexistent_stage(self):
        result = stage_io("nonexistent")
        assert "error" in result
        assert "available" in result

    def test_json_serializable(self):
        result = stage_io()
        json.dumps(result)


class TestListStages:
    def test_returns_9_stages(self):
        stages = list_stages()
        assert len(stages) == 9
        assert "validate" in stages
        assert "export" in stages

    def test_order(self):
        stages = list_stages()
        assert stages[0] == "validate"
        assert stages[-1] == "export"


class TestPipelineGraph:
    def test_returns_fields_mapping(self):
        graph = pipeline_graph()
        assert "fields" in graph
        assert len(graph["fields"]) > 0

    def test_scene_producer_is_validate(self):
        graph = pipeline_graph()
        scene = graph["fields"].get("scene", {})
        assert "validate" in scene.get("producers", [])

    def test_scene_consumers_include_build(self):
        graph = pipeline_graph()
        scene = graph["fields"].get("scene", {})
        assert "build" in scene.get("consumers", [])

    def test_config_has_no_producer(self):
        graph = pipeline_graph()
        config = graph["fields"].get("config", {})
        assert config.get("producers", []) == []

    def test_json_serializable(self):
        graph = pipeline_graph()
        json.dumps(graph)


class TestDataBundleSchema:
    def test_has_filling_rules(self):
        schema = data_bundle_schema()
        assert "filling_rules" in schema
        assert "supervised" in schema["filling_rules"]
        assert "self_supervised" in schema["filling_rules"]

    def test_supervised_rule(self):
        schema = data_bundle_schema()
        rule = schema["filling_rules"]["supervised"]
        assert rule["train"] == "required"
        assert rule["test"] == "required"
        assert rule["unsupervised"] == "forbidden"

    def test_self_supervised_rule(self):
        schema = data_bundle_schema()
        rule = schema["filling_rules"]["self_supervised"]
        assert rule["train"] == "forbidden"
        assert rule["unsupervised"] == "required"
        assert rule["supervised_finetune"] == "required"

    def test_json_serializable(self):
        schema = data_bundle_schema()
        json.dumps(schema)


class TestDataBundleDescribe:
    def test_empty_bundle_supervised(self):
        bundle = DatasetBundle()
        desc = data_bundle_describe(bundle, "supervised")
        assert desc["filled_fields"] == []
        assert len(desc["validation_errors"]) > 0  # train/test required

    def test_empty_bundle_self_supervised(self):
        bundle = DatasetBundle()
        desc = data_bundle_describe(bundle, "self_supervised")
        assert len(desc["validation_errors"]) > 0

    def test_json_serializable(self):
        bundle = DatasetBundle()
        desc = data_bundle_describe(bundle)
        json.dumps(desc)


class TestDataProfileSchema:
    def test_has_new_fields(self):
        schema = data_profile_schema()
        names = [f["name"] for f in schema["fields"]]
        assert "dtypes" in names
        assert "feature_names" in names
        assert "nullable" in names
        assert "shapes" in names

    def test_json_serializable(self):
        schema = data_profile_schema()
        json.dumps(schema)


class TestDataProfileDescribe:
    def test_empty_profile(self):
        profile = DataProfile()
        desc = data_profile_describe(profile)
        assert "n_features" in desc
        assert "dtype_distribution" in desc
        assert "nullable_ratio" in desc

    def test_json_serializable(self):
        profile = DataProfile()
        desc = data_profile_describe(profile)
        json.dumps(desc)
