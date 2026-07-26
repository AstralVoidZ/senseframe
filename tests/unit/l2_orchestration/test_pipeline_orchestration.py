"""L2 编排层测试：Pipeline 运行时行为。

锚点：RFC Phase D（Pipeline 9 stage）、设计文档 0.6 节（PipelineRun 状态机）、
RFC-004 方案 F（确定性资源释放）。

L2 测试验证编排运行时行为（stage 间数据流、跳过/续跑、hook 顺序、
错误传播、资源清理），不验证契约（L1）或算法（L3）。
"""
from __future__ import annotations

import json

import pytest

from senseframe.engine.config import (
    ExperimentConfig,
    InputFeature,
    OutputFeature,
    SceneConfig,
    TrainerConfig,
)
from senseframe.engine.runner.pipeline.context import PipelineContext
from senseframe.engine.runner.pipeline.runtime import Pipeline

from tests.fakes.fake_trainer import FakeTrainer

pytestmark = pytest.mark.l2_orchestration


@pytest.fixture
def minimal_config():
    """构造最小 ExperimentConfig，仅满足 PipelineContext 构造需求。"""
    return ExperimentConfig(
        scene=SceneConfig(
            name="test_scene",
            dataset="test_dataset",
            model_id="TestModel",
            learning_mode="supervised",
            data_root="/tmp",
        ),
        input_features=[InputFeature(name="x", type="tabular", shape=[10])],
        output_features=[OutputFeature(name="y", type="category", num_classes=2)],
        trainer=TrainerConfig(epochs=1, batch_size=4),
    )


class TestStageDataFlow:
    """stage 间数据流传递。"""

    def test_stage_data_flow_output_feeds_next_stage(self, minimal_config):
        """stage_a 写入 ctx.model_id，stage_b 能读到。"""

        def stage_a(ctx):
            ctx.model_id = "test_model_42"
            return ctx

        def stage_b(ctx):
            ctx.extra["read_value"] = ctx.model_id
            return ctx

        pipeline = Pipeline(stages=[("a", stage_a), ("b", stage_b)])
        ctx = PipelineContext(config=minimal_config)
        result = pipeline.run(ctx)

        assert result.error is None
        assert result.context.model_id == "test_model_42"
        assert result.context.extra["read_value"] == "test_model_42"


class TestSkipAndResume:
    """跳过/续跑行为。"""

    def test_pipeline_run_skips_completed_stages(self, minimal_config):
        """ctx.completed_stages 中的 stage 被跳过，其余正常执行。"""
        call_log = []

        def stage_a(ctx):
            call_log.append("a")
            return ctx

        def stage_b(ctx):
            call_log.append("b")
            return ctx

        def stage_c(ctx):
            call_log.append("c")
            return ctx

        pipeline = Pipeline(stages=[("a", stage_a), ("b", stage_b), ("c", stage_c)])
        ctx = PipelineContext(config=minimal_config)
        ctx.completed_stages = ["a"]
        result = pipeline.run(ctx)

        assert result.error is None
        assert "a" not in call_log
        assert "b" in call_log
        assert "c" in call_log

    def test_non_serializable_stages_forced_rerun(self, minimal_config, tmp_path):
        """checkpoint 中的不可序列化 stage（load）被移除并强制重跑。"""
        call_log = []

        def stage_validate(ctx):
            call_log.append("validate")
            return ctx

        def stage_load(ctx):
            call_log.append("load")
            return ctx

        def stage_export(ctx):
            call_log.append("export")
            return ctx

        # 构造 checkpoint：validate + load 均标记为已完成
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        ckpt_data = {
            "completed_stages": ["validate", "load"],
            "config_hash": "",  # 空 → 跳过 hash 校验
            "stage_outputs": {},
        }
        ckpt_path.write_text(json.dumps(ckpt_data), encoding="utf-8")

        pipeline = Pipeline(stages=[
            ("validate", stage_validate),
            ("load", stage_load),
            ("export", stage_export),
        ])
        ctx = PipelineContext(config=minimal_config)
        ctx.stage_checkpoint_path = ckpt_path
        ctx.output_dir = output_dir
        result = pipeline.run(ctx)

        assert result.error is None
        # validate 是可序列化 stage，被跳过
        assert "validate" not in call_log
        # load 是不可序列化 stage（_NON_SERIALIZABLE_STAGES），被强制重跑
        assert "load" in call_log
        # export 未在 completed_stages 中，正常执行
        assert "export" in call_log


class TestHooks:
    """hook 插入与执行顺序。"""

    def test_hook_execution_order(self, minimal_config):
        """before → 原始 stage → after 顺序执行。"""
        call_log = []

        def stage_train(ctx):
            call_log.append("train")
            return ctx

        def before_hook(ctx):
            call_log.append("before_train")
            return ctx

        def after_hook(ctx):
            call_log.append("after_train")
            return ctx

        pipeline = Pipeline(stages=[("train", stage_train)])
        pipeline.before("train", before_hook)
        pipeline.after("train", after_hook)

        ctx = PipelineContext(config=minimal_config)
        result = pipeline.run(ctx)

        assert result.error is None
        assert call_log == ["before_train", "train", "after_train"]


class TestErrorPropagation:
    """错误传播与 pipeline 中止。"""

    def test_error_propagation_stops_pipeline(self, minimal_config):
        """stage b 抛异常 → result 含 error，failed_stage="b"，stage c 未执行。"""
        call_log = []

        def stage_a(ctx):
            call_log.append("a")
            return ctx

        def stage_b(ctx):
            call_log.append("b")
            raise RuntimeError("stage b exploded")

        def stage_c(ctx):
            call_log.append("c")
            return ctx

        pipeline = Pipeline(stages=[("a", stage_a), ("b", stage_b), ("c", stage_c)])
        ctx = PipelineContext(config=minimal_config)
        result = pipeline.run(ctx)

        assert result.error is not None
        assert isinstance(result.error, RuntimeError)
        assert result.context.failed_stage == "b"
        assert "a" in call_log
        assert "b" in call_log
        assert "c" not in call_log


class TestResourceCleanup:
    """RFC-004 方案 F：确定性资源释放。"""

    def test_resource_cleanup_on_failure(self, minimal_config):
        """stage 失败后 release_resources 仍被调用（finally 块），trainer 置 None。"""

        def stage_fail(ctx):
            raise RuntimeError("boom")

        pipeline = Pipeline(stages=[("fail", stage_fail)])
        ctx = PipelineContext(config=minimal_config)
        ctx.trainer = FakeTrainer()
        result = pipeline.run(ctx)

        assert result.error is not None
        # release_resources 在 finally 中执行，trainer 被置 None
        assert ctx.trainer is None


class TestCheckpoint:
    """checkpoint 写入行为。"""

    def test_checkpoint_written_after_each_stage(self, minimal_config, tmp_path):
        """pipeline 成功完成后 output_dir 下存在 pipeline_checkpoint.json。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        def stage_a(ctx):
            return ctx

        def stage_b(ctx):
            return ctx

        pipeline = Pipeline(stages=[("a", stage_a), ("b", stage_b)])
        ctx = PipelineContext(config=minimal_config)
        ctx.output_dir = output_dir
        result = pipeline.run(ctx)

        assert result.error is None
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        assert ckpt_path.exists()
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        assert "a" in ckpt["completed_stages"]
        assert "b" in ckpt["completed_stages"]
