"""端到端集成测试：完整 run_pipeline 流程。

验证 validate→preflight→resolve→load→build→probe_vram→train→eval→export 9 个 stage
能跑通一个最小数据集。

标记 @pytest.mark.e2e，默认跳过，使用 -m e2e 显式启用。

实现说明：
- 用 generic_test 场景（_TestableGenericContainer，见 conftest.py）+ 合成 CSV
- 直接调用 pipeline.run(ctx) 而非 run_pipeline(config)，以便访问 ctx.exploration_history / ctx.extra
- 预置一次探索试验（ctx.record_trial），让 stage_eval 能回写 feedback 到 exploration_history
"""
import json
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestFullPipeline:
    """完整 pipeline 端到端测试。"""

    def test_generic_csv_pipeline(self, experiment_config):
        """generic 场景 + 合成 CSV → run_pipeline → 验证输出。"""
        pytest.importorskip("torch")
        pytest.importorskip("pytorch_lightning")

        from senseframe.engine.runner import Pipeline, PipelineContext

        config = experiment_config

        # 构造 PipelineContext，预置一次探索试验
        # （stage_eval 仅在 exploration_history 非空时回写 feedback）
        pipeline = Pipeline.default()
        ctx = PipelineContext(config=config)
        ctx.record_trial(strategy={"loss": "cross_entropy", "learning_rate": 0.001})

        # 运行 pipeline（9 个 stage 依次执行）
        result = pipeline.run(ctx)
        ctx = result.context

        # 1. 验证无异常且成功
        assert result.error is None, f"pipeline 抛异常: {result.error}"
        assert ctx.output is not None, "ctx.output 未被 stage_preflight 初始化"
        assert ctx.output.status == "success", (
            f"pipeline 状态: {ctx.output.status}, "
            f"error_code: {getattr(ctx.output, 'error_code', None)}"
        )

        # 2. 验证输出目录与产物（stage_export 应写入）
        output_dir = Path(ctx.output.output_dir)
        assert output_dir.exists(), f"输出目录不存在: {output_dir}"
        assert (output_dir / "metadata.json").exists(), "metadata.json 未生成"
        assert (output_dir / "exploration.json").exists(), "exploration.json 未生成"

        # 3. 验证 metadata.json 内容
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["model_id"] == "GenericMLP"
        assert metadata["dataset"] == "synthetic"
        assert metadata["learning_mode"] == "supervised"
        assert "final_eval" in metadata
        assert "config" in metadata

        # 4. 验证 exploration_history 已回写 feedback（闭合训练→反馈回路）
        assert len(ctx.exploration_history) > 0, "exploration_history 为空"
        last_trial = ctx.exploration_history[-1]
        assert last_trial.get("feedback") is not None, "最近 trial 未回写 feedback"
        assert "status" in last_trial["feedback"], "feedback 缺少 status 字段"
        assert last_trial.get("status") == "completed", (
            f"trial status 应为 completed，实际: {last_trial.get('status')}"
        )
        assert last_trial.get("result") is not None, "trial 未回写 result"

    def test_pipeline_context_get_set(self, experiment_config):
        """验证 PipelineContext.get/set 路由逻辑（顺带验证 extra 字段）。"""
        pytest.importorskip("torch")

        from senseframe.engine.runner import PipelineContext

        ctx = PipelineContext(config=experiment_config)

        # 已定义字段走 setattr
        ctx.set("trial_id", "test-trial-001")
        assert ctx.trial_id == "test-trial-001"
        assert ctx.get("trial_id") == "test-trial-001"

        # 未定义字段走 extra
        ctx.set("custom_key", {"nested": "value"})
        assert ctx.get("custom_key") == {"nested": "value"}
        assert ctx.extra["custom_key"] == {"nested": "value"}

        # get 默认值
        assert ctx.get("nonexistent", "default") == "default"
