"""端到端集成测试：探索-反馈闭环。

验证训练 → feedback → recommend_next 的完整链路：
- 首次训练生成结构化 feedback（stage_eval 的 analyze_training_result）
- feedback 回写到 exploration_history 最后一次 trial
- ExplorationTracker.recommend_next 感知 feedback status，生成定向推荐

标记 @pytest.mark.e2e，默认跳过，使用 -m e2e 显式启用。
"""
import pytest


@pytest.mark.e2e
class TestClosedLoop:
    """探索-反馈闭环端到端测试。"""

    def test_feedback_drives_recommendation(self, experiment_config):
        """训练 → feedback → recommend_next 验证推荐方向匹配 feedback status。"""
        pytest.importorskip("torch")
        pytest.importorskip("pytorch_lightning")

        from senseframe.engine.runner import Pipeline, PipelineContext
        from senseframe.exploration import ExplorationTracker

        config = experiment_config

        # 首次训练：预置探索试验，让 stage_eval 能回写 feedback
        pipeline = Pipeline.default()
        ctx = PipelineContext(config=config)
        ctx.record_trial(strategy={"loss": "cross_entropy", "learning_rate": 0.001})

        result = pipeline.run(ctx)
        ctx = result.context

        # 1. pipeline 应成功完成
        assert result.error is None, f"pipeline 抛异常: {result.error}"
        assert ctx.output is not None
        assert ctx.output.status == "success", (
            f"pipeline 状态: {ctx.output.status}"
        )

        # 2. 读取 feedback（stage_eval 写入 ctx.extra）
        # P5 P2-7 阶段2：feedback 现在是 FeedbackResult dataclass，用属性访问
        feedback = ctx.extra.get("feedback")
        assert feedback is not None, "ctx.extra 未生成 feedback（stage_eval 未执行或异常）"
        # 兼容 FeedbackResult dataclass 和 dict 两种形态
        if hasattr(feedback, "status"):
            assert feedback.status is not None, "feedback 缺少 status 字段"
            assert hasattr(feedback, "diagnosis")
            assert hasattr(feedback, "suggestions")
            fb_status = feedback.status
        else:
            assert "status" in feedback, "feedback 缺少 status 字段"
            assert "diagnosis" in feedback
            assert "suggestions" in feedback
            fb_status = feedback["status"]

        # 3. feedback 应已回写到 exploration_history 最后一次 trial
        assert len(ctx.exploration_history) > 0
        last_trial = ctx.exploration_history[-1]
        assert last_trial.get("feedback") is not None, "最近 trial 未回写 feedback"
        # tracker 中的 feedback 已被 hpo.py 转为 dict
        assert last_trial["feedback"]["status"] == fb_status

        # 4. recommend_next 应返回非空列表
        tracker = ExplorationTracker(ctx.exploration_history)
        recs = tracker.recommend_next(task_type="classification", top_k=5)
        assert len(recs) > 0, "recommend_next 未返回推荐"

        # 5. 验证推荐方向匹配 feedback status
        # feedback 感知推荐优先级最高，置于 recs 列表前列（见 ExplorationTracker.recommend_next）
        status = fb_status
        first = recs[0]
        strategy_str = str(first.get("strategy", {}))

        if status == "overfitting":
            # 过拟合 → 推荐数据增强 / weight_decay / dropout
            assert "augment" in strategy_str or "weight_decay" in strategy_str \
                or "dropout" in strategy_str, (
                f"overfitting → 首条推荐应含 augment/weight_decay/dropout，实际: {first}"
            )
        elif status == "underfitting":
            # 欠拟合 → 推荐更强 loss / epochs_scale
            assert "loss" in strategy_str or "epochs_scale" in strategy_str, (
                f"underfitting → 首条推荐应含 loss/epochs_scale，实际: {first}"
            )
        elif status == "numerical_instability":
            # 数值不稳定 → 推荐稳定 loss / gradient_clip_val
            assert "loss" in strategy_str or "gradient_clip_val" in strategy_str, (
                f"numerical_instability → 首条推荐应含 stable loss/gradient_clip_val，"
                f"实际: {first}"
            )
        # converged / success：推荐方向不强制断言（兼容性矩阵/HPO 候选即可）

        # 6. tracker.last_feedback 应与 pipeline 生成的 feedback 一致
        assert tracker.last_feedback()["status"] == fb_status
