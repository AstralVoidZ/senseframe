"""端到端集成测试：HPO 与 ExplorationTracker 统一。

验证 run_hpo 的 tracker 参数正确记录 trial，统一 HPO 数值超参搜索
与策略空间搜索的探索视图（RFC-002 P2）。

标记 @pytest.mark.e2e，默认跳过，使用 -m e2e 显式启用。

实现说明：
- run_hpo 默认目标函数调用 run_experiment（完整 pipeline），故需 generic_test 场景
- n_trials=1 + epochs=1 控制耗时（CPU 上约 30-60s）
- sampler=random 提升确定性，pruner=none 避免提前剪枝导致 tracker 无记录
"""
import pytest


@pytest.mark.e2e
class TestHPOUnified:
    """HPO + ExplorationTracker 联动测试。"""

    def test_hpo_writes_to_tracker(self, experiment_config):
        """run_hpo(config, tracker=my_tracker) 后 tracker.history 含 trial。"""
        pytest.importorskip("optuna")
        pytest.importorskip("torch")
        pytest.importorskip("pytorch_lightning")

        from senseframe.engine.hpo import run_hpo
        from senseframe.exploration import ExplorationTracker

        config = experiment_config
        # 启用 HPO：1 个 trial，random sampler 提升确定性，none pruner 避免 trial 被剪枝
        config.hpo.enabled = True
        config.hpo.n_trials = 1
        config.hpo.sampler = "random"
        config.hpo.pruner = "none"
        config.hpo.metric = "val_loss"
        config.hpo.direction = "minimize"
        # 安全网：避免极端超参组合导致 hang（120s 足够 1 trial）
        config.hpo.timeout = 120

        tracker = ExplorationTracker()
        output = run_hpo(config, tracker=tracker)

        # 1. tracker 应记录至少 1 个 trial（成功或失败均记录）
        assert len(tracker.history) > 0, "tracker.history 为空，HPO 未写入 trial"

        # 2. tracker 中 trial 结构应完整
        first_trial = tracker.history[0]
        assert "strategy" in first_trial, "trial 缺少 strategy 字段"
        assert "timestamp" in first_trial, "trial 缺少 timestamp 字段"
        assert "trial_id" in first_trial, "trial 缺少 trial_id 字段"
        # HPO 默认目标函数写入 feedback（status=success 或 numerical_instability）
        assert first_trial.get("feedback") is not None, "trial 未写入 feedback"
        assert "status" in first_trial["feedback"]

        # 3. output.tracker 应与传入的 tracker 是同一对象（统一探索视图）
        assert output.tracker is tracker, "output.tracker 与传入 tracker 不是同一对象"

        # 4. HPOOutput 基本字段
        assert output.n_trials >= 1, "HPOOutput.n_trials 应 >= 1"
        assert output.direction == "minimize"
        assert output.metric == "val_loss"
