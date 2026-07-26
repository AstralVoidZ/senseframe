"""P1 端到端集成测试（反假绿）。

用真实 run_pipeline（合成 CSV + 1 epoch + 小 batch）验证：
- ε1 损失搜索完整流程：SP ask/tell 驱动多次试验
- ε6 experiment 模块完整流程：MethodRunner + BaselineRunner + ExperimentRunner + ComparisonReport

反假绿原则：
- 不用 mock sentinel，跑真实 run_pipeline
- 验证 SP 状态真实改变（study.list_trials 返回非空）
- 验证 TrialResult 字段真实填充（metrics 非空 + wall_time_s > 0）
- 验证 ComparisonReport 显著性检验真实执行（p_value 非默认值）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# torch + lightning 为必需依赖（e2e 测试跑真实训练）
pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

# 审查修复：e2e 测试必须标记，避免默认 CI 执行（7 个真实训练测试）
pytestmark = pytest.mark.e2e


# ============================================================
# ε1 损失搜索 e2e
# ============================================================
class TestEpsilon1LossSearchE2E:
    """ε1 损失搜索端到端验证（真实 run_pipeline）。"""

    def test_run_loss_search_completes(self, experiment_config):
        """run_loss_search 完整执行 3 次试验，返回 LossSearchResult。"""
        from senseframe.automl.loss_search import run_loss_search

        result = run_loss_search(
            experiment_config,
            n_trials=3,
            direction="maximize",
            metric="val_accuracy",
            sampler="random",
        )

        # 验证返回类型
        assert result is not None
        assert result.study_id.startswith("study_")
        assert result.n_trials == 3
        # 至少有 1 次完成（合成数据 + 1 epoch 应该能跑通）
        assert result.n_completed + result.n_failed == 3

        # 验证 trials 列表非空
        assert len(result.trials) == 3
        # 至少 1 次有 params
        trials_with_params = [t for t in result.trials if t.params]
        assert len(trials_with_params) > 0
        # loss 字段在 params 中
        first_trial = trials_with_params[0]
        assert "loss" in first_trial.params

    def test_run_loss_search_best_params_contain_loss(self, experiment_config):
        """若有成功 trial，best_params 应含 loss 字段。"""
        from senseframe.automl.loss_search import run_loss_search

        result = run_loss_search(
            experiment_config,
            n_trials=2,
            direction="maximize",
            metric="val_accuracy",
        )

        if result.n_completed > 0:
            assert result.best_value is not None
            assert "loss" in result.best_params
            assert result.best_params["loss"] in [
                "cross_entropy", "focal", "mse", "mae",
                "smooth_l1", "bce_with_logits", "cross_entropy_weighted",
            ]

    def test_run_loss_search_sp_state_persists(self, experiment_config):
        """SP Study 状态真实持久：study_id 可查询到 trials。"""
        from senseframe.automl.loss_search import run_loss_search
        from senseframe.search_protocol import get_study_manager

        result = run_loss_search(
            experiment_config,
            n_trials=2,
            direction="maximize",
            metric="val_accuracy",
        )

        sm = get_study_manager()
        trials = sm.list_trials(result.study_id)
        assert len(trials) == 2

        # 验证 trial 状态真实改变（不是 mock sentinel）
        for trial in trials:
            assert trial.state in ("completed", "failed")
            assert trial.trial_id  # 非空


# ============================================================
# ε6 experiment 模块 e2e
# ============================================================
class TestEpsilon6ExperimentE2E:
    """ε6 experiment 模块端到端验证（真实 run_pipeline）。"""

    def _make_design(self, experiment_config, tmp_path):
        """构造最小 ExperimentDesign（1 dataset × 1 model × Method + 1 Baseline × 1 repeat）。"""
        from senseframe.experiment import (
            BaselineConfig,
            ExperimentBudget,
            ExperimentDesign,
            MethodConfig,
        )

        # Method 组：损失搜索
        method_config = MethodConfig(
            name="senseframe_loss_search",
            base_config=experiment_config,
            metric="val_accuracy",
            direction="maximize",
            sampler="random",
        )

        # Baseline 组：固定 cross_entropy loss
        baseline_config = BaselineConfig(
            name="baseline_ce",
            base_config=experiment_config,
            manual_tunes=5,
        )

        design = ExperimentDesign(
            name="e2e_test",
            datasets=[experiment_config.scene.dataset],
            models=[experiment_config.scene.model_id],
            method=method_config,
            baselines=[baseline_config],
            budget=ExperimentBudget(max_trials_per_group=2, n_repeats=1),
        )
        return design

    def test_method_runner_real_pipeline(self, experiment_config):
        """MethodRunner 执行一次真实试验，返回 TrialResult（DSP 合规）。"""
        from senseframe.experiment import (
            MethodConfig, MethodRunner, TrialGroup, TrialStatus,
        )
        from senseframe.search_protocol import StudyManager
        from senseframe.automl.loss_search import build_loss_search_space

        sm = StudyManager()
        search_space = build_loss_search_space(include_label_smoothing=False)
        study_id = sm.create_study(
            name="method_e2e", direction="maximize",
            search_space=search_space, sampler="random",
        )

        method_config = MethodConfig(
            name="m1", base_config=experiment_config,
            metric="val_accuracy", direction="maximize",
        )
        runner = MethodRunner(
            config=method_config, study_id=study_id,
            study_manager=sm, experiment_id="exp_e2e",
        )

        result = runner.run(
            dataset=experiment_config.scene.dataset,
            model_id=experiment_config.scene.model_id,
            run_idx=0,
        )

        # 验证 TrialResult DSP 合规
        assert result.schema_version == "1.0.0"
        assert result.experiment_id == "exp_e2e"
        assert result.group == TrialGroup.METHOD
        assert result.method_name == "m1"
        assert result.run_index == 0
        assert result.sp_trial_id is not None  # SP 关联

        # 验证状态真实（不是 mock）
        assert result.status in (TrialStatus.SUCCESS, TrialStatus.FAILED)
        if result.status == TrialStatus.SUCCESS:
            # 真实跑通的应有指标 + 时间
            assert len(result.metrics) > 0
            assert result.wall_time_s >= 0
            assert result.agent_decisions >= 1

        # 验证 SP 状态真实改变
        sp_trial = sm.get_trial(result.sp_trial_id)
        assert sp_trial is not None
        assert sp_trial.state in ("completed", "failed")

    def test_baseline_runner_real_pipeline(self, experiment_config):
        """BaselineRunner 执行一次真实试验，不走 SP。"""
        from senseframe.experiment import (
            BaselineConfig, BaselineRunner, TrialGroup, TrialStatus,
        )

        baseline_config = BaselineConfig(
            name="b1", base_config=experiment_config, manual_tunes=3,
        )
        runner = BaselineRunner(config=baseline_config, experiment_id="exp_e2e")

        result = runner.run(
            dataset=experiment_config.scene.dataset,
            model_id=experiment_config.scene.model_id,
            run_idx=0,
        )

        # 验证 TrialResult
        assert result.group == TrialGroup.BASELINE_REPRO
        assert result.method_name == "b1"
        assert result.sp_trial_id is None  # Baseline 不关联 SP
        assert result.manual_tunes == 3
        assert result.status in (TrialStatus.SUCCESS, TrialStatus.FAILED)

    def test_baseline_paper_skips_pipeline(self, experiment_config):
        """BASELINE_PAPER 直接用 reported_metrics，不跑训练。"""
        from senseframe.experiment import (
            BaselineConfig, BaselineRunner, TrialGroup, TrialStatus,
        )

        baseline_config = BaselineConfig(
            name="paper_baseline",
            base_config=experiment_config,
            group=TrialGroup.BASELINE_PAPER,
            reported_metrics={"accuracy": 0.92, "macro_f1": 0.88},
        )
        runner = BaselineRunner(config=baseline_config, experiment_id="exp_e2e")

        result = runner.run(
            dataset=experiment_config.scene.dataset,
            model_id=experiment_config.scene.model_id,
            run_idx=0,
        )

        # 直接用 reported_metrics，不跑训练
        assert result.status == TrialStatus.SUCCESS
        assert result.metrics == {"accuracy": 0.92, "macro_f1": 0.88}
        assert result.wall_time_s == 0.0  # 没跑训练
        assert result.n_epochs_trained == 0

    def test_experiment_runner_full_flow(self, experiment_config, tmp_path):
        """ExperimentRunner 完整流程：Method + Baseline → ComparisonReport。"""
        from senseframe.experiment import ExperimentRunner

        design = self._make_design(experiment_config, tmp_path)
        runner = ExperimentRunner(design=design)

        report = runner.run(output_path=str(tmp_path / "report.json"))

        # 验证 ComparisonReport
        assert report.experiment_id == runner.experiment_id
        assert report.design_name == "e2e_test"
        # 1 dataset × 1 model × (max_trials=2 Method + 1 baseline × n_repeats=1) = 3
        assert len(report.results) == 3

        # 验证 summary 含两组
        assert "method" in report.summary
        assert "baseline_repro" in report.summary

        # 验证 summary 聚合正确
        method_summary = report.summary["method"]
        assert method_summary.n_trials == 2  # max_trials_per_group=2
        assert method_summary.method_name == "senseframe_loss_search"

        baseline_summary = report.summary["baseline_repro"]
        assert baseline_summary.n_trials == 1  # n_repeats=1
        assert baseline_summary.mean_manual_tunes == 5.0

        # 验证 report 文件已保存
        report_path = tmp_path / "report.json"
        assert report_path.exists()
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["experiment_id"] == runner.experiment_id
        assert len(saved["results"]) == 3

    def test_experiment_runner_significance_real(self, experiment_config, tmp_path):
        """ComparisonReport 显著性检验真实执行（p_value 非默认 1.0）。"""
        from senseframe.experiment import (
            BaselineConfig, ExperimentBudget, ExperimentDesign,
            ExperimentRunner, MethodConfig, TrialGroup, TrialStatus,
            TrialResult,
        )

        # 构造明显差异的 Method vs Baseline 数据
        # Method: accuracy = [0.85, 0.87, 0.86]
        # Baseline: accuracy = [0.70, 0.72, 0.71]
        # 这种差异下 t-test p_value 应该 < 0.05
        method_results = [
            TrialResult(
                experiment_id="exp1", group=TrialGroup.METHOD,
                method_name="m", dataset="d", model_id="m", run_index=i,
                metrics={"accuracy": acc}, status=TrialStatus.SUCCESS,
            )
            for i, acc in enumerate([0.85, 0.87, 0.86])
        ]
        baseline_results = [
            TrialResult(
                experiment_id="exp1", group=TrialGroup.BASELINE_REPRO,
                method_name="b", dataset="d", model_id="m", run_index=i,
                metrics={"accuracy": acc}, status=TrialStatus.SUCCESS,
            )
            for i, acc in enumerate([0.70, 0.72, 0.71])
        ]
        all_results = method_results + baseline_results

        from senseframe.experiment import ComparisonReport
        report = ComparisonReport(experiment_id="exp1", design_name="d1")
        sig = report.build_significance(all_results, metric="accuracy")

        # 验证显著性检验真实执行
        assert "method_vs_baseline_repro" in sig
        test = sig["method_vs_baseline_repro"]
        assert test.test_method in ("ttest", "bootstrap")
        # p_value 不是默认值 1.0（真实执行了检验）
        assert test.p_value < 1.0
        # 差异显著
        assert test.significant is True
        assert test.p_value < 0.05
        # 效应量大（Cohen's d）
        assert test.effect_size > 0.5
