"""P1 ε6 experiment 模块单元测试（反假绿）。

测试原则：
- DSP 合规用 dataclasses.fields 反射验证（不硬编码字段列表）
- SP 驱动验证用 grep 实证（检查 method.py 含 sm.ask / sm.tell）
- BaselineRunner 不走 SP 用 grep 实证（检查 baseline.py 不含 sm.ask）
- 显著性检验用真实数据（不 mock scipy）
- 完整流程测试在 integration/test_p1_e2e.py（用真实 run_pipeline）
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from senseframe.experiment import (
    BaselineConfig,
    BaselineRunner,
    ComparisonReport,
    ExperimentBudget,
    ExperimentDesign,
    GroupSummary,
    MethodConfig,
    MethodRunner,
    SignificanceTest,
    TrialGroup,
    TrialResult,
    TrialStatus,
    run_significance_test,
)


# ============================================================
# TrialResult DSP 合规测试
# ============================================================
class TestTrialResultDSP:
    """验证 TrialResult 满足 DSP（数据结构协议）。"""

    def test_schema_version_default(self):
        """schema_version 默认为 '1.0.0'。"""
        tr = TrialResult()
        assert tr.schema_version == "1.0.0"

    def test_schema_returns_json_schema(self):
        """schema() 返回 JSON Schema dict（含 title / properties / schema_version）。"""
        schema = TrialResult.schema()
        assert isinstance(schema, dict)
        assert schema["title"] == "TrialResult"
        assert "properties" in schema
        assert "schema_version" in schema["properties"]

    def test_schema_covers_all_fields(self):
        """schema() 用 dataclasses.fields 反射，覆盖所有字段（不硬编码）。"""
        schema = TrialResult.schema()
        props = schema["properties"]
        # 用反射获取所有字段名，验证 schema 覆盖
        field_names = {f.name for f in fields(TrialResult)}
        for name in field_names:
            assert name in props, f"schema 缺少字段: {name}"

    def test_schema_has_enum_for_group_and_status(self):
        """schema 中 group / status 字段含 enum 约束。"""
        schema = TrialResult.schema()
        assert "enum" in schema["properties"]["group"]
        assert set(schema["properties"]["group"]["enum"]) == {
            g.value for g in TrialGroup
        }
        assert "enum" in schema["properties"]["status"]
        assert set(schema["properties"]["status"]["enum"]) == {
            s.value for s in TrialStatus
        }

    def test_describe_returns_runtime_state(self):
        """describe() 返回运行时状态摘要。"""
        tr = TrialResult(
            experiment_id="exp1",
            group=TrialGroup.METHOD,
            method_name="senseframe_loss",
            dataset="UT_HAR",
            model_id="mlp",
            run_index=0,
            metrics={"accuracy": 0.85},
            wall_time_s=120.5,
            agent_decisions=10,
        )
        d = tr.describe()
        assert d["schema_version"] == "1.0.0"
        assert d["experiment_id"] == "exp1"
        assert d["group"] == "method"
        assert d["n_metrics"] == 1
        assert d["wall_time_s"] == 120.5
        assert d["agent_decisions"] == 10

    def test_to_dict_serializable(self):
        """to_dict 返回可 JSON 序列化的 dict（Enum 转 value）。"""
        tr = TrialResult(
            experiment_id="exp1",
            group=TrialGroup.BASELINE_PAPER,
            status=TrialStatus.FAILED,
            metrics={"accuracy": 0.8},
        )
        d = tr.to_dict()
        # Enum 应转为 value
        assert d["group"] == "baseline_paper"
        assert d["status"] == "failed"
        # 可序列化
        json.dumps(d)

    def test_from_dict_roundtrip(self):
        """from_dict 反序列化（Enum 自动转换）+ 往返一致。"""
        original = TrialResult(
            experiment_id="exp1",
            group=TrialGroup.METHOD,
            method_name="m1",
            dataset="d1",
            model_id="mlp",
            run_index=2,
            metrics={"accuracy": 0.9},
            status=TrialStatus.SUCCESS,
        )
        d = original.to_dict()
        restored = TrialResult.from_dict(d)
        assert restored.experiment_id == original.experiment_id
        assert restored.group == original.group
        assert restored.status == original.status
        assert restored.metrics == original.metrics
        assert restored.run_index == original.run_index


# ============================================================
# Enum 测试
# ============================================================
class TestEnums:
    """验证 TrialGroup / TrialStatus Enum。"""

    def test_trial_group_values(self):
        """TrialGroup 含 METHOD / BASELINE_PAPER / BASELINE_REPRO。"""
        assert TrialGroup.METHOD.value == "method"
        assert TrialGroup.BASELINE_PAPER.value == "baseline_paper"
        assert TrialGroup.BASELINE_REPRO.value == "baseline_repro"

    def test_trial_status_values(self):
        """TrialStatus 含 SUCCESS / FAILED / PRUNED。"""
        assert TrialStatus.SUCCESS.value == "success"
        assert TrialStatus.FAILED.value == "failed"
        assert TrialStatus.PRUNED.value == "pruned"

    def test_trial_group_is_str_enum(self):
        """TrialGroup 是 str Enum（可直接比较字符串）。"""
        assert TrialGroup.METHOD == "method"


# ============================================================
# ExperimentDesign 验证测试
# ============================================================
class TestExperimentDesignValidation:
    """验证 ExperimentDesign / MethodConfig / BaselineConfig validate()。"""

    def _make_method_config(self):
        """构造最小 MethodConfig（不依赖真实数据集）。"""
        from senseframe.engine.config import (
            ExperimentConfig, InputFeature, OutputFeature,
            SceneConfig, TrainerConfig,
        )
        config = ExperimentConfig(
            scene=SceneConfig(name="generic_test", dataset="d", model_id="m",
                              learning_mode="supervised", data_root="/tmp",
                              params={"data_root": "/tmp"}),
            input_features=[InputFeature(name="f", type="tabular", shape=[10])],
            output_features=[OutputFeature(name="l", type="category", num_classes=3)],
            trainer=TrainerConfig(epochs=1, batch_size=4, logger="csv"),
            output_dir="/tmp",
        )
        return MethodConfig(name="m1", base_config=config)

    def test_method_config_validate_rejects_empty_name(self):
        """MethodConfig.validate 拒绝空 name。"""
        mc = self._make_method_config()
        mc.name = ""
        with pytest.raises(ValueError, match="name"):
            mc.validate()

    def test_method_config_validate_rejects_invalid_direction(self):
        """MethodConfig.validate 拒绝无效 direction。"""
        mc = self._make_method_config()
        mc.direction = "invalid"
        with pytest.raises(ValueError, match="direction"):
            mc.validate()

    def test_baseline_config_validate_rejects_method_group(self):
        """BaselineConfig.validate 拒绝 group=METHOD。"""
        mc = self._make_method_config()
        bc = BaselineConfig(name="b1", base_config=mc.base_config, group=TrialGroup.METHOD)
        with pytest.raises(ValueError, match="METHOD"):
            bc.validate()

    def test_experiment_budget_validate(self):
        """ExperimentBudget.validate 拒绝 <=0 的 max_trials_per_group。"""
        b = ExperimentBudget(max_trials_per_group=0)
        with pytest.raises(ValueError):
            b.validate()
        b2 = ExperimentBudget(n_repeats=0)
        with pytest.raises(ValueError):
            b2.validate()

    def test_experiment_design_validate_rejects_empty_datasets(self):
        """ExperimentDesign.validate 拒绝空 datasets。"""
        mc = self._make_method_config()
        design = ExperimentDesign(
            name="d1", datasets=[], models=["m"], method=mc,
        )
        with pytest.raises(ValueError, match="datasets"):
            design.validate()

    def test_experiment_design_validate_rejects_empty_models(self):
        """ExperimentDesign.validate 拒绝空 models。"""
        mc = self._make_method_config()
        design = ExperimentDesign(
            name="d1", datasets=["d"], models=[], method=mc,
        )
        with pytest.raises(ValueError, match="models"):
            design.validate()


# ============================================================
# MethodRunner SP 驱动验证（grep 实证）
# ============================================================
class TestMethodRunnerSPDriven:
    """反假绿：用 grep 实证验证 MethodRunner 通过 SP ask/tell 驱动。"""

    def test_method_source_contains_sm_ask(self):
        """method.py 源码含 sm.ask 调用。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "method.py"
        source = source_path.read_text(encoding="utf-8")
        assert ".ask(" in source, "method.py 未调用 sm.ask()"
        assert ".tell(" in source, "method.py 未调用 sm.tell()"

    def test_method_source_uses_apply_params(self):
        """method.py 源码调用 apply_params 应用 SP 参数。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "method.py"
        source = source_path.read_text(encoding="utf-8")
        assert "apply_params" in source, "method.py 未调用 apply_params"

    def test_method_source_uses_run_pipeline(self):
        """method.py 源码调用 run_pipeline 执行训练。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "method.py"
        source = source_path.read_text(encoding="utf-8")
        assert "run_pipeline" in source, "method.py 未调用 run_pipeline"

    def test_method_constructor_accepts_study_id(self):
        """MethodRunner 构造函数接受 study_id 参数。"""
        import inspect
        sig = inspect.signature(MethodRunner.__init__)
        assert "study_id" in sig.parameters
        assert "config" in sig.parameters


# ============================================================
# BaselineRunner 不走 SP 验证（grep 实证）
# ============================================================
class TestBaselineRunnerNoSP:
    """反假绿：用 grep 实证验证 BaselineRunner 不走 SP。"""

    def test_baseline_source_no_sm_ask(self):
        """baseline.py 源码不含 sm.ask 调用（Baseline 不走 SP 搜索）。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "baseline.py"
        source = source_path.read_text(encoding="utf-8")
        # baseline.py 不应调用 sm.ask（不走 SP 搜索）
        # 注意：允许 sm.tell 也不应存在（Baseline 不报告到 SP）
        assert ".ask(" not in source, "baseline.py 不应调用 sm.ask()（Baseline 不走 SP）"
        assert ".tell(" not in source, "baseline.py 不应调用 sm.tell()（Baseline 不报告到 SP）"

    def test_baseline_source_no_study_id(self):
        """baseline.py 源码不含 study_id（Baseline 不关联 SP Study）。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "baseline.py"
        source = source_path.read_text(encoding="utf-8")
        assert "study_id" not in source, "baseline.py 不应引用 study_id"

    def test_baseline_constructor_no_study_id(self):
        """BaselineRunner 构造函数不接受 study_id 参数。"""
        import inspect
        sig = inspect.signature(BaselineRunner.__init__)
        assert "study_id" not in sig.parameters

    def test_baseline_source_uses_run_pipeline(self):
        """baseline.py 源码调用 run_pipeline（BASELINE_REPRO 走训练）。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "experiment" / "baseline.py"
        source = source_path.read_text(encoding="utf-8")
        assert "run_pipeline" in source, "baseline.py 未调用 run_pipeline"


# ============================================================
# ComparisonReport 聚合测试
# ============================================================
class TestComparisonReport:
    """验证 ComparisonReport 聚合 + 显著性检验。"""

    def _make_results(self):
        """构造测试用 TrialResult 列表（Method + Baseline 各 3 个）。"""
        results = []
        # Method 组：accuracy = [0.85, 0.87, 0.86]
        for i, acc in enumerate([0.85, 0.87, 0.86]):
            results.append(TrialResult(
                experiment_id="exp1",
                group=TrialGroup.METHOD,
                method_name="senseframe",
                dataset="d1", model_id="m1", run_index=i,
                metrics={"accuracy": acc},
                wall_time_s=100.0,
                agent_decisions=5,
                status=TrialStatus.SUCCESS,
            ))
        # Baseline 组：accuracy = [0.75, 0.77, 0.76]
        for i, acc in enumerate([0.75, 0.77, 0.76]):
            results.append(TrialResult(
                experiment_id="exp1",
                group=TrialGroup.BASELINE_REPRO,
                method_name="sensefi",
                dataset="d1", model_id="m1", run_index=i,
                metrics={"accuracy": acc},
                wall_time_s=200.0,
                manual_tunes=10,
                status=TrialStatus.SUCCESS,
            ))
        return results

    def test_build_summary_groups_by_group(self):
        """build_summary 按 group 分组聚合。"""
        report = ComparisonReport(experiment_id="exp1", design_name="d1")
        results = self._make_results()
        summary = report.build_summary(results)
        assert "method" in summary
        assert "baseline_repro" in summary

    def test_build_summary_mean_metrics(self):
        """build_summary 计算均值指标正确。"""
        report = ComparisonReport(experiment_id="exp1", design_name="d1")
        results = self._make_results()
        summary = report.build_summary(results)
        method_summary = summary["method"]
        # Method accuracy = [0.85, 0.87, 0.86], mean = 0.86
        assert abs(method_summary.mean_metrics["accuracy"] - 0.86) < 1e-6
        assert method_summary.n_trials == 3
        assert method_summary.n_success == 3

    def test_build_summary_manual_tunes(self):
        """build_summary 聚合 Baseline 的 manual_tunes。"""
        report = ComparisonReport(experiment_id="exp1", design_name="d1")
        results = self._make_results()
        summary = report.build_summary(results)
        baseline_summary = summary["baseline_repro"]
        assert baseline_summary.mean_manual_tunes == 10.0

    def test_build_significance_method_vs_baseline(self):
        """build_significance 生成 method_vs_baseline_repro 检验。"""
        report = ComparisonReport(experiment_id="exp1", design_name="d1")
        results = self._make_results()
        sig = report.build_significance(results, metric="accuracy")
        assert "method_vs_baseline_repro" in sig
        test = sig["method_vs_baseline_repro"]
        assert isinstance(test, SignificanceTest)
        assert test.test_method in ("ttest", "bootstrap")
        # Method (0.86) vs Baseline (0.76) 差异显著
        assert test.p_value < 0.05
        assert test.significant is True

    def test_significance_insufficient_samples(self):
        """样本数 < 2 时返回 insufficient_samples。"""
        test = run_significance_test(
            pair="p1", metric="acc",
            method_values=[0.85], baseline_values=[0.75],
        )
        assert test.test_method == "insufficient_samples"
        assert test.significant is False

    def test_report_to_dict_serializable(self):
        """to_dict 返回可 JSON 序列化的 dict。"""
        report = ComparisonReport(
            experiment_id="exp1", design_name="d1",
            results=self._make_results(),
        )
        report.summary = report.build_summary(report.results)
        report.significance = report.build_significance(report.results, metric="accuracy")
        d = report.to_dict()
        json.dumps(d)  # 不抛异常即可
        assert d["experiment_id"] == "exp1"

    def test_report_save_to_file(self, tmp_path):
        """save 写入 JSON 文件。"""
        report = ComparisonReport(
            experiment_id="exp1", design_name="d1",
            results=self._make_results(),
        )
        report.summary = report.build_summary(report.results)
        output = tmp_path / "report.json"
        report.save(output)
        assert output.exists()
        # 验证文件内容是合法 JSON
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["experiment_id"] == "exp1"


# ============================================================
# 顶层导出测试
# ============================================================
class TestTopLevelExport:
    """验证 ε6 符号顶层可达。"""

    def test_import_from_senseframe(self):
        """from senseframe import ExperimentRunner 可达。"""
        import senseframe
        assert hasattr(senseframe, "ExperimentRunner")
        assert hasattr(senseframe, "ExperimentDesign")
        assert hasattr(senseframe, "MethodRunner")
        assert hasattr(senseframe, "BaselineRunner")
        assert hasattr(senseframe, "ComparisonReport")
        assert hasattr(senseframe, "ExperimentTrialResult")

    def test_sp_trial_result_not_overwritten(self):
        """顶层 TrialResult 仍是 SP 版本（未被 experiment 覆盖）。"""
        import senseframe
        # SP TrialResult 应有 trial_id + value + intermediate_values
        sp_tr = senseframe.TrialResult(
            trial_id="t1", study_id="s1", params={}, value=0.9,
        )
        assert sp_tr.trial_id == "t1"
        assert sp_tr.value == 0.9

    def test_experiment_trial_result_alias(self):
        """ExperimentTrialResult 是 experiment.TrialResult 的别名。"""
        import senseframe
        from senseframe.experiment import TrialResult as ExpTrialResult
        assert senseframe.ExperimentTrialResult is ExpTrialResult
