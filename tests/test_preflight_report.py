"""PreflightReport 和 dynamic_validation status 保障测试。

覆盖场景：
1. to_dict() 中 dynamic_validation 空结果的 status 不应为 "passed"（all([]) 空真值 bug 守护）
2. to_dict() 中 dynamic_validation 非空全 ok 时 status 为 "passed"
3. to_dict() 中 dynamic_validation 非空有失败时 status 为 "failed"
4. add_category 空 list 不更新 PreflightReport.status
5. CheckResult.to_dict 字段完整性
6. _build_unified_report 保留 failed 状态及 error/error_code 字段
7. layout 声明一致性：NTU-Fi_HAR=nested, NTU-Fi-HumanID=nested
"""

import pytest

from senseframe.engine.runner.preflight import (
    CheckResult,
    PreflightReport,
)


class TestCheckResult:
    """CheckResult dataclass 基本行为。"""

    def test_to_dict_fields_complete(self):
        """to_dict 应包含所有字段。"""
        c = CheckResult(
            name="test_check",
            ok=False,
            severity="error",
            detail={"key": "value"},
            error_code="TEST_ERROR",
            remediation="fix it",
        )
        d = c.to_dict()
        assert d["name"] == "test_check"
        assert d["ok"] is False
        assert d["severity"] == "error"
        assert d["detail"] == {"key": "value"}
        assert d["error_code"] == "TEST_ERROR"
        assert d["remediation"] == "fix it"

    def test_to_dict_defaults(self):
        """to_dict 默认值正确。"""
        c = CheckResult(name="ok_check", ok=True)
        d = c.to_dict()
        assert d["severity"] == "info"
        assert d["detail"] is None
        assert d["error_code"] is None
        assert d["remediation"] is None


class TestPreflightReportDynamicValidation:
    """dynamic_validation status 推导逻辑测试（核心 bug 守护）。"""

    def test_empty_results_status_is_failed(self):
        """空结果列表时 status 应为 "failed"，而非 "passed"。

        这是 all([]) 空真值 bug 的直接守护测试。
        旧逻辑：all([]) == True → status="passed"（错误）
        修复后：results and all(...) → [] and True → [] → falsy → status="failed"
        """
        rpt = PreflightReport(status="ok")
        rpt.add_category("model_contract", [])
        d = rpt.to_dict()
        assert d["dynamic_validation"]["status"] == "failed", (
            "空检查项列表应视为失败（动态校验异常导致无检查项），不应因 all([])==True 而标为 passed"
        )
        assert d["dynamic_validation"]["checks"] == []

    def test_all_ok_results_status_is_passed(self):
        """全部通过时 status 应为 "passed"。"""
        checks = [
            CheckResult(name="forward_pass", ok=True),
            CheckResult(name="backward_pass", ok=True),
        ]
        rpt = PreflightReport(status="ok")
        rpt.add_category("model_contract", checks)
        d = rpt.to_dict()
        assert d["dynamic_validation"]["status"] == "passed"
        assert len(d["dynamic_validation"]["checks"]) == 2

    def test_has_failure_status_is_failed(self):
        """有失败项时 status 应为 "failed"。"""
        checks = [
            CheckResult(name="forward_pass", ok=True),
            CheckResult(name="backward_pass", ok=False, severity="error"),
        ]
        rpt = PreflightReport(status="ok")
        rpt.add_category("model_contract", checks)
        d = rpt.to_dict()
        assert d["dynamic_validation"]["status"] == "failed"

    def test_single_failure_status_is_failed(self):
        """仅 1 项且失败时 status 应为 "failed"。"""
        checks = [CheckResult(name="forward_pass", ok=False, severity="error")]
        rpt = PreflightReport(status="ok")
        rpt.add_category("model_contract", checks)
        d = rpt.to_dict()
        assert d["dynamic_validation"]["status"] == "failed"


class TestPreflightReportAddCategory:
    """add_category 对 PreflightReport.status 的影响。"""

    def test_empty_list_does_not_update_status(self):
        """空 list 不更新 PreflightReport.status（保持初始值）。"""
        rpt = PreflightReport(status="ok")
        rpt.add_category("model_contract", [])
        assert rpt.status == "ok"

    def test_error_updates_status_to_blocked(self):
        """error 级失败更新 status 为 "blocked"。"""
        rpt = PreflightReport(status="ok")
        rpt.add_category("config_semantics", [
            CheckResult(name="epochs_positive", ok=False, severity="error"),
        ])
        assert rpt.status == "blocked"

    def test_warning_updates_status_to_warning(self):
        """warning 级失败更新 status 为 "warning"。"""
        rpt = PreflightReport(status="ok")
        rpt.add_category("config_semantics", [
            CheckResult(name="batch_size_large", ok=False, severity="warning"),
        ])
        assert rpt.status == "warning"

    def test_blocked_not_downgraded_by_warning(self):
        """blocked 状态不被 warning 降级。"""
        rpt = PreflightReport(status="blocked")
        rpt.add_category("other", [
            CheckResult(name="warn_check", ok=False, severity="warning"),
        ])
        assert rpt.status == "blocked"


class TestBuildUnifiedReportFailedPreservation:
    """_build_unified_report 保留 failed 状态及 error/error_code 字段。

    修复 bug：旧逻辑只覆写 "skipped" 状态，"failed" 状态被 to_dict 丢失。
    """

    def _make_minimal_config(self):
        """构造 minimal ExperimentConfig 实例。"""
        from senseframe.engine.config import (
            ExperimentConfig, SceneConfig, InputFeature, OutputFeature, TrainerConfig,
        )
        return ExperimentConfig(
            scene=SceneConfig(name="test", dataset="test", model_id="test"),
            input_features=[InputFeature(name="x", type="csi", shape=[1, 250, 90])],
            output_features=[OutputFeature(name="y", type="category", num_classes=7)],
            trainer=TrainerConfig(),
        )

    def test_failed_dyn_val_preserved_in_unified_report(self):
        """_build_unified_report 应保留 failed 状态和 error/error_code 字段。"""
        from senseframe.cli import _build_unified_report

        config = self._make_minimal_config()
        route_config = {"level": "cpu_standard", "device": "cpu"}

        # 构造 dynamic_validation 失败的 report
        report = {
            "status": "blocked",
            "blocked_reason": "dynamic_validation: DYNAMIC_VALIDATION_ERROR - FileNotFoundError",
            "dynamic_validation": {
                "status": "failed",
                "error": "CSIDataset: layout='flat' but no '*.mat' under /path",
                "error_code": "DYNAMIC_VALIDATION_ERROR",
                "checks": [],
            },
        }

        unified = _build_unified_report(report, config, route_config)

        # failed 状态应被保留（而非被 to_dict 的 all([]) 错误设为 "passed"）
        assert unified["dynamic_validation"]["status"] == "failed", (
            "failed 状态应被保留，不应被 to_dict 的 all([]) 空真值 bug 覆盖为 passed"
        )
        # error/error_code 字段应被保留
        assert unified["dynamic_validation"]["error_code"] == "DYNAMIC_VALIDATION_ERROR"
        assert "CSIDataset" in unified["dynamic_validation"]["error"]
        # blocked_reason 应被保留
        assert "dynamic_validation" in unified.get("blocked_reason", "")

    def test_skipped_dyn_val_preserved_in_unified_report(self):
        """_build_unified_report 应保留 skipped 状态（回归测试）。"""
        from senseframe.cli import _build_unified_report

        config = self._make_minimal_config()
        route_config = {"level": "cpu_standard", "device": "cpu"}

        report = {
            "status": "ok",
            "dynamic_validation": {
                "status": "skipped",
                "reason": "static_only mode",
                "checks": [],
            },
        }

        unified = _build_unified_report(report, config, route_config)
        assert unified["dynamic_validation"]["status"] == "skipped"
        assert unified["dynamic_validation"]["reason"] == "static_only mode"


class TestDatasetLayoutConsistency:
    """数据集 layout 声明一致性测试。

    确保 NTU-Fi_HAR=nested（按类别子目录），NTU-Fi-HumanID=nested（按类别子目录）。
    防止 layout 声明与实际数据集目录结构不匹配的 bug 再次出现。
    """

    @pytest.fixture(autouse=True)
    def _activate_scenes(self):
        """每个测试前激活 lazy scenes，确保 registry 有数据集 spec。"""
        import senseframe as sf
        sf.activate_lazy_scenes()

    def test_ntu_fi_har_layout_is_nested(self):
        """NTU-Fi_HAR 的 layout 应为 nested（官方数据集按类别子目录组织）。"""
        from senseframe.registry import get_dataset_spec
        spec = get_dataset_spec("NTU-Fi_HAR")
        assert spec.layout == "nested", (
            f"NTU-Fi_HAR layout 应为 'nested'（类别子目录），实际为 '{spec.layout}'"
        )

    def test_ntu_fi_humanid_layout_is_nested(self):
        """NTU-Fi-HumanID 的 layout 应为 nested（类别子目录，如 test_amp/015/*.mat）。"""
        from senseframe.registry import get_dataset_spec
        spec = get_dataset_spec("NTU-Fi-HumanID")
        assert spec.layout == "nested", (
            f"NTU-Fi-HumanID layout 应为 'nested'（类别子目录），实际为 '{spec.layout}'"
        )

    def test_widar_layout_is_nested(self):
        """Widar 的 layout 应为 nested（类别子目录）。"""
        from senseframe.registry import get_dataset_spec
        spec = get_dataset_spec("Widar")
        assert spec.layout == "nested"

    def test_ut_har_layout_is_flat(self):
        """UT_HAR_data 的 layout 应为 flat（.npy 文件）。"""
        from senseframe.registry import get_dataset_spec
        spec = get_dataset_spec("UT_HAR_data")
        assert spec.layout == "flat"
