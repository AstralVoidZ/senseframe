"""senseframe.core.validators 模块测试。

覆盖 5 个验证器 + ValidationResult + compose + run_validation。
"""

import pytest

from senseframe.core.validators import (
    ValidationResult,
    shape_validator,
    numerical_stability_validator,
    signature_validator,
    performance_validator,
    transform_pipeline_validator,
    compose,
    run_validation,
)


class TestValidationResult:
    """ValidationResult 的 ok / fail / merge 基本行为。"""

    def test_ok_default(self):
        r = ValidationResult.ok()
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []

    def test_ok_with_warnings(self):
        r = ValidationResult.ok(warnings=["w1", "w2"])
        assert r.passed is True
        assert r.warnings == ["w1", "w2"]

    def test_fail(self):
        r = ValidationResult.fail(["err"])
        assert r.passed is False
        assert r.errors == ["err"]
        assert r.warnings == []

    def test_fail_with_warnings(self):
        r = ValidationResult.fail(["err"], warnings=["w"])
        assert r.passed is False
        assert r.errors == ["err"]
        assert r.warnings == ["w"]

    def test_merge_both_pass(self):
        a = ValidationResult.ok()
        b = ValidationResult.ok()
        m = a.merge(b)
        assert m.passed is True
        assert m.errors == []

    def test_merge_one_fail(self):
        a = ValidationResult.ok()
        b = ValidationResult.fail(["err"])
        m = a.merge(b)
        assert m.passed is False
        assert m.errors == ["err"]

    def test_merge_accumulates_errors_and_warnings(self):
        a = ValidationResult.fail(["e1"], warnings=["w1"])
        b = ValidationResult.fail(["e2"], warnings=["w2"])
        m = a.merge(b)
        assert m.passed is False
        assert m.errors == ["e1", "e2"]
        assert m.warnings == ["w1", "w2"]


class TestShapeValidator:
    """shape_validator 对合法/非法 transform 的验证。"""

    def test_valid_transform_no_warning(self):
        validator = shape_validator((30,))

        def good_transform(x, y):
            return x

        result = validator(good_transform)
        assert result.passed is True
        assert not any("参数数" in w for w in result.warnings)

    def test_invalid_signature_warns(self):
        validator = shape_validator((30,))

        def bad_transform(x):
            return x

        result = validator(bad_transform)
        # shape_validator 总是通过，仅对参数不足产生 warning
        assert result.passed is True
        assert any("参数数" in w for w in result.warnings)


class TestNumericalStabilityValidator:
    """numerical_stability_validator 的数值稳定性验证。"""

    def test_stable_loss_passes(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        validator = numerical_stability_validator()
        result = validator(nn.CrossEntropyLoss)
        assert result.passed is True

    def test_nan_loss_fails(self):
        torch = pytest.importorskip("torch")

        def nan_loss_factory():
            def loss(pred, target):
                return torch.tensor(float("nan"))

            return loss

        validator = numerical_stability_validator()
        result = validator(nan_loss_factory)
        assert result.passed is False
        assert any("NaN" in e for e in result.errors)

    def test_no_torch_graceful_degrade(self):
        """无 torch 时优雅降级，仅 warning。"""
        import sys
        from unittest import mock

        validator = numerical_stability_validator()
        with mock.patch.dict(sys.modules, {"torch": None}):
            result = validator(lambda: None)
        assert result.passed is True
        assert any("torch" in w.lower() for w in result.warnings)


class TestSignatureValidator:
    """signature_validator 的工厂签名兼容性验证。"""

    def test_factory_with_kwargs_passes(self):
        validator = signature_validator(["alpha", "beta"])

        def factory(**kwargs):
            pass

        result = validator(factory)
        assert result.passed is True
        assert result.errors == []

    def test_factory_missing_param_fails(self):
        validator = signature_validator(["alpha", "beta"])

        def factory(alpha):
            pass

        result = validator(factory)
        assert result.passed is False
        assert any("beta" in e for e in result.errors)


class TestPerformanceValidator:
    """performance_validator 测量快速函数。"""

    def test_fast_function_passes(self):
        pytest.importorskip("torch")

        def fast_factory():
            def fn(x):
                return x + 1

            return fn

        validator = performance_validator(baseline_time_ms=1000.0, num_runs=3)
        result = validator(fast_factory)
        assert result.passed is True


class TestTransformPipelineValidator:
    """transform_pipeline_validator 的 pipeline 验证。"""

    def test_stable_pipeline_passes(self):
        pytest.importorskip("torch")
        from senseframe.scenes.wifi_csi.transforms import compose_transforms

        pipeline = compose_transforms(["hampel"])
        validator = transform_pipeline_validator()
        result = validator(pipeline)
        assert result.passed is True

    def test_nan_pipeline_fails(self):
        import numpy as np

        def nan_pipeline(x, y=None):
            return np.full_like(np.asarray(x, dtype=np.float64), np.nan), y

        validator = transform_pipeline_validator()
        result = validator(nan_pipeline)
        assert result.passed is False
        assert any("NaN" in e for e in result.errors)


class TestCompose:
    """compose 组合多个验证器，全部通过才通过。"""

    def test_all_pass(self):
        v1 = lambda obj: ValidationResult.ok()
        v2 = lambda obj: ValidationResult.ok()
        c = compose(v1, v2)
        assert c("anything").passed is True

    def test_one_fails(self):
        v1 = lambda obj: ValidationResult.ok()
        v2 = lambda obj: ValidationResult.fail(["err"])
        c = compose(v1, v2)
        result = c("anything")
        assert result.passed is False
        assert result.errors == ["err"]

    def test_none_validators_skipped(self):
        c = compose(None, None)
        assert c("anything").passed is True


class TestRunValidation:
    """run_validation：validator=None 时直接通过。"""

    def test_none_validator_passes(self):
        result = run_validation(None, "obj")
        assert result.passed is True

    def test_with_validator(self):
        v = lambda obj: ValidationResult.fail(["err"])
        result = run_validation(v, "obj")
        assert result.passed is False

    def test_exception_caught(self):
        def bad_validator(obj):
            raise RuntimeError("boom")

        result = run_validation(bad_validator, "obj")
        assert result.passed is False
        assert any("验证器异常" in e for e in result.errors)
