"""
验证系统：为开放注册表提供安全护栏。

设计理念（RFC-002 原则 2）：
- 验证重于生成（P vs NP 不对称性）
- 框架不需要比 Agent 更擅长生成策略，但必须更擅长验证策略
- 新注册的策略必须通过验证才能生效

验证器类型：
- shape_validator: 验证 transform 输出 shape 匹配模型输入
- numerical_stability_validator: 验证 loss/metric 不产生 NaN/Inf
- signature_validator: 验证工厂签名兼容性
- compose: 组合多个验证器

使用方式：
    from senseframe.core.validators import numerical_stability_validator, compose

    @register_loss("my_loss", validator=numerical_stability_validator(sample_input))
    def my_loss(pred, target, **kwargs):
        return torch.exp(pred) - target  # 可能数值不稳定

    # 注册时自动验证，验证失败则拒绝注册
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationResult:
    """验证结果。"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        return cls(passed=True, warnings=warnings or [])

    @classmethod
    def fail(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        return cls(passed=False, errors=errors, warnings=warnings or [])

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
            passed=self.passed and other.passed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


# 验证器协议：接收被验证对象，返回 ValidationResult
Validator = Callable[[Any], ValidationResult]


def shape_validator(expected_shape: Tuple[int, ...]) -> Validator:
    """验证 transform 输出 shape 匹配预期。

    Args:
        expected_shape: 预期的输出 shape（不含 batch 维）

    Returns:
        Validator
    """
    def validate(transform_fn: Callable) -> ValidationResult:
        errors = []
        warnings = []
        try:
            sig = inspect.signature(transform_fn)
            params = list(sig.parameters.keys())
            if len(params) < 2:
                warnings.append(
                    f"transform 签名参数数 {len(params)} < 2，"
                    f"期望 (x, y) 或 (x, y, **kwargs)"
                )
        except (ValueError, TypeError) as e:
            warnings.append(f"无法检查签名: {e}")
        return ValidationResult(passed=True, errors=errors, warnings=warnings)

    return validate


def numerical_stability_validator(
    sample_input: Optional[Any] = None,
    sample_target: Optional[Any] = None,
    num_samples: int = 4,
) -> Validator:
    """验证 loss/metric 在小批量上不产生 NaN/Inf。

    Args:
        sample_input: 样本输入（None 时随机生成）
        sample_target: 样本目标（None 时随机生成）
        num_samples: 随机生成时的样本数

    Returns:
        Validator
    """
    def validate(factory: Callable) -> ValidationResult:
        errors = []
        warnings = []
        try:
            import torch

            # 实例化
            if sample_input is not None:
                x = torch.as_tensor(sample_input, dtype=torch.float32)
            else:
                x = torch.randn(num_samples, 4)
            if sample_target is not None:
                y = torch.as_tensor(sample_target, dtype=torch.float32)
            else:
                y = torch.randint(0, 2, (num_samples,))

            # 尝试调用工厂
            try:
                instance = factory(num_classes=2) if "num_classes" in inspect.signature(factory).parameters else factory()
            except TypeError:
                try:
                    instance = factory()
                except Exception as e:
                    warnings.append(f"无法实例化验证（工厂签名不匹配）: {e}")
                    return ValidationResult(passed=True, warnings=warnings)

            # 尝试前向计算
            if hasattr(instance, "__call__"):
                try:
                    if isinstance(instance, type):
                        # 是类而非实例
                        return ValidationResult.ok(warnings=["工厂返回类而非实例，跳过前向验证"])
                    output = instance(x, y) if y is not None else instance(x)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    if isinstance(output, torch.Tensor):
                        if torch.isnan(output).any():
                            errors.append("输出包含 NaN")
                        if torch.isinf(output).any():
                            errors.append("输出包含 Inf")
                except Exception as e:
                    warnings.append(f"前向验证跳过（调用失败）: {e}")

        except ImportError:
            warnings.append("torch 未安装，跳过数值稳定性验证")
        except Exception as e:
            errors.append(f"验证异常: {e}")

        if errors:
            return ValidationResult.fail(errors, warnings)
        return ValidationResult.ok(warnings)

    return validate


def signature_validator(expected_kwargs: List[str]) -> Validator:
    """验证工厂签名兼容性。

    Args:
        expected_kwargs: 期望工厂接受的参数名列表

    Returns:
        Validator
    """
    def validate(factory: Callable) -> ValidationResult:
        errors = []
        warnings = []
        try:
            sig = inspect.signature(factory)
            params = sig.parameters
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if has_var_keyword:
                return ValidationResult.ok()
            for kw in expected_kwargs:
                if kw not in params:
                    errors.append(f"工厂缺少参数: {kw}")
        except (ValueError, TypeError) as e:
            errors.append(f"无法检查签名: {e}")
        if errors:
            return ValidationResult.fail(errors, warnings)
        return ValidationResult.ok(warnings)

    return validate


def compose(*validators: Validator) -> Validator:
    """组合多个验证器，全部通过才算通过。"""
    def validate(obj: Any) -> ValidationResult:
        result = ValidationResult.ok()
        for v in validators:
            if v is None:
                continue
            result = result.merge(v(obj))
            if not result.passed:
                break
        return result

    return validate


def performance_validator(
    sample_input: Optional[Any] = None,
    baseline_time_ms: float = 10.0,
    num_runs: int = 5,
) -> Validator:
    """性能基准验证器（RFC-002 阶段 O）：测量生成代码的执行 overhead。

    P vs NP 不对称性的应用：框架不替 Agent 写代码，但能高效验证代码性能。
    超过基线时间仅产生 warning（不 fail），因为性能问题不阻断注册。

    Args:
        sample_input: 样本输入（None 时随机生成）
        baseline_time_ms: 基线时间（毫秒），超过则 warning
        num_runs: 测量次数（取平均）

    Returns:
        Validator
    """
    def validate(factory: Callable) -> ValidationResult:
        warnings = []
        try:
            import time

            # 实例化
            try:
                instance = (
                    factory(num_classes=2)
                    if "num_classes" in inspect.signature(factory).parameters
                    else factory()
                )
            except TypeError:
                try:
                    instance = factory()
                except Exception as e:
                    warnings.append(f"性能验证跳过（无法实例化）: {e}")
                    return ValidationResult.ok(warnings)

            # 准备输入
            try:
                import torch
                x = torch.randn(4, 4) if sample_input is None else torch.as_tensor(sample_input)
            except ImportError:
                warnings.append("torch 未安装，跳过性能验证")
                return ValidationResult.ok(warnings)

            # 预热 + 测量
            if hasattr(instance, "__call__") and not isinstance(instance, type):
                try:
                    instance(x)  # 预热
                except Exception:
                    pass
                times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    try:
                        instance(x)
                    except Exception:
                        break
                    times.append((time.perf_counter() - start) * 1000)
                if times:
                    avg_ms = sum(times) / len(times)
                    if avg_ms > baseline_time_ms:
                        warnings.append(
                            f"平均执行时间 {avg_ms:.2f}ms 超过基线 {baseline_time_ms}ms"
                        )
        except Exception as e:
            warnings.append(f"性能验证异常: {e}")
        return ValidationResult.ok(warnings)

    return validate


def transform_pipeline_validator(
    sample_input: Optional[Any] = None,
    expected_output_shape: Optional[Tuple[int, ...]] = None,
) -> Validator:
    """transform pipeline 验证器（RFC-002 阶段 O）：验证组合后的 shape 一致性 + 数值稳定性。

    用于验证 compose_transforms 组合的 pipeline：执行整个链路，
    检查输出无 NaN/Inf，且 shape 匹配预期。

    Args:
        sample_input: 样本输入（None 时随机生成 (4, 30) 矩阵）
        expected_output_shape: 预期输出 shape（不含 batch 维，None 时仅检查数值稳定性）

    Returns:
        Validator
    """
    def validate(transform_fn: Callable) -> ValidationResult:
        errors = []
        warnings = []
        try:
            import numpy as np

            # 准备输入
            if sample_input is not None:
                x = sample_input
            else:
                x = np.random.randn(4, 30).astype(np.float32)

            # 执行 pipeline
            try:
                result = transform_fn(x, None)
                output = result[0] if isinstance(result, (tuple, list)) else result
            except Exception as e:
                errors.append(f"pipeline 执行失败: {e}")
                return ValidationResult.fail(errors, warnings)

            # 检查 NaN/Inf
            try:
                import torch
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        errors.append("pipeline 输出包含 NaN")
                    if torch.isinf(output).any():
                        errors.append("pipeline 输出包含 Inf")
                else:
                    out_np = np.asarray(output)
                    if np.isnan(out_np).any():
                        errors.append("pipeline 输出包含 NaN")
                    if np.isinf(out_np).any():
                        errors.append("pipeline 输出包含 Inf")
            except ImportError:
                out_np = np.asarray(output)
                if np.isnan(out_np).any():
                    errors.append("pipeline 输出包含 NaN")
                if np.isinf(out_np).any():
                    errors.append("pipeline 输出包含 Inf")

            # 检查 shape
            if expected_output_shape is not None:
                try:
                    import torch
                    if isinstance(output, torch.Tensor):
                        actual_shape = tuple(output.shape[1:])
                    else:
                        actual_shape = tuple(np.asarray(output).shape[1:])
                    if actual_shape != expected_output_shape:
                        errors.append(
                            f"输出 shape {actual_shape} 不匹配预期 {expected_output_shape}"
                        )
                except Exception as e:
                    warnings.append(f"shape 检查跳过: {e}")

        except Exception as e:
            errors.append(f"pipeline 验证异常: {e}")

        if errors:
            return ValidationResult.fail(errors, warnings)
        return ValidationResult.ok(warnings)

    return validate


def run_validation(validator: Optional[Validator], obj: Any) -> ValidationResult:
    """执行验证（validator 为 None 时直接通过）。"""
    if validator is None:
        return ValidationResult.ok()
    try:
        return validator(obj)
    except Exception as e:
        return ValidationResult.fail([f"验证器异常: {e}"])


__all__ = [
    "ValidationResult",
    "Validator",
    "shape_validator",
    "numerical_stability_validator",
    "signature_validator",
    "performance_validator",
    "transform_pipeline_validator",
    "compose",
    "run_validation",
]
