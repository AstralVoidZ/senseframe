"""L1 契约测试：SenseFrame Search Protocol（Sampler / Pruner）。

锚点来源：senseframe.search_protocol 中的 Sampler / Pruner Protocol 定义
（@runtime_checkable Protocol，项目内协议声明作为 L1 契约）。

注意：这不是 Optuna 协议，是 SenseFrame 自有 Protocol。锚点是
search_protocol.py 中的 Protocol 类定义（接口声明），不是具体实现类的源码。
断言锚点是 Protocol 类本身（hasattr(Sampler, 'sample')），实现类仅作为
"被测对象"验证其符合 Protocol 契约。
"""
from __future__ import annotations

import inspect
from typing import Protocol

import pytest

from senseframe.search_protocol import (
    ASHASampler,
    GridSampler,
    HyperbandSampler,
    Pruner,
    RandomSampler,
    Sampler,
)
from tests.fakes.fake_pruner import FakePruner
from tests.fakes.fake_sampler import FakeSampler


@pytest.mark.l1_contract
class TestSamplerProtocolContract:
    """验证 SenseFrame Sampler Protocol 契约（锚点：Sampler Protocol 声明）。"""

    def test_sampler_is_protocol(self):
        """L1 anchor: Sampler 是 typing.Protocol 子类，锚点 Sampler Protocol 声明。"""
        assert issubclass(Sampler, Protocol)

    def test_sampler_is_runtime_checkable(self):
        """L1 anchor: Sampler 标注 @runtime_checkable，支持 isinstance 运行时检查。"""
        # @runtime_checkable 装饰器设置 _is_runtime_protocol = True
        assert getattr(Sampler, "_is_runtime_protocol", False) is True

    def test_sampler_declares_name_attribute(self):
        """L1 anchor: Sampler Protocol 声明 name 类属性，锚点 Sampler Protocol 接口。"""
        # Protocol 中 name: str 是类型注解（非实际类属性），通过 __annotations__ 验证声明
        assert "name" in getattr(Sampler, "__annotations__", {}), \
            "Sampler Protocol 必须声明 name 属性"

    def test_sampler_declares_sample_method(self):
        """L1 anchor: Sampler Protocol 声明 sample 方法，锚点 Sampler Protocol 接口。"""
        assert callable(getattr(Sampler, "sample", None)), \
            "Sampler Protocol 必须声明 sample 方法"

    def test_sampler_declares_warm_start_method(self):
        """L1 anchor: Sampler Protocol 声明 warm_start 方法，锚点 Sampler Protocol 接口。"""
        assert callable(getattr(Sampler, "warm_start", None)), \
            "Sampler Protocol 必须声明 warm_start 方法"

    def test_sampler_sample_signature(self):
        """L1 anchor: sample 签名含 search_space / history 参数，锚点 Sampler Protocol 接口。"""
        sig = inspect.signature(Sampler.sample)
        params = list(sig.parameters.keys())
        assert "search_space" in params, "sample 必须含 search_space 参数"
        assert "history" in params, "sample 必须含 history 参数"

    def test_sampler_warm_start_signature(self):
        """L1 anchor: warm_start 签名含 source_history 参数，锚点 Sampler Protocol 接口。"""
        sig = inspect.signature(Sampler.warm_start)
        params = list(sig.parameters.keys())
        assert "source_history" in params, "warm_start 必须含 source_history 参数"

    @pytest.mark.parametrize(
        "sampler_cls", [RandomSampler, GridSampler, ASHASampler, HyperbandSampler]
    )
    def test_builtin_sampler_satisfies_protocol(self, sampler_cls):
        """L1 anchor: 内置 Sampler 通过 isinstance(x, Sampler)，锚点 Sampler Protocol。"""
        instance = sampler_cls()
        assert isinstance(instance, Sampler), \
            f"{sampler_cls.__name__} 应满足 Sampler Protocol"

    def test_fake_sampler_satisfies_protocol(self):
        """L1 anchor: FakeSampler 测试替身通过 isinstance(x, Sampler)，锚点 Sampler Protocol。"""
        assert isinstance(FakeSampler(), Sampler)

    def test_non_sampler_fails_isinstance(self):
        """L1 anchor: 不满足 Sampler 契约的对象不通过 isinstance（反向用例）。"""
        class _NotASampler:
            pass
        assert not isinstance(_NotASampler(), Sampler)


@pytest.mark.l1_contract
class TestPrunerProtocolContract:
    """验证 SenseFrame Pruner Protocol 契约（锚点：Pruner Protocol 声明）。"""

    def test_pruner_is_protocol(self):
        """L1 anchor: Pruner 是 typing.Protocol 子类，锚点 Pruner Protocol 声明。"""
        assert issubclass(Pruner, Protocol)

    def test_pruner_is_runtime_checkable(self):
        """L1 anchor: Pruner 标注 @runtime_checkable，支持 isinstance 运行时检查。"""
        assert getattr(Pruner, "_is_runtime_protocol", False) is True

    def test_pruner_declares_name_attribute(self):
        """L1 anchor: Pruner Protocol 声明 name 类属性，锚点 Pruner Protocol 接口。"""
        # Protocol 中 name: str 是类型注解（非实际类属性），通过 __annotations__ 验证声明
        assert "name" in getattr(Pruner, "__annotations__", {}), \
            "Pruner Protocol 必须声明 name 属性"

    def test_pruner_declares_should_prune_method(self):
        """L1 anchor: Pruner Protocol 声明 should_prune 方法，锚点 Pruner Protocol 接口。"""
        assert callable(getattr(Pruner, "should_prune", None)), \
            "Pruner Protocol 必须声明 should_prune 方法"

    def test_pruner_should_prune_signature(self):
        """L1 anchor: should_prune 签名含 trial_id/intermediate_values/rung 参数。"""
        sig = inspect.signature(Pruner.should_prune)
        params = list(sig.parameters.keys())
        assert "trial_id" in params, "should_prune 必须含 trial_id 参数"
        assert "intermediate_values" in params, "should_prune 必须含 intermediate_values 参数"
        assert "rung" in params, "should_prune 必须含 rung 参数"

    @pytest.mark.parametrize("pruner_cls", [ASHASampler, HyperbandSampler])
    def test_builtin_pruner_satisfies_protocol(self, pruner_cls):
        """L1 anchor: 内置 Pruner 通过 isinstance(x, Pruner)，锚点 Pruner Protocol。"""
        instance = pruner_cls()
        assert isinstance(instance, Pruner), \
            f"{pruner_cls.__name__} 应满足 Pruner Protocol"

    def test_fake_pruner_satisfies_protocol(self):
        """L1 anchor: FakePruner 测试替身通过 isinstance(x, Pruner)，锚点 Pruner Protocol。"""
        assert isinstance(FakePruner(), Pruner)

    def test_non_pruner_fails_isinstance(self):
        """L1 anchor: 不满足 Pruner 契约的对象不通过 isinstance（反向用例）。"""
        class _NotAPruner:
            pass
        assert not isinstance(_NotAPruner(), Pruner)

    def test_pruner_does_not_declare_optuna_prune_method(self):
        """L1 anchor: SP Pruner 声明 should_prune（非 Optuna prune），锚点 SP Protocol 设计。"""
        # SenseFrame Pruner Protocol 是自有协议，不声明 Optuna 的 prune 方法
        assert not hasattr(Pruner, "prune"), \
            "SP Pruner 不应声明 Optuna 的 prune 方法（自有 Protocol）"
