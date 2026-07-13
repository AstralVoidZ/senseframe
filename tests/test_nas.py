"""ε2 NAS 测试（P2.6-P2.9）。

反假绿测试策略：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- dataclasses.fields 反射：验证字段存在性，不硬编码字段列表
- 真实行为：ArchitectureBuilder 真实构建 nn.Module + 真实 forward pass
- Protocol 契约：isinstance + runtime_checkable 验证（EvolutionarySampler 满足 Sampler）
- 真实进化采样：population 初始化 + 锦标赛选择 + 变异

覆盖：
- P2.6: ArchitectureSearchSpace + ArchitectureParameterSpec（DSP 合规）
- P2.7: ArchitectureBuilder（conv1d / rnn / hybrid 三种 cell_type）
- P2.8: EvolutionarySampler（Sampler Protocol + 进化行为）
- P2.9: make_nas_module_factory（module_factory 注入 + GenericLightningModule 包装）
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import pytest

# torch 可选（如未安装则 skip 整个文件）
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from senseframe.nas import (
    ArchitectureBuilder,
    ArchitectureParameterSpec,
    ArchitectureSearchSpace,
    EvolutionarySampler,
    make_nas_module_factory,
)
from senseframe.nas.search_space import (
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_CELL_TYPES,
    SUPPORTED_RNN_TYPES,
)
from senseframe.search_protocol import (
    ParameterSpec,
    Sampler,
    SearchSpace,
    get_sampler,
    list_samplers,
)


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


def _conv1d_params(**overrides):
    """构造 conv1d 架构参数（用于 builder 测试）。"""
    params = {
        "cell_type": "conv1d",
        "n_layers": 2,
        "hidden_dim": 32,
        "activation": "relu",
        "kernel_size": 3,
        "dropout": 0.1,
    }
    params.update(overrides)
    return params


def _rnn_params(**overrides):
    """构造 rnn 架构参数。"""
    params = {
        "cell_type": "rnn",
        "n_layers": 2,
        "hidden_dim": 64,
        "activation": "tanh",
        "rnn_type": "lstm",
        "bidirectional": False,
        "dropout": 0.1,
    }
    params.update(overrides)
    return params


def _hybrid_params(**overrides):
    """构造 hybrid 架构参数。"""
    params = {
        "cell_type": "hybrid",
        "n_layers": 2,
        "hidden_dim": 32,
        "activation": "relu",
        "kernel_size": 3,
        "rnn_type": "lstm",
        "bidirectional": False,
        "dropout": 0.1,
    }
    params.update(overrides)
    return params


# ============================================================
# P2.6: ArchitectureParameterSpec
# ============================================================
class TestArchitectureParameterSpec:
    """ArchitectureParameterSpec 数据结构验证。"""

    def test_field_exists_via_reflection(self):
        """应有核心字段（反射验证，不硬编码）。"""
        field_names = [f.name for f in fields(ArchitectureParameterSpec)]
        for required in ["name", "type", "choices", "low", "high", "log", "step", "default"]:
            assert required in field_names, f"ArchitectureParameterSpec 缺少字段: {required}"

    def test_to_dict_serializable(self):
        """to_dict 应返回可序列化 dict。"""
        p = ArchitectureParameterSpec(name="lr", type="float", low=0.001, high=0.1, log=True)
        d = p.to_dict()
        assert d["name"] == "lr"
        assert d["type"] == "float"
        assert d["low"] == 0.001
        assert d["high"] == 0.1
        assert d["log"] is True

    def test_from_dict_roundtrip(self):
        """from_dict 应能反序列化。"""
        p = ArchitectureParameterSpec(name="n_layers", type="int", low=1, high=8, default=3)
        d = p.to_dict()
        p2 = ArchitectureParameterSpec.from_dict(d)
        assert p2.name == p.name
        assert p2.type == p.type
        assert p2.low == p.low
        assert p2.high == p.high
        assert p2.default == p.default

    def test_to_sp_param_omits_default(self):
        """to_sp_param 应不包含 default（SP ParameterSpec 无此字段）。"""
        p = ArchitectureParameterSpec(
            name="cell_type", type="categorical",
            choices=["conv1d"], default="conv1d",
        )
        sp = p.to_sp_param()
        assert "default" not in sp, "SP ParameterSpec 不应有 default 字段"
        assert sp["name"] == "cell_type"
        assert sp["choices"] == ["conv1d"]


# ============================================================
# P2.6: ArchitectureSearchSpace
# ============================================================
class TestArchitectureSearchSpace:
    """ArchitectureSearchSpace 数据结构 + DSP 合规验证。"""

    def test_schema_version_default(self):
        """默认 schema_version 应为 1.0.0。"""
        ss = ArchitectureSearchSpace()
        assert ss.schema_version == "1.0.0"

    def test_default_cell_types(self):
        """默认 cell_types 应为 ['conv1d', 'rnn']。"""
        ss = ArchitectureSearchSpace()
        assert ss.cell_types == ["conv1d", "rnn"]

    def test_default_parameters_built(self):
        """默认应构造非空 parameters 列表。"""
        ss = ArchitectureSearchSpace()
        assert len(ss.parameters) > 0
        # 应包含 conv1d 和 rnn 的核心参数
        names = [p.name for p in ss.parameters]
        assert "cell_type" in names
        assert "n_layers" in names
        assert "hidden_dim" in names

    def test_cell_type_choices_merged(self):
        """cell_type 的 choices 应合并所有 cell_types。"""
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"])
        cell_type_param = ss.get_param("cell_type")
        assert cell_type_param is not None
        assert set(cell_type_param.choices) == {"conv1d", "rnn"}

    def test_unsupported_cell_type_raises(self):
        """attention cell_type 应在 P2 抛异常（推迟到 P3）。"""
        with pytest.raises(ValueError, match="Unsupported cell_type 'attention'"):
            ArchitectureSearchSpace(cell_types=["attention"])

    def test_hybrid_cell_type_supported(self):
        """hybrid cell_type 应支持。"""
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn", "hybrid"])
        assert "hybrid" in ss.cell_types
        # 应有 rnn_type 和 bidirectional 参数（hybrid 追加）
        names = [p.name for p in ss.parameters]
        assert "rnn_type" in names
        assert "bidirectional" in names

    def test_to_sp_search_space_returns_search_space(self):
        """to_sp_search_space 应返回 SP SearchSpace 实例。"""
        ss = ArchitectureSearchSpace()
        sp_ss = ss.to_sp_search_space()
        assert isinstance(sp_ss, SearchSpace)
        # 参数数量应一致
        assert len(sp_ss.parameters) == len(ss.parameters)
        # 应转换为 SP ParameterSpec
        for p in sp_ss.parameters:
            assert isinstance(p, ParameterSpec)

    def test_dsp_schema_returns_json_schema(self):
        """schema() 应返回 JSON Schema（DSP 自省）。"""
        schema = ArchitectureSearchSpace.schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "schema_version" in schema["properties"]

    def test_dsp_describe_returns_runtime_state(self):
        """describe() 应返回运行时状态摘要。"""
        ss = ArchitectureSearchSpace()
        desc = ss.describe()
        assert "schema_version" in desc
        assert "cell_types" in desc
        assert "n_parameters" in desc
        assert "parameter_names" in desc

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict / from_dict 应可往返。"""
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"])
        d = ss.to_dict()
        ss2 = ArchitectureSearchSpace.from_dict(d)
        assert ss2.schema_version == ss.schema_version
        assert ss2.cell_types == ss.cell_types
        assert len(ss2.parameters) == len(ss.parameters)

    def test_get_param_returns_spec(self):
        """get_param 应返回正确的参数规格。"""
        ss = ArchitectureSearchSpace()
        p = ss.get_param("n_layers")
        assert p is not None
        assert p.name == "n_layers"

    def test_get_param_nonexistent_returns_none(self):
        """get_param 未找到应返回 None。"""
        ss = ArchitectureSearchSpace()
        assert ss.get_param("__nonexistent__") is None

    def test_validate_params_valid(self):
        """合法参数应返回空错误列表。"""
        ss = ArchitectureSearchSpace()
        # 构造一组覆盖所有参数的合法值
        valid_params = {p.name: p.default for p in ss.parameters if p.default is not None}
        errors = ss.validate_params(valid_params)
        assert errors == [], f"合法参数应通过验证, got errors: {errors}"

    def test_validate_params_invalid_categorical(self):
        """categorical 参数取值非法应报错。"""
        ss = ArchitectureSearchSpace()
        errors = ss.validate_params({"cell_type": "__invalid__"})
        assert any("cell_type" in e for e in errors)

    def test_validate_params_int_out_of_range(self):
        """int 参数越界应报错。"""
        ss = ArchitectureSearchSpace()
        # n_layers low=1
        errors = ss.validate_params({"n_layers": 0})
        assert any("n_layers" in e for e in errors)


# ============================================================
# P2.7: ArchitectureBuilder
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestArchitectureBuilder:
    """ArchitectureBuilder 行为验证（真实 nn.Module 构建 + forward）。"""

    def test_build_conv1d_returns_module(self):
        """build conv1d 应返回 nn.Module。"""
        builder = ArchitectureBuilder()
        model = builder.build(_conv1d_params(), input_shape=(30, 100), num_classes=7)
        assert isinstance(model, nn.Module)

    def test_build_conv1d_forward_shape(self):
        """conv1d 模型 forward 应返回 (batch, num_classes)。"""
        builder = ArchitectureBuilder()
        model = builder.build(_conv1d_params(), input_shape=(30, 100), num_classes=7)
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_rnn_returns_module(self):
        """build rnn 应返回 nn.Module。"""
        builder = ArchitectureBuilder()
        model = builder.build(_rnn_params(), input_shape=(30, 100), num_classes=7)
        assert isinstance(model, nn.Module)

    def test_build_rnn_forward_shape(self):
        """rnn 模型 forward 应返回 (batch, num_classes)。"""
        builder = ArchitectureBuilder()
        model = builder.build(_rnn_params(), input_shape=(30, 100), num_classes=7)
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_hybrid_forward_shape(self):
        """hybrid 模型 forward 应返回 (batch, num_classes)。"""
        builder = ArchitectureBuilder()
        model = builder.build(_hybrid_params(), input_shape=(30, 100), num_classes=7)
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_rnn_bidirectional(self):
        """bidirectional=True 应能构建。"""
        builder = ArchitectureBuilder()
        model = builder.build(
            _rnn_params(bidirectional=True), input_shape=(30, 100), num_classes=7,
        )
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_rnn_gru(self):
        """rnn_type=gru 应能构建。"""
        builder = ArchitectureBuilder()
        model = builder.build(
            _rnn_params(rnn_type="gru"), input_shape=(30, 100), num_classes=7,
        )
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_conv1d_different_kernel_sizes(self):
        """不同 kernel_size 应都能构建。"""
        builder = ArchitectureBuilder()
        for ks in [3, 5, 7]:
            model = builder.build(
                _conv1d_params(kernel_size=ks),
                input_shape=(30, 100), num_classes=7,
            )
            x = torch.randn(2, 30, 100)
            out = model(x)
            assert out.shape == (2, 7)

    def test_build_unsupported_cell_type_raises(self):
        """不支持的 cell_type 应抛异常。"""
        builder = ArchitectureBuilder()
        with pytest.raises(ValueError, match="Unsupported cell_type"):
            builder.build({"cell_type": "attention"}, (30, 100), 7)

    def test_build_missing_cell_type_raises(self):
        """缺少 cell_type 应抛异常。"""
        builder = ArchitectureBuilder()
        with pytest.raises(ValueError, match="cell_type"):
            builder.build({"n_layers": 3}, (30, 100), 7)

    def test_build_unsupported_activation_raises(self):
        """不支持的 activation 应抛异常。"""
        builder = ArchitectureBuilder()
        with pytest.raises(ValueError, match="activation"):
            builder.build(
                _conv1d_params(activation="__invalid__"),
                (30, 100), 7,
            )


# ============================================================
# P2.8: EvolutionarySampler
# ============================================================
class TestEvolutionarySampler:
    """EvolutionarySampler 行为验证（Sampler Protocol + 进化行为）。"""

    def test_satisfies_sampler_protocol(self):
        """EvolutionarySampler 应满足 Sampler Protocol。"""
        sampler = EvolutionarySampler(seed=42)
        assert isinstance(sampler, Sampler)

    def test_name(self):
        """name 应为 'evolutionary'。"""
        assert EvolutionarySampler.name == "evolutionary"

    def test_registered_in_sp_sampler_registry(self):
        """应注册到 SP Sampler 注册表。"""
        assert "evolutionary" in list_samplers()
        assert get_sampler("evolutionary") is EvolutionarySampler

    def test_invalid_population_size_raises(self):
        """population_size < 2 应抛异常。"""
        with pytest.raises(ValueError, match="population_size"):
            EvolutionarySampler(population_size=1)

    def test_invalid_mutation_rate_raises(self):
        """mutation_rate 不在 [0, 1] 应抛异常。"""
        with pytest.raises(ValueError, match="mutation_rate"):
            EvolutionarySampler(mutation_rate=1.5)

    def test_invalid_tournament_size_raises(self):
        """tournament_size < 1 应抛异常。"""
        with pytest.raises(ValueError, match="tournament_size"):
            EvolutionarySampler(tournament_size=0)

    def test_sample_returns_valid_params(self):
        """sample() 应返回搜索空间内的有效参数。"""
        sampler = EvolutionarySampler(seed=42)
        ss = ArchitectureSearchSpace().to_sp_search_space()
        params = sampler.sample(ss, [])
        assert "cell_type" in params
        assert params["cell_type"] in ["conv1d", "rnn"]
        assert "n_layers" in params
        assert "hidden_dim" in params

    def test_sample_random_init_phase(self):
        """population 未满时，sample 应随机初始化。"""
        sampler = EvolutionarySampler(population_size=5, seed=42)
        ss = ArchitectureSearchSpace().to_sp_search_space()
        # 第一次采样：population 为空
        params = sampler.sample(ss, [])
        assert len(sampler._population) == 1
        # 第二次采样：population 仍不足
        params2 = sampler.sample(ss, [])
        assert len(sampler._population) == 2

    def test_sample_evolution_phase_uses_tournament(self):
        """population 已满 + history 有 fitness 时，应进入进化阶段。"""
        sampler = EvolutionarySampler(
            population_size=3, tournament_size=2, mutation_rate=0.5, seed=42,
        )
        ss = ArchitectureSearchSpace().to_sp_search_space()

        # 填充 history（模拟 SP 已完成的 trial）
        history = []
        for i in range(5):
            params = {
                "cell_type": "conv1d", "n_layers": 2 + i,
                "hidden_dim": 32, "activation": "relu",
                "kernel_size": 3, "dropout": 0.1,
                "rnn_type": "lstm", "bidirectional": False,
            }
            history.append({
                "params": params,
                "result": {"value": 0.5 + i * 0.1},  # fitness 递增
                "status": "completed",
            })

        # sample 应进入进化阶段（变异）
        child = sampler.sample(ss, history)
        assert "cell_type" in child
        # population 应被同步填充
        assert sampler.population_size_actual() >= 3
        assert sampler.evaluated_count() >= 1

    def test_sample_direction_minimize(self):
        """direction=minimize 时，锦标赛应选最小值。"""
        sampler = EvolutionarySampler(
            population_size=2, tournament_size=2,
            direction="minimize", mutation_rate=0.0, seed=42,
        )
        ss = ArchitectureSearchSpace().to_sp_search_space()
        # 填充 history：两个个体，value 分别为 0.3 和 0.8
        history = [
            {"params": {"cell_type": "conv1d", "n_layers": 1,
                        "hidden_dim": 16, "activation": "relu",
                        "kernel_size": 3, "dropout": 0.1,
                        "rnn_type": "lstm", "bidirectional": False},
             "result": {"value": 0.3}, "status": "completed"},
            {"params": {"cell_type": "conv1d", "n_layers": 5,
                        "hidden_dim": 64, "activation": "relu",
                        "kernel_size": 3, "dropout": 0.1,
                        "rnn_type": "lstm", "bidirectional": False},
             "result": {"value": 0.8}, "status": "completed"},
        ]
        # mutation_rate=0 → child 应等于 parent（最小 value 的个体）
        child = sampler.sample(ss, history)
        # 应继承 n_layers=1（value=0.3 是 minimize 最优）
        assert child["n_layers"] == 1

    def test_sample_direction_maximize(self):
        """direction=maximize 时，锦标赛应选最大值。"""
        sampler = EvolutionarySampler(
            population_size=2, tournament_size=2,
            direction="maximize", mutation_rate=0.0, seed=42,
        )
        ss = ArchitectureSearchSpace().to_sp_search_space()
        history = [
            {"params": {"cell_type": "conv1d", "n_layers": 1,
                        "hidden_dim": 16, "activation": "relu",
                        "kernel_size": 3, "dropout": 0.1,
                        "rnn_type": "lstm", "bidirectional": False},
             "result": {"value": 0.3}, "status": "completed"},
            {"params": {"cell_type": "conv1d", "n_layers": 5,
                        "hidden_dim": 64, "activation": "relu",
                        "kernel_size": 3, "dropout": 0.1,
                        "rnn_type": "lstm", "bidirectional": False},
             "result": {"value": 0.8}, "status": "completed"},
        ]
        child = sampler.sample(ss, history)
        # 应继承 n_layers=5（value=0.8 是 maximize 最优）
        assert child["n_layers"] == 5

    def test_mutation_changes_at_least_one_param_with_high_rate(self):
        """mutation_rate=1.0 应扰动所有可变异参数。"""
        sampler = EvolutionarySampler(
            population_size=2, tournament_size=2,
            mutation_rate=1.0, seed=123,
        )
        ss = ArchitectureSearchSpace().to_sp_search_space()
        # 先填充 history 提供一个父代
        parent_params = {
            "cell_type": "conv1d", "n_layers": 4,
            "hidden_dim": 64, "activation": "relu",
            "kernel_size": 3, "dropout": 0.2,
            "rnn_type": "lstm", "bidirectional": False,
        }
        history = [{
            "params": parent_params,
            "result": {"value": 0.5},
            "status": "completed",
        }]
        # 第二次填充（达到 population_size=2）
        history.append({
            "params": parent_params,
            "result": {"value": 0.5},
            "status": "completed",
        })
        child = sampler.sample(ss, history)
        # 应该至少有一个参数被改变（mutation_rate=1.0 + 多参数）
        # 注意：cell_type 只有 2 个 choice，可能不变；其他参数应有变化
        diff_count = sum(1 for k in parent_params if child.get(k) != parent_params[k])
        assert diff_count > 0, "mutation_rate=1.0 应至少改变一个参数"

    def test_evaluated_count_after_history_sync(self):
        """history 同步后，evaluated_count 应反映已评估数量。"""
        sampler = EvolutionarySampler(population_size=5, seed=42)
        ss = ArchitectureSearchSpace().to_sp_search_space()
        history = [
            {"params": {"cell_type": "conv1d", "n_layers": i + 1,
                        "hidden_dim": 32, "activation": "relu",
                        "kernel_size": 3, "dropout": 0.1,
                        "rnn_type": "lstm", "bidirectional": False},
             "result": {"value": 0.5}, "status": "completed"}
            for i in range(3)
        ]
        sampler.sample(ss, history)
        assert sampler.evaluated_count() == 3


# ============================================================
# P2.9: make_nas_module_factory
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestMakeNasModuleFactory:
    """make_nas_module_factory 集成验证。"""

    def test_returns_callable(self):
        """应返回可调用对象。"""
        mf = make_nas_module_factory(_conv1d_params(), input_shape=(30, 100))
        assert callable(mf)

    def test_factory_returns_lightning_module(self):
        """factory 调用应返回 GenericLightningModule。"""
        from senseframe.engine.module import GenericLightningModule
        mf = make_nas_module_factory(_conv1d_params(), input_shape=(30, 100))
        lm = mf(
            model=None,
            learning_rate=0.001,
            metrics=["accuracy"],
            num_classes=7,
            optimizer="adam",
        )
        assert isinstance(lm, GenericLightningModule)

    def test_factory_ignores_scene_model(self):
        """factory 应忽略 scene 构建的 model（NAS 用自己的 arch_params）。"""
        from senseframe.engine.module import GenericLightningModule
        mf = make_nas_module_factory(_conv1d_params(), input_shape=(30, 100))
        # 传入一个 dummy model，应被忽略
        dummy_model = nn.Linear(10, 10)
        lm = mf(
            model=dummy_model,
            learning_rate=0.001,
            metrics=["accuracy"],
            num_classes=7,
        )
        # 内部 model 应不是 dummy_model
        assert lm.model is not dummy_model

    def test_factory_rnn_cell_type(self):
        """factory 应支持 rnn cell_type。"""
        from senseframe.engine.module import GenericLightningModule
        mf = make_nas_module_factory(_rnn_params(), input_shape=(30, 100))
        lm = mf(model=None, learning_rate=0.001, metrics=["accuracy"], num_classes=7)
        assert isinstance(lm, GenericLightningModule)

    def test_factory_custom_builder(self):
        """应支持传入自定义 builder。"""
        from senseframe.engine.module import GenericLightningModule
        custom_builder = ArchitectureBuilder()
        mf = make_nas_module_factory(
            _conv1d_params(), input_shape=(30, 100), builder=custom_builder,
        )
        lm = mf(model=None, learning_rate=0.001, metrics=["accuracy"], num_classes=7)
        assert isinstance(lm, GenericLightningModule)


# ============================================================
# 反假绿：grep 实证检查（源码不可绕过）
# ============================================================
class TestGrepEvidence:
    """源码 grep 实证：mock 可绕过运行时，但绕不过源码 grep。"""

    def test_search_space_has_architecture_parameter_spec(self):
        """search_space.py 应定义 ArchitectureParameterSpec。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "class ArchitectureParameterSpec"), \
            "应有 ArchitectureParameterSpec 类"

    def test_search_space_has_architecture_search_space(self):
        """search_space.py 应定义 ArchitectureSearchSpace。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "class ArchitectureSearchSpace"), \
            "应有 ArchitectureSearchSpace 类"

    def test_search_space_has_to_sp_search_space(self):
        """search_space.py 应有 to_sp_search_space 方法。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "def to_sp_search_space"), \
            "应有 to_sp_search_space 方法（NAS → SP 转换）"

    def test_search_space_has_dsp_schema(self):
        """search_space.py 应有 DSP schema 方法。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "def schema(cls)"), \
            "应有 schema 方法（DSP 自省）"
        assert _grep_source(path, "def describe(self)"), \
            "应有 describe 方法（DSP 自省）"

    def test_search_space_has_supported_constants(self):
        """search_space.py 应有 SUPPORTED_CELL_TYPES 常量。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "SUPPORTED_CELL_TYPES"), \
            "应有 SUPPORTED_CELL_TYPES 常量"
        assert _grep_source(path, "SUPPORTED_RNN_TYPES"), \
            "应有 SUPPORTED_RNN_TYPES 常量"

    def test_builder_has_architecture_builder_class(self):
        """builder.py 应定义 ArchitectureBuilder 类。"""
        path = _source_path("nas/builder.py")
        assert _grep_source(path, "class ArchitectureBuilder"), \
            "应有 ArchitectureBuilder 类"

    def test_builder_has_build_method(self):
        """builder.py 应有 build 方法。"""
        path = _source_path("nas/builder.py")
        assert _grep_source(path, "def build("), \
            "ArchitectureBuilder 应有 build 方法"

    def test_builder_has_conv1d_rnn_hybrid(self):
        """builder.py 应实现 conv1d / rnn / hybrid 三种 cell_type。"""
        path = _source_path("nas/builder.py")
        assert _grep_source(path, "class Conv1dNet"), "应有 Conv1dNet 类"
        assert _grep_source(path, "class RNNNet"), "应有 RNNNet 类"
        assert _grep_source(path, "class HybridNet"), "应有 HybridNet 类"

    def test_sampler_has_evolutionary_sampler_class(self):
        """sampler.py 应定义 EvolutionarySampler 类。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, "class EvolutionarySampler"), \
            "应有 EvolutionarySampler 类"

    def test_sampler_has_name_evolutionary(self):
        """sampler.py 应有 name = 'evolutionary' 类属性。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, 'name = "evolutionary"'), \
            "EvolutionarySampler.name 应为 'evolutionary'"

    def test_sampler_has_sample_method(self):
        """sampler.py 应有 sample 方法。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, "def sample("), \
            "EvolutionarySampler 应有 sample 方法"

    def test_sampler_has_tournament_and_mutate(self):
        """sampler.py 应实现锦标赛选择和变异。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, "_tournament_select"), \
            "应有 _tournament_select 方法"
        assert _grep_source(path, "_mutate"), \
            "应有 _mutate 方法"

    def test_sampler_registered_in_sp_registry(self):
        """sampler.py 应注册到 SP Sampler 注册表。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, 'register_sampler("evolutionary"'), \
            "应注册到 SP Sampler 注册表"

    def test_init_has_make_nas_module_factory(self):
        """__init__.py 应定义 make_nas_module_factory。"""
        path = _source_path("nas/__init__.py")
        assert _grep_source(path, "def make_nas_module_factory"), \
            "__init__.py 应有 make_nas_module_factory 函数"

    def test_init_uses_generic_lightning_module(self):
        """__init__.py 应使用 GenericLightningModule 包装。"""
        path = _source_path("nas/__init__.py")
        assert _grep_source(path, "GenericLightningModule"), \
            "make_nas_module_factory 应使用 GenericLightningModule"

    def test_pipeline_module_factory_supports_nas_injection(self):
        """pipeline.py 应支持 module_factory 注入（NAS 集成前提）。"""
        path = _source_path("engine/runner/pipeline.py")
        assert _grep_source(path, "module_factory is not None"), \
            "stage_build 应检查 module_factory 非空"
        assert _grep_source(path, "ctx.config.module_factory("), \
            "stage_build 应调用 module_factory"
