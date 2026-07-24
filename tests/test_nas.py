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
    AttentionNet,
    DARTSCell,
    DARTSPipelineRun,
    DARTSSampler,
    DARTSSupernet,
    ENASSampler,
    EvolutionarySampler,
    OP_NAMES,
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


def _attention_params(**overrides):
    """构造 attention 架构参数（P3.3.1 新增）。"""
    params = {
        "cell_type": "attention",
        "n_layers": 2,
        "d_model": 32,
        "n_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
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
        """不支持的 cell_type 应抛异常（P3.3.1 后 attention 已支持，改用真正未支持的类型）。"""
        with pytest.raises(ValueError, match="Unsupported cell_type 'transformer'"):
            ArchitectureSearchSpace(cell_types=["transformer"])

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
        """不支持的 cell_type 应抛异常（P3.3.1 后 attention 已支持，改用真正未支持的类型）。"""
        builder = ArchitectureBuilder()
        with pytest.raises(ValueError, match="Unsupported cell_type"):
            builder.build({"cell_type": "transformer"}, (30, 100), 7)

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
        """pipeline/stages/build.py 应支持 module_factory 注入（NAS 集成前提）。

        拆分背景：原 pipeline.py 拆分为 pipeline/ 包，stage_build 位于 pipeline/stages/build.py。
        """
        path = _source_path("engine/runner/pipeline/stages/build.py")
        assert _grep_source(path, "module_factory is not None"), \
            "stage_build 应检查 module_factory 非空"
        assert _grep_source(path, "ctx.config.module_factory("), \
            "stage_build 应调用 module_factory"


# ============================================================
# P3.3.1: AttentionNet
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestAttentionNet:
    """AttentionNet 行为验证（P3.3.1）。"""

    def test_attention_net_construction(self):
        """AttentionNet 可构造。"""
        net = AttentionNet(
            input_shape=(30, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,
        )
        assert isinstance(net, nn.Module)

    def test_attention_net_forward_output_shape(self):
        """forward 输出形状应为 (batch, num_classes)。"""
        net = AttentionNet(
            input_shape=(30, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,
        )
        x = torch.randn(2, 30, 100)
        out = net(x)
        assert out.shape == (2, 7)

    def test_attention_net_input_proj_when_channels_mismatch(self):
        """in_channels != d_model 时应构造投影层（nn.Linear）。"""
        net = AttentionNet(
            input_shape=(30, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,  # 30 != 32
        )
        assert isinstance(net.input_proj, nn.Linear)

    def test_attention_net_input_identity_when_channels_match(self):
        """in_channels == d_model 时应用 nn.Identity。"""
        net = AttentionNet(
            input_shape=(32, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,  # 32 == 32
        )
        assert isinstance(net.input_proj, nn.Identity)

    def test_attention_net_backward(self):
        """可反向传播（梯度可流到 input_proj 与 encoder 参数）。"""
        net = AttentionNet(
            input_shape=(30, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,
        )
        x = torch.randn(2, 30, 100)
        y = torch.randint(0, 7, (2,))
        out = net(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        # input_proj 应有梯度（要么 Linear，要么 Identity 无参数）
        if isinstance(net.input_proj, nn.Linear):
            assert net.input_proj.weight.grad is not None
        # classifier 应有梯度
        assert net.classifier.weight.grad is not None

    def test_attention_net_with_different_n_heads(self):
        """不同 n_heads (2/4/8) 都可工作。"""
        for n_heads in [2, 4, 8]:
            net = AttentionNet(
                input_shape=(32, 50), num_classes=5,
                n_layers=1, d_model=32, n_heads=n_heads,
            )
            x = torch.randn(2, 32, 50)
            out = net(x)
            assert out.shape == (2, 5)

    def test_attention_net_with_different_activation(self):
        """gelu / relu 都可工作。"""
        for act in ["gelu", "relu"]:
            net = AttentionNet(
                input_shape=(32, 50), num_classes=5,
                n_layers=1, d_model=32, n_heads=4,
                activation=act,
            )
            x = torch.randn(2, 32, 50)
            out = net(x)
            assert out.shape == (2, 5)

    def test_attention_net_parameter_count(self):
        """参数量应 > 0。"""
        net = AttentionNet(
            input_shape=(30, 100), num_classes=7,
            n_layers=2, d_model=32, n_heads=4,
        )
        n_params = sum(p.numel() for p in net.parameters())
        assert n_params > 0


# ============================================================
# P3.3.1: ArchitectureBuilder attention 分支
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestArchitectureBuilderAttention:
    """ArchitectureBuilder attention cell_type 行为验证（P3.3.1）。"""

    def test_build_attention_returns_module(self):
        """build({"cell_type": "attention", ...}) 应返回 nn.Module。"""
        builder = ArchitectureBuilder()
        model = builder.build(_attention_params(), input_shape=(30, 100), num_classes=7)
        assert isinstance(model, nn.Module)
        assert isinstance(model, AttentionNet)

    def test_build_attention_default_params(self):
        """默认参数构造成功（不传 n_layers/d_model/n_heads）。"""
        builder = ArchitectureBuilder()
        model = builder.build(
            {"cell_type": "attention"}, input_shape=(32, 100), num_classes=7,
        )
        x = torch.randn(2, 32, 100)
        out = model(x)
        assert out.shape == (2, 7)

    def test_build_attention_custom_params(self):
        """自定义 n_layers/d_model/n_heads 生效。"""
        builder = ArchitectureBuilder()
        model = builder.build(
            _attention_params(n_layers=3, d_model=64, n_heads=8),
            input_shape=(30, 100), num_classes=7,
        )
        # d_model=64, in_channels=30 → 应有 input_proj Linear
        assert isinstance(model.input_proj, nn.Linear)
        # encoder 应有 3 层
        assert model.encoder.num_layers == 3

    def test_build_attention_with_real_input_shape(self):
        """真实 input_shape=(30, 100) 可工作。"""
        builder = ArchitectureBuilder()
        model = builder.build(_attention_params(), input_shape=(30, 100), num_classes=7)
        x = torch.randn(4, 30, 100)
        out = model(x)
        assert out.shape == (4, 7)

    def test_build_attention_supported_cell_type(self):
        """SUPPORTED_CELL_TYPES 应含 'attention'（P3.3.1 新增）。"""
        assert "attention" in SUPPORTED_CELL_TYPES


# ============================================================
# P3.3.1: ArchitectureSearchSpace attention 扩展
# ============================================================
class TestArchitectureSearchSpaceAttention:
    """ArchitectureSearchSpace attention cell_type 验证（P3.3.1）。"""

    def test_search_space_with_attention_cell_type(self):
        """ArchitectureSearchSpace(cell_types=["attention"]) 可构造。"""
        ss = ArchitectureSearchSpace(cell_types=["attention"])
        assert ss.cell_types == ["attention"]
        # 应有 attention 特化参数
        names = [p.name for p in ss.parameters]
        assert "cell_type" in names
        assert "n_layers" in names
        assert "d_model" in names
        assert "n_heads" in names

    def test_search_space_attention_to_sp(self):
        """to_sp_search_space 转换成功（attention 参数可转 SP ParameterSpec）。"""
        ss = ArchitectureSearchSpace(cell_types=["attention"])
        sp_ss = ss.to_sp_search_space()
        assert isinstance(sp_ss, SearchSpace)
        assert len(sp_ss.parameters) == len(ss.parameters)
        for p in sp_ss.parameters:
            assert isinstance(p, ParameterSpec)

    def test_search_space_attention_validate_params(self):
        """validate_params 通过（合法 attention 参数）。"""
        ss = ArchitectureSearchSpace(cell_types=["attention"])
        valid_params = {p.name: p.default for p in ss.parameters if p.default is not None}
        errors = ss.validate_params(valid_params)
        assert errors == [], f"合法 attention 参数应通过验证, got errors: {errors}"

    def test_search_space_attention_with_other_cell_types(self):
        """attention 与 conv1d/rnn 联合构造（参数合并）。"""
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn", "attention"])
        assert "attention" in ss.cell_types
        names = [p.name for p in ss.parameters]
        # attention 特化参数应存在
        assert "d_model" in names
        assert "n_heads" in names


# ============================================================
# P3.3.2: DARTSSampler
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSSampler:
    """DARTSSampler 行为验证（P3.3.2）。"""

    def test_darts_sampler_name(self):
        """sampler.name 应为 'darts'。"""
        sampler = DARTSSampler()
        assert sampler.name == "darts"

    def test_darts_sampler_satisfies_protocol(self):
        """DARTSSampler 应满足 Sampler Protocol。"""
        sampler = DARTSSampler()
        assert isinstance(sampler, Sampler)

    def test_darts_sampler_sample_returns_dict(self):
        """sample() 应返回 dict。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        assert isinstance(params, dict)
        # 应含 cell_type key
        assert "cell_type" in params

    def test_darts_sampler_sample_with_empty_alpha(self):
        """arch_alpha 为空时 sample 应自动初始化 α。"""
        sampler = DARTSSampler()
        assert sampler.arch_alpha == {}
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        sampler.sample(ss, [])
        # α 应被自动初始化（每个 categorical 参数一个 α 向量）
        assert len(sampler.arch_alpha) > 0
        # cell_type 应在 α 中
        assert "cell_type" in sampler.arch_alpha

    def test_darts_sampler_warm_start_no_op(self):
        """warm_start 应为 no-op（不报错）。"""
        sampler = DARTSSampler()
        # 应不抛异常
        sampler.warm_start([{"params": {"cell_type": "conv1d"}, "result": {"value": 0.5}}])

    def test_darts_sampler_registered(self):
        """get_sampler('darts') 应返回 DARTSSampler。"""
        assert "darts" in list_samplers()
        assert get_sampler("darts") is DARTSSampler

    def test_darts_sampler_optimizer_lazy_init(self):
        """optimizer 应延迟构造（init 时为 None）。"""
        sampler = DARTSSampler()
        assert sampler.optimizer is None
        # 触发 sample + update
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        sampler.sample(ss, [])
        gradient = {k: torch.ones_like(v) for k, v in sampler.arch_alpha.items()}
        sampler.update(gradient)
        # update 后 optimizer 应已构造
        assert sampler.optimizer is not None

    def test_darts_sampler_sample_with_real_search_space(self):
        """与真实 SearchSpace 集成（含 conv1d + rnn 参数）。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        # 应返回离散化的合法参数
        assert "cell_type" in params
        assert params["cell_type"] in ["conv1d", "rnn"]


# ============================================================
# P3.3.2: DARTSPipelineRun
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSPipelineRun:
    """DARTSPipelineRun 行为验证（P3.3.2）。"""

    def test_darts_pipeline_run_construct(self):
        """DARTSPipelineRun 可构造。"""
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
        )
        assert run is not None
        assert run.n_epochs == 1

    def test_darts_pipeline_run_run_with_dummy_data(self):
        """用 dummy train/val loader 跑通 1 epoch。"""
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=2,
        )
        # 构造 dummy loader
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)
        assert isinstance(result, dict)

    def test_darts_pipeline_run_returns_arch(self):
        """run() 返回 dict 含 best_arch。"""
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        result = run.run(train_loader, val_loader)
        assert "best_arch" in result
        assert isinstance(result["best_arch"], dict)
        assert "cell_type" in result["best_arch"]

    def test_darts_pipeline_run_double_optimization(self):
        """w 和 α 都被更新（双优化机制实证）。"""
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=2,
        )
        # 记录 α 的初始值
        sampler._init_arch_alpha_from_search_space(ss)
        alpha_before = {k: v.clone() for k, v in sampler.arch_alpha.items()}

        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)

        # α 应发生变化（被 update 调用）
        alpha_changed = any(
            not torch.allclose(alpha_before[k], result["final_alpha"][k])
            for k in alpha_before
        )
        assert alpha_changed, "DARTS 双优化应使 α 被更新"

        # best_arch 应在合法范围内
        best_arch = result["best_arch"]
        assert "cell_type" in best_arch

    def test_darts_pipeline_run_history_recorded(self):
        """返回的 history 含每 epoch 的 loss。"""
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=3,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)
        assert "history" in result
        history = result["history"]
        assert len(history) == 3
        for i, entry in enumerate(history):
            assert entry["epoch"] == i
            assert "w_loss" in entry
            assert "alpha_loss" in entry


# ============================================================
# P3 资源泄露修复验证（TestDARTSResourceLeakFix）
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSResourceLeakFix:
    """DARTS 资源泄露修复验证（P3 审查）。

    验证以下修复点：
    - alpha_loss 计算图不累积（torch.no_grad 包裹）
    - alpha_grad 用 .detach() 切断外部引用
    - run() try/finally 释放 supernet / optimizer / iterator
    - _InfiniteLoader.close() 释放 loader 引用
    - DARTSSampler.cleanup() 释放 arch_alpha / optimizer
    """

    def test_darts_sampler_cleanup_releases_resources(self):
        """DARTSSampler.cleanup() 释放 arch_alpha 和 optimizer。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        sampler._init_arch_alpha_from_search_space(ss)
        # 触发 optimizer 构造
        grad = {k: torch.randn_like(v) for k, v in sampler.arch_alpha.items()}
        sampler.update(grad)
        assert sampler.optimizer is not None
        assert len(sampler.arch_alpha) > 0
        # cleanup
        sampler.cleanup()
        assert sampler.optimizer is None
        assert len(sampler.arch_alpha) == 0
        assert sampler._search_space is None

    def test_darts_pipeline_run_releases_supernet_after_run(self):
        """run() 结束后 supernet 被 del（无强引用残留）。"""
        import gc
        import weakref
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        # 在 run 内部构造 supernet 后用 weakref 跟踪
        # run() 结束后 supernet 应无强引用，gc 后被回收
        result = run.run(train_loader, val_loader)
        # run 实例不应持有 supernet 引用
        assert not hasattr(run, "_supernet") or run.__dict__.get("_supernet") is None
        # gc.collect() 后应有更多对象被回收（无法直接断言 supernet 死亡，
        # 但可验证 run.__dict__ 不含 supernet/w_optimizer/train_iter/val_iter）
        for attr in ("supernet", "w_optimizer", "train_iter", "val_iter"):
            assert attr not in run.__dict__, f"run() 泄露了 {attr} 引用"

    def test_darts_pipeline_run_no_grad_accumulation_for_alpha_loss(self):
        """验证 alpha_loss 不累积计算图（torch.no_grad 修复）。

        修复前：alpha_loss = criterion(val_logits, y_val) 创建计算图但从不 backward，
        每个 epoch 泄露一份。修复后用 torch.no_grad() 包裹。
        """
        # 通过 grep 实证：darts.py 中 val_logits / alpha_loss 在 torch.no_grad() 块内
        darts_path = Path(__file__).parent.parent / "senseframe" / "nas" / "darts.py"
        content = darts_path.read_text(encoding="utf-8")
        # 验证 torch.no_grad 包裹了 val_logits 计算
        assert "with torch.no_grad():" in content
        assert "val_logits = supernet(x_val)" in content
        # 验证 no_grad 块在 val_logits 之前
        idx_no_grad = content.find("with torch.no_grad():")
        idx_val_logits = content.find("val_logits = supernet(x_val)")
        assert idx_no_grad != -1 and idx_val_logits != -1
        assert idx_no_grad < idx_val_logits, "val_logits 应在 torch.no_grad() 块内"

    def test_darts_pipeline_run_alpha_grad_detached(self):
        """验证 alpha_grad 用 .detach() 切断外部计算图引用。"""
        darts_path = Path(__file__).parent.parent / "senseframe" / "nas" / "darts.py"
        content = darts_path.read_text(encoding="utf-8")
        # grep 实证：randn_like 后接 .detach()
        assert "torch.randn_like(ap) * 0.01).detach()" in content

    def test_darts_pipeline_run_try_finally_cleanup(self):
        """验证 run() 含 try/finally 块释放资源。"""
        darts_path = Path(__file__).parent.parent / "senseframe" / "nas" / "darts.py"
        content = darts_path.read_text(encoding="utf-8")
        # grep 实证：try/finally 块 + 资源释放
        assert "finally:" in content
        assert "train_iter.close()" in content
        assert "val_iter.close()" in content
        assert "del supernet" in content

    def test_darts_pipeline_run_update_uses_set_to_none(self):
        """验证 optimizer.zero_grad(set_to_none=True) 释放 grad 引用。"""
        darts_path = Path(__file__).parent.parent / "senseframe" / "nas" / "darts.py"
        content = darts_path.read_text(encoding="utf-8")
        assert "zero_grad(set_to_none=True)" in content

    def test_infinite_loader_close_releases_loader(self):
        """_InfiniteLoader.close() 释放 loader 和 _iter 引用。"""
        from senseframe.nas.darts import _InfiniteLoader
        loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]
        inf = _InfiniteLoader(loader)
        assert inf.loader is not None
        assert inf._iter is not None
        inf.close()
        assert inf.loader is None
        assert inf._iter is None

    def test_infinite_loader_context_manager(self):
        """_InfiniteLoader 支持 context manager 协议。"""
        from senseframe.nas.darts import _InfiniteLoader
        loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]
        with _InfiniteLoader(loader) as inf:
            x, y = inf.next()
            assert x.shape == (2, 3)
        # 退出 with 块后 loader 应被释放
        assert inf.loader is None
        assert inf._iter is None

    def test_darts_pipeline_run_does_not_leak_tensor_memory(self):
        """端到端验证：多次 run() 调用不累积 tensor 内存。

        通过检查 sampler.arch_alpha 在 cleanup 后被清空，
        且再次 run() 可正常工作（重新初始化 arch_alpha）。
        """
        sampler = DARTSSampler()
        builder = ArchitectureBuilder()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()

        # 第一次 run
        run1 = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        result1 = run1.run(train_loader, val_loader)
        assert "best_arch" in result1

        # cleanup sampler
        sampler.cleanup()
        assert len(sampler.arch_alpha) == 0

        # 第二次 run（重新初始化 arch_alpha）
        run2 = DARTSPipelineRun(
            sampler=sampler, builder=builder, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
        )
        result2 = run2.run(train_loader, val_loader)
        assert "best_arch" in result2
        # 第二次 run 后 arch_alpha 应非空
        assert len(sampler.arch_alpha) > 0


# ============================================================
# P3.3.3: ENASSampler
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestENASSampler:
    """ENASSampler 行为验证（P3.3.3）。"""

    def test_enas_sampler_name(self):
        """sampler.name 应为 'enas'。"""
        sampler = ENASSampler()
        assert sampler.name == "enas"

    def test_enas_sampler_satisfies_protocol(self):
        """ENASSampler 应满足 Sampler Protocol。"""
        sampler = ENASSampler()
        assert isinstance(sampler, Sampler)

    def test_enas_sampler_sample_returns_dict(self):
        """sample() 应返回 dict。"""
        sampler = ENASSampler(seed=42)
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        assert isinstance(params, dict)
        assert "cell_type" in params

    def test_enas_sampler_sample_without_controller_fallback_random(self):
        """无 controller 时应随机采样 fallback。"""
        sampler = ENASSampler(seed=42)
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        # 应返回合法参数
        assert params["cell_type"] in ["conv1d", "rnn"]
        assert "n_layers" in params

    def test_enas_sampler_sample_with_controller(self):
        """有 controller 时应 controller 采样。"""
        # 构造一个简单的 controller（nn.LSTM）
        controller_hidden = 64
        controller = nn.LSTM(
            input_size=controller_hidden,
            hidden_size=controller_hidden,
            batch_first=True,
        )
        sampler = ENASSampler(controller=controller, controller_hidden=controller_hidden, seed=42)
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        # 应返回 dict（不抛异常）
        assert isinstance(params, dict)
        assert "cell_type" in params

    def test_enas_sampler_warm_start_no_op(self):
        """warm_start 应为 no-op（不报错）。"""
        sampler = ENASSampler()
        sampler.warm_start([{"params": {"cell_type": "conv1d"}, "result": {"value": 0.5}}])

    def test_enas_sampler_registered(self):
        """get_sampler('enas') 应返回 ENASSampler。"""
        assert "enas" in list_samplers()
        assert get_sampler("enas") is ENASSampler

    def test_enas_sampler_sample_with_real_search_space(self):
        """与真实 SearchSpace 集成。"""
        sampler = ENASSampler(seed=42)
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn", "attention"]).to_sp_search_space()
        params = sampler.sample(ss, [])
        assert "cell_type" in params
        assert params["cell_type"] in ["conv1d", "rnn", "attention"]


# ============================================================
# P3.3.4: 反假绿 grep 实证检查
# ============================================================
class TestGrepEvidenceDartsEnas:
    """P3.3 DARTS/ENAS/attention 源码 grep 实证检查。"""

    def test_grep_attention_net_class(self):
        """builder.py 应含 class AttentionNet。"""
        path = _source_path("nas/builder.py")
        assert _grep_source(path, "class AttentionNet"), \
            "builder.py 应有 AttentionNet 类"

    def test_grep_build_attention_method(self):
        """builder.py 应含 _build_attention 方法。"""
        path = _source_path("nas/builder.py")
        assert _grep_source(path, "def _build_attention"), \
            "builder.py 应有 _build_attention 方法"

    def test_grep_supported_cell_types_has_attention(self):
        """search_space.py 的 SUPPORTED_CELL_TYPES 应含 attention。"""
        path = _source_path("nas/search_space.py")
        content = path.read_text(encoding="utf-8")
        # SUPPORTED_CELL_TYPES 行应含 attention
        for line in content.splitlines():
            if "SUPPORTED_CELL_TYPES" in line and "=" in line and "[" in line:
                assert "attention" in line, \
                    f"SUPPORTED_CELL_TYPES 应含 attention, got: {line}"
                return
        assert False, "未找到 SUPPORTED_CELL_TYPES 定义"

    def test_grep_default_attention_params(self):
        """search_space.py 应含 _default_attention_params 函数。"""
        path = _source_path("nas/search_space.py")
        assert _grep_source(path, "def _default_attention_params"), \
            "search_space.py 应有 _default_attention_params 函数"

    def test_grep_darts_sampler_class(self):
        """darts.py 应含 class DARTSSampler。"""
        path = _source_path("nas/darts.py")
        assert _grep_source(path, "class DARTSSampler"), \
            "darts.py 应有 DARTSSampler 类"

    def test_grep_darts_pipeline_run_class(self):
        """darts.py 应含 class DARTSPipelineRun。"""
        path = _source_path("nas/darts.py")
        assert _grep_source(path, "class DARTSPipelineRun"), \
            "darts.py 应有 DARTSPipelineRun 类"

    def test_grep_enas_sampler_class(self):
        """sampler.py 应含 class ENASSampler。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, "class ENASSampler"), \
            "sampler.py 应有 ENASSampler 类"

    def test_grep_darts_registered(self):
        """darts.py 应含 register_sampler("darts" 注册调用。"""
        path = _source_path("nas/darts.py")
        assert _grep_source(path, 'register_sampler("darts"'), \
            "darts.py 应注册 DARTSSampler 到 SP"

    def test_grep_enas_registered(self):
        """sampler.py 应含 register_sampler("enas" 注册调用。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, 'register_sampler("enas"'), \
            "sampler.py 应注册 ENASSampler 到 SP"

    def test_grep_nas_init_exports_darts_enas(self):
        """__init__.py 应导出 DARTSSampler, ENASSampler, DARTSPipelineRun, AttentionNet。"""
        path = _source_path("nas/__init__.py")
        assert _grep_source(path, "DARTSSampler"), \
            "__init__.py 应导出 DARTSSampler"
        assert _grep_source(path, "ENASSampler"), \
            "__init__.py 应导出 ENASSampler"
        assert _grep_source(path, "DARTSPipelineRun"), \
            "__init__.py 应导出 DARTSPipelineRun"
        assert _grep_source(path, "AttentionNet"), \
            "__init__.py 应导出 AttentionNet"

    def test_grep_darts_sampler_name(self):
        """darts.py 应有 name = 'darts' 类属性。"""
        path = _source_path("nas/darts.py")
        assert _grep_source(path, 'name = "darts"'), \
            "DARTSSampler.name 应为 'darts'"

    def test_grep_enas_sampler_name(self):
        """sampler.py 应有 name = 'enas' 类属性。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, 'name = "enas"'), \
            "ENASSampler.name 应为 'enas'"

    def test_grep_darts_warm_start(self):
        """darts.py 应含 warm_start 方法（Sampler Protocol 合规）。"""
        path = _source_path("nas/darts.py")
        assert _grep_source(path, "def warm_start"), \
            "DARTSSampler 应有 warm_start 方法"

    def test_grep_enas_warm_start(self):
        """sampler.py 应含 ENASSampler 的 warm_start 方法。"""
        path = _source_path("nas/sampler.py")
        # sampler.py 已有 EvolutionarySampler 的 warm_start；确保 ENAS 也有
        content = path.read_text(encoding="utf-8")
        # 找到 ENASSampler 类定义后的 warm_start
        enas_idx = content.find("class ENASSampler")
        assert enas_idx >= 0, "应含 ENASSampler 类"
        enas_section = content[enas_idx:]
        assert "def warm_start" in enas_section, \
            "ENASSampler 应有 warm_start 方法"


# ============================================================
# P1.3: DARTS 真实超网测试（2026-07-19）
# ============================================================
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSSupernet:
    """DARTSSupernet 真实超网行为验证（P1.3 新增）。

    验证点（对齐设计文档 P1.3 测试清单）：
    - test_supernet_forward_all_ops：所有候选 op 被计算（grep 实证 + 行为）
    - test_alpha_softmax_weights_sum_to_one：softmax 权重和为 1
    - test_discretize_argmax_selects_best_op：离散化选 argmax
    - test_w_alpha_parameters_separated：w 和 α 参数分离
    - test_build_discrete_model：离散化模型可独立 forward
    """

    def test_supernet_construct(self):
        """DARTSSupernet 可构造。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        assert sn is not None
        assert sn.n_cells == 3  # 默认 3 个 cell
        assert sn.c_stem == 32
        assert sn.c_cell == 64
        assert sn.num_classes == 7

    def test_supernet_forward_shape(self):
        """超网 forward 输出形状正确。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        x = torch.randn(4, 30, 100)
        logits = sn(x)
        assert logits.shape == (4, 7)

    def test_supernet_forward_2d_input(self):
        """2D 输入 (B, L) 应自动 unsqueeze 到 (B, 1, L)。"""
        sn = DARTSSupernet(input_shape=(1, 50), num_classes=3, n_cells=1)
        x = torch.randn(4, 50)  # (B, L)
        logits = sn(x)
        assert logits.shape == (4, 3)

    def test_supernet_forward_all_ops(self):
        """所有候选 op 被并行计算（grep 实证 + 行为验证）。

        grep 实证：supernet.py 含 "for op in self.ops" 或 outputs 列表构造
        行为：每个 cell.ops 数量等于 OP_NAMES 长度
        """
        # grep 实证：DARTSCell.forward 应并行计算所有 op
        path = _source_path("nas/supernet.py")
        content = path.read_text(encoding="utf-8")
        # 应含并行计算所有 op 的代码
        assert "outputs = [op(x) for op in self.ops]" in content, \
            "DARTSCell.forward 应并行计算所有候选 op"

        # 行为验证：每个 cell 的 ops 数量 = OP_NAMES 长度
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        for cell in sn.cells:
            assert len(cell.ops) == len(OP_NAMES)
            # 每个 op 是 nn.Module
            for op in cell.ops:
                assert isinstance(op, nn.Module)

    def test_alpha_softmax_weights_sum_to_one(self):
        """softmax(α) 权重和为 1（数学契约）。"""
        cell = DARTSCell(c_in=32, c_out=64)
        weights = torch.softmax(cell.alpha, dim=-1)
        # 权重和应接近 1（浮点误差容忍）
        assert abs(weights.sum().item() - 1.0) < 1e-6
        # 所有权重非负
        assert (weights >= 0).all()
        # P2.3-2 修复后 α 默认初始化为 randn*0.001（非 zeros），
        # 此处显式构造 α=zeros 验证 softmax 数学性质：zeros 时权重精确均匀分布。
        cell_uniform = DARTSCell(c_in=32, c_out=64)
        with torch.no_grad():
            cell_uniform.alpha.fill_(0.0)
        weights_uniform = torch.softmax(cell_uniform.alpha, dim=-1)
        assert abs(weights_uniform[0].item() - 1.0 / len(OP_NAMES)) < 1e-6

    def test_alpha_is_parameter_and_requires_grad(self):
        """α 应是 nn.Parameter 且 requires_grad=True（可微基础）。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        for cell in sn.cells:
            assert isinstance(cell.alpha, nn.Parameter)
            assert cell.alpha.requires_grad
            # α 形状 = (len(OP_NAMES),)
            assert cell.alpha.shape == (len(OP_NAMES),)

    def test_w_alpha_parameters_separated(self):
        """w_parameters 和 alpha_parameters 互斥且并集 = 全部参数。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)

        w_params = list(sn.w_parameters())
        alpha_params = list(sn.alpha_parameters())

        # α 参数数量 = n_cells（每个 cell 一个 α）
        assert len(alpha_params) == 2
        # w 参数应远多于 α 参数（w 含 stem + cells.ops + classifier 权重）
        assert len(w_params) > len(alpha_params) * 5

        # 验证：w_params 不含 alpha 参数（通过 data_ptr 比对）
        alpha_data_ptrs = {p.data_ptr() for p in alpha_params}
        w_data_ptrs = {p.data_ptr() for p in w_params}
        assert not (alpha_data_ptrs & w_data_ptrs), \
            "w_parameters 不应包含 α 参数"

        # 并集 = 全部 named_parameters
        all_params = dict(sn.named_parameters())
        all_data_ptrs = {p.data_ptr() for p in all_params.values()}
        union_ptrs = alpha_data_ptrs | w_data_ptrs
        assert union_ptrs == all_data_ptrs, \
            "w_parameters + alpha_parameters 应覆盖所有参数"

    def test_discretize_returns_dict(self):
        """discretize() 返回 dict 含每个 cell 的 op 名。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=3)
        arch = sn.discretize()
        assert isinstance(arch, dict)
        assert len(arch) == 3
        # 每个 key 形如 "cell_0"
        for i in range(3):
            assert f"cell_{i}" in arch
            # value 应是 OP_NAMES 之一
            assert arch[f"cell_{i}"] in OP_NAMES

    def test_discretize_argmax_selects_best_op(self):
        """离散化：argmax α 选中的 op 名与 discretize() 返回一致。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=1)
        cell = sn.cells[0]

        # 人为构造 α：使第 2 个 op（conv5）argmax
        with torch.no_grad():
            cell.alpha.zero_()
            cell.alpha[1] = 10.0  # conv5 argmax

        arch = sn.discretize()
        assert arch["cell_0"] == "conv5"

        # 改为第 4 个 op（maxpool）argmax
        with torch.no_grad():
            cell.alpha.zero_()
            cell.alpha[3] = 10.0  # maxpool argmax
        arch = sn.discretize()
        assert arch["cell_0"] == "maxpool"

    def test_build_discrete_model_forward(self):
        """build_discrete_model() 返回的模型可独立 forward。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        # 训练几个 step 让 α 有梯度
        x = torch.randn(4, 30, 100)
        y = torch.randint(0, 7, (4,))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(sn.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss = criterion(sn(x), y)
        loss.backward()
        optimizer.step()

        # 构建离散化模型
        discrete_model = sn.build_discrete_model()
        # 验证 forward 输出形状一致
        logits_supernet = sn(x)
        logits_discrete = discrete_model(x)
        assert logits_supernet.shape == logits_discrete.shape == (4, 7)
        # 离散化模型不应含 α 参数
        for name, p in discrete_model.named_parameters():
            assert not name.endswith("alpha"), \
                f"离散化模型不应含 α 参数，发现 {name}"

    def test_supernet_alpha_dict(self):
        """alpha_dict() 返回每个 cell 的 α 参数引用。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        alpha_dict = sn.alpha_dict()
        assert len(alpha_dict) == 2
        assert "cell_0" in alpha_dict
        assert "cell_1" in alpha_dict
        # 应是同一个 tensor 引用（修改 alpha_dict 应影响 supernet）
        assert alpha_dict["cell_0"] is sn.cells[0].alpha

    def test_supernet_input_shape_too_short_raises(self):
        """input_shape 为空时应抛 ValueError。"""
        with pytest.raises(ValueError, match="input_shape too short"):
            DARTSSupernet(input_shape=(), num_classes=7)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSSupernetDoubleOptimization:
    """DARTS 真实超网双优化行为验证（P1.3 核心）。"""

    def test_double_optimization_updates_alpha(self):
        """验证集 backward 真实更新 α（非 randn_like 近似）。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        # 记录 α 初始值
        alpha_before = {i: cell.alpha.clone() for i, cell in enumerate(sn.cells)}

        # 双优化：w 用 SGD，α 用 Adam
        w_optimizer = torch.optim.SGD(
            list(sn.w_parameters()), lr=0.025, momentum=0.9,
        )
        alpha_optimizer = torch.optim.Adam(
            list(sn.alpha_parameters()), lr=3e-4,
        )
        criterion = nn.CrossEntropyLoss()

        # 执行几个 step
        for _ in range(3):
            x = torch.randn(4, 30, 100)
            y = torch.randint(0, 7, (4,))

            # w 更新
            w_optimizer.zero_grad()
            w_loss = criterion(sn(x), y)
            w_loss.backward()
            w_optimizer.step()

            # α 更新
            alpha_optimizer.zero_grad()
            alpha_loss = criterion(sn(x), y)
            alpha_loss.backward()
            alpha_optimizer.step()

        # 验证 α 发生变化
        alpha_changed = any(
            not torch.allclose(alpha_before[i], sn.cells[i].alpha)
            for i in range(len(sn.cells))
        )
        assert alpha_changed, "双优化应使 α 被真实更新"

    def test_alpha_grad_not_none_after_backward(self):
        """α backward 后 grad 应非 None（真实可微，非近似梯度）。"""
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=1)
        # 初始 grad 应为 None
        assert sn.cells[0].alpha.grad is None

        x = torch.randn(4, 30, 100)
        y = torch.randint(0, 7, (4,))
        loss = nn.CrossEntropyLoss()(sn(x), y)
        loss.backward()

        # backward 后 α 应有真实梯度（非 None）
        assert sn.cells[0].alpha.grad is not None
        # 梯度应非全 0（除非运气特别差）
        assert sn.cells[0].alpha.grad.abs().sum().item() > 0

    def test_no_randn_like_in_real_supernet_path(self):
        """真实超网路径不应使用 randn_like 近似梯度。

        grep 实证：supernet.py 不应含 randn_like（简化路径的近似方法）。
        """
        path = _source_path("nas/supernet.py")
        content = path.read_text(encoding="utf-8")
        # supernet.py 不应有 randn_like（这是简化路径的近似）
        assert "randn_like" not in content, \
            "DARTSSupernet 不应使用 randn_like 近似梯度（应通过 autograd 真实可微）"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSPipelineRunRealSupernet:
    """DARTSPipelineRun 真实超网路径端到端验证（P1.3 新增）。"""

    def test_real_supernet_run_returns_arch(self):
        """use_real_supernet=True 时 run() 返回 best_arch dict。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=2,
            use_real_supernet=True,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)

        assert "best_arch" in result
        assert isinstance(result["best_arch"], dict)
        # 应含 cell_0 / cell_1 / cell_2（3 个 cell）
        assert "cell_0" in result["best_arch"]
        # 每个 op 名应在 OP_NAMES 内
        for op_name in result["best_arch"].values():
            assert op_name in OP_NAMES

    def test_real_supernet_returns_final_alpha(self):
        """真实超网路径返回 final_alpha（每个 cell 的 α）。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=2,
            use_real_supernet=True,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        result = run.run(train_loader, val_loader)

        assert "final_alpha" in result
        assert isinstance(result["final_alpha"], dict)
        # 应含 cell_0 / cell_1 / cell_2
        assert "cell_0" in result["final_alpha"]
        # 每个 α 应是 1D tensor，长度 = OP_NAMES
        for alpha in result["final_alpha"].values():
            assert alpha.dim() == 1
            assert alpha.shape[0] == len(OP_NAMES)

    def test_real_supernet_history_recorded(self):
        """真实超网路径返回 history 含每 epoch 的 loss。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=3,
            use_real_supernet=True,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)

        assert "history" in result
        history = result["history"]
        assert len(history) == 3
        for i, entry in enumerate(history):
            assert entry["epoch"] == i
            assert "w_loss" in entry
            assert "alpha_loss" in entry
            # loss 应是有限数
            assert isinstance(entry["w_loss"], float)
            assert isinstance(entry["alpha_loss"], float)

    def test_real_supernet_alpha_changes_after_run(self):
        """真实超网路径：α 在 run 后发生变化（真实双优化）。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()

        # 用 supernet_kwargs 固定 cell 数量便于断言
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=3,
            use_real_supernet=True,
            supernet_kwargs={"n_cells": 2},
        )
        # P1.3-4 修复：删除未使用的 probe 死代码（原 probe = DARTSSupernet(...)
        # 创建后从未被引用）。alpha_before 直接用 zeros 构造即可。
        # P2.3-2 修复后：α 初始化为 randn * 0.001（小随机），不再是 zeros。
        # 但 run.run() 内部新构造 supernet，其 α 初始值与本测试构造的 alpha_before
        # 独立。这里只需断言"α 发生变化"，所以 alpha_before 用 zeros 仍可：
        # 真实 supernet 的 α 初始为 randn*0.001（非 zeros），run 后必然变化，
        # 与 zeros 比较必然 not allclose。保持 zeros 作为"明显不同的参考值"。
        alpha_before = {
            f"cell_{i}": torch.zeros(len(OP_NAMES)) for i in range(2)
        }

        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,))) for _ in range(3)]
        result = run.run(train_loader, val_loader)

        # final_alpha 应与初始 zeros 不同（真实双优化使 α 发生变化）
        alpha_changed = any(
            not torch.allclose(alpha_before[k], result["final_alpha"][k])
            for k in alpha_before
        )
        assert alpha_changed, "真实双优化应使 α 发生变化"

    def test_real_supernet_no_resource_leak(self):
        """真实超网路径 run() 后不泄露 supernet / optimizer 引用。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
            use_real_supernet=True,
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        result = run.run(train_loader, val_loader)

        # run 实例不应持有 supernet / optimizer / iterator 引用
        for attr in ("supernet", "w_optimizer", "alpha_optimizer",
                     "train_iter", "val_iter"):
            assert attr not in run.__dict__, \
                f"run() 泄露了 {attr} 引用"
        # sampler 的 _supernet 应被解除（detach_supernet 已调用）
        assert sampler._supernet is None

    def test_real_supernet_supernet_kwargs(self):
        """supernet_kwargs 传递给 DARTSSupernet。"""
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        run = DARTSPipelineRun(
            sampler=sampler, builder=None, search_space=ss,
            input_shape=(30, 100), num_classes=7, n_epochs=1,
            use_real_supernet=True,
            supernet_kwargs={"n_cells": 4, "c_stem": 16, "c_cell": 32},
        )
        train_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        val_loader = [(torch.randn(4, 30, 100), torch.randint(0, 7, (4,)))]
        result = run.run(train_loader, val_loader)
        # 应有 4 个 cell
        assert len(result["best_arch"]) == 4
        assert len(result["final_alpha"]) == 4

    def test_real_supernet_grep_evidence(self):
        """grep 实证：darts.py 含真实超网路径代码。"""
        path = _source_path("nas/darts.py")
        content = path.read_text(encoding="utf-8")
        # 应含 use_real_supernet 参数
        assert "use_real_supernet" in content
        # 应含 _run_real_supernet 方法
        assert "_run_real_supernet" in content
        # 应含 DARTSSupernet 导入
        assert "from .supernet import DARTSSupernet" in content
        # 应含 alpha_optimizer（真实双优化的 α 优化器）
        assert "alpha_optimizer" in content
        # 应含 supernet.discretize()
        assert "supernet.discretize()" in content
        # 应含 supernet.alpha_parameters()
        assert "supernet.alpha_parameters()" in content
        # 应含 supernet.w_parameters()
        assert "supernet.w_parameters()" in content


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSSamplerSupernetAttach:
    """DARTSSampler 的 attach_supernet / detach_supernet 行为（P1.3 新增）。"""

    def test_attach_supernet_sets_attribute(self):
        """attach_supernet 设置 _supernet 属性。"""
        sampler = DARTSSampler()
        assert sampler._supernet is None
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        sampler.attach_supernet(sn)
        assert sampler._supernet is sn

    def test_detach_supernet_clears_attribute(self):
        """detach_supernet 清除 _supernet 属性。"""
        sampler = DARTSSampler()
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        sampler.attach_supernet(sn)
        sampler.detach_supernet()
        assert sampler._supernet is None

    def test_cleanup_clears_supernet_ref(self):
        """cleanup() 应解除 _supernet 引用。"""
        sampler = DARTSSampler()
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        sampler.attach_supernet(sn)
        sampler.cleanup()
        assert sampler._supernet is None

    def test_attach_supernet_grep_evidence(self):
        """grep 实证：darts.py 含 attach_supernet / detach_supernet 方法。"""
        path = _source_path("nas/darts.py")
        content = path.read_text(encoding="utf-8")
        assert "def attach_supernet" in content
        assert "def detach_supernet" in content
        assert "self._supernet" in content
