"""P3 阶段 8 PEFT 微调策略搜索单元测试（反假绿）。

测试原则：
- grep 实证：源码检查不可绕过（sm.ask / sm.tell / PEFTBuilder.build）
- dataclasses.fields 反射：验证字段存在性
- 真实行为：构建真实 nn.Module + 真实 forward pass
- Protocol 契约：isinstance + runtime_checkable
"""
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from senseframe.automl.peft_builder import (
    PEFTBuilder,
    PEFTModel,
    LoRALayer,
    AdapterLayer,
    PrefixTuningLayer,
    PromptTuningLayer,
)
from senseframe.automl.peft_search import (
    build_peft_search_space,
    run_peft_search,
    PEFTSearchResult,
)
from senseframe.core.foundation_model import (
    SensingFoundationModel,
    PretrainConfig,
    PEFTConfig,
)
from senseframe.search_protocol import SearchSpace, ParameterSpec


# ============================================================
# 测试用 backbone
# ============================================================
class _SimpleBackbone(nn.Module):
    """简单 MLP backbone，含 query / value 命名 Linear 便于注入测试。"""

    def __init__(self, in_dim: int = 10, hidden: int = 20, out_dim: int = 3):
        super().__init__()
        self.query = nn.Linear(in_dim, hidden)
        self.value = nn.Linear(in_dim, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.query(x) + self.value(x)
        return self.head(h)


class _SequenceBackbone(nn.Module):
    """序列 backbone，接受 (batch, seq, d_model) 输入。"""

    def __init__(self, d_model: int = 16, out_dim: int = 3):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (batch, seq, d_model)
        h = self.proj(x)
        return self.head(h.mean(dim=1))


# ============================================================
# TestBuildPEFTSearchSpace
# ============================================================
class TestBuildPEFTSearchSpace:
    """验证 build_peft_search_space 返回正确的 SearchSpace 结构。"""

    def test_returns_search_space_instance(self):
        ss = build_peft_search_space()
        assert isinstance(ss, SearchSpace), f"expected SearchSpace, got {type(ss)}"

    def test_default_has_all_parameters(self):
        """默认含全部 9 个参数。"""
        ss = build_peft_search_space()
        names = [p.name for p in ss.parameters]
        expected = {
            "peft_method", "peft_rank", "peft_alpha", "peft_dropout",
            "peft_target_modules", "learning_rate", "adapter_bottleneck",
            "prompt_length", "freeze_backbone",
        }
        assert set(names) == expected, f"missing: {expected - set(names)}, extra: {set(names) - expected}"
        assert len(names) == 9

    def test_peft_method_is_categorical(self):
        ss = build_peft_search_space()
        param = next(p for p in ss.parameters if p.name == "peft_method")
        assert param.type == "categorical"
        assert param.choices is not None
        assert len(param.choices) > 0

    def test_default_methods_include_all_5(self):
        """默认 choices 含全部 5 种方法。"""
        ss = build_peft_search_space()
        param = next(p for p in ss.parameters if p.name == "peft_method")
        for m in ["lora", "adapter", "prefix_tuning", "prompt_tuning", "full"]:
            assert m in param.choices, f"missing method: {m}"

    def test_include_methods_filter(self):
        """传 include_methods 时 choices 只含指定方法。"""
        ss = build_peft_search_space(include_methods=["lora", "full"])
        param = next(p for p in ss.parameters if p.name == "peft_method")
        assert set(param.choices) == {"lora", "full"}

    def test_peft_rank_choices(self):
        ss = build_peft_search_space()
        param = next(p for p in ss.parameters if p.name == "peft_rank")
        assert param.type == "categorical"
        assert param.choices == [4, 8, 16, 32, 64]

    def test_learning_rate_is_log_float(self):
        ss = build_peft_search_space()
        param = next(p for p in ss.parameters if p.name == "learning_rate")
        assert param.type == "float"
        assert param.log is True, "learning_rate 必须是 log-uniform"
        assert param.low == 1e-5
        assert param.high == 1e-3

    def test_freeze_backbone_choices_are_bool(self):
        """freeze_backbone choices 含 True / False。"""
        ss = build_peft_search_space()
        param = next(p for p in ss.parameters if p.name == "freeze_backbone")
        assert param.type == "categorical"
        assert True in param.choices
        assert False in param.choices


# ============================================================
# TestPEFTBuilder
# ============================================================
class TestPEFTBuilder:
    """验证 PEFTBuilder 各方法的构建行为。"""

    def test_build_lora_returns_peft_model(self):
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": 8,
            "peft_alpha": 1,
            "peft_target_modules": "all",
        })
        assert isinstance(result, PEFTModel)
        assert result.peft_method == "lora"

    def test_build_adapter_returns_peft_model(self):
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "adapter",
            "adapter_bottleneck": 64,
            "peft_target_modules": "all",
        })
        assert isinstance(result, PEFTModel)
        assert result.peft_method == "adapter"

    def test_build_prefix_tuning_returns_peft_model(self):
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "prefix_tuning",
            "prompt_length": 10,
        })
        assert isinstance(result, PEFTModel)
        assert result.peft_method == "prefix_tuning"
        assert result.prefix_layer is not None

    def test_build_prompt_tuning_returns_peft_model(self):
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "prompt_tuning",
            "prompt_length": 10,
        })
        assert isinstance(result, PEFTModel)
        assert result.peft_method == "prompt_tuning"
        assert result.prompt_layer is not None

    def test_build_full_returns_backbone_unchanged(self):
        """full 方法不注入 PEFT 模块，backbone 参数全部可训练。"""
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {"peft_method": "full"})
        assert isinstance(result, PEFTModel)
        assert result.peft_method == "full"
        # 无 PEFT 模块注入
        assert len(result.lora_modules) == 0
        assert len(result.adapter_modules) == 0
        assert result.prefix_layer is None
        assert result.prompt_layer is None
        # backbone 参数全部可训练
        for p in result.backbone.parameters():
            assert p.requires_grad, f"param requires_grad should be True for full method"

    def test_build_unknown_method_raises(self):
        backbone = _SimpleBackbone()
        with pytest.raises(ValueError, match="Unknown peft_method"):
            PEFTBuilder.build(backbone, {"peft_method": "nonexistent_method"})

    def test_lora_freezes_backbone(self):
        """LoRA 注入后原始 backbone 权重 requires_grad=False。"""
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": 8,
            "peft_alpha": 1,
            "peft_target_modules": "all",
            "freeze_backbone": True,
        })
        # 每个 LoRA 层的 original 权重应被冻结
        for lora in result.lora_modules:
            for p in lora.original.parameters():
                assert not p.requires_grad, "LoRA original weight should be frozen"
        # LoRA A/B 应可训练
        for lora in result.lora_modules:
            assert lora.A.requires_grad, "LoRA A should be trainable"
            assert lora.B.requires_grad, "LoRA B should be trainable"

    def test_lora_injects_lora_layers(self):
        """LoRA 注入后 lora_modules 非空。"""
        backbone = _SimpleBackbone()
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": 8,
            "peft_alpha": 1,
            "peft_target_modules": "all",
        })
        assert len(result.lora_modules) > 0, "lora_modules should not be empty"
        # _SimpleBackbone 有 3 个 Linear（query/value/head）
        assert len(result.lora_modules) == 3

    def test_lora_forward_shape_preserved(self):
        """LoRA 注入后输入输出 shape 一致。"""
        backbone = _SimpleBackbone(in_dim=10, out_dim=3)
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": 8,
            "peft_alpha": 1,
            "peft_target_modules": "all",
        })
        x = torch.randn(4, 10)
        out = result(x)
        assert out.shape == (4, 3), f"shape mismatch: {out.shape} vs (4, 3)"

    def test_lora_initial_output_is_zero(self):
        """B 初始化为 zeros，初始时 LoRA 输出为 0（与原始 backbone 输出一致）。"""
        backbone = _SimpleBackbone(in_dim=10, out_dim=3)
        x = torch.randn(4, 10)
        # 原始 backbone 输出
        orig_out = backbone(x).clone().detach()
        # 构建 LoRA（in-place 修改 backbone）
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": 8,
            "peft_alpha": 1,
            "peft_target_modules": "all",
        })
        peft_out = result(x).detach()
        # LoRA 初始输出应等于原始输出（B=zeros，旁路贡献为 0）
        assert torch.allclose(orig_out, peft_out, atol=1e-6), (
            "LoRA initial output should equal original backbone output (B=zeros)"
        )

    def test_lora_alpha_scaling(self):
        """alpha/rank 缩放正确。"""
        backbone = _SimpleBackbone()
        rank, alpha = 8, 2
        result = PEFTBuilder.build(backbone, {
            "peft_method": "lora",
            "peft_rank": rank,
            "peft_alpha": alpha,
            "peft_target_modules": "all",
        })
        for lora in result.lora_modules:
            assert lora.scaling == alpha / rank, (
                f"scaling={lora.scaling}, expected={alpha / rank}"
            )

    def test_adapter_bottleneck_dim(self):
        """Adapter bottleneck 维度正确。"""
        backbone = _SimpleBackbone()
        bottleneck = 64
        result = PEFTBuilder.build(backbone, {
            "peft_method": "adapter",
            "adapter_bottleneck": bottleneck,
            "peft_target_modules": "all",
        })
        assert len(result.adapter_modules) > 0
        for adapter in result.adapter_modules:
            assert adapter.bottleneck == bottleneck
            # 验证 down/up Linear 的形状
            assert adapter.adapter_down.out_features == bottleneck
            assert adapter.adapter_up.in_features == bottleneck

    def test_prefix_tuning_prefix_length(self):
        """Prefix tuning 的 prefix 长度正确。"""
        backbone = _SimpleBackbone()
        prefix_len = 20
        result = PEFTBuilder.build(backbone, {
            "peft_method": "prefix_tuning",
            "prompt_length": prefix_len,
        })
        assert result.prefix_layer is not None
        assert result.prefix_layer.prefix_len == prefix_len
        assert result.prefix_layer.prefix.shape[0] == prefix_len

    def test_prompt_tuning_prompt_length(self):
        """Prompt tuning 的 prompt 长度正确。"""
        backbone = _SimpleBackbone()
        prompt_len = 15
        result = PEFTBuilder.build(backbone, {
            "peft_method": "prompt_tuning",
            "prompt_length": prompt_len,
        })
        assert result.prompt_layer is not None
        assert result.prompt_layer.prompt_len == prompt_len
        assert result.prompt_layer.prompt.shape[0] == prompt_len


# ============================================================
# TestPEFTSearchResult
# ============================================================
class TestPEFTSearchResult:
    """验证 PEFTSearchResult 数据结构。"""

    def test_default_values(self):
        r = PEFTSearchResult(study_id="test_study")
        assert r.study_id == "test_study"
        assert r.n_trials == 0
        assert r.n_completed == 0
        assert r.n_failed == 0
        assert r.trials == []
        assert r.best_params == {}
        assert r.best_value is None

    def test_to_dict_serializable(self):
        """to_dict 返回可 JSON 序列化的 dict。"""
        import json
        r = PEFTSearchResult(
            study_id="s1",
            best_params={"peft_method": "lora"},
            best_value=0.85,
            n_trials=3,
            n_completed=2,
            n_failed=1,
        )
        d = r.to_dict()
        json.dumps(d)  # 不抛异常即可
        assert d["study_id"] == "s1"
        assert d["best_params"] == {"peft_method": "lora"}
        assert d["best_value"] == 0.85


# ============================================================
# TestRunPEFTSearch（轻量单元测试 — grep 实证，不跑真实 pipeline）
# ============================================================
class TestRunPEFTSearch:
    """反假绿：用 grep 实证验证 run_peft_search 通过 SP ask/tell 驱动。"""

    @property
    def _source(self) -> str:
        source_path = (
            Path(__file__).parent.parent
            / "senseframe" / "automl" / "peft_search.py"
        )
        return source_path.read_text(encoding="utf-8")

    def test_source_uses_sp_ask_tell(self):
        """peft_search.py 源码含 sm.ask / sm.tell 调用。"""
        source = self._source
        assert ".ask(" in source, "peft_search.py 未调用 sm.ask()"
        assert ".tell(" in source, "peft_search.py 未调用 sm.tell()"

    def test_source_uses_build_peft_search_space(self):
        """peft_search.py 源码在 run_peft_search 中调用 build_peft_search_space。"""
        source = self._source
        assert "build_peft_search_space(" in source, (
            "peft_search.py 未调用 build_peft_search_space()"
        )

    def test_source_uses_peft_builder(self):
        """peft_search.py 源码含 PEFTBuilder.build 调用。"""
        source = self._source
        assert "PEFTBuilder.build" in source, (
            "peft_search.py 未调用 PEFTBuilder.build()"
        )

    def test_run_peft_search_signature(self):
        """run_peft_search 签名含关键参数。"""
        sig = inspect.signature(run_peft_search)
        params = sig.parameters
        for expected in ["config", "foundation_model", "n_trials", "direction", "metric"]:
            assert expected in params, f"missing param: {expected}"


# ============================================================
# TestSensingFoundationModelProtocol
# ============================================================
class TestSensingFoundationModelProtocol:
    """验证 SensingFoundationModel Protocol 契约。"""

    def test_protocol_is_runtime_checkable(self):
        """Protocol 被 @runtime_checkable 装饰，isinstance 可用。"""
        # 非实现者应不满足
        class _NotAFoundationModel:
            pass
        assert not isinstance(_NotAFoundationModel(), SensingFoundationModel)

    def test_minimal_impl_satisfies_protocol(self):
        """最小实现满足 Protocol。"""

        class _MinimalFoundationModel:
            @property
            def model_id(self) -> str:
                return "test-model"

            @property
            def modality(self) -> str:
                return "csi"

            def pretrain(self, unlabeled_data, config):
                pass

            def encode(self, x):
                return x

            def get_peft_module(self, peft_config):
                return nn.Linear(1, 1)

        instance = _MinimalFoundationModel()
        assert isinstance(instance, SensingFoundationModel), (
            "minimal impl should satisfy SensingFoundationModel Protocol"
        )

    def test_peft_config_has_expected_fields(self):
        """PEFTConfig 含全部期望字段（dataclass fields 反射）。"""
        field_names = {f.name for f in fields(PEFTConfig)}
        expected = {
            "peft_method", "peft_rank", "peft_alpha", "peft_dropout",
            "peft_target_modules", "adapter_bottleneck", "prompt_length",
            "freeze_backbone",
        }
        assert expected.issubset(field_names), (
            f"missing fields: {expected - field_names}"
        )

    def test_pretrain_config_has_expected_fields(self):
        """PretrainConfig 含全部期望字段。"""
        field_names = {f.name for f in fields(PretrainConfig)}
        expected = {"method", "epochs", "batch_size", "learning_rate", "mask_ratio", "augmentations"}
        assert expected.issubset(field_names), (
            f"missing fields: {expected - field_names}"
        )


# ============================================================
# 顶层导出测试
# ============================================================
class TestTopLevelExport:
    """验证 P3 阶段 8 符号顶层可达。"""

    def test_import_from_automl(self):
        """from senseframe.automl import PEFTBuilder 等可达。"""
        from senseframe.automl import (
            PEFTBuilder as PB,
            PEFTModel as PM,
            LoRALayer,
            AdapterLayer,
            PrefixTuningLayer,
            PromptTuningLayer,
            build_peft_search_space,
            run_peft_search as rps,
            PEFTSearchResult,
        )
        assert PB is not None
        assert PM is not None
        assert LoRALayer is not None
        assert AdapterLayer is not None
        assert PrefixTuningLayer is not None
        assert PromptTuningLayer is not None
        assert build_peft_search_space is not None
        assert rps is not None
        assert PEFTSearchResult is not None

    def test_import_from_core(self):
        """from senseframe.core import SensingFoundationModel 等可达。"""
        from senseframe.core import (
            SensingFoundationModel as SFM,
            PretrainConfig,
            PEFTConfig,
        )
        assert SFM is not None
        assert PretrainConfig is not None
        assert PEFTConfig is not None
