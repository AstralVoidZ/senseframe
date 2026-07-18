"""ε5 Multi-fidelity 早停测试（P2.1-P2.4）。

反假绿测试策略：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- dataclasses.fields 反射：验证字段存在性，不硬编码字段列表
- 真实行为：ASHA/Hyperband 剪枝逻辑用真实数值验证，非 mock
- Protocol 契约：isinstance + runtime_checkable 验证

覆盖：
- P2.1: Pruner Protocol + 注册表
- P2.2: ASHASampler + HyperbandSampler（双契约 Sampler+Pruner）
- P2.3: IntermediateMetricLogger Callback + PipelineContext.intermediate_values
- P2.4: MethodRunner 早停检查（pruner 注入 + should_prune + PRUNED 状态）
"""
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from senseframe.search_protocol import (
    Pruner,
    Sampler,
    ASHASampler,
    HyperbandSampler,
    SearchSpace,
    ParameterSpec,
    StudyManager,
    register_pruner,
    get_pruner,
    list_pruners,
    list_samplers,
)
from senseframe.engine.runner.orchestrator import IntermediateMetricLogger
from senseframe.engine.runner.pipeline import PipelineContext, _FIELD_FILL_STAGE
from senseframe.experiment.method import MethodRunner
from senseframe.experiment.design import MethodConfig
from senseframe.experiment.types import TrialGroup, TrialStatus


# ============================================================
# 辅助
# ============================================================
def _make_search_space() -> SearchSpace:
    """构造测试用搜索空间。"""
    return SearchSpace(parameters=[
        ParameterSpec(name="lr", type="float", low=0.001, high=0.1, log=True),
        ParameterSpec(name="hidden", type="int", low=16, high=128),
        ParameterSpec(name="opt", type="categorical", choices=["adam", "sgd"]),
    ])


def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


# ============================================================
# P2.1: Pruner Protocol + 注册表
# ============================================================
class TestPrunerProtocol:
    """Pruner Protocol 契约验证。"""

    def test_pruner_is_protocol(self):
        """Pruner 应为 Protocol 子类（@runtime_checkable）。"""
        from typing import Protocol
        assert hasattr(Pruner, "_is_protocol"), "Pruner 应为 Protocol"
        assert Pruner._is_protocol is True

    def test_pruner_has_name_and_should_prune(self):
        """Pruner Protocol 应声明 name 属性和 should_prune 方法。"""
        # Protocol 的方法在 __protocol_attrs__ 中
        attrs = getattr(Pruner, "__protocol_attrs__", set())
        assert "name" in attrs, "Pruner 应声明 name 属性"
        assert "should_prune" in attrs, "Pruner 应声明 should_prune 方法"

    def test_isinstance_valid_pruner(self):
        """满足 Pruner 契约的类应通过 isinstance 检查。"""
        class _ValidPruner:
            name = "valid"
            def should_prune(self, trial_id, intermediate_values, rung):
                return False
        assert isinstance(_ValidPruner(), Pruner)

    def test_isinstance_invalid_pruner(self):
        """不满足 Pruner 契约的类不应通过 isinstance 检查。"""
        class _NotPruner:
            pass
        assert not isinstance(_NotPruner(), Pruner)

    def test_isinstance_missing_should_prune(self):
        """只有 name 没有 should_prune 的类不应通过 isinstance。"""
        class _OnlyName:
            name = "only_name"
        assert not isinstance(_OnlyName(), Pruner)

    def test_should_prune_signature(self):
        """should_prune 方法签名应为 (trial_id, intermediate_values, rung) -> bool。"""
        sig = inspect.signature(Pruner.should_prune)
        params = list(sig.parameters.keys())
        # Protocol 方法的第一个参数是 self
        assert params == ["self", "trial_id", "intermediate_values", "rung"], \
            f"should_prune 参数应为 [self, trial_id, intermediate_values, rung], got {params}"


class TestPrunerRegistry:
    """Pruner 注册表 CRUD。"""

    def test_register_and_get(self):
        """注册 Pruner 后应可通过 get_pruner 获取。"""
        class _TestPruner:
            name = "test_registry"
            def should_prune(self, trial_id, intermediate_values, rung):
                return False
        register_pruner("__test_registry__", _TestPruner)
        assert get_pruner("__test_registry__") is _TestPruner

    def test_get_nonexistent(self):
        """未注册的 Pruner 应返回 None。"""
        assert get_pruner("__nonexistent_pruner__") is None

    def test_list_pruners_contains_builtin(self):
        """list_pruners 应包含内置的 asha 和 hyperband。"""
        pruners = list_pruners()
        assert "asha" in pruners, "asha 应在 pruner 注册表"
        assert "hyperband" in pruners, "hyperband 应在 pruner 注册表"


# ============================================================
# P2.2: ASHASampler
# ============================================================
class TestASHASampler:
    """ASHASampler 行为验证（双契约 Sampler+Pruner）。"""

    def test_satisfies_sampler_protocol(self):
        """ASHASampler 应满足 Sampler Protocol。"""
        asha = ASHASampler()
        assert isinstance(asha, Sampler)

    def test_satisfies_pruner_protocol(self):
        """ASHASampler 应满足 Pruner Protocol。"""
        asha = ASHASampler()
        assert isinstance(asha, Pruner)

    def test_name(self):
        """ASHASampler.name 应为 'asha'。"""
        assert ASHASampler.name == "asha"

    def test_sample_returns_valid_params(self):
        """sample() 应返回搜索空间内的有效参数。"""
        asha = ASHASampler()
        ss = _make_search_space()
        params = asha.sample(ss, [])
        assert "lr" in params
        assert "hidden" in params
        assert "opt" in params
        assert params["opt"] in ["adam", "sgd"]
        assert 0.001 <= params["lr"] <= 0.1
        assert 16 <= params["hidden"] <= 128

    def test_should_prune_data_insufficient(self):
        """数据点不足 eta 时不剪枝。"""
        asha = ASHASampler(eta=3)
        # 1 个 trial，不足 eta=3
        assert asha.should_prune("t1", {0: 0.5}, 0) is False
        # 2 个 trial，仍不足
        assert asha.should_prune("t2", {0: 0.6}, 0) is False

    def test_should_prune_maximize_keeps_high(self):
        """maximize 方向：保留高值，剪枝低值。"""
        asha = ASHASampler(eta=3, direction="maximize")
        asha.should_prune("t1", {0: 0.5}, 0)  # False (不足)
        asha.should_prune("t2", {0: 0.6}, 0)  # False (不足)
        # 3 个 trial, eta=3, n_keep=1, top 1 = 0.7 (t3)
        assert asha.should_prune("t3", {0: 0.7}, 0) is False  # t3 保留（最高）
        # 4 个 trial, n_keep=4//3=1, top 1 = 0.7 (t3)
        assert asha.should_prune("t4", {0: 0.4}, 0) is True   # t4 剪枝（最低）

    def test_should_prune_minimize_keeps_low(self):
        """minimize 方向：保留低值，剪枝高值。"""
        asha = ASHASampler(eta=3, direction="minimize")
        asha.should_prune("t1", {0: 0.5}, 0)
        asha.should_prune("t2", {0: 0.6}, 0)
        # t3=0.4 保留（最低值）
        assert asha.should_prune("t3", {0: 0.4}, 0) is False
        # t4=0.7 剪枝（最高值）
        assert asha.should_prune("t4", {0: 0.7}, 0) is True

    def test_should_prune_rung_not_in_values(self):
        """rung 不在 intermediate_values 中时不剪枝。"""
        asha = ASHASampler(eta=3)
        assert asha.should_prune("t1", {1: 0.5}, 0) is False

    def test_should_prune_no_duplicate_recording(self):
        """同一 trial_id 多次调用不应重复记录。"""
        asha = ASHASampler(eta=3, direction="maximize")
        asha.should_prune("t1", {0: 0.5}, 0)  # 记录 t1
        asha.should_prune("t1", {0: 0.5}, 0)  # 不应重复记录
        # rung 0 应只有 1 个 entry
        assert len(asha._rungs[0]) == 1

    def test_should_prune_multiple_rungs(self):
        """多 rung 独立跟踪。"""
        asha = ASHASampler(eta=3, direction="maximize")
        # rung 0
        asha.should_prune("t1", {0: 0.5, 1: 0.6}, 0)
        asha.should_prune("t2", {0: 0.4, 1: 0.5}, 0)
        asha.should_prune("t3", {0: 0.6, 1: 0.7}, 0)  # 不足 eta，不剪枝
        # rung 1（独立跟踪）
        assert asha.should_prune("t1", {0: 0.5, 1: 0.6}, 1) is False  # 不足
        asha.should_prune("t2", {0: 0.4, 1: 0.5}, 1)
        asha.should_prune("t3", {0: 0.6, 1: 0.7}, 1)  # 仍不足
        # rung 0 和 rung 1 应独立
        assert 0 in asha._rungs
        assert 1 in asha._rungs


# ============================================================
# P2.2: HyperbandSampler
# ============================================================
class TestHyperbandSampler:
    """HyperbandSampler 行为验证（多 bracket）。"""

    def test_satisfies_sampler_protocol(self):
        """HyperbandSampler 应满足 Sampler Protocol。"""
        assert isinstance(HyperbandSampler(), Sampler)

    def test_satisfies_pruner_protocol(self):
        """HyperbandSampler 应满足 Pruner Protocol。"""
        assert isinstance(HyperbandSampler(), Pruner)

    def test_name(self):
        """HyperbandSampler.name 应为 'hyperband'。"""
        assert HyperbandSampler.name == "hyperband"

    def test_n_brackets_computation(self):
        """n_brackets = floor(log_eta(max_resource)) + 1。"""
        # max_resource=27, eta=3 → log3(27)=3, n_brackets=4
        h1 = HyperbandSampler(max_resource=27, eta=3)
        assert h1.n_brackets == 4
        # max_resource=9, eta=3 → log3(9)=2, n_brackets=3
        h2 = HyperbandSampler(max_resource=9, eta=3)
        assert h2.n_brackets == 3
        # max_resource=1, eta=3 → n_brackets=1（边界）
        h3 = HyperbandSampler(max_resource=1, eta=3)
        assert h3.n_brackets == 1

    def test_bracket_assignment_deterministic(self):
        """相同 trial_id 应分配到相同 bracket。"""
        h = HyperbandSampler(max_resource=27, eta=3)
        b1 = h._get_bracket("trial_abc")
        b2 = h._get_bracket("trial_abc")
        assert b1 == b2

    def test_bracket_assignment_in_range(self):
        """bracket index 应在 [0, n_brackets) 范围内。"""
        h = HyperbandSampler(max_resource=27, eta=3)
        for i in range(20):
            idx = h._get_bracket(f"trial_{i}")
            assert 0 <= idx < h.n_brackets

    def test_should_prune_data_insufficient(self):
        """数据不足时不剪枝。"""
        h = HyperbandSampler(max_resource=9, eta=3)
        assert h.should_prune("test_t1", {0: 0.5}, 0) is False

    def test_should_prune_per_bracket_isolation(self):
        """不同 bracket 的 trial 独立比较。"""
        h = HyperbandSampler(max_resource=9, eta=3, direction="maximize")
        # 找到两个不同 bracket 的 trial_ids
        bracket_trials: Dict[int, List[str]] = {}
        for i in range(100):
            tid = f"iso_trial_{i}"
            b = h._get_bracket(tid)
            if b not in bracket_trials:
                bracket_trials[b] = []
            bracket_trials[b].append(tid)
            if len(bracket_trials) >= 2 and all(len(v) >= 4 for v in bracket_trials.values()):
                break

        # 取两个不同 bracket 的 trials
        brackets = list(bracket_trials.keys())[:2]
        b0_trials = bracket_trials[brackets[0]][:4]
        b1_trials = bracket_trials[brackets[1]][:4]

        # 在 bracket 0 中填入 trials
        for i, tid in enumerate(b0_trials):
            h.should_prune(tid, {0: 0.1 * i}, 0)

        # 在 bracket 1 中填入 trials
        for i, tid in enumerate(b1_trials):
            h.should_prune(tid, {0: 0.9 - 0.1 * i}, 0)

        # 两个 bracket 的 rung 0 应有独立的 entries
        assert len(h._brackets[brackets[0]][0]) == 4
        assert len(h._brackets[brackets[1]][0]) == 4


# ============================================================
# P2.2: 注册表交叉验证
# ============================================================
class TestRegistration:
    """ASHA/Hyperband 双注册表验证。"""

    def test_asha_in_both_registries(self):
        """ASHA 应同时在 sampler 和 pruner 注册表。"""
        assert "asha" in list_samplers()
        assert "asha" in list_pruners()

    def test_hyperband_in_both_registries(self):
        """Hyperband 应同时在 sampler 和 pruner 注册表。"""
        assert "hyperband" in list_samplers()
        assert "hyperband" in list_pruners()

    def test_get_pruner_returns_correct_class(self):
        """get_pruner 应返回正确的类。"""
        assert get_pruner("asha") is ASHASampler
        assert get_pruner("hyperband") is HyperbandSampler


# ============================================================
# P2.3: IntermediateMetricLogger Callback
# ============================================================
class TestIntermediateMetricLogger:
    """IntermediateMetricLogger 回调验证。"""

    def test_writes_to_intermediate_values(self):
        """回调应将指标写入 intermediate_values dict（1-indexed epoch key）。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        # 模拟 trainer（P0-3: sanity_checking=False 才会写入）
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_accuracy": 0.85}
        cb.on_validation_epoch_end(trainer, None)
        # P1 修复：1-indexed（current_epoch + 1），与 training_log/CSV epoch 对齐
        assert iv == {1: 0.85}

    def test_writes_multiple_epochs(self):
        """多个 epoch 应写入多个 entry（1-indexed epoch key）。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        for epoch, val in [(0, 0.5), (1, 0.6), (2, 0.7)]:
            trainer = MagicMock()
            trainer.current_epoch = epoch
            trainer.sanity_checking = False
            trainer.callback_metrics = {"val_accuracy": val}
            cb.on_validation_epoch_end(trainer, None)
        # P1 修复：1-indexed（current_epoch + 1），与 training_log/CSV epoch 对齐
        assert iv == {1: 0.5, 2: 0.6, 3: 0.7}

    def test_none_intermediate_values_noop(self):
        """intermediate_values=None 时应 no-op。"""
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=None)
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_accuracy": 0.85}
        # 不应抛异常
        cb.on_validation_epoch_end(trainer, None)

    def test_missing_metric_no_write(self):
        """指标不存在时不写入。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_loss": 0.5}  # 没有 val_accuracy
        cb.on_validation_epoch_end(trainer, None)
        assert iv == {}

    def test_handles_tensor_values(self):
        """应处理 Tensor 类型值（Lightning callback_metrics 返回 Tensor）。1-indexed key。"""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_accuracy": torch.tensor(0.85)}
        cb.on_validation_epoch_end(trainer, None)
        # P1 修复：1-indexed（current_epoch + 1）
        # float32 精度容差（torch.tensor(0.85).item() = 0.8500000238418579）
        assert 1 in iv
        assert iv[1] == pytest.approx(0.85, abs=1e-6)
        assert isinstance(iv[1], float)

    def test_handles_float_values(self):
        """应处理 float 类型值。1-indexed key。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_accuracy": 0.85}
        cb.on_validation_epoch_end(trainer, None)
        # P1 修复：1-indexed（current_epoch + 1）
        assert iv == {1: 0.85}
        assert isinstance(iv[1], float)

    def test_skips_sanity_check(self):
        """P0-3: sanity_check 阶段不应写入 intermediate_values。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = True  # 模拟 sanity_check 阶段
        trainer.callback_metrics = {"val_accuracy": 0.85}
        cb.on_validation_epoch_end(trainer, None)
        assert iv == {}  # 应被跳过

    def test_inactive_in_eval_stage(self):
        """P0-1: stage_eval 中 callback 应被 set_active('eval') 置为 inactive。"""
        iv: Dict[int, float] = {}
        cb = IntermediateMetricLogger(metric="val_accuracy", intermediate_values=iv)
        cb.set_active("eval")  # 模拟 stage_eval 入口
        trainer = MagicMock()
        trainer.current_epoch = 0
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val_accuracy": 0.85}
        cb.on_validation_epoch_end(trainer, None)
        assert iv == {}  # 应被 is_active() 跳过

    def test_frozen_dict_raises_on_write(self):
        """P0-1: FrozenDict 写入应抛 RuntimeError（防御性兜底）。"""
        from senseframe.engine.runner.callbacks import FrozenDict
        fd = FrozenDict({0: 0.85})
        with pytest.raises(RuntimeError, match="frozen after stage_train"):
            fd[1] = 0.90


# ============================================================
# P2.3: PipelineContext intermediate_values 字段
# ============================================================
class TestPipelineContextIntermediateValues:
    """PipelineContext.intermediate_values 字段验证（反射 + grep）。"""

    def test_field_exists_via_reflection(self):
        """PipelineContext 应有 intermediate_values 字段（反射验证，不硬编码）。"""
        field_names = [f.name for f in fields(PipelineContext)]
        assert "intermediate_values" in field_names, \
            "PipelineContext 应有 intermediate_values 字段"

    def test_field_default_empty_dict(self):
        """intermediate_values 默认应为空 dict。"""
        # PipelineContext 需要 config 参数，用 MagicMock
        ctx = PipelineContext(config=MagicMock())
        assert ctx.intermediate_values == {}
        assert isinstance(ctx.intermediate_values, dict)

    def test_field_fill_stage_mapping(self):
        """_FIELD_FILL_STAGE 应映射 intermediate_values → stage_train。"""
        assert _FIELD_FILL_STAGE.get("intermediate_values") == "stage_train", \
            "intermediate_values 应映射到 stage_train"

    def test_field_type_annotation(self):
        """intermediate_values 类型注解应为 Dict[int, float]。"""
        field = next(f for f in fields(PipelineContext) if f.name == "intermediate_values")
        # 类型注解可能为 Dict[int, float] 或 dict
        type_str = str(field.type)
        assert "Dict" in type_str or "dict" in type_str, \
            f"intermediate_values 类型应为 Dict, got {type_str}"


# ============================================================
# P2.4: MethodRunner 早停检查
# ============================================================
class TestMethodRunnerPrunerIntegration:
    """MethodRunner Pruner 集成验证。"""

    def _make_method_config(self):
        """构造测试用 MethodConfig。"""
        from senseframe.engine.config import (
            ExperimentConfig, InputFeature, OutputFeature, SceneConfig, TrainerConfig,
        )
        base_config = ExperimentConfig(
            scene=SceneConfig(name="test", dataset="synthetic", model_id="MLP"),
            input_features=[InputFeature(name="features", type="tabular", shape=[10])],
            output_features=[OutputFeature(name="label", type="category", num_classes=3)],
            trainer=TrainerConfig(epochs=1, batch_size=4, enable_progress_bar=False, logger="csv"),
            output_dir="/tmp/test",
        )
        return MethodConfig(
            name="test_method",
            base_config=base_config,
            search_space=SearchSpace(parameters=[
                ParameterSpec(name="lr", type="float", low=0.001, high=0.1),
            ]),
            metric="val_accuracy",
            direction="maximize",
        )

    def _make_mock_train_output(self, intermediate_values=None):
        """构造 mock TrainOutput。"""
        mock = MagicMock()
        mock.status = "success"
        mock.model_path = "/tmp/model.pt"
        mock.output_dir = "/tmp/output"
        mock.error = None
        mock.error_code = None
        mock.final_eval = {"val_accuracy": 0.8, "val_loss": 0.5}
        # P5 P2-7 阶段2：training 现在是 TrainingSummary dataclass
        from senseframe.schemas import TrainingSummary
        mock.training = TrainingSummary(
            epochs_trained=2,
            early_stopped=False,
            duration_s=10.0,
            intermediate_values=intermediate_values or {},
        )
        return mock

    def test_pruner_none_no_pruning(self):
        """pruner=None 时不应剪枝（向后兼容）。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=None,  # 无 pruner
        )

        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output({0: 0.5, 1: 0.8})
            result = runner.run("synthetic", "MLP", 0)

        assert result.status == TrialStatus.SUCCESS

    def test_pruner_returns_true_trial_pruned(self):
        """pruner 返回 True 时 trial 应标记为 PRUNED。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        # 构造总是剪枝的 pruner
        class _AlwaysPrune:
            name = "always_prune"
            def should_prune(self, trial_id, intermediate_values, rung):
                return True

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_AlwaysPrune(),
        )

        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output({0: 0.5, 1: 0.8})
            result = runner.run("synthetic", "MLP", 0)

        assert result.status == TrialStatus.PRUNED
        # SP trial 状态应为 pruned
        sp_trials = sm.list_trials(study_id)
        assert len(sp_trials) == 1
        assert sp_trials[0].state == "pruned"

    def test_pruner_returns_false_trial_success(self):
        """pruner 返回 False 时 trial 应标记为 SUCCESS。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        class _NeverPrune:
            name = "never_prune"
            def should_prune(self, trial_id, intermediate_values, rung):
                return False

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_NeverPrune(),
        )

        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output({0: 0.5, 1: 0.8})
            result = runner.run("synthetic", "MLP", 0)

        assert result.status == TrialStatus.SUCCESS
        sp_trials = sm.list_trials(study_id)
        assert sp_trials[0].state == "completed"

    def test_pruner_exception_graceful_degradation(self):
        """Pruner 抛异常时应降级为不剪枝。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        class _CrashPruner:
            name = "crash"
            def should_prune(self, trial_id, intermediate_values, rung):
                raise RuntimeError("pruner crashed")

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_CrashPruner(),
        )

        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output({0: 0.5, 1: 0.8})
            result = runner.run("synthetic", "MLP", 0)

        # 降级为 SUCCESS，不剪枝
        assert result.status == TrialStatus.SUCCESS

    def test_intermediate_values_passed_to_sm_tell(self):
        """intermediate_values 应传递给 sm.tell。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        class _NeverPrune:
            name = "never"
            def should_prune(self, trial_id, intermediate_values, rung):
                return False

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_NeverPrune(),
        )

        iv = {0: 0.5, 1: 0.8}
        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output(iv)
            result = runner.run("synthetic", "MLP", 0)

        # SP TrialResult 应包含 intermediate_values
        sp_trial = sm.list_trials(study_id)[0]
        assert sp_trial.intermediate_values == iv

    def test_empty_intermediate_values_no_pruning_check(self):
        """intermediate_values 为空时不调用 pruner。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        call_count = [0]
        class _CountingPruner:
            name = "counting"
            def should_prune(self, trial_id, intermediate_values, rung):
                call_count[0] += 1
                return False

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_CountingPruner(),
        )

        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            # intermediate_values 为空
            mock_pipeline.return_value = self._make_mock_train_output({})
            result = runner.run("synthetic", "MLP", 0)

        # pruner 不应被调用
        assert call_count[0] == 0
        assert result.status == TrialStatus.SUCCESS

    def test_rung_is_max_epoch(self):
        """rung 应为 intermediate_values 的最大 key（最后 epoch）。"""
        config = self._make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )

        captured_rung = [None]
        class _CapturingPruner:
            name = "capturing"
            def should_prune(self, trial_id, intermediate_values, rung):
                captured_rung[0] = rung
                return False

        runner = MethodRunner(
            config=config, study_id=study_id, study_manager=sm,
            pruner=_CapturingPruner(),
        )

        iv = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.9}
        with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = self._make_mock_train_output(iv)
            runner.run("synthetic", "MLP", 0)

        # rung 应为最大 epoch key
        assert captured_rung[0] == 3


# ============================================================
# 反假绿：grep 实证检查（源码不可绕过）
# ============================================================
class TestGrepEvidence:
    """源码 grep 实证：mock 可绕过运行时，但绕不过源码 grep。"""

    def test_search_protocol_has_pruner_protocol(self):
        """search_protocol.py 应定义 Pruner Protocol。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "class Pruner(Protocol)"), \
            "search_protocol.py 应定义 class Pruner(Protocol)"
        assert _grep_source(path, "@runtime_checkable"), \
            "Pruner 应有 @runtime_checkable 装饰器"

    def test_search_protocol_has_pruner_registry(self):
        """search_protocol.py 应有 Pruner 注册表三件套。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "def register_pruner("), "应有 register_pruner"
        assert _grep_source(path, "def get_pruner("), "应有 get_pruner"
        assert _grep_source(path, "def list_pruners("), "应有 list_pruners"

    def test_search_protocol_has_asha_and_hyperband(self):
        """search_protocol.py 应定义 ASHASampler 和 HyperbandSampler。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "class ASHASampler:"), "应有 ASHASampler 类"
        assert _grep_source(path, "class HyperbandSampler:"), "应有 HyperbandSampler 类"
        assert _grep_source(path, 'register_pruner("asha"'), "应注册 asha 为 pruner"
        assert _grep_source(path, 'register_pruner("hyperband"'), "应注册 hyperband 为 pruner"

    def test_method_py_has_pruner_integration(self):
        """method.py 应包含 pruner 集成代码。"""
        path = _source_path("experiment/method.py")
        assert _grep_source(path, "self.pruner"), "MethodRunner 应有 self.pruner 属性"
        assert _grep_source(path, "should_prune("), "应调用 should_prune"
        assert _grep_source(path, "TrialStatus.PRUNED"), "应使用 TrialStatus.PRUNED"
        assert _grep_source(path, 'state="pruned"'), "应 tell state=pruned"
        assert _grep_source(path, "intermediate_values"), "应处理 intermediate_values"

    def test_pipeline_has_intermediate_values_field(self):
        """pipeline.py 应定义 intermediate_values 字段。"""
        path = _source_path("engine/runner/pipeline.py")
        assert _grep_source(path, "intermediate_values: Dict[int, float]"), \
            "PipelineContext 应有 intermediate_values: Dict[int, float] 字段"

    def test_pipeline_has_intermediate_metric_logger(self):
        """pipeline.py 应注入 IntermediateMetricLogger 回调。"""
        path = _source_path("engine/runner/pipeline.py")
        assert _grep_source(path, "IntermediateMetricLogger"), \
            "stage_build 应使用 IntermediateMetricLogger"
        assert _grep_source(path, "intermediate_values=ctx.intermediate_values"), \
            "应将 ctx.intermediate_values 传给回调"

    def test_orchestrator_has_intermediate_metric_logger_class(self):
        """orchestrator.py 应定义 IntermediateMetricLogger 类。"""
        path = _source_path("engine/runner/orchestrator.py")
        assert _grep_source(path, "class IntermediateMetricLogger"), \
            "orchestrator.py 应定义 IntermediateMetricLogger 类"
        assert _grep_source(path, "on_validation_epoch_end"), \
            "IntermediateMetricLogger 应实现 on_validation_epoch_end"

    def test_pipeline_export_intermediate_values_to_train_output(self):
        """pipeline.py 应将 intermediate_values 写入 TrainOutput.training。"""
        path = _source_path("engine/runner/pipeline.py")
        # P5 P2-7 阶段2：构造点改为 validate_training_summary 调用
        assert _grep_source(path, '"intermediate_values": ctx.intermediate_values'), \
            "stage_export 应将 intermediate_values 写入 TrainingSummary 构造"

    def test_field_fill_stage_has_intermediate_values(self):
        """_FIELD_FILL_STAGE 应包含 intermediate_values 映射。"""
        path = _source_path("engine/runner/pipeline.py")
        assert _grep_source(path, '"intermediate_values": "stage_train"'), \
            "_FIELD_FILL_STAGE 应映射 intermediate_values → stage_train"
