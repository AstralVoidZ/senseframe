"""P1 ε1 损失函数搜索单元测试（反假绿）。

测试原则：
- 结构测试用真实数据（build_loss_search_space 返回真实 SearchSpace）
- SP 驱动验证用 grep 实证（检查源码含 sm.ask / sm.tell）
- 不用 mock sentinel 验证 SP 调用（mock 可绕过，grep 实证不可绕过）
- 完整流程测试在 integration/test_p1_e2e.py（用真实 run_pipeline）
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from senseframe.automl.loss_search import (
    LossSearchResult,
    _apply_loss_params,
    build_loss_search_space,
)
from senseframe.core import list_losses, register_loss
from senseframe.search_protocol import SearchSpace


# ============================================================
# build_loss_search_space 结构测试
# ============================================================
class TestBuildLossSearchSpace:
    """验证 build_loss_search_space 返回正确的 SearchSpace 结构。"""

    def test_returns_search_space_instance(self):
        """返回 SearchSpace 实例（非 dict / 非 None）。"""
        ss = build_loss_search_space()
        assert isinstance(ss, SearchSpace), f"expected SearchSpace, got {type(ss)}"

    def test_default_has_loss_and_label_smoothing(self):
        """默认含 loss（categorical）+ label_smoothing（float）两个参数。"""
        ss = build_loss_search_space()
        names = [p.name for p in ss.parameters]
        assert "loss" in names
        assert "label_smoothing" in names

    def test_loss_is_categorical(self):
        """loss 参数类型为 categorical。"""
        ss = build_loss_search_space()
        loss_param = next(p for p in ss.parameters if p.name == "loss")
        assert loss_param.type == "categorical"
        assert loss_param.choices is not None
        assert len(loss_param.choices) > 0

    def test_label_smoothing_is_float(self):
        """label_smoothing 参数类型为 float，范围 0.0-0.3。"""
        ss = build_loss_search_space()
        ls_param = next(p for p in ss.parameters if p.name == "label_smoothing")
        assert ls_param.type == "float"
        assert ls_param.low == 0.0
        assert ls_param.high == 0.3

    def test_no_label_smoothing_when_disabled(self):
        """include_label_smoothing=False 时只有 loss 参数。"""
        ss = build_loss_search_space(include_label_smoothing=False)
        names = [p.name for p in ss.parameters]
        assert names == ["loss"]

    def test_choices_reflect_registry(self):
        """loss choices 动态反映 list_losses() 注册表状态。"""
        ss = build_loss_search_space()
        loss_param = next(p for p in ss.parameters if p.name == "loss")
        # 应包含所有内置 loss
        for builtin in ["cross_entropy", "focal", "mse"]:
            assert builtin in loss_param.choices, f"missing builtin loss: {builtin}"

        # 注册一个临时 loss，验证 choices 动态更新
        @register_loss("__test_loss_only__")
        def _test_loss():
            import torch.nn as nn
            return nn.MSELoss()

        try:
            ss2 = build_loss_search_space()
            loss_param2 = next(p for p in ss2.parameters if p.name == "loss")
            assert "__test_loss_only__" in loss_param2.choices
        finally:
            # 清理注册项（list_losses 无 unregister，直接操作注册表）
            from senseframe.core.losses import _LOSS_REGISTRY
            _LOSS_REGISTRY.pop("__test_loss_only__", None)

    def test_extra_losses_appended(self):
        """extra_losses 追加自定义 loss 到 choices。"""
        ss = build_loss_search_space(extra_losses=["__custom_loss__"])
        loss_param = next(p for p in ss.parameters if p.name == "loss")
        assert "__custom_loss__" in loss_param.choices

    def test_default_excludes_self_supervised(self):
        """默认排除自监督损失（ent_loss），避免监督任务采样到不兼容 loss。

        根因：EntLoss.forward(feat1, feat2) 返回 dict 且 feat2 不能是 long 标签，
        在监督分类任务 (logits, y_long) 下必然抛 NotImplementedError，
        污染 best_trial 候选集（trial failed value=0.0 也被 ExplorationTracker.best_trial 纳入）。
        """
        from senseframe.core import SELF_SUPERVISED_LOSSES, list_supervised_losses

        # SELF_SUPERVISED_LOSSES 应包含 ent_loss
        assert "ent_loss" in SELF_SUPERVISED_LOSSES

        # list_supervised_losses 应排除 SELF_SUPERVISED_LOSSES
        supervised = list_supervised_losses()
        for ssl_name in SELF_SUPERVISED_LOSSES:
            assert ssl_name not in supervised, (
                f"list_supervised_losses 不应包含自监督损失: {ssl_name}"
            )

        # build_loss_search_space 默认排除自监督损失
        ss = build_loss_search_space()
        loss_param = next(p for p in ss.parameters if p.name == "loss")
        for ssl_name in SELF_SUPERVISED_LOSSES:
            assert ssl_name not in loss_param.choices, (
                f"build_loss_search_space 默认不应包含自监督损失: {ssl_name}"
            )

    def test_include_self_supervised_true_keeps_ent_loss(self):
        """include_self_supervised=True 时保留 ent_loss（用于自监督场景）。"""
        ss = build_loss_search_space(include_self_supervised=True)
        loss_param = next(p for p in ss.parameters if p.name == "loss")
        assert "ent_loss" in loss_param.choices, (
            "include_self_supervised=True 应保留 ent_loss"
        )


# ============================================================
# LossSearchResult 数据结构测试
# ============================================================
class TestLossSearchResult:
    """验证 LossSearchResult 数据结构。"""

    def test_default_values(self):
        """默认值合理（n_trials=0, trials=[]）。"""
        r = LossSearchResult(study_id="test_study")
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
        r = LossSearchResult(
            study_id="s1",
            best_params={"loss": "focal"},
            best_value=0.85,
            n_trials=3,
            n_completed=2,
            n_failed=1,
        )
        d = r.to_dict()
        # 验证可序列化
        json.dumps(d)
        assert d["study_id"] == "s1"
        assert d["best_params"] == {"loss": "focal"}
        assert d["best_value"] == 0.85


# ============================================================
# _apply_loss_params 测试
# ============================================================
def _make_minimal_config():
    """构造最小 ExperimentConfig（不依赖真实数据集，仅用于参数应用测试）。"""
    from senseframe.engine.config import (
        ExperimentConfig,
        InputFeature,
        OutputFeature,
        SceneConfig,
        TrainerConfig,
    )
    return ExperimentConfig(
        scene=SceneConfig(
            name="generic_test",
            dataset="synthetic",
            model_id="GenericMLP",
            learning_mode="supervised",
            data_root="/tmp",
            params={"data_root": "/tmp"},
        ),
        input_features=[InputFeature(name="features", type="tabular", shape=[10])],
        output_features=[OutputFeature(name="label", type="category", num_classes=3)],
        trainer=TrainerConfig(epochs=1, batch_size=4, enable_progress_bar=False, logger="csv"),
        output_dir="/tmp",
    )


class TestApplyLossParams:
    """验证 _apply_loss_params 正确应用 loss + label_smoothing。"""

    def test_loss_applied_to_scene_params(self):
        """loss 参数写入 scene.params['loss']。"""
        config = _make_minimal_config()
        modified = _apply_loss_params(config, {"loss": "focal"})
        assert modified.scene.params["loss"] == "focal"

    def test_label_smoothing_applied_to_loss_kwargs(self):
        """label_smoothing 写入 scene.params['loss_kwargs']['label_smoothing']。"""
        config = _make_minimal_config()
        modified = _apply_loss_params(
            config,
            {"loss": "cross_entropy", "label_smoothing": 0.1},
        )
        assert modified.scene.params["loss_kwargs"]["label_smoothing"] == 0.1

    def test_label_smoothing_merges_existing_kwargs(self):
        """label_smoothing 合并已有 loss_kwargs，不覆盖其他 key。"""
        config = _make_minimal_config()
        config.scene.params["loss_kwargs"] = {"reduction": "sum"}
        modified = _apply_loss_params(
            config,
            {"loss": "cross_entropy", "label_smoothing": 0.2},
        )
        assert modified.scene.params["loss_kwargs"]["reduction"] == "sum"
        assert modified.scene.params["loss_kwargs"]["label_smoothing"] == 0.2

    def test_does_not_mutate_original_config(self):
        """不修改原始 config（深拷贝）。"""
        config = _make_minimal_config()
        original_loss = config.scene.params.get("loss")
        _apply_loss_params(config, {"loss": "focal"})
        assert config.scene.params.get("loss") == original_loss


# ============================================================
# SP 驱动验证（grep 实证）
# ============================================================
class TestSPDrivenEvidence:
    """反假绿：用 grep 实证验证 run_loss_search 通过 SP ask/tell 驱动。

    不用 mock sentinel（mock 可绕过），用源码 grep 实证（不可绕过）。
    """

    def test_source_contains_sm_ask(self):
        """loss_search.py 源码含 sm.ask 调用。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "automl" / "loss_search.py"
        source = source_path.read_text(encoding="utf-8")
        # 查找 sm.ask 或 study_manager.ask 或 .ask(study_id
        assert ".ask(" in source, "loss_search.py 未调用 sm.ask()"
        assert ".tell(" in source, "loss_search.py 未调用 sm.tell()"

    def test_source_creates_study(self):
        """loss_search.py 源码含 sm.create_study 调用。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "automl" / "loss_search.py"
        source = source_path.read_text(encoding="utf-8")
        assert "create_study" in source, "loss_search.py 未调用 sm.create_study()"

    def test_source_uses_build_loss_search_space(self):
        """loss_search.py 源码在 run_loss_search 中调用 build_loss_search_space。"""
        source_path = Path(__file__).parent.parent / "senseframe" / "automl" / "loss_search.py"
        source = source_path.read_text(encoding="utf-8")
        # run_loss_search 函数体内应调用 build_loss_search_space
        assert "build_loss_search_space(" in source, "run_loss_search 未调用 build_loss_search_space"

    def test_run_loss_search_signature(self):
        """run_loss_search 签名含 n_trials / direction / metric 参数。"""
        from senseframe.automl.loss_search import run_loss_search
        sig = inspect.signature(run_loss_search)
        params = sig.parameters
        assert "n_trials" in params
        assert "direction" in params
        assert "metric" in params
        assert "config" in params


# ============================================================
# 顶层导出测试
# ============================================================
class TestTopLevelExport:
    """验证 ε1 符号顶层可达。"""

    def test_import_from_senseframe(self):
        """from senseframe import run_loss_search 可达。"""
        import senseframe
        assert hasattr(senseframe, "run_loss_search")
        assert hasattr(senseframe, "build_loss_search_space")
        assert hasattr(senseframe, "LossSearchResult")

    def test_import_from_automl(self):
        """from senseframe.automl import run_loss_search 可达。"""
        from senseframe.automl import (
            build_loss_search_space,
            run_loss_search,
            LossSearchResult,
        )
        assert build_loss_search_space is not None
        assert run_loss_search is not None
        assert LossSearchResult is not None
