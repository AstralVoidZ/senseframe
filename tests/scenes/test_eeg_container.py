"""EEG 场景容器契约验证（P1.2 落地）。

验证 SceneContainer 抽象在 EEG 模态下的可移植性，
特别是自监督模式下 DatasetBundle.filling_rule 的契约校验：
- 4 抽象方法契约：meta / load_dataset / build_model_for_dataset / get_dataset_info
- SceneMeta 字段完整性（支持 supervised + self_supervised）
- 自监督 filling_rule：unsupervised + supervised_finetune 必填，train forbidden
- 模型工厂绑定（含自监督 EEGLowEncoder）
- CSP / 时频分析 / 通道标准化变换
- HPO 搜索空间

反假绿测试策略：
- grep 实证：源码检查不可绕过
- dataclasses.fields 反射：验证字段存在性
- 真实行为：模型实例化 + forward pass 真实验证，非 mock
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from senseframe.scenes import (
    get_scene, has_scene, list_scenes, activate_lazy_scenes,
    SceneContainer, SceneMeta,
)
from senseframe.scenes.base import DatasetBundle
from senseframe.scenes.eeg import container as eeg_container_mod
from senseframe.scenes.eeg import models as eeg_models_mod
from senseframe.scenes.eeg import datasets as eeg_datasets_mod
from senseframe.scenes.eeg import transforms as eeg_transforms_mod
from senseframe.scenes.eeg.container import EEGContainer


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


# ============================================================
# SceneMeta + 延迟注册
# ============================================================
class TestEEGSceneMeta:
    """EEG 场景元数据验证。"""

    def test_eeg_lazy_declared(self):
        """eeg 应在 _LAZY_SCENES 中声明。"""
        assert has_scene("eeg")

    def test_eeg_meta_fields(self):
        """eeg 元数据应包含完整字段。"""
        scenes = list_scenes()
        assert "eeg" in scenes
        meta = scenes["eeg"]
        assert isinstance(meta, SceneMeta)
        assert meta.name == "eeg"
        assert "classification" in meta.supported_tasks
        assert "self_supervised" in meta.supported_tasks
        assert "EEGNet" in meta.supported_models
        assert "DeepConvNet" in meta.supported_models
        assert "TransformerEEG" in meta.supported_models
        assert "BCI_Competition_IV_2a" in meta.supported_datasets
        assert "PhysioNet_MI" in meta.supported_datasets

    def test_eeg_modality_is_eeg(self):
        """eeg 场景应显式声明 modality='eeg'。"""
        scenes = list_scenes()
        meta = scenes["eeg"]
        assert meta.modality == "eeg"

    def test_eeg_supported_learning_modes(self):
        """eeg 场景应支持 supervised + self_supervised。"""
        scenes = list_scenes()
        meta = scenes["eeg"]
        assert "supervised" in meta.supported_learning_modes
        assert "self_supervised" in meta.supported_learning_modes


# ============================================================
# 4 抽象方法契约
# ============================================================
class TestEEGContainerContract:
    """EEGContainer 4 抽象方法契约验证。"""

    def test_container_is_scene_container(self):
        """EEGContainer 应满足 SceneContainer Protocol。"""
        scene = get_scene("eeg")
        for method in ("meta", "load_dataset", "build_model_for_dataset", "get_dataset_info"):
            assert callable(getattr(scene, method, None)), \
                f"EEGContainer 缺少必需方法: {method}"

    def test_meta_returns_scene_meta(self):
        """meta() 应返回 SceneMeta 实例。"""
        scene = get_scene("eeg")
        meta = scene.meta()
        assert isinstance(meta, SceneMeta)
        assert meta.name == "eeg"

    def test_load_dataset_supervised_returns_bundle(self, tmp_path):
        """supervised 模式 load_dataset 应返回正确的 DatasetBundle。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="supervised",
        )
        assert isinstance(bundle, DatasetBundle)
        assert bundle.train is not None
        assert bundle.test is not None
        assert bundle.learning_mode == "supervised"
        assert bundle.unsupervised is None
        assert bundle.supervised_finetune is None

    def test_load_dataset_self_supervised_returns_bundle(self, tmp_path):
        """self_supervised 模式 load_dataset 应返回正确的 DatasetBundle。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="self_supervised",
        )
        assert isinstance(bundle, DatasetBundle)
        assert bundle.learning_mode == "self_supervised"
        # 自监督契约：unsupervised + supervised_finetune 必填
        assert bundle.unsupervised is not None
        assert bundle.supervised_finetune is not None
        # train 应为 None（forbidden）
        assert bundle.train is None
        # test 必填
        assert bundle.test is not None

    def test_build_model_supervised_returns_nn_module(self):
        """supervised 模式 build_model 应返回 nn.Module。"""
        scene = get_scene("eeg")
        model = scene.build_model_for_dataset(
            "EEGNet", "BCI_Competition_IV_2a", num_classes=4,
            learning_mode="supervised",
        )
        assert isinstance(model, torch.nn.Module)

    def test_build_model_self_supervised_returns_nn_module(self):
        """self_supervised 模式 build_model 应返回自监督模型。"""
        scene = get_scene("eeg")
        model = scene.build_model_for_dataset(
            "EEGLowEncoder", "BCI_Competition_IV_2a", num_classes=None,
            learning_mode="self_supervised",
        )
        assert isinstance(model, torch.nn.Module)

    def test_get_dataset_info_returns_dict(self):
        """get_dataset_info 应返回数据集信息 dict。"""
        scene = get_scene("eeg")
        info = scene.get_dataset_info("BCI_Competition_IV_2a")
        assert isinstance(info, dict)
        assert info["num_classes"] == 4
        assert info["modality"] == "eeg"
        assert info["channels"] == 22


# ============================================================
# 模型行为验证（真实 forward pass）
# ============================================================
class TestEEGModelsForward:
    """EEG 模型真实 forward pass 验证。"""

    @pytest.mark.parametrize("model_id", ["EEGNet", "DeepConvNet", "TransformerEEG"])
    def test_model_forward_pass(self, model_id):
        """每个监督模型应能完成 forward pass 并输出正确形状。"""
        scene = get_scene("eeg")
        info = scene.get_dataset_info("BCI_Competition_IV_2a")
        in_channels = info["input_shape"][0]
        signal_length = info["input_shape"][1]
        num_classes = info["num_classes"]

        model = scene.build_model_for_dataset(
            model_id, "BCI_Competition_IV_2a", num_classes=num_classes,
        )
        # EEG 输入: (B, C, T)
        x = torch.randn(4, in_channels, signal_length)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, num_classes), \
            f"{model_id} forward 输出形状错误: {out.shape} vs expected (4, {num_classes})"

    def test_self_supervised_model_forward(self):
        """自监督模型应能完成 forward pass 输出表示向量。"""
        scene = get_scene("eeg")
        info = scene.get_dataset_info("BCI_Competition_IV_2a")
        in_channels = info["input_shape"][0]
        signal_length = info["input_shape"][1]

        model = scene.build_model_for_dataset(
            "EEGLowEncoder", "BCI_Competition_IV_2a", num_classes=None,
            learning_mode="self_supervised",
        )
        x = torch.randn(4, in_channels, signal_length)
        with torch.no_grad():
            out = model(x)
        # EEGLowEncoder 输出 (B, feature_dim=128)
        assert out.dim() == 2
        assert out.shape[0] == 4

    def test_invalid_supervised_model_in_self_supervised_raises(self):
        """self_supervised 模式下使用监督模型应 raise。"""
        scene = get_scene("eeg")
        with pytest.raises(ValueError, match="Unknown self-supervised"):
            scene.build_model_for_dataset(
                "EEGNet", "BCI_Competition_IV_2a", num_classes=4,
                learning_mode="self_supervised",
            )

    def test_invalid_dataset_raises(self):
        """未知数据集应 raise ValueError。"""
        scene = get_scene("eeg")
        with pytest.raises(ValueError, match="Unknown eeg dataset"):
            scene.get_dataset_info("NonExistent")


# ============================================================
# 变换契约验证
# ============================================================
class TestEEGTransforms:
    """EEG 场景变换配置验证。"""

    def test_get_transforms_default(self):
        """默认 get_transforms 应返回 TransformConfig。"""
        scene = get_scene("eeg")
        cfg = scene.get_transforms("BCI_Competition_IV_2a")
        assert cfg.train_transform is not None
        assert cfg.eval_transform is not None

    def test_transform_pipeline_executes(self):
        """变换 pipeline 应能执行。"""
        scene = get_scene("eeg")
        cfg = scene.get_transforms("BCI_Competition_IV_2a")
        # EEG 数据 (22, 1000)
        x = torch.randn(22, 1000)
        y = torch.tensor(2)
        x_t, y_t = cfg.train_transform(x, y)
        assert x_t.dim() >= 1
        assert y_t == y

    def test_self_supervised_supervised_transform(self):
        """自监督模式下 supervised_transform 应非 None。"""
        scene = get_scene("eeg")
        cfg = scene.get_transforms(
            "BCI_Competition_IV_2a",
            params={"learning_mode": "self_supervised"},
        )
        assert cfg.supervised_transform is not None, \
            "自监督模式下 supervised_transform 应非 None"

    def test_csp_transform(self):
        """应支持 CSP 特征提取。"""
        from senseframe.scenes.eeg.transforms import compose_transforms
        fn = compose_transforms(["csp_features"], n_components=6, n_channels=22)
        x = torch.randn(22, 1000)
        y = torch.tensor(0)
        x_t, _ = fn(x, y)
        # CSP 输出: (6, 1000)
        assert x_t.shape[0] == 6

    def test_normalize_eeg_transform(self):
        """应支持 EEG 通道标准化。"""
        from senseframe.scenes.eeg.transforms import compose_transforms
        fn = compose_transforms(["normalize_eeg"])
        x = torch.randn(22, 1000) * 50 + 10  # 模拟 μV 信号
        y = torch.tensor(0)
        x_t, _ = fn(x, y)
        # 标准化后 mean ≈ 0, std ≈ 1
        assert abs(x_t.mean().item()) < 1e-5
        assert abs(x_t.std().item() - 1.0) < 0.1


# ============================================================
# HPO 搜索空间
# ============================================================
class TestEEGSearchSpace:
    """EEG 场景 HPO 搜索空间验证。"""

    def test_search_space_not_empty(self):
        scene = get_scene("eeg")
        ss = scene.get_search_space("EEGNet", "BCI_Competition_IV_2a")
        assert not ss.is_empty()
        assert "learning_rate" in ss.params
        assert "dropout" in ss.params


# ============================================================
# DatasetBundle filling_rule 契约（核心：自监督模式）
# ============================================================
class TestEEGDatasetBundleFilling:
    """EEG 场景 DatasetBundle filling_rule 契约验证。

    核心验证：自监督模式下 unsupervised + supervised_finetune 必填，
    train forbidden；supervised 模式反之。
    """

    def test_supervised_filling_rule(self):
        """supervised 模式 filling_rule 应正确。"""
        rule = DatasetBundle.filling_rule("supervised")
        assert rule["train"] == "required"
        assert rule["test"] == "required"
        assert rule["unsupervised"] == "forbidden"
        assert rule["supervised_finetune"] == "forbidden"

    def test_self_supervised_filling_rule(self):
        """self_supervised 模式 filling_rule 应正确。"""
        rule = DatasetBundle.filling_rule("self_supervised")
        assert rule["train"] == "forbidden"
        assert rule["test"] == "required"
        assert rule["unsupervised"] == "required"
        assert rule["supervised_finetune"] == "required"

    def test_supervised_bundle_validates(self, tmp_path):
        """supervised 模式 bundle 应通过 validate_filling。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="supervised",
        )
        errors = bundle.validate_filling("supervised")
        assert errors == [], f"supervised bundle validate 失败: {errors}"

    def test_self_supervised_bundle_validates(self, tmp_path):
        """self_supervised 模式 bundle 应通过 validate_filling。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="self_supervised",
        )
        errors = bundle.validate_filling("self_supervised")
        assert errors == [], f"self_supervised bundle validate 失败: {errors}"

    def test_self_supervised_bundle_train_is_none(self, tmp_path):
        """self_supervised 模式 bundle.train 应为 None（forbidden 字段）。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="self_supervised",
        )
        assert bundle.train is None, \
            "self_supervised 模式下 bundle.train 应为 None（forbidden 契约）"

    def test_supervised_bundle_unsupervised_is_none(self, tmp_path):
        """supervised 模式 bundle.unsupervised 应为 None（forbidden 字段）。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="supervised",
        )
        assert bundle.unsupervised is None, \
            "supervised 模式下 bundle.unsupervised 应为 None（forbidden 契约）"

    def test_bundle_describe_correct_mode_supervised(self, tmp_path):
        """supervised bundle.describe() 应返回正确的 learning_mode。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="supervised",
        )
        desc = bundle.describe()
        assert desc["learning_mode"] == "supervised"

    def test_bundle_describe_correct_mode_self_supervised(self, tmp_path):
        """self_supervised bundle.describe() 应返回正确的 learning_mode。"""
        scene = get_scene("eeg")
        bundle = scene.load_dataset(
            "BCI_Competition_IV_2a", str(tmp_path), learning_mode="self_supervised",
        )
        desc = bundle.describe()
        assert desc["learning_mode"] == "self_supervised", \
            f"自监督 bundle.describe 返回错误 learning_mode: {desc['learning_mode']}"


# ============================================================
# grep 实证（防止代码漂移）
# ============================================================
class TestEEGGrepEvidence:
    """EEG 场景源码反射实证。"""

    @pytest.mark.parametrize("desc,check", [
        ("eeg.container module", lambda: importlib.import_module("senseframe.scenes.eeg.container") is not None),
        ("EEGContainer class", lambda: hasattr(eeg_container_mod, "EEGContainer")),
        ("EEGNet", lambda: hasattr(eeg_models_mod, "EEGNet")),
        ("DeepConvNet", lambda: hasattr(eeg_models_mod, "DeepConvNet")),
        ("TransformerEEG", lambda: hasattr(eeg_models_mod, "TransformerEEG")),
        ("EEGLowEncoder", lambda: hasattr(eeg_models_mod, "EEGLowEncoder")),
        ("SELFSUP_MODEL_REGISTRY", lambda: hasattr(eeg_models_mod, "SELFSUP_MODEL_REGISTRY")),
        ("LazyEEGContainer", lambda: hasattr(importlib.import_module("senseframe.scenes._eeg_lazy"), "LazyEEGContainer")),
        ("TRANSFORM_REGISTRY", lambda: hasattr(importlib.import_module("senseframe.scenes.eeg.transforms"), "TRANSFORM_REGISTRY")),
        ("register function", lambda: callable(getattr(importlib.import_module("senseframe.scenes.eeg._register"), "register", None))),
        ("EEGContainer.meta", lambda: callable(getattr(EEGContainer, "meta", None))),
        ("EEGContainer.load_dataset", lambda: callable(getattr(EEGContainer, "load_dataset", None))),
        ("EEGContainer.build_model_for_dataset", lambda: callable(getattr(EEGContainer, "build_model_for_dataset", None))),
        ("EEGContainer.get_dataset_info", lambda: callable(getattr(EEGContainer, "get_dataset_info", None))),
    ])
    def test_module_integrity(self, desc, check):
        """模块 / 类 / 函数存在性检查。"""
        assert check(), f"EEG 场景缺少: {desc}"

    def test_eeg_datasets_defined(self):
        assert hasattr(eeg_datasets_mod, 'DATASET_INFO')
        assert "BCI_Competition_IV_2a" in eeg_datasets_mod.DATASET_INFO
        assert "PhysioNet_MI" in eeg_datasets_mod.DATASET_INFO

    def test_eeg_registered_in_scenes_init(self):
        path = _source_path("scenes/__init__.py")
        assert _grep_source(path, 'declare_lazy_scene("eeg"')  # ARCHITECTURE_TRIPWIRE: 延迟注册是架构契约，反射无法验证注册调用是否写在源码中

    def test_eeg_meta_modality_eeg(self):
        meta = list_scenes()["eeg"]
        assert meta.modality == "eeg", \
            "eeg 场景 modality 应为 'eeg'"

    def test_eeg_meta_self_supervised_mode(self):
        """eeg 应声明 supported_learning_modes 含 self_supervised。"""
        meta = list_scenes()["eeg"]
        assert "self_supervised" in meta.supported_learning_modes

    def test_eeg_container_handles_self_supervised(self):
        """EEGContainer 应处理 self_supervised 模式。"""
        src = inspect.getsource(EEGContainer)
        assert "self_supervised" in src, \
            "EEGContainer 未处理 self_supervised 模式"

    def test_eeg_datasets_supports_self_supervised(self):
        """eeg/datasets.py 应支持 self_supervised 模式加载。"""
        src = inspect.getsource(eeg_datasets_mod)
        assert "self_supervised" in src
