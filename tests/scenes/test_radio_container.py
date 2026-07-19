"""Radio 场景容器契约验证（P1.2 落地）。

验证 SceneContainer 抽象在无线电信号模态下的可移植性：
- 4 抽象方法契约：meta / load_dataset / build_model_for_dataset / get_dataset_info
- SceneMeta 字段完整性
- 延迟注册机制
- 模型工厂绑定
- 数据变换契约
- HPO 搜索空间

反假绿测试策略：
- grep 实证：源码检查不可绕过
- dataclasses.fields 反射：验证字段存在性
- 真实行为：模型实例化 + forward pass 真实验证，非 mock
"""
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Dict

import pytest
import torch

from senseframe.scenes import (
    get_scene, has_scene, list_scenes, activate_lazy_scenes,
    SceneContainer, SceneMeta,
)
from senseframe.scenes.base import DatasetBundle


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


# ============================================================
# SceneMeta + 延迟注册
# ============================================================
class TestRadioSceneMeta:
    """Radio 场景元数据验证。"""

    def test_radio_lazy_declared(self):
        """radio 应在 _LAZY_SCENES 中声明（未激活时也可查询元数据）。"""
        assert has_scene("radio"), "radio 场景应在 _LAZY_SCENES 中声明"

    def test_radio_meta_fields(self):
        """未激活时通过 list_scenes 也应能查到 radio 元数据。"""
        scenes = list_scenes()
        assert "radio" in scenes
        meta = scenes["radio"]
        assert isinstance(meta, SceneMeta)
        assert meta.name == "radio"
        assert "classification" in meta.supported_tasks
        assert "CNN1D" in meta.supported_models
        assert "ResNet1D" in meta.supported_models
        assert "Transformer1D" in meta.supported_models
        assert "RadioML2016A" in meta.supported_datasets
        assert "RadioML2018" in meta.supported_datasets

    def test_radio_modality_is_iq(self):
        """radio 场景应显式声明 modality='iq'。"""
        scenes = list_scenes()
        meta = scenes["radio"]
        assert meta.modality == "iq", \
            f"radio 场景 modality 应为 'iq'，实际 {meta.modality!r}"

    def test_radio_supported_learning_modes(self):
        """radio 场景应仅支持 supervised 模式。"""
        scenes = list_scenes()
        meta = scenes["radio"]
        assert meta.supported_learning_modes == ["supervised"], \
            f"radio 应仅支持 supervised，实际 {meta.supported_learning_modes}"


# ============================================================
# 4 抽象方法契约
# ============================================================
class TestRadioContainerContract:
    """RadioContainer 4 抽象方法契约验证。"""

    def test_container_is_scene_container(self):
        """RadioContainer 应满足 SceneContainer Protocol（duck typing）。"""
        scene = get_scene("radio")
        # duck typing 校验：4 必需方法存在
        for method in ("meta", "load_dataset", "build_model_for_dataset", "get_dataset_info"):
            assert callable(getattr(scene, method, None)), \
                f"RadioContainer 缺少必需方法: {method}"

    def test_meta_returns_scene_meta(self):
        """meta() 应返回 SceneMeta 实例。"""
        scene = get_scene("radio")
        meta = scene.meta()
        assert isinstance(meta, SceneMeta)
        assert meta.name == "radio"

    def test_load_dataset_returns_dataset_bundle(self, tmp_path):
        """load_dataset 应返回 DatasetBundle（supervised 模式 train+test 必填）。"""
        scene = get_scene("radio")
        bundle = scene.load_dataset("RadioML2016A", str(tmp_path), learning_mode="supervised")
        assert isinstance(bundle, DatasetBundle)
        assert bundle.train is not None
        assert bundle.test is not None
        assert bundle.learning_mode == "supervised"
        # supervised 模式下 unsupervised/supervised_finetune 应为 None
        assert bundle.unsupervised is None
        assert bundle.supervised_finetune is None

    def test_build_model_returns_nn_module(self):
        """build_model_for_dataset 应返回 nn.Module 实例。"""
        scene = get_scene("radio")
        model = scene.build_model_for_dataset(
            "CNN1D", "RadioML2016A", num_classes=11,
            learning_mode="supervised",
        )
        assert isinstance(model, torch.nn.Module)

    def test_get_dataset_info_returns_dict(self):
        """get_dataset_info 应返回数据集信息 dict。"""
        scene = get_scene("radio")
        info = scene.get_dataset_info("RadioML2016A")
        assert isinstance(info, dict)
        assert info["num_classes"] == 11
        assert info["modality"] == "iq"
        assert "input_shape" in info
        assert len(info["input_shape"]) == 2  # (2, 128)


# ============================================================
# 模型行为验证（真实 forward pass）
# ============================================================
class TestRadioModelsForward:
    """Radio 模型真实 forward pass 验证。"""

    @pytest.mark.parametrize("model_id", ["CNN1D", "ResNet1D", "Transformer1D"])
    def test_model_forward_pass(self, model_id):
        """每个模型应能完成 forward pass 并输出正确形状。"""
        scene = get_scene("radio")
        info = scene.get_dataset_info("RadioML2016A")
        in_channels = info["input_shape"][0]
        signal_length = info["input_shape"][1]
        num_classes = info["num_classes"]

        model = scene.build_model_for_dataset(
            model_id, "RadioML2016A", num_classes=num_classes,
        )
        # 构造 batch: (B, C, L)
        x = torch.randn(4, in_channels, signal_length)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, num_classes), \
            f"{model_id} forward 输出形状错误: {out.shape} vs expected (4, {num_classes})"

    def test_invalid_model_raises(self):
        """未知模型应 raise ValueError。"""
        scene = get_scene("radio")
        with pytest.raises(ValueError, match="Unknown radio model"):
            scene.build_model_for_dataset(
                "NonExistent", "RadioML2016A", num_classes=11,
            )

    def test_invalid_dataset_raises(self):
        """未知数据集应 raise ValueError。"""
        scene = get_scene("radio")
        with pytest.raises(ValueError, match="Unknown radio dataset"):
            scene.get_dataset_info("NonExistentDataset")

    def test_invalid_learning_mode_raises(self, tmp_path):
        """radio 场景应拒绝 self_supervised 模式。"""
        scene = get_scene("radio")
        with pytest.raises(ValueError, match="不支持 learning_mode"):
            scene.load_dataset("RadioML2016A", str(tmp_path), learning_mode="self_supervised")


# ============================================================
# 变换契约验证
# ============================================================
class TestRadioTransforms:
    """Radio 场景变换配置验证。"""

    def test_get_transforms_returns_config(self):
        """get_transforms 应返回 TransformConfig。"""
        scene = get_scene("radio")
        cfg = scene.get_transforms("RadioML2016A")
        assert cfg.train_transform is not None
        assert cfg.eval_transform is not None

    def test_transform_pipeline_executes(self):
        """变换 pipeline 应能执行：IQ → complex → normalize。"""
        scene = get_scene("radio")
        cfg = scene.get_transforms("RadioML2016A")
        # 模拟 IQ 数据 (2, 128)
        x = torch.randn(2, 128)
        y = torch.tensor(3)
        x_t, y_t = cfg.train_transform(x, y)
        assert x_t.dim() >= 1
        assert y_t == y

    def test_iq_to_spectrogram_transform(self):
        """应支持 IQ → 时频图变换。"""
        from senseframe.scenes.radio.transforms import compose_transforms
        fn = compose_transforms(["iq_to_spectrogram"])
        x = torch.randn(2, 128)
        y = torch.tensor(0)
        x_t, _ = fn(x, y)
        # STFT 输出: (2, F, T_frames)
        assert x_t.dim() == 3
        assert x_t.shape[0] == 2  # 双通道


# ============================================================
# HPO 搜索空间
# ============================================================
class TestRadioSearchSpace:
    """Radio 场景 HPO 搜索空间验证。"""

    def test_search_space_not_empty(self):
        """搜索空间应非空。"""
        scene = get_scene("radio")
        ss = scene.get_search_space("CNN1D", "RadioML2016A")
        assert not ss.is_empty()
        assert "learning_rate" in ss.params
        assert "batch_size" in ss.params
        assert "dropout" in ss.params

    def test_search_space_param_types(self):
        """搜索空间参数类型应正确。"""
        scene = get_scene("radio")
        ss = scene.get_search_space("CNN1D", "RadioML2016A")
        assert ss.params["learning_rate"]["type"] == "float"
        assert ss.params["batch_size"]["type"] == "categorical"
        assert ss.params["dropout"]["type"] == "float"


# ============================================================
# DatasetBundle filling_rule 契约
# ============================================================
class TestRadioDatasetBundleFilling:
    """Radio 场景 DatasetBundle filling_rule 契约验证。"""

    def test_supervised_filling_rule(self):
        """supervised 模式 filling_rule 应正确。"""
        rule = DatasetBundle.filling_rule("supervised")
        assert rule["train"] == "required"
        assert rule["test"] == "required"
        assert rule["unsupervised"] == "forbidden"
        assert rule["supervised_finetune"] == "forbidden"

    def test_bundle_validate_filling_passes(self, tmp_path):
        """supervised 模式 bundle 应通过 validate_filling。"""
        scene = get_scene("radio")
        bundle = scene.load_dataset("RadioML2016A", str(tmp_path), learning_mode="supervised")
        errors = bundle.validate_filling("supervised")
        assert errors == [], f"supervised bundle validate 失败: {errors}"

    def test_bundle_describe_returns_correct_mode(self, tmp_path):
        """bundle.describe() 应返回正确的 learning_mode。"""
        scene = get_scene("radio")
        bundle = scene.load_dataset("RadioML2016A", str(tmp_path), learning_mode="supervised")
        desc = bundle.describe()
        assert desc["learning_mode"] == "supervised"
        assert "train" in desc["filled_fields"]
        assert "test" in desc["filled_fields"]


# ============================================================
# grep 实证（防止代码漂移）
# ============================================================
class TestRadioGrepEvidence:
    """Radio 场景源码 grep 实证。"""

    def test_radio_container_exists(self):
        """radio/container.py 应存在。"""
        path = _source_path("scenes/radio/container.py")
        assert path.exists(), f"radio container 不存在: {path}"

    def test_radio_container_class_defined(self):
        """RadioContainer 类应在 container.py 中定义。"""
        path = _source_path("scenes/radio/container.py")
        assert _grep_source(path, "class RadioContainer"), \
            "radio/container.py 未定义 RadioContainer 类"

    def test_radio_models_defined(self):
        """radio/models.py 应定义 CNN1D / ResNet1D / Transformer1D。"""
        path = _source_path("scenes/radio/models.py")
        assert _grep_source(path, "class CNN1D")
        assert _grep_source(path, "class ResNet1D")
        assert _grep_source(path, "class Transformer1D")

    def test_radio_datasets_defined(self):
        """radio/datasets.py 应定义 DATASET_INFO 含 RadioML2016A 和 RadioML2018。"""
        path = _source_path("scenes/radio/datasets.py")
        assert _grep_source(path, "RadioML2016A")
        assert _grep_source(path, "RadioML2018")

    def test_radio_lazy_proxy_exists(self):
        """_radio_lazy.py 应存在并定义 LazyRadioContainer。"""
        path = _source_path("scenes/_radio_lazy.py")
        assert path.exists()
        assert _grep_source(path, "class LazyRadioContainer")

    def test_radio_registered_in_scenes_init(self):
        """scenes/__init__.py 应通过 declare_lazy_scene 注册 radio。"""
        path = _source_path("scenes/__init__.py")
        assert _grep_source(path, 'declare_lazy_scene("radio"')

    def test_radio_meta_modality_iq(self):
        """scenes/__init__.py 中 radio 的 modality 应为 'iq'。"""
        path = _source_path("scenes/__init__.py")
        content = path.read_text(encoding="utf-8")
        # 简化：检查 radio 块内含 modality="iq"
        assert 'modality="iq"' in content or "modality='iq'" in content, \
            "scenes/__init__.py 未设置 radio 的 modality='iq'"

    def test_radio_transforms_module_exists(self):
        """radio/transforms.py 应存在。"""
        path = _source_path("scenes/radio/transforms.py")
        assert path.exists()
        assert _grep_source(path, "TRANSFORM_REGISTRY")

    def test_radio_register_module_exists(self):
        """radio/_register.py 应存在并定义 register 函数。"""
        path = _source_path("scenes/radio/_register.py")
        assert path.exists()
        assert _grep_source(path, "def register()")

    def test_radio_container_4_abstract_methods(self):
        """RadioContainer 应实现 4 个抽象方法。"""
        path = _source_path("scenes/radio/container.py")
        for method in ("def meta(", "def load_dataset(",
                       "def build_model_for_dataset(", "def get_dataset_info("):
            assert _grep_source(path, method), \
                f"radio/container.py 缺少抽象方法: {method}"
