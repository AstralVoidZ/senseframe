"""Radio 场景容器：封装 RadioML 数据集 + 1D 信号模型 + IQ 变换。

P1.2 落地：验证 SceneContainer 抽象在新模态（无线电信号）下的可移植性。

实现 4 抽象方法：meta / load_dataset / build_model_for_dataset / get_dataset_info
实现 6 可选方法：get_transforms / get_default_config / get_search_space /
              get_model_info / get_normalization_info / get_task_spec / get_feature_spec
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import (
    DatasetBundle,
    DefaultConfig,
    SceneContainer,
    SceneMeta,
    SearchSpace,
    TransformConfig,
)
from ...common.transforms import ChainedTransform
from .datasets import DATASET_INFO, load_radioml_dataset
from .models import MODEL_REGISTRY


class RadioContainer(SceneContainer):
    """无线电信号场景容器。

    封装：
    - 2 个 RadioML 数据集（RadioML2016A: 11 类, RadioML2018: 24 类）
    - 3 个 1D 信号模型（CNN1D / ResNet1D / Transformer1D）
    - IQ 数据变换（复数谱图 / 时频图 / 标准化）
    - 监督学习模式
    """

    def meta(self) -> SceneMeta:
        """返回场景元数据。"""
        return SceneMeta(
            name="radio",
            supported_tasks=["classification"],
            supported_models=list(MODEL_REGISTRY.keys()),
            supported_datasets=list(DATASET_INFO.keys()),
            requires_custom_dataloader=False,
            supported_learning_modes=["supervised"],
            is_dynamic_dataset=False,
            # P0 修复：显式声明模态，覆盖 profiler shape 启发式
            # IQ (2, 128) 与 image (C, H, W) / csi 在 shape 上易混淆
            modality="iq",
        )

    def load_dataset(self, dataset_name: str, root: str,
                     learning_mode: str = "supervised",
                     **kwargs) -> DatasetBundle:
        """加载 RadioML 数据集。

        Args:
            dataset_name: "RadioML2016A" 或 "RadioML2018"
            root: 数据根目录
            learning_mode: 仅支持 "supervised"
        """
        if learning_mode != "supervised":
            raise ValueError(
                f"Radio 场景不支持 learning_mode='{learning_mode}'，"
                f"仅支持 'supervised'"
            )
        result = load_radioml_dataset(dataset_name, root, learning_mode)
        return DatasetBundle(
            train=result["train"],
            val=result["val"],
            test=result["test"],
            learning_mode="supervised",
        )

    def build_model_for_dataset(self, model_id: str, dataset: str,
                                num_classes: int,
                                learning_mode: str = "supervised",
                                **kwargs) -> nn.Module:
        """构建指定数据集的 Radio 模型。"""
        if model_id not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown radio model: {model_id}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        if dataset not in DATASET_INFO:
            raise ValueError(
                f"Unknown radio dataset: {dataset}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset]
        # num_classes 优先用调用方传入的，否则用数据集默认值
        n_cls = num_classes if num_classes is not None else info["num_classes"]
        in_channels = info["input_shape"][0]
        cls = MODEL_REGISTRY[model_id]
        return cls(in_channels=in_channels, num_classes=n_cls)

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """返回数据集信息。"""
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown radio dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset_name]
        # 转换 tuple 为 list 以便 JSON 序列化
        result = dict(info)
        if "input_shape" in result and isinstance(result["input_shape"], tuple):
            result["input_shape"] = list(result["input_shape"])
        return result

    def get_default_config(self, model_id: str, dataset_name: str,
                           **kwargs) -> DefaultConfig:
        """返回默认训练配置。"""
        return DefaultConfig(
            epochs=50,
            learning_rate=1e-3,
            batch_size=128,
            extra={
                "paradigm": "cnn1d",
                "estimated_vram_mb": 1024,
                "estimated_params_m": 1.5,
                "requires_gpu": False,
            },
        )

    def get_search_space(self, model_id: str, dataset_name: str,
                         **kwargs) -> SearchSpace:
        """Radio 场景的 HPO 搜索空间。"""
        return SearchSpace(params={
            "learning_rate": {
                "type": "float", "low": 1e-5, "high": 1e-2, "log": True,
            },
            "batch_size": {
                "type": "categorical", "values": [64, 128, 256],
            },
            "weight_decay": {
                "type": "float", "low": 1e-6, "high": 1e-3, "log": True,
            },
            "dropout": {
                "type": "float", "low": 0.1, "high": 0.5,
            },
        })

    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """返回模型属性。"""
        info_map = {
            "CNN1D": {
                "paradigm": "cnn1d",
                "estimated_vram_mb": 512,
                "estimated_params_m": 0.5,
                "requires_gpu": False,
                "default_lr": 1e-3,
            },
            "ResNet1D": {
                "paradigm": "resnet1d",
                "estimated_vram_mb": 1024,
                "estimated_params_m": 1.5,
                "requires_gpu": False,
                "default_lr": 1e-3,
            },
            "Transformer1D": {
                "paradigm": "transformer1d",
                "estimated_vram_mb": 2048,
                "estimated_params_m": 3.0,
                "requires_gpu": True,
                "default_lr": 5e-4,
            },
        }
        return info_map.get(model_id, {})

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        """返回数据集的变换配置。

        默认变换：IQ → 复数 magnitude → 标准化
        Agent 可通过 params.transform.pipeline 配置其他原语序列。
        """
        params = kwargs.get("params") or kwargs.get("scene_params") or {}
        transform_cfg_params = params.get("transform", {}) if isinstance(params, dict) else {}

        # 默认 pipeline：iq_to_complex + normalize_iq
        pipeline = transform_cfg_params.get("pipeline", ["iq_to_complex", "normalize_iq"])
        pipeline_params = transform_cfg_params.get("pipeline_params", {})
        transform_seed = transform_cfg_params.get("seed")

        if pipeline:
            from .transforms import compose_transforms
            transform_fn = compose_transforms(
                pipeline, seed=transform_seed, **pipeline_params
            )
            return TransformConfig(
                train_transform=transform_fn,
                eval_transform=transform_fn,
            )
        return TransformConfig()

    def get_normalization_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回归一化信息（IQ 数据按通道标准化）。"""
        if dataset_name not in DATASET_INFO:
            return None
        return {
            "type": "per_sample_standardization",
            "description": "IQ 信号按样本沿时间轴标准化（mean=0, std=1）",
        }
