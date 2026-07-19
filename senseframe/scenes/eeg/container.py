"""EEG 场景容器：BCI Competition / PhysioNet MI + EEGNet/DeepConvNet/TransformerEEG。

P1.2 落地：验证 SceneContainer 抽象在 EEG 模态下的可移植性，
特别是自监督模式下 DatasetBundle.filling_rule 的契约校验。

实现 4 抽象方法：meta / load_dataset / build_model_for_dataset / get_dataset_info
实现 7 可选方法：get_transforms / get_default_config / get_search_space /
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
from .datasets import DATASET_INFO, load_eeg_dataset
from .models import MODEL_REGISTRY, SELFSUP_MODEL_REGISTRY


class EEGContainer(SceneContainer):
    """脑电信号场景容器。

    封装：
    - 2 个 EEG 数据集（BCI Competition IV-2a: 4 类, PhysioNet MI: 2 类）
    - 3 个监督模型（EEGNet / DeepConvNet / TransformerEEG）
    - 1 个自监督模型（EEGLowEncoder）
    - 监督 + 自监督两阶段学习
    - CSP / 时频分析 / 通道标准化变换
    """

    def meta(self) -> SceneMeta:
        """返回场景元数据。"""
        return SceneMeta(
            name="eeg",
            supported_tasks=["classification", "self_supervised"],
            supported_models=list(MODEL_REGISTRY.keys()),
            supported_datasets=list(DATASET_INFO.keys()),
            requires_custom_dataloader=False,
            # 关键：声明支持监督 + 自监督两种学习模式
            supported_learning_modes=["supervised", "self_supervised"],
            is_dynamic_dataset=False,
            # P0 修复：显式声明 EEG 模态
            modality="eeg",
        )

    def load_dataset(self, dataset_name: str, root: str,
                     learning_mode: str = "supervised",
                     **kwargs) -> DatasetBundle:
        """加载 EEG 数据集。

        Args:
            dataset_name: "BCI_Competition_IV_2a" 或 "PhysioNet_MI"
            root: 数据根目录
            learning_mode: "supervised" 或 "self_supervised"
        """
        result = load_eeg_dataset(dataset_name, root, learning_mode)

        if learning_mode == "self_supervised":
            # 自监督模式契约：train=None, unsupervised=required,
            # supervised_finetune=required, test=required
            return DatasetBundle(
                train=None,
                unsupervised=result["unsupervised"],
                supervised_finetune=result["supervised_finetune"],
                val=result["val"],
                test=result["test"],
                learning_mode="self_supervised",
            )
        # supervised 模式契约：train=required, test=required,
        # unsupervised=None, supervised_finetune=None
        return DatasetBundle(
            train=result["train"],
            val=result["val"],
            test=result["test"],
            unsupervised=None,
            supervised_finetune=None,
            learning_mode="supervised",
        )

    def build_model_for_dataset(self, model_id: str, dataset: str,
                                num_classes: int,
                                learning_mode: str = "supervised",
                                **kwargs) -> nn.Module:
        """构建指定数据集的 EEG 模型。"""
        if dataset not in DATASET_INFO:
            raise ValueError(
                f"Unknown eeg dataset: {dataset}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset]
        n_cls = num_classes if num_classes is not None else info["num_classes"]
        in_channels = info["input_shape"][0]

        if learning_mode == "self_supervised":
            if model_id not in SELFSUP_MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown self-supervised eeg model: {model_id}. "
                    f"Available: {list(SELFSUP_MODEL_REGISTRY.keys())}"
                )
            cls = SELFSUP_MODEL_REGISTRY[model_id]
            return cls(in_channels=in_channels)

        if model_id not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown eeg model: {model_id}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        cls = MODEL_REGISTRY[model_id]
        return cls(in_channels=in_channels, num_classes=n_cls)

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """返回数据集信息。"""
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown eeg dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset_name]
        result = dict(info)
        if "input_shape" in result and isinstance(result["input_shape"], tuple):
            result["input_shape"] = list(result["input_shape"])
        return result

    def get_default_config(self, model_id: str, dataset_name: str,
                           **kwargs) -> DefaultConfig:
        """返回默认训练配置。"""
        learning_mode = kwargs.get("learning_mode", "supervised")
        return DefaultConfig(
            epochs=100 if learning_mode == "supervised" else 50,
            learning_rate=1e-3,
            batch_size=64,
            extra={
                "paradigm": "eeg_cnn",
                "estimated_vram_mb": 1024,
                "estimated_params_m": 0.5,
                "requires_gpu": False,
            },
        )

    def get_search_space(self, model_id: str, dataset_name: str,
                         **kwargs) -> SearchSpace:
        """EEG 场景的 HPO 搜索空间。"""
        return SearchSpace(params={
            "learning_rate": {
                "type": "float", "low": 1e-5, "high": 1e-2, "log": True,
            },
            "batch_size": {
                "type": "categorical", "values": [32, 64, 128],
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
            "EEGNet": {
                "paradigm": "eeg_cnn",
                "estimated_vram_mb": 512,
                "estimated_params_m": 0.05,
                "requires_gpu": False,
                "default_lr": 1e-3,
            },
            "DeepConvNet": {
                "paradigm": "eeg_deep_cnn",
                "estimated_vram_mb": 1024,
                "estimated_params_m": 0.5,
                "requires_gpu": False,
                "default_lr": 1e-3,
            },
            "TransformerEEG": {
                "paradigm": "eeg_transformer",
                "estimated_vram_mb": 2048,
                "estimated_params_m": 1.0,
                "requires_gpu": True,
                "default_lr": 5e-4,
            },
            "EEGLowEncoder": {
                "paradigm": "eeg_self_supervised",
                "estimated_vram_mb": 512,
                "estimated_params_m": 0.1,
                "requires_gpu": False,
                "default_lr": 1e-3,
            },
        }
        return info_map.get(model_id, {})

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        """返回数据集的变换配置。

        自监督模式下 supervised_transform 配置较弱的增强，
        预训练 unsupervised 用较强的增强（仅 train_transform）。

        默认 pipeline：normalize_eeg
        Agent 可通过 params.transform.pipeline 配置其他原语。
        """
        params = kwargs.get("params") or kwargs.get("scene_params") or {}
        transform_cfg_params = params.get("transform", {}) if isinstance(params, dict) else {}
        learning_mode = params.get("learning_mode", "supervised") if isinstance(params, dict) else "supervised"

        # 默认 pipeline
        default_pipeline = ["normalize_eeg"]
        pipeline = transform_cfg_params.get("pipeline", default_pipeline)
        pipeline_params = transform_cfg_params.get("pipeline_params", {})
        transform_seed = transform_cfg_params.get("seed")

        if pipeline:
            from .transforms import compose_transforms
            transform_fn = compose_transforms(
                pipeline, seed=transform_seed, **pipeline_params
            )

            if learning_mode == "self_supervised":
                # 自监督：预训练用强增强（train_transform），
                # 微调用弱增强（supervised_transform）
                from .transforms import compose_transforms as _compose
                supervised_pipeline = transform_cfg_params.get(
                    "supervised_pipeline", default_pipeline
                )
                supervised_fn = _compose(
                    supervised_pipeline, seed=transform_seed, **pipeline_params
                ) if supervised_pipeline else transform_fn
                return TransformConfig(
                    train_transform=transform_fn,
                    eval_transform=transform_fn,
                    supervised_transform=supervised_fn,
                )
            return TransformConfig(
                train_transform=transform_fn,
                eval_transform=transform_fn,
            )
        return TransformConfig()

    def get_normalization_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回归一化信息（EEG 信号按通道标准化）。"""
        if dataset_name not in DATASET_INFO:
            return None
        return {
            "type": "per_channel_standardization",
            "description": "EEG 信号按通道沿时间轴标准化（mean=0, std=1）",
        }
