"""
WiFi CSI 场景容器：封装 4 数据集 + 11 模型 + 归一化。

设计原则：
- 委托模式：不移动现有 data.py/registry.py 逻辑，而是委托调用
- 向后兼容：现有 senseframe.data / senseframe.registry 接口保持不变
- 场景容器作为新入口，引擎通过容器访问领域逻辑

Phase 3.1：NORMALIZATION_CONSTANTS 单一来源在 data.py，此处仅做 re-export
"""

from typing import Any, Dict

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
from ...data import CSIDataModule, NORMALIZATION_CONSTANTS  # Phase 3.1：从 data.py 导入
from ...data.normalization import Normalize  # R-fix：统一归一化接口
from ...registry import (
    DATASET_INFO,
    EPOCHS_TABLE,
    MODEL_TABLE,
    get_default_epochs,
    get_model,
    get_model_info,
    get_normalization,  # R-fix：通过策略注册表获取归一化，替代直接读 NORMALIZATION_CONSTANTS
    list_models,
)

# Phase 3.1：NORMALIZATION_CONSTANTS 从 data.py 导入并 re-export
# 保持 `from senseframe.scenes.wifi_csi.container import NORMALIZATION_CONSTANTS` 向后兼容
__all__ = ["WiFiCSIContainer", "NORMALIZATION_CONSTANTS"]


# ============================================================
# Phase 7.3：数据变换函数（从 data.py 的 Dataset.__getitem__ 迁移）
# ============================================================
# R-fix：归一化统一走 registry 策略注册表（Normalize 类委托 get_normalization），
# 消除直接读 NORMALIZATION_CONSTANTS 的重复实现。
# RFC Phase C：变换参数可配置，通过工厂函数生成。
def _make_ntu_fi_transform(stride: int = 4, reshape=(3, 114, 500), norm_name: str = "NTU-Fi_HAR"):
    """构造 NTU-Fi 变换函数（RFC Phase C：参数可配置）。

    Agent 可通过 scene.params.transform 注入 stride/reshape/norm_name，
    消除硬编码封装死点。
    """
    def transform(x, y):
        x = Normalize(norm_name)(x, None)[0]
        x = x[:, ::stride]
        x = x.reshape(*reshape)
        return torch.FloatTensor(x), y
    return transform


def _make_widar_transform(reshape=(22, 20, 20), norm_name: str = "Widar"):
    """构造 Widar 变换函数（RFC Phase C：参数可配置）。"""
    def transform(x, y):
        x = Normalize(norm_name)(x, None)[0]
        x = x.reshape(*reshape)
        return torch.FloatTensor(x), y
    return transform


# 向后兼容：保留原硬编码函数（默认参数）
_ntu_fi_transform = _make_ntu_fi_transform()
_widar_transform = _make_widar_transform()


class WiFiCSIContainer(SceneContainer):
    """
    WiFi CSI 动作识别场景容器。

    封装：
    - 4 个 CSI 数据集（UT_HAR / NTU-Fi_HAR / NTU-Fi-HumanID / Widar）
    - 11 个模型（MLP/LeNet/ResNet/RNN/GRU/LSTM/BiLSTM/CNN+GRU/ViT）
    - 监督 + 自监督两阶段学习
    - 数据集特定归一化
    """

    def meta(self) -> SceneMeta:
        """返回场景元数据。"""
        return SceneMeta(
            name="wifi_csi",
            supported_tasks=["classification", "self_supervised"],
            supported_models=list(MODEL_TABLE.keys()),
            supported_datasets=list(DATASET_INFO.keys()),
            requires_custom_dataloader=True,
            # Phase 6.2：显式声明支持的学习模式
            supported_learning_modes=["supervised", "self_supervised"],
        )

    def load_dataset(self, dataset_name: str, root: str,
                     learning_mode: str = "supervised",
                     **kwargs) -> DatasetBundle:
        """
        加载 CSI 数据集。

        Phase 9.1：统一返回 DatasetBundle，解决监督/自监督模式 arity 不一致。
        - supervised: bundle.train / bundle.test
        - self_supervised: bundle.unsupervised / bundle.supervised_finetune / bundle.test

        内部使用 CSIDataModule，调用 setup() 后提取 dataset。
        """
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )

        dm = CSIDataModule(
            dataset_name=dataset_name,
            root=root,
            learning_mode=learning_mode,
        )
        dm.setup()

        if learning_mode == "self_supervised":
            # 自监督模式：unsupervised_dataset 用于预训练，
            # supervised_dataset 用于微调，test_dataset 用于验证
            return DatasetBundle(
                unsupervised=dm.unsupervised_dataset,
                supervised_finetune=dm.supervised_dataset,
                test=dm.test_dataset,
            )
        return DatasetBundle(
            train=dm.train_dataset,
            test=dm.test_dataset,
        )

    def build_model_for_dataset(self, model_id: str, dataset: str,
                                num_classes: int,
                                learning_mode: str = "supervised",
                                **kwargs) -> nn.Module:
        """
        Phase 9.3：构建指定数据集的 CSI 模型（唯一模型构建入口）。

        CSI 场景的模型工厂按数据集分组，因此需要 dataset 参数选择正确工厂。
        旧的 build_model 抽象方法已删除，统一走此方法。
        """
        return get_model(model_id, dataset, num_classes, learning_mode, scene_name="wifi_csi")

    def normalize(self, x, dataset_name: str):
        """应用数据集特定的归一化。

        R-fix：统一走 registry 策略注册表（get_normalization），
        消除直接读 NORMALIZATION_CONSTANTS 的重复实现。
        """
        return get_normalization(dataset_name).apply(x)

    def get_normalization_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回数据集的归一化信息（供引擎保存 metadata）。

        R-fix：从策略注册表派生，保持单一数据源。
        返回 {"mean":..., "std":...} 格式（向后兼容旧 NORMALIZATION_CONSTANTS 格式）。
        """
        strategy = get_normalization(dataset_name)
        if strategy.is_noop():
            return None
        info = strategy.to_dict()
        # 统一为 {"mean", "std"} 格式，向后兼容
        if info.get("type") == "ZScoreStrategy":
            return {"mean": info["mean"], "std": info["std"]}
        return info

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """返回数据集信息（num_classes, input_shape）。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset_name]
        # 转换 tuple 为 list 以便 JSON 序列化
        result = dict(info)
        if "input_shape" in result and isinstance(result["input_shape"], tuple):
            result["input_shape"] = list(result["input_shape"])
        return result

    def get_default_config(self, model_id: str, dataset_name: str, **kwargs) -> DefaultConfig:
        """返回 (model_id, dataset) 的默认训练配置。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        epochs = get_default_epochs(model_id, dataset_name, scene_name="wifi_csi")
        model_info = MODEL_TABLE.get(model_id, {})
        return DefaultConfig(
            epochs=epochs,
            learning_rate=model_info.get("default_lr", 1e-3),
            batch_size=64,
            extra={
                "paradigm": model_info.get("paradigm"),
                "estimated_vram_mb": model_info.get("estimated_vram_mb"),
                "estimated_params_m": model_info.get("estimated_params_m"),
                "requires_gpu": model_info.get("requires_gpu", False),
            },
        )

    def get_search_space(self, model_id: str, dataset_name: str, **kwargs) -> SearchSpace:
        """CSI 场景的 HPO 搜索空间。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
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
        })

    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """返回模型属性（委托给 registry.get_model_info）。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        return get_model_info(model_id)

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        """
        Phase 7.3：返回数据集的变换配置（RFC Phase C：可配置预处理）。

        RFC-002 阶段 N：支持 pipeline 组合多个信号处理原语。
        Agent 可通过 params.transform.pipeline 配置原语序列，
        通过 params.transform.augment 注入数据增强（仅 train）。

        支持的 params.transform 字段：
        - stride: 采样 stride（默认 4，NTU-Fi 用）
        - reshape: 目标 shape（默认按数据集预设）
        - normalization: 归一化策略名（默认按数据集预设）
        - pipeline: 原语名列表，如 ["hampel", "phase_unwrap", "stft"]，
                    在基础变换之后叠加（见 transforms.compose_transforms）
        - augment: 增强原语名列表，如 ["time_jitter", "freq_masking"]（仅 train）
        - pipeline_params: 原语参数，如 {"hampel": {"window": 7}}

        变换函数签名：fn(x, y) -> (x_tensor, y)
        """
        # RFC Phase C：从 kwargs/params 读取变换配置
        params = kwargs.get("params") or kwargs.get("scene_params") or {}
        transform_cfg_params = params.get("transform", {}) if isinstance(params, dict) else {}

        # 构建基础变换（归一化 + reshape），向后兼容
        if dataset_name in ("NTU-Fi_HAR", "NTU-Fi-HumanID"):
            stride = transform_cfg_params.get("stride", 4)
            reshape = transform_cfg_params.get("reshape", (3, 114, 500))
            norm_name = transform_cfg_params.get("normalization", "NTU-Fi_HAR")
            base_transform = _make_ntu_fi_transform(stride, reshape, norm_name)
        elif dataset_name == "Widar":
            reshape = transform_cfg_params.get("reshape", (22, 20, 20))
            norm_name = transform_cfg_params.get("normalization", "Widar")
            base_transform = _make_widar_transform(reshape, norm_name)
        else:
            # UT_HAR_data：归一化在加载时批量完成，无需逐样本变换
            base_transform = None

        # RFC-002 阶段 N：pipeline 组合信号处理原语
        pipeline = transform_cfg_params.get("pipeline")
        augment = transform_cfg_params.get("augment")
        pipeline_params = transform_cfg_params.get("pipeline_params", {})

        train_transform = base_transform
        eval_transform = base_transform

        if pipeline:
            from .transforms import compose_transforms
            pipeline_fn = compose_transforms(pipeline, **pipeline_params)

            if base_transform is not None:
                def _combined(x, y, _base=base_transform, _pipe=pipeline_fn):
                    x, y = _base(x, y)
                    return _pipe(x, y)
                train_transform = _combined
                eval_transform = _combined
            else:
                train_transform = pipeline_fn
                eval_transform = pipeline_fn

        # RFC-002 阶段 N：数据增强（仅 train，eval 不增强）
        if augment:
            from .transforms import compose_transforms
            augment_fn = compose_transforms(augment, **pipeline_params)

            if train_transform is not None:
                def _train_with_aug(x, y, _base=train_transform, _aug=augment_fn):
                    x, y = _base(x, y)
                    return _aug(x, y)
                train_transform = _train_with_aug
            else:
                train_transform = augment_fn

        return TransformConfig(
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def list_models(self, dataset: str = None, paradigm: str = None,
                    enabled_only: bool = True) -> list:
        """列出模型（委托给 registry.list_models）。"""
        return list_models(dataset=dataset, paradigm=paradigm, enabled_only=enabled_only)

    def get_catalog(self):
        """返回 wifi_csi 场景的技术目录（RFC-002 阶段 U）。"""
        from .catalog import CATALOG
        return CATALOG
