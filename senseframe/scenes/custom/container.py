"""
自定义场景容器：基于 manifest 的零代码数据集接入。

设计目标：
- 用户写一个 manifest JSON/YAML，即可用 senseframe 训练自定义数据集
- 无需继承 SceneContainer，无需改源码
- 复用 GenericMLP 或用户提供的模型

使用方式：
    # 1. 写 manifest.json
    {
        "name": "my_dataset",
        "num_classes": 5,
        "input_shape": [3, 114, 500],
        "file_format": "npy",
        "normalization": "auto",
        "samples": [...]
    }

    # 2. YAML 配置
    scene:
      name: custom
      dataset: my_dataset
      model_id: GenericMLP
      params:
        manifest_path: manifests/my_dataset.json

    # 3. 训练
    python -m senseframe.cli experiment --config configs/my_exp.yaml

兼容性：
- CustomContainer 实现 SceneContainer 全部抽象方法
- Phase 9.1：load_dataset 返回 DatasetBundle（监督模式填 train/test）
- 支持的模型：GenericMLP（内置），或通过 register_model 注册的自定义模型
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import logging
import numpy as np
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
from ..generic.container import GenericMLP
from ...data.manifest import (
    DatasetManifest,
    build_datasets_from_manifest,
    load_manifest,
)

_logger = logging.getLogger(__name__)


# ============================================================
# manifest 缓存（避免重复 IO + 重复计算归一化）
# ============================================================
_MANIFEST_CACHE: Dict[str, DatasetManifest] = {}
_NORMALIZATION_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _load_manifest_cached(path: str) -> DatasetManifest:
    """加载 manifest 并缓存（按绝对路径缓存）。"""
    abs_path = str(Path(path).resolve())
    if abs_path not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[abs_path] = load_manifest(abs_path)
    return _MANIFEST_CACHE[abs_path]


def _get_normalization_cached(manifest: DatasetManifest) -> Optional[Dict[str, Any]]:
    """获取归一化常量并缓存（按 manifest.name 缓存）。"""
    name = manifest.name
    if name not in _NORMALIZATION_CACHE:
        _, _, norm_stats = build_datasets_from_manifest(manifest)
        _NORMALIZATION_CACHE[name] = norm_stats
    return _NORMALIZATION_CACHE[name]


# ============================================================
# 模块级变换函数（可 pickle，支持 DataLoader 多进程）
# ============================================================
def _flatten_transform(x, y):
    """将多维输入展平为 (features,) 以适配 GenericMLP。"""
    return x.flatten(0) if x.dim() == 3 else x.view(-1), y


# ============================================================
# CustomContainer
# ============================================================
class CustomContainer(SceneContainer):
    """
    自定义场景容器：基于 manifest 接入任意数据集。

    通过 scene.params.manifest_path 指定 manifest 文件路径。
    支持的模型：GenericMLP（内置）+ 通过 register_model 注册的模型。

    场景元数据（supported_datasets/supported_models）动态返回：
    - supported_datasets: [manifest.name]（从 manifest 读取）
    - supported_models: ["GenericMLP"] + 已注册的自定义模型
    """

    def __init__(self):
        self._manifest_cache: Dict[str, DatasetManifest] = {}

    def _get_manifest(self, params: Optional[Dict[str, Any]]) -> DatasetManifest:
        """从 scene.params 获取 manifest 实例。"""
        if params is None or "manifest_path" not in params:
            raise ValueError(
                "CustomContainer 需要 scene.params.manifest_path 指向 manifest 文件。"
                "请在 YAML 配置中添加：\n"
                "  scene:\n"
                "    name: custom\n"
                "    params:\n"
                "      manifest_path: path/to/manifest.json"
            )
        return _load_manifest_cached(params["manifest_path"])

    def meta(self) -> SceneMeta:
        """返回场景元数据。

        supported_datasets 动态返回，但 meta() 通常在无 params 上下文时调用，
        故返回空列表，由 runner 在校验时通过 get_dataset_info 二次确认。
        """
        return SceneMeta(
            name="custom",
            supported_tasks=["classification"],
            supported_models=["GenericMLP"],
            supported_datasets=[],  # 动态：由 manifest 决定
            input_shape_hint=None,
            requires_custom_dataloader=False,
            supported_learning_modes=["supervised"],
            is_dynamic_dataset=True,
            modality="tabular",  # P5 P1-D：显式声明模态，消除 shape-based fallback
        )

    def get_catalog(self):
        """P5 P3-3：显式覆写，custom 场景技术目录由 manifest 决定，无静态目录。"""
        return None

    def load_dataset(
        self,
        dataset_name: str,
        root: str,
        learning_mode: str = "supervised",
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DatasetBundle:
        """
        根据 manifest 加载数据集。

        Phase 9.1：统一返回 DatasetBundle（监督模式填 train/test）。
        R4：加载时自动将数据集注册到注册表（幂等），使 get_model() 等查询可用。

        Args:
            dataset_name: 应与 manifest.name 一致（仅做校验）
            root: 未使用（manifest 内含 data_root）
            learning_mode: 仅支持 "supervised"
            params: scene.params，必须含 manifest_path
        """
        if learning_mode != "supervised":
            raise ValueError(
                f"CustomContainer 仅支持 supervised 模式，实际: '{learning_mode}'"
            )

        manifest = self._get_manifest(params)

        if dataset_name != manifest.name:
            raise ValueError(
                f"dataset_name '{dataset_name}' 与 manifest.name '{manifest.name}' 不一致"
            )

        # R4：自动注册数据集到注册表（幂等，已注册则跳过）
        self._ensure_dataset_registered(manifest)

        train_ds, test_ds, _ = build_datasets_from_manifest(manifest)
        return DatasetBundle(train=train_ds, test=test_ds)

    def _ensure_dataset_registered(self, manifest: DatasetManifest) -> None:
        """将 manifest 数据集注册到注册表（幂等）。"""
        from ...registry import is_dataset_registered, register_dataset
        if not is_dataset_registered(manifest.name):
            register_dataset(
                manifest.name,
                num_classes=manifest.num_classes,
                input_shape=tuple(manifest.input_shape),
                file_format=manifest.file_format,
                loader_type="manifest",
            )

    def build_model_for_dataset(
        self,
        model_id: str,
        dataset: str,
        num_classes: int,
        learning_mode: str = "supervised",
        data_root: Optional[str] = None,
        input_dim: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Phase 9.3：构建模型（唯一模型构建入口）。

        支持三种模型来源：
        1. GenericMLP：内置，将 input_shape 展平为 input_dim
        2. 已注册模型：通过 resolve_factory 查找（场景级或全局级）
        3. 未注册模型：raise ValueError
        """
        if learning_mode != "supervised":
            raise ValueError(
                f"CustomContainer 仅支持 supervised 模式，实际: '{learning_mode}'"
            )

        manifest = self._get_manifest(params)

        if input_dim is None:
            input_dim = int(np.prod(manifest.input_shape))

        if model_id == "GenericMLP":
            hidden_dims = kwargs.get("hidden_dims", [128, 64])
            dropout = kwargs.get("dropout", 0.1)
            return GenericMLP(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        # 尝试从注册表查找已注册模型
        from ...registry import is_model_registered, resolve_factory, get_model
        if is_model_registered(model_id):
            try:
                return get_model(model_id, dataset, num_classes, learning_mode, scene_name="custom")
            except (KeyError, ValueError):
                # 注册表中存在但无匹配工厂，尝试通配符
                try:
                    return get_model(model_id, "*", num_classes, learning_mode, scene_name="custom")
                except (KeyError, ValueError):
                    pass

        raise ValueError(
            f"未知模型 '{model_id}'。CustomContainer 支持: ['GenericMLP'] + 已注册模型。"
            f"请通过 register_model + bind_model_factory 注册自定义模型。"
        )

    def normalize(self, x, dataset_name: str):
        """归一化已在 ManifestDataset.__getitem__ 中完成，此处直接返回。"""
        return x

    def get_normalization_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回 manifest 数据集的归一化信息。"""
        params = kwargs.get("params")
        if params is None:
            return None
        try:
            manifest = self._get_manifest(params)
            return _get_normalization_cached(manifest)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            _logger.warning("Failed to load manifest for normalization info: %s", e)
            return None
        except (TypeError, AttributeError, ValueError) as e:
            _logger.error(
                "Failed to get normalization info from manifest: %s", e, exc_info=True
            )
            return None

    def get_manifest_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回 manifest 信息（供引擎保存 metadata）。"""
        params = kwargs.get("params")
        if params is None:
            return None
        try:
            manifest = self._get_manifest(params)
            return {
                "name": manifest.name,
                "file_format": manifest.file_format,
                "mat_key": manifest.mat_key,
                "input_shape": list(manifest.input_shape),
                "num_classes": manifest.num_classes,
            }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            _logger.warning("Failed to load manifest for manifest info: %s", e)
            return None
        except (TypeError, AttributeError, ValueError) as e:
            _logger.error(
                "Failed to get manifest info: %s", e, exc_info=True
            )
            return None

    def get_dataset_info(
        self,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """返回数据集信息（从 manifest 读取）。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        manifest = self._get_manifest(params)
        if dataset_name != manifest.name:
            raise ValueError(
                f"dataset_name '{dataset_name}' 与 manifest.name '{manifest.name}' 不一致"
            )
        return {
            "num_classes": manifest.num_classes,
            "input_shape": tuple(manifest.input_shape),
            "classes": manifest.num_classes,
            "n_features": int(np.prod(manifest.input_shape)),
            "n_samples": len(manifest.samples),
            "file_format": manifest.file_format,
            "label_map": manifest.label_map,
        }

    def get_default_config(
        self,
        model_id: str,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> DefaultConfig:
        """返回默认训练配置。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        return DefaultConfig(
            epochs=50,
            learning_rate=1e-3,
            batch_size=32,
            extra={
                "paradigm": "traditional_ml",
                "estimated_vram_mb": 256,
                "estimated_params_m": 0.1,
                "requires_gpu": False,
            },
        )

    def get_search_space(
        self,
        model_id: str,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SearchSpace:
        """HPO 搜索空间。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        return SearchSpace(params={
            "learning_rate": {
                "type": "float", "low": 1e-5, "high": 1e-1, "log": True,
            },
            "batch_size": {
                "type": "categorical", "values": [16, 32, 64, 128],
            },
            "weight_decay": {
                "type": "float", "low": 1e-6, "high": 1e-2, "log": True,
            },
        })

    def get_transforms(
        self,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TransformConfig:
        """
        返回变换配置。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。

        ManifestDataset 已在 __getitem__ 中完成归一化。
        此处添加 flatten 变换，将 input_shape 展平为 (features,) 以适配 GenericMLP。
        若用户注册了支持多维输入的自定义模型，可覆写此方法。
        """
        manifest = self._get_manifest(params)
        if len(manifest.input_shape) > 1:
            # 多维输入需 flatten 给 GenericMLP（使用模块级函数以支持 pickle）
            return TransformConfig(
                train_transform=_flatten_transform,
                eval_transform=_flatten_transform,
            )
        return TransformConfig()

    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """返回模型信息。

        Phase 9.2：通过 **kwargs 接收上下文，符合 LSP。
        """
        if model_id == "GenericMLP":
            return {
                "id": 100,
                "paradigm": "traditional_ml",
                "enabled": True,
                "requires_gpu": False,
                "estimated_vram_mb": 256,
                "estimated_params_m": 0.1,
                "default_lr": 1e-3,
            }
        return {}


__all__ = ["CustomContainer"]
