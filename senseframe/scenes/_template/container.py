"""{SCENE_NAME} 场景容器（自动生成模板）。

实现 SceneContainer 的 4 个必须方法：meta / load_dataset / build_model_for_dataset / get_dataset_info。
"""
from typing import Any, Dict, List, Optional

from ..base import SceneContainer, SceneMeta, TransformConfig


class {SCENE_NAME_CLASS}Container(SceneContainer):
    """{SCENE_NAME} 场景容器。"""

    def meta(self) -> SceneMeta:
        """返回场景元数据。"""
        return SceneMeta(
            name="{SCENE_NAME}",
            supported_tasks=["classification"],
            supported_datasets=["{SCENE_NAME}_dataset"],
            supported_models=["mlp"],
            is_dynamic_dataset=False,
            modality="tabular",  # P5 P1-D：显式声明模态，消除 shape-based fallback
        )

    def load_dataset(self, dataset_name: str, root: str,
                     learning_mode: str = "supervised", **kwargs):
        """加载数据集，返回 DatasetBundle。"""
        # TODO: 实现数据加载逻辑
        raise NotImplementedError(f"load_dataset not implemented for {SCENE_NAME}")

    def build_model_for_dataset(self, model_id: str, dataset: str, num_classes: int,
                                learning_mode: str = "supervised", **kwargs):
        """构建模型。"""
        # TODO: 实现模型构建逻辑
        raise NotImplementedError(f"build_model_for_dataset not implemented for {SCENE_NAME}")

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """返回数据集信息。"""
        return {
            "num_classes": 3,
            "n_features": 10,
        }
