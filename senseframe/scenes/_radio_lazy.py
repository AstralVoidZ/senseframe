"""Radio 场景延迟加载代理。

P1.2 落地：对齐 _wifi_csi_lazy.py 的延迟加载模式。
首次调用任何方法时才触发 radio 子包导入和注册。
"""
from typing import Any, Dict

from .base import (
    DatasetBundle,
    DefaultConfig,
    SceneContainer,
    SceneMeta,
    SearchSpace,
    TransformConfig,
)


class LazyRadioContainer(SceneContainer):
    """Radio 场景延迟加载代理。

    首次调用方法时才导入 RadioContainer 并触发注册，
    之后所有调用委托给真实容器。
    """

    def __init__(self):
        self._real: SceneContainer = None

    def _ensure(self) -> SceneContainer:
        if self._real is not None:
            return self._real
        from .radio._register import register
        register()
        from .radio.container import RadioContainer
        self._real = RadioContainer()
        return self._real

    def meta(self) -> SceneMeta:
        return self._ensure().meta()

    def load_dataset(self, dataset_name: str, root: str,
                     learning_mode: str = "supervised",
                     **kwargs) -> DatasetBundle:
        return self._ensure().load_dataset(dataset_name, root, learning_mode, **kwargs)

    def build_model_for_dataset(self, model_id: str, dataset: str,
                                num_classes: int,
                                learning_mode: str = "supervised",
                                **kwargs):
        return self._ensure().build_model_for_dataset(
            model_id, dataset, num_classes, learning_mode, **kwargs)

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        return self._ensure().get_dataset_info(dataset_name, **kwargs)

    def get_default_config(self, model_id: str, dataset_name: str,
                           **kwargs) -> DefaultConfig:
        return self._ensure().get_default_config(model_id, dataset_name, **kwargs)

    def get_search_space(self, model_id: str, dataset_name: str,
                         **kwargs) -> SearchSpace:
        return self._ensure().get_search_space(model_id, dataset_name, **kwargs)

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        return self._ensure().get_transforms(dataset_name, **kwargs)

    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        return self._ensure().get_model_info(model_id, **kwargs)

    def get_normalization_info(self, dataset_name: str, **kwargs):
        return self._ensure().get_normalization_info(dataset_name, **kwargs)
