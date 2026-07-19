"""RFC-003 DSP-1：PipelineContext 字段类型协议。

Protocol 是结构子类型（structural subtyping），运行时定义不依赖实际类型。
现有 SceneContainer / nn.Module / pl.LightningDataModule / pl.Trainer 等
自动满足对应 Protocol，无需修改源码。
@runtime_checkable 使 isinstance 检查可用（仅校验方法存在性，不校验签名）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from ....scenes.base import SceneMeta  # noqa: F401


@runtime_checkable
class SceneProtocol(Protocol):
    """场景容器结构契约（RFC-003 DSP-1）。"""

    def meta(self): ...
    def load_dataset(self, dataset_name: str, root: str, learning_mode: str = "supervised", **kwargs): ...
    def build_model_for_dataset(self, model_id: str, dataset: str, num_classes: int, learning_mode: str = "supervised", **kwargs): ...
    def get_dataset_info(self, dataset_name: str, **kwargs): ...
    def get_transforms(self, dataset_name: str, **kwargs): ...
    def get_catalog(self): ...


@runtime_checkable
class ModelProtocol(Protocol):
    """模型结构契约（RFC-003 DSP-1）。"""

    def forward(self, x): ...
    def parameters(self): ...
    def state_dict(self, *args, **kwargs): ...
    def load_state_dict(self, state_dict, *args, **kwargs): ...


@runtime_checkable
class DataModuleProtocol(Protocol):
    """数据模块结构契约（RFC-003 DSP-1）。"""

    def train_dataloader(self): ...
    def val_dataloader(self): ...
    def test_dataloader(self): ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """训练器结构契约（RFC-003 DSP-1）。"""

    def fit(self, model, *args, **kwargs): ...
    def validate(self, model, *args, **kwargs): ...
    def test(self, model, *args, **kwargs): ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """PyTorch Lightning Logger 结构契约（P2.1: 替代 Any）。

    仅声明 SenseFrame 使用的接口，不要求完整 pl.Logger。
    """
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...


@runtime_checkable
class SceneMetaProtocol(Protocol):
    """场景元数据结构契约（RFC-003 DSP-1）。"""

    is_dynamic_dataset: bool
    supported_datasets: List[str]
    supported_models: List[str]
    supported_learning_modes: List[str]


@runtime_checkable
class TaskSpecProtocol(Protocol):
    """任务规格结构契约（RFC-003 DSP-1）。

    effective_loss / effective_metrics 在 TaskSpec 中为 @property，
    Protocol 以数据属性声明即可满足（PEP 544）。
    """

    task_type: str
    effective_loss: str
    effective_metrics: List[str]

    def to_dict(self) -> Dict[str, Any]: ...


@runtime_checkable
class FeatureSpecProtocol(Protocol):
    """特征规格结构契约（RFC-003 DSP-1）。

    feature_names / dtypes 为 RFC-003 DSP-4 计划新增字段，
    在此先行声明契约，DSP-4 实施后 FeatureSpec 自动满足。
    """

    feature_dim: Optional[int]
    feature_names: List[str]
    dtypes: List[str]

    def to_dict(self) -> Dict[str, Any]: ...
