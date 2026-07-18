"""
场景容器抽象基类：定义场景必须实现的接口契约。

设计理念：
- 将领域逻辑（数据加载/模型构建/归一化）与通用训练流程解耦
- 引擎通过 SceneContainer 接口与场景交互，无需感知具体领域
- 新增场景只需实现此接口，无需修改引擎代码

参考：Ludwig 的 input_features/output_features 声明式配置理念，
但保留 senseframe 的 Lightning 训练核心，避免引入重型依赖。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from senseframe.core.params import SceneParams


@dataclass
class SceneMeta:
    """场景元数据：供引擎查询场景能力。

    Phase 6.2：新增 supported_learning_modes 字段，显式声明场景支持的
    学习模式（supervised/self_supervised），供 Agent 程序化查询。
    Phase R-fix：新增 is_dynamic_dataset 标志，标记数据集在运行时动态决定
    （如 custom/generic 场景），引擎跳过静态 dataset 校验。
    P0 修复：新增 modality 字段，场景显式声明数据模态（csi/image/text/tabular/sequence/audio），
    覆盖 profiler 的 shape 启发式（CSI (1,250,90) 与 image (1,H,W) 在 shape 上不可区分）。
    """
    name: str                              # 场景标识，如 "wifi_csi"
    supported_tasks: List[str]             # ["classification", "self_supervised"]
    supported_models: List[str]            # ["MLP", "LeNet", ...]
    supported_datasets: List[str]          # ["UT_HAR_data", ...]
    input_shape_hint: Optional[List[int]] = None
    requires_custom_dataloader: bool = True
    # Phase 6.2：显式声明支持的学习模式（与 supported_tasks 正交：
    # tasks 描述 ML 任务类型如 classification/regression，
    # learning_modes 描述训练范式如 supervised/self_supervised）
    supported_learning_modes: List[str] = field(default_factory=lambda: ["supervised"])
    # R-fix：动态数据集场景标志（custom/generic），引擎据此跳过静态 dataset 校验
    is_dynamic_dataset: bool = False
    # P0 修复：场景显式声明数据模态，覆盖 profiler shape 启发式
    # profiler 遇到 "unknown" 会 raise ValueError；所有场景容器必须在 meta() 中显式声明
    modality: str = "unknown"


@dataclass
class DefaultConfig:
    """场景返回的默认训练配置。"""
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 64
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchSpace:
    """HPO 搜索空间定义（供 Optuna 集成使用）。

    每个参数的 spec 格式：
    - float: {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
    - int:   {"type": "int", "low": 16, "high": 256}
    - categorical: {"type": "categorical", "values": [32, 64, 128]}
    """
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.params) == 0


@dataclass
class TransformConfig:
    """Phase 2.1a：数据变换配置。

    场景容器通过 get_transforms 返回此配置，DataModule 在 __getitem__ 后应用。
    仅支持 sample-level（样本级）变换，在 Dataset.__getitem__ 返回 (x, y) 后、
    collate 前应用。batch 级变换（如 mixup / batch 归一化）不在本配置支持范围内
    ——P1 清理：原 batch_transform 字段全代码库无消费者（DataModule / collate 均未读取），
    属未实现的预留能力，已删除以避免误导调用方。如需 batch 级增强，应通过
    GenericDataModule.collate_fn 注入。

    - train_transform: 训练阶段变换（含数据增强），签名 fn(x, y) -> (x, y)
    - eval_transform:  验证/测试阶段变换（无增强，仅预处理），签名 fn(x, y) -> (x, y)
    - supervised_transform: Phase 2 微调专用变换（自监督模式下 supervised_dataset 使用）。
      None 时 DataModule 回退到 train_transform（向后兼容）；显式设置时，允许微调阶段
      使用与预训练不同的增强强度（例如更弱的增强或无增强）。P2-2 上策：消除
      supervised_dataset 与 train_transform 的隐式耦合，让 transform 归属由数据用途决定。
    None 表示该阶段无变换（使用 Dataset 原始返回）。
    """
    train_transform: Optional[Callable] = None
    eval_transform: Optional[Callable] = None
    supervised_transform: Optional[Callable] = None


@dataclass
class DatasetBundle:
    """Phase 9.1：统一 load_dataset 返回值。

    解决 WiFiCSI 监督/自监督模式返回元组 arity 不一致（2 vs 3）的 LSP 违规。
    所有场景的 load_dataset 统一返回此数据类，调用方按属性访问。

    向后兼容：提供 to_tuple() 方法，按学习模式返回对应元组。

    根因修复（数据结构契约对齐）：保留 `supervised_finetune`（自监督专用）的同时
    显式声明 `train`（监督专用）字段。stage_build 在监督路径访问 `bundle.train`，
    自监督路径访问 `bundle.supervised_finetune`；两个字段通过 filling_rule 互斥
    （supervised 模式 train=required/supervised_finetune=forbidden，反之亦然），
    既满足 stage_build 的字段期望，又通过 validate_filling 强制契约一致性。

    P0-B 修复（2026-07-18）：新增 `learning_mode` 字段，由 scene container 在
    `load_dataset` 时显式传入。旧实现 `describe(learning_mode="supervised")` 默认
    参数导致自监督模式下 `bundle.describe()` 误报 `learning_mode='supervised'`。
    现在学习模式作为一等字段随 bundle 携带，describe() 在未传参时优先读
    `self.learning_mode`，再退化为从 filled_fields 推断，确保调用方不传参也能
    得到正确结果。
    """
    train: Optional[Dataset] = None
    test: Optional[Dataset] = None
    val: Optional[Dataset] = None
    # 自监督模式专用
    unsupervised: Optional[Dataset] = None
    supervised_finetune: Optional[Dataset] = None
    # P0-B：学习模式作为一等字段随 bundle 携带，由 scene container 显式传入。
    # 默认 "supervised" 保持向后兼容（旧调用方不传时仍为监督模式）。
    learning_mode: str = "supervised"

    def to_tuple(self, learning_mode: Optional[str] = None) -> Tuple:
        """
        向后兼容：按学习模式返回元组。

        - supervised: (train, test)
        - self_supervised: (unsupervised, supervised_finetune, test)

        Args:
            learning_mode: 显式覆盖；None 时读 self.learning_mode（P0-B 修复）
        """
        mode = learning_mode if learning_mode is not None else self.learning_mode
        if mode == "self_supervised":
            return (self.unsupervised, self.supervised_finetune, self.test)
        return (self.train, self.test)

    @classmethod
    def filling_rule(cls, learning_mode: str) -> Dict[str, str]:
        """返回填充规则表（RFC-003 DSP-2）。

        返回每个字段的填充状态：required / forbidden / optional
        """
        if learning_mode == "self_supervised":
            return {
                "train": "forbidden",
                "test": "required",
                "val": "optional",
                "unsupervised": "required",
                "supervised_finetune": "required",
            }
        # supervised（默认）
        return {
            "train": "required",
            "test": "required",
            "val": "optional",
            "unsupervised": "forbidden",
            "supervised_finetune": "forbidden",
        }

    def filled_fields(self) -> List[str]:
        """返回当前非 None 的字段名列表（RFC-003 DSP-2）。"""
        return [k for k in ["train", "test", "val", "unsupervised", "supervised_finetune"]
                if getattr(self, k) is not None]

    def validate_filling(self, learning_mode: str) -> List[str]:
        """校验填充是否符合契约（RFC-003 DSP-2）。

        Returns:
            错误列表，空列表表示通过
        """
        errors = []
        rule = self.filling_rule(learning_mode)
        for field_name, status in rule.items():
            value = getattr(self, field_name)
            if status == "required" and value is None:
                errors.append(f"Field '{field_name}' is required for learning_mode='{learning_mode}' but is None")
            elif status == "forbidden" and value is not None:
                errors.append(f"Field '{field_name}' is forbidden for learning_mode='{learning_mode}' but is not None")
        return errors

    def _infer_learning_mode(self) -> str:
        """从 filled_fields 推断 learning_mode（describe 兜底）。

        - unsupervised 或 supervised_finetune 非 None → "self_supervised"
        - 否则 → "supervised"
        """
        if self.unsupervised is not None or self.supervised_finetune is not None:
            return "self_supervised"
        return "supervised"

    @staticmethod
    def schema() -> Dict[str, Any]:
        """返回字段结构 + 填充规则表（RFC-003 DSP-2）。"""
        return {
            "schema_version": "1.1.0",  # P0-B: 新增 learning_mode 字段
            "fields": [
                {"name": "train", "type": "Optional[Dataset]"},
                {"name": "test", "type": "Optional[Dataset]"},
                {"name": "val", "type": "Optional[Dataset]"},
                {"name": "unsupervised", "type": "Optional[Dataset]"},
                {"name": "supervised_finetune", "type": "Optional[Dataset]"},
                {"name": "learning_mode", "type": "str", "default": "supervised"},
            ],
            "filling_rules": {
                "supervised": DatasetBundle.filling_rule("supervised"),
                "self_supervised": DatasetBundle.filling_rule("self_supervised"),
            },
        }

    def describe(self, learning_mode: Optional[str] = None) -> Dict[str, Any]:
        """返回运行时状态（RFC-003 DSP-2）。

        P0-B 修复：learning_mode 形参默认改为 None，触发兜底逻辑：
        1. 显式传参 → 用传入值
        2. None + self.learning_mode 已由 container 设置 → 用 self.learning_mode
        3. None + self.learning_mode 是默认值 "supervised" 但 filled_fields
           显示自监督填充 → 从 filled_fields 推断（防御性兜底）

        这样调用方不传参也能得到正确结果，消除旧实现"默认 supervised 导致
        自监督模式误报"的 bug。
        """
        if learning_mode is None:
            # 优先读 self.learning_mode（container 显式传入的值）
            # 若 self.learning_mode 仍是默认 "supervised" 但 filled_fields
            # 显示自监督填充，则从 filled_fields 推断（防御性兜底）
            if self.learning_mode == "supervised":
                inferred = self._infer_learning_mode()
                if inferred != self.learning_mode:
                    # filled_fields 显示自监督但 learning_mode 字段未设置，
                    # 用推断值覆盖（兼容未显式传 learning_mode 的旧 container）
                    effective_mode = inferred
                else:
                    effective_mode = self.learning_mode
            else:
                effective_mode = self.learning_mode
        else:
            effective_mode = learning_mode
        return {
            "filled_fields": self.filled_fields(),
            "learning_mode": effective_mode,
            "validation_errors": self.validate_filling(effective_mode),
        }


class SceneContainer(ABC):
    """
    场景容器：封装特定领域的数据加载、模型构建、归一化等逻辑。

    引擎通过此接口与场景交互，实现领域逻辑与通用训练流程的解耦。
    新增场景（如通用表格、图像、时序）只需继承此类并实现抽象方法。

    Phase 9 接口修正：
    - load_dataset 统一返回 DatasetBundle（解决 arity 不一致）
    - 删除 build_model 抽象方法，统一用 build_model_for_dataset
    - 精简抽象方法：6 必需 → 4 必需（meta/load_dataset/build_model_for_dataset/get_dataset_info）
    - normalize 降级为可选（已被 TransformConfig 取代）
    - get_default_config 降级为可选（提供默认实现）
    """

    @abstractmethod
    def meta(self) -> SceneMeta:
        """返回场景元数据。"""

    @abstractmethod
    def load_dataset(
        self,
        dataset_name: str,
        root: str,
        learning_mode: str = "supervised",
        **kwargs,
    ) -> DatasetBundle:
        """
        加载数据集，返回 DatasetBundle。

        Phase 9.1：统一返回 DatasetBundle，解决 arity 不一致。
        向后兼容：调用方可通过 bundle.to_tuple(learning_mode) 获取元组。

        Args:
            dataset_name: 数据集名称
            root: 数据根目录
            learning_mode: "supervised" 或 "self_supervised"
            **kwargs: 场景特定参数（如 params 上下文）

        Returns:
            DatasetBundle: 含 train/test/unsupervised/supervised_finetune
        """

    @abstractmethod
    def build_model_for_dataset(
        self,
        model_id: str,
        dataset: str,
        num_classes: int,
        learning_mode: str = "supervised",
        **kwargs,
    ) -> nn.Module:
        """
        Phase 9.3：构建指定数据集的模型实例（含 dataset 上下文）。

        Phase 9.3 起此方法为唯一抽象的模型构建入口，
        旧的 build_model 抽象方法已删除。

        Args:
            model_id: 模型 ID
            dataset: 数据集名（用于选择正确的模型工厂）
            num_classes: 类别数
            learning_mode: "supervised" 或 "self_supervised"
            **kwargs: 场景特定参数（如 data_root、input_dim、params）

        Returns:
            nn.Module 模型实例
        """

    @abstractmethod
    def get_dataset_info(
        self,
        dataset_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """返回数据集信息（num_classes, input_shape 等）。

        Phase 9.2：子类可通过 kwargs 接收 params 等上下文，
        不再偷偷加参数，符合 LSP。
        """

    def normalize(self, x, dataset_name: str):
        """
        Phase 9.4：降级为可选方法（已被 TransformConfig 取代）。

        默认实现直接返回 x（无归一化）。
        旧场景（WiFi CSI）保留实现以向后兼容。
        """
        return x

    def get_default_config(
        self,
        model_id: str,
        dataset_name: str,
        **kwargs,
    ) -> DefaultConfig:
        """
        Phase 9.4：降级为可选方法，提供默认实现。

        子类可覆盖以提供场景特定的默认配置。
        """
        return DefaultConfig()

    def get_search_space(
        self,
        model_id: str,
        dataset_name: str,
        **kwargs,
    ) -> SearchSpace:
        """
        返回 HPO 搜索空间（可选实现）。

        默认返回空空间，表示不支持 HPO。
        子类可覆盖此方法以提供场景特定的搜索空间。
        """
        return SearchSpace()

    def get_transforms(
        self,
        dataset_name: str,
        **kwargs,
    ) -> TransformConfig:
        """
        Phase 2.1a：返回数据集的变换配置（可选实现）。

        默认返回空 TransformConfig（无变换），子类可覆盖以提供
        场景特定的数据预处理与增强逻辑。

        变换函数签名：fn(x: Tensor, y: Any) -> (x, y)
        DataModule 在 Dataset.__getitem__ 返回原始 (x, y) 后应用。

        Args:
            dataset_name: 数据集名
            **kwargs: 场景特定参数（如 params 上下文）

        Returns:
            TransformConfig: 含 train_transform / eval_transform
        """
        return TransformConfig()

    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """
        返回模型信息（可选实现）。

        默认返回空字典，子类可覆盖以提供模型属性
       （如 estimated_vram_mb, paradigm 等）。
        """
        return {}

    def get_feature_spec(
        self,
        dataset_name: str,
        **kwargs,
    ):
        """
        Phase 11.3：返回数据集的特征规格（可选实现）。

        默认从 get_dataset_info() 派生 FeatureSpec。子类可覆盖以提供
        场景特定的特征规格（如额外 metadata、C 维数、序列长度等）。
        """
        from ..core.features import FeatureSpec
        info = self.get_dataset_info(dataset_name, **kwargs)
        return FeatureSpec.from_dataset_info(info, modality=self.meta().modality)

    def get_task_spec(
        self,
        dataset_name: str,
        model_id: str = "",
        **kwargs,
    ):
        """
        Phase 11.1：返回任务规格（可选实现）。

        默认从 dataset_info 派生分类 TaskSpec(num_classes=...)。
        子类可覆盖以支持回归/检测/分割等任务。
        """
        from ..core.task import TaskSpec
        info = self.get_dataset_info(dataset_name, **kwargs)
        num_classes = info.get("classes") or info.get("num_classes")
        if num_classes is None:
            return TaskSpec(task_type="regression")
        return TaskSpec.classification(num_classes=int(num_classes))

    def postprocess(self, model_output: Any, **kwargs) -> Any:
        """
        Phase 13.3：模型输出后处理钩子（可选实现）。

        默认直接返回输入（无后处理）。检测场景可覆写以应用 NMS、
        分割场景可覆写以应用 CRF 等。

        Args:
            model_output: 模型原始输出（Tensor / dict / tuple）
            **kwargs: 场景特定参数（如 scene_params、score_threshold 等）

        Returns:
            后处理结果（格式由子类决定）
        """
        return model_output

    def get_normalization_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回数据集的归一化信息（可选实现）。

        供引擎在保存 metadata 时调用，避免引擎越层访问 data 模块。

        Args:
            dataset_name: 数据集名
            **kwargs: 场景特定参数（如 params 上下文）

        Returns:
            归一化信息 dict（如 {"mean": 0.0, "std": 1.0}），无归一化则返回 None
        """
        return None

    def get_manifest_info(self, dataset_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """返回 manifest 信息（仅 CustomContainer 场景实现）。

        供引擎在保存 metadata 时调用，避免引擎越层访问场景私有函数。

        Args:
            dataset_name: 数据集名
            **kwargs: 场景特定参数（如 params 上下文）

        Returns:
            manifest 信息 dict，非 manifest 场景返回 None
        """
        return None

    def get_catalog(self) -> Optional[List[Dict[str, Any]]]:
        """返回场景的技术目录（可选实现）。

        默认返回 None 表示场景无技术目录。
        子类可覆盖以提供场景特定的技术目录，供 SearchSpaceMap
        聚合查询与 Agent 探索使用。

        Returns:
            技术目录条目列表，每项含 name/category/description/applicable/
            params/implemented 等字段；无目录时返回 None
        """
        return None

    def get_scene_params(self, dataset_name: str, **kwargs) -> Optional["SceneParams"]:
        """返回场景参数（可选实现）。

        P5 P3-2：基类显式声明此方法，消除文档-代码契约违规。
        默认返回 None 表示本场景不使用 SceneParams 概念；
        需后处理参数的场景（如 detection）可覆写返回 SceneParams 实例。

        注意：当前引擎层 scene.params 仍为 Dict[str, Any]，
        SceneParams 仅服务于场景内部的后处理流程。Phase 11.4
        正交化落地后，此方法将成为场景参数的统一入口。

        Args:
            dataset_name: 数据集名
            **kwargs: 场景特定参数上下文

        Returns:
            SceneParams 实例，无场景参数则返回 None
        """
        return None
