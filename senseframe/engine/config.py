"""
声明式实验配置 schema：dataclass + 校验。

设计理念（参考 Ludwig 的 input_features/output_features 声明式配置）：
- 用 dataclass 表达实验配置的层级结构，提供类型安全的字段定义
- from_dict() 解析嵌套 dict（YAML/JSON）为 dataclass 实例，并做基本校验
- to_dict() 反向序列化，便于持久化和日志
- validate() 做语义校验（如 optimizer 名称、HPO 方向、early_stopping 正负等）

不引入 pydantic 等重型依赖，仅用标准库 dataclasses + 显式校验。

Phase R3（架构重构）：合并顶层 config.py 的 DEFAULT_DATA_ROOT 到此文件。
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 默认数据根目录（相对于项目根目录）
# Phase R3：从顶层 config.py 迁入，统一配置入口
DEFAULT_DATA_ROOT = str(Path(__file__).parent.parent.parent / "CSI_DATASETS" / "Data")


# ============================================================
# 支持的取值范围（集中管理，便于扩展）
# RFC-002 阶段 K：改为从注册表派生，Agent 可运行时扩展
# ============================================================
# 内置默认值（Agent 可通过 register_* 扩展）
_BUILTIN_FEATURE_TYPES: Tuple[str, ...] = ("csi", "tabular", "image", "text", "sequence")
_BUILTIN_OUTPUT_TYPES: Tuple[str, ...] = ("category", "number", "binary")
_BUILTIN_OPTIMIZERS: Tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")
_BUILTIN_SCHEDULERS: Tuple[Optional[str], ...] = (None, "cosine", "step")
_BUILTIN_GRAD_CLIP_ALGOS: Tuple[str, ...] = ("norm", "value")
_BUILTIN_LEARNING_MODES: Tuple[str, ...] = ("supervised", "self_supervised")
_BUILTIN_HPO_SAMPLERS: Tuple[str, ...] = ("tpe", "random", "cmaes")
_BUILTIN_HPO_PRUNERS: Tuple[str, ...] = ("median", "none", "hyperband")
_BUILTIN_HPO_DIRECTIONS: Tuple[str, ...] = ("minimize", "maximize")
# Phase 2.2a：支持的 logger 后端（csv 始终可用，tensorboard/wandb 按需安装）
_BUILTIN_LOGGERS: Tuple[str, ...] = ("csv", "tensorboard", "wandb", "none")

# 可扩展注册表
_EXTENSION_OPTIMIZERS: set = set()
_EXTENSION_SCHEDULERS: set = set()
_EXTENSION_LOGGERS: set = set()


def register_optimizer(name: str) -> None:
    """注册新 optimizer（RFC-002 阶段 K）。"""
    _EXTENSION_OPTIMIZERS.add(name)


def register_scheduler(name: str) -> None:
    """注册新 scheduler（RFC-002 阶段 K）。"""
    _EXTENSION_SCHEDULERS.add(name)


def register_logger(name: str) -> None:
    """注册新 logger 后端（RFC-002 阶段 K）。"""
    _EXTENSION_LOGGERS.add(name)


# 从注册表派生的 SUPPORTED_* 常量（动态属性）
def _get_supported_optimizers() -> Tuple[str, ...]:
    return _BUILTIN_OPTIMIZERS + tuple(_EXTENSION_OPTIMIZERS)


def _get_supported_schedulers() -> Tuple:
    return _BUILTIN_SCHEDULERS + tuple(_EXTENSION_SCHEDULERS)


def _get_supported_loggers() -> Tuple[str, ...]:
    return _BUILTIN_LOGGERS + tuple(_EXTENSION_LOGGERS)


# 向后兼容：保留 SUPPORTED_* 名称，但改为动态计算
class _SupportedRegistry:
    """动态 SUPPORTED_* 注册表（RFC-002 阶段 K）。"""
    @property
    def FEATURE_TYPES(self): return _BUILTIN_FEATURE_TYPES
    @property
    def OUTPUT_TYPES(self): return _BUILTIN_OUTPUT_TYPES
    @property
    def OPTIMIZERS(self): return _get_supported_optimizers()
    @property
    def SCHEDULERS(self): return _get_supported_schedulers()
    @property
    def GRAD_CLIP_ALGOS(self): return _BUILTIN_GRAD_CLIP_ALGOS
    @property
    def LEARNING_MODES(self): return _BUILTIN_LEARNING_MODES
    @property
    def HPO_SAMPLERS(self): return _BUILTIN_HPO_SAMPLERS
    @property
    def HPO_PRUNERS(self): return _BUILTIN_HPO_PRUNERS
    @property
    def HPO_DIRECTIONS(self): return _BUILTIN_HPO_DIRECTIONS
    @property
    def LOGGERS(self): return _get_supported_loggers()


# 向后兼容：保留模块级常量名（静态检查用），但 validate 时用动态注册表
SUPPORTED_FEATURE_TYPES = _BUILTIN_FEATURE_TYPES
SUPPORTED_OUTPUT_TYPES = _BUILTIN_OUTPUT_TYPES
SUPPORTED_OPTIMIZERS = _BUILTIN_OPTIMIZERS
SUPPORTED_SCHEDULERS = _BUILTIN_SCHEDULERS
SUPPORTED_GRAD_CLIP_ALGOS = _BUILTIN_GRAD_CLIP_ALGOS
SUPPORTED_LEARNING_MODES = _BUILTIN_LEARNING_MODES
SUPPORTED_HPO_SAMPLERS = _BUILTIN_HPO_SAMPLERS
SUPPORTED_HPO_PRUNERS = _BUILTIN_HPO_PRUNERS
SUPPORTED_HPO_DIRECTIONS = _BUILTIN_HPO_DIRECTIONS
SUPPORTED_LOGGERS = _BUILTIN_LOGGERS


# ============================================================
# 子配置 dataclass
# ============================================================
@dataclass
class InputFeature:
    """输入特征声明。"""
    name: str
    type: str                            # 见 SUPPORTED_FEATURE_TYPES
    shape: List[int] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("InputFeature.name 不能为空")
        if self.type not in SUPPORTED_FEATURE_TYPES:
            raise ValueError(
                f"InputFeature.type '{self.type}' 不支持，"
                f"可选: {SUPPORTED_FEATURE_TYPES}"
            )
        if not isinstance(self.shape, (list, tuple)):
            raise ValueError(f"InputFeature.shape 必须是 list/tuple，实际: {type(self.shape)}")


@dataclass
class OutputFeature:
    """输出特征声明。"""
    name: str
    type: str                            # 见 SUPPORTED_OUTPUT_TYPES
    num_classes: Optional[int] = None    # type=category/binary 时必填

    def validate(self) -> None:
        if not self.name:
            raise ValueError("OutputFeature.name 不能为空")
        if self.type not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"OutputFeature.type '{self.type}' 不支持，"
                f"可选: {SUPPORTED_OUTPUT_TYPES}"
            )
        if self.type in ("category", "binary") and self.num_classes is None:
            raise ValueError(
                f"OutputFeature.type='{self.type}' 需要指定 num_classes"
            )
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError(f"num_classes 必须 >= 2，实际: {self.num_classes}")


@dataclass
class TrainerConfig:
    """训练器配置。

    RFC-004 方案 E：默认即最佳实践 — 默认配置含正则化 + 早停 + scheduler，
    新手开箱即用，专家可覆盖。
    """
    epochs: int = 100
    # Phase 1.1b: learning_rate 默认 None，由 resolve_config 三级回退填充
    # （YAML > 模型 default_lr > 1e-3），修复模型级 default_lr 不生效的 bug
    learning_rate: Optional[float] = None
    batch_size: int = 64
    optimizer: str = "adam"
    # RFC-004 方案 E：默认 L2 正则（1e-4 是 Adam 族的常用值）
    weight_decay: float = 1e-4
    # RFC-004 方案 E：默认早停（patience=5，5 epoch 无 val_loss 提升则停）
    early_stopping: Optional[int] = 5       # patience，None 表示不启用
    early_stopping_min_delta: float = 0.001  # val_loss 提升阈值，低于此值视为无提升
    deterministic: bool = False
    max_time: Optional[str] = None           # DD:HH:MM:SS 格式，None 不限时
    seed: int = 42
    # 优化 7：进度条配置（后台进程可关闭，依赖日志回调监控）
    enable_progress_bar: bool = True
    # Phase 1.1a：scheduler 正式入 schema（不再依赖 scene.params 透传）
    # RFC-004 方案 E：默认 cosine 衰减（比 None 更稳定的训练动态）
    scheduler: Optional[str] = "cosine"      # None / cosine / step
    # Phase 1.2a：梯度裁剪（Lightning Trainer 原生支持）
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"    # norm / value
    # Phase 1.2a：梯度累积（大模型等效大 batch）
    accumulate_grad_batches: int = 1
    # Phase 2.2a：logger 后端选择（csv / tensorboard / wandb / none）
    # csv 始终可用；tensorboard/wandb 需安装对应包；none 关闭日志
    logger: str = "csv"
    # Phase 11.4：scene.params 正交化 — 常用键提升为顶层字段
    # 向后兼容：scene.params 中的同名键仍可覆盖（escape hatch）
    self_supervised_epochs: int = 100        # 自监督预训练轮数
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "macro_f1"])
    gpu: Optional[int] = None                # 指定 GPU ID，None=自动
    resume: Optional[str] = None            # resume checkpoint 路径，None=不恢复
    # Phase 14.3.3：mixed_precision 正交化（原 scene.params 透传）
    # True=16-mixed, False=32, 字符串直接透传（如 "bf16-mixed"），None=自动
    mixed_precision: Optional[Any] = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs 必须 > 0，实际: {self.epochs}")
        # learning_rate=None 合法（由 resolve_config 填充），仅校验非 None 的情况
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须 > 0，实际: {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，实际: {self.batch_size}")
        # RFC-002 阶段 K：从动态注册表查询
        if self.optimizer not in _get_supported_optimizers():
            raise ValueError(
                f"optimizer '{self.optimizer}' 不支持，"
                f"可选: {_get_supported_optimizers()}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay 必须 >= 0，实际: {self.weight_decay}")
        if self.early_stopping is not None and self.early_stopping <= 0:
            raise ValueError(f"early_stopping 必须 > 0，实际: {self.early_stopping}")
        # RFC-004 方案 E：early_stopping_min_delta 校验
        if hasattr(self, "early_stopping_min_delta") and self.early_stopping_min_delta < 0:
            raise ValueError(
                f"early_stopping_min_delta 必须 >= 0，实际: {self.early_stopping_min_delta}"
            )
        if self.scheduler not in _get_supported_schedulers():
            raise ValueError(
                f"scheduler '{self.scheduler}' 不支持，"
                f"可选: {_get_supported_schedulers()}"
            )
        if self.gradient_clip_algorithm not in SUPPORTED_GRAD_CLIP_ALGOS:
            raise ValueError(
                f"gradient_clip_algorithm '{self.gradient_clip_algorithm}' 不支持，"
                f"可选: {SUPPORTED_GRAD_CLIP_ALGOS}"
            )
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val 必须 > 0，实际: {self.gradient_clip_val}")
        if self.accumulate_grad_batches <= 0:
            raise ValueError(f"accumulate_grad_batches 必须 > 0，实际: {self.accumulate_grad_batches}")
        # Phase 11.4：self_supervised_epochs 校验
        if self.self_supervised_epochs <= 0:
            raise ValueError(f"self_supervised_epochs 必须 > 0，实际: {self.self_supervised_epochs}")
        # RFC-002 阶段 K：从动态注册表查询
        if self.logger not in _get_supported_loggers():
            raise ValueError(
                f"logger '{self.logger}' 不支持，"
                f"可选: {_get_supported_loggers()}"
            )


@dataclass
class HPOConfig:
    """
    超参搜索配置（Stage 4 Optuna 集成使用）。

    Phase 4.1：新增持久化与断点续搜支持：
    - storage: 持久化后端 URL（如 "sqlite:///runs/hpo_study.db"），None=内存
    - study_name: study 名称，配合 load_if_exists 实现断点续搜
    - load_if_exists: True 时若同名 study 已存在则恢复，否则新建
    - export_path: HPO 结果导出 JSON 路径，None=不导出
    - timeout: 搜索超时秒数，None=不限制
    """
    enabled: bool = False
    n_trials: int = 20
    sampler: str = "tpe"
    pruner: str = "median"
    metric: str = "val_loss"
    direction: str = "minimize"
    # Phase 4.1：持久化与断点续搜
    storage: Optional[str] = None
    study_name: Optional[str] = None
    load_if_exists: bool = False
    export_path: Optional[str] = None
    # Phase 4.1：搜索超时（秒），与 n_trials 任一满足即停止
    timeout: Optional[float] = None

    def validate(self) -> None:
        if self.enabled:
            if self.n_trials <= 0:
                raise ValueError(f"hpo.n_trials 必须 > 0，实际: {self.n_trials}")
            if self.sampler not in SUPPORTED_HPO_SAMPLERS:
                raise ValueError(
                    f"hpo.sampler '{self.sampler}' 不支持，"
                    f"可选: {SUPPORTED_HPO_SAMPLERS}"
                )
            if self.pruner not in SUPPORTED_HPO_PRUNERS:
                raise ValueError(
                    f"hpo.pruner '{self.pruner}' 不支持，"
                    f"可选: {SUPPORTED_HPO_PRUNERS}"
                )
            if self.direction not in SUPPORTED_HPO_DIRECTIONS:
                raise ValueError(
                    f"hpo.direction '{self.direction}' 不支持，"
                    f"可选: {SUPPORTED_HPO_DIRECTIONS}"
                )
            if not self.metric:
                raise ValueError("hpo.metric 不能为空")
            # Phase 4.1：load_if_exists=True 时 study_name 必须指定
            if self.load_if_exists and not self.study_name:
                raise ValueError(
                    "hpo.load_if_exists=True 时必须指定 hpo.study_name"
                )
            # Phase 4.1：timeout 校验
            if self.timeout is not None and self.timeout <= 0:
                raise ValueError(f"hpo.timeout 必须 > 0，实际: {self.timeout}")


# R-fix：TaskSpecField 合并到 core.task.TaskSpec，消除重复定义
# TaskSpec 已支持 loss_kwargs 字段和 validate() 方法
from ..core.task import TaskSpec as TaskSpecField


@dataclass
class SceneConfig:
    """场景配置：声明使用的场景容器、数据集、模型与学习范式。"""
    name: str                                    # 场景名，如 "wifi_csi"
    dataset: str                                 # 数据集名
    model_id: str                                # 模型 ID
    learning_mode: str = "supervised"            # 见 SUPPORTED_LEARNING_MODES
    data_root: Optional[str] = None              # 数据根目录，None 用场景默认
    params: Dict[str, Any] = field(default_factory=dict)  # 场景特定参数
    # Phase 12.3：task_spec 字段，None 表示用 scene 默认
    task_spec: Optional[TaskSpecField] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("scene.name 不能为空")
        if not self.dataset:
            raise ValueError("scene.dataset 不能为空")
        if not self.model_id:
            raise ValueError("scene.model_id 不能为空")
        if self.learning_mode not in SUPPORTED_LEARNING_MODES:
            raise ValueError(
                f"scene.learning_mode '{self.learning_mode}' 不支持，"
                f"可选: {SUPPORTED_LEARNING_MODES}"
            )
        if self.task_spec is not None:
            self.task_spec.validate()


# ============================================================
# 顶层实验配置
# ============================================================
@dataclass
class ExperimentConfig:
    """
    顶层实验配置：组合 scene / features / trainer / hpo。

    RFC Phase E：新增工厂注入字段，Agent 可注入自定义
    LightningModule / DataModule / Callbacks，消除硬编码实例化。

    使用方式：
        cfg = ExperimentConfig.from_dict(yaml_dict)
        cfg.validate()  # 显式校验
        # ... 传给 engine.run_experiment(cfg)
    """
    scene: SceneConfig
    input_features: List[InputFeature]
    output_features: List[OutputFeature]
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    output_dir: str = "runs"
    save_model: bool = True
    # s2: 导出格式（纳入 schema，替代 cli.py 的 setattr 动态注入）
    export_formats: List[str] = field(default_factory=list)  # 如 ["state_dict", "onnx"]
    # RFC Phase E：工厂注入字段（消除硬编码实例化，Agent 可注入自定义实现）
    # 均为可选，None 时使用框架默认（GenericLightningModule / GenericDataModule / 标准callbacks）
    # 注意：这些字段不参与 from_dict/to_dict（Python 对象，非声明式配置）
    module_factory: Optional[Any] = None      # Callable(model, **kwargs) -> pl.LightningModule
    datamodule_factory: Optional[Any] = None   # Callable(train_ds, test_ds, **kwargs) -> pl.LightningDataModule
    extra_callbacks: List[Any] = field(default_factory=list)  # List[pl.Callback]
    # RFC-002 阶段 K：Trainer 工厂注入，Agent 可自定义 Trainer 构造
    # None 时使用框架默认 pl.Trainer 构造逻辑
    trainer_factory: Optional[Any] = None      # Callable(**kwargs) -> pl.Trainer

    # ------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """
        从 dict（YAML/JSON 解析结果）构造 ExperimentConfig。

        Args:
            d: 配置字典，必须包含 scene / input_features / output_features

        Raises:
            ValueError: 缺少必需字段或字段类型错误
        """
        if not isinstance(d, dict):
            raise ValueError(f"ExperimentConfig.from_dict 需要 dict，实际: {type(d)}")

        # 必需字段检查
        for required in ("scene", "input_features", "output_features"):
            if required not in d:
                raise ValueError(f"缺少必需字段: '{required}'")

        # scene
        scene_dict = dict(d["scene"])
        # Phase 12.3：递归解析 task_spec 子字段
        if "task_spec" in scene_dict and scene_dict["task_spec"] is not None:
            scene_dict["task_spec"] = _parse_dataclass(
                TaskSpecField, scene_dict["task_spec"]
            )
        scene = _parse_dataclass(SceneConfig, scene_dict)

        # input_features
        input_features_raw = d["input_features"]
        if not isinstance(input_features_raw, list) or len(input_features_raw) == 0:
            raise ValueError("input_features 必须是非空 list")
        input_features = [_parse_dataclass(InputFeature, f) for f in input_features_raw]

        # output_features
        output_features_raw = d["output_features"]
        if not isinstance(output_features_raw, list) or len(output_features_raw) == 0:
            raise ValueError("output_features 必须是非空 list")
        output_features = [_parse_dataclass(OutputFeature, f) for f in output_features_raw]

        # trainer（可选，缺省用默认值）
        trainer = _parse_dataclass(TrainerConfig, d.get("trainer", {}))

        # hpo（可选）
        hpo = _parse_dataclass(HPOConfig, d.get("hpo", {}))

        return cls(
            scene=scene,
            input_features=input_features,
            output_features=output_features,
            trainer=trainer,
            hpo=hpo,
            output_dir=d.get("output_dir", "runs"),
            save_model=d.get("save_model", True),
            export_formats=d.get("export_formats", []),
        )

    # ------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------
    def validate(self) -> None:
        """递归校验所有子配置。"""
        self.scene.validate()
        if not self.input_features:
            raise ValueError("input_features 不能为空")
        if not self.output_features:
            raise ValueError("output_features 不能为空")
        for f in self.input_features:
            f.validate()
        for f in self.output_features:
            f.validate()
        self.trainer.validate()
        self.hpo.validate()
        if not self.output_dir:
            raise ValueError("output_dir 不能为空")
        # s2: 校验 export_formats
        valid_formats = {"state_dict", "torchscript", "onnx", "quantized_onnx"}
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid export_format '{fmt}'. Supported: {valid_formats}"
                )

    # ------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """反向序列化为 dict（可写 YAML/JSON）。"""
        return {
            "scene": _dataclass_to_dict(self.scene),
            "input_features": [_dataclass_to_dict(f) for f in self.input_features],
            "output_features": [_dataclass_to_dict(f) for f in self.output_features],
            "trainer": _dataclass_to_dict(self.trainer),
            "hpo": _dataclass_to_dict(self.hpo),
            "output_dir": self.output_dir,
            "save_model": self.save_model,
            "export_formats": self.export_formats,
        }


# ============================================================
# 内部工具：dict <-> dataclass 转换
# ============================================================
def _parse_dataclass(cls, data: Any):
    """
    将 dict 转换为 cls 的实例。

    - 仅取 cls 已定义的字段，忽略 dict 中多余键（前向兼容）
    - 缺失字段使用 dataclass 默认值
    - 嵌套 dataclass 不自动递归（本 schema 仅顶层 ExperimentConfig 含嵌套，
      已在 from_dict 中显式处理）
    """
    if not is_dataclass(cls):
        raise ValueError(f"_parse_dataclass 需要 dataclass 类型，实际: {cls}")
    if not isinstance(data, dict):
        raise ValueError(f"解析 {cls.__name__} 需要 dict，实际: {type(data)}")

    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> Any:
    """递归将 dataclass 实例转为 dict（含嵌套 dataclass 与 list）。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


__all__ = [
    "InputFeature",
    "OutputFeature",
    "TrainerConfig",
    "HPOConfig",
    "SceneConfig",
    "TaskSpecField",
    "ExperimentConfig",
    "SUPPORTED_FEATURE_TYPES",
    "SUPPORTED_OUTPUT_TYPES",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_SCHEDULERS",
    "SUPPORTED_GRAD_CLIP_ALGOS",
    "SUPPORTED_LEARNING_MODES",
    "SUPPORTED_HPO_SAMPLERS",
    "SUPPORTED_HPO_PRUNERS",
    "SUPPORTED_HPO_DIRECTIONS",
    "SUPPORTED_LOGGERS",
]
