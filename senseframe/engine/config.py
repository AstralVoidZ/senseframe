"""
声明式实验配置 schema：pydantic v2 + dataclass 混合（演进中）。

设计理念（参考 Ludwig 的 input_features/output_features 声明式配置）：
- 用 pydantic BaseModel 表达用户面配置层级结构，提供类型安全 + 自动校验 + JSON Schema
- from_dict() / to_dict() / validate() 保留为薄封装，向后兼容 23 个调用点
- 训练热路径对象（PipelineContext/Batch 等）保留 dataclass，避免 pydantic 构造开销

演进历史：
- Phase 1（初版）：纯 dataclass + 手写校验（42 处 raise ValueError）
- Phase 2（2026-07-18 起）：核心 6 个配置类迁移到 pydantic v2
  - 收益：自动字段路径错误、自动 JSON Schema 生成、统一序列化协议
  - 兼容：from_dict/to_dict/validate 保留为薄封装，调用点无需改造
  - 禁区：训练热路径（Batch/ModelState）保留 dataclass

Phase R3（架构重构）：合并顶层 config.py 的 DEFAULT_DATA_ROOT 到此文件。
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# P2 演进（2026-07-18）：from_dict 路径抛 ConfigValidationError 而非纯 ValueError，
# 让 classify_error 能直接命中 SenseFrameError 分支，错误码精确为 CONFIG_VALIDATION_ERROR。
# ConfigValidationError 同时继承 ValueError，现有 `except ValueError` 调用点无需改造。
# 延迟导入避免循环依赖：engine.runner.orchestrator 反向依赖 engine.config。


def _config_validation_error(message: str):
    """构造 ConfigValidationError（延迟导入避免循环依赖）。

    ConfigValidationError 继承 SenseFrameError + ValueError：
    - 现有 `except ValueError` 调用点无需改造
    - classify_error 直接命中 SenseFrameError 分支，返回 CONFIG_VALIDATION_ERROR 错误码
    """
    from .runner.errors import ConfigValidationError
    return ConfigValidationError(message)


# 数据根目录：框架不猜测路径，由调用者显式提供（YAML scene.data_root / CLI --data-root
# / env SENSEFRAME_DATA_ROOT）。SceneConfig.data_root 必填，validate 校验非空。
# 删除原 DEFAULT_DATA_ROOT 模块级常量：import 时推导路径是反模式，且掩盖配置缺失。


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
# 子配置（pydantic v2 BaseModel，P1 演进）
# ============================================================
class InputFeature(BaseModel):
    """输入特征声明。

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    保留 validate() / to_dict() / from_dict() 兼容方法。
    P2 演进（2026-07-18）：extra="forbid" 捕获 YAML 字段拼写错误（如 shapes/nam/typ）。
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    type: str                            # 见 SUPPORTED_FEATURE_TYPES
    shape: List[int] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("InputFeature.name 不能为空")
        return v

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in SUPPORTED_FEATURE_TYPES:
            raise ValueError(
                f"InputFeature.type '{v}' 不支持，"
                f"可选: {SUPPORTED_FEATURE_TYPES}"
            )
        return v

    def validate(self) -> None:
        """向后兼容：pydantic 构造时已校验，此方法仅做二次确认（no-op）。"""
        # pydantic v2 在 __init__ 时自动校验，这里保留方法签名兼容旧调用
        # 若需显式重新校验，调用 self.model_validate(self.model_dump())
        pass

    def to_dict(self) -> Dict[str, Any]:
        """向后兼容：委托给 model_dump。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InputFeature":
        """向后兼容：委托给 model_validate。"""
        return cls.model_validate(d)


class OutputFeature(BaseModel):
    """输出特征声明。

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    P2 演进（2026-07-18）：extra="forbid" 捕获 YAML 字段拼写错误（如 num_class/n_classes）。
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    type: str                            # 见 SUPPORTED_OUTPUT_TYPES
    num_classes: Optional[int] = None    # type=category/binary 时必填

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("OutputFeature.name 不能为空")
        return v

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"OutputFeature.type '{v}' 不支持，"
                f"可选: {SUPPORTED_OUTPUT_TYPES}"
            )
        return v

    @model_validator(mode="after")
    def _validate_num_classes(self) -> "OutputFeature":
        if self.type in ("category", "binary") and self.num_classes is None:
            raise ValueError(
                f"OutputFeature.type='{self.type}' 需要指定 num_classes"
            )
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError(f"num_classes 必须 >= 2，实际: {self.num_classes}")
        return self

    def validate(self) -> None:
        """向后兼容：pydantic 构造时已校验，此方法仅做二次确认（no-op）。"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """向后兼容：委托给 model_dump。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputFeature":
        """向后兼容：委托给 model_validate。"""
        return cls.model_validate(d)


class TrainerConfig(BaseModel):
    """训练器配置。

    RFC-004 方案 E：默认即最佳实践 — 默认配置含正则化 + 早停 + scheduler，
    新手开箱即用，专家可覆盖。

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    - 3 个 Optional[Any] 字段收窄为 Union：
      - mixed_precision: Optional[Union[bool, str]]（True/False/"bf16-mixed" 等）
      - limit_train_batches: Optional[Union[int, float]]（1 或 1.0）
      - limit_val_batches: Optional[Union[int, float]]
    - 动态注册表（optimizer/scheduler/logger）用 @field_validator 查询
    - 保留 validate() / to_dict() / from_dict() 兼容方法
    P2 演进（2026-07-18）：extra="forbid" 捕获 YAML 字段拼写错误（如 max_epochs/lr/patience），
    在 schema 层强制 test_doc_contract.py 已有的 max_epochs 文档契约。
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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
    # P2-3 修复：EarlyStopping/ModelCheckpoint monitor 可配置化
    # 默认 "val_loss"，与 module.py validation_step 的 log key 对齐
    early_stopping_monitor: str = "val_loss"
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
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "macro_f1"])
    gpu: Optional[int] = None                # 指定 GPU ID，None=自动
    resume: Optional[str] = None            # resume checkpoint 路径，None=不恢复
    # Phase 14.3.3：mixed_precision 正交化（原 scene.params 透传）
    # P1 演进：Optional[Any] → Optional[Union[bool, str]]，收窄类型
    # True=16-mixed, False=32, 字符串直接透传（如 "bf16-mixed"），None=自动
    mixed_precision: Optional[Union[bool, str]] = None
    # P2-3: dry-run 动态校验支持 — 限制训练/验证 batch 数
    # P1 演进：Optional[Any] → Optional[Union[int, float]]，收窄类型
    # None=不限制（默认，向后兼容），1.0/1=只跑 1 batch（dry-run 用）
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    # num_workers: DataLoader 并行加载进程数。
    # None=由 routing 按资源优先级自动派生（默认），显式值覆盖 routing 决策。
    # Windows + Python 3.14 spawn 模式下 num_workers>0 需 if __name__=='__main__' 保护；
    # 用户脚本若无此保护，可显式设为 0 规避 multiprocessing 错误。
    num_workers: Optional[int] = None
    # Part 4：自动 LR 标定（Lightning LR Range Test）。
    # True 时 stage_train 在 fit 前用独立 Trainer 调 trainer.tune()，基于前向+反向
    # 1 epoch 数据自动建议学习率。耗时约 1 epoch，默认 False（显式启用）。
    # 仅监督模式生效（自监督模式的 LR 需不同策略）。
    # 设计决策（风险推演 R3）：用独立 tune_trainer 隔离副作用，不污染训练 Trainer 状态。
    auto_lr_find: bool = False

    @field_validator("epochs")
    @classmethod
    def _validate_epochs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"epochs 必须 > 0，实际: {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, v: Optional[float]) -> Optional[float]:
        # learning_rate=None 合法（由 resolve_config 填充），仅校验非 None 的情况
        if v is not None and v <= 0:
            raise ValueError(f"learning_rate 必须 > 0，实际: {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"batch_size 必须 > 0，实际: {v}")
        return v

    @field_validator("optimizer")
    @classmethod
    def _validate_optimizer(cls, v: str) -> str:
        # RFC-002 阶段 K：从动态注册表查询（运行时可扩展）
        if v not in _get_supported_optimizers():
            raise ValueError(
                f"optimizer '{v}' 不支持，"
                f"可选: {_get_supported_optimizers()}"
            )
        return v

    @field_validator("weight_decay")
    @classmethod
    def _validate_weight_decay(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"weight_decay 必须 >= 0，实际: {v}")
        return v

    @field_validator("early_stopping")
    @classmethod
    def _validate_early_stopping(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"early_stopping 必须 > 0，实际: {v}")
        return v

    @field_validator("early_stopping_min_delta")
    @classmethod
    def _validate_early_stopping_min_delta(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"early_stopping_min_delta 必须 >= 0，实际: {v}")
        return v

    @field_validator("scheduler")
    @classmethod
    def _validate_scheduler(cls, v: Optional[str]) -> Optional[str]:
        if v not in _get_supported_schedulers():
            raise ValueError(
                f"scheduler '{v}' 不支持，"
                f"可选: {_get_supported_schedulers()}"
            )
        return v

    @field_validator("gradient_clip_algorithm")
    @classmethod
    def _validate_gradient_clip_algorithm(cls, v: str) -> str:
        if v not in SUPPORTED_GRAD_CLIP_ALGOS:
            raise ValueError(
                f"gradient_clip_algorithm '{v}' 不支持，"
                f"可选: {SUPPORTED_GRAD_CLIP_ALGOS}"
            )
        return v

    @field_validator("gradient_clip_val")
    @classmethod
    def _validate_gradient_clip_val(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"gradient_clip_val 必须 > 0，实际: {v}")
        return v

    @field_validator("accumulate_grad_batches")
    @classmethod
    def _validate_accumulate_grad_batches(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"accumulate_grad_batches 必须 > 0，实际: {v}")
        return v

    @field_validator("self_supervised_epochs")
    @classmethod
    def _validate_self_supervised_epochs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"self_supervised_epochs 必须 > 0，实际: {v}")
        return v

    @field_validator("logger")
    @classmethod
    def _validate_logger(cls, v: str) -> str:
        # RFC-002 阶段 K：从动态注册表查询（运行时可扩展）
        if v not in _get_supported_loggers():
            raise ValueError(
                f"logger '{v}' 不支持，"
                f"可选: {_get_supported_loggers()}"
            )
        return v

    def validate(self) -> None:
        """向后兼容：pydantic 构造时已校验，此方法仅做二次确认（no-op）。

        旧调用方（如 cli.py 的 config.validate()）仍可工作，实际校验在
        BaseModel.__init__ 时已完成。若需显式重新校验，调用
        self.model_validate(self.model_dump())。
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """向后兼容：委托给 model_dump。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        """向后兼容：委托给 model_validate。"""
        return cls.model_validate(d)


class HPOConfig(BaseModel):
    """超参搜索配置（Stage 4 Optuna 集成使用）。

    Phase 4.1：新增持久化与断点续搜支持：
    - storage: 持久化后端 URL（如 "sqlite:///runs/hpo_study.db"），None=内存
    - study_name: study 名称，配合 load_if_exists 实现断点续搜
    - load_if_exists: True 时若同名 study 已存在则恢复，否则新建
    - export_path: HPO 结果导出 JSON 路径，None=不导出
    - timeout: 搜索超时秒数，None=不限制

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    - 保留 enabled=False 时跳过校验的语义（@model_validator 内分支）
    - 动态注册表（sampler/pruner/direction）用 @field_validator 查询
    - 保留 validate() / to_dict() / from_dict() 兼容方法
    P2 演进（2026-07-18）：extra="forbid" 捕获 YAML 字段拼写错误（如 trials/n_trial/prune）。
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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

    @field_validator("n_trials")
    @classmethod
    def _validate_n_trials(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"hpo.n_trials 必须 > 0，实际: {v}")
        return v

    @field_validator("sampler")
    @classmethod
    def _validate_sampler(cls, v: str) -> str:
        if v not in SUPPORTED_HPO_SAMPLERS:
            raise ValueError(
                f"hpo.sampler '{v}' 不支持，"
                f"可选: {SUPPORTED_HPO_SAMPLERS}"
            )
        return v

    @field_validator("pruner")
    @classmethod
    def _validate_pruner(cls, v: str) -> str:
        if v not in SUPPORTED_HPO_PRUNERS:
            raise ValueError(
                f"hpo.pruner '{v}' 不支持，"
                f"可选: {SUPPORTED_HPO_PRUNERS}"
            )
        return v

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, v: str) -> str:
        if v not in SUPPORTED_HPO_DIRECTIONS:
            raise ValueError(
                f"hpo.direction '{v}' 不支持，"
                f"可选: {SUPPORTED_HPO_DIRECTIONS}"
            )
        return v

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        if not v:
            raise ValueError("hpo.metric 不能为空")
        return v

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"hpo.timeout 必须 > 0，实际: {v}")
        return v

    @model_validator(mode="after")
    def _validate_load_if_exists_requires_study_name(self) -> "HPOConfig":
        """load_if_exists=True 时 study_name 必须指定（Phase 4.1）。"""
        if self.load_if_exists and not self.study_name:
            raise ValueError(
                "hpo.load_if_exists=True 时必须指定 hpo.study_name"
            )
        return self

    def validate(self) -> None:
        """向后兼容：pydantic 构造时已校验，此方法仅做二次确认（no-op）。

        旧实现根据 enabled 短路校验，pydantic 版本在构造时已对所有字段
        做结构校验（n_trials/sampler/pruner/direction/metric/timeout），
        跨字段约束（load_if_exists + study_name）由 @model_validator 处理。
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """向后兼容：委托给 model_dump。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HPOConfig":
        """向后兼容：委托给 model_validate。"""
        return cls.model_validate(d)


# R-fix：TaskSpecField 合并到 core.task.TaskSpec，消除重复定义
# TaskSpec 已支持 loss_kwargs 字段和 validate() 方法
from ..core.task import TaskSpec as TaskSpecField


class SceneConfig(BaseModel):
    """场景配置：声明使用的场景容器、数据集、模型与学习范式。

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    - name/dataset/model_id/learning_mode 用 @field_validator 校验
    - data_root 校验保留在 validate() 中（CLI/env 后填充，不能在构造时校验）
    - params: Optional[Any] 保留（运行时为 SceneParams，用 Any 避免 import cycle）
    - task_spec: Optional[TaskSpec] 保留（arbitrary_types_allowed=True）
    - 保留 validate() / to_dict() / from_dict() 兼容方法
    P2 演进（2026-07-18）：extra="forbid" 捕获 YAML 字段拼写错误（如 model/data/scene_name）。
    注意：scene.params 是声明字段，其中的任意键不算 extra，escape hatch 不受影响。
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str                                    # 场景名，如 "wifi_csi"
    dataset: str                                 # 数据集名
    model_id: str                                # 模型 ID
    learning_mode: str = "supervised"            # 见 SUPPORTED_LEARNING_MODES
    data_root: str = ""                          # 数据根目录，必填（YAML/CLI/env 三选一）
    # P5 P3-4 完整正交化：params 从 Dict[str, Any] 收窄为 Optional[SceneParams]
    # SceneParams 提供 dict-like 兼容层（[]/= /in/.get()/.items()），下游零改动
    params: Optional[Any] = None  # 运行时为 Optional[SceneParams]，用 Any 避免 import cycle
    # Phase 12.3：task_spec 字段，None 表示用 scene 默认
    task_spec: Optional[TaskSpecField] = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("scene.name 不能为空")
        return v

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, v: str) -> str:
        if not v:
            raise ValueError("scene.dataset 不能为空")
        return v

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, v: str) -> str:
        if not v:
            raise ValueError("scene.model_id 不能为空")
        return v

    @field_validator("learning_mode")
    @classmethod
    def _validate_learning_mode(cls, v: str) -> str:
        if v not in SUPPORTED_LEARNING_MODES:
            raise ValueError(
                f"scene.learning_mode '{v}' 不支持，"
                f"可选: {SUPPORTED_LEARNING_MODES}"
            )
        return v

    def validate(self) -> None:
        """向后兼容：data_root 校验保留在此处（CLI/env 后填充）。

        name/dataset/model_id/learning_mode 已由 @field_validator 在构造时校验。
        data_root 可能在构造后由 CLI --data-root 或 env SENSEFRAME_DATA_ROOT 填充，
        因此不能在构造时校验，必须延后到显式 validate() 调用。
        task_spec.validate() 也在此处调用（TaskSpec 是 dataclass，无 pydantic 校验）。
        """
        if not self.data_root:
            raise ValueError(
                "scene.data_root 必填。提供方式（三选一）：\n"
                "  - YAML: scene.data_root: /path/to/CSI_DATASETS\n"
                "  - CLI: --data-root /path/to/CSI_DATASETS\n"
                "  - Env: SENSEFRAME_DATA_ROOT=/path/to/CSI_DATASETS"
            )
        if self.task_spec is not None:
            self.task_spec.validate()

    def to_dict(self) -> Dict[str, Any]:
        """向后兼容：委托给 model_dump。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneConfig":
        """向后兼容：委托给 model_validate。"""
        return cls.model_validate(d)


# ============================================================
# 顶层实验配置
# ============================================================
# P2 演进（2026-07-18）：工厂字段拆分到独立 RuntimeInjections dataclass。
# 原先 4 个工厂字段直接声明在 ExperimentConfig 中，用 Field(exclude=True) 排除序列化，
# model_json_schema() 覆盖移除。现拆分到 RuntimeInjections，让 ExperimentConfig
# 完全只含声明式配置，model_fields 不再含工厂字段，schema 自动纯净。
#
# 访问兼容性（属性代理）：ExperimentConfig 通过 @property + setter 代理 4 个工厂字段，
# 所有访问点（ctx.config.module_factory / cfg.extra_callbacks.append / config.X = ...）
# 无需修改，零回归风险。
_RUNTIME_FACTORY_FIELDS: Tuple[str, ...] = (
    "module_factory", "datamodule_factory", "extra_callbacks", "trainer_factory",
)


@dataclass
class RuntimeInjections:
    """运行时注入对象集合（非声明式配置，不参与序列化）。

    P2 演进（2026-07-18）：从 ExperimentConfig 拆分出 4 个工厂字段。
    这些字段是运行时 Python 对象（Callable / Callback），不可序列化，
    不属于声明式配置，YAML/JSON 中不出现。

    设计决策：用 dataclass 而非 pydantic BaseModel
    - 工厂对象类型是 Any（Callable / Callback），pydantic 校验无收益
    - dataclass 构造开销低（运行时热路径外，但仍避免不必要开销）
    - 避免 pydantic arbitrary_types_allowed 的类型穿透问题

    访问方式：
        # 直接访问（推荐）
        cfg.runtime.module_factory
        cfg.runtime.extra_callbacks.append(cb)

        # 兼容代理（旧代码无需修改）
        cfg.module_factory  # 等价于 cfg.runtime.module_factory
        cfg.extra_callbacks  # 等价于 cfg.runtime.extra_callbacks
    """
    module_factory: Optional[Any] = None       # Callable(model, **kwargs) -> pl.LightningModule
    datamodule_factory: Optional[Any] = None   # Callable(train_ds, test_ds, **kwargs) -> pl.LightningDataModule
    extra_callbacks: List[Any] = field(default_factory=list)  # List[pl.Callback]
    trainer_factory: Optional[Any] = None      # Callable(**kwargs) -> pl.Trainer


class ExperimentConfig(BaseModel):
    """顶层实验配置：组合 scene / features / trainer / hpo。

    RFC Phase E：新增工厂注入字段，Agent 可注入自定义
    LightningModule / DataModule / Callbacks，消除硬编码实例化。

    P1 演进（2026-07-18）：从 dataclass 迁移到 pydantic v2 BaseModel。
    - input_features/output_features 非空检查用 @model_validator
    - export_formats 合法值校验用 @field_validator
    - 保留 from_dict() / to_dict() / validate() 兼容方法

    P2 演进（2026-07-18）：
    - extra="forbid" 捕获 YAML 字段拼写错误（如 train/output/save）
    - 分布式训练字段（devices/strategy/num_nodes/sync_batchnorm/num_processes）提升为
      声明字段，修复"文档化顶层 YAML 字段被 extra='ignore' 静默丢弃"的预存在 bug
    - 工厂字段拆分到 RuntimeInjections dataclass（消除 4 个不可序列化 Any 字段），
      ExperimentConfig 通过 @property 代理访问，所有调用点零改动

    使用方式：
        cfg = ExperimentConfig.from_dict(yaml_dict)
        cfg.validate()  # 显式校验
        # ... 传给 engine.run_experiment(cfg)

        # 工厂注入（向后兼容，等价于 cfg.runtime.module_factory = ...）
        cfg.module_factory = my_factory
        cfg.extra_callbacks.append(my_callback)
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scene: SceneConfig
    input_features: List[InputFeature]
    output_features: List[OutputFeature]
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    output_dir: str = "runs"
    save_model: bool = True
    # s2: 导出格式（纳入 schema，替代 cli.py 的 setattr 动态注入）
    export_formats: List[str] = Field(default_factory=list)  # 如 ["state_dict", "onnx"]
    # P2 演进：工厂字段拆分到 RuntimeInjections dataclass（非声明式配置，排除序列化）。
    # 通过 @property 代理访问，cfg.module_factory 等价于 cfg.runtime.module_factory。
    # model_validator(mode='before') 提取构造时传入的工厂字段，转发到 runtime。
    runtime: RuntimeInjections = Field(default_factory=RuntimeInjections, exclude=True)
    # P1-1: 严格 schema 校验模式（True 时 training_log 类型污染直接抛错，
    # False 时降级保留原始 entry 保持向后兼容）
    strict_schema: bool = False
    # P2 修复（2026-07-18）：分布式训练字段提升为声明字段。
    # 旧实现这些字段在 config_schema.md 中文档化为"YAML 顶层字段"，但未在
    # ExperimentConfig 中声明，被 extra="ignore" 静默丢弃，routing.py 永远读到
    # 默认值——文档化的分布式训练 YAML 配置实际不生效（预存在 bug）。
    # 现在提升为声明字段，experiment_config_to_dict 输出后由 routing.py 消费。
    # 默认值与 routing.py 的 fallback 默认值对齐（devices=1/strategy=None/
    # num_nodes=1/sync_batchnorm=False/num_processes=1）。
    devices: Union[int, str] = 1             # GPU 数量（int 或 "auto"），默认 1（单卡）
    strategy: Optional[str] = None           # 分布式策略（"ddp"/"ddp_spawn"/"fsfp"），None=单设备
    num_nodes: int = 1                       # 多节点训练节点数
    sync_batchnorm: bool = False             # 分布式同步 BatchNorm
    num_processes: int = 1                   # CPU 模式并行进程数（仅 device=cpu 时生效）

    @model_validator(mode="before")
    @classmethod
    def _extract_factory_fields(cls, data: Any) -> Any:
        """提取构造时传入的工厂字段，转发到 runtime。

        向后兼容：允许 ExperimentConfig(scene=..., module_factory=...) 构造，
        内部转发到 runtime=RuntimeInjections(module_factory=...)。

        安全：拒绝从 dict 构造 runtime（防止 YAML 注入运行时对象）。
        runtime 只能通过 Python 代码传入 RuntimeInjections 实例。
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # shallow copy，不修改调用方传入的 dict
        # 提取工厂字段
        factory_kwargs: Dict[str, Any] = {}
        for f in _RUNTIME_FACTORY_FIELDS:
            if f in data:
                factory_kwargs[f] = data.pop(f)
        # runtime 字段不允许从 dict 构造（仅 Python 代码可注入 RuntimeInjections 实例）
        if "runtime" in data and not isinstance(data["runtime"], RuntimeInjections):
            raise _config_validation_error(
                "ExperimentConfig.runtime 不允许从 dict 构造，仅 Python 代码可注入 "
                "RuntimeInjections 实例。工厂字段请通过 module_factory/datamodule_factory/"
                "extra_callbacks/trainer_factory 传入。"
            )
        # 合并工厂字段到 runtime
        # Review 修复（2026-07-18）：合并时创建新 RuntimeInjections 实例，
        # 不修改用户传入的原实例（避免副作用：共享 ri 被构造污染）。
        if factory_kwargs:
            existing_runtime = data.get("runtime")
            if isinstance(existing_runtime, RuntimeInjections):
                # 复制原实例字段，再用 factory_kwargs 覆盖
                merged = RuntimeInjections(
                    module_factory=existing_runtime.module_factory,
                    datamodule_factory=existing_runtime.datamodule_factory,
                    extra_callbacks=list(existing_runtime.extra_callbacks),
                    trainer_factory=existing_runtime.trainer_factory,
                )
                for k, v in factory_kwargs.items():
                    setattr(merged, k, v)
                data["runtime"] = merged
            else:
                data["runtime"] = RuntimeInjections(**factory_kwargs)
        return data

    @field_validator("output_dir")
    @classmethod
    def _validate_output_dir(cls, v: str) -> str:
        if not v:
            raise ValueError("output_dir 不能为空")
        return v

    @field_validator("export_formats")
    @classmethod
    def _validate_export_formats(cls, v: List[str]) -> List[str]:
        valid_formats = {"state_dict", "torchscript", "onnx", "quantized_onnx"}
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid export_format '{fmt}'. Supported: {valid_formats}"
                )
        return v

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> Dict[str, Any]:
        """生成 JSON Schema，排除 runtime 字段（运行时注入对象，非声明式配置）。

        P2 演进：Field(exclude=True) 仅排除 model_dump()，不排除 model_json_schema()。
        此处覆盖 model_json_schema 移除 runtime 字段，使生成的 schema 仅含声明式配置。
        """
        schema = super().model_json_schema(**kwargs)
        props = schema.get("properties", {})
        props.pop("runtime", None)
        # 防御性移除 required 中的 runtime（如有）
        required = schema.get("required")
        if required and "runtime" in required:
            required.remove("runtime")
        return schema

    # ============================================================
    # 工厂字段属性代理（向后兼容所有访问点）
    # ============================================================
    # P2 演进：4 个 @property 代理让 cfg.module_factory 等价于 cfg.runtime.module_factory，
    # 所有 pipeline.py / hpo.py / autoaugment / nas 的访问点零改动。
    # setter 支持赋值（cfg.module_factory = ...）和 list 操作（cfg.extra_callbacks.append）。

    @property
    def module_factory(self) -> Optional[Any]:
        """代理到 runtime.module_factory（向后兼容）。"""
        return self.runtime.module_factory

    @module_factory.setter
    def module_factory(self, value: Optional[Any]) -> None:
        self.runtime.module_factory = value

    @property
    def datamodule_factory(self) -> Optional[Any]:
        """代理到 runtime.datamodule_factory（向后兼容）。"""
        return self.runtime.datamodule_factory

    @datamodule_factory.setter
    def datamodule_factory(self, value: Optional[Any]) -> None:
        self.runtime.datamodule_factory = value

    @property
    def extra_callbacks(self) -> List[Any]:
        """代理到 runtime.extra_callbacks（向后兼容）。

        返回的是 runtime.extra_callbacks 的引用，append/extend 操作直接生效。
        """
        return self.runtime.extra_callbacks

    @extra_callbacks.setter
    def extra_callbacks(self, value: List[Any]) -> None:
        self.runtime.extra_callbacks = value

    @property
    def trainer_factory(self) -> Optional[Any]:
        """代理到 runtime.trainer_factory（向后兼容）。"""
        return self.runtime.trainer_factory

    @trainer_factory.setter
    def trainer_factory(self, value: Optional[Any]) -> None:
        self.runtime.trainer_factory = value

    # ------------------------------------------------------------
    # pydantic v2 内部方法覆写：model_copy
    # ------------------------------------------------------------
    # 遗留问题 4 修复（2026-07-19）：pydantic v2 BaseModel.model_copy 绕过 __init__
    # 和 model_validator，直接写 __dict__：values = {**self.__dict__, **update} →
    # new_model.__dict__ = values。但工厂字段是 @property（data descriptor），
    # 描述符协议优先于 __dict__ 访问，导致 update={"module_factory": X} 后访问
    # new_config.module_factory 返回 self.runtime.module_factory（原值），
    # __dict__["module_factory"] 被遮蔽，更新静默失效。
    #
    # 修复策略：从 update 中提取工厂字段，调用 super().model_copy 处理剩余字段
    # （声明字段 + runtime 字段），然后通过 setattr 应用工厂字段——setattr 触发
    # @module_factory.setter，写入 self.runtime.module_factory，绕过 __dict__ 遮蔽。
    def model_copy(
        self,
        *,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "ExperimentConfig":
        """覆写 pydantic v2 model_copy，处理工厂字段代理。

        Args:
            update: 字段更新字典，可含：
                - 声明字段（如 epochs, learning_rate）→ 交给 super().model_copy
                - 工厂字段（module_factory/datamodule_factory/extra_callbacks/trainer_factory）
                  → 通过 setter 代理写入 runtime
                - runtime 字段（RuntimeInjections 实例）→ 交给 super().model_copy
                  （runtime 是 pydantic Field，非 @property，无遮蔽问题）
            deep: True 时深拷贝所有字段（含 runtime）。
                  super().model_copy(deep=True) 会深拷贝 runtime，
                  此后 setattr 写入的是深拷贝后的 runtime，不影响原 cfg。

        Returns:
            新 ExperimentConfig 实例，工厂字段已正确应用。

        Examples:
            # 浅拷贝 + 更新工厂字段
            new_cfg = cfg.model_copy(update={"module_factory": my_factory})
            assert new_cfg.module_factory is my_factory
            assert new_cfg.runtime.module_factory is my_factory

            # 深拷贝（runtime 也被深拷贝，不影响原 cfg）
            new_cfg = cfg.model_copy(deep=True)
            assert new_cfg.runtime is not cfg.runtime

            # 同时更新声明字段和工厂字段
            new_cfg = cfg.model_copy(update={
                "epochs": 200,
                "module_factory": my_factory,
            })
            assert new_cfg.trainer.epochs == 200
            assert new_cfg.module_factory is my_factory
        """
        update = dict(update) if update else {}
        # 提取工厂字段（不能传给 super().model_copy，会被 property 遮蔽）
        factory_updates: Dict[str, Any] = {}
        for f in _RUNTIME_FACTORY_FIELDS:
            if f in update:
                factory_updates[f] = update.pop(f)
        # 剩余字段（声明字段 + runtime）交给 pydantic 默认实现
        new_config = super().model_copy(
            update=update if update else None,
            deep=deep,
        )
        # 浅拷贝副作用防御：deep=False 时 new_config.runtime is self.runtime（共享），
        # 若此时通过 setattr 应用工厂字段，会修改原 cfg 的 runtime——违背 model_copy
        # 返回独立副本的语义。故在应用工厂字段前先复制 runtime 实例（浅复制字段值）。
        # deep=True 时 super().model_copy 已深拷贝 runtime，无需再复制。
        if factory_updates and not deep and new_config.runtime is self.runtime:
            original_runtime = new_config.runtime
            new_config.runtime = RuntimeInjections(
                module_factory=original_runtime.module_factory,
                datamodule_factory=original_runtime.datamodule_factory,
                extra_callbacks=list(original_runtime.extra_callbacks),
                trainer_factory=original_runtime.trainer_factory,
            )
        # 应用工厂字段（通过 setter 代理写入 runtime）
        for f, v in factory_updates.items():
            setattr(new_config, f, v)
        return new_config

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
            ConfigValidationError: 缺少必需字段或字段类型错误（同时是 ValueError 子类，
                现有 `except ValueError` 调用点无需改造；classify_error 能精确命中
                CONFIG_VALIDATION_ERROR 错误码）
            pydantic.ValidationError: 子配置（SceneConfig/TrainerConfig 等）字段校验失败
                （pydantic ValidationError 也是 ValueError 子类）
        """
        if not isinstance(d, dict):
            raise _config_validation_error(
                f"ExperimentConfig.from_dict 需要 dict，实际: {type(d).__name__}"
            )

        # P2 演进（2026-07-18）：extra="forbid" 显式检查。
        # from_dict 手动 cherry-pick 字段，绕过了 pydantic 的 extra="forbid" 检查。
        # 此处显式校验 YAML 顶层只含声明字段，捕获拼写错误（如 train→trainer, output→output_dir）。
        # P2 演进（工厂字段拆分）：工厂字段已从 model_fields 移除（拆到 RuntimeInjections），
        # 但仍需显式拒绝 YAML 中的工厂字段（它们是运行时对象，不可序列化）。
        _YAML_ALLOWED_KEYS = set(cls.model_fields.keys()) - {"runtime"}
        unknown_keys = set(d.keys()) - _YAML_ALLOWED_KEYS
        if unknown_keys:
            factory_in_yaml = unknown_keys & set(_RUNTIME_FACTORY_FIELDS)
            runtime_in_yaml = "runtime" in unknown_keys
            if factory_in_yaml:
                raise _config_validation_error(
                    f"ExperimentConfig 不允许在 YAML 中声明运行时工厂字段: "
                    f"{sorted(factory_in_yaml)}（这些字段仅 Python 代码可注入）"
                )
            if runtime_in_yaml:
                raise _config_validation_error(
                    "ExperimentConfig.runtime 不允许在 YAML 中声明"
                    "（运行时注入对象，仅 Python 代码可构造）"
                )
            raise _config_validation_error(
                f"ExperimentConfig 含未知字段: {sorted(unknown_keys)}。"
                f"常见拼写错误：train→trainer, output→output_dir, save→save_model, "
                f"features→input_features。完整字段列表见 model_json_schema()。"
            )

        # 必需字段检查
        for required in ("scene", "input_features", "output_features"):
            if required not in d:
                raise _config_validation_error(
                    f"ExperimentConfig 缺少必需字段: '{required}'"
                )

        # scene
        scene_dict = dict(d["scene"])
        # Phase 12.3：递归解析 task_spec 子字段（TaskSpec 是 dataclass，pydantic 无法自动转换）
        if "task_spec" in scene_dict and scene_dict["task_spec"] is not None:
            scene_dict["task_spec"] = _parse_dataclass(
                TaskSpecField, scene_dict["task_spec"]
            )
        # P5 P3-4 完整正交化：params dict → SceneParams 实例
        if "params" in scene_dict and scene_dict["params"] is not None:
            from ..core.params import SceneParams
            if isinstance(scene_dict["params"], dict):
                scene_dict["params"] = SceneParams.from_dict(scene_dict["params"])
            elif isinstance(scene_dict["params"], SceneParams):
                pass  # 已是 SceneParams，无需转换
        scene = _parse_dataclass(SceneConfig, scene_dict)

        # input_features
        input_features_raw = d["input_features"]
        if not isinstance(input_features_raw, list) or len(input_features_raw) == 0:
            raise _config_validation_error(
                "ExperimentConfig.input_features 必须是非空 list"
            )
        input_features = [_parse_dataclass(InputFeature, f) for f in input_features_raw]

        # output_features
        output_features_raw = d["output_features"]
        if not isinstance(output_features_raw, list) or len(output_features_raw) == 0:
            raise _config_validation_error(
                "ExperimentConfig.output_features 必须是非空 list"
            )
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
            strict_schema=d.get("strict_schema", False),
            # P2 修复：分布式训练字段从 YAML 顶层读取（config_schema.md 文档化）
            devices=d.get("devices", 1),
            strategy=d.get("strategy"),
            num_nodes=d.get("num_nodes", 1),
            sync_batchnorm=d.get("sync_batchnorm", False),
            num_processes=d.get("num_processes", 1),
        )

    # ------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------
    def validate(self) -> None:
        """递归校验所有子配置。

        向后兼容：pydantic 构造时已校验大部分字段，但 SceneConfig.data_root
        和 TaskSpec 需要显式校验（data_root 可能由 CLI/env 后填充）。
        input_features/output_features 非空检查保留在此处（允许构造时为空，
        由 validate() 显式触发），与旧 dataclass 行为一致。
        """
        self.scene.validate()
        if not self.input_features:
            raise ValueError("input_features 不能为空")
        if not self.output_features:
            raise ValueError("output_features 不能为空")
        # 子配置 validate() 为 no-op（pydantic 构造时已校验），保留调用以兼容
        for f in self.input_features:
            f.validate()
        for f in self.output_features:
            f.validate()
        self.trainer.validate()
        self.hpo.validate()

    # ------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """反向序列化为 dict（可写 YAML/JSON）。

        P2 演进：工厂字段已拆分到 RuntimeInjections dataclass，
        runtime 字段用 Field(exclude=True) 排除序列化，不出现在 to_dict 输出中。
        """
        return {
            "scene": _dataclass_to_dict(self.scene),
            "input_features": [_dataclass_to_dict(f) for f in self.input_features],
            "output_features": [_dataclass_to_dict(f) for f in self.output_features],
            "trainer": _dataclass_to_dict(self.trainer),
            "hpo": _dataclass_to_dict(self.hpo),
            "output_dir": self.output_dir,
            "save_model": self.save_model,
            "export_formats": self.export_formats,
            "strict_schema": self.strict_schema,
            # P2 修复：分布式训练字段
            "devices": self.devices,
            "strategy": self.strategy,
            "num_nodes": self.num_nodes,
            "sync_batchnorm": self.sync_batchnorm,
            "num_processes": self.num_processes,
        }


# ============================================================
# 内部工具：dict <-> 配置对象转换（兼容 dataclass 与 pydantic BaseModel）
# ============================================================
def _is_pydantic_model(cls) -> bool:
    """判断 cls 是否为 pydantic v2 BaseModel 子类。"""
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _parse_dataclass(cls, data: Any):
    """
    将 dict 转换为 cls 的实例（兼容 dataclass 与 pydantic BaseModel）。

    - pydantic BaseModel：委托 cls.model_validate(data)，extra='ignore' 已在
      model_config 中配置，自动忽略多余键
    - dataclass：仅取 cls 已定义的字段，忽略 dict 中多余键（前向兼容）
    - 缺失字段使用默认值
    - 嵌套结构不自动递归（本 schema 仅顶层 ExperimentConfig 含嵌套，
      已在 from_dict 中显式处理）
    """
    if _is_pydantic_model(cls):
        if not isinstance(data, dict):
            raise ValueError(f"解析 {cls.__name__} 需要 dict，实际: {type(data)}")
        return cls.model_validate(data)
    if not is_dataclass(cls):
        raise ValueError(f"_parse_dataclass 需要 dataclass 或 pydantic BaseModel 类型，实际: {cls}")
    if not isinstance(data, dict):
        raise ValueError(f"解析 {cls.__name__} 需要 dict，实际: {type(data)}")

    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> Any:
    """递归将配置对象转为 dict（兼容 dataclass 与 pydantic BaseModel，含嵌套与 list）。"""
    # P5 P3-4：SceneParams 用 to_flat_dict() 序列化（扁平结构，而非 asdict 嵌套）
    if hasattr(obj, "to_flat_dict") and callable(obj.to_flat_dict):
        return obj.to_flat_dict()
    # pydantic v2 BaseModel：用 model_dump() 序列化，再递归处理嵌套的
    # arbitrary_types_allowed 字段（如 TaskSpec dataclass / SceneParams）
    if isinstance(obj, BaseModel):
        return {k: _dataclass_to_dict(v) for k, v in obj.model_dump().items()}
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
