"""
RFC Phase D：可编程训练流程 — Stage Pipeline。

将 run_experiment 拆解为可重组的 stage，Agent 可：
- 调用单个 stage（如只做 preflight 预检）
- 替换 stage（如自定义 eval）
- 插入 pre/post hook
- 跳过 stage
- 编排自定义 pipeline

8 个顶层 stage（RFC 决策：不进一步拆解）：
1. validate   — 校验配置 schema + 场景注册
2. preflight  — 资源探测 + 路由 + 预检
3. load       — 加载数据 + 数据画像
4. resolve    — 解析 TaskSpec / FeatureSpec / 配置
5. build      — 构建模型 / DataModule / LightningModule
6. train      — 训练执行
7. eval       — 评估
8. export     — 导出

设计原则（RFC 原则 4）：
- 训练流程是可组合的 stage pipeline，不是黑盒函数
- 每个 stage 独立可调用，支持 pre/post hook、替换、跳过

向后兼容：run_experiment 保持不变，Pipeline 是并行入口供 Agent 使用。
"""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, field, fields as _dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
    runtime_checkable,
)

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import yaml

from ..config import DEFAULT_DATA_ROOT, ExperimentConfig
from ...observability import IncrementalLogWriter, Timer, setup_logging as _setup_logging
from ...observability_otel import (
    record_training_metric, record_trial_metric,
    ML_TRAIN_LOSS, ML_VAL_LOSS, ML_VAL_ACCURACY,
    ML_STAGE, ML_EPOCH, ML_MODEL_ID, ML_DATASET,
    ML_TRIAL_COUNT, ML_TRIAL_BEST_METRIC,
)
from ...routing import ResourceProbe, ResourceRouter
from ...schemas import TrainOutput
from ...scenes import get_scene, has_scene, list_scenes

from .preflight import set_seed, preflight_check, build_env_snapshot, build_logger
from .resolver import (
    experiment_config_to_dict,
    load_manifest_for_metadata,
    resolve_task_spec,
    resolve_feature_spec,
    validate_scene_capabilities,
)
from .errors import (
    classify_error,
    SceneNotRegisteredError,
    DatasetNotSupportedError,
    ModelNotSupportedError,
    ConfigValidationError,
)
# RFC-004 方案 G：训练产物溯源体系
from .artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    sha256_file,
    sha256_str,
    verify_artifacts as _verify_artifacts,
)

# RFC-003 DSP-1：TYPE_CHECKING 下导入实际类型用于注解，运行时不强制导入
if TYPE_CHECKING:
    from ..datamodule import GenericDataModule  # noqa: F401
    from ..module import GenericLightningModule  # noqa: F401
    from ...core.features import FeatureSpec  # noqa: F401
    from ...core.profiler import DataProfile  # noqa: F401
    from ...core.task import TaskSpec  # noqa: F401
    from ...observability import TrainingMonitor  # noqa: F401
    from ...routing import ResourceReport  # noqa: F401  (re-exported via routing)
    from ...scenes.base import DatasetBundle, SceneMeta  # noqa: F401

_logger = _setup_logging()


# ============================================================
# RFC-005 资源泄露修复：Lightning Logger / Trainer 清理辅助
# ============================================================

def _finalize_lightning_logger(logger: Any) -> None:
    """Finalize Lightning Logger（释放文件句柄/wandb 进程/SummaryWriter）。

    覆盖 CSVLogger / TensorBoardLogger / WandbLogger 三种后端：
    - WandbLogger: 调用 wandb.finish() 终止 run 进程 + 网络 socket
    - TensorBoardLogger: 关闭 SummaryWriter（events.out.tfevents 文件句柄 + 写入线程）
    - CSVLogger: 关闭 experiment（metrics.csv 文件句柄）
    """
    # 1. WandbLogger：调用 wandb.finish() 终止 run
    _cls_name = type(logger).__name__.lower()
    if "wandb" in _cls_name:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
        return

    # 2. 通用：调用 finalize(status)（Lightning Logger 协议）
    if hasattr(logger, "finalize"):
        try:
            logger.finalize("success")
        except Exception:
            pass

    # 3. 关闭 experiment（CSVLogger.experiment / TensorBoardLogger.experiment = SummaryWriter）
    try:
        exp = getattr(logger, "experiment", None)
        if exp is not None:
            # TensorBoard SummaryWriter 有 close()
            if hasattr(exp, "close"):
                exp.close()
            # CSV experiment 可能有 _fileio 等
            elif hasattr(exp, "flush"):
                exp.flush()
    except Exception:
        pass


# ============================================================
# RFC-003 DSP-1：PipelineContext 字段类型协议
# ============================================================
# Protocol 是结构子类型（structural subtyping），运行时定义不依赖实际类型。
# 现有 SceneContainer / nn.Module / pl.LightningDataModule / pl.Trainer 等
# 自动满足对应 Protocol，无需修改源码。
# @runtime_checkable 使 isinstance 检查可用（仅校验方法存在性，不校验签名）。


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


# ============================================================
# RFC-003 DSP-1：字段填充时机映射
# ============================================================
# 字段名 → 首次填充的 stage 名（基于附录 C 的 stage writes 实证）。
# 用于 filled_at() / schema() 的字段契约输出。
# "init" 表示构造函数注入；"agent" 表示 Agent 运行时控制。
_FIELD_FILL_STAGE: Dict[str, str] = {
    # init（构造函数注入）
    "config": "init",
    # stage_validate
    "scene": "stage_validate",
    "meta": "stage_validate",
    "model_id": "stage_validate",
    "dataset": "stage_validate",
    "learning_mode": "stage_validate",
    # stage_preflight
    "report": "stage_preflight",
    "route_level": "stage_preflight",
    "route_config": "stage_preflight",
    "output": "stage_preflight",
    # stage_resolve
    "scene_info": "stage_resolve",
    "num_classes": "stage_resolve",
    "task_spec": "stage_resolve",
    "feature_spec": "stage_resolve",
    "resolved": "stage_resolve",
    "lightning_params": "stage_resolve",
    "distributed_kwargs": "stage_resolve",
    # stage_load
    "scene_kwargs": "stage_load",
    "bundle": "stage_load",
    "data_profile": "stage_load",
    "output_dir": "stage_load",
    "log_writer": "stage_load",
    # stage_build
    "model": "stage_build",
    "datamodule": "stage_build",
    "module": "stage_build",
    "callbacks": "stage_build",
    "pl_logger": "stage_build",
    "csv_logger": "stage_build",
    "monitor": "stage_build",
    # stage_train
    "trainer": "stage_train",
    "training_duration_s": "stage_train",
    "best_model_path": "stage_train",
    "best_model_score": "stage_train",
    # stage_eval
    "final_eval": "stage_eval",
    "training_log": "stage_eval",
    "early_stopped": "stage_eval",
    "feedback": "stage_eval",
    # agent-controlled（RFC-002 探索状态 + 断点续跑）
    "trial_id": "agent",
    "parent_trial_id": "agent",
    "exploration_history": "agent",
    "extra": "agent",
    "completed_stages": "agent",
    "stage_checkpoint_path": "agent",
}


# ============================================================
# RFC-004 方案 C：training_log 字段结构契约
# ============================================================
# epoch_entry 的字段结构契约，生产者（module.py on_validation_epoch_end）
# 与消费者（analyze_training_result）必须对齐。为 DSP-4 TypedDict 化铺路。
_TRAINING_LOG_ENTRY_SCHEMA: Dict[str, Any] = {
    "epoch": "int",                          # epoch 序号（1-based）
    "lr": "Optional[float]",                 # 学习率
    "train_loss": "float",                   # 训练 loss
    "train_accuracy": "Optional[float]",     # 分类任务 train 指标
    "train_macro_f1": "Optional[float]",     # 分类任务 train 指标
    "val_loss": "float",                     # 验证 loss
    "val_accuracy": "Optional[float]",       # 分类任务 val 指标
    "val_macro_f1": "Optional[float]",       # 分类任务 val 指标
}


@dataclass
class PipelineContext:
    """Stage 间共享的上下文（Agent 可读取和修改）。

    每个 stage 接收 context，返回更新后的 context。
    Agent 可在 stage 间读取 context 状态做决策。

    RFC-002 阶段 J：新增探索状态字段，支持回溯与兼容性查询。
    RFC-003 DSP-1：字段类型强化为 Protocol / 具体类型（forward reference 字符串
    避免循环导入），新增 completed_fields / filled_at / schema / describe 内省方法。
    """
    config: ExperimentConfig
    # stage 间传递的状态
    scene: Optional["SceneProtocol"] = None
    meta: Optional["SceneMetaProtocol"] = None
    model_id: str = ""
    dataset: str = ""
    learning_mode: str = "supervised"
    num_classes: Optional[int] = None
    task_spec: Optional["TaskSpecProtocol"] = None
    feature_spec: Optional["FeatureSpecProtocol"] = None
    data_profile: Optional["DataProfile"] = None  # DataProfile（具体类型，DSP-4 强化）
    bundle: Optional["DatasetBundle"] = None
    model: Optional["ModelProtocol"] = None
    datamodule: Optional["DataModuleProtocol"] = None
    module: Optional["GenericLightningModule"] = None  # pl.LightningModule 子类
    trainer: Optional["TrainerProtocol"] = None
    output: Optional[TrainOutput] = None
    output_dir: Optional[Path] = None
    # 解析后的配置
    resolved: Dict[str, Any] = field(default_factory=dict)
    scene_kwargs: Dict[str, Any] = field(default_factory=dict)
    scene_info: Dict[str, Any] = field(default_factory=dict)
    lightning_params: Dict[str, Any] = field(default_factory=dict)
    distributed_kwargs: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[Any] = field(default_factory=list)  # pl.Callback 列表，类型异构
    pl_logger: Optional["LoggerProtocol"] = None  # P2.1: pl.Logger（LoggerProtocol 契约）
    log_writer: Optional["IncrementalLogWriter"] = None
    csv_logger: Optional["LoggerProtocol"] = None  # P2.1: pl.Logger 实例（LoggerProtocol 契约）
    report: Optional["ResourceReport"] = None
    route_level: str = ""
    route_config: Dict[str, Any] = field(default_factory=dict)
    # RFC-002 阶段 J：探索状态
    trial_id: str = ""                              # 当前试验 ID
    parent_trial_id: Optional[str] = None           # 父试验 ID（支持回溯）
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)  # 已探索策略组合
    # 训练结果（first-class 字段，替代 extra 中的框架内部值）
    training_duration_s: float = 0.0                # stage_train 写入
    best_model_path: Optional[str] = None           # stage_train 写入（从 ModelCheckpoint 读取）
    best_model_score: Optional[float] = None        # stage_train 写入
    final_eval: Dict[str, Any] = field(default_factory=dict)       # stage_eval 写入
    training_log: List[Any] = field(default_factory=list)          # stage_eval 写入
    early_stopped: bool = False                                     # stage_eval 写入
    feedback: Optional[Dict[str, Any]] = None                      # stage_eval 写入
    # 错误状态（Pipeline.run 异常时写入）
    failed_stage: Optional[str] = None
    failed_error: Optional[str] = None
    # 自由扩展字段（Agent 自由扩展区，框架代码不得写入）
    extra: Dict[str, Any] = field(default_factory=dict)
    # P1：stage 级断点续跑 — 记录已完成 stage + checkpoint 路径
    completed_stages: List[str] = field(default_factory=list)  # 已完成的 stage 名
    stage_checkpoint_path: Optional[Path] = None  # pipeline_checkpoint.json 路径
    # P2：训练实时监控
    monitor: Optional["TrainingMonitor"] = None  # TrainingMonitor 实例
    # RFC-004 方案 G：产物溯源注册表 — 各 stage 注册其产出的文件
    artifact_registry: List[ArtifactDescriptor] = field(default_factory=list)

    def get(self, key: str, default=None):
        """获取上下文属性（含 extra 字段）。"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)

    def set(self, key: str, value: Any):
        """设置上下文属性（自动路由到 extra 或字段）。"""
        if hasattr(self, key) and key != "extra":
            setattr(self, key, value)
        else:
            self.extra[key] = value

    # ============================================================
    # RFC-004 方案 G：产物溯源注册
    # ============================================================

    def register_artifact(
        self,
        name: str,
        path: Path,
        kind: str,
        producer_stage: str,
        content_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[ArtifactDescriptor]:
        """stage 调用此方法注册产物到 artifact_registry。

        幂等：同名产物已注册时跳过（返回 None）。
        文件不存在时跳过并 warning（不抛异常，避免 stage 失败）。

        Args:
            name: 逻辑名（如 "model_weights" / "training_log"）
            path: 产物文件路径（绝对路径或相对 output_dir）
            kind: 产物类型（model/metrics/config/log/profile/feedback/metadata）
            producer_stage: 生产者 stage 名（如 "stage_export"）
            content_schema: 内容契约（字段名/类型），可选

        Returns:
            注册成功的 ArtifactDescriptor，跳过时返回 None
        """
        # 幂等：同名产物已注册
        if any(a.name == name for a in self.artifact_registry):
            return None

        path = Path(path)
        if not path.exists():
            _logger.warning(
                f"register_artifact: file not found, skipping: {name}={path}"
            )
            return None

        # 相对路径存储（便于 output_dir 整体迁移）
        try:
            if self.output_dir is not None and path.is_absolute():
                rel_path = str(path.relative_to(self.output_dir))
            else:
                rel_path = str(path)
        except ValueError:
            # path 不在 output_dir 下，保留绝对路径
            rel_path = str(path)

        try:
            desc = ArtifactDescriptor(
                name=name,
                path=rel_path,
                kind=kind,
                producer_stage=producer_stage,
                content_hash=sha256_file(path),
                size_bytes=path.stat().st_size,
                content_schema=content_schema or {},
            )
            self.artifact_registry.append(desc)
            return desc
        except Exception as e:
            _logger.warning(f"register_artifact failed for {name}: {e}")
            return None

    def record_trial(self, strategy: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        """记录一次探索试验（RFC-002 阶段 J）。

        Args:
            strategy: 本次试验使用的策略组合（如 {"loss": "focal", "lr": 0.001}）
            result: 试验结果（如 {"val_accuracy": 0.85}），None 表示未完成
        """
        self.exploration_history.append({
            "trial_id": self.trial_id,
            "parent_trial_id": self.parent_trial_id,
            "strategy": strategy,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    # ============================================================
    # RFC-003 DSP-1：运行时内省协议
    # ============================================================

    def completed_fields(self) -> List[str]:
        """返回当前已填充（非 None）的字段名列表（RFC-003 DSP-1）。

        注意：default_factory 字段（list/dict）即使为空也算已填充（有默认容器）。
        本方法只考虑 None 判定，覆盖标量 / 对象字段。
        """
        return [f.name for f in _dataclass_fields(self) if getattr(self, f.name, None) is not None]

    def filled_at(self, stage_name: str) -> List[str]:
        """返回某 stage 应填充的字段名（RFC-003 DSP-1）。

        Args:
            stage_name: stage 名（如 "stage_validate" / "stage_load" / "init" / "agent"）

        Returns:
            该 stage 首次填充的字段名列表（按 _FIELD_FILL_STAGE 顺序）
        """
        return [k for k, v in _FIELD_FILL_STAGE.items() if v == stage_name]

    @classmethod
    def schema(cls) -> dict:
        """返回完整字段契约（RFC-003 DSP-1）。

        返回 JSON 可序列化 dict，含 schema_version 与每个字段的
        name / type / fill_stage / has_default 元信息。
        """
        return {
            "schema_version": "1.0.0",
            "fields": [
                {
                    "name": f.name,
                    "type": str(f.type) if hasattr(f, "type") else "Any",
                    "fill_stage": _FIELD_FILL_STAGE.get(f.name, "unknown"),
                    "has_default": (
                        f.default is not MISSING
                        or f.default_factory is not MISSING  # type: ignore[misc]
                    ),
                }
                for f in _dataclass_fields(cls)
            ],
        }

    def describe(self) -> dict:
        """返回运行时状态（RFC-003 DSP-1）。

        返回 JSON 可序列化 dict，含已填充字段名 / extra 自由键 /
        trial_id / 已完成 stage 名，供 Agent 与 introspect 模块消费。
        """
        return {
            "completed_fields": self.completed_fields(),
            "extra_keys": list(self.extra.keys()),
            "trial_id": self.trial_id,
            "completed_stages": list(self.completed_stages),
        }

    # ============================================================
    # RFC-004 方案 F：确定性资源生命周期
    # ============================================================
    # Pipeline 必须在结束（成功/失败/OOM）时显式释放大对象引用，
    # 不依赖 GC。资源生命周期与 Pipeline 生命周期绑定。
    # 保留可序列化结果字段（training_log/final_eval/best_model_path），
    # 释放不可序列化大对象（bundle/model/trainer/module）。

    _RESOURCE_FIELDS: Tuple[str, ...] = (
        "trainer", "module", "model", "datamodule",
        "bundle", "monitor", "log_writer",
        "pl_logger", "csv_logger", "scene", "meta", "data_profile",
    )

    def release_resources(self) -> None:
        """显式释放大对象引用（RFC-004 方案 F + RFC-005 资源泄露修复）。

        在 Pipeline.run() 的 finally 块中调用，确保成功/失败/OOM 路径
        都释放资源。释放后 ctx 仍保留可序列化结果字段，可供 Agent 读取。

        幂等：多次调用安全（已 None 的字段不会重复释放）。

        资源释放顺序（RFC-005 修复 10805 线程 / 249068 句柄泄露）：
        1. log_writer close（文件句柄）
        2. Logger finalize（CSVLogger/TBLogger/WandbLogger 的文件句柄/进程）
        3. Trainer _teardown（Lightning accelerator/strategy/loops 状态清理）
        4. DataModule teardown（清空缓存的 DataLoader 引用；worker 进程的 IPC pipe
           由 datamodule.py 模块级 patch _patched_shutdown_workers 在 iterator
           析构时关闭 — 这是 +24 pipe/run 泄露的根因）
        5. 模型 .cpu()（GPU 显存引用释放，打破循环引用）
        6. 置 None + 清空 callbacks
        7. CUDA synchronize + empty_cache
        8. gc.collect
        """
        # 1. log_writer 先 close（异常路径可能未关闭）
        if self.log_writer is not None:
            try:
                self.log_writer.close()
            except Exception:
                pass

        # 2. Logger finalize（CSVLogger/TensorBoardLogger/WandbLogger 持有文件句柄/wandb 进程）
        for _lg_field in ("pl_logger", "csv_logger"):
            _lg = getattr(self, _lg_field, None)
            if _lg is not None:
                try:
                    _finalize_lightning_logger(_lg)
                except Exception:
                    pass

        # 3. Trainer teardown（释放 accelerator/strategy/callbacks + 关闭 fit 阶段 DataLoader workers）
        # RFC-005：Lightning Trainer 只有私有 _teardown()（清理 accelerator/strategy/loops 状态），
        # 没有公开 teardown() 方法。之前调用 trainer.teardown(stage="fit") 会 AttributeError
        # 被 except 吞掉，导致清理从未执行。
        # _teardown() 不关闭 persistent_workers 的 IPC pipe（+24 pipe/run 泄露），
        # pipe 的关闭由 datamodule.py 模块级 patch 在 iterator 析构时负责。
        if self.trainer is not None:
            # 3a. 调用 Lightning 私有 _teardown（清理 accelerator/strategy/loops 状态）
            if hasattr(self.trainer, "_teardown"):
                try:
                    self.trainer._teardown()
                except Exception:
                    pass
            # 3b. 兼容：某些 Lightning 版本可能有 teardown 钩子
            if hasattr(self.trainer, "teardown"):
                try:
                    self.trainer.teardown(stage="fit")
                except Exception:
                    pass

        # 4. DataModule teardown（终止 persistent_workers 子进程）
        if self.datamodule is not None:
            try:
                self.datamodule.teardown(stage="fit")
            except Exception:
                pass

        # 5. 模型迁回 CPU（释放 GPU 显存引用，打破 Trainer↔Module↔Optimizer 循环引用）
        for _model_field in ("module", "model"):
            _m = getattr(self, _model_field, None)
            if _m is not None:
                try:
                    _m.cpu()
                except Exception:
                    pass

        # 6. 置 None + 清空 callbacks
        for name in self._RESOURCE_FIELDS:
            setattr(self, name, None)
        if self.callbacks:
            self.callbacks = []

        # 7. GPU 显存：empty_cache 前同步，确保异步操作完成
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

        # 8. 强制回收引用循环（Trainer/LightningModule 内部常见）
        import gc
        gc.collect()


# Stage 函数类型：接收 context，返回更新后的 context
StageFn = Callable[[PipelineContext], PipelineContext]


@dataclass
class StageResult:
    """Stage 执行结果。"""
    context: PipelineContext
    skipped: bool = False
    error: Optional[Exception] = None


@dataclass
class ReadinessReport:
    """Stage 就绪度报告（RFC-004 原则 9，advisory）。

    available=False 不阻断执行，仅提示 Agent 缺失的 reads 字段。
    """
    stage_name: str
    available: bool
    missing_reads: List[str]


@dataclass
class DanglingRef:
    """Dangling reference：stage 声明读取但无 stage 声明产出的字段（RFC-004 原则 9）。"""
    stage_name: str
    field_name: str
    reason: str


# ============================================================
# RFC-003 DSP-3：Stage IO 声明
# ============================================================

@dataclass
class FieldSpec:
    """字段规格（RFC-003 DSP-3）。

    声明 stage 读取/写入的字段名与类型，作为 stage IO 契约的最小单位。
    """
    name: str
    type: str = "Any"
    required: bool = True
    description: str = ""


@dataclass
class StageSpec:
    """Stage IO 规格（RFC-003 DSP-3）。

    声明 stage 的 name / reads / writes / description，
    由 @stage 装饰器附加到 stage 函数的 _stage_spec 属性。
    """
    name: str
    reads: List[FieldSpec] = field(default_factory=list)
    writes: List[FieldSpec] = field(default_factory=list)
    description: str = ""


def stage(name: str, reads: List[str], writes: List[str], description: str = ""):
    """Stage IO 声明装饰器（RFC-003 DSP-3）。

    将 StageSpec 附加为函数属性 `_stage_spec`，不改变函数运行时行为。

    Args:
        name: stage 名（如 "validate"）
        reads: 该 stage 读取的 PipelineContext 字段名列表
        writes: 该 stage 写入的 PipelineContext 字段名列表
        description: stage 用途说明

    Returns:
        装饰器函数，将被装饰函数原样返回（仅附加 _stage_spec 属性）
    """
    def decorator(fn: StageFn) -> StageFn:
        fn._stage_spec = StageSpec(
            name=name,
            reads=[FieldSpec(name=n) for n in reads],
            writes=[FieldSpec(name=n) for n in writes],
            description=description,
        )
        return fn
    return decorator


@stage(
    name="validate",
    reads=["config"],
    writes=["scene", "meta", "model_id", "dataset", "learning_mode"],
    description="Stage 1: 校验配置 schema + 场景注册",
)
def stage_validate(ctx: PipelineContext) -> PipelineContext:
    """Stage 1: 校验配置 schema + 场景注册。"""
    ctx.config.validate()

    if not has_scene(ctx.config.scene.name):
        raise SceneNotRegisteredError(
            f"Scene '{ctx.config.scene.name}' not registered. "
            f"Available: {list(list_scenes().keys())}"
        )

    ctx.scene = get_scene(ctx.config.scene.name)
    ctx.meta = ctx.scene.meta()
    ctx.model_id = ctx.config.scene.model_id
    ctx.dataset = ctx.config.scene.dataset
    ctx.learning_mode = ctx.config.scene.learning_mode

    if not ctx.meta.is_dynamic_dataset:
        if ctx.dataset not in ctx.meta.supported_datasets:
            raise DatasetNotSupportedError(
                f"Dataset '{ctx.dataset}' not supported by scene '{ctx.config.scene.name}'. "
                f"Supported: {ctx.meta.supported_datasets}"
            )
    if ctx.model_id not in ctx.meta.supported_models:
        raise ModelNotSupportedError(
            f"Model '{ctx.model_id}' not supported by scene '{ctx.config.scene.name}'. "
            f"Supported: {ctx.meta.supported_models}"
        )

    return ctx


@stage(
    name="preflight",
    reads=["config", "model_id", "dataset", "learning_mode"],  # P2.1: 对齐函数体真实读取
    writes=["report", "route_level", "route_config", "output"],
    description="Stage 2: 资源探测 + 路由 + 预检",
)
def stage_preflight(ctx: PipelineContext) -> PipelineContext:
    """Stage 2: 资源探测 + 路由 + 预检。"""
    # GPU 隔离
    gpu = ctx.config.trainer.gpu
    if gpu is None and ctx.config.scene.params:
        gpu = ctx.config.scene.params.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # 随机种子
    deterministic = ctx.config.trainer.deterministic
    set_seed(ctx.config.trainer.seed, deterministic=deterministic)

    # 资源探测 + 路由
    ctx.report = ResourceProbe.probe()
    ctx.route_level = ResourceRouter.route(ctx.report)
    ctx.route_config = ResourceRouter.get_route_config(ctx.route_level)

    # 初始化 TrainOutput
    ctx.output = TrainOutput(
        status="error",
        model_id=ctx.model_id,
        dataset=ctx.dataset,
        learning_mode=ctx.learning_mode,
    )
    ctx.output.resource = ctx.report.to_dict()
    ctx.output.route_config = {"route_level": ctx.route_level, **ctx.route_config}

    return ctx


@stage(
    name="resolve",
    reads=["config", "scene", "dataset", "num_classes", "data_profile",
           "scene_kwargs", "report", "route_config", "meta", "model_id"],
    writes=["scene_info", "num_classes", "task_spec", "feature_spec",
            "resolved", "lightning_params", "distributed_kwargs"],
    description="Stage 4: 解析 TaskSpec / FeatureSpec / 最终配置",
)
def stage_resolve(ctx: PipelineContext) -> PipelineContext:
    """Stage 4: 解析 TaskSpec / FeatureSpec / 最终配置。"""
    # scene_kwargs 由 stage_load 前置填充
    # 获取 num_classes
    ctx.scene_info = ctx.scene.get_dataset_info(ctx.dataset, **ctx.scene_kwargs)
    ctx.num_classes = ctx.scene_info["num_classes"]

    # 自监督模式特殊处理
    is_self_supervised = (ctx.learning_mode == "self_supervised")
    if is_self_supervised:
        from ...registry import get_dataset_spec, is_dataset_registered
        spec = get_dataset_spec(ctx.dataset) if is_dataset_registered(ctx.dataset) else None
        supervised_source = spec.supervised_source if spec else ""
        if not supervised_source:
            if ctx.dataset != "NTU-Fi_HAR":
                raise ConfigValidationError(
                    f"Self-supervised mode requires dataset with supervised_source, "
                    f"got '{ctx.dataset}' (no supervised_source defined)."
                )
            supervised_source = "NTU-Fi-HumanID"
        from ...registry import get_dataset_spec as _gds
        src_spec = _gds(supervised_source)
        ctx.num_classes = src_spec.num_classes

    # 解析 TaskSpec（支持数据画像推断）
    ctx.task_spec = resolve_task_spec(
        ctx.config, ctx.scene, ctx.dataset, ctx.model_id,
        ctx.num_classes, scene_kwargs=ctx.scene_kwargs,
        data_profile=ctx.data_profile,
    )

    # 解析 FeatureSpec
    ctx.feature_spec = resolve_feature_spec(
        ctx.config, ctx.scene, ctx.dataset, scene_kwargs=ctx.scene_kwargs,
    )

    # 校验场景能力
    validate_scene_capabilities(ctx.meta, ctx.task_spec, ctx.learning_mode, ctx.config.scene.name)

    # 解析最终配置
    config_dict = experiment_config_to_dict(ctx.config)
    model_info = ctx.scene.get_model_info(ctx.model_id)
    preflight_check(
        config_dict, model_info, ctx.report, ctx.dataset,
        scene_name=ctx.config.scene.name,
        scene_params=ctx.config.scene.params,
    )
    ctx.resolved = ResourceRouter.resolve_config(config_dict, ctx.route_config, model_info, ctx.report)
    ctx.resolved["deterministic"] = ctx.config.trainer.deterministic

    # Lightning Trainer 参数
    ctx.lightning_params = ResourceRouter.to_lightning_params(ctx.resolved)

    # 分布式训练参数
    ctx.distributed_kwargs = {}
    if "strategy" in ctx.lightning_params:
        ctx.distributed_kwargs["strategy"] = ctx.lightning_params["strategy"]
    if "num_nodes" in ctx.lightning_params:
        ctx.distributed_kwargs["num_nodes"] = ctx.lightning_params["num_nodes"]
    if "sync_batchnorm" in ctx.lightning_params:
        ctx.distributed_kwargs["sync_batchnorm"] = ctx.lightning_params["sync_batchnorm"]

    return ctx


@stage(
    name="load",
    reads=["config", "scene", "dataset", "learning_mode", "output"],
    writes=["scene_kwargs", "bundle", "data_profile", "output_dir", "log_writer"],
    description="Stage 3: 加载数据 + 数据画像",
)
def stage_load(ctx: PipelineContext) -> PipelineContext:
    """Stage 3: 加载数据 + 数据画像。"""
    from ...core.profiler import DataProfiler

    # scene_kwargs 前置计算（供 load_dataset 使用，也供后续 resolve 读取）
    ctx.scene_kwargs = {"params": ctx.config.scene.params} if ctx.config.scene.params else {}

    data_root = ctx.config.scene.data_root or DEFAULT_DATA_ROOT

    ctx.bundle = ctx.scene.load_dataset(
        ctx.dataset, data_root, learning_mode=ctx.learning_mode,
        **ctx.scene_kwargs,
    )

    # 创建输出目录（在数据画像前，便于落盘）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    ctx.output_dir = Path(ctx.config.output_dir) / f"{ctx.model_id}_{ctx.dataset}_{timestamp}_{pid}"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.output:
        ctx.output.output_dir = str(ctx.output_dir)

    # 数据画像（落盘到 output_dir）
    try:
        profiler = DataProfiler(max_samples=500)
        ctx.data_profile = profiler.profile_bundle(ctx.bundle, dataset_name=ctx.dataset)
        if ctx.data_profile is not None:
            profile_path = ctx.output_dir / "data_profile.json"
            ctx.data_profile.save(profile_path)
            # RFC-004 方案 G：注册 data_profile 产物
            ctx.register_artifact(
                "data_profile", profile_path,
                kind="profile", producer_stage="stage_load",
                content_schema={"n_samples": int, "input_shape": list,
                                "class_distribution": dict},
            )
    except Exception:
        ctx.data_profile = None

    # 增量日志写入器
    ctx.log_writer = IncrementalLogWriter(ctx.output_dir / "training_log.jsonl")

    return ctx


@stage(
    name="build",
    reads=["config", "scene", "model_id", "dataset", "num_classes", "feature_spec",
           "bundle", "task_spec", "resolved", "output_dir", "scene_info",
           "route_config", "log_writer", "learning_mode"],
    writes=["model", "datamodule", "module", "callbacks", "pl_logger", "csv_logger", "monitor"],
    description="Stage 5: 构建模型 / DataModule / LightningModule",
)
def stage_build(ctx: PipelineContext) -> PipelineContext:
    """Stage 5: 构建模型 / DataModule / LightningModule。"""
    from ...engine.datamodule import GenericDataModule
    from ...engine.module import GenericLightningModule
    from ...engine.self_supervised import SelfSupervisedModule

    is_self_supervised = (ctx.learning_mode == "self_supervised")
    data_root = ctx.config.scene.data_root or DEFAULT_DATA_ROOT

    # 构建模型
    ctx.model = ctx.scene.build_model_for_dataset(
        ctx.model_id, ctx.dataset, ctx.num_classes,
        learning_mode=ctx.learning_mode,
        data_root=data_root,
        input_dim=ctx.feature_spec.feature_dim or ctx.scene_info.get("n_features"),
        feature_spec=ctx.feature_spec,
        **ctx.scene_kwargs,
    )

    # metrics
    if ctx.config.scene.task_spec is not None:
        metrics = ctx.task_spec.effective_metrics
    else:
        metrics = ctx.resolved.get("metrics", ["accuracy", "macro_f1"])

    # Logger
    logger_type = ctx.resolved.get("logger", "csv")
    ctx.pl_logger = build_logger(logger_type, ctx.output_dir, ctx.model_id, ctx.dataset)
    ctx.csv_logger = ctx.pl_logger

    # Callbacks
    ctx.callbacks = []
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ctx.output_dir / "checkpoints"),
        filename="best-{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    ctx.callbacks.append(ckpt_cb)

    early_stopping_patience = ctx.config.trainer.early_stopping
    if early_stopping_patience is not None:
        # RFC-004 方案 E：使用 min_delta 避免微小波动误触发早停
        early_stopping_min_delta = getattr(
            ctx.config.trainer, "early_stopping_min_delta", 0.0
        )
        ctx.callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
        ))

    # P2: 创建 TrainingMonitor，供 EpochLogCallback 写入实时指标
    from ...observability import TrainingMonitor
    ctx.monitor = TrainingMonitor()

    from .orchestrator import EpochLogCallback
    ctx.callbacks.append(EpochLogCallback(log_every_n=10, monitor=ctx.monitor))

    if ctx.config.extra_callbacks:
        ctx.callbacks.extend(ctx.config.extra_callbacks)

    if is_self_supervised:
        # 自监督模式
        unsup_ds = ctx.bundle.unsupervised
        sup_ds = ctx.bundle.supervised_finetune
        test_ds = ctx.bundle.test
        transform_cfg = ctx.scene.get_transforms(ctx.dataset)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=sup_ds, test_dataset=test_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )
        else:
            ctx.datamodule = GenericDataModule(
                train_dataset=sup_ds, test_dataset=test_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )

        ctx.module = SelfSupervisedModule(
            model=ctx.model,
            learning_rate=ctx.resolved["learning_rate"],
            weight_decay=ctx.resolved["weight_decay"],
            metrics=metrics,
            num_classes=ctx.num_classes,
            incremental_log_writer=ctx.log_writer,
        )
    else:
        # 监督模式
        train_ds = ctx.bundle.train
        test_ds = ctx.bundle.test
        transform_cfg = ctx.scene.get_transforms(ctx.dataset, **ctx.scene_kwargs)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=train_ds, test_dataset=test_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="supervised",
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )
        else:
            ctx.datamodule = GenericDataModule(
                train_dataset=train_ds, test_dataset=test_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="supervised",
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )

        epochs = ctx.config.trainer.epochs
        max_epochs = ctx.route_config.get("max_epochs", float("inf"))
        if epochs > max_epochs:
            epochs = max_epochs

        if ctx.config.module_factory is not None:
            ctx.module = ctx.config.module_factory(
                model=ctx.model,
                learning_rate=ctx.resolved["learning_rate"],
                metrics=metrics,
                num_classes=ctx.num_classes,
                optimizer=ctx.resolved["optimizer"],
                weight_decay=ctx.resolved["weight_decay"],
                scheduler=ctx.resolved["scheduler"],
                max_epochs=epochs,
                incremental_log_writer=ctx.log_writer,
                task_spec=ctx.task_spec,
            )
        else:
            ctx.module = GenericLightningModule(
                model=ctx.model,
                learning_rate=ctx.resolved["learning_rate"],
                metrics=metrics,
                num_classes=ctx.num_classes,
                optimizer=ctx.resolved["optimizer"],
                weight_decay=ctx.resolved["weight_decay"],
                scheduler=ctx.resolved["scheduler"],
                max_epochs=epochs,
                incremental_log_writer=ctx.log_writer,
                task_spec=ctx.task_spec,
            )

    return ctx


# ============================================================
# P3: OOM 回退辅助
# ============================================================
def _is_oom_error(exc: Exception) -> bool:
    """判断异常是否为 CUDA/内存 OOM。"""
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", type(None))):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            return True
    return False


def _fit_with_oom_fallback(
    ctx: PipelineContext,
    build_trainer: Callable[[], "pl.Trainer"],
    fit_fn: Callable[["pl.Trainer"], None],
    *,
    min_batch_size: int = 4,
) -> "pl.Trainer":
    """执行 trainer.fit() 并在 OOM 时自动减半 batch_size 重试一次。

    P3: 闭环 OOM 恢复——Agent 选的 batch_size 可能超出显存，
    框架自动降级而非直接失败，减少 Agent 重试往返。

    Args:
        ctx: PipelineContext（读取/写入 resolved["batch_size"] 和 datamodule.batch_size）
        build_trainer: 无参 callable，返回新的 Trainer 实例（重试时重新调用）
        fit_fn: 接收 trainer 的 callable，内部调用 trainer.fit(...)；
                重试时重新调用，应每次重新获取 dataloader 以反映新 batch_size
        min_batch_size: 最小 batch_size，低于此值不再重试

    Returns:
        成功完成 fit 的 Trainer 实例
    """
    import logging
    _log = logging.getLogger(__name__)

    trainer = build_trainer()
    try:
        fit_fn(trainer)
        return trainer
    except Exception as e:
        if not _is_oom_error(e):
            raise
        current_bs = ctx.resolved.get("batch_size", 64)
        if current_bs <= min_batch_size:
            _log.warning(
                f"OOM at batch_size={current_bs} (<= min {min_batch_size}), not retrying"
            )
            raise
        new_bs = max(min_batch_size, current_bs // 2)
        _log.warning(
            f"OOM at batch_size={current_bs}, retrying with batch_size={new_bs}"
        )
        ctx.resolved["batch_size"] = new_bs
        # 更新 datamodule batch_size（Lightning DataModule 在 fit 时重新调用 dataloader 方法）
        if hasattr(ctx.datamodule, "batch_size"):
            ctx.datamodule.batch_size = new_bs
        # RFC-005：清理旧 Trainer（_teardown + del + CUDA 同步），避免残留 worker/显存泄露
        # 注意：用 _teardown()（Lightning 私有），teardown() 不存在会静默失败
        if hasattr(trainer, "_teardown"):
            try:
                trainer._teardown()
            except Exception:
                pass
        del trainer
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        # 重建 trainer（旧实例已 teardown）
        trainer = build_trainer()
        fit_fn(trainer)
        return trainer


@stage(
    name="train",
    reads=["config", "model", "datamodule", "module", "callbacks",
           "lightning_params", "pl_logger", "csv_logger", "resolved",
           "route_config", "distributed_kwargs", "learning_mode"],
    writes=["trainer"],
    description="Stage 6: 训练执行",
)
def stage_train(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: 训练执行。

    RFC-002 阶段 K：支持 trainer_factory 注入，Agent 可自定义 Trainer 构造。
    """
    is_self_supervised = (ctx.learning_mode == "self_supervised")
    deterministic = ctx.config.trainer.deterministic
    enable_progress_bar = ctx.config.trainer.enable_progress_bar
    max_time = ctx.config.trainer.max_time or "00:02:00:00"

    # checkpoint 恢复
    resume_ckpt = ctx.config.trainer.resume
    if resume_ckpt is None and ctx.config.scene.params:
        resume_ckpt = ctx.config.scene.params.get("resume")

    timer = Timer("training")
    timer.__enter__()

    # RFC-002 阶段 K：Trainer 构造参数
    def _build_trainer_kwargs(**overrides):
        kwargs = {
            "accelerator": ctx.lightning_params["accelerator"],
            "devices": ctx.lightning_params["devices"],
            "precision": ctx.lightning_params["precision"],
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": False,
            "deterministic": deterministic,
            "max_time": max_time,
            "gradient_clip_val": ctx.resolved.get("gradient_clip_val"),
            "gradient_clip_algorithm": ctx.resolved.get("gradient_clip_algorithm", "norm"),
            "accumulate_grad_batches": ctx.resolved.get("accumulate_grad_batches", 1),
            **ctx.distributed_kwargs,
        }
        kwargs.update(overrides)
        return kwargs

    if is_self_supervised:
        ss_epochs = ctx.resolved.get("self_supervised_epochs", 100)
        sup_epochs = ctx.config.trainer.epochs

        # Phase 1: 自监督预训练（P3: OOM 回退）
        ctx.module.phase = "self_supervised"
        def _build_ss_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=ss_epochs,
                    logger=ctx.csv_logger,
                    enable_checkpointing=False,
                    limit_val_batches=0,
                    **_build_trainer_kwargs(),
                )
            return pl.Trainer(
                max_epochs=ss_epochs,
                logger=ctx.csv_logger,
                enable_checkpointing=False,
                limit_val_batches=0,
                **_build_trainer_kwargs(),
            )
        def _fit_ss(trainer):
            # 每次重新获取 dataloader，OOM 重试时反映新 batch_size
            trainer.fit(ctx.module, train_dataloaders=ctx.datamodule.train_dataloader())
        # RFC-005：存 SS Phase 1 Trainer 返回值，fit 后显式 _teardown 释放
        ss_trainer = _fit_with_oom_fallback(ctx, _build_ss_trainer, _fit_ss)
        if hasattr(ss_trainer, "_teardown"):
            try:
                ss_trainer._teardown()
            except Exception:
                pass
        del ss_trainer

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

        # Phase 2: 监督微调（P3: OOM 回退）
        ctx.module.phase = "supervised"
        ctx.module._current_epoch_loss = 0.0
        ctx.module._current_epoch_steps = 0
        def _build_sup_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=sup_epochs,
                    callbacks=ctx.callbacks,
                    logger=ctx.csv_logger,
                    enable_checkpointing=True,
                    **_build_trainer_kwargs(),
                )
            return pl.Trainer(
                max_epochs=sup_epochs,
                callbacks=ctx.callbacks,
                logger=ctx.csv_logger,
                enable_checkpointing=True,
                **_build_trainer_kwargs(),
            )
        def _fit_sup(trainer):
            trainer.fit(
                ctx.module,
                train_dataloaders=ctx.datamodule.supervised_dataloader(),
                val_dataloaders=ctx.datamodule.val_dataloader(),
                ckpt_path=resume_ckpt,
            )
        ctx.trainer = _fit_with_oom_fallback(ctx, _build_sup_trainer, _fit_sup)
    else:
        epochs = ctx.config.trainer.epochs
        max_epochs = ctx.route_config.get("max_epochs", float("inf"))
        if epochs > max_epochs:
            epochs = max_epochs

        # P3: OOM 回退——trainer 构造与 fit 分离，便于 OOM 时重建重试
        def _build_supervised_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=epochs,
                    callbacks=ctx.callbacks,
                    logger=ctx.csv_logger,
                    enable_checkpointing=True,
                    **_build_trainer_kwargs(),
                )
            return pl.Trainer(
                max_epochs=epochs,
                callbacks=ctx.callbacks,
                logger=ctx.csv_logger,
                enable_checkpointing=True,
                **_build_trainer_kwargs(),
            )

        def _fit_supervised(trainer):
            trainer.fit(ctx.module, datamodule=ctx.datamodule, ckpt_path=resume_ckpt)

        ctx.trainer = _fit_with_oom_fallback(ctx, _build_supervised_trainer, _fit_supervised)

    # 训练结束：停止计时器 + 提取 checkpoint 信息到 first-class 字段
    timer.__exit__()
    ctx.training_duration_s = round(timer.elapsed, 2)
    for cb in ctx.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ctx.best_model_path = cb.best_model_path or None
            ctx.best_model_score = float(cb.best_model_score) if cb.best_model_score is not None else None
            break

    return ctx


def analyze_training_result(
    final_eval: Dict[str, Any],
    training_log: List[Any],
    early_stopped: bool,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """分析训练结果，输出结构化反馈（RFC-002 阶段 L）。

    闭合探索-反馈回路：eval 结果 → 失败分类 + 改进建议 → Agent 调整策略。

    Returns:
        {"status", "diagnosis", "suggestions"}
    """
    import math

    # 1. 数值不稳定：指标含 NaN/Inf
    for k, v in final_eval.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return {
                "status": "numerical_instability",
                "diagnosis": f"指标 {k} 包含 NaN/Inf，训练数值不稳定",
                "suggestions": [
                    "降低 learning_rate",
                    "检查 loss 函数数值稳定性（可用 numerical_stability_validator）",
                    "启用梯度裁剪 (gradient_clip_val)",
                    "检查输入数据是否已归一化",
                ],
            }

    # 2. 从 training_log 提取末轮 train/val metric
    last_train_acc = None
    last_val_acc = None
    for entry in reversed(training_log) if isinstance(training_log, list) else []:
        if not isinstance(entry, dict):
            continue
        if last_train_acc is None:
            last_train_acc = entry.get("train_accuracy") or entry.get("train_acc")
        if last_val_acc is None:
            last_val_acc = entry.get("val_accuracy") or entry.get("val_acc")
        if last_train_acc is not None and last_val_acc is not None:
            break

    val_acc = final_eval.get("val_accuracy") or final_eval.get("accuracy") or last_val_acc

    # 3. 欠拟合：验证准确率过低
    if val_acc is not None and val_acc < 0.5 and task_type == "classification":
        return {
            "status": "underfitting",
            "diagnosis": f"验证准确率 {val_acc:.3f} 偏低，模型欠拟合",
            "suggestions": [
                "增大模型容量（更多层/更宽）",
                "增加训练轮数 (epochs)",
                "降低正则化强度 (weight_decay)",
                "尝试更丰富的特征工程 pipeline",
            ],
        }

    # 4. 过拟合：train-val gap 过大
    if last_train_acc is not None and last_val_acc is not None:
        gap = last_train_acc - last_val_acc
        if gap > 0.15:
            return {
                "status": "overfitting",
                "diagnosis": f"train-val gap {gap:.3f}（train={last_train_acc:.3f}, val={last_val_acc:.3f}），模型过拟合",
                "suggestions": [
                    "增加数据增强 (params.transform.augment)",
                    "增大 weight_decay",
                    "启用/增加 dropout",
                    "减小模型容量",
                    "启用 early_stopping",
                ],
            }

    # 5. 已收敛：早停
    if early_stopped:
        return {
            "status": "converged",
            "diagnosis": "训练早停，模型已收敛",
            "suggestions": [
                "尝试更激进的策略（更大 lr、不同 loss）",
                "尝试不同的特征工程 pipeline（见 catalog.suggest_pipeline）",
                "探索兼容性矩阵中的其他组合",
            ],
        }

    # 6. 正常完成
    return {
        "status": "success",
        "diagnosis": "训练正常完成",
        "suggestions": [
            "记录当前策略到技能库供复用 (save_skill)",
            "探索 ExplorationTracker.recommend_next 推荐的下一步",
        ],
    }


@stage(
    name="eval",
    reads=["config", "trainer", "module", "datamodule",
           "task_spec", "exploration_history", "learning_mode"],
    writes=["output", "exploration_history"],  # P2.1: 对齐函数体（写 exploration_history.feedback）
    description="Stage 7: 评估",
)
def stage_eval(ctx: PipelineContext) -> PipelineContext:
    """Stage 7: 评估。

    RFC-002 阶段 L：输出结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路。
    """
    is_self_supervised = (ctx.learning_mode == "self_supervised")

    ctx.module._is_final_validation = True
    if is_self_supervised:
        ctx.trainer.validate(ctx.module, dataloaders=ctx.datamodule.val_dataloader())
    else:
        ctx.trainer.validate(ctx.module, datamodule=ctx.datamodule)

    # 收集结果
    final_eval = ctx.module.get_final_metrics()
    training_log = ctx.module.training_log
    early_stopped = any(
        isinstance(cb, EarlyStopping) and cb.stopped_epoch >= 0
        for cb in ctx.trainer.callbacks
    )

    # 保存结果到 first-class 字段
    ctx.final_eval = final_eval
    ctx.training_log = training_log
    ctx.early_stopped = early_stopped

    # RFC-002 阶段 L：结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路
    task_type = ctx.task_spec.task_type if ctx.task_spec else "classification"
    ctx.feedback = analyze_training_result(
        final_eval, training_log, early_stopped, task_type=task_type,
    )

    # RFC-002 阶段 R：feedback 回写到最近一次探索试验，闭合"训练→反馈→推荐"回路
    # recommend_next 将基于此 feedback 调整优先级
    if ctx.exploration_history:
        feedback = ctx.feedback
        last_trial = ctx.exploration_history[-1]
        last_trial["feedback"] = feedback
        last_trial["result"] = {
            k: v for k, v in final_eval.items()
            if isinstance(v, (int, float, str)) or v is None
        }
        last_trial["status"] = "completed"

    # P0.2: OBP 评估指标埋点（OTel 未初始化时 no-op）
    _val_acc = final_eval.get("val_accuracy") or final_eval.get("val_acc")
    _val_loss = final_eval.get("val_loss")
    if _val_acc is not None:
        record_training_metric(ML_VAL_ACCURACY, value=float(_val_acc),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    if _val_loss is not None:
        record_training_metric(ML_VAL_LOSS, value=float(_val_loss),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    # 记录 trial count
    record_trial_metric(
        ML_TRIAL_COUNT, value=len(ctx.exploration_history),
        trial_id=ctx.trial_id,
    )

    return ctx


@stage(
    name="export",
    reads=["config", "model", "module", "output", "output_dir",
           "scene", "scene_info", "scene_kwargs", "meta", "report",
           "route_level", "task_spec", "feature_spec", "resolved",
           "log_writer", "exploration_history", "num_classes",
           "model_id", "dataset", "learning_mode"],
    writes=["output"],
    description="Stage 8: 导出",
)
def stage_export(ctx: PipelineContext) -> PipelineContext:
    """Stage 8: 导出。"""
    final_eval = ctx.final_eval
    training_log = ctx.training_log
    early_stopped = ctx.early_stopped

    # 保存模型 + metadata
    model_path = None
    if ctx.config.save_model:
        model_path = ctx.output_dir / "model.pth"
        torch.save(ctx.model.state_dict(), model_path)
        model_path = str(model_path)

        normalization_info = ctx.scene.get_normalization_info(ctx.dataset, **ctx.scene_kwargs)
        label_map = {}
        manifest_info = None
        if ctx.meta.is_dynamic_dataset:
            manifest_info = ctx.scene.get_manifest_info(ctx.dataset, **ctx.scene_kwargs)
            if manifest_info is not None:
                try:
                    manifest = load_manifest_for_metadata(ctx.config.scene.params)
                    label_map = manifest.label_map
                except Exception:
                    pass

        metadata = {
            "model_id": ctx.model_id,
            "dataset": ctx.dataset,
            "learning_mode": ctx.learning_mode,
            "num_classes": ctx.num_classes,
            "input_shape": list(ctx.scene_info.get("input_shape", [])),
            "normalization": normalization_info,
            "label_map": {str(k): v for k, v in label_map.items()},
            "manifest": manifest_info,
            "config": ctx.resolved,
            "metrics": list(final_eval.keys()),
            "final_eval": final_eval,
            "env": build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed}),
            "resource": ctx.report.to_dict(),
            "route_level": ctx.route_level,
            "task_spec": ctx.task_spec.to_dict(),
            "feature_spec": ctx.feature_spec.to_dict(),
            "created_at": datetime.now().isoformat(),
        }
        (ctx.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (ctx.output_dir / "config.yaml").write_text(
            yaml.dump(ctx.resolved, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    # 构建 TrainOutput
    if ctx.output:
        ctx.output.status = "success"
        ctx.output.training = {
            "epochs_trained": len(training_log),
            "early_stopped": early_stopped,
            "log": training_log,
            "duration_s": ctx.training_duration_s,
            "best_val_loss": ctx.best_model_score,
            "best_checkpoint": ctx.best_model_path,
        }
        ctx.output.final_eval = final_eval
        ctx.output.model_path = model_path
        ctx.output.env_snapshot = build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed})

    # 可选多格式导出
    export_formats = getattr(ctx.config, "export_formats", None)
    if export_formats and model_path:
        try:
            from ...export import export_model
            export_dir = ctx.output_dir / "exports"
            export_result = export_model(
                model=ctx.model,
                output_dir=export_dir,
                formats=export_formats,
                input_shape=list(ctx.scene_info.get("input_shape", [])),
                metadata={
                    "model_id": ctx.model_id,
                    "dataset": ctx.dataset,
                    "learning_mode": ctx.learning_mode,
                    "num_classes": ctx.num_classes,
                    "final_eval": final_eval,
                },
            )
            if ctx.output:
                ctx.output.export = export_result.to_dict()
        except Exception as e:
            if ctx.output:
                ctx.output.export = {"error": str(e)}

    # 关闭日志写入器（release_resources 也会关闭，此处保留双保险）
    if ctx.log_writer is not None:
        try:
            ctx.log_writer.close()
        except Exception:
            pass

    # 清理显存
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

    # RFC-002 阶段 L + P1.5：持久化探索历史 + 结构化反馈 + 自动推荐，闭合探索-反馈回路
    feedback = ctx.feedback
    if feedback:
        (ctx.output_dir / "feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if ctx.exploration_history:
        from ...exploration import ExplorationTracker
        tracker = ExplorationTracker(ctx.exploration_history)
        tracker.save(ctx.output_dir / "exploration.json")

        # P1.5：自动推荐下一步策略（闭合探索-反馈回路）
        task_type = ctx.task_spec.task_type if ctx.task_spec else None
        try:
            recommendations = tracker.recommend_next(task_type=task_type, top_k=5)
            if recommendations:
                (ctx.output_dir / "recommendations.json").write_text(
                    json.dumps(recommendations, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception as e:
            _logger.warning(f"recommend_next failed: {e}")

        # P1.8：success status 自动沉淀技能（闭合 Voyager 检索复用回路）
        if feedback and feedback.get("status") == "success":
            try:
                from ...skills import save_skill as _save_skill
                skill_name = f"{ctx.model_id}_{ctx.dataset}"
                _save_skill(
                    name=skill_name,
                    code=(
                        f"# Auto-saved from trial {ctx.trial_id}\n"
                        f"# model={ctx.model_id}, dataset={ctx.dataset}, "
                        f"learning_mode={ctx.learning_mode}\n"
                    ),
                    description=f"Auto-saved: {ctx.model_id} on {ctx.dataset}",
                    tags=[ctx.model_id, ctx.dataset, "auto"],
                )
                _logger.info(f"Auto-saved skill: {skill_name}")
            except Exception as e:
                _logger.warning(f"Auto save_skill failed: {e}")

    # ============================================================
    # RFC-004 方案 G：产物溯源注册 + 缺失产物补齐
    # ============================================================
    # 补齐 env_snapshot.json（独立文件，不再仅嵌在 metadata.json）
    try:
        env_snap = build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed})
        env_snap_path = ctx.output_dir / "env_snapshot.json"
        env_snap_path.write_text(
            json.dumps(env_snap, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        ctx.register_artifact(
            "env_snapshot", env_snap_path,
            kind="log", producer_stage="stage_export",
            content_schema={"python": str, "torch": str, "cuda": bool, "device": str},
        )
    except Exception as e:
        _logger.warning(f"Failed to save env_snapshot.json: {e}")

    # 补齐 metrics.csv（从 training_log 派生，Grafana/Pandas 友好）
    if training_log:
        try:
            import csv as _csv
            metrics_csv_path = ctx.output_dir / "metrics.csv"
            # 收集所有可能的字段名（union of keys across entries）
            fieldnames: list = []
            for entry in training_log:
                if isinstance(entry, dict):
                    for k in entry.keys():
                        if k not in fieldnames:
                            fieldnames.append(k)
            with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in training_log:
                    if isinstance(entry, dict):
                        writer.writerow({k: entry.get(k, "") for k in fieldnames})
            ctx.register_artifact(
                "metrics", metrics_csv_path,
                kind="metrics", producer_stage="stage_export",
                content_schema=_TRAINING_LOG_ENTRY_SCHEMA,
            )
        except Exception as e:
            _logger.warning(f"Failed to save metrics.csv: {e}")

    # 注册核心产物（model/metadata/config/training_log/feedback/exploration）
    if model_path is not None:
        ctx.register_artifact(
            "model_weights", Path(model_path),
            kind="model", producer_stage="stage_export",
            content_schema={"format": "state_dict", "num_classes": int},
        )
    metadata_path = ctx.output_dir / "metadata.json"
    if metadata_path.exists():
        ctx.register_artifact(
            "model_metadata", metadata_path,
            kind="metadata", producer_stage="stage_export",
            content_schema={"model_id": str, "dataset": str, "final_eval": dict},
        )
    config_yaml_path = ctx.output_dir / "config.yaml"
    if config_yaml_path.exists():
        ctx.register_artifact(
            "config", config_yaml_path,
            kind="config", producer_stage="stage_export",
            content_schema={"scene": str, "dataset": str, "model_id": str, "trainer": dict},
        )
    training_log_path = ctx.output_dir / "training_log.jsonl"
    if training_log_path.exists():
        ctx.register_artifact(
            "training_log", training_log_path,
            kind="log", producer_stage="stage_train",
            content_schema=_TRAINING_LOG_ENTRY_SCHEMA,
        )
    feedback_path = ctx.output_dir / "feedback.json"
    if feedback_path.exists():
        ctx.register_artifact(
            "feedback", feedback_path,
            kind="feedback", producer_stage="stage_eval",
            content_schema={"status": str, "diagnosis": str, "suggestions": list},
        )
    exploration_path = ctx.output_dir / "exploration.json"
    if exploration_path.exists():
        ctx.register_artifact(
            "exploration", exploration_path,
            kind="log", producer_stage="stage_eval",
            content_schema={"trial_id": str, "strategy": dict, "result": dict},
        )
    recommendations_path = ctx.output_dir / "recommendations.json"
    if recommendations_path.exists():
        ctx.register_artifact(
            "recommendations", recommendations_path,
            kind="log", producer_stage="stage_eval",
            content_schema={"strategy": dict, "priority": str},
        )

    return ctx


# P0.2：不可序列化 stage — 产出对象引用（bundle/model/trainer）无法从 JSON checkpoint 恢复。
# resume 时这些 stage 必须强制重跑，仅跳过纯计算 stage（validate/preflight/resolve）。
_NON_SERIALIZABLE_STAGES = frozenset({"load", "build", "train"})

# P2：pipeline checkpoint 版本号，结构变更时递增
_PIPELINE_VERSION = "2.0"


def _compute_config_hash(config: ExperimentConfig) -> str:
    """计算 config 关键字段的哈希，用于 resume 时检测 config 变更。"""
    import hashlib
    key_fields = {
        "scene": config.scene.name,
        "dataset": config.scene.dataset,
        "model_id": config.scene.model_id,
        "learning_mode": config.scene.learning_mode,
        "epochs": config.trainer.epochs,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.trainer.learning_rate,
        "optimizer": config.trainer.optimizer,
    }
    raw = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================
# RFC-004 方案 G：manifest.json 生成
# ============================================================
def _generate_manifest(ctx: PipelineContext) -> Optional[Path]:
    """从 ctx.artifact_registry 生成 manifest.json（RFC-004 方案 G）。

    在 Pipeline.run() 的 finally 块中调用，记录所有产物的路径/hash/大小/生产者/内容契约。
    成功/失败/异常路径都会生成 manifest（失败时仅含已产出的部分产物）。

    Returns:
        manifest.json 路径，失败时返回 None
    """
    import uuid
    try:
        from ... import __version__ as sf_version
    except Exception:
        sf_version = "unknown"

    try:
        config_hash = sha256_str(
            json.dumps(experiment_config_to_dict(ctx.config), sort_keys=True, default=str)
        )
    except Exception:
        config_hash = ""

    # data_hash：可选，大数据集采样 hash（此处留空，未来 DVC 集成时填充）
    data_hash = ""

    manifest = ArtifactManifest(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        senseframe_version=sf_version,
        pipeline_version=_PIPELINE_VERSION,
        config_hash=config_hash,
        data_hash=data_hash,
        artifacts=list(ctx.artifact_registry),  # copy
    )
    return manifest.save(ctx.output_dir)


# ============================================================
# Pipeline 编排器
# ============================================================
@dataclass
class Pipeline:
    """可重组的 stage pipeline。

    Agent 可：
    - 使用默认 pipeline：Pipeline.default()
    - 自定义 pipeline：Pipeline(stages=[...])
    - 替换单个 stage：pipeline.replace_stage("train", my_train)
    - 插入 hook：pipeline.before("train", my_hook)
    - 跳过 stage：pipeline.skip("export")
    """

    stages: List[tuple] = field(default_factory=list)  # [(name, fn), ...]

    @classmethod
    def default(cls) -> "Pipeline":
        """默认 pipeline（8 个 stage）。"""
        return cls(stages=[
            ("validate", stage_validate),
            ("preflight", stage_preflight),
            ("load", stage_load),
            ("resolve", stage_resolve),
            ("build", stage_build),
            ("train", stage_train),
            ("eval", stage_eval),
            ("export", stage_export),
        ])

    def replace_stage(self, name: str, fn: StageFn) -> "Pipeline":
        """替换指定 stage。"""
        self.stages = [(n, fn if n == name else f) for n, f in self.stages]
        return self

    def before(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 前插入 hook。"""
        new_stages = []
        for n, f in self.stages:
            if n == name:
                new_stages.append((f"before_{name}", hook))
            new_stages.append((n, f))
        self.stages = new_stages
        return self

    def after(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 后插入 hook。"""
        new_stages = []
        for n, f in self.stages:
            new_stages.append((n, f))
            if n == name:
                new_stages.append((f"after_{name}", hook))
        self.stages = new_stages
        return self

    def skip(self, name: str) -> "Pipeline":
        """跳过指定 stage。"""
        self.stages = [(n, f) for n, f in self.stages if n != name]
        return self

    def stages_with_spec(self) -> List[StageSpec]:
        """返回全部 stage 的 Spec（RFC-003 DSP-3）。

        遍历当前 pipeline 的所有 stage 函数，读取 @stage 装饰器附加的
        `_stage_spec` 属性。未声明的 stage 返回空 reads/writes 的 StageSpec。
        """
        specs: List[StageSpec] = []
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if spec is None:
                spec = StageSpec(name=name)
            specs.append(spec)
        return specs

    def check_readiness(self, ctx: PipelineContext, stage_name: str) -> "ReadinessReport":
        """检查指定 stage 的 reads 字段是否已在 ctx 中就绪（RFC-004 原则 9）。

        Advisory 查询：available=False 不阻断执行，仅记录信息。
        Agent 可据此决定是否跳过 stage 或手动填充缺失字段。

        Args:
            ctx: 当前 PipelineContext
            stage_name: 要检查的 stage 名

        Returns:
            ReadinessReport
        """
        spec = None
        for name, fn in self.stages:
            if name == stage_name:
                spec = getattr(fn, "_stage_spec", None)
                break
        if spec is None:
            return ReadinessReport(stage_name=stage_name, available=True, missing_reads=[])

        missing = []
        for field_spec in spec.reads:
            if field_spec.required:
                val = getattr(ctx, field_spec.name, None)
                if val is None or (hasattr(val, "__len__") and len(val) == 0 and not isinstance(val, (str, bytes))):
                    missing.append(field_spec.name)
        return ReadinessReport(
            stage_name=stage_name,
            available=len(missing) == 0,
            missing_reads=missing,
        )

    def validate_graph(self) -> List["DanglingRef"]:
        """编译期检查：reads 声明的字段是否有对应 stage 声明产出（RFC-004 原则 9）。

        遍历所有 stage 的 writes，构建"可产出字段集"，
        然后检查每个 stage 的 reads 是否有字段不在该集合中（dangling reference）。

        Advisory：返回非空列表不阻断执行，仅提示 Agent 数据通路可能断裂。
        config / extra / completed_stages 等 agent/init 填充字段视为已就绪。

        Returns:
            DanglingRef 列表（空列表表示无 dangling reference）
        """
        # 收集所有 stage 声明产出的字段
        produced: set = set()
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if spec:
                for w in spec.writes:
                    produced.add(w.name)
        # init/agent 填充的字段视为已产出（不由 stage 产出，由构造函数或 Agent 注入）
        for k, v in _FIELD_FILL_STAGE.items():
            if v in ("init", "agent"):
                produced.add(k)

        dangling: List["DanglingRef"] = []
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if not spec:
                continue
            for r in spec.reads:
                if r.name not in produced:
                    dangling.append(DanglingRef(
                        stage_name=name,
                        field_name=r.name,
                        reason="field declared as read but no stage produces it",
                    ))
        return dangling

    def run(self, ctx: PipelineContext) -> StageResult:
        """执行 pipeline（P1：支持断点续跑）。

        依次执行所有 stage，返回最终结果。
        任一 stage 抛异常则停止并返回错误。
        每个 stage 完成后写 checkpoint；失败时也写 checkpoint（标记 failed_stage）。
        若 ctx.stage_checkpoint_path 存在，加载后跳过已完成的 stage。
        """
        # P1：加载 checkpoint（若存在）
        if ctx.stage_checkpoint_path and ctx.stage_checkpoint_path.exists():
            ckpt = json.loads(ctx.stage_checkpoint_path.read_text(encoding="utf-8"))
            ctx.completed_stages = ckpt.get("completed_stages", [])

            # P2：config_hash 校验 — 若 config 变更，全部重跑
            saved_hash = ckpt.get("config_hash", "")
            current_hash = _compute_config_hash(ctx.config)
            if saved_hash and saved_hash != current_hash:
                _logger.warning(
                    f"Config changed since last run (hash {saved_hash} → {current_hash}), "
                    f"re-running all stages"
                )
                ctx.completed_stages = []
            else:
                # P0.2：不可序列化 stage 强制重跑（bundle/model/trainer 无法从 checkpoint 恢复）
                replay = [s for s in ctx.completed_stages if s in _NON_SERIALIZABLE_STAGES]
                if replay:
                    ctx.completed_stages = [s for s in ctx.completed_stages if s not in _NON_SERIALIZABLE_STAGES]
                    _logger.info(
                        f"Resumed pipeline: re-running non-serializable stages {replay} "
                        f"(object refs lost on restart), skipping {len(ctx.completed_stages)} pure stages: "
                        f"{ctx.completed_stages}"
                    )
                else:
                    _logger.info(
                        f"Resumed pipeline, skipping {len(ctx.completed_stages)} completed stages: "
                        f"{ctx.completed_stages}"
                    )

        # RFC-004 方案 F：try/finally 确保所有出口（成功/失败/异常）都释放资源
        try:
            for name, fn in self.stages:
                # P1：跳过已完成 stage
                if name in ctx.completed_stages:
                    _logger.info(f"Skipping completed stage: {name}")
                    continue

                try:
                    ctx = fn(ctx)
                    # P1：记录完成 + 写 checkpoint
                    ctx.completed_stages.append(name)
                    self._write_checkpoint(ctx)
                    # P0.2: OBP 训练指标埋点（stage 完成时记录，OTel 未初始化时 no-op）
                    record_training_metric(
                        f"senseframe.stage.{name}.duration_s",
                        value=0.0,  # Timer 在 stage_train 内部管理
                        stage=name,
                        model_id=ctx.config.scene.model_id if hasattr(ctx, "config") else "",
                        dataset=ctx.config.scene.dataset if hasattr(ctx, "config") else "",
                    )
                except Exception as e:
                    # P1：记录失败 stage + 写 checkpoint
                    ctx.failed_stage = name
                    ctx.failed_error = str(e)
                    self._write_checkpoint(ctx, failed_stage=name)

                    # 异常时 traceback 落盘
                    import traceback as _tb
                    tb = _tb.format_exc()
                    if ctx.output:
                        ctx.output.status = "error"
                        ctx.output.error = str(e)
                        ctx.output.error_traceback = tb
                        ctx.output.error_code = classify_error(e, stage=name)
                    if ctx.output_dir and ctx.output_dir.exists():
                        try:
                            (ctx.output_dir / "FAILED").write_text(tb, encoding="utf-8")
                            for p in ctx.output_dir.glob("*.pth"):
                                p.unlink()
                        except Exception:
                            pass
                    # 关闭日志写入器（release_resources 也会关闭，此处保留双保险）
                    if ctx.log_writer is not None:
                        try:
                            ctx.log_writer.close()
                        except Exception:
                            pass
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                    return StageResult(context=ctx, skipped=False, error=e)

            return StageResult(context=ctx, skipped=False)
        finally:
            # RFC-004 方案 G：生成 manifest.json（产物溯源清单）
            # 在 release_resources 前，artifact_registry 已被各 stage 填充
            try:
                if ctx.output_dir is not None and ctx.output_dir.exists():
                    _generate_manifest(ctx)
            except Exception as e:
                _logger.warning(f"Failed to generate manifest.json: {e}")
            # RFC-004 方案 F：确定性资源释放（成功/失败/异常路径均执行）
            # 幂等：release_resources 内部对已 None 字段安全
            ctx.release_resources()

    def _write_checkpoint(self, ctx: PipelineContext, failed_stage: Optional[str] = None) -> None:
        """P1：写 stage checkpoint 到 output_dir/pipeline_checkpoint.json。

        Args:
            ctx: PipelineContext（含 completed_stages、trial_id、output_dir）
            failed_stage: 若不为 None，表示在指定 stage 失败，记录到 checkpoint
        """
        if ctx.output_dir is None:
            return
        ckpt_path = ctx.output_dir / "pipeline_checkpoint.json"
        ctx.stage_checkpoint_path = ckpt_path
        data = {
            "pipeline_version": _PIPELINE_VERSION,
            "config_hash": _compute_config_hash(ctx.config),
            "completed_stages": ctx.completed_stages,
            "trial_id": ctx.trial_id,
            "timestamp": datetime.now().isoformat(),
        }
        if failed_stage:
            data["failed_stage"] = failed_stage
        try:
            ckpt_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            _logger.warning(f"Failed to write pipeline checkpoint: {e}")

    @classmethod
    def resume(cls, output_dir, pipeline_run=None) -> Tuple["Pipeline", List[str]]:
        """P1：从 output_dir 恢复 pipeline。

        读取 pipeline_checkpoint.json，返回默认 pipeline 与已完成的 stage 名列表。
        调用方可据此构造 PipelineContext 并设置 completed_stages +
        stage_checkpoint_path 以跳过已完成 stage。

        P1.2：支持传入 PipelineRun 实例，由 PipelineRun.phase 和
        PipelineRun.stages 状态驱动 completed_stages 恢复，实现 OP-3
        状态机集成。PipelineRun 优先于 checkpoint JSON。

        Args:
            output_dir: 之前的 pipeline 输出目录（含 pipeline_checkpoint.json）
            pipeline_run: PipelineRun 实例（OP 编排器提供），None 时走 JSON checkpoint

        Returns:
            (pipeline, completed_stages): 默认 pipeline 和已完成 stage 名列表

        Raises:
            FileNotFoundError: 若 checkpoint 不存在且 pipeline_run 为 None
        """
        pipeline = cls.default()

        if pipeline_run is not None:
            # P1.2: 从 PipelineRun 状态机恢复 completed_stages
            completed = [
                s.name for s in pipeline_run.stages
                if s.phase == "succeeded"
            ]
            return pipeline, completed

        # 向后兼容：从 JSON checkpoint 恢复
        output_dir = Path(output_dir)
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pipeline checkpoint found at {ckpt_path}")

        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        completed = ckpt.get("completed_stages", [])

        return pipeline, completed


def run_pipeline(config: ExperimentConfig, pipeline: Optional[Pipeline] = None) -> TrainOutput:
    """Pipeline 入口（P1：失败时输出可恢复提示）。

    Agent 可传入自定义 pipeline，或使用默认 pipeline。
    默认 pipeline 执行完整的 8 stage 流程。
    P1 简化方案：失败时不自动 retry（重建 datamodule 过于复杂），
    而是输出 resume 提示到 stderr，引导用户从失败 stage 续跑。

    Args:
        config: ExperimentConfig 实例
        pipeline: 自定义 pipeline（None 时使用默认）

    Returns:
        TrainOutput
    """
    if pipeline is None:
        pipeline = Pipeline.default()

    ctx = PipelineContext(config=config)
    result = pipeline.run(ctx)

    # 确保 output 存在
    if ctx.output is None:
        ctx.output = TrainOutput(
            status="error" if result.error else "success",
            model_id=config.scene.model_id,
            dataset=config.scene.dataset,
            learning_mode=config.scene.learning_mode,
        )
        if result.error:
            ctx.output.error = str(result.error)

    # P1：失败时输出可恢复提示
    if result.error is not None and ctx.output_dir is not None:
        import sys
        failed_stage = ctx.failed_stage or "unknown"
        print(
            f"Pipeline failed at stage '{failed_stage}'. To resume: "
            f"Pipeline.resume('{ctx.output_dir}')",
            file=sys.stderr,
        )

    return ctx.output


__all__ = [
    "PipelineContext",
    "Pipeline",
    "StageResult",
    "ReadinessReport",
    "DanglingRef",
    "StageFn",
    # RFC-003 DSP-1：Protocol 类型
    "SceneProtocol",
    "ModelProtocol",
    "DataModuleProtocol",
    "TrainerProtocol",
    "SceneMetaProtocol",
    "TaskSpecProtocol",
    "FeatureSpecProtocol",
    # RFC-003 DSP-3：Stage IO 声明
    "FieldSpec",
    "StageSpec",
    "stage",
    # 8 个默认 stage
    "stage_validate",
    "stage_preflight",
    "stage_resolve",
    "stage_load",
    "stage_build",
    "stage_train",
    "stage_eval",
    "stage_export",
    "run_pipeline",
    # RFC-004 方案 G：产物溯源体系
    "ArtifactDescriptor",
    "ArtifactManifest",
    "load_manifest",
    "verify_artifacts",
]


# ============================================================
# RFC-004 方案 G：公共溯源 API
# ============================================================
def load_manifest(output_dir) -> ArtifactManifest:
    """加载训练产物清单，供训练后分析。

    Args:
        output_dir: 训练输出目录（含 manifest.json）或 manifest.json 路径

    Returns:
        ArtifactManifest 实例
    """
    return ArtifactManifest.load(Path(output_dir))


def verify_artifacts(output_dir) -> Dict[str, bool]:
    """校验 output_dir 中所有产物的 hash，检测是否被篡改。

    Args:
        output_dir: 训练输出目录（含 manifest.json）

    Returns:
        {产物名: hash 是否匹配}
    """
    return _verify_artifacts(Path(output_dir))
