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
    ClassVar,
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

from ..config import ExperimentConfig
from ...observability import IncrementalLogWriter, Timer, setup_logging as _setup_logging
from ...observability_otel import (
    record_training_metric, record_trial_metric,
    ML_TRAIN_LOSS, ML_VAL_LOSS, ML_VAL_ACCURACY,
    # 对称性修复：新增 test 指标常量
    ML_TEST_LOSS, ML_TEST_ACCURACY,
    ML_STAGE, ML_EPOCH, ML_MODEL_ID, ML_DATASET,
    ML_TRIAL_COUNT, ML_TRIAL_BEST_METRIC,
    # P2-2: 数据侧 OTel 指标常量
    ML_DATA_LOAD_DURATION_S, ML_DATA_N_SAMPLES,
    ML_DATA_N_CLASSES, ML_DATA_IMBALANCE_RATIO,
)
from ...routing import ResourceProbe, ResourceRouter
from ...schemas import TrainOutput, validate_training_log_entry
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
    # 任务4：Pipeline.run 异常重新分类使用的具体异常类
    OOMError,
    ModelBuildError,
    TrainingError,
    DataCorruptedError,
    CheckpointError,
    SaveError,
    PreflightError,
)
# RFC-004 方案 G：训练产物溯源体系
from .artifacts import (
    ArtifactDescriptor,
    ArtifactManifest,
    sha256_file,
    sha256_str,
    verify_artifacts as _verify_artifacts,
    verify_artifacts_recursive as _verify_artifacts_recursive,
    verify_manifest_schema as _verify_manifest_schema,
    verify_artifacts_full as _verify_artifacts_full,
)
from .callbacks import StageAwareCallback, FrozenDict

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
    "dry_run": "init",
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
    "data_hash": "stage_load",  # 任务2：数据集元数据哈希（路径+大小+mtime）
    # stage_build
    "model": "stage_build",
    "datamodule": "stage_build",
    "module": "stage_build",
    "callbacks": "stage_build",
    "pl_logger": "stage_build",
    "csv_logger": "stage_build",
    "monitor": "stage_build",
    # stage_probe_vram（方案 B：前向+反向+optimizer step 探测实测显存）
    "vram_probe_result": "stage_probe_vram",
    # stage_train
    "trainer": "stage_train",
    "training_duration_s": "stage_train",
    "best_model_path": "stage_train",
    "best_model_score": "stage_train",
    "best_epoch": "stage_train",  # 任务1：best checkpoint 的 epoch 号（从 best_model_path 文件名解析）
    "intermediate_values": "stage_train",  # P2.3: ε5 Multi-fidelity（IntermediateMetricLogger 回调写入）
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
    "failed_stage": "agent",       # P0.1：Pipeline.run except 块写入（错误路径，agent 可观测）
    "failed_error": "agent",       # P0.1：同上
    # RFC-004 方案 G：产物溯源注册表（各 stage 注册，stage_export 是主要注册点）
    "artifact_registry": "stage_export",  # P0.1：声明填充 stage，消除 schema "unknown"
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
    # 任务3：dry-run 标志（True 时 stage_train 跳过 trainer.fit()，仅输出训练 plan）。
    # 由 Pipeline.run(dry_run=True) 或调用方直接设置 ctx.dry_run=True 注入。
    dry_run: bool = False
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
    # 任务2：数据集元数据哈希（stage_load 计算，_generate_manifest 写入 manifest）
    data_hash: str = ""
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
    # 任务1（P0）：best checkpoint 的 epoch 号。
    # 旧逻辑 feedback 用 final epoch 的 train/val 指标算 gap，但 final_eval 来自
    # best checkpoint（stage_train 已加载 best 权重到 ctx.model），数据源不一致
    # 导致过拟合误报。best_epoch 用于在 stage_eval 中让 analyze_training_result
    # 从 training_log 取 best epoch 那轮的 train 指标，与 final_eval 的 val 指标配对。
    best_epoch: Optional[int] = None                # stage_train 写入（从 best_model_path 文件名解析）
    final_eval: Dict[str, Any] = field(default_factory=dict)       # stage_eval 写入
    training_log: List[Any] = field(default_factory=list)          # stage_eval 写入
    # P2.3: ε5 Multi-fidelity — epoch 级中间值，IntermediateMetricLogger 回调写入
    # 供 MethodRunner 早停检查（P2.4）与 SP Pruner should_prune 使用
    intermediate_values: Dict[int, float] = field(default_factory=dict)
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
    # 方案 B：显存探测结果（stage_probe_vram 写入，stage_export 写入 metadata.resource.vram_probe）
    # None 表示探测被跳过（CPU/MPS 路由或 dry_run）；dict 含 measured_vram_mb/needed_vram_mb/free_vram_mb/ok/batch_size/precision
    vram_probe_result: Optional[Dict[str, Any]] = None
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
        # H3 修复：拒绝存储绝对路径或逃逸 output_dir 的路径
        # （旧逻辑 fallback 保留绝对路径，被 verify_artifacts 的 pathlib 拼接放大为路径穿越）
        # 路径双重嵌套修复：先 resolve 成绝对路径，避免相对项目根的路径
        # （如 output_dir / "data_profile.json"）被 safe_relative_path 当作
        # 相对 output_dir 的路径处理，导致 output_dir / path 双重嵌套
        from ...common.path_safe import safe_relative_path
        try:
            abs_path = path.resolve()
            rel_path = safe_relative_path(self.output_dir, abs_path)
        except ValueError:
            _logger.warning(
                f"register_artifact: path escapes output_dir, skipping: {name}={path}"
            )
            return None

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

    _RESOURCE_FIELDS: ClassVar[Tuple[str, ...]] = (
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
            # 注意：_teardown 是 Lightning 私有 API，版本升级可能失效或行为变化。
            # P3 修复：降级为 DEBUG（正常清理路径不应产生 WARNING 噪音）。
            # 版本兼容性风险已在代码注释记录，无需每次训练都 WARNING 提醒。
            if hasattr(self.trainer, "_teardown"):
                _logger.debug(
                    "Calling Lightning private _teardown() for resource cleanup; "
                    "this private API may break across versions."
                )
                try:
                    self.trainer._teardown()
                except Exception as e:
                    _logger.warning("trainer._teardown() failed: %s", e, exc_info=True)
            # 3b. 兼容：某些 Lightning 版本可能有 teardown 钩子
            if hasattr(self.trainer, "teardown"):
                try:
                    self.trainer.teardown(stage="fit")
                except Exception as e:
                    _logger.debug("trainer.teardown(stage='fit') failed: %s", e, exc_info=True)

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
            f"Available: {[k for k in list_scenes().keys() if k != '_unavailable']}"
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

    # P3-6：class_weight 自动注入桥接。
    # DataProfile 检测 imbalance_ratio>5 时自动计算 inverse frequency 权重，
    # 此处注入到 TaskSpec.loss_kwargs["weights"]，并将 loss 切换为 cross_entropy_weighted。
    # 设计原则：与 recommended_loss → resolver 自动注入是同一架构模式。
    # 用户显式指定 loss（config.scene.task_spec.loss 非 None）时不覆盖 loss，
    # 但仍注入 weights（供 cross_entropy_weighted 或 focal 的 alpha 使用）。
    if (ctx.data_profile is not None
            and ctx.data_profile.recommended_class_weights is not None
            and ctx.task_spec is not None):
        weights = ctx.data_profile.recommended_class_weights
        current_loss = ctx.task_spec.effective_loss
        # 仅当当前 loss 为普通 cross_entropy 时自动升级为 cross_entropy_weighted
        if current_loss == "cross_entropy":
            from ...core.losses import has_loss
            if has_loss("cross_entropy_weighted"):
                ctx.task_spec.loss = "cross_entropy_weighted"
                _logger.info(
                    "P3-6 auto-inject class_weight: ratio=%.2f, loss %s→cross_entropy_weighted, "
                    "weights=%s",
                    ctx.data_profile.imbalance_ratio, current_loss, weights,
                )
        # 注入 weights 到 loss_kwargs（合并已有 kwargs，不覆盖其他 key）
        existing_kwargs = dict(ctx.task_spec.loss_kwargs or {})
        existing_kwargs["weights"] = weights
        ctx.task_spec.loss_kwargs = existing_kwargs

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
    writes=["scene_kwargs", "bundle", "data_profile", "output_dir", "log_writer",
            "data_hash"],  # 任务2：新增 data_hash 写入声明
    description="Stage 3: 加载数据 + 数据画像",
)
def stage_load(ctx: PipelineContext) -> PipelineContext:
    """Stage 3: 加载数据 + 数据画像。"""
    import time as _time
    from ...core.profiler import DataProfiler

    # P1-4: stage 入口摘要日志
    _logger.info(
        "stage_load input: data_root=%s, dataset=%s, learning_mode=%s",
        ctx.config.scene.data_root or "(default)", ctx.dataset, ctx.learning_mode,
    )

    # P2-2: 数据加载耗时计时（包含 load_dataset + DataProfiler 全流程）
    load_timer = _time.time()

    # scene_kwargs 前置计算（供 load_dataset 使用，也供后续 resolve 读取）
    ctx.scene_kwargs = {"params": ctx.config.scene.params} if ctx.config.scene.params else {}

    # data_root 已由 SceneConfig.validate() 校验非空（YAML/CLI/env 三选一）
    data_root = ctx.config.scene.data_root

    ctx.bundle = ctx.scene.load_dataset(
        ctx.dataset, data_root, learning_mode=ctx.learning_mode,
        **ctx.scene_kwargs,
    )

    # 任务2：计算 data_hash（数据集元数据哈希）。
    # 不读取文件内容做全量 hash，只 hash 元数据（路径+大小+mtime），
    # 性能远优于全量 hash，且能检测数据集变更/损坏/缺失。
    # 存入 ctx.data_hash，供 _generate_manifest 写入 manifest.data_hash。
    try:
        ctx.data_hash = _compute_data_hash(data_root)
    except Exception as e:
        _logger.warning(f"Failed to compute data_hash: {e}")
        ctx.data_hash = ""

    # 创建输出目录（在数据画像前，便于落盘）
    # M2 修复：model_id / dataset 来自配置（不可信），清洗后再拼接，避免路径逃逸
    # P5 P2-9：dry_run 使用 tempfile.mkdtemp 隔离临时产物，run() finally 中 rmtree
    import tempfile
    if ctx.dry_run:
        ctx.output_dir = Path(tempfile.mkdtemp(prefix="senseframe_dryrun_"))
        ctx.config.save_model = False
    else:
        from ...common.path_safe import sanitize_path_component
        safe_model_id = sanitize_path_component(ctx.model_id)
        safe_dataset = sanitize_path_component(ctx.dataset)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        ctx.output_dir = Path(ctx.config.output_dir).resolve() / f"{safe_model_id}_{safe_dataset}_{timestamp}_{pid}"
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.output:
        ctx.output.output_dir = str(ctx.output_dir)

    # 数据画像（落盘到 output_dir）
    # P2-4: 异常不再静默——DataProfiler 失败时记录 error 日志（含 traceback），
    # 仍降级为 None 不中断 stage_load，但留痕供 Agent 排查。
    try:
        profiler = DataProfiler(max_samples=500)
        # P0 修复：从 SceneMeta.modality 读取场景显式声明的数据模态，
        # 覆盖 profiler 的 shape 启发式（CSI (1,250,90) 与 image (1,H,W) 不可区分）
        modality_hint = getattr(ctx.meta, "modality", None)
        # P1 修复：透传 learning_mode，让 profile_bundle 按学习模式选择采样源
        # （自监督用 unsupervised 集，监督用 train 集），避免用 test 集做画像造成数据泄露。
        ctx.data_profile = profiler.profile_bundle(
            ctx.bundle, dataset_name=ctx.dataset, modality_hint=modality_hint,
            learning_mode=ctx.learning_mode,
        )
        if ctx.data_profile is not None:
            profile_path = ctx.output_dir / "data_profile.json"
            ctx.data_profile.save(profile_path)
            # RFC-004 方案 G：注册 data_profile 产物
            # P2-4: content_schema 补全 DataProfile 全部字段（与 profiler.py dataclass 对齐）
            ctx.register_artifact(
                "data_profile", profile_path,
                kind="profile", producer_stage="stage_load",
                content_schema={
                    "n_samples": "int",
                    "input_shape": "list",
                    "n_features": "int",
                    "n_classes": "int",
                    "class_distribution": "dict",
                    "imbalance_ratio": "float",
                    "missing_rate": "float",
                    "value_range": "list",
                    "mean": "float",
                    "std": "float",
                    "is_spatial": "bool",
                    "is_temporal": "bool",
                    "modality": "str",
                    "recommended_task_type": "str",
                    "recommended_loss": "str",
                    "recommended_metrics": "list",
                    "recommended_normalization": "str",
                    "dataset_name": "str",
                    "dtypes": "dict",
                    "feature_names": "list",
                    "nullable": "dict",
                    "shapes": "dict",
                    "profile_source": "str",
                },
            )
    except Exception as e:
        # P2-4: 留痕而非静默吞掉（旧逻辑 `except Exception: pass` 完全静默）
        _logger.error(f"DataProfiler failed: {e}", exc_info=True)
        ctx.data_profile = None

    # P2-2: 数据侧 OTel 指标埋点
    # 埋点失败不能中断 stage_load（用 try/except 兜底），OTel 未初始化时 no-op。
    load_duration = _time.time() - load_timer
    try:
        record_training_metric(
            ML_DATA_LOAD_DURATION_S,
            value=load_duration,
            stage="load",
            model_id=ctx.config.scene.model_id,
            dataset=ctx.config.scene.dataset,
        )
        if ctx.data_profile is not None:
            record_training_metric(
                ML_DATA_N_SAMPLES,
                value=ctx.data_profile.n_samples or 0,
                stage="load",
                model_id=ctx.config.scene.model_id,
                dataset=ctx.config.scene.dataset,
            )
            # n_classes 可能为 None（回归任务），OTel gauge 需数值，None 时记 0
            record_training_metric(
                ML_DATA_N_CLASSES,
                value=ctx.data_profile.n_classes or 0,
                stage="load",
                model_id=ctx.config.scene.model_id,
                dataset=ctx.config.scene.dataset,
            )
            # 类别不平衡比率：根因修复（P2）— 改用 DataProfile.imbalance_ratio
            # （自监督模式下 class_distribution 为空 → imbalance_ratio 为 None）。
            # 消费前做 None 守卫，避免对 None 求值。
            if ctx.data_profile.imbalance_ratio is not None:
                record_training_metric(
                    ML_DATA_IMBALANCE_RATIO,
                    value=float(ctx.data_profile.imbalance_ratio),
                    stage="load",
                    model_id=ctx.config.scene.model_id,
                    dataset=ctx.config.scene.dataset,
                )
    except Exception as e:
        _logger.debug("OTel data metrics recording failed: %s", e)

    # 增量日志写入器
    ctx.log_writer = IncrementalLogWriter(ctx.output_dir / "training_log.jsonl")

    # P1-4: stage 出口摘要日志
    train_samples = len(ctx.bundle.train) if ctx.bundle and getattr(ctx.bundle, "train", None) is not None else 0
    test_samples = len(ctx.bundle.test) if ctx.bundle and getattr(ctx.bundle, "test", None) is not None else 0
    _logger.info(
        "stage_load output: bundle.train_samples=%d, bundle.test_samples=%d, "
        "data_profile=%s, output_dir=%s, load_duration_s=%.3f",
        train_samples, test_samples,
        "present" if ctx.data_profile else "missing",
        str(ctx.output_dir) if ctx.output_dir else "None",
        load_duration,
    )

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
    # data_root 已由 SceneConfig.validate() 校验非空（YAML/CLI/env 三选一）
    data_root = ctx.config.scene.data_root

    # P1-4: stage 入口摘要日志
    _input_shape = ctx.scene_info.get("input_shape", []) if ctx.scene_info else []
    _feature_dim = getattr(ctx.feature_spec, "feature_dim", None) if ctx.feature_spec else None
    _logger.info(
        "stage_build input: model_id=%s, dataset=%s, learning_mode=%s, "
        "input_shape=%s, feature_dim=%s, num_classes=%s",
        ctx.model_id, ctx.dataset, ctx.learning_mode,
        _input_shape, _feature_dim, ctx.num_classes,
    )

    # 构建模型
    ctx.model = ctx.scene.build_model_for_dataset(
        ctx.model_id, ctx.dataset, ctx.num_classes,
        learning_mode=ctx.learning_mode,
        data_root=data_root,
        input_dim=ctx.feature_spec.feature_dim or ctx.scene_info.get("n_features"),
        feature_spec=ctx.feature_spec,
        **ctx.scene_kwargs,
    )
    # 修复（5.11）：stage_build 模型构建无日志
    # 旧逻辑：模型构造后无任何日志，参数量/输入输出形状/是否 DeviceMap 全无
    try:
        n_params = sum(p.numel() for p in ctx.model.parameters())
        n_trainable_params = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)
        # 探测输入形状（从 scene_info 或 feature_spec）
        input_shape = ctx.scene_info.get("input_shape", [])
        is_device_map = hasattr(ctx.model, "hf_device_map") or hasattr(ctx.model, "device_map")
        _logger.info(
            f"stage_build: model constructed, model_id={ctx.model_id}, "
            f"dataset={ctx.dataset}, model_class={type(ctx.model).__name__}, "
            f"total_params={n_params:,}, trainable_params={n_trainable_params:,}, "
            f"input_shape={input_shape}, is_device_map={is_device_map}"
        )
    except Exception as e:
        _logger.debug(f"stage_build: failed to log model info: {e}")

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
    # P2-3 修复：monitor 可配置化（默认 val_loss，支持自定义指标）
    monitor_metric = getattr(ctx.config.trainer, "early_stopping_monitor", "val_loss")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ctx.output_dir / "checkpoints"),
        filename=f"best-{{epoch}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
        # PL 2.6.5: save_on_train_epoch_end=None 默认推断为 True，
        # 在 on_train_epoch_end 检查 val_loss（此时 validation 尚未执行）
        # 触发 "could not find the monitored key" 警告。
        # 显式设 False，只在 on_validation_epoch_end 检查。
        # 与 EarlyStopping(check_on_train_epoch_end=False) 对称修复。
        save_on_train_epoch_end=False,
    )
    ctx.callbacks.append(ckpt_cb)

    early_stopping_patience = ctx.config.trainer.early_stopping
    if early_stopping_patience is not None:
        # RFC-004 方案 E：使用 min_delta 避免微小波动误触发早停
        early_stopping_min_delta = getattr(
            ctx.config.trainer, "early_stopping_min_delta", 0.0
        )
        ctx.callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
            # pytorch_lightning 2.6.5: check_on_train_epoch_end=None 默认推断为 True，
            # 导致 on_train_epoch_end 时 val_loss 不可用而抛 RuntimeError。
            # 显式设为 False，只在 on_validation_epoch_end 检查（val_loss 在 validation 后才可用）。
            check_on_train_epoch_end=False,
        ))

    # P2: 创建 TrainingMonitor，供 EpochLogCallback 写入实时指标
    from ...observability import TrainingMonitor
    ctx.monitor = TrainingMonitor()

    from .orchestrator import EpochLogCallback, IntermediateMetricLogger
    ctx.callbacks.append(EpochLogCallback(log_every_n=10, monitor=ctx.monitor))
    # P2.3: ε5 Multi-fidelity — 捕获 epoch 级中间值供 Pruner should_prune 使用
    # 回调写入 ctx.intermediate_values（dict 引用），stage_train 不需修改
    ctx.callbacks.append(IntermediateMetricLogger(
        metric="val_accuracy",
        intermediate_values=ctx.intermediate_values,
    ))

    if ctx.config.extra_callbacks:
        ctx.callbacks.extend(ctx.config.extra_callbacks)

    # 修复（2.8）：若 ctx 含 Optuna trial 对象（通过 extra 传入），注册
    # OptunaReportingCallback 桥接 Lightning 中间指标到 trial.report()，
    # 让 Pruner 基于 epoch 级指标剪枝。Optuna 未安装时降级 warning。
    _optuna_trial = ctx.extra.get("optuna_trial") if ctx.extra else None
    if _optuna_trial is not None:
        try:
            from .orchestrator import OptunaReportingCallback
            ctx.callbacks.append(
                OptunaReportingCallback(
                    trial=_optuna_trial,
                    metric=ctx.resolved.get("hpo_metric", "val_accuracy"),
                )
            )
        except ImportError:
            _logger.warning(
                "ctx.extra['optuna_trial'] set but OptunaReportingCallback "
                "unavailable (optuna not installed); HPO pruner will not "
                "receive intermediate values."
            )

    if is_self_supervised:
        # 自监督模式
        unsup_ds = ctx.bundle.unsupervised
        sup_ds = ctx.bundle.supervised_finetune
        val_ds = ctx.bundle.val  # P2-3 修复：传递独立 val_dataset
        test_ds = ctx.bundle.test
        # P0 修复：自监督分支漏传 scene_kwargs，导致 get_transforms 无法读取
        # params（如 transform.pipeline/augment 配置），与监督分支行为不一致。
        # 监督分支已透传 **ctx.scene_kwargs，此处对齐。
        transform_cfg = ctx.scene.get_transforms(ctx.dataset, **ctx.scene_kwargs)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=sup_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
                supervised_transform=transform_cfg.supervised_transform,
            )
        else:
            ctx.datamodule = GenericDataModule(
                train_dataset=sup_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
                supervised_transform=transform_cfg.supervised_transform,
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
        val_ds = ctx.bundle.val  # P2-3 修复：传递独立 val_dataset
        test_ds = ctx.bundle.test
        transform_cfg = ctx.scene.get_transforms(ctx.dataset, **ctx.scene_kwargs)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=train_ds, test_dataset=test_ds,
                val_dataset=val_ds,
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
                val_dataset=val_ds,
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

    # P2-4: 注入 DataProfile 到 module，供 on_train_start 一致性校验使用
    # （如 num_classes 与 data_profile.n_classes 不匹配时提前告警）
    # 自监督和监督模式统一注入；data_profile 为 None 时跳过。
    if ctx.data_profile is not None and ctx.module is not None:
        try:
            ctx.module.data_profile = ctx.data_profile
            _logger.debug("DataProfile injected into module for on_train_start validation")
        except Exception as e:
            # module 可能是 frozen dataclass 或不允许 setattr，留痕不中断
            _logger.debug(f"Failed to inject DataProfile into module: {e}")
    else:
        _logger.debug("DataProfile is None or module is None, skip injection into module")

    # P1-4: stage 出口摘要日志
    try:
        _model_class = type(ctx.model).__name__ if ctx.model is not None else "None"
        _total_params = sum(p.numel() for p in ctx.model.parameters()) if ctx.model is not None else 0
        _dm_batch_size = getattr(ctx.datamodule, "batch_size", None) if ctx.datamodule is not None else None
        _module_class = type(ctx.module).__name__ if ctx.module is not None else "None"
    except Exception:
        _model_class = "unknown"
        _total_params = 0
        _dm_batch_size = None
        _module_class = "unknown"
    _logger.info(
        "stage_build output: model_class=%s, total_params=%d, "
        "datamodule_batch_size=%s, module_class=%s, callbacks_count=%d",
        _model_class, _total_params, _dm_batch_size, _module_class,
        len(ctx.callbacks) if ctx.callbacks else 0,
    )

    return ctx


@stage(
    name="probe_vram",
    reads=["model", "datamodule", "module", "resolved", "report",
           "dry_run", "route_level", "model_id"],
    writes=["vram_probe_result"],
    description="Stage 5.5: 动态显存探测（方案 B：前向+反向+optimizer step 测峰值显存）",
)
def _run_probe_in_subprocess(params: Dict[str, Any]) -> Dict[str, Any]:
    """在子进程中执行显存探测，隔离 CUDA 计算不影响主进程。

    设计目的：probe 的 CUDA 计算会初始化 cuBLAS/cuDNN handle，这些全局状态
    无法被 set_seed 重置，会改变后续 trainer.fit() 首步的 CUDA 状态。
    子进程隔离让 probe 的 CUDA 上下文在子进程退出时销毁，主进程不受影响。

    通信协议：
    - 主进程 → 子进程：命令行参数（标量）+ JSON 文件（复杂参数）
    - 子进程 → 主进程：JSON stdout（成功含 measured_vram_mb，失败含 error）

    Args:
        params: 探测参数 dict，含 model_id/dataset/num_classes/batch_size 等

    Returns:
        探测结果 dict（含 measured_vram_mb/needed_vram_mb/free_vram_mb/ok/breakdown_mb）

    Raises:
        PreflightError: 子进程启动失败、超时、退出码非 0 或输出非 JSON
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    # 1. 构造命令行参数
    cmd = [
        sys.executable, "-m", "senseframe.engine.runner.probe_worker",
        "--model-id", str(params["model_id"]),
        "--dataset", str(params["dataset"]),
        "--num-classes", str(params["num_classes"]),
        "--learning-mode", str(params.get("learning_mode", "supervised")),
        "--batch-size", str(params["batch_size"]),
        "--precision", str(params.get("precision", "32")),
        "--optimizer", str(params.get("optimizer", "adam")),
        "--data-root", str(params["data_root"]),
        "--scene-name", str(params["scene_name"]),
    ]

    # 2. 复杂参数写入临时 JSON 文件（feature_spec, scene_kwargs, scene_info）
    params_file = None
    complex_params = {}
    if params.get("feature_spec"):
        complex_params["feature_spec"] = params["feature_spec"]
    if params.get("scene_kwargs"):
        complex_params["scene_kwargs"] = params["scene_kwargs"]
    if params.get("scene_info"):
        complex_params["scene_info"] = params["scene_info"]
    if complex_params:
        fd, params_file = tempfile.mkstemp(suffix=".json", prefix="probe_params_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(complex_params, f, ensure_ascii=False)
        cmd.extend(["--params-file", params_file])

    # 3. 启动子进程
    try:
        _logger.info(
            "probe subprocess: model_id=%s, dataset=%s, batch_size=%s",
            params["model_id"], params["dataset"], params["batch_size"],
        )
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 分钟超时
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        raise PreflightError(
            f"VRAM probe subprocess timed out (120s). "
            f"model_id={params['model_id']}, dataset={params['dataset']}"
        )
    except FileNotFoundError as e:
        raise PreflightError(
            f"VRAM probe subprocess 启动失败（Python 不可用？）: {e}"
        )
    finally:
        # 清理临时文件
        if params_file and os.path.exists(params_file):
            os.unlink(params_file)

    # 4. 解析子进程输出
    if proc.returncode != 0:
        # 子进程异常退出，尝试解析 stderr
        stderr_snippet = (proc.stderr or "")[:500]
        # 也尝试从 stdout 解析 error JSON
        try:
            error_result = json.loads(proc.stdout.strip())
            if "error" in error_result:
                raise PreflightError(
                    f"VRAM probe subprocess 失败: {error_result['error']} "
                    f"(type={error_result.get('error_type', 'unknown')})"
                )
        except (json.JSONDecodeError, ValueError):
            pass
        raise PreflightError(
            f"VRAM probe subprocess 退出码 {proc.returncode}: {stderr_snippet}"
        )

    # 5. 解析结果 JSON
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        stdout_snippet = (proc.stdout or "")[:500]
        raise PreflightError(
            f"VRAM probe subprocess 输出非 JSON: {e}. "
            f"stdout 前 500 字符: {stdout_snippet}"
        )

    # 6. 检查 error 字段
    if "error" in result:
        raise PreflightError(
            f"VRAM probe subprocess 内部错误: {result['error']} "
            f"(type={result.get('error_type', 'unknown')})"
        )

    return result


def stage_probe_vram(ctx: PipelineContext) -> PipelineContext:
    """Stage 5.5: 动态显存探测（子进程隔离）。

    方案 B 完整实现：在 stage_build 构造模型后、stage_train 正式训练前，
    在子进程中跑 1 个 batch 的前向，测量峰值显存（含参数+梯度+optimizer
    state+激活），与 gpu_free_vram_mb 比较。

    子进程隔离（2026-07-11）：probe 在独立 Python 进程中运行，子进程退出时
    CUDA 上下文销毁，主进程的 CUDA 状态不受影响。主进程 trainer.fit() 首步
    就是进程中首次 CUDA 计算，等同无 probe 路径（N0 基线），不需要 GPU warmup。

    与现有三层防御的关系：
    - 第一层（stage_resolve.preflight_check）：静态粗筛，快速失败
    - 第二层（本 stage）：动态精确探测，给 batch_size 建议
    - 第三层（stage_train._fit_with_oom_fallback）：运行时兜底

    跳过条件（写 vram_probe_result=None）：
    - dry_run 模式（无实际训练，无需探测）
    - 非 CUDA 路由（CPU/MPS 无 CUDA 显存测量 API）
    - ctx.model 或 ctx.datamodule 为 None（无法探测）
    """
    # P1-4: stage 入口摘要日志
    _logger.info(
        "stage_probe_vram input: dry_run=%s, has_cuda=%s, route_level=%s, "
        "batch_size=%s, precision=%s",
        ctx.dry_run,
        ctx.report.has_cuda if ctx.report else False,
        ctx.route_level,
        ctx.resolved.get("batch_size") if ctx.resolved else None,
        ctx.resolved.get("precision") if ctx.resolved else None,
    )

    # 跳过条件 1：dry_run 模式无实际训练，探测无意义
    if ctx.dry_run:
        ctx.vram_probe_result = {"skipped": "dry_run", "measured_vram_mb": None}
        _logger.info("stage_probe_vram: skipped (dry_run mode)")
        return ctx

    # 跳过条件 2：非 CUDA 路由（CPU/MPS 无 CUDA 显存测量 API）
    has_cuda = ctx.report.has_cuda if ctx.report else False
    if not has_cuda:
        ctx.vram_probe_result = {"skipped": "no_cuda", "measured_vram_mb": None}
        _logger.info("stage_probe_vram: skipped (no CUDA, route_level=%s)", ctx.route_level)
        return ctx

    # 跳过条件 3：模型或 datamodule 缺失（无法构造探测输入）
    if ctx.model is None or ctx.datamodule is None:
        ctx.vram_probe_result = {"skipped": "missing_model_or_data", "measured_vram_mb": None}
        _logger.warning(
            "stage_probe_vram: skipped (model=%s, datamodule=%s)",
            ctx.model is not None, ctx.datamodule is not None,
        )
        return ctx

    # 子进程隔离：probe 在独立进程中运行，主进程不执行任何 CUDA 计算，
    # 因此不需要 set_seed / RNG 保存恢复 / 模式保存恢复 / empty_cache。
    # 主进程的 CUDA 状态完全干净，trainer.fit() 首步就是首次 CUDA 计算。

    # 构造子进程探测参数
    from ...common.paths import resolve_data_root
    data_root = ctx.config.scene.data_root
    try:
        data_root = str(resolve_data_root(data_root))
    except FileNotFoundError:
        pass  # 子进程会报告具体错误

    # 序列化 feature_spec（如果是 dataclass，转 dict）
    feature_spec_dict = None
    if ctx.feature_spec is not None:
        try:
            from dataclasses import asdict
            feature_spec_dict = asdict(ctx.feature_spec)
        except Exception:
            feature_spec_dict = None

    probe_params = {
        "model_id": ctx.model_id,
        "dataset": ctx.dataset,
        "num_classes": ctx.num_classes,
        "learning_mode": ctx.learning_mode,
        "batch_size": ctx.resolved.get("batch_size", 64),
        "precision": ctx.resolved.get("precision", "32"),
        "optimizer": ctx.resolved.get("optimizer", "adam"),
        "data_root": data_root,
        "scene_name": ctx.config.scene.name,
        "feature_spec": feature_spec_dict,
        "scene_kwargs": ctx.scene_kwargs,
        "scene_info": ctx.scene_info,
    }

    # 方案 A：batch_size 自动适配（二分搜索）
    # 首次探测用当前 batch_size；若超限，按比例降低并重测，直到通过或达到迭代上限。
    # 每次迭代启动新子进程（子进程 CUDA 上下文已污染，不可复用）。
    result = _run_probe_in_subprocess(probe_params)
    original_batch_size = result.get("batch_size")
    max_iterations = 5
    iteration = 0

    while not result.get("ok") and result.get("measured_vram_mb") is not None:
        iteration += 1
        if iteration > max_iterations:
            # 超过迭代上限仍未通过，raise 给出最终建议
            batch_size = result["batch_size"]
            free_vram_mb = result["free_vram_mb"]
            needed_vram_mb = result["needed_vram_mb"]
            suggested_bs = max(4, int(batch_size * free_vram_mb / max(needed_vram_mb, 1)))
            ctx.vram_probe_result = result
            raise PreflightError(
                f"VRAM probe failed after {max_iterations} iterations: "
                f"measured {result['measured_vram_mb']}MB "
                f"(needed {needed_vram_mb:.1f}MB with 15% margin) > free {free_vram_mb}MB. "
                f"建议：batch_size {original_batch_size} → {suggested_bs}，"
                f"或改 CPU route（device=cpu），或减小模型（当前 {ctx.model_id}）。"
            )

        # 计算建议 batch_size（激活显存约与 batch_size 成正比）
        current_bs = result["batch_size"]
        free_vram_mb = result["free_vram_mb"]
        needed_vram_mb = result["needed_vram_mb"]
        suggested_bs = max(4, int(current_bs * free_vram_mb / max(needed_vram_mb, 1)))

        if suggested_bs >= current_bs:
            # 建议值未降低，无法通过降 batch_size 解决（固定显存部分已超限）
            ctx.vram_probe_result = result
            raise PreflightError(
                f"VRAM probe failed: measured {result['measured_vram_mb']}MB "
                f"(needed {needed_vram_mb:.1f}MB) > free {free_vram_mb}MB. "
                f"batch_size 已降至 {current_bs}，无法进一步降低。"
                f"建议：改 CPU route（device=cpu），或减小模型（当前 {ctx.model_id}）。"
            )

        # 应用新 batch_size，重新探测（新子进程）
        _logger.info(
            "stage_probe_vram: batch_size %d → %d (iteration %d/%d, "
            "measured=%.1fMB > free=%.1fMB)",
            current_bs, suggested_bs, iteration, max_iterations,
            result["measured_vram_mb"], free_vram_mb,
        )
        ctx.resolved["batch_size"] = suggested_bs
        if hasattr(ctx.datamodule, "batch_size"):
            ctx.datamodule.batch_size = suggested_bs
        probe_params["batch_size"] = suggested_bs
        result = _run_probe_in_subprocess(probe_params)

    # 探测通过（或本就通过），记录最终结果
    if result.get("batch_size") != original_batch_size:
        _logger.info(
            "stage_probe_vram: batch_size auto-fitted %d → %d "
            "(measured=%.1fMB, needed=%.1fMB, free=%.1fMB)",
            original_batch_size, result["batch_size"],
            result["measured_vram_mb"], result["needed_vram_mb"],
            result["free_vram_mb"],
        )

    ctx.vram_probe_result = result
    _logger.info(
        "stage_probe_vram: measured=%.1fMB, needed=%.1fMB (15%% margin), "
        "free=%.1fMB, ok=%s, batch_size=%s, precision=%s",
        result.get("measured_vram_mb", 0),
        result.get("needed_vram_mb", 0),
        result.get("free_vram_mb", 0),
        result.get("ok"),
        result.get("batch_size"),
        result.get("precision"),
    )

    return ctx


# _run_vram_probe 已移除（2026-07-12）：
# 旧方案使用 deepcopy 副本在主进程中探测显存，已被子进程隔离方案取代。
# 当前 stage_probe_vram 调用 _run_probe_in_subprocess，probe 逻辑在
# probe_worker._do_probe 中独立实现（不依赖 PipelineContext）。
# 保留 _run_probe_in_subprocess 作为子进程隔离入口。


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
    trainer = build_trainer()
    try:
        fit_fn(trainer)
        return trainer
    except Exception as e:
        if not _is_oom_error(e):
            # 修复（2.10）：非 OOM 异常时 teardown trainer，避免资源泄露
            # （Trainer 内部持有 CUDA/dataloader worker 等资源，不 teardown 会泄露）
            if hasattr(trainer, "_teardown"):
                try:
                    trainer._teardown()
                except Exception:
                    pass
            raise
        current_bs = ctx.resolved.get("batch_size", 64)
        if current_bs <= min_batch_size:
            _logger.warning(
                f"OOM at batch_size={current_bs} (<= min {min_batch_size}), not retrying"
            )
            raise
        new_bs = max(min_batch_size, current_bs // 2)
        _logger.warning(
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
        # 修复（2.9）：del 后需 gc.collect() 打断 Trainer/LightningModule 内部循环引用，
        # 否则 empty_cache() 时引用计数未归零，显存未真正释放。
        import gc
        gc.collect()
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
    writes=["trainer", "training_duration_s", "best_model_path", "best_model_score",
            "intermediate_values"],  # 任务3：补报 intermediate_values（FrozenDict 冻结写入）
    description="Stage 6: 训练执行",
)
def stage_train(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: 训练执行。

    RFC-002 阶段 K：支持 trainer_factory 注入，Agent 可自定义 Trainer 构造。
    """
    is_self_supervised = (ctx.learning_mode == "self_supervised")
    deterministic = ctx.config.trainer.deterministic

    # 子进程隔离方案（2026-07-11）：probe 在独立子进程中运行，主进程不消耗 RNG，
    # 不需要在 stage_train 入口重新 set_seed。set_seed 仅在 stage_preflight 调用
    # 一次，之后 RNG 自然流经 stage_load/resolve/build，与 N0 基线（无 probe）
    # 路径一致。在 stage_train 入口额外 set_seed 会重置 RNG，导致 DataLoader
    # shuffle 顺序与 N0 基线不同（实测 ep0 从 1.210943 变为 1.258861）。

    enable_progress_bar = ctx.config.trainer.enable_progress_bar
    max_time = ctx.config.trainer.max_time or "00:02:00:00"

    # checkpoint 恢复
    resume_ckpt = ctx.config.trainer.resume
    if resume_ckpt is None and ctx.config.scene.params:
        resume_ckpt = ctx.config.scene.params.get("resume")

    # P1-4: stage 入口摘要日志
    _epochs = ctx.config.trainer.epochs
    _batch_size = ctx.resolved.get("batch_size") if ctx.resolved else None
    _lr = ctx.resolved.get("learning_rate") if ctx.resolved else None
    _optimizer = ctx.resolved.get("optimizer") if ctx.resolved else None
    _scheduler = ctx.resolved.get("scheduler") if ctx.resolved else None
    _logger.info(
        "stage_train input: epochs=%s, batch_size=%s, learning_rate=%s, "
        "optimizer=%s, scheduler=%s, learning_mode=%s, resume_ckpt=%s",
        _epochs, _batch_size, _lr, _optimizer, _scheduler,
        ctx.learning_mode, resume_ckpt,
    )

    # 修复（任务3 / P0）：dry-run 模式跳过 trainer.fit()，仅输出训练 plan。
    # 旧逻辑用 limit_train_batches=1 近似 dry-run，但仍执行完整 fit/validation/
    # checkpoint，产生副作用（写 checkpoint、占显存、跑验证）。改为入口直接短路。
    if ctx.dry_run:
        plan = {
            "epochs": _epochs,
            "batch_size": _batch_size,
            "learning_rate": _lr,
            "optimizer": _optimizer,
            "scheduler": _scheduler,
            "device": ctx.lightning_params.get("accelerator") if ctx.lightning_params else None,
            "devices": ctx.lightning_params.get("devices") if ctx.lightning_params else None,
            "precision": ctx.lightning_params.get("precision") if ctx.lightning_params else None,
            "max_time": max_time,
            "learning_mode": ctx.learning_mode,
            "resume_ckpt": resume_ckpt,
        }
        _logger.info("stage_train dry-run plan: %s", json.dumps(plan, default=str))

        # 修复（任务2 / P1）：dry-run 短路改为前向传播验证。
        # 旧逻辑 dry-run 完全跳过 fit()，不执行任何前向传播，无法验证模型可前向。
        # CLI 的 _cmd_dry_run 动态校验需要"1 epoch + 1 batch 前向"验证模型可前向。
        # 方案：从 datamodule 取 1 个 batch，验证模型可前向。失败不阻断 dry-run
        # （只 warning），因为前向验证的目的是验证模型可前向，非阻断性校验。
        try:
            if hasattr(ctx.datamodule, 'setup'):
                try:
                    ctx.datamodule.setup()
                except Exception:
                    pass
            train_dl = ctx.datamodule.train_dataloader() if hasattr(ctx.datamodule, 'train_dataloader') else None
            if train_dl is not None and ctx.model is not None:
                batch = next(iter(train_dl))
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get('x') or batch.get('input') or list(batch.values())[0]
                else:
                    x = batch
                with torch.no_grad():
                    output = ctx.model(x)
                _logger.info(
                    "stage_train dry-run: forward pass OK, output shape=%s",
                    output.shape if hasattr(output, 'shape') else type(output).__name__,
                )
        except Exception as e:
            _logger.warning("stage_train dry-run: forward pass failed: %s", e)
            # 前向失败不阻断 dry-run，只在报告中标记

        ctx.training_duration_s = 0.0
        ctx.best_model_path = None
        ctx.best_model_score = None
        ctx.best_epoch = None  # 任务1：dry-run 无训练，best_epoch 置 None
        _logger.info(
            "stage_train dry-run: skipped trainer.fit(), forward validation done"
        )
        return ctx

    timer = Timer("training")
    timer.__enter__()

    # P1-5.8: 训练入口 log 显存占用 + 梯度裁剪配置（可观测性补全）
    try:
        import torch as _torch
        _grad_clip_val = ctx.resolved.get("gradient_clip_val")
        _grad_clip_algo = ctx.resolved.get("gradient_clip_algorithm", "norm")
        if _grad_clip_val is not None:
            _logger.info(
                "stage_train: gradient_clip configured (val=%s, algorithm=%s)",
                _grad_clip_val, _grad_clip_algo,
            )
        else:
            _logger.info("stage_train: gradient_clip disabled (val=None)")
        if _torch.cuda.is_available():
            _allocated = _torch.cuda.memory_allocated() / (1024 ** 3)
            _reserved = _torch.cuda.memory_reserved() / (1024 ** 3)
            _logger.info(
                "stage_train: GPU memory before fit (allocated=%.3f GB, reserved=%.3f GB)",
                _allocated, _reserved,
            )
    except Exception as _e:
        _logger.debug("stage_train: failed to log GPU memory / gradient config: %s", _e)

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
        # P2-3: 从 config 读取 limit_train_batches / limit_val_batches
        # 仅在非 None 时添加（默认 None 保持向后兼容，dry-run 动态校验时设为 1）
        # 调用方可通过 overrides 覆盖（如自监督阶段 limit_val_batches=0）
        _limit_train = getattr(ctx.config.trainer, "limit_train_batches", None)
        _limit_val = getattr(ctx.config.trainer, "limit_val_batches", None)
        if _limit_train is not None:
            kwargs["limit_train_batches"] = _limit_train
        if _limit_val is not None:
            kwargs["limit_val_batches"] = _limit_val
        # Part 4：自动 LR 标定注入 Trainer 构造参数
        if ctx.config.trainer.auto_lr_find:
            kwargs["auto_lr_find"] = True
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
                    **_build_trainer_kwargs(limit_val_batches=0),
                )
            return pl.Trainer(
                max_epochs=ss_epochs,
                logger=ctx.csv_logger,
                enable_checkpointing=False,
                **_build_trainer_kwargs(limit_val_batches=0),
            )
        def _fit_ss(trainer):
            # 每次重新获取 dataloader，OOM 重试时反映新 batch_size
            trainer.fit(ctx.module, train_dataloaders=ctx.datamodule.train_dataloader())
        # RFC-005：存 SS Phase 1 Trainer 返回值，fit 后显式 _teardown 释放
        # 修复（2.10）：非 OOM 异常时 ss_trainer 资源泄露——用 try/finally 确保
        # 异常路径也 teardown。同时修复（2.9）：del 后加 gc.collect() 打断循环引用。
        ss_trainer = _fit_with_oom_fallback(ctx, _build_ss_trainer, _fit_ss)
        try:
            if hasattr(ss_trainer, "_teardown"):
                try:
                    ss_trainer._teardown()
                except Exception:
                    pass
        finally:
            del ss_trainer
            import gc
            gc.collect()
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

        # Part 4（风险推演 R3）：自动 LR 标定。
        # 用独立 tune_trainer 隔离副作用——trainer.tune() 内部跑 1 epoch 训练，
        # 会触发回调写入 training_log、更新 metric 状态、可能触发 checkpoint。
        # 用独立 Trainer（关闭 checkpoint/validation/logger）隔离，tune 后清理状态。
        if ctx.config.trainer.auto_lr_find:
            _logger.info("stage_train: auto_lr_find enabled, running LR Range Test...")
            try:
                tune_trainer = pl.Trainer(
                    **_build_trainer_kwargs(
                        max_epochs=1,
                        enable_checkpointing=False,
                        limit_val_batches=0,
                        logger=False,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                    ),
                    auto_lr_find=True,
                )
                tune_result = tune_trainer.tune(ctx.module, datamodule=ctx.datamodule)
                suggestion = tune_result.get("lr_find", {}).get("suggestion")
                if suggestion is not None:
                    ctx.module.learning_rate = suggestion
                    ctx.resolved["learning_rate"] = suggestion
                    _logger.info(
                        "stage_train: auto_lr_find suggested lr=%.6f", suggestion
                    )
                else:
                    _logger.warning(
                        "stage_train: auto_lr_find failed to suggest lr, using default"
                    )
                # 清理 tune_trainer（释放显存）
                del tune_trainer
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 清理 tune 期间的副作用：清空 training_log + 重置累加器
                # tune_trainer.tune() 会触发 on_train_epoch_end 写入 epoch 0 entry
                ctx.module.training_log.clear()
                ctx.module._current_epoch_loss = 0.0
                ctx.module._current_epoch_steps = 0
                ctx.module._current_val_epoch_loss = 0.0
                ctx.module._current_val_epoch_steps = 0
                ctx.module._has_validation_run = False
                # 重置 metric 状态（tune 期间更新了 torchmetrics）
                for metric_dict in [ctx.module.train_metrics, ctx.module.val_metrics]:
                    for name in metric_dict:
                        try:
                            metric_dict[name].reset()
                        except Exception:
                            pass
                _logger.info("stage_train: auto_lr_find done, training_log cleared, starting fit")
            except Exception as e:
                _logger.warning(
                    "stage_train: auto_lr_find failed: %s, using default lr", e
                )

        ctx.trainer = _fit_with_oom_fallback(ctx, _build_supervised_trainer, _fit_supervised)

    # 训练结束：停止计时器 + 提取 checkpoint 信息到 first-class 字段
    timer.__exit__()
    ctx.training_duration_s = round(timer.elapsed, 2)
    for cb in ctx.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ctx.best_model_path = cb.best_model_path or None
            ctx.best_model_score = float(cb.best_model_score) if cb.best_model_score is not None else None
            # 任务1（P0）：从 best_model_path 文件名解析 best_epoch。
            # ModelCheckpoint filename 格式 "best-{epoch}-{val_loss:.3f}" 实际生成
            # "best-epoch=14-val_loss=0.066.ckpt"，用正则 r"epoch=(\d+)" 解析 epoch 号。
            # 解析失败时回退到 len(module.training_log)（最后完成的 epoch 号，1-based）。
            # best_epoch 用于 stage_eval 的 analyze_training_result 取 best epoch 那轮
            # train 指标，与 final_eval 的 val 指标配对算 gap，避免数据源不一致的过拟合误报。
            if ctx.best_model_path:
                import re as _re
                _m = _re.search(r"epoch=(\d+)", ctx.best_model_path)
                if _m:
                    ctx.best_epoch = int(_m.group(1))
                else:
                    _tl = getattr(ctx.module, "training_log", None) if ctx.module else None
                    ctx.best_epoch = len(_tl) if _tl else None
            else:
                ctx.best_epoch = None
            break

    # 修复：best model 加载回 ctx.model。
    # 旧逻辑训练结束后 ctx.model 仍是最后一代权重，导出的 model.pth 是最后一代
    # 而非最优，final_eval 反映最后一代性能（可能因 early stopping 远差于 best）。
    # 改为：若有 best checkpoint，加载回 ctx.model，确保后续 export/eval 用最优权重。
    if ctx.best_model_path:
        import os
        if os.path.exists(ctx.best_model_path):
            try:
                # Lightning checkpoint 含 state_dict + optimizer + scheduler 等，
                # 加载时只取 state_dict，避免 optimizer/scheduler 状态覆盖
                ckpt = torch.load(ctx.best_model_path, map_location="cpu", weights_only=False)
                state_dict_key = "state_dict"
                if state_dict_key in ckpt:
                    # LightningModule 的 state_dict 键前缀是 "model."，去掉后加载到裸 model
                    raw_state = {k[len("model."):]: v for k, v in ckpt[state_dict_key].items()
                                 if k.startswith("model.")}
                    if raw_state:
                        ctx.model.load_state_dict(raw_state)
                        _logger.info(
                            f"stage_train: loaded best model weights from {ctx.best_model_path} "
                            f"into ctx.model (best_score={ctx.best_model_score})"
                        )
                    else:
                        _logger.warning(
                            f"stage_train: best checkpoint {ctx.best_model_path} has no "
                            f"'model.' prefixed keys in state_dict, skip loading"
                        )
                else:
                    _logger.warning(
                        f"stage_train: best checkpoint {ctx.best_model_path} missing "
                        f"'state_dict' key, skip loading"
                    )
            except Exception as e:
                _logger.warning(
                    f"stage_train: failed to load best checkpoint {ctx.best_model_path}: {e}",
                    exc_info=True,
                )
        else:
            _logger.warning(
                f"stage_train: best_model_path does not exist: {ctx.best_model_path}"
            )

    # P0-1 防御性兜底：stage_train 后冻结 intermediate_values
    # 防止 stage_eval 的 trainer.validate() 触发 IntermediateMetricLogger 写入
    ctx.intermediate_values = FrozenDict(ctx.intermediate_values)
    _logger.info(
        "intermediate_values frozen with %d entries after stage_train",
        len(ctx.intermediate_values),
    )

    # P1-4: stage 出口摘要日志
    _logger.info(
        "stage_train output: best_model_score=%s, best_model_path=%s, "
        "training_duration_s=%s, intermediate_values_count=%d",
        ctx.best_model_score, ctx.best_model_path,
        ctx.training_duration_s,
        len(ctx.intermediate_values),
    )

    return ctx


def analyze_training_result(
    final_eval: Dict[str, Any],
    training_log: List[Any],
    early_stopped: bool,
    task_type: str = "classification",
    best_epoch: Optional[int] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """分析训练结果，输出结构化反馈（RFC-002 阶段 L）。

    闭合探索-反馈回路：eval 结果 → 失败分类 + 改进建议 → Agent 调整策略。

    任务1（P0）修复：新增 best_epoch 参数。旧逻辑反向遍历 training_log 找
    **最后一轮**的 train/val 指标算 gap，但 final_eval 来自 best checkpoint
    （stage_train 已加载 best 权重到 ctx.model），两个数据源不一致导致过拟合误报
    （实测 best epoch val=0.982 但末轮 val=0.823，gap=0.161 误报 overfitting）。
    修复后：若 best_epoch 提供，从 training_log 找 entry["epoch"] == best_epoch
    的条目，用该条目的 train/val 指标算 gap，数据源与 final_eval 一致。

    对称性修复：新增 val-test gap 泛化分析。P2-3 修复后 val/test 分离，
    final_eval 同时含 val_* 和 test_* 指标。val-test gap 过大表示
    模型在 val 上调参（early_stopping）后在 test 上泛化能力下降。

    Args:
        final_eval: 最终评估指标（含 val_* 和 test_* 前缀）
        training_log: 训练日志（每 epoch 1 条）
        early_stopped: 是否早停
        task_type: 任务类型（classification/regression）
        best_epoch: best checkpoint 的 epoch 号（None 时回退到末轮逻辑）

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

    # 2. 从 training_log 提取 train/val metric
    # 修复（任务1 / P0）：feedback 基于 best epoch 而非 final epoch。
    # 旧逻辑反向遍历找最后一轮的 train/val 指标算 gap，但 final_eval 来自 best
    # checkpoint（stage_train 已加载 best 权重），两数据源不一致导致过拟合误报。
    # 修复：若 best_epoch 提供，从 training_log 找 entry["epoch"] == best_epoch
    # 的条目，用该条目的 train/val 指标算 gap，数据源与 final_eval 一致。
    # best_epoch 为 None 或找不到对应条目时回退原逻辑（反向遍历找最后一轮）。
    # Part 3（风险推演 R1）：过滤掉 final_eval 行，避免回退反向遍历取到
    # final validation 的 val_accuracy（来自 best checkpoint）与 epoch N 的
    # train_accuracy 错配，重新引入数据源不一致问题。
    # phase 字段可选，默认 "train_val"（向后兼容无 phase 字段的旧 entry）。
    trainable_log = [
        e for e in (training_log if isinstance(training_log, list) else [])
        if isinstance(e, dict) and e.get("phase", "train_val") != "final_eval"
    ]
    last_train_acc = None
    last_val_acc = None
    best_entry = None
    if best_epoch is not None:
        for entry in trainable_log:
            if isinstance(entry, dict) and entry.get("epoch") == best_epoch:
                best_entry = entry
                break
    if best_entry is not None:
        last_train_acc = best_entry.get("train_accuracy") or best_entry.get("train_acc")
        last_val_acc = best_entry.get("val_accuracy") or best_entry.get("val_acc")
    else:
        # 回退：反向遍历找最后一轮
        for entry in reversed(trainable_log):
            if not isinstance(entry, dict):
                continue
            if last_train_acc is None:
                last_train_acc = entry.get("train_accuracy") or entry.get("train_acc")
            if last_val_acc is None:
                last_val_acc = entry.get("val_accuracy") or entry.get("val_acc")
            if last_train_acc is not None and last_val_acc is not None:
                break

    # P4-5：val_acc 提取改为显式 is not None 检查链。
    # 旧代码用 or 链，当 val_accuracy=0.0（falsy）时会错误跳到 accuracy 或 last_val_acc，
    # 导致 underfitting 检查被静默跳过。
    val_acc = None
    for _key in ("val_accuracy", "accuracy"):
        _v = final_eval.get(_key)
        if _v is not None:
            val_acc = _v
            break
    if val_acc is None:
        val_acc = last_val_acc

    # 3. 欠拟合：验证准确率过低
    # P4-5：阈值动态化，基于 n_classes 计算随机猜测基线。
    # 旧代码硬编码 0.5，对多分类（如 7 类，随机基线≈0.143）过宽松。
    # 新阈值 = max(2/n_classes, 0.3)，即随机基线的 2 倍与 0.3 取较大值。
    if val_acc is not None and task_type == "classification":
        if n_classes is not None and n_classes > 1:
            underfit_threshold = max(2.0 / n_classes, 0.3)
        else:
            underfit_threshold = 0.5
        if val_acc < underfit_threshold:
            return {
                "status": "underfitting",
                "diagnosis": (
                    f"验证准确率 {val_acc:.3f} 低于阈值 {underfit_threshold:.3f}"
                    f"（n_classes={n_classes}, 随机基线≈{1.0/(n_classes or 2):.3f}），模型欠拟合"
                ),
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

    # 对称性修复：val-test gap 泛化分析
    # P2-3 修复后 val/test 分离，final_eval 同时含 val_* 和 test_* 指标
    # val-test gap 过大表示模型在 val 上调参（early_stopping）后在 test 上泛化能力下降
    _test_acc = final_eval.get("test_accuracy") or final_eval.get("test_acc")
    if (val_acc is not None and _test_acc is not None
            and task_type == "classification"):
        val_test_gap = val_acc - _test_acc
        if val_test_gap > 0.10:
            return {
                "status": "generalization_gap",
                "diagnosis": (f"val-test gap {val_test_gap:.3f}"
                              f"（val={val_acc:.3f}, test={_test_acc:.3f}），"
                              f"模型在 val 上调参后 test 泛化能力下降"),
                "suggestions": [
                    "增大 val_split_ratio（如 0.1 → 0.2）以获得更稳健的 val 估计",
                    "检查 val/test 分布是否一致（domain shift）",
                    "使用 k-fold 交叉验证替代单次 split",
                    "增大 early_stopping patience 容忍 val 波动",
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
    writes=["output", "exploration_history",  # P2.1: 对齐函数体（写 exploration_history.feedback）
            "final_eval", "training_log", "early_stopped", "feedback"],  # 任务3：补报 stage_eval 写入字段
    description="Stage 7: 评估",
)
def stage_eval(ctx: PipelineContext) -> PipelineContext:
    """Stage 7: 评估。

    RFC-002 阶段 L：输出结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路。

    P4-2 文档澄清：本 stage 内部调用 ctx.trainer.validate()，Lightning Trainer
    会对每个 validation batch 调用 LightningModule.validation_step（约 N 次，
    N = validation batch 数）。这是 Lightning 的固有行为，与 replace_stage 无关。

    replace_stage("eval", fn) 的语义是完全取代 stage 函数，原 stage_eval 不执行。
    若需"eval 后钩子"，应使用 after("eval", hook) 且 hook 内部不调用 trainer.validate()。
    """
    # P5 P1-N：dry_run 模式下跳过评估（ctx.trainer 为 None，trainer.validate() 会崩溃）
    if ctx.dry_run:
        _logger.info("Skipping stage_eval in dry_run mode")
        return ctx

    is_self_supervised = (ctx.learning_mode == "self_supervised")

    # 修复（2.7）：_is_final_validation 标志在 trainer.validate() 完成后必须 reset，
    # 否则模块复用时（如 HPO 多 trial 复用同一 module）状态污染，后续训练中验证
    # 误走 final_validation 路径。用 try/finally 确保异常时也 reset。
    ctx.module._is_final_validation = True
    try:
        if is_self_supervised:
            ctx.trainer.validate(ctx.module, dataloaders=ctx.datamodule.val_dataloader())
        else:
            ctx.trainer.validate(ctx.module, datamodule=ctx.datamodule)
    finally:
        ctx.module._is_final_validation = False

    # 对称性修复：在 trainer.validate() 后新增 trainer.test() 调用
    # P2-3 修复后 val/test 分离，test 集需要独立评估以报告泛化能力
    # trainer.test() 触发 test_step → on_test_epoch_end，存储 _last_test_metrics
    # get_final_metrics 会合并 val_* 和 test_* 指标
    ctx.module._is_final_test = True
    try:
        if is_self_supervised:
            ctx.trainer.test(ctx.module, dataloaders=ctx.datamodule.test_dataloader())
        else:
            ctx.trainer.test(ctx.module, dataloaders=ctx.datamodule.test_dataloader())
    finally:
        ctx.module._is_final_test = False

    # 收集结果（get_final_metrics 现在合并 val_* 和 test_* 指标）
    final_eval = ctx.module.get_final_metrics()
    training_log = ctx.module.training_log
    early_stopped = any(
        isinstance(cb, EarlyStopping) and cb.stopped_epoch >= 0
        for cb in ctx.trainer.callbacks
    )
    # 修复（5.8）：early stopping 触发时无日志，加 INFO 留痕
    if early_stopped:
        stopped_epoch = -1
        for cb in ctx.trainer.callbacks:
            if isinstance(cb, EarlyStopping) and cb.stopped_epoch >= 0:
                stopped_epoch = cb.stopped_epoch
                break
        _logger.info(
            f"early stopping triggered at epoch {stopped_epoch} "
            f"(monitor={getattr(cb, 'monitor', 'val_loss')})"
        )

    # 保存结果到 first-class 字段
    ctx.final_eval = final_eval
    ctx.training_log = training_log
    ctx.early_stopped = early_stopped

    # RFC-002 阶段 L：结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路
    task_type = ctx.task_spec.task_type if ctx.task_spec else "classification"
    # 任务1（P0）：传入 best_epoch，让 analyze_training_result 从 training_log
    # 取 best epoch 那轮的 train 指标，与 final_eval 的 val 指标配对算 gap，
    # 避免数据源不一致（final epoch train vs best checkpoint val）导致过拟合误报。
    ctx.feedback = analyze_training_result(
        final_eval, training_log, early_stopped, task_type=task_type,
        best_epoch=ctx.best_epoch,
        n_classes=ctx.num_classes,
    )

    # P5 P2-7 阶段2：在 analyze_training_result 出口做类型校验并切换为 FeedbackResult 实例。
    # 下游消费方已迁移为属性访问 + to_dict() 序列化兼容。
    # hpo.py 在传给 tracker 时会调用 .to_dict() 转为 dict。
    from ...schemas import validate_feedback
    ctx.feedback = validate_feedback(ctx.feedback)

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
    # 对称性修复：test 指标 OTel 埋点（与 val 对称）
    _test_acc = final_eval.get("test_accuracy") or final_eval.get("test_acc")
    _test_loss = final_eval.get("test_loss")
    if _test_acc is not None:
        record_training_metric(ML_TEST_ACCURACY, value=float(_test_acc),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    if _test_loss is not None:
        record_training_metric(ML_TEST_LOSS, value=float(_test_loss),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    # 记录 trial count
    record_trial_metric(
        ML_TRIAL_COUNT, value=len(ctx.exploration_history),
        trial_id=ctx.trial_id,
    )

    return ctx


def _merge_metrics_csv(csv_path: Path) -> None:
    """合并 Lightning CSVLogger 的 train+val 分行为 1 行/epoch（任务4 / P2）。

    根因：Lightning CSVLogger 在 on_train_epoch_end 和 on_validation_epoch_end
    分别写入一行，导致每 epoch 有 2 行（train 行含 train_* 指标但 val_* 为空，
    val 行含 val_* 指标但 train_* 为空）。这与 training_log.jsonl 的 1 行/epoch
    格式不一致，下游消费者（如 Agent 分析、manifest 校验）难以对齐。

    合并后：每 epoch 1 行，train_* 和 val_* 在同一行，与 training_log.jsonl 对齐。
    同一 epoch 的多行合并时，非空值覆盖空值（train 行的 train_* + val 行的 val_*）。
    """
    import csv
    lines = csv_path.read_text(encoding='utf-8').splitlines()
    if not lines:
        return
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames
    if not fieldnames:
        return

    # 按 epoch 聚合：同一 epoch 的多行合并，非空值覆盖空值
    merged = {}  # epoch -> row dict
    epoch_order = []
    for row in reader:
        try:
            ep = int(float(row.get('epoch', 0)))
        except (ValueError, TypeError):
            continue
        if ep not in merged:
            merged[ep] = {'epoch': ep}
            epoch_order.append(ep)
        for k, v in row.items():
            if k == 'epoch':
                continue
            if v is not None and v != '':
                merged[ep][k] = v

    # 覆写 csv 文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epoch_order:
            writer.writerow(merged[ep])


@stage(
    name="export",
    reads=["config", "model", "module", "output", "output_dir",
           "scene", "scene_info", "scene_kwargs", "meta", "report",
           "route_level", "task_spec", "feature_spec", "resolved",
           "log_writer", "exploration_history", "num_classes",
           "model_id", "dataset", "learning_mode",
           # 任务3：补报 stage_export 读取字段（函数体实际访问的 ctx.xxx）
           "bundle", "data_profile",
           "training_duration_s", "best_model_path", "best_model_score",
           # 方案 B：显存探测结果写入 metadata.resource.vram_probe
           "vram_probe_result"],
    writes=["output", "artifact_registry"],  # 任务3：补报 artifact_registry（register_artifact 写入）
    description="Stage 8: 导出",
)
def stage_export(ctx: PipelineContext) -> PipelineContext:
    """Stage 8: 导出。"""
    # P5 P1-N：dry_run 模式下跳过导出
    if ctx.dry_run:
        _logger.info("Skipping stage_export in dry_run mode")
        return ctx

    final_eval = ctx.final_eval
    training_log = ctx.training_log
    early_stopped = ctx.early_stopped

    # P1-1: training_log schema 校验（拦截 LR 污染等类型错误）
    # strict_schema=True 时校验失败直接抛错；False 时降级保留原始 entry
    validated_log: List[Any] = []
    for entry in training_log:
        try:
            validated_entry = validate_training_log_entry(entry)
            validated_log.append(validated_entry.to_dict())
        except (ValueError, TypeError) as e:
            _logger.error(
                f"training_log entry failed schema validation: {e}",
                exc_info=True,
            )
            if getattr(ctx.config, "strict_schema", False):
                raise
            validated_log.append(entry)  # 保留原始（可能含污染）
    # 后续产物写入（ctx.output.training["log"] / metrics.csv）使用校验后的版本
    training_log = validated_log

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
            # metadata.config 是完整配置快照，供实验复现与下游消费者（generate_inference 等）使用。
            # 根因修复：ctx.resolved 仅含路由运行时字段（device/batch_size/precision/...），
            # 缺失 14 个训练级字段（epochs/seed/deterministic/max_time/...）和场景级字段（data_root/...）。
            # 方案 D：合并 experiment_config_to_dict(ctx.config)（声明式配置完整快照）
            # 与 ctx.resolved（路由解析后实际生效值），重叠字段以 ctx.resolved 为准。
            # 这样复现所需字段（epochs/seed/data_root/learning_mode/...）全部进入 metadata.config，
            # 且未来 ExperimentConfig 新增字段自动进入，无需逐字段补录。
            "config": {
                **experiment_config_to_dict(ctx.config),
                **ctx.resolved,
            },
            "metrics": list(final_eval.keys()),
            "final_eval": final_eval,
            # 对称性修复：显式提取 test_eval 字段，便于下游消费者直接访问 test 指标
            "test_eval": {
                k: v for k, v in final_eval.items()
                if k.startswith("test_")
            } if any(k.startswith("test_") for k in final_eval) else None,
            "env": build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed}),
            "resource": {
                **ctx.report.to_dict(),
                # 方案 B：动态显存探测结果（stage_probe_vram 写入）
                # None/跳过时记 skipped 原因；探测成功时含 measured_vram_mb/needed_vram_mb/free_vram_mb/ok
                "vram_probe": ctx.vram_probe_result,
            },
            "route_level": ctx.route_level,
            "task_spec": ctx.task_spec.to_dict(),
            "feature_spec": ctx.feature_spec.to_dict(),
            # Part 2：best checkpoint 溯源 + epoch 利用率（风险推演 R1/R4）
            # best_epoch/best_model_path/best_model_score 从 ctx 读取（stage_train 写入）
            # epoch_utilization = best_epoch / epochs，供 Agent 判断预算是否合理
            # （<0.3 预算过大，>0.9 预算不足）
            "best_epoch": ctx.best_epoch,
            "best_model_path": ctx.best_model_path,
            "best_model_score": ctx.best_model_score,
            "epoch_utilization": round(ctx.best_epoch / ctx.config.trainer.epochs, 3) if ctx.best_epoch and ctx.config.trainer.epochs else None,
            "created_at": datetime.now().isoformat(),
        }
        # P5 P2-6：strict_schema=True 时对 metadata 关键字段做类型校验
        # 旧代码 strict_schema 仅控制 training_log，metadata 无类型校验，
        # 允许 num_classes=str / best_epoch=float 等类型污染传播到下游
        if getattr(ctx.config, "strict_schema", False):
            _type_checks = [
                ("model_id", ctx.model_id, str),
                ("dataset", ctx.dataset, str),
                ("num_classes", ctx.num_classes, int),
                ("best_epoch", ctx.best_epoch, (type(None), int)),
                ("best_model_score", ctx.best_model_score, (type(None), float, int)),
                ("created_at", metadata["created_at"], str),
            ]
            for field_name, value, expected in _type_checks:
                if not isinstance(value, expected):
                    raise TypeError(
                        f"metadata.{field_name} type error: expected {expected}, "
                        f"got {type(value).__name__}={value!r}"
                    )
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
        # P5 P2-7 阶段2：构造 TrainingSummary dataclass 实例（不再还原为 dict）。
        # 下游消费方已迁移为属性访问 + to_dict() 序列化兼容。
        # TrainOutput.to_dict() 已有多态序列化 helper，会自动调用 .to_dict()。
        from ...schemas import validate_training_summary, validate_env_snapshot
        ctx.output.training = validate_training_summary({
            "epochs_trained": len(training_log),
            "early_stopped": early_stopped,
            "log": training_log,
            "duration_s": ctx.training_duration_s,
            "best_val_loss": ctx.best_model_score,
            "best_checkpoint": ctx.best_model_path,
            "intermediate_values": ctx.intermediate_values,  # P2.3: ε5 Multi-fidelity
        })
        ctx.output.final_eval = final_eval
        ctx.output.model_path = model_path
        env_snapshot_dict = build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed})
        ctx.output.env_snapshot = validate_env_snapshot(env_snapshot_dict)

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
            # P4-3：导出有 errors（如 onnx 包缺失）时记录 warning，不再静默。
            if export_result.errors:
                _logger.warning(
                    "Export completed with errors: %s", export_result.errors
                )
        except Exception as e:
            # P4-3：导出异常不再静默吞没，记录 warning 供排查。
            _logger.warning("Export failed: %s", e)
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
        # 对称性修复：在 feedback 中附加 test 指标摘要
        # P2-3 修复后 val/test 分离，feedback 应同时包含 val 和 test 指标
        if ctx.final_eval:
            _test_metrics_summary = {
                k: v for k, v in ctx.final_eval.items()
                if k.startswith("test_") and not k.startswith("test_confusion")
            }
            if _test_metrics_summary:
                # P5 P2-7 阶段2：feedback 现在是 FeedbackResult dataclass，
                # test_metrics 是可选字段，直接属性赋值
                feedback.test_metrics = _test_metrics_summary
        # P5 P2-7 阶段2：feedback.json 写入时调用 to_dict() 序列化
        (ctx.output_dir / "feedback.json").write_text(
            json.dumps(feedback.to_dict(), ensure_ascii=False, indent=2),
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
        # P5 P2-7 阶段2：feedback 是 FeedbackResult dataclass，用属性访问
        if feedback and feedback.status == "success":
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

    # 修复（任务4 / P1）：metrics.csv 双重写入修复。
    # 旧逻辑：此处手动写顶层 metrics.csv，与 CSVLogger（build_logger 中
    # CSVLogger(save_dir=output_dir, name="metrics", version="")）写入的
    # metrics/metrics.csv 内容重复，导致运行目录同时存在两份相同 metrics.csv。
    # 方案：删除手动顶层写入，仅注册 CSVLogger 产出的 metrics/metrics.csv 为产物。
    # CSVLogger 在训练过程中按 epoch 增量写入，内容与 training_log 一致。
    if training_log:
        # 修复（任务3 / P2）：CSVLogger 的 finalize 在 Pipeline.run finally 块中调用，
        # 晚于 stage_export，导致 metrics.csv 未 flush 时注册 artifact 失败
        # （warning "CSVLogger metrics.csv not found"）。
        # 方案：在注册 metrics artifact 前先 finalize csv_logger，确保文件已落盘。
        if ctx.csv_logger is not None:
            _finalize_lightning_logger(ctx.csv_logger)

        csv_logger_metrics_path = ctx.output_dir / "metrics" / "metrics.csv"
        if csv_logger_metrics_path.exists():
            # 修复（任务4 / P2）：合并 metrics.csv 的 train+val 分行为 1 行/epoch，
            # 与 training_log.jsonl 格式对齐。Lightning CSVLogger 在
            # on_train_epoch_end 和 on_validation_epoch_end 分别写入一行，
            # 导致每 epoch 有 2 行（train 行 + val 行），合并后每 epoch 1 行。
            try:
                _merge_metrics_csv(csv_logger_metrics_path)
            except Exception as e:
                _logger.warning("Failed to merge metrics.csv rows: %s", e)
            ctx.register_artifact(
                "metrics", csv_logger_metrics_path,
                kind="metrics", producer_stage="stage_build",
                content_schema=_TRAINING_LOG_ENTRY_SCHEMA,
            )
        else:
            _logger.warning(
                "CSVLogger metrics.csv not found at %s; metrics artifact not registered",
                csv_logger_metrics_path,
            )

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
            # 对称性修复：final_eval 现在含 val_* 和 test_* 指标
            content_schema={"model_id": str, "dataset": str, "final_eval": dict,
                            "test_eval": dict},
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
            # 对称性修复：content_schema 新增 test_metrics 字段
            content_schema={"status": str, "diagnosis": str, "suggestions": list,
                            "test_metrics": dict},
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
# probe_vram 依赖 ctx.model/ctx.datamodule 对象引用，同样不可序列化恢复。
_NON_SERIALIZABLE_STAGES = frozenset({"load", "build", "probe_vram", "train", "eval"})

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


def _compute_data_hash(data_root: str) -> str:
    """计算数据集目录的元数据哈希（任务2）。

    性能策略：不读取文件内容做全量 hash，只 hash 元数据
    （排序后的文件相对路径 + 文件大小 + 文件 mtime 的拼接）。
    大数据集（10k+ 文件）仍可在秒级完成。

    Args:
        data_root: 数据集根目录路径

    Returns:
        SHA256 十六进制字符串；目录不存在或为空时返回空字符串
    """
    root = Path(data_root)
    if not root.exists():
        return ""

    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            entries.append(f"{rel}|{stat.st_size}|{stat.st_mtime}")

    if not entries:
        return ""

    return sha256_str("\n".join(entries))


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

    # 任务1：config_hash 覆盖声明式配置 + 路由解析后实际生效值。
    # 旧逻辑仅 hash ExperimentConfig，不含 ctx.resolved（路由后的
    # device/batch_size/precision/...），导致同名配置不同路由产生相同
    # config_hash，溯源无法区分。合并后 config_hash 唯一标识运行时配置。
    try:
        config_hash = sha256_str(
            json.dumps(
                {**experiment_config_to_dict(ctx.config), **ctx.resolved},
                sort_keys=True, default=str,
            )
        )
    except Exception:
        config_hash = ""

    # 任务2：data_hash 从 ctx.data_hash 读取（stage_load 计算的数据集元数据哈希）。
    # 旧逻辑恒为空字符串，manifest.data_hash 无溯源价值。
    data_hash = ctx.data_hash or ""

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


def _classify_runtime_error(e: Exception, stage_name: str) -> Exception:
    """根据异常类型和 stage 上下文重新分类为具体 SenseFrame 异常类（任务4）。

    Pipeline.run 的 except 块捕获 Exception 后，调用此函数将通用异常
    包装为具体异常类（OOMError/ModelBuildError/TrainingError/
    DataCorruptedError/CheckpointError/SaveError），使 Agent 可基于
    异常类型做精确恢复决策（如 OOM → 降 batch_size，Checkpoint → 删旧 ckpt）。

    分类优先级（从高到低）：
    1. torch.cuda.OutOfMemoryError → OOMError
    2. stage="build" + RuntimeError → ModelBuildError
    3. stage="train" + RuntimeError → TrainingError
    4. stage="load" + "corrupt" in msg → DataCorruptedError
    5. "checkpoint" / ".ckpt" in msg → CheckpointError
    6. "save" / "permission" in msg → SaveError
    7. 其他 → 保持原异常

    Args:
        e: 原始异常
        stage_name: 当前执行的 stage 名（如 "build" / "train" / "load"）

    Returns:
        重新分类后的异常（具体异常类实例或原异常）
    """
    msg = str(e).lower()

    # 1. torch.cuda.OutOfMemoryError → OOMError（最高优先级，任何 stage 都需精确识别）
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", type(None))):
        return OOMError(str(e))

    # 2. stage="build" 且 RuntimeError → ModelBuildError
    if stage_name == "build" and isinstance(e, RuntimeError):
        return ModelBuildError(str(e))

    # 3. stage="train" 且 RuntimeError → TrainingError
    if stage_name == "train" and isinstance(e, RuntimeError):
        return TrainingError(str(e))

    # 4. stage="load" 且 "corrupt" in msg → DataCorruptedError
    if stage_name == "load" and "corrupt" in msg:
        return DataCorruptedError(str(e))

    # 5. "checkpoint" / ".ckpt" in msg → CheckpointError
    if "checkpoint" in msg or ".ckpt" in msg:
        return CheckpointError(str(e))

    # 6. "save" / "permission" in msg → SaveError
    if "save" in msg or "permission" in msg:
        return SaveError(str(e))

    # 7. 其他 → 保持原异常
    return e


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
            ("probe_vram", stage_probe_vram),
            ("train", stage_train),
            ("eval", stage_eval),
            ("export", stage_export),
        ])

    def replace_stage(self, name: str, fn: StageFn) -> "Pipeline":
        """替换指定 stage。"""
        # 修复（5.5）：replace_stage 完全静默，加 INFO 日志记录替换操作
        found = any(n == name for n, _ in self.stages)
        self.stages = [(n, fn if n == name else f) for n, f in self.stages]
        _logger.info(
            f"Pipeline.replace_stage: stage='{name}', found={found}, "
            f"new_fn={getattr(fn, '__name__', repr(fn))}"
        )
        return self

    def before(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 前插入 hook。"""
        # 修复（5.5）：before 完全静默，加 INFO 日志记录插入操作
        new_stages = []
        inserted = False
        for n, f in self.stages:
            if n == name:
                new_stages.append((f"before_{name}", hook))
                inserted = True
            new_stages.append((n, f))
        self.stages = new_stages
        _logger.info(
            f"Pipeline.before: inserted hook before stage='{name}', "
            f"inserted={inserted}, hook_fn={getattr(hook, '__name__', repr(hook))}"
        )
        return self

    def after(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 后插入 hook。"""
        # 修复（5.5）：after 完全静默，加 INFO 日志记录插入操作
        new_stages = []
        inserted = False
        for n, f in self.stages:
            new_stages.append((n, f))
            if n == name:
                new_stages.append((f"after_{name}", hook))
                inserted = True
        self.stages = new_stages
        _logger.info(
            f"Pipeline.after: inserted hook after stage='{name}', "
            f"inserted={inserted}, hook_fn={getattr(hook, '__name__', repr(hook))}"
        )
        return self

    def skip(self, name: str) -> "Pipeline":
        """跳过指定 stage。"""
        # 修复（5.5）：skip 完全静默，加 INFO 日志记录跳过操作
        before_count = len(self.stages)
        self.stages = [(n, f) for n, f in self.stages if n != name]
        removed = before_count - len(self.stages)
        _logger.info(
            f"Pipeline.skip: removed stage='{name}', removed_entries={removed}, "
            f"stages_before={before_count}, stages_after={len(self.stages)}"
        )
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

    def run(self, ctx: PipelineContext, *, dry_run: bool = False) -> StageResult:
        """执行 pipeline（P1：支持断点续跑）。

        依次执行所有 stage，返回最终结果。
        任一 stage 抛异常则停止并返回错误。
        每个 stage 完成后写 checkpoint；失败时也写 checkpoint（标记 failed_stage）。
        若 ctx.stage_checkpoint_path 存在，加载后跳过已完成的 stage。

        Args:
            ctx: Pipeline 上下文
            dry_run: dry-run 标志（任务3），True 时 stage_train 跳过 trainer.fit()，
                     仅输出训练 plan。也可直接在调用 run 前设置 ctx.dry_run=True。
        """
        # 修复（任务3 / P0）：从 kwargs 设置 dry-run 标志到 ctx，
        # 供 stage_train 检查后跳过 trainer.fit()，避免 dry-run 仍执行
        # 完整 fit/validation/checkpoint 产生副作用。
        if dry_run:
            ctx.dry_run = True
        # 修复（OTel 全链路失效）：Pipeline.run 入口调用 init_otel，
        # 否则 record_training_metric 全部 no-op，所有 OTel 埋点失效。
        # 旧逻辑 init_otel 从未在训练流程被调用，用户以为指标在采集实际全丢。
        try:
            from ...observability_otel import init_otel
            init_otel(
                pipeline_run_id=str(ctx.output_dir) if ctx.output_dir else "",
                trial_id=getattr(ctx, "trial_id", "") or "",
                model_id=ctx.config.scene.model_id if hasattr(ctx, "config") else "",
                dataset=ctx.config.scene.dataset if hasattr(ctx, "config") else "",
            )
        except Exception as e:
            _logger.warning(f"OTel init failed (training metrics will be no-op): {e}")

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

                # P0-1: 在 stage 边界设置 callback active 状态
                # P5 P3-15：dry_run 下 ctx.trainer 为 None，需同时检查 ctx.callbacks
                callback_lists = []
                if ctx.trainer is not None and hasattr(ctx.trainer, "callbacks"):
                    callback_lists.append(ctx.trainer.callbacks)
                if getattr(ctx, "callbacks", None):
                    callback_lists.append(ctx.callbacks)
                for cb_list in callback_lists:
                    for cb in cb_list:
                        if isinstance(cb, StageAwareCallback):
                            cb.set_active(name)
                            _logger.debug(
                                "callback %s active=%s in stage=%s",
                                type(cb).__name__, cb.is_active(), name,
                            )

                # 修复（stage 边界日志 + stage duration Timer）：
                # 旧逻辑无 stage starting/completed 边界日志，Agent 无法追踪执行进度；
                # 旧逻辑 record_training_metric value=0.0 硬编码，stage duration 恒为 0。
                # 改为：用 Timer 包裹 fn(ctx)，回填实际耗时到 OTel 指标 + 加边界日志。
                _logger.info(f"[Stage {name}] starting")
                stage_timer = Timer()
                stage_timer.__enter__()
                try:
                    ctx = fn(ctx)
                    stage_timer.__exit__()
                    stage_duration = round(stage_timer.elapsed, 3)
                    # P1：记录完成 + 写 checkpoint
                    ctx.completed_stages.append(name)
                    self._write_checkpoint(ctx)
                    # P0.2: OBP 训练指标埋点（stage 完成时记录实际耗时）
                    record_training_metric(
                        f"senseframe.stage.{name}.duration_s",
                        value=stage_duration,
                        stage=name,
                        model_id=ctx.config.scene.model_id if hasattr(ctx, "config") else "",
                        dataset=ctx.config.scene.dataset if hasattr(ctx, "config") else "",
                    )
                    _logger.info(f"[Stage {name}] completed (duration={stage_duration}s)")
                except Exception as e:
                    try:
                        stage_timer.__exit__()
                    except Exception:
                        pass
                    _logger.error(f"[Stage {name}] failed: {e}", exc_info=True)
                    # 任务4：根据异常类型和 stage 上下文重新分类为具体异常类
                    # （OOMError/ModelBuildError/TrainingError/DataCorruptedError/
                    # CheckpointError/SaveError），使 Agent 可基于异常类型精确恢复。
                    actual_error = _classify_runtime_error(e, name)
                    # P1：记录失败 stage + 写 checkpoint
                    ctx.failed_stage = name
                    ctx.failed_error = repr(actual_error)
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
                            # P3-5：重命名为 FAILED_ 前缀，隔离失败目录，
                            # 避免失败目录的 manifest/checkpoint 干扰新 run 的扫描。
                            # 保留全部失败信息（checkpoint、metrics、logs、traceback）
                            # 供 resume 和诊断；更新 ctx.output_dir 指向新位置，
                            # 让 finally 分支的 _generate_manifest 写入隔离目录。
                            failed_dir = ctx.output_dir.parent / f"FAILED_{ctx.output_dir.name}"
                            if not failed_dir.exists():
                                ctx.output_dir.rename(failed_dir)
                                _logger.info(
                                    "P3-5: failed output_dir moved to %s", failed_dir
                                )
                                ctx.output_dir = failed_dir
                                if ctx.output is not None:
                                    ctx.output.output_dir = str(failed_dir)
                            else:
                                _logger.warning(
                                    "P3-5: FAILED_ dir already exists: %s, keeping original",
                                    failed_dir,
                                )
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
            # P4-4：资源释放后刷新 checkpoint，持久化 resources_released=True。
            # 旧代码 finally 块内无 checkpoint 刷新，导致 checkpoint 永远记录释放前的状态
            #（resources_released=False），跨进程无法验证资源是否已释放。
            try:
                self._write_checkpoint(ctx)
            except Exception as e:
                _logger.warning(f"Failed to write post-release checkpoint: {e}")
            # P5 P2-9：dry_run 模式清理临时目录
            if ctx.dry_run and ctx.output_dir is not None:
                import shutil
                try:
                    shutil.rmtree(ctx.output_dir)
                    _logger.info(f"dry_run cleanup: removed temp dir {ctx.output_dir}")
                except Exception as e:
                    _logger.warning(f"dry_run cleanup failed: {e}")

    def _write_checkpoint(self, ctx: PipelineContext, failed_stage: Optional[str] = None) -> None:
        """P1：写 stage checkpoint 到 output_dir/pipeline_checkpoint.json。

        P0.8 扩展：增加 stage_outputs 字段（仅可序列化字段），统一为 OP-4 真源。

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
            # P0.8：stage 输出快照（仅可序列化字段），作为 OP-4 唯一真源
            "stage_outputs": self._serialize_stage_outputs(ctx),
            # P4-4：资源释放状态（方案 F 持久化）。
            # finally 块的 release_resources() 后会追加一次 checkpoint 刷新，
            # 此时 trainer/module 已置 None，resources_released=True。
            # 若 checkpoint 在 try 块内写入（stage 执行后），此值为 False（尚未释放）。
            "resources_released": ctx.trainer is None and ctx.module is None,
        }
        if failed_stage:
            data["failed_stage"] = failed_stage
        try:
            ckpt_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 修复（5.6）：checkpoint 写入成功路径无日志，加 INFO 留痕
            # 旧逻辑只在失败时 warning，成功路径完全静默，无法追踪 checkpoint 落盘时机
            _logger.info(
                f"checkpoint written: {ckpt_path} "
                f"(completed_stages={len(ctx.completed_stages)}, "
                f"failed_stage={failed_stage})"
            )
        except Exception as e:
            _logger.warning(f"Failed to write pipeline checkpoint: {e}")

    def _serialize_stage_outputs(self, ctx: PipelineContext) -> Dict[str, Any]:
        """序列化 ctx 中可 JSON 化的轻量字段（P0.8，OP-4 真源扩展）。

        仅提取跨 stage 传递的"结果类"字段（str/int/float/bool/list/dict），
        跳过 torch/lightning 等不可序列化对象。
        final_eval/training_log 可能含 tensor，逐项 try/except。

        Returns:
            Dict[str, Any]: 可 JSON 序列化的 stage 输出快照
        """
        snapshot: Dict[str, Any] = {}
        # 简单可序列化字段（str/int/float/bool）
        simple_fields = [
            "model_id", "dataset", "learning_mode", "num_classes",
            "trial_id", "parent_trial_id",
            "training_duration_s", "best_model_path", "best_model_score",
            "best_epoch",  # Part 2：持久化 best_epoch
            "early_stopped", "failed_stage", "failed_error",
            "route_level",
        ]
        for name in simple_fields:
            val = getattr(ctx, name, None)
            if val is None:
                continue
            try:
                json.dumps(val)
                snapshot[name] = val
            except (TypeError, ValueError):
                # 不可序列化字段跳过（如 Path 对象转 str）
                if isinstance(val, Path):
                    snapshot[name] = str(val)
                else:
                    snapshot[name] = repr(val)

        # final_eval: dict，逐项 try/except
        if ctx.final_eval:
            serializable_eval: Dict[str, Any] = {}
            for k, v in ctx.final_eval.items():
                try:
                    json.dumps(v)
                    serializable_eval[k] = v
                except (TypeError, ValueError):
                    serializable_eval[k] = repr(v)
            snapshot["final_eval"] = serializable_eval

        # P5 P2-8：feedback 序列化（FeedbackResult dataclass，调用 to_dict() 后逐项处理）
        if ctx.feedback is not None:
            # P5 P2-7 阶段2：ctx.feedback 现在是 FeedbackResult dataclass
            _feedback_dict = ctx.feedback.to_dict()
            serializable_feedback: Dict[str, Any] = {}
            for k, v in _feedback_dict.items():
                try:
                    json.dumps(v)
                    serializable_feedback[k] = v
                except (TypeError, ValueError):
                    serializable_feedback[k] = repr(v)
            snapshot["feedback"] = serializable_feedback

        # P5 P2-8：training_log 序列化（list，逐项 try/except，可能含 tensor）
        if ctx.training_log:
            serializable_log: List[Any] = []
            for entry in ctx.training_log:
                try:
                    json.dumps(entry)
                    serializable_log.append(entry)
                except (TypeError, ValueError):
                    if hasattr(entry, "to_dict"):
                        try:
                            serializable_log.append(entry.to_dict())
                        except Exception:
                            serializable_log.append(repr(entry))
                    else:
                        serializable_log.append(repr(entry))
            snapshot["training_log"] = serializable_log

        # completed_stages: list[str]，必可序列化
        if ctx.completed_stages:
            snapshot["completed_stages"] = list(ctx.completed_stages)

        return snapshot

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
        # P3-5：自动检测 FAILED_ 前缀。Pipeline.run 失败时将 output_dir 重命名
        # 为 FAILED_{原名}；resume 时若原路径不存在但 FAILED_ 候选存在，则从
        # 失败目录恢复（保留 checkpoint/metrics/logs 供续跑与诊断）。
        if not output_dir.exists():
            failed_candidate = output_dir.parent / f"FAILED_{output_dir.name}"
            if failed_candidate.exists():
                _logger.info(
                    "P3-5: detected FAILED_ prefix, resuming from %s", failed_candidate
                )
                output_dir = failed_candidate
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pipeline checkpoint found at {ckpt_path}")

        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        completed = ckpt.get("completed_stages", [])

        # 任务5：读取 failed_error 做诊断，输出恢复建议。
        # 不改变续跑行为（仍从 completed_stages 推断），仅增加诊断日志，
        # 帮助 Agent 理解上次失败原因并采取针对性措施。
        # failed_error 可能存于顶层（旧格式）或 stage_outputs 内（_serialize_stage_outputs）
        stage_outputs = ckpt.get("stage_outputs", {})
        failed_error = ckpt.get("failed_error") or stage_outputs.get("failed_error") or ""
        if failed_error:
            failed_error_lower = failed_error.lower()
            if "oom" in failed_error_lower or "outofmemory" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因 OOM 失败，建议降低 batch_size "
                    "(ctx.resolved['batch_size']) 或减少 num_workers 后重试"
                )
            if "datacorrupted" in failed_error_lower or "corrupt" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因数据损坏失败，建议检查数据集完整性 "
                    "(文件是否完整、未损坏) 后重试"
                )
            if "checkpoint" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因 checkpoint 问题失败，建议检查 checkpoint "
                    "文件是否损坏（可能需要删除旧 checkpoint 重新训练）"
                )

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
        # P4-5：从 ctx 复制 feedback 到 TrainOutput，供 HPO tracker 消费。
        # 旧代码未复制，导致 hpo.py 硬编码 feedback={"status":"success"}，
        # 探索-反馈回路断裂（underfitting trial 被误标为 success）。
        if ctx.feedback is not None:
            ctx.output.feedback = ctx.feedback
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
    "verify_artifacts_recursive",
    "verify_manifest_schema",
    "verify_artifacts_full",
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


def verify_artifacts_recursive(output_dir, max_depth: int = 3) -> Dict[str, Dict[str, bool]]:
    """递归校验 output_dir 及子目录中所有 manifest.json 的产物 hash（P3-4）。

    用于 HPO 多 trial 场景：output_dir/ 下可能有 trial_0/、trial_1/ 等子目录，
    每个子目录有自己的 manifest.json。单 run 场景请用 verify_artifacts。

    Args:
        output_dir: 根输出目录
        max_depth: 最大递归深度

    Returns:
        {子目录相对路径: {产物名: hash 是否匹配}}，根目录用 "." 表示
    """
    return _verify_artifacts_recursive(Path(output_dir), max_depth=max_depth)


def verify_manifest_schema(manifest_path) -> List[str]:
    """校验 manifest.json schema 完整性，返回缺失字段列表（P5 P3-11）。

    Args:
        manifest_path: manifest.json 路径

    Returns:
        缺失的 manifest 字段名列表（空列表表示完整）
    """
    return _verify_manifest_schema(Path(manifest_path))


def verify_artifacts_full(output_dir) -> Dict[str, Any]:
    """完整校验：hash + schema + 必填产物（P5 P3-11）。

    Args:
        output_dir: 训练输出目录

    Returns:
        {
            "hash_check": {产物名: bool},
            "manifest_schema_missing": [缺失字段],
            "missing_artifacts": [缺失产物名],
        }
    """
    return _verify_artifacts_full(Path(output_dir))
