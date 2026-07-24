"""PipelineContext + Stage 结果类型 + 字段填充映射 + 资源清理辅助。

包含：
- _FIELD_FILL_STAGE：字段名 → 首次填充的 stage 名映射
- _PSEUDO_STAGES：伪 stage 集合（init/agent）
- _TRAINING_LOG_ENTRY_SCHEMA：training_log 字段结构契约
- _finalize_lightning_logger：Lightning Logger 清理辅助
- PipelineContext：Stage 间共享的上下文
- StageResult / ReadinessReport / DanglingRef
"""
from __future__ import annotations

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
    Tuple,
    TYPE_CHECKING,
)

import torch

from ...config import ExperimentConfig
from ....common import load_checkpoint_flexible
from ....observability import IncrementalLogWriter, setup_logging as _setup_logging
from ....routing import ResourceReport  # noqa: F401 (re-exported via routing)
from ....schemas import TrainOutput
from ..artifacts import ArtifactDescriptor, sha256_file, sha256_str

if TYPE_CHECKING:
    from ...datamodule import GenericDataModule  # noqa: F401
    from ...module import GenericLightningModule  # noqa: F401
    from ....core.features import FeatureSpec  # noqa: F401
    from ....core.profiler import DataProfile  # noqa: F401
    from ....core.task import TaskSpec  # noqa: F401
    from ....observability import TrainingMonitor  # noqa: F401
    from ....scenes.base import DatasetBundle, SceneMeta  # noqa: F401
    from .protocols import (
        SceneProtocol, SceneMetaProtocol, TaskSpecProtocol,
        FeatureSpecProtocol, ModelProtocol, DataModuleProtocol,
        TrainerProtocol, LoggerProtocol,
    )

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
    "pretrain_checkpoint": "stage_load",  # v2 差距 3：预训练 checkpoint 路径（scene.params.pretrain_source 触发）
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
    # P1.1 Multi-fidelity 实时早停修复 — IntermediateMetricLogger 回调写入（trainer.should_stop=True 触发）
    "pruned": "stage_train",       # 训练是否被实时剪枝（True 表示 pruner 决定提前终止）
    "pruned_epoch": "stage_train",  # 剪枝发生时的 epoch 号（1-indexed，None 表示未剪枝）
    # agent-controlled（RFC-002 探索状态 + 断点续跑）
    "trial_id": "agent",
    "parent_trial_id": "agent",
    "exploration_history": "agent",
    "extra": "agent",
    "completed_stages": "agent",
    "stage_checkpoint_path": "agent",
    "failed_stage": "agent",       # P0.1：Pipeline.run except 块写入（错误路径，agent 可观测）
    "failed_error": "agent",       # P0.1：同上
    # P1.1 Multi-fidelity 实时早停修复 — agent 通过 ctx 注入 pruner 实例
    # stage_build 读取 ctx.pruner 并注入到 IntermediateMetricLogger，让每个 epoch end 调用 should_prune
    "pruner": "agent",
    # stage_eval
    "final_eval": "stage_eval",
    "training_log": "stage_eval",
    "early_stopped": "stage_eval",
    "feedback": "stage_eval",
    # RFC-004 方案 G：产物溯源注册表（各 stage 注册，stage_export 是主要注册点）
    "artifact_registry": "stage_export",  # P0.1：声明填充 stage，消除 schema "unknown"
}


# 伪 stage 集合（模块级常量，避免被 dataclasses.fields 识别为 PipelineContext 字段）：
# 不由任何 pipeline stage 产出，由构造函数或 Agent 运行时注入。
# schema() 用此集合设置 is_pseudo_stage=True，让 Agent 程序化区分真实 stage 产出
# 与非 stage 产出字段，避免误把 "init"/"agent" 当作 stage 名传给 stage_io()。
_PSEUDO_STAGES: frozenset = frozenset({"init", "agent"})


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
    log_writer: Optional[IncrementalLogWriter] = None
    # 任务2：数据集元数据哈希（stage_load 计算，_generate_manifest 写入 manifest）
    data_hash: str = ""
    # v2 差距 3：预训练 checkpoint 路径（stage_load 写入，scene.params.pretrain_source 触发）
    pretrain_checkpoint: Optional[str] = None
    csv_logger: Optional["LoggerProtocol"] = None  # P2.1: pl.Logger 实例（LoggerProtocol 契约）
    report: Optional["ResourceReport"] = None
    route_level: str = ""
    route_config: Dict[str, Any] = field(default_factory=dict)
    # RFC-002 阶段 J：探索状态
    trial_id: str = ""                              # 当前试验 ID
    parent_trial_id: Optional[str] = None           # 父试验 ID（支持回溯）
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)  # 已探索策略组合
    # P1.1 Multi-fidelity 实时早停修复 — agent 注入 pruner 实例
    # stage_build 读取此字段注入到 IntermediateMetricLogger，每个 epoch end 调 should_prune
    # None 时退化为旧路径（MethodRunner 事后剪枝），向后兼容
    pruner: Optional[Any] = None  # Pruner Protocol 实例（duck-typed，避免循环导入用 Any）
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
    # P1.1 Multi-fidelity 实时早停修复 — 替代 MethodRunner 事后剪枝
    # 旧路径（事后剪枝）：训练完整跑完 → MethodRunner 调 pruner.should_prune → 标记 PRUNED
    #   问题：浪费已无意义的训练算力（如 epoch 5 已可剪枝，但训练到 epoch 50 才停止）
    # 新路径（实时早停）：每个 epoch end → IntermediateMetricLogger 调 pruner.should_prune
    #   → True 则 trainer.should_stop=True → Lightning 提前终止训练
    # stage_train 写入 pruned/pruned_epoch；stage_build 通过 ctx.pruner 注入到回调
    pruned: bool = False                                            # stage_train 写入（trainer.should_stop 触发后置 True）
    pruned_epoch: Optional[int] = None                              # stage_train 写入（剪枝发生的 epoch 号，1-indexed）
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
        from ....common.path_safe import safe_relative_path
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

    # 伪 stage 集合：不由任何 pipeline stage 产出，由构造函数或 Agent 运行时注入。
    # schema() 用此集合设置 is_pseudo_stage=True，让 Agent 程序化区分真实 stage 产出
    # 与非 stage 产出字段，避免误把 "init"/"agent" 当作 stage 名传给 stage_io()。
    # （模块级常量 _PSEUDO_STAGES 定义在 _FIELD_FILL_STAGE 旁，避免被 dataclasses.fields
    # 识别为 PipelineContext 字段触发 test_field_fill_stage_complete 误报。）

    @classmethod
    def schema(cls) -> dict:
        """返回完整字段契约（RFC-003 DSP-1）。

        返回 JSON 可序列化 dict，含 schema_version 与每个字段的元信息：
        - name: 字段名
        - type: 类型字符串
        - fill_stage: _FIELD_FILL_STAGE 中的原始值（"init"/"agent"/"stage_validate"/...）
        - stage_name: 真实 pipeline stage 名（去掉 "stage_" 前缀），伪 stage 为 None。
                      与 list_stages() / stage_io() / Pipeline.default() 返回的 stage 名对齐。
        - is_pseudo_stage: True 表示 fill_stage 是伪 stage（init/agent），
                          非 pipeline stage 产出，由构造函数或 Agent 注入。
        - has_default: 是否有默认值
        """
        fields_info = []
        for f in _dataclass_fields(cls):
            fill_stage = _FIELD_FILL_STAGE.get(f.name, "unknown")
            is_pseudo = fill_stage in _PSEUDO_STAGES
            # 真实 stage 名：去掉 "stage_" 前缀（如 "stage_validate" → "validate"），
            # 与 Pipeline.default() 的 tuple 第一项 / list_stages() 输出对齐。
            # 伪 stage / "unknown" 无对应真实 stage，stage_name=None。
            if is_pseudo or fill_stage == "unknown":
                stage_name = None
            elif fill_stage.startswith("stage_"):
                stage_name = fill_stage[len("stage_"):]
            else:
                stage_name = fill_stage
            fields_info.append({
                "name": f.name,
                "type": str(f.type) if hasattr(f, "type") else "Any",
                "fill_stage": fill_stage,
                "stage_name": stage_name,
                "is_pseudo_stage": is_pseudo,
                "has_default": (
                    f.default is not MISSING
                    or f.default_factory is not MISSING  # type: ignore[misc]
                ),
            })
        return {
            "schema_version": "1.1.0",  # 1.1: 新增 stage_name + is_pseudo_stage 字段
            "fields": fields_info,
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
