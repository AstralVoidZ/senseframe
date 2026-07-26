"""RFC-003 OP：编排协议 — 让 SenseFrame 可被外部编排器托管。

对齐 K8s Operator Pattern（CRD + Reconciliation Loop）+ Argo Workflows（Pipeline/PipelineRun 分离）+ CloudEvents 1.0。

让 SenseFrame Pipeline 可被 K8s Operator / Argo / Airflow / 自研 AutoML 主控托管。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from typing import Protocol, runtime_checkable

if TYPE_CHECKING:
    # 仅用于类型注解，避免运行时循环导入
    from .orchestration_store import OrchestrationStore

# 修复：模块顶部未 import logging，load_checkpoint 等方法引用 logger 时 NameError。
# 旧逻辑传含 .. 的 output_dir 时直接崩溃，无法记录审计日志。
logger = logging.getLogger(__name__)


# ============================================================
# OP-1: Pipeline 定义（声明式，与运行实例分离）
# ============================================================

@dataclass
class StageTemplate:
    """Stage 模板（OP-1）。"""
    name: str
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryPolicy:
    """重试策略（OP-1）。"""
    max_retries: int = 3
    backoff: float = 1.0  # 退避秒数
    retry_on_stages: List[str] = field(default_factory=list)  # 空=所有 stage


@dataclass
class CheckpointPolicy:
    """Checkpoint 策略（OP-1）。"""
    enabled: bool = True
    interval: int = 1  # 每 N 个 stage 保存一次
    storage_uri: str = ""  # 空=本地


@dataclass
class PipelineDef:
    """Pipeline 定义（OP-1，声明式，与运行实例分离）。

    可序列化为 YAML/JSON。同一定义可被多次实例化为 PipelineRun。
    """
    name: str
    stages: List[StageTemplate] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    scene: Dict[str, Any] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stages": [{"name": s.name, "reads": s.reads, "writes": s.writes, "config": s.config} for s in self.stages],
            "config": self.config,
            "scene": self.scene,
            "retry_policy": {"max_retries": self.retry_policy.max_retries, "backoff": self.retry_policy.backoff, "retry_on_stages": self.retry_policy.retry_on_stages},
            "checkpoint_policy": {"enabled": self.checkpoint_policy.enabled, "interval": self.checkpoint_policy.interval, "storage_uri": self.checkpoint_policy.storage_uri},
        }

    @classmethod
    def default(cls, name: str = "default") -> "PipelineDef":
        """默认 Pipeline 定义（9 stage）。

        P0.7：stage 顺序与 Pipeline.default() 对齐，
        使 PipelineDef.default().materialize() 与 Pipeline.default() 等价。
        方案 B：新增 probe_vram stage（位于 build 和 train 之间，动态显存探测）。
        """
        return cls(
            name=name,
            stages=[
                StageTemplate("validate", reads=["config"], writes=["scene", "meta", "model_id", "dataset", "learning_mode"]),
                StageTemplate("preflight", reads=["config", "scene"], writes=["report", "route_level", "route_config", "output"]),
                StageTemplate("load", reads=["config", "scene", "dataset"], writes=["bundle", "data_profile", "output_dir"]),
                StageTemplate("resolve", reads=["config", "scene", "dataset"], writes=["task_spec", "feature_spec", "resolved", "lightning_params"]),
                StageTemplate("build", reads=["config", "scene", "model_id", "bundle"], writes=["model", "datamodule", "module", "callbacks"]),
                StageTemplate("probe_vram", reads=["model", "datamodule", "resolved", "report"], writes=["vram_probe_result"]),
                StageTemplate("train", reads=["config", "model", "datamodule", "module"], writes=["trainer", "output"]),
                StageTemplate("eval", reads=["config", "trainer", "module", "datamodule"], writes=["output"]),
                StageTemplate("export", reads=["config", "model", "module", "output"], writes=["output"]),
            ],
        )

    def materialize(self) -> "Pipeline":
        """将声明式定义物化为可执行 Pipeline（P0.7，OP-1）。

        按 stages 列表查找 stage 函数，构造 Pipeline 执行器。
        复用 Pipeline.default() 内部结构，不引入新全局表。

        仅支持内置 8 个 stage（validate/preflight/load/resolve/build/train/eval/export）。
        自定义 stage 需通过 Pipeline.replace_stage 注入。

        Returns:
            Pipeline: 可执行 Pipeline 实例，stages 列表与 PipelineDef.stages 顺序一致

        Raises:
            ValueError: 若 stages 中含未知 stage 名

        验证（P0.9 test_pipelinedef_materialize）：
            PipelineDef.default().materialize().stages 名列表与
            Pipeline.default().stages 名列表一致
        """
        # 局部导入避免循环依赖（pipeline.py 不依赖 orchestration.py）
        from .engine.runner.pipeline import Pipeline

        # 复用 Pipeline.default() 内部结构作为 stage 函数查找表
        builtin_stages: Dict[str, Any] = dict(Pipeline.default().stages)

        stages: List[tuple] = []
        for stage_tmpl in self.stages:
            stage_fn = builtin_stages.get(stage_tmpl.name)
            if stage_fn is None:
                raise ValueError(
                    f"unknown stage: {stage_tmpl.name!r}; "
                    f"builtin stages: {list(builtin_stages.keys())}. "
                    f"自定义 stage 请用 Pipeline.replace_stage 注入。"
                )
            stages.append((stage_tmpl.name, stage_fn))
        return Pipeline(stages=stages)


# ============================================================
# OP-2: PipelineRun 实例 + OP-3: 生命周期状态机
# ============================================================

@dataclass
class StageStatus:
    """Stage 运行状态（OP-2）。"""
    name: str
    phase: str = "pending"  # pending / running / succeeded / failed / skipped
    started_at: str = ""
    finished_at: str = ""
    checkpoint_uri: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "phase": self.phase, "started_at": self.started_at,
                "finished_at": self.finished_at, "checkpoint_uri": self.checkpoint_uri, "error": self.error}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageStatus":
        """从 dict 构造 StageStatus（P3.4.1，反序列化）。

        缺失字段使用默认值（与 dataclass 默认值对齐），
        保证旧版序列化数据可向后兼容加载。
        """
        return cls(
            name=d.get("name", ""),
            phase=d.get("phase", "pending"),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            checkpoint_uri=d.get("checkpoint_uri", ""),
            error=d.get("error", ""),
        )


# OP-3: PipelineRun 状态机 phase 常量
PHASE_PENDING = "pending"
PHASE_RUNNING = "running"
PHASE_SUCCEEDED = "succeeded"
PHASE_FAILED = "failed"
PHASE_PAUSED = "paused"

# 合法状态转换
_VALID_TRANSITIONS = {
    PHASE_PENDING: {PHASE_RUNNING},
    PHASE_RUNNING: {PHASE_SUCCEEDED, PHASE_FAILED, PHASE_PAUSED},
    PHASE_PAUSED: {PHASE_RUNNING},
    PHASE_FAILED: {PHASE_RUNNING},  # retry
    PHASE_SUCCEEDED: set(),  # 终态
}


@dataclass
class PipelineRun:
    """PipelineRun 实例（OP-2 + OP-3 状态机）。

    运行实例，含 status.phase 状态机。
    """
    run_id: str
    pipeline_ref: str  # PipelineDef.name
    owner_reference: Optional[str] = None  # P2.2: 归属 PipelineDef 的强引用（K8s CRD owner_reference 语义）
    params: Dict[str, Any] = field(default_factory=dict)
    checkpoint_uri: str = ""
    # status
    phase: str = PHASE_PENDING
    stages: List[StageStatus] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    output_uri: str = ""
    error: str = ""
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "pipeline_ref": self.pipeline_ref,
            "owner_reference": self.owner_reference,
            "params": self.params, "checkpoint_uri": self.checkpoint_uri,
            "status": {
                "phase": self.phase,
                "stages": [s.to_dict() for s in self.stages],
                "started_at": self.started_at, "finished_at": self.finished_at,
                "output_uri": self.output_uri, "error": self.error,
                "retry_count": self.retry_count,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineRun":
        """从 dict 构造 PipelineRun（P3.4.1，反序列化）。

        递归调用 StageStatus.from_dict 构造 stages 列表。
        缺失字段使用默认值，保证旧版序列化数据可向后兼容加载。

        to_dict 将 status 字段（phase/stages/started_at/finished_at/output_uri/error/retry_count）
        嵌套在 "status" key 下，from_dict 从该 key 解包。
        """
        status = d.get("status", {}) or {}
        stages_data = status.get("stages", []) or []
        return cls(
            run_id=d.get("run_id", ""),
            pipeline_ref=d.get("pipeline_ref", ""),
            owner_reference=d.get("owner_reference", None),
            params=d.get("params", {}) or {},
            checkpoint_uri=d.get("checkpoint_uri", ""),
            phase=status.get("phase", PHASE_PENDING),
            stages=[StageStatus.from_dict(s) for s in stages_data],
            started_at=status.get("started_at", ""),
            finished_at=status.get("finished_at", ""),
            output_uri=status.get("output_uri", ""),
            error=status.get("error", ""),
            retry_count=status.get("retry_count", 0),
        )

    def transition(self, new_phase: str) -> None:
        """状态转换（OP-3）。非法转换抛 ValueError。"""
        if new_phase not in _VALID_TRANSITIONS.get(self.phase, set()):
            raise ValueError(f"Invalid transition: {self.phase} → {new_phase}")
        self.phase = new_phase
        if new_phase == PHASE_RUNNING and not self.started_at:
            self.started_at = datetime.now().isoformat()
        if new_phase in (PHASE_SUCCEEDED, PHASE_FAILED):
            self.finished_at = datetime.now().isoformat()


# ============================================================
# OP-4: Checkpoint（冷启动/热续跑）
# ============================================================

@dataclass
class CheckpointSpec:
    """Checkpoint 规格（OP-4）。"""
    run_id: str
    stage_name: str
    checkpoint_uri: str
    stage_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "stage_name": self.stage_name,
                "checkpoint_uri": self.checkpoint_uri,
                "stage_snapshot": self.stage_snapshot, "timestamp": self.timestamp}


# ============================================================
# OP-5: CloudEvent 事件流（对齐 CloudEvents 1.0）
# ============================================================

@dataclass
class CloudEvent:
    """CloudEvent（OP-5，对齐 CloudEvents 1.0）。"""
    specversion: str = "1.0"
    id: str = ""
    source: str = ""
    type: str = ""
    time: str = ""
    datacontenttype: str = "application/json"
    subject: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        if not self.time:
            self.time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specversion": self.specversion, "id": self.id,
            "source": self.source, "type": self.type,
            "time": self.time, "datacontenttype": self.datacontenttype,
            "subject": self.subject,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# 事件类型常量
EVENT_PIPELINE_STARTED = "senseframe.pipeline.started"
EVENT_PIPELINE_SUCCEEDED = "senseframe.pipeline.succeeded"
EVENT_PIPELINE_FAILED = "senseframe.pipeline.failed"
EVENT_PIPELINE_PAUSED = "senseframe.pipeline.paused"
EVENT_PIPELINE_RESUMED = "senseframe.pipeline.resumed"
EVENT_STAGE_STARTED = "senseframe.stage.started"
EVENT_STAGE_SUCCEEDED = "senseframe.stage.succeeded"
EVENT_STAGE_FAILED = "senseframe.stage.failed"
EVENT_TRIAL_COMPLETED = "senseframe.trial.completed"
EVENT_INFERENCE_SERVED = "senseframe.inference.served"


def make_event(event_type: str, run_id: str, data: Dict[str, Any]) -> CloudEvent:
    """创建 CloudEvent（OP-5）。"""
    return CloudEvent(
        source=f"/senseframe/pipeline/{run_id}",
        type=event_type,
        data=data,
    )


# ============================================================
# P3.4.3: CloudEvent 外部 sink
# ============================================================

@runtime_checkable
class EventSink(Protocol):
    """CloudEvent 外部 sink 协议（P3.4.3）。

    任何含 ``emit(event: CloudEvent) -> None`` 方法的对象均满足此协议。
    Orchestrator._event_sink 接受任何 EventSink 实现（FileEventSink / Kafka / Webhook 等）。
    """

    def emit(self, event: CloudEvent) -> None: ...


class FileEventSink:
    """CloudEvent 文件 sink（P3 默认实现，P3.4.3）。

    将 CloudEvent 以 JSONL 格式追加写入日志文件（每行一个 event JSON）。
    sink 异常不影响主流程（Orchestrator._emit_event 中 try/except 兜底）。
    """

    def __init__(self, log_path: Any):
        """Args:
            log_path: 日志文件路径（str / Path）。父目录自动创建。
        """
        self.log_path = Path(log_path)
        # 父目录自动创建（与 FileOrchestrationStore 行为一致）
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: CloudEvent) -> None:
        """将 CloudEvent 追加写入 JSONL 日志（每行一个 event JSON）。"""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")


# ============================================================
# P3.4.4: K8s Operator 适配层（接口准备，P4 真实实现）
# ============================================================

class K8sOperatorAdapter:
    """K8s Operator 适配层（P3 接口准备，P4 真实实现）。

    将 OP PipelineRun 映射为 K8s Custom Resource（CRD: senseframe.io/v1 PipelineRun）。
    P3 仅提供 to_cr_manifest / from_cr_manifest 双向序列化方法，
    真实 Operator 驱动（reconciliation loop、CR watch、status subresource 回写）
    推迟到 P4（需 kopf/argo 依赖 + K8s 集群验证）。

    CR manifest 结构对齐 K8s CRD 规范：
        apiVersion: senseframe.io/v1
        kind: PipelineRun
        metadata:
          name: <run_id>
          ownerReferences: [...]  # 仅当 run.owner_reference 非 None
        spec:
          pipelineRef / params / checkpointUri
        status:
          phase / stages / startedAt / finishedAt / outputUri / error / retryCount
    """

    API_VERSION = "senseframe.io/v1"
    KIND = "PipelineRun"
    OWNER_API_VERSION = "senseframe.io/v1"
    OWNER_KIND = "PipelineDef"

    def to_cr_manifest(self, run: PipelineRun) -> Dict[str, Any]:
        """PipelineRun → K8s Custom Resource manifest（P3.4.4）。

        ownerReferences 仅在 run.owner_reference 非 None 时包含（K8s CRD 语义）。
        """
        manifest: Dict[str, Any] = {
            "apiVersion": self.API_VERSION,
            "kind": self.KIND,
            "metadata": {
                "name": run.run_id,
            },
            "spec": {
                "pipelineRef": run.pipeline_ref,
                "params": run.params,
                "checkpointUri": run.checkpoint_uri,
            },
            "status": {
                "phase": run.phase,
                "startedAt": run.started_at,
                "finishedAt": run.finished_at,
                "outputUri": run.output_uri,
                "error": run.error,
                "retryCount": run.retry_count,
                "stages": [s.to_dict() for s in run.stages],
            },
        }
        if run.owner_reference:
            manifest["metadata"]["ownerReferences"] = [{
                "apiVersion": self.OWNER_API_VERSION,
                "kind": self.OWNER_KIND,
                "name": run.owner_reference,
            }]
        return manifest

    def from_cr_manifest(self, manifest: Dict[str, Any]) -> PipelineRun:
        """K8s Custom Resource → PipelineRun（P3.4.4）。

        支持最小 manifest（仅 metadata.name + spec.pipelineRef），
        缺失字段使用默认值（与 PipelineRun.from_dict 行为一致）。
        """
        meta = manifest.get("metadata", {}) or {}
        spec = manifest.get("spec", {}) or {}
        status = manifest.get("status", {}) or {}
        stages_data = status.get("stages", []) or []
        owner_refs = meta.get("ownerReferences", []) or []
        owner_reference = owner_refs[0].get("name") if owner_refs else None
        return PipelineRun(
            run_id=meta.get("name", ""),
            pipeline_ref=spec.get("pipelineRef", ""),
            owner_reference=owner_reference,
            params=spec.get("params", {}) or {},
            checkpoint_uri=spec.get("checkpointUri", ""),
            phase=status.get("phase", PHASE_PENDING),
            stages=[StageStatus.from_dict(s) for s in stages_data],
            started_at=status.get("startedAt", ""),
            finished_at=status.get("finishedAt", ""),
            output_uri=status.get("outputUri", ""),
            error=status.get("error", ""),
            retry_count=status.get("retryCount", 0),
        )


# ============================================================
# OP-6: 编排器接口
# ============================================================

class Orchestrator:
    """编排器（OP-6）。

    供 AutoML 主控调用，管理 Pipeline 生命周期。
    对齐 K8s Controller reconciliation loop。
    P0.1: reconcile() 桥接 Pipeline 执行器，实现「观察状态 → 驱动 stage → 回写状态」闭环。

    P3.4 扩展：
    - ``store`` 参数（可选）：OrchestrationStore 持久化后端，None 时退化为纯内存（向后兼容）。
    - ``event_sink`` 参数（可选）：CloudEvent 外部 sink，None 时不写外部日志（向后兼容）。
    - ``recover()`` 方法：从 store 加载所有 run，重启后恢复状态。
    """

    def __init__(self, store: Optional["OrchestrationStore"] = None,
                 event_sink: Optional[EventSink] = None):
        self._pipelines: Dict[str, PipelineDef] = {}  # name -> PipelineDef
        self._runs: Dict[str, PipelineRun] = {}  # run_id -> PipelineRun
        self._checkpoints: Dict[str, List[CheckpointSpec]] = {}  # run_id -> checkpoints
        self._subscribers: Dict[str, List[Callable[[CloudEvent], None]]] = {}  # event_type -> callbacks
        # P0.1: PipelineRun ↔ PipelineContext 映射（reconcile 驱动执行）
        self._contexts: Dict[str, Any] = {}  # run_id -> PipelineContext
        self._lock = threading.RLock()  # O4: RLock 可重入，避免嵌套获取死锁
        # P2.11: 异步执行支持（reconcile 真循环）
        # ThreadPoolExecutor 用于 start_and_execute 异步提交 _execute_pipeline
        # 单独的 _run_futures 跟踪每个 run 的 Future（用于 wait_for_completion）
        self._executor: Optional[ThreadPoolExecutor] = None
        self._run_futures: Dict[str, Future] = {}
        self._async_lock = threading.Lock()  # 保护 _run_futures
        # P3.4.2: 持久化后端（None 时退化为纯内存，向后兼容）
        self._store: Optional["OrchestrationStore"] = store
        # P3.4.3: CloudEvent 外部 sink（None 时不写外部日志，向后兼容）
        self._event_sink: Optional[EventSink] = event_sink

    @property
    def is_shutdown(self) -> bool:
        """返回 Orchestrator 是否已 shutdown。"""
        return self._executor is None

    @property
    def store(self):
        """返回 OrchestrationStore（可能为 None）。"""
        return self._store

    @property
    def event_sink(self):
        """返回 EventSink（可能为 None）。"""
        return self._event_sink

    def _persist_run(self, run: PipelineRun) -> None:
        """持久化单个 run 到 store（P3.4.2 内部辅助）。

        store 为 None 时 no-op（向后兼容）。store 异常不影响主流程
        （捕获后吞掉，避免持久化失败阻断编排——K8s Controller 同样语义：
        status subresource 回写失败不应阻断 reconciliation）。
        """
        if self._store is None:
            return
        try:
            self._store.save_run(run)
        except Exception as e:
            # O5: 禁止静默吞持久化异常，至少 warning 留痕（与 _emit_event 风格对齐）
            logger.warning(
                "persist_run failed (run_id=%s): %s",
                run.run_id, e,
            )

    def create_pipeline(self, pipeline_def: PipelineDef) -> str:
        """注册 Pipeline 定义（OP-1/6）。"""
        with self._lock:
            self._pipelines[pipeline_def.name] = pipeline_def
        return pipeline_def.name

    def create_run(self, pipeline_id: str, params: Optional[Dict[str, Any]] = None,
                   checkpoint_uri: str = "") -> str:
        """创建 PipelineRun（OP-2/6）。"""
        if pipeline_id not in self._pipelines:
            raise KeyError(f"Pipeline '{pipeline_id}' not found")
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        pdef = self._pipelines[pipeline_id]
        run = PipelineRun(
            run_id=run_id, pipeline_ref=pipeline_id,
            owner_reference=pipeline_id,  # P2.2: 归属 PipelineDef
            params=params or {}, checkpoint_uri=checkpoint_uri,
            stages=[StageStatus(name=s.name) for s in pdef.stages],
        )
        with self._lock:
            self._runs[run_id] = run
            self._checkpoints[run_id] = []
            self._contexts[run_id] = None  # P0.1: 待绑定 PipelineContext
        # P3.4.2: 持久化（store 为 None 时 no-op）
        self._persist_run(run)
        return run_id

    def start(self, run_id: str) -> None:
        """启动 PipelineRun（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_RUNNING)
            # 修复（5.14）：状态转换只发 CloudEvent 不写日志，加 INFO 留痕
            logger.info(
                "OP start: run_id=%s, phase transition: %s -> %s",
                run_id, old_phase, run.phase,
            )
            self._emit_event(EVENT_PIPELINE_STARTED, run_id, {"phase": run.phase})
            self._persist_run(run)

    def pause(self, run_id: str) -> None:
        """暂停 PipelineRun（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_PAUSED)
            logger.info(
                "OP pause: run_id=%s, phase transition: %s -> %s",
                run_id, old_phase, run.phase,
            )
            self._emit_event(EVENT_PIPELINE_PAUSED, run_id, {"phase": run.phase})
            self._persist_run(run)

    def resume(self, run_id: str) -> None:
        """恢复 PipelineRun（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_RUNNING)
            logger.info(
                "OP resume: run_id=%s, phase transition: %s -> %s",
                run_id, old_phase, run.phase,
            )
            self._emit_event(EVENT_PIPELINE_RESUMED, run_id, {"phase": run.phase})
            self._persist_run(run)

    def retry(self, run_id: str) -> None:
        """重试 PipelineRun（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_RUNNING)
            run.retry_count += 1
            logger.info(
                "OP retry: run_id=%s, phase transition: %s -> %s, retry_count=%d",
                run_id, old_phase, run.phase, run.retry_count,
            )
            self._emit_event(EVENT_PIPELINE_STARTED, run_id, {"phase": run.phase, "retry": run.retry_count})
            self._persist_run(run)

    def stop(self, run_id: str) -> None:
        """停止 PipelineRun（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_FAILED)
            run.error = "Stopped by orchestrator"
            logger.info(
                "OP stop: run_id=%s, phase transition: %s -> %s",
                run_id, old_phase, run.phase,
            )
            self._emit_event(EVENT_PIPELINE_FAILED, run_id, {"phase": run.phase, "error": run.error})
            self._persist_run(run)

    def complete(self, run_id: str, output_uri: str = "") -> None:
        """标记 PipelineRun 成功完成（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_SUCCEEDED)
            run.output_uri = output_uri
            logger.info(
                "OP complete: run_id=%s, phase transition: %s -> %s, output_uri=%s",
                run_id, old_phase, run.phase, output_uri,
            )
            self._emit_event(EVENT_PIPELINE_SUCCEEDED, run_id, {"phase": run.phase, "output_uri": output_uri})
            self._persist_run(run)

    def fail(self, run_id: str, error: str, stage_name: str = "") -> None:
        """标记 PipelineRun 失败（OP-6）。"""
        with self._lock:
            run = self._get_run(run_id)
            old_phase = run.phase
            run.transition(PHASE_FAILED)
            run.error = error
            if stage_name:
                for s in run.stages:
                    if s.name == stage_name:
                        s.phase = "failed"
                        s.error = error
                        break
            logger.info(
                "OP fail: run_id=%s, phase transition: %s -> %s, stage=%s, error=%s",
                run_id, old_phase, run.phase, stage_name, error,
            )
            self._emit_event(EVENT_PIPELINE_FAILED, run_id, {"phase": run.phase, "error": error, "stage": stage_name})
            self._persist_run(run)

    def update_stage(self, run_id: str, stage_name: str, phase: str,
                     checkpoint_uri: str = "", error: str = "") -> None:
        """更新 stage 状态（OP-6）。"""
        run = self._get_run(run_id)
        for s in run.stages:
            if s.name == stage_name:
                old_phase = s.phase
                s.phase = phase
                if phase == "running" and not s.started_at:
                    s.started_at = datetime.now().isoformat()
                if phase in ("succeeded", "failed") and not s.finished_at:
                    s.finished_at = datetime.now().isoformat()
                if checkpoint_uri:
                    s.checkpoint_uri = checkpoint_uri
                if error:
                    s.error = error
                # 发 stage 事件
                event_type = {
                    "running": EVENT_STAGE_STARTED,
                    "succeeded": EVENT_STAGE_SUCCEEDED,
                    "failed": EVENT_STAGE_FAILED,
                }.get(phase)
                if event_type and old_phase != phase:
                    self._emit_event(event_type, run_id, {"stage_name": stage_name, "phase": phase})
                break

    def save_checkpoint(self, run_id: str, stage_name: str, checkpoint_uri: str,
                        stage_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """保存 Checkpoint（OP-4/6）。

        P0.8：建议改用 save_checkpoint_from_file，以 pipeline_checkpoint.json
        为唯一真源，避免两套快照并行。此方法保留向后兼容（外部调用方可用）。
        """
        ckpt = CheckpointSpec(
            run_id=run_id, stage_name=stage_name, checkpoint_uri=checkpoint_uri,
            stage_snapshot=stage_snapshot or {}, timestamp=datetime.now().isoformat(),
        )
        with self._lock:
            self._checkpoints.setdefault(run_id, []).append(ckpt)

    def save_checkpoint_from_file(self, run_id: str, stage_name: str,
                                   output_dir: Any) -> Optional[CheckpointSpec]:
        """从 pipeline_checkpoint.json 构造 CheckpointSpec（P0.8，OP-4 唯一真源）。

        以 pipeline_checkpoint.json 为唯一真源，避免 CheckpointSpec 内存快照
        与文件快照并行存在的不一致问题。

        Args:
            run_id: PipelineRun ID
            stage_name: 当前 stage 名
            output_dir: PipelineContext.output_dir（pipeline_checkpoint.json 所在目录）

        Returns:
            CheckpointSpec 或 None（文件不存在时返回 None，不抛异常）

        验证（P0.9 test_checkpoint_unified）：
            get_checkpoints(run_id) 返回的 stage_snapshot 含 stage_outputs 字段，
            且与 pipeline_checkpoint.json 内容一致
        """
        # M4 修复：output_dir 来自外部（不可信），resolve 后记录审计日志
        # 正常流程下 output_dir 由 pipeline.py 创建（已受 M2 保护），此处防御直接调用
        output_dir_resolved = Path(output_dir).resolve()
        # 检测路径含 .. 的可疑输入（resolve 前后差异大）
        if ".." in str(output_dir):
            logger.warning(
                "load_checkpoint: output_dir contains '..' (suspicious): %s -> %s",
                output_dir, output_dir_resolved,
            )
        ckpt_path = output_dir_resolved / "pipeline_checkpoint.json"
        if not ckpt_path.exists():
            return None

        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # 文件损坏不阻断主流程，仅记录
            self._checkpoints.setdefault(run_id, [])
            return None

        ckpt = CheckpointSpec(
            run_id=run_id,
            stage_name=stage_name,
            checkpoint_uri=str(ckpt_path),
            stage_snapshot=data,  # 完整快照（含 stage_outputs）
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        with self._lock:
            self._checkpoints.setdefault(run_id, []).append(ckpt)
        return ckpt

    def get_checkpoints(self, run_id: str) -> List[CheckpointSpec]:
        """获取 Checkpoint 列表（OP-4）。"""
        return list(self._checkpoints.get(run_id, []))

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """查询 PipelineRun 状态（OP-6）。"""
        return self._runs.get(run_id)

    def list_runs(self, filter_phase: Optional[str] = None) -> List[PipelineRun]:
        """列出 PipelineRun（OP-6）。"""
        runs = list(self._runs.values())
        if filter_phase:
            runs = [r for r in runs if r.phase == filter_phase]
        return runs

    def recover(self) -> List[str]:
        """从持久化存储恢复所有 run（P3.4.2，重启后调用）。

        从 store 加载所有 PipelineRun 到 ``_runs``，并初始化对应的
        ``_checkpoints`` / ``_contexts`` 占位。PipelineContext 不可序列化，
        需 Agent 在恢复后重新绑定（``bind_context``）。

        Returns:
            恢复的 run_id 列表（按 store.list_runs() 顺序）

        Note:
            store 为 None 时返回空 list（向后兼容，纯内存模式无可恢复数据）。
            store 异常不抛出（捕获后视为无 run 可恢复）。
        """
        if self._store is None:
            return []
        try:
            runs = self._store.list_runs()
        except Exception:
            return []
        recovered: List[str] = []
        for run in runs:
            with self._lock:
                self._runs[run.run_id] = run
                self._checkpoints.setdefault(run.run_id, [])
                self._contexts.setdefault(run.run_id, None)
            recovered.append(run.run_id)
        return recovered

    def subscribe(self, event_type: str, callback: Callable[[CloudEvent], None]) -> Callable[[], None]:
        """订阅事件（OP-5/6）。

        返回取消订阅函数。
        """
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

        def unsubscribe():
            with self._lock:
                self._subscribers.get(event_type, []).remove(callback)

        return unsubscribe

    # ============================================================
    # P0.1: Reconciliation — 编排器驱动 Pipeline 执行
    # ============================================================
    def bind_context(self, run_id: str, ctx: Any) -> None:
        """绑定 PipelineContext 到 PipelineRun（P0.1）。

        reconcile() 执行 stage 时需要 PipelineContext 传递跨 stage 状态。
        必须在 start() 之前调用。

        Args:
            run_id: PipelineRun ID
            ctx: PipelineContext 实例
        """
        with self._lock:
            self._contexts[run_id] = ctx

    def reconcile(
        self,
        run_id: str,
        pipeline: Any = None,
    ) -> Dict[str, Any]:
        """Reconciliation Loop（OP-3 核心闭环）。

        委托 Pipeline.run() 执行 stage（OOM 回退、checkpoint、OBP 指标统一在此），
        reconcile 仅负责 CloudEvent 发射 + PipelineRun 状态回写。
        不复制 stage 循环逻辑，消除第三条执行路径。

        Args:
            run_id: PipelineRun ID
            pipeline: Pipeline 实例。若 None，使用 Pipeline.default()

        Returns:
            {"status": "succeeded"|"failed"|"paused", "completed_stages": [...],
             "failed_stage": str|None, "error": str|None}
        """
        run = self._get_run(run_id)
        if run.phase not in (PHASE_RUNNING,):
            return {"status": run.phase, "completed_stages": [], "failed_stage": None, "error": None}

        ctx = self._contexts.get(run_id)
        if ctx is None:
            return {"status": "failed", "completed_stages": [], "failed_stage": None,
                    "error": f"No PipelineContext bound to run {run_id}. Call bind_context() first."}

        if pipeline is None:
            try:
                from .engine.runner.pipeline import Pipeline as _Pipeline
                pipeline = _Pipeline.default()
            except ImportError:
                return {"status": "failed", "completed_stages": [], "failed_stage": None,
                        "error": "Pipeline not available"}

        # 从 PipelineRun.stage 状态恢复 completed_stages
        completed = [s.name for s in run.stages if s.phase == "succeeded"]
        if hasattr(ctx, "completed_stages"):
            ctx.completed_stages = list(completed)

        # 包装每个 stage：CloudEvent 发射 + PipelineRun 状态回写
        # Pipeline.run() 负责 OOM 回退、checkpoint、error handling
        original_stages = list(pipeline.stages)
        pipeline.stages = [
            (name, self._wrap_stage_for_reconcile(run_id, name, fn))
            for name, fn in original_stages
        ]

        # 委托执行
        # O1+O2: pipeline.run 抛异常时，run 会卡在 RUNNING 且 stages 保持包装状态。
        # 用 try/except/finally 包裹：except 标记 run failed，finally 还原 stages。
        try:
            result = pipeline.run(ctx)
        except Exception as exc:
            self.fail(run_id, error=f"reconcile crashed: {exc}")
            return {"status": "failed", "completed_stages": [], "failed_stage": None,
                    "error": str(exc)}
        finally:
            # 恢复原始 stages（pipeline 实例可能被复用）
            pipeline.stages = original_stages

        # 映射 StageResult → PipelineRun 状态
        if result.error is None:
            self.complete(run_id, output_uri=str(getattr(ctx, "output_dir", "")))
            return {"status": "succeeded", "completed_stages": list(ctx.completed_stages),
                    "failed_stage": None, "error": None}
        else:
            failed_stage = ctx.failed_stage or "unknown"
            self.update_stage(run_id, failed_stage, "failed", error=str(result.error))
            self.fail(run_id, error=str(result.error), stage_name=failed_stage)
            return {"status": "failed", "completed_stages": list(ctx.completed_stages),
                    "failed_stage": failed_stage, "error": str(result.error)}

    def _wrap_stage_for_reconcile(self, run_id: str, name: str, fn: Any) -> Any:
        """包装 stage 函数：执行前标记 running，执行后标记 succeeded + 保存 checkpoint。

        异常由 Pipeline.run() 统一捕获（不在此处理），
        reconcile 在 pipeline.run() 返回后从 ctx.failed_stage 读取失败 stage。

        P0.8：checkpoint 改用 save_checkpoint_from_file，以 pipeline_checkpoint.json
        为唯一真源，消除两套快照并行。
        """
        def wrapped(ctx):
            self.update_stage(run_id, name, "running")
            result = fn(ctx)
            self.update_stage(run_id, name, "succeeded")
            # P0.8：从 pipeline_checkpoint.json 读取完整快照作为唯一真源
            if hasattr(ctx, "output_dir") and ctx.output_dir:
                self.save_checkpoint_from_file(run_id, name, ctx.output_dir)
            return result
        return wrapped

    def _get_run(self, run_id: str) -> PipelineRun:
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(f"PipelineRun '{run_id}' not found")
        return run

    def emit_event(self, event_type: str, run_id: str, data: Dict[str, Any]) -> None:
        """发射 CloudEvent（OP-5 + P3.4.3 外部 sink）。"""
        event = make_event(event_type, run_id, data)
        # 通知进程内订阅者（原有逻辑）
        callbacks = self._subscribers.get(event_type, []) + self._subscribers.get("*", [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                # 修复（5.15）：禁止静默吞订阅者异常，至少 warning 留痕
                # 旧逻辑 except Exception: pass → CloudEvent 丢失无告警
                logger.warning(
                    "event subscriber failed (event_type=%s, run_id=%s): %s",
                    event_type, run_id, e,
                )
        # P3.4.3: 外部 sink（FileEventSink / Kafka / Webhook 等）
        # sink 异常不影响主流程（与订阅者同语义）
        if self._event_sink is not None:
            try:
                self._event_sink.emit(event)
            except Exception as e:
                # 修复（5.15）：禁止静默吞 sink 异常，至少 warning 留痕
                logger.warning(
                    "event sink emit failed (event_type=%s, run_id=%s): %s",
                    event_type, run_id, e,
                )

    # backward compat
    _emit_event = emit_event

    # ============================================================
    # P2.11: 异步执行（reconcile 真循环 + wait_for_completion）
    # ============================================================
    def start_and_execute(
        self,
        run_id: str,
        pipeline: Any = None,
    ) -> "Future":
        """异步启动并执行 PipelineRun（P2.11，OP-6 异步扩展）。

        与同步 reconcile() 的区别：
        - reconcile()：阻塞当前线程直到 Pipeline 执行完成
        - start_and_execute()：提交到 ThreadPoolExecutor，立即返回 Future
          调用方可通过 wait_for_completion(run_id) 阻塞等待结果

        异步任务 _execute_pipeline 内部调用 reconcile() 逻辑，
        复用 stage 包装 + CloudEvent 发射 + checkpoint 保存。

        Args:
            run_id: PipelineRun ID
            pipeline: Pipeline 实例（None 时用 Pipeline.default()）

        Returns:
            concurrent.futures.Future（异步任务句柄）

        Raises:
            KeyError: run_id 不存在
            RuntimeError: run 已在执行中
        """
        run = self._get_run(run_id)
        # 防止重复提交
        with self._async_lock:
            if run_id in self._run_futures and not self._run_futures[run_id].done():
                raise RuntimeError(f"Run '{run_id}' is already executing")
            # 确保线程池存在
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orch")
            future = self._executor.submit(self._execute_pipeline, run_id, pipeline)
            self._run_futures[run_id] = future
        return future

    def _execute_pipeline(self, run_id: str, pipeline: Any = None) -> Dict[str, Any]:
        """异步执行 Pipeline（P2.11，内部方法）。

        在 ThreadPoolExecutor 工作线程中调用 reconcile()，
        reconcile 内部完成 stage 循环 + CloudEvent 发射 + 状态回写。

        异常处理：
        - reconcile 已捕获 stage 异常并标记 run 为 failed
        - 此处仅作为兜底，捕获 reconcile 自身的异常（如 PipelineContext 未绑定）

        Args:
            run_id: PipelineRun ID
            pipeline: Pipeline 实例

        Returns:
            reconcile 返回的 dict（status / completed_stages / failed_stage / error）
        """
        try:
            # start() 触发 PHASE_RUNNING + emit EVENT_PIPELINE_STARTED
            # （若已是 RUNNING 则 transition 会被忽略，这里幂等）
            run = self._get_run(run_id)
            if run.phase == PHASE_PENDING:
                self.start(run_id)
            return self.reconcile(run_id, pipeline=pipeline)
        except Exception as e:
            # 兜底：reconcile 自身异常
            try:
                self.fail(run_id, error=f"_execute_pipeline crashed: {e}")
            except Exception:
                pass
            return {
                "status": "failed",
                "completed_stages": [],
                "failed_stage": None,
                "error": str(e),
            }

    def wait_for_completion(
        self,
        run_id: str,
        timeout: Optional[float] = None,
    ) -> PipelineRun:
        """阻塞等待 PipelineRun 收敛（P2.11，OP-6 异步扩展）。

        阻塞当前线程直到 PipelineRun 进入终态（succeeded / failed）或超时。
        若 run 通过 start_and_execute 启动，等待异步任务完成；
        若 run 通过同步 reconcile 启动，立即返回（已是终态）。

        Args:
            run_id: PipelineRun ID
            timeout: 最大等待秒数（None 表示无限等待）

        Returns:
            PipelineRun 实例（终态）

        Raises:
            KeyError: run_id 不存在
            TimeoutError: 等待超时
        """
        run = self._get_run(run_id)
        # 若有异步 future，先等 future 完成
        with self._async_lock:
            future = self._run_futures.get(run_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except Exception:
                # future 异常已在 _execute_pipeline 中处理为 failed
                pass

        # 即使 future 完成，也要确认 run.phase 已是终态（轮询保险）
        import time
        # 终态轮询间隔：50ms 平衡响应延迟与 CPU 开销
        _PHASE_POLL_INTERVAL_S = 0.05
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while run.phase not in (PHASE_SUCCEEDED, PHASE_FAILED):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run '{run_id}' did not complete within {timeout}s "
                    f"(current phase: {run.phase})"
                )
            time.sleep(_PHASE_POLL_INTERVAL_S)
        return run

    def shutdown(self) -> None:
        """关闭线程池（P2.11，资源清理）。

        在 Orchestrator 不再使用时调用，释放 ThreadPoolExecutor 资源。
        通常在测试 tearDown 或应用退出时调用。
        """
        with self._async_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None
            self._run_futures.clear()


# 全局单例
_orchestrator: Optional[Orchestrator] = None
_orchestrator_lock = threading.Lock()

def get_orchestrator() -> Orchestrator:
    """获取全局 Orchestrator 单例（线程安全，双重检查锁定）。"""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = Orchestrator()
    return _orchestrator


__all__ = [
    # OP-1
    "StageTemplate", "RetryPolicy", "CheckpointPolicy", "PipelineDef",
    # OP-2/3
    "StageStatus", "PipelineRun",
    "PHASE_PENDING", "PHASE_RUNNING", "PHASE_SUCCEEDED", "PHASE_FAILED", "PHASE_PAUSED",
    # OP-4
    "CheckpointSpec",
    # OP-5
    "CloudEvent", "make_event",
    "EVENT_PIPELINE_STARTED", "EVENT_PIPELINE_SUCCEEDED", "EVENT_PIPELINE_FAILED",
    "EVENT_PIPELINE_PAUSED", "EVENT_PIPELINE_RESUMED",
    "EVENT_STAGE_STARTED", "EVENT_STAGE_SUCCEEDED", "EVENT_STAGE_FAILED",
    "EVENT_TRIAL_COMPLETED", "EVENT_INFERENCE_SERVED",
    # OP-6
    "Orchestrator", "get_orchestrator",
    # P3.4.3: CloudEvent 外部 sink
    "EventSink", "FileEventSink",
    # P3.4.4: K8s Operator 适配层
    "K8sOperatorAdapter",
]
