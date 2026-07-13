"""RFC-003 OP：编排协议 — 让 SenseFrame 可被外部编排器托管。

对齐 K8s Operator Pattern（CRD + Reconciliation Loop）+ Argo Workflows（Pipeline/PipelineRun 分离）+ CloudEvents 1.0。

让 SenseFrame Pipeline 可被 K8s Operator / Argo / Airflow / 自研 AutoML 主控托管。
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


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
        """默认 Pipeline 定义（8 stage）。

        P0.7：stage 顺序与 Pipeline.default() 对齐，
        使 PipelineDef.default().materialize() 与 Pipeline.default() 等价。
        """
        return cls(
            name=name,
            stages=[
                StageTemplate("validate", reads=["config"], writes=["scene", "meta", "model_id", "dataset", "learning_mode"]),
                StageTemplate("preflight", reads=["config", "scene"], writes=["report", "route_level", "route_config", "output"]),
                StageTemplate("load", reads=["config", "scene", "dataset"], writes=["bundle", "data_profile", "output_dir"]),
                StageTemplate("resolve", reads=["config", "scene", "dataset"], writes=["task_spec", "feature_spec", "resolved", "lightning_params"]),
                StageTemplate("build", reads=["config", "scene", "model_id", "bundle"], writes=["model", "datamodule", "module", "callbacks"]),
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
# OP-6: 编排器接口
# ============================================================

class Orchestrator:
    """编排器（OP-6）。

    供 AutoML 主控调用，管理 Pipeline 生命周期。
    对齐 K8s Controller reconciliation loop。
    P0.1: reconcile() 桥接 Pipeline 执行器，实现「观察状态 → 驱动 stage → 回写状态」闭环。
    """

    def __init__(self):
        self._pipelines: Dict[str, PipelineDef] = {}  # name -> PipelineDef
        self._runs: Dict[str, PipelineRun] = {}  # run_id -> PipelineRun
        self._checkpoints: Dict[str, List[CheckpointSpec]] = {}  # run_id -> checkpoints
        self._subscribers: Dict[str, List[Callable[[CloudEvent], None]]] = {}  # event_type -> callbacks
        # P0.1: PipelineRun ↔ PipelineContext 映射（reconcile 驱动执行）
        self._contexts: Dict[str, Any] = {}  # run_id -> PipelineContext
        self._lock = threading.Lock()

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
        return run_id

    def start(self, run_id: str) -> None:
        """启动 PipelineRun（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_RUNNING)
        self._emit_event(EVENT_PIPELINE_STARTED, run_id, {"phase": run.phase})

    def pause(self, run_id: str) -> None:
        """暂停 PipelineRun（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_PAUSED)
        self._emit_event(EVENT_PIPELINE_PAUSED, run_id, {"phase": run.phase})

    def resume(self, run_id: str) -> None:
        """恢复 PipelineRun（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_RUNNING)
        self._emit_event(EVENT_PIPELINE_RESUMED, run_id, {"phase": run.phase})

    def retry(self, run_id: str) -> None:
        """重试 PipelineRun（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_RUNNING)
        run.retry_count += 1
        self._emit_event(EVENT_PIPELINE_STARTED, run_id, {"phase": run.phase, "retry": run.retry_count})

    def stop(self, run_id: str) -> None:
        """停止 PipelineRun（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_FAILED)
        run.error = "Stopped by orchestrator"
        self._emit_event(EVENT_PIPELINE_FAILED, run_id, {"phase": run.phase, "error": run.error})

    def complete(self, run_id: str, output_uri: str = "") -> None:
        """标记 PipelineRun 成功完成（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_SUCCEEDED)
        run.output_uri = output_uri
        self._emit_event(EVENT_PIPELINE_SUCCEEDED, run_id, {"phase": run.phase, "output_uri": output_uri})

    def fail(self, run_id: str, error: str, stage_name: str = "") -> None:
        """标记 PipelineRun 失败（OP-6）。"""
        run = self._get_run(run_id)
        run.transition(PHASE_FAILED)
        run.error = error
        if stage_name:
            for s in run.stages:
                if s.name == stage_name:
                    s.phase = "failed"
                    s.error = error
                    break
        self._emit_event(EVENT_PIPELINE_FAILED, run_id, {"phase": run.phase, "error": error, "stage": stage_name})

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
        ckpt_path = Path(output_dir) / "pipeline_checkpoint.json"
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
        result = pipeline.run(ctx)

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

    def _emit_event(self, event_type: str, run_id: str, data: Dict[str, Any]) -> None:
        """发射 CloudEvent（OP-5）。"""
        event = make_event(event_type, run_id, data)
        # 通知订阅者
        callbacks = self._subscribers.get(event_type, []) + self._subscribers.get("*", [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # 订阅者异常不影响主流程


# 全局单例
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator() -> Orchestrator:
    """获取全局 Orchestrator 单例。"""
    global _orchestrator
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
]
