# Orchestration 与 CloudEvent 参考

> 锚点：SenseFrame 编排协议，对齐 K8s Operator 模式 + CloudEvents 1.0 规范

## 概述

SenseFrame Orchestration 子系统（RFC-003 OP）将训练 Pipeline 从「函数调用」升级为「可被外部编排器托管」的声明式工作流。它对齐三类业界模式：

- **K8s Operator Pattern**：CRD（PipelineDef/PipelineRun）+ Reconciliation Loop（`reconcile`）
- **Argo Workflows**：Pipeline 定义与运行实例分离（PipelineDef ↔ PipelineRun）
- **CloudEvents 1.0**：标准化事件流，便于对接 Kafka / Webhook / 自动驾驶主控

子系统由六个 OP（Orchestration Protocol）模块组成：

| 模块 | 职责 |
|------|------|
| OP-1 | 声明式 Pipeline 定义（PipelineDef / StageTemplate / RetryPolicy / CheckpointPolicy） |
| OP-2 | 运行实例（PipelineRun / StageStatus） |
| OP-3 | 生命周期状态机（5 phase + 合法转换矩阵） |
| OP-4 | Checkpoint（冷启动 / 热续跑，以 `pipeline_checkpoint.json` 为唯一真源） |
| OP-5 | CloudEvent 事件流（10 个事件类型常量 + EventSink 协议） |
| OP-6 | Orchestrator 编排器（生命周期管理 + Reconciliation + 异步执行 + K8s CR 适配） |

核心源码：`senseframe/orchestration.py`（约 1090 行）、`senseframe/orchestration_store.py`（持久化后端）。

---

## 声明式定义（OP-1）

### PipelineDef

`PipelineDef` 是声明式 Pipeline 定义，可序列化为 YAML/JSON，同一定义可被多次实例化为 `PipelineRun`。

**关键字段**：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `str` | — | Pipeline 定义名（`create_run` 的查找键） |
| `stages` | `List[StageTemplate]` | `[]` | Stage 模板列表，按执行顺序 |
| `config` | `Dict[str, Any]` | `{}` | 全局配置 |
| `scene` | `Dict[str, Any]` | `{}` | 场景配置 |
| `retry_policy` | `RetryPolicy` | `RetryPolicy()` | 重试策略 |
| `checkpoint_policy` | `CheckpointPolicy` | `CheckpointPolicy()` | Checkpoint 策略 |

**方法**：

- `to_dict() -> Dict[str, Any]`：序列化为 dict（含嵌套 retry_policy / checkpoint_policy）。
- `PipelineDef.default(name="default") -> PipelineDef`：返回 **9 stage 默认定义**（与 `Pipeline.default()` 顺序对齐），stages 为：
  1. `validate` — reads: `[config]`, writes: `[scene, meta, model_id, dataset, learning_mode]`
  2. `preflight` — reads: `[config, scene]`, writes: `[report, route_level, route_config, output]`
  3. `load` — reads: `[config, scene, dataset]`, writes: `[bundle, data_profile, output_dir]`
  4. `resolve` — reads: `[config, scene, dataset]`, writes: `[task_spec, feature_spec, resolved, lightning_params]`
  5. `build` — reads: `[config, scene, model_id, bundle]`, writes: `[model, datamodule, module, callbacks]`
  6. `probe_vram` — reads: `[model, datamodule, resolved, report]`, writes: `[vram_probe_result]`
  7. `train` — reads: `[config, model, datamodule, module]`, writes: `[trainer, output]`
  8. `eval` — reads: `[config, trainer, module, datamodule]`, writes: `[output]`
  9. `export` — reads: `[config, model, module, output]`, writes: `[output]`
- `materialize() -> Pipeline`：将声明式定义物化为可执行 `Pipeline`。从 `Pipeline.default().stages` 查找内置 stage 函数；未知 stage 名抛 `ValueError`。自定义 stage 需通过 `Pipeline.replace_stage` 注入。

### StageTemplate / RetryPolicy / CheckpointPolicy

**StageTemplate**（Stage 模板）：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `str` | — | Stage 名（如 `"train"`） |
| `reads` | `List[str]` | `[]` | 该 stage 读取的上下文字段 |
| `writes` | `List[str]` | `[]` | 该 stage 产出的上下文字段 |
| `config` | `Dict[str, Any]` | `{}` | Stage 级配置 |

**RetryPolicy**（重试策略）：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_retries` | `int` | `3` | 最大重试次数 |
| `backoff` | `float` | `1.0` | 退避秒数 |
| `retry_on_stages` | `List[str]` | `[]` | 限定可重试的 stage（空 = 所有 stage） |

**CheckpointPolicy**（Checkpoint 策略）：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enabled` | `bool` | `True` | 是否启用 checkpoint |
| `interval` | `int` | `1` | 每 N 个 stage 保存一次 |
| `storage_uri` | `str` | `""` | 存储位置（空 = 本地） |

---

## 运行实例与状态机（OP-2/3）

### StageStatus

`StageStatus` 描述单个 stage 的运行状态。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `str` | — | Stage 名 |
| `phase` | `str` | `"pending"` | 取值：`pending` / `running` / `succeeded` / `failed` / `skipped` |
| `started_at` | `str` | `""` | 启动时间（ISO 格式） |
| `finished_at` | `str` | `""` | 完成时间（ISO 格式） |
| `checkpoint_uri` | `str` | `""` | 该 stage 的 checkpoint URI |
| `error` | `str` | `""` | 失败原因 |

`to_dict()` / `StageStatus.from_dict(d)` 完成双向序列化；`from_dict` 缺失字段使用默认值，保证旧版数据向后兼容。

### PipelineRun

`PipelineRun` 是 Pipeline 定义的一次运行实例。

**关键字段**：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `run_id` | `str` | — | 运行实例 ID（`run_{uuid.hex[:8]}`） |
| `pipeline_ref` | `str` | — | 引用的 `PipelineDef.name` |
| `owner_reference` | `Optional[str]` | `None` | 归属 PipelineDef 强引用（K8s CRD owner_reference 语义） |
| `params` | `Dict[str, Any]` | `{}` | 运行参数 |
| `checkpoint_uri` | `str` | `""` | Checkpoint 基 URI |
| `phase` | `str` | `PHASE_PENDING` | 当前状态机 phase |
| `stages` | `List[StageStatus]` | `[]` | 各 stage 状态 |
| `started_at` | `str` | `""` | Run 启动时间 |
| `finished_at` | `str` | `""` | Run 完成时间 |
| `output_uri` | `str` | `""` | 产物 URI |
| `error` | `str` | `""` | 失败原因 |
| `retry_count` | `int` | `0` | 已重试次数 |

**方法**：

- `to_dict() -> Dict[str, Any]`：序列化。`status` 字段（phase/stages/started_at/finished_at/output_uri/error/retry_count）嵌套在 `"status"` key 下。
- `PipelineRun.from_dict(d) -> PipelineRun`：反序列化。递归调用 `StageStatus.from_dict`，从 `"status"` key 解包；缺失字段使用默认值，保证旧版数据向后兼容。
- `transition(new_phase) -> None`：状态转换。非法转换抛 `ValueError`。副作用：
  - 转入 `RUNNING` 且 `started_at` 为空时，写入 `started_at`
  - 转入 `SUCCEEDED` / `FAILED` 时，写入 `finished_at`

### 状态机

**5 个 phase 常量**（orchestration.py 第 178–182 行）：

| 常量 | 值 |
|------|----|
| `PHASE_PENDING` | `"pending"` |
| `PHASE_RUNNING` | `"running"` |
| `PHASE_SUCCEEDED` | `"succeeded"` |
| `PHASE_FAILED` | `"failed"` |
| `PHASE_PAUSED` | `"paused"` |

**合法状态转换矩阵**（`_VALID_TRANSITIONS`）：

| 当前 phase \ 可转入 | RUNNING | SUCCEEDED | FAILED | PAUSED |
|--------------------|---------|-----------|--------|--------|
| PENDING            | ✅      | ❌        | ❌     | ❌     |
| RUNNING            | ❌      | ✅        | ✅     | ✅     |
| PAUSED             | ✅      | ❌        | ❌     | ❌     |
| FAILED             | ✅(retry)| ❌       | ❌     | ❌     |
| SUCCEEDED          | ❌      | ❌（终态）| ❌     | ❌     |

要点：`SUCCEEDED` 为终态无出边；`FAILED → RUNNING` 仅用于 retry；`PAUSED ↔ RUNNING` 双向。

**状态机图示**：

```
                  start/retry
       ┌────────────────────────┐
       │                        │
       ▼                        │
   ┌────────┐    start    ┌─────────┐
   │PENDING │────────────▶│ RUNNING │
   └────────┘             └─────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       complete/succeed    fail/stop        pause
              │                │                │
              ▼                ▼                ▼
        ┌──────────┐    ┌────────┐       ┌────────┐
        │SUCCEEDED │    │ FAILED │       │ PAUSED │
        │ (终态)   │    └────────┘       └────────┘
        └──────────┘         ▲                │
                             │   retry        │
                             └────────────────┘
                                  resume
                             (PAUSED → RUNNING)
```

**典型生命周期路径**：

- 正常完成：`PENDING → RUNNING → SUCCEEDED`
- 失败重试：`PENDING → RUNNING → FAILED → RUNNING → SUCCEEDED`
- 暂停恢复：`PENDING → RUNNING → PAUSED → RUNNING → SUCCEEDED`
- 主动停止：`PENDING → RUNNING → FAILED`（`error="Stopped by orchestrator"`）

---

## Checkpoint（OP-4）

### CheckpointSpec

`CheckpointSpec` 描述单个 stage 的 checkpoint 快照。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `run_id` | `str` | — | 所属 PipelineRun |
| `stage_name` | `str` | — | Checkpoint 对应的 stage 名 |
| `checkpoint_uri` | `str` | — | Checkpoint 文件 URI |
| `stage_snapshot` | `Dict[str, Any]` | `{}` | Stage 产出快照 |
| `timestamp` | `str` | `""` | Checkpoint 时间戳 |

`to_dict()` 提供序列化方法。Checkpoint 内存列表保存在 `Orchestrator._checkpoints[run_id]`。

### Checkpoint 操作

Orchestrator 提供三类 Checkpoint 方法：

- `save_checkpoint(run_id, stage_name, checkpoint_uri, stage_snapshot=None)`：内存 Checkpoint 保存（向后兼容接口）。构造 `CheckpointSpec` 并追加到 `_checkpoints[run_id]`。
- `save_checkpoint_from_file(run_id, stage_name, output_dir) -> Optional[CheckpointSpec]`：**以 `pipeline_checkpoint.json` 为唯一真源**。从 `output_dir/pipeline_checkpoint.json` 读取完整快照（含 `stage_outputs`），构造 `CheckpointSpec`：
  - `output_dir` 经 `Path(...).resolve()` 规范化；含 `..` 时记 warning 审计日志
  - 文件不存在或 JSON 损坏时返回 `None`，不抛异常（不阻断主流程）
  - `checkpoint_uri` 设为 `pipeline_checkpoint.json` 的绝对路径
  - `stage_snapshot` 设为完整 JSON 内容，`timestamp` 取自 JSON 的 `timestamp` 字段
- `get_checkpoints(run_id) -> List[CheckpointSpec]`：返回该 run 的所有 Checkpoint 列表（拷贝）。

**唯一真源原则**（P0.8）：`pipeline_checkpoint.json` 由 `Pipeline._write_checkpoint` 在每个 stage 完成后写入，包含 `pipeline_version` / `config_hash` / `completed_stages` / `trial_id` / `timestamp` / `stage_outputs` / `resources_released` / `failed_stage`。Orchestrator 的 `reconcile` 流程通过 `save_checkpoint_from_file` 读取该文件作为 OP-4 真源，避免内存快照与文件快照并行存在的不一致问题。

---

## CloudEvent 事件流（OP-5）

### CloudEvent

`CloudEvent` 对齐 CloudEvents 1.0 规范。

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `specversion` | `str` | `"1.0"` | CloudEvents 规范版本 |
| `id` | `str` | `""` | 事件 ID（`__post_init__` 中若空则 `uuid.uuid4().hex`） |
| `source` | `str` | `""` | 事件源（`make_event` 中设为 `/senseframe/pipeline/{run_id}`） |
| `type` | `str` | `""` | 事件类型（见下表） |
| `time` | `str` | `""` | 事件时间（`__post_init__` 中若空则 `datetime.now().isoformat()`） |
| `datacontenttype` | `str` | `"application/json"` | 数据内容类型 |
| `subject` | `str` | `""` | 事件主题 |
| `data` | `Dict[str, Any]` | `{}` | 事件负载 |

**方法**：

- `to_dict() -> Dict[str, Any]`：返回完整 CloudEvent dict。
- `to_json() -> str`：返回 JSON 字符串（`ensure_ascii=False`）。

**工厂函数** `make_event(event_type, run_id, data) -> CloudEvent`：构造 `source=f"/senseframe/pipeline/{run_id}"`、`type=event_type`、`data=data` 的 CloudEvent。

### 事件类型

10 个事件类型常量（orchestration.py 第 321–330 行）：

| 常量 | 值 | 触发位置 |
|------|----|---------|
| `EVENT_PIPELINE_STARTED` | `"senseframe.pipeline.started"` | `Orchestrator.start` / `retry` |
| `EVENT_PIPELINE_SUCCEEDED` | `"senseframe.pipeline.succeeded"` | `Orchestrator.complete` |
| `EVENT_PIPELINE_FAILED` | `"senseframe.pipeline.failed"` | `Orchestrator.stop` / `fail` |
| `EVENT_PIPELINE_PAUSED` | `"senseframe.pipeline.paused"` | `Orchestrator.pause` |
| `EVENT_PIPELINE_RESUMED` | `"senseframe.pipeline.resumed"` | `Orchestrator.resume` |
| `EVENT_STAGE_STARTED` | `"senseframe.stage.started"` | `Orchestrator.update_stage`（phase=running） |
| `EVENT_STAGE_SUCCEEDED` | `"senseframe.stage.succeeded"` | `Orchestrator.update_stage`（phase=succeeded） |
| `EVENT_STAGE_FAILED` | `"senseframe.stage.failed"` | `Orchestrator.update_stage`（phase=failed） |
| `EVENT_TRIAL_COMPLETED` | `"senseframe.trial.completed"` | HPO trial 完成 |
| `EVENT_INFERENCE_SERVED` | `"senseframe.inference.served"` | 推理服务上线 |

事件发射由 `Orchestrator._emit_event(event_type, run_id, data)` 统一处理：
1. 构造 CloudEvent（`make_event`）
2. 通知进程内订阅者（按 `event_type` + 通配 `"*"` 订阅列表）；订阅者异常被 `try/except` 捕获并记 warning（不静默吞）
3. 若 `event_sink` 非 None，调用 `event_sink.emit(event)`；sink 异常同样被捕获并记 warning

### EventSink

`EventSink` 是 CloudEvent 外部 sink 协议（`@runtime_checkable Protocol`）：

```python
class EventSink(Protocol):
    def emit(self, event: CloudEvent) -> None: ...
```

任何含 `emit(event: CloudEvent) -> None` 方法的对象均满足此协议，可作为 `Orchestrator(store=..., event_sink=...)` 的 `event_sink` 参数（FileEventSink / Kafka / Webhook 等）。

**FileEventSink**：默认实现，将 CloudEvent 以 JSONL 格式追加写入日志文件。

| 方法 | 行为 |
|------|------|
| `__init__(log_path)` | 自动创建父目录；`log_path` 接受 `str` / `Path` |
| `emit(event)` | 以 `"a"` 模式打开 `log_path`，每行写入 `event.to_json() + "\n"` |

sink 异常不影响主流程（Orchestrator 在 `_emit_event` 中 `try/except` 兜底）。

---

## K8s Operator 适配（OP-6）

### K8sOperatorAdapter

`K8sOperatorAdapter` 将 OP `PipelineRun` 映射为 K8s Custom Resource（CRD: `senseframe.io/v1` `PipelineRun`）。

**类常量**：

| 常量 | 值 |
|------|----|
| `API_VERSION` | `"senseframe.io/v1"` |
| `KIND` | `"PipelineRun"` |
| `OWNER_API_VERSION` | `"senseframe.io/v1"` |
| `OWNER_KIND` | `"PipelineDef"` |

**CR manifest 结构**（对齐 K8s CRD 规范）：

```yaml
apiVersion: senseframe.io/v1
kind: PipelineRun
metadata:
  name: <run_id>
  ownerReferences:         # 仅当 run.owner_reference 非 None
    - apiVersion: senseframe.io/v1
      kind: PipelineDef
      name: <owner_reference>
spec:
  pipelineRef: <pipeline_ref>
  params: {...}
  checkpointUri: <checkpoint_uri>
status:
  phase: <phase>
  startedAt: <started_at>
  finishedAt: <finished_at>
  outputUri: <output_uri>
  error: <error>
  retryCount: <retry_count>
  stages: [...]
```

**方法**：

- `to_cr_manifest(run: PipelineRun) -> Dict[str, Any]`：PipelineRun → K8s CR。`ownerReferences` 仅在 `run.owner_reference` 非 None 时包含。
- `from_cr_manifest(manifest: Dict[str, Any]) -> PipelineRun`：K8s CR → PipelineRun。支持最小 manifest（仅 `metadata.name` + `spec.pipelineRef`），缺失字段使用默认值；`ownerReferences[0].name` 映射回 `owner_reference`。

> P3 仅提供双向序列化；真实 Operator 驱动（reconciliation loop、CR watch、status subresource 回写）推迟到 P4（需 kopf/argo 依赖 + K8s 集群验证）。

---

## Orchestrator 编排器

`Orchestrator` 是 OP-6 的核心，供 AutoML 主控调用，管理 Pipeline 生命周期，对齐 K8s Controller reconciliation loop。

### 构造与内部状态

```python
Orchestrator(store: Optional[OrchestrationStore] = None,
             event_sink: Optional[EventSink] = None)
```

| 内部字段 | 类型 | 说明 |
|---------|------|------|
| `_pipelines` | `Dict[str, PipelineDef]` | name → PipelineDef 注册表 |
| `_runs` | `Dict[str, PipelineRun]` | run_id → PipelineRun |
| `_checkpoints` | `Dict[str, List[CheckpointSpec]]` | run_id → Checkpoint 列表 |
| `_subscribers` | `Dict[str, List[Callable]]` | event_type → 回调列表 |
| `_contexts` | `Dict[str, Any]` | run_id → PipelineContext |
| `_lock` | `threading.Lock` | 保护上述 dict |
| `_executor` | `Optional[ThreadPoolExecutor]` | 异步执行线程池（max_workers=4） |
| `_run_futures` | `Dict[str, Future]` | run_id → Future |
| `_async_lock` | `threading.Lock` | 保护 `_run_futures` |
| `_store` | `Optional[OrchestrationStore]` | 持久化后端（None 时纯内存） |
| `_event_sink` | `Optional[EventSink]` | CloudEvent 外部 sink |

`store` 与 `event_sink` 均为可选，None 时退化为纯内存 / 不写外部日志（向后兼容）。

### 生命周期管理

| 方法 | 行为 |
|------|------|
| `create_pipeline(pipeline_def) -> str` | 注册 PipelineDef，返回 name |
| `create_run(pipeline_id, params=None, checkpoint_uri="") -> str` | 创建 PipelineRun。若 pipeline_id 未注册抛 `KeyError`；`run_id` 形如 `run_{uuid.hex[:8]}`；`owner_reference` 设为 `pipeline_id`；`stages` 由 `pdef.stages` 构造为 `StageStatus(name=s.name)` |
| `start(run_id)` | `transition(RUNNING)` + emit `EVENT_PIPELINE_STARTED` + 持久化 |
| `pause(run_id)` | `transition(PAUSED)` + emit `EVENT_PIPELINE_PAUSED` + 持久化 |
| `resume(run_id)` | `transition(RUNNING)` + emit `EVENT_PIPELINE_RESUMED` + 持久化 |
| `retry(run_id)` | `transition(RUNNING)` + `retry_count += 1` + emit `EVENT_PIPELINE_STARTED`（含 `retry` 字段） + 持久化 |
| `stop(run_id)` | `transition(FAILED)` + `error="Stopped by orchestrator"` + emit `EVENT_PIPELINE_FAILED` + 持久化 |
| `complete(run_id, output_uri="")` | `transition(SUCCEEDED)` + 写 `output_uri` + emit `EVENT_PIPELINE_SUCCEEDED` + 持久化 |
| `fail(run_id, error, stage_name="")` | `transition(FAILED)` + 写 `error`；若 `stage_name` 非空，将该 stage 标 `failed` 并写入 `error` + emit `EVENT_PIPELINE_FAILED` + 持久化 |
| `update_stage(run_id, stage_name, phase, checkpoint_uri="", error="")` | 更新 stage 状态：`running` 时写 `started_at`，`succeeded/failed` 时写 `finished_at`；阶段变化时 emit `EVENT_STAGE_*`（按 phase 映射） |

所有状态变更方法调用 `transition`，非法转换会抛 `ValueError`；每次变更后通过 `_persist_run(run)` 写 store（store 为 None 时 no-op；store 异常被吞，不阻断主流程）。

### Checkpoint 管理

见上文 [Checkpoint 操作](#checkpoint-操作)。

### 查询与恢复

| 方法 | 行为 |
|------|------|
| `get_run(run_id) -> Optional[PipelineRun]` | 查询单个 run，不存在返回 None |
| `list_runs(filter_phase=None) -> List[PipelineRun]` | 列出所有 run；`filter_phase` 非 None 时按 phase 过滤 |
| `recover() -> List[str]` | 从 store 加载所有 run 到 `_runs`，并初始化 `_checkpoints` / `_contexts` 占位。store 为 None 时返回 `[]`；store 异常被吞，视为无 run 可恢复。**PipelineContext 不可序列化**，Agent 恢复后需重新调 `bind_context`。返回恢复的 run_id 列表 |

### 事件订阅

```python
unsubscribe = orchestrator.subscribe(event_type, callback)
# ... 后续 unsubscribe() 取消订阅
```

`subscribe(event_type, callback) -> Callable[[], None]`：将 `callback` 加入 `_subscribers[event_type]`，返回取消订阅函数。订阅者收到事件时异常被 `_emit_event` 捕获并记 warning（不阻断其他订阅者）。

支持通配 `"*"` event_type：`_emit_event` 会同时通知 `event_type` 订阅者与 `"*"` 订阅者。

### Reconciliation

Reconciliation Loop 是 OP-3 核心闭环，对齐 K8s Controller 模式。

| 方法 | 行为 |
|------|------|
| `bind_context(run_id, ctx)` | 绑定 PipelineContext 到 PipelineRun。**必须在 `start()` 之前调用**，`reconcile` 执行 stage 时需要 PipelineContext 传递跨 stage 状态 |
| `reconcile(run_id, pipeline=None) -> Dict[str, Any]` | 核心闭环。委托 `Pipeline.run()` 执行 stage，不复制 stage 循环逻辑 |

`reconcile` 执行流程：

1. 取 `run = self._get_run(run_id)`；若 `phase != RUNNING`，立即返回当前状态
2. 取 `ctx = self._contexts.get(run_id)`；若 None，返回 `{"status": "failed", ..., "error": "No PipelineContext bound..."}`
3. 若 `pipeline` 为 None，尝试 `Pipeline.default()`；导入失败返回 failed
4. 从 `run.stages` 中 `phase == "succeeded"` 的 stage 名恢复 `ctx.completed_stages`
5. 用 `_wrap_stage_for_reconcile` 包装每个 stage 函数（执行前标 `running`，执行后标 `succeeded` + `save_checkpoint_from_file`），临时替换 `pipeline.stages`
6. 调 `pipeline.run(ctx)` 委托执行（OOM 回退、checkpoint、OBP 指标统一在 `Pipeline.run` 内）
7. 恢复 `pipeline.stages` 为原始 stages（pipeline 实例可复用）
8. 映射结果：`result.error is None` → `self.complete(run_id, output_uri=str(ctx.output_dir))`，返回 succeeded；否则 `self.update_stage(run_id, failed_stage, "failed", ...)` + `self.fail(...)`，返回 failed

返回 dict 结构：`{"status": "succeeded"|"failed"|"paused", "completed_stages": [...], "failed_stage": str|None, "error": str|None}`。

`_wrap_stage_for_reconcile(run_id, name, fn)`：返回闭包 `wrapped(ctx)`，依次调用 `update_stage(running)` → `fn(ctx)` → `update_stage(succeeded)` → 若 `ctx.output_dir` 非空则 `save_checkpoint_from_file`。异常由 `Pipeline.run()` 统一捕获。

### Pipeline.run() 执行流程（reconcile 委托目标）

`reconcile` 委托的 `Pipeline.run(ctx, *, dry_run=False) -> StageResult`（`senseframe/engine/runner/pipeline/runtime.py`）承担实际 stage 循环：

1. **OTel 初始化**：调用 `init_otel(pipeline_run_id, trial_id, model_id, dataset)`，否则所有 `record_training_metric` 埋点 no-op。
2. **Checkpoint 加载**：若 `ctx.stage_checkpoint_path` 存在，加载 JSON 取 `completed_stages`：
   - **config_hash 校验**：若 `config_hash` 变更，全部重跑（清空 `completed_stages`）
   - **不可序列化 stage 强制重跑**：`_NON_SERIALIZABLE_STAGES = frozenset({"load", "build", "probe_vram", "train", "eval"})` 中的 stage 即使在 `completed_stages` 中也强制重跑（`bundle/model/trainer` 等对象引用跨进程不可恢复）
   - **stage_outputs 恢复**：`_restore_stage_outputs(ctx)` 从 `pipeline_checkpoint.json` 的 `stage_outputs` 字段恢复 `report` / `route_config` / `task_spec` / `feature_spec` 等可序列化产出
3. **Stage 循环**（`try/finally` 包裹）：
   - 跳过 `completed_stages` 中的 stage（含补偿逻辑：`validate` 重建 `scene/meta`，`preflight` 重做 `set_seed`）
   - 设置 `StageAwareCallback` 的 active 状态
   - 用 `Timer` 包裹 `fn(ctx)`，记录 `senseframe.stage.{name}.duration_s` 到 OTel
   - 成功：`completed_stages.append(name)` + `_write_checkpoint(ctx)`
   - 异常：`_classify_runtime_error` 重新分类（OOMError/ModelBuildError/TrainingError/DataCorruptedError/CheckpointError/SaveError），写 `FAILED` 文件 + 删 `*.pth` + 重命名 `output_dir` 为 `FAILED_{原名}` + 返回 `StageResult(error=e)`
4. **finally 分支**：生成 `manifest.json`（产物溯源清单） + `ctx.release_resources()` + 再次写 checkpoint（持久化 `resources_released=True`） + dry_run 时清理临时目录

**`pipeline_checkpoint.json` 结构**（`_write_checkpoint` 写入）：

```json
{
  "pipeline_version": "<_PIPELINE_VERSION>",
  "config_hash": "<hash>",
  "completed_stages": ["validate", "preflight", ...],
  "trial_id": "<id>",
  "timestamp": "<ISO>",
  "stage_outputs": {  // OP-4 唯一真源
    "model_id": "...", "dataset": "...",
    "report": {...}, "route_config": {...},
    "task_spec": {...}, "feature_spec": {...},
    "final_eval": {...}, "training_log": [...],
    "feedback": {...}, ...
  },
  "resources_released": false,
  "failed_stage": "train"  // 仅失败时存在
}
```

### 异步执行

| 方法 | 行为 |
|------|------|
| `start_and_execute(run_id, pipeline=None) -> Future` | 提交到 `ThreadPoolExecutor`（max_workers=4），立即返回 Future。run 已在执行中抛 `RuntimeError`。内部调 `_execute_pipeline` |
| `_execute_pipeline(run_id, pipeline=None) -> Dict` | 在工作线程中：若 `phase == PENDING` 先 `start()`，再调 `reconcile`；reconcile 自身异常被捕获并调 `fail(run_id, ...)` 兜底 |
| `wait_for_completion(run_id, timeout=None) -> PipelineRun` | 阻塞等待 run 收敛到 `SUCCEEDED` / `FAILED`。先等 `_run_futures[run_id]` Future 完成；之后轮询 `run.phase`（50ms 间隔）直到终态或超时。超时抛 `TimeoutError` |
| `shutdown()` | 关闭 `ThreadPoolExecutor`（`wait=False`），清空 `_run_futures`。测试 tearDown / 应用退出时调用 |

### 全局单例

```python
def get_orchestrator() -> Orchestrator:
    """获取全局 Orchestrator 单例。"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
```

模块级 `_orchestrator: Optional[Orchestrator] = None`，首次调用 `get_orchestrator()` 时构造纯内存实例（无 store / 无 event_sink）。

---

## 持久化后端

### OrchestrationStore Protocol

`OrchestrationStore`（`orchestration_store.py`）是 OP 持久化后端协议（`@runtime_checkable Protocol`）：

| 方法 | 说明 |
|------|------|
| `save_run(run)` | 保存 / 覆盖单个 PipelineRun |
| `load_run(run_id) -> Optional[PipelineRun]` | 加载单个 run（不存在返回 None） |
| `list_runs(pipeline_ref=None) -> List[PipelineRun]` | 列出所有 run（可按 pipeline_ref 过滤） |
| `delete_run(run_id)` | 删除单个 run（不存在不报错，幂等） |

### FileOrchestrationStore

文件系统默认实现。

| 方法 | 行为 |
|------|------|
| `__init__(base_dir)` | 存储目录（str / Path） |
| `save_run(run)` | 自动创建 `base_dir`（含父目录）；写 `{base_dir}/{run_id}.json`（`indent=2`，`ensure_ascii=False`）；重复调用覆盖旧文件 |
| `load_run(run_id)` | 文件不存在返回 None；JSON 损坏 / OSError 返回 None（不抛异常） |
| `list_runs(pipeline_ref=None)` | 目录不存在返回 `[]`；按 `sorted(glob("*.json"))` 顺序加载；JSON 损坏跳过；`pipeline_ref` 非 None 时按 `run.pipeline_ref` 过滤 |
| `delete_run(run_id)` | 文件不存在不报错（幂等） |

> P4 扩展点：`K8sOrchestrationStore` 以 K8s CRD 为后端（save_run → PUT CR，load_run → GET CR，list_runs → LIST CR，delete_run → DELETE CR），需 kopf/kubernetes-client 依赖。

---

## 使用示例

### 示例 1：声明式 Pipeline 定义 + 运行

```python
from senseframe.orchestration import (
    PipelineDef, Orchestrator, FileOrchestrationStore, FileEventSink,
)
from senseframe.engine.runner.pipeline import PipelineContext

# 1. 声明式定义（用 default 9 stage）
pdef = PipelineDef.default(name="wifi_csi_train")
pdef.retry_policy.max_retries = 2
pdef.checkpoint_policy.enabled = True

# 2. 构造编排器（含持久化 + 事件 sink）
orch = Orchestrator(
    store=FileOrchestrationStore("./runs"),
    event_sink=FileEventSink("./events/cloud_events.jsonl"),
)
pid = orch.create_pipeline(pdef)

# 3. 创建 run + 绑定 ctx + 异步执行
run_id = orch.create_run(pid, params={"scene": "wifi_csi"})
ctx = PipelineContext(config=...)  # 由调用方构造
orch.bind_context(run_id, ctx)

future = orch.start_and_execute(run_id)
run = orch.wait_for_completion(run_id, timeout=3600)
print(f"phase={run.phase}, output_uri={run.output_uri}")
```

### 示例 2：断点续跑（从 store 恢复）

```python
# 进程重启后
orch = Orchestrator(store=FileOrchestrationStore("./runs"))
recovered = orch.recover()  # 返回 ["run_abc12345", ...]

for run_id in recovered:
    run = orch.get_run(run_id)
    if run.phase == "paused":
        # 重新绑定 PipelineContext（不可序列化，必须重建）
        ctx = PipelineContext(config=...)
        orch.bind_context(run_id, ctx)
        orch.resume(run_id)
        orch.start_and_execute(run_id)
    elif run.phase == "failed":
        # 检查已完成的 stage，从失败 stage 续跑
        ckpts = orch.get_checkpoints(run_id)
        print(f"已有 {len(ckpts)} 个 checkpoint")
        orch.retry(run_id)
```

### 示例 3：事件订阅

```python
def on_stage_succeeded(event):
    print(f"stage succeeded: {event.data}")

def on_any_event(event):  # 通配订阅
    print(f"[{event.type}] {event.source}")

orch.subscribe("senseframe.stage.succeeded", on_stage_succeeded)
orch.subscribe("*", on_any_event)
```

---

## 参考文件

| 文件 | 内容 |
|------|------|
| `senseframe/orchestration.py` | OP-1 至 OP-6 主模块（PipelineDef / PipelineRun / CloudEvent / Orchestrator / K8sOperatorAdapter） |
| `senseframe/orchestration_store.py` | OrchestrationStore Protocol + FileOrchestrationStore |
| `senseframe/engine/runner/pipeline/runtime.py` | `Pipeline` 类与 `Pipeline.run()` 方法（reconcile 委托的执行器） |
| `senseframe/engine/runner/orchestrator.py` | `EpochLogCallback` 等训练回调（非 OP 编排器） |
