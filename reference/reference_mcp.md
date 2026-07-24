# MCP 子系统参考

> 锚点：SenseFrame MCP 服务器实现，对齐 MCP (Model Context Protocol) 规范

## 概述

SenseFrame MCP 子系统是 Agent 与感知域 AutoML 引擎之间的协议层。它通过 stdio 传输与上层 Agent 交互，对外暴露三类能力：

- **工具调用（Tools）**：29 个 `senseframe_*` 工具，覆盖配置解析、流水线编排、Study 搜索、HPO、AutoML、产物校验、技能库等域。
- **资源自省（Resources）**：12 个 `senseframe://` URI，提供 schema、pipeline graph、scene catalog 等只读元数据。
- **错误路由（Error Routing）**：业务异常经 `to_tool_error` 桥接为 `ToolError`，封装为 `(code, message, category)` 信封供客户端程序化解析。

服务器实现位于 `senseframe/mcp/`，入口为 `server.py`，使用 `FastMCP` 框架。Agent 持有控制循环：MCP 工具仅提供原子原语，不持有长任务状态机（HPO/NAS 通过 Ask-Tell 三步接口暴露给 Agent）。

## 服务器启动

`server.py` 的 `main()` 是唯一入口，流程如下：

1. `validate_config()`：一次性校验所有环境变量，失败时收集全部错误后 raise。
2. `configure_logging()`：配置 root logger 仅写 stderr（stdout 保留给 JSON-RPC）。
3. `_install_signal_handlers()`：注册 SIGTERM/SIGINT 处理器。第一信号设置 `_shutdown_event`（`threading.Event`），等待 in-flight 任务最多 10s 后 `exit 0`；第二信号强制 `exit 1`。
4. `mcp.run(transport="stdio")`：启动 stdio 传输。

`FastMCP` 实例配置：

- `name="senseframe-mcp"`
- `instructions`：提示 Agent 推荐流程（`config_parse → pipeline_create → pipeline_get/advance → artifact_verify`，长任务用 Ask-Tell）
- `lifespan=_lifespan`：startup 记录日志；shutdown 设置 drain flag（SenseFrame 不依赖 SQLite，故不做 DB 连通性检查，区别于 pipeflow）

工具与资源在模块加载时通过 `_register_tools_and_resources()` 程序化注册到 `mcp`（非 `@mcp.tool` 装饰器），便于 AST 守卫测试钉死 `EXPECTED_TOOLS` 与 `_TOOL_REGISTRY` 名称集合一致。

可配置环境变量（`senseframe/mcp/config.py`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SENSEFRAME_LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |
| `SENSEFRAME_LOG_FORMAT` | `text` | 日志格式（text/json） |
| `SENSEFRAME_RATE_LIMIT` | `60` | per-tool 限流（calls/min，0 禁用） |

## 工具（Tools）

29 个工具统一通过 `senseframe/mcp/tool_dispatch.py` 的 `_TOOL_REGISTRY` 集中注册，每个工具是 async 函数，签名约定：

- 必须接受 `ctx: Context[Any, Any, Any] | None = None` 作为最后一个参数（注入 request_id）
- 通过 `MiddlewareStack(RequestIdMiddleware(), RateLimitMiddleware(...))` 包装
- 异常经 `to_tool_error(exc)` 桥接为 `ToolError`
- 返回值是 `FrozenModel` 子类（强类型响应视图）

每个工具在 `_ANNOTATIONS` 矩阵中声明 `ToolAnnotations`（readOnlyHint / destructiveHint / idempotentHint / openWorldHint）。

### config 工具

#### `senseframe_config_parse`

解析 YAML 配置字符串为 `ExperimentConfig`（含 `extra='forbid'` 校验）。

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `config_yaml` | `str` | 是 | — | YAML 格式配置字符串 |
| `ctx` | `Context` | 否 | `None` | MCP Context |

返回 `ConfigParseResponse`（含解析后的 `config` dict）。YAML 语法错误或校验失败 → `ToolError`（config category）。

### pipeline 工具

7 个工具围绕 `PipelineRun` 状态机（Pending → Running → Succeeded/Failed/Paused 等），共享进程级 `PipelineRunStore` 单例。

#### `senseframe_pipeline_create`

声明式创建 PipelineRun。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `config` | `dict[str, Any]` | 是 | — |
| `stages` | `list[str]` | 是 | — |
| `trial_id` | `str \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `PipelineCreateResponse`（`run_id` + `state=Pending` + `_transitions`）。

#### `senseframe_pipeline_advance`

推进状态机（单一状态变更入口，幂等）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `run_id` | `str` | 是 | — |
| `action` | `Literal["start","complete","fail","retry","skip","pause","resume"]` | 是 | — |
| `completed_stage` | `str \| None` | 否 | `None` |
| `failed_stage` | `str \| None` | 否 | `None` |
| `error_message` | `str \| None` | 否 | `None` |
| `trial_id` | `str \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `PipelineAdvanceResponse`（`previous_state` + `new_state` + `action` + `_transitions`）。

#### `senseframe_pipeline_run`

执行完整 pipeline（黑盒，阻塞，调用 `run_pipeline`）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `config` | `dict[str, Any]` | 是 | — |
| `run_id` | `str \| None` | 否 | `None` |
| `stages` | `list[str] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `PipelineRunView`（`state=Succeeded/Failed`）。训练异常自动 `fail` 并记录 `error_message`。

#### `senseframe_pipeline_get`

查询 run 状态（含 `_transitions` HATEOAS）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `run_id` | `str` | 是 | — |
| `ctx` | `Context` | 否 | `None` |

返回 `PipelineRunView`。

#### `senseframe_pipeline_list`

列出所有 run（cursor 分页）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `cursor` | `str \| None` | 否 | `None` |
| `limit` | `int` | 否 | `50` |
| `filter_dict` | `dict[str, Any] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `PipelineRunListView`（`items` + `next_cursor` + `total_count` + `limit`）。`limit` 钳制到 `[1, 200]`。

#### `senseframe_pipeline_pause` / `senseframe_pipeline_resume`

幂等暂停 / 恢复 run。参数仅 `run_id` + `ctx`，返回 `PipelineAdvanceResponse`（`action="pause"/"resume"`）。

### study 工具

7 个工具包装 `StudyManager`，实现 L4 SP（搜索协议）的 Ask-Tell 三步接口。

#### `senseframe_study_create`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `name` | `str` | 是 | — |
| `direction` | `str` | 否 | `"maximize"` |
| `search_space` | `dict \| list \| None` | 否 | `None` |
| `sampler` | `str` | 否 | `"random"` |
| `seed` | `int \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

`sampler` 支持 `random / grid / asha / hyperband`。返回 `StudyCreateResponse`（`study_id` + `transitions=[ask, stop]`）。

#### `senseframe_study_ask`

采样下一个 trial。参数：`study_id: str` + `ctx`。返回 `StudyAskResponse`（`trial_id` + `params` + `transitions=[tell, ask]`）。

#### `senseframe_study_tell`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `trial_id` | `str` | 是 | — |
| `value` | `float` | 是 | — |
| `intermediate_values` | `dict[int, float] \| None` | 否 | `None` |
| `state` | `str` | 否 | `"completed"` |
| `feedback` | `dict[str, Any] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `StudyTellResponse`（`trial_id` + `state` + `value` + `transitions=[ask, stop, get]`）。

#### `senseframe_study_get`

查询 study 状态 + 最佳 trial。参数：`study_id: str` + `ctx`。返回 `StudyView`（含 `n_trials / n_completed / best_value`）。

#### `senseframe_study_list`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `cursor` | `str \| None` | 否 | `None` |
| `limit` | `int` | 否 | `50` |
| `filter_dict` | `dict[str, Any] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

`filter_dict` 支持 `status / direction / sampler / name` 等值过滤。返回 `StudyListView`。

#### `senseframe_study_compare`

多 study 对比（结构化对比表）。参数：`study_ids: list[str]`（至少 2 个）+ `ctx`。返回 `StudyCompareView`（`studies` + `comparison_table` + `best_study_id`，方向感知推导最佳）。

#### `senseframe_study_stop`

停止 study（终态，幂等）。参数：`study_id: str` + `ctx`。返回 `StudyView`（`status=stopped`）。

### hpo 工具

#### `senseframe_hpo_setup`

把 `ExperimentConfig` 的 HPOConfig 转换为 Study 搜索空间（Ask-Tell 入口）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `config` | `dict[str, Any]` | 是 | — |
| `n_trials` | `int` | 否 | `20` |
| `sampler` | `str` | 否 | `"random"` |
| `direction` | `str` | 否 | `"minimize"` |
| `ctx` | `Context` | 否 | `None` |

优先从场景 `get_search_space()` 获取，失败回退到空 `SearchSpace`。返回 `StudyCreateResponse`。

### exploration 工具

#### `senseframe_exploration_recommend`

基于当前 study 的 feedback 推荐下一策略（闭合探索-反馈回路）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `study_id` | `str` | 是 | — |
| `task_type` | `str \| None` | 否 | `None` |
| `top_k` | `int` | 否 | `5` |
| `ctx` | `Context` | 否 | `None` |

`top_k` 钳制到 `[1, 50]`。返回 `ExplorationRecommendationView`（`recommendations` 列表，每项含 `strategy / reason / priority / recommendation_id`）。

### automl 工具

4 个工具操作 `AutoMLOrchestrator`，stages 元素必须是 `nas / hpo / autoaugment`。

#### `senseframe_automl_create`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `config` | `dict[str, Any]` | 是 | — |
| `stages` | `list[str]` | 是 | — |
| `ctx` | `Context` | 否 | `None` |

返回 `AutoMLCreateResponse`（`pipeline_id` + `state=Pending` + `transitions=[start, get]`）。

#### `senseframe_automl_advance`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `pipeline_id` | `str` | 是 | — |
| `action` | `str` | 是 | — |
| `study_id` | `str \| None` | 否 | `None` |
| `error_message` | `str \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

`action` 支持 `start/complete/fail/pause/resume/retry`。返回 `AutoMLAdvanceResponse`。

#### `senseframe_automl_get`

查询流水线状态。参数：`pipeline_id: str` + `ctx`。返回 `AutoMLPipelineView`。

#### `senseframe_automl_list`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `cursor` | `str \| None` | 否 | `None` |
| `limit` | `int` | 否 | `50` |
| `filter_dict` | `dict[str, Any] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

`filter_dict` 支持 `state / failed_stage`。返回 `AutoMLPipelineListView`。

### param_bridge 工具

#### `senseframe_apply_params_extended`

应用采样参数 + 注入工厂字段到 `ExperimentConfig`（联合搜索）。

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `config` | `dict[str, Any]` | 是 | — |
| `params` | `dict[str, Any]` | 是 | — |
| `module_factory` | `Any \| None` | 否 | `None` |
| `datamodule_factory` | `Any \| None` | 否 | `None` |
| `extra_callbacks` | `list[Any] \| None` | 否 | `None` |
| `trainer_factory` | `Any \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

工厂字段不可序列化，仅同进程调用有效（MCP 客户端通常传 None）。返回 `ApplyParamsExtendedResponse`（`config` + `applied_params` + `injected_factories`）。

### artifact 工具

3 个工具复用 `senseframe.engine.runner.pipeline.artifacts_api` 的薄包装层。

#### `senseframe_artifact_verify`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `output_dir` | `str` | 是 | — |
| `recursive` | `bool` | 否 | `False` |
| `ctx` | `Context` | 否 | `None` |

三重校验：hash + manifest schema + 必填产物（`config / metadata / training_log`）。返回 `ArtifactVerifyResponse`（`hash_check / manifest_schema_missing / missing_artifacts / overall_ok`）。`recursive=True` 时扫描子目录（max_depth=3）。

#### `senseframe_artifact_list`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `output_dir` | `str` | 是 | — |
| `cursor` | `str \| None` | 否 | `None` |
| `limit` | `int` | 否 | `50` |
| `filter_dict` | `dict[str, Any] \| None` | 否 | `None` |
| `ctx` | `Context` | 否 | `None` |

返回 `ArtifactListView`（`items` + `next_cursor` + `total_count` + `limit` + `run_id`）。

#### `senseframe_artifact_export`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `output_dir` | `str` | 是 | — |
| `artifact_names` | `list[str] \| None` | 否 | `None` |
| `format` | `str` | 否 | `"zip"` |
| `ctx` | `Context` | 否 | `None` |

`format` 支持 `zip / tar / manifest`。导出到 `output_dir/exports/`。返回 `ArtifactExportResponse`（`output_path / format / artifact_count / total_size_bytes / content_hash / run_id`，`content_hash` 为导出文件的 SHA256）。

### skill 工具

4 个工具复用 `senseframe.skills` 的 `SkillLibrary` 单例。

#### `senseframe_skill_save`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `name` | `str` | 是 | — |
| `code` | `str` | 是 | — |
| `description` | `str` | 否 | `""` |
| `tags` | `list[str] \| None` | 否 | `None` |
| `source_path` | `str` | 否 | `""` |
| `version` | `str` | 否 | `"1.0.0"` |
| `ctx` | `Context` | 否 | `None` |

`code` 通过 `compile` 验证语法。验证失败时返回 `saved=False` + `validation_errors`（不抛异常）。返回 `SkillSaveResponse`。

#### `senseframe_skill_get`

参数：`name: str` + `version: str | None = None` + `ctx`。返回 `SkillView`。技能不存在 → `SkillNotFoundError`。

#### `senseframe_skill_search`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `query` | `str` | 是 | — |
| `top_k` | `int` | 否 | `5` |
| `ctx` | `Context` | 否 | `None` |

`top_k` 钳制到 `[1, 50]`。返回 `SkillSearchResponse`（`items` 按 `score` 降序，仅含 `score > 0`）。

#### `senseframe_skill_remove`

| 参数 | 类型 | 必填 | 默认 |
|------|------|------|------|
| `name` | `str` | 是 | — |
| `force` | `bool` | 否 | `False` |
| `ctx` | `Context` | 否 | `None` |

`force=False` 时有依赖则拒绝（`SkillHasDependentsError`）。返回 `SkillRemoveResponse`。

## 资源（Resources）

12 个 Resource 端点通过 `_RESOURCE_REGISTRY`（`resources/__init__.py`）注册，URI scheme 为 `senseframe://`。Agent 通过 ISP（Introspection Protocol）建立 SenseFrame 心智模型。

| URI | ISP | 用途 |
|-----|-----|------|
| `senseframe://introspect` | ISP-0 | 聚合索引：所有 Resource URI 列表 + 协议版本 + 服务器版本 + tool 清单 |
| `senseframe://schemas/pipeline` | ISP-1 | `PipelineContext` schema |
| `senseframe://schemas/stage` | ISP-2 | `StageSpec` schema |
| `senseframe://schemas/config` | ISP-3 | `ExperimentConfig` JSON schema（调用 `model_json_schema`） |
| `senseframe://schemas/errors` | ISP-4 | 错误码 × 恢复策略映射 |
| `senseframe://tools/output-schemas` | ISP-11 | Tool 输出 schema 索引 |
| `senseframe://pipeline/{run_id}/graph` | ISP-7 | Stage 数据流图（field → producer/consumers） |
| `senseframe://pipeline/{run_id}/readiness` | ISP-9 | 运行时数据就绪度（advisory） |
| `senseframe://scenes` | ISP-8 | Scene 目录（调用 `list_scenes`） |
| `senseframe://scenes/{name}/capabilities` | ISP-9 | Scene 能力（`SceneMeta`） |
| `senseframe://search-space/{scene}/{model_id}` | ISP-10 | 搜索空间（`ParameterSpec` 列表） |
| `senseframe://protocols` | ISP-10 alt | 已注册的 Protocol 类型 |

分层不变量：`resources/` 不得 import `tools`（AST 守卫测试钉死）。

## 错误处理

错误路由机制分三层：

### 1. 异常分类（`errors.py`）

- **协议层错误**（9 子类，继承 `MCPProtocolError`）：`PipelineNotFound / StageNotFound / IllegalTransition / StageOrderViolation / MaxRetriesExceeded / SchemaValidationError / TimeBudgetExceeded / InvalidPathError / RateLimitExceeded`
- **分页错误**（继承 `InvalidPathError`）：`CursorFilterMismatch / InvalidCursor`
- **Artifact 域错误**（继承 `ArtifactError`）：`ManifestNotFoundError / ManifestSchemaError / ArtifactHashMismatchError / MissingRequiredArtifactError / ArtifactPathEscapeError / UnsupportedExportFormatError`
- **Skill 域错误**（继承 `SkillError`）：`SkillNotFoundError / SkillHasDependentsError / SkillValidationError`
- **ML 业务错误**（从 `senseframe.engine.runner.errors` 导入）：`SenseFrameError` 及其子类（`SceneNotRegisteredError / OOMError / CheckpointError / TrainingError / ConfigValidationError` 等），每个异常类有 `error_code` 字符串属性（如 `PIPELINE_NOT_FOUND`）。

### 2. Category 路由（`views/tool_error.py` 的 `_CATEGORY_BY_EXC`）

7 类 category（严格不可增减，定义在 `CategoryT = Literal[...]`）：

| category | 异常类 |
|----------|--------|
| `pipeline` | `PipelineNotFound / StageNotFound / IllegalTransition / StageOrderViolation / MaxRetriesExceeded / TimeBudgetExceeded` |
| `config` | `SchemaValidationError / InvalidPathError / PreflightError / ConfigValidationError / pydantic.ValidationError / MetadataVersionError` |
| `artifact` | `ArtifactError`（及所有子类） |
| `internal` | `RateLimitExceeded / SkillError`（及子类）`/ OOMError / CheckpointError / TrainingError / ModelBuildError / SaveError` |
| `study` | `KeyError`（`StudyManager.ask/tell/get_study` 在 ID 不存在时抛出） |
| `scene` | `SceneNotRegisteredError / DatasetNotSupportedError / ModelNotSupportedError / DataNotFoundError / DataCorruptedError` |
| `search` | 预留（当前未映射，待 HPO 异常落地后启用） |

映射顺序敏感（`isinstance` 按顺序匹配，具体类在前）：`ArtifactError` 与 `SkillError` 子类放在 `KeyError` 之前，避免被 study 兜底匹配。未知异常 → `category='internal'`（脱敏兜底）。

### 3. ToolError 信封（`tools/_errors.py`）

`to_tool_error(exc)` 流程：

1. `ToolErrorResponse.envelope_from(exc)` 路由异常到 category，构造 `(code, message, category)` 信封
2. `logger.exception` 仅记录安全元数据（`code` + `category`），不记录 `message`（M19 修复：`message` 可能含路径/输入等敏感信息）
3. 返回 `ToolError(envelope.model_dump_json())`，客户端可程序化解析 JSON 信封

## 中间件

`middleware.py` 提供基于 async context manager 的轻量中间件链（FastMCP stdio 传输无原生中间件钩子）。

### `MiddlewareStack`

洋葱模式调用：`before` 钩子按注册顺序执行，`after` 钩子按反向顺序执行（由 `finally` 块保证）。用法：

```python
async with _stack.instrument("senseframe_pipeline_create", ctx):
    return pipeline_tools.create(...)
```

每个 tool 模块声明自己的 `_stack`（如 `pipeline._stack`、`study._study_stack`、`automl._automl_stack` 等），统一组合 `RequestIdMiddleware` + `RateLimitMiddleware`。

### 内置中间件

- **`RequestIdMiddleware`**：将 `ctx.request_id` 注入模块级 `ContextVar`（`_request_id_ctx`），tool 调用期间所有层（DAO / orchestrator / FSM）可通过 `get_request_id()` 读取；调用完成后清空为 `"-"`。
- **`RateLimitMiddleware`**：委托 `RateLimiter` 实现（默认 `TokenBucketLimiter`），per-tool bucket。`SENSEFRAME_RATE_LIMIT=0` 完全禁用。错误不退还 token。
- **`TokenBucketLimiter`**：仅依赖 `time.monotonic()`，零外部依赖。`calls_per_minute=0` 完全禁用。拒绝时抛 `RateLimitExceeded`。

## 分页

`pagination/` 目录实现 cursor 分页机制，所有 list 工具统一响应格式：

```json
{
  "items": [...],
  "next_cursor": "str | null",
  "total_count": 100,
  "limit": 50
}
```

### Cursor wire format（`cursor.py`）

```
base64-urlsafe(repr(last_id) || "|" || filter_fingerprint)  # padding stripped
```

- `last_id` 是 str（SenseFrame 的 `run_id / study_id / pipeline_id` 均为 uuid4 hex 或字典序字符串）
- `filter_fingerprint` = `sha256(canonical_json(filter_dict))[:8]`（8 字符 hex）；空 filter 归一化为 `"00000000"`
- 客户端必须将 cursor 视为不透明

### 关键函数

- `encode_cursor(last_id, filter_dict)`：编码
- `decode_cursor(cursor)` → `(last_id, fingerprint)`，失败抛 `InvalidCursor`
- `assert_fingerprint_matches(cursor, filter_dict)` → `last_id | None`，fingerprint 不一致抛 `CursorFilterMismatch`（要求重启 `cursor=None`）
- `clamp_limit(limit)`：钳制到 `[1, 200]`，默认 50
- `build_page(items, total_count, limit, has_more, last_id_fn, filter_dict)`：limit+1 技巧，`next_cursor` 从保留的最后一行 id 编码

## 使用示例

### 示例 1：配置解析 + Pipeline 创建 + 推进

```python
# 1. 解析 YAML 配置（含 extra='forbid' 校验）
config_resp = await session.call_tool("senseframe_config_parse", {
    "config_yaml": "scene:\n  name: eeg\n  model_id: eegnet\n  dataset: bcic_iv_2a\n..."
})
# 2. 声明式创建 PipelineRun（state=Pending）
create_resp = await session.call_tool("senseframe_pipeline_create", {
    "config": config_resp.result["config"],
    "stages": ["validate", "preflight", "load", "build", "train", "eval", "export"]
})
run_id = create_resp.result["run_id"]
# 3. 启动 run（读 _transitions HATEOAS 决定下一步）
await session.call_tool("senseframe_pipeline_advance", {"run_id": run_id, "action": "start"})
state = await session.call_tool("senseframe_pipeline_get", {"run_id": run_id})
```

### 示例 2：HPO Ask-Tell 循环（长任务，Agent 持有控制循环）

```python
# 1. hpo_setup → study_id
study = await session.call_tool("senseframe_hpo_setup", {
    "config": base_config, "n_trials": 20, "direction": "maximize", "sampler": "asha"})
study_id = study.result["study_id"]

# 2. Ask → trial_id + params
trial = await session.call_tool("senseframe_study_ask", {"study_id": study_id})

# 3. apply_params_extended + pipeline_run（阻塞执行）
applied = await session.call_tool("senseframe_apply_params_extended", {
    "config": base_config, "params": trial.result["params"]})
run = await session.call_tool("senseframe_pipeline_run", {"config": applied.result["config"]})

# 4. Tell 上报结果（含 feedback 供 exploration 闭合回路）
await session.call_tool("senseframe_study_tell", {
    "trial_id": trial.result["trial_id"], "value": run.result["best_val_acc"],
    "state": "completed", "feedback": {"status": "converging"}})
recs = await session.call_tool("senseframe_exploration_recommend", {
    "study_id": study_id, "top_k": 3})

# 5. 循环结束 → 取最佳 trial + 校验产物
best = await session.call_tool("senseframe_study_get", {"study_id": study_id})
await session.call_tool("senseframe_artifact_verify", {"output_dir": run.result["output_dir"]})
```
