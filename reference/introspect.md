# 自省协议与探索状态 API 参考

RFC-003 引入的 DSP-1/2/3 内省协议、探索状态管理、技能库、断点续跑的完整 API 参考。Agent 可程序化查询字段契约与填充时机，无需读源码。

## DSP-1：PipelineContext 字段契约

`PipelineContext` 是 stage 间共享的上下文对象。`schema()` 查询字段契约，`filled_at()` 查询每个 stage 填充哪些字段。

### schema() — 字段契约

```python
import senseframe as sf

sf.context_schema()  # 顶层便捷入口
# 等价于 PipelineContext.schema()
```

返回结构：

```python
{
    "schema_version": "1.0.0",
    "fields": [
        {"name": "config", "type": "ExperimentConfig", "fill_stage": "init", "has_default": False},
        {"name": "scene", "type": "SceneContainer", "fill_stage": "stage_validate", "has_default": False},
        {"name": "bundle", "type": "Optional[DatasetBundle]", "fill_stage": "stage_load", "has_default": True},
        ...
    ]
}
```

### 字段 → Stage 映射

| Stage | 填充字段 |
|-------|----------|
| `init`（构造注入） | `config`, `dry_run` |
| `stage_validate` | `scene`, `meta`, `model_id`, `dataset`, `learning_mode` |
| `stage_preflight` | `report`, `route_level`, `route_config`, `output` |
| `stage_resolve` | `scene_info`, `num_classes`, `task_spec`, `feature_spec`, `resolved`, `lightning_params`, `distributed_kwargs` |
| `stage_load` | `scene_kwargs`, `bundle`, `data_profile`, `output_dir`, `log_writer`, `data_hash` |
| `stage_build` | `model`, `datamodule`, `module`, `callbacks`, `pl_logger`, `csv_logger`, `monitor` |
| `stage_probe_vram` | `vram_probe_result`（方案 B：动态显存探测结果，写入 metadata.resource.vram_probe） |
| `stage_train` | `trainer`, `training_duration_s`, `best_model_path`, `best_model_score`, `best_epoch`, `intermediate_values` |
| `stage_export`（metadata） | `best_epoch`, `best_model_path`, `best_model_score`, `epoch_utilization` 持久化到 metadata.json + pipeline_checkpoint.json |
| `stage_eval` | `final_eval`, `training_log`, `early_stopped`, `feedback` |
| `stage_export` | `artifact_registry` |
| `agent`（运行时） | `trial_id`, `parent_trial_id`, `exploration_history`, `extra`, `completed_stages`, `stage_checkpoint_path`, `failed_stage`, `failed_error` |

### 运行时状态查询

```python
ctx = sf.PipelineContext(config=my_config)

ctx.filled_at("stage_load")    # ["bundle", "data_profile", "output_dir", "log_writer"]
ctx.completed_fields()         # 当前已填充（非 None）的字段名列表
ctx.describe()                 # 运行时状态摘要
```

`describe()` 返回：

```python
{
    "completed_fields": ["config", "scene", "meta", ...],
    "extra_keys": [],     # extra 字典的键（用户自定义扩展，框架不写入；feedback/failed_stage 是 first-class 字段，不在 extra 中）
    "trial_id": "trial_0001",
    "completed_stages": ["validate", "preflight", "resolve", "load"],
}
```

## DSP-2：DatasetBundle 填充契约

`DatasetBundle` 按 `learning_mode` 校验 required/forbidden 字段。

### filling_rule(learning_mode)

```python
from senseframe.scenes.base import DatasetBundle

DatasetBundle.filling_rule("supervised")
# {"train": "required", "test": "required", "val": "optional",
#  "unsupervised": "forbidden", "supervised_finetune": "forbidden"}

DatasetBundle.filling_rule("self_supervised")
# {"train": "forbidden", "test": "required", "val": "optional",
#  "unsupervised": "required", "supervised_finetune": "required"}
```

### 校验与自省

```python
bundle = scene.load_dataset(...)

bundle.filled_fields()                      # ["train", "test"]（非 None 字段）
errors = bundle.validate_filling("supervised")  # 错误列表，空列表表示通过
bundle.describe("self_supervised")          # 运行时状态
```

`validate_filling` 错误格式：
- `"Field '{field}' is required for learning_mode='{mode}' but is None"`
- `"Field '{field}' is forbidden for learning_mode='{mode}' but is not None"`

`describe()` 返回：

```python
{
    "filled_fields": ["train", "test"],
    "learning_mode": "supervised",
    "validation_errors": [],
}
```

### schema()

```python
sf.data_bundle_schema()  # 顶层便捷入口
```

返回结构：

```python
{
    "schema_version": "1.0.0",
    "fields": [
        {"name": "train", "type": "Optional[Dataset]"},
        {"name": "test", "type": "Optional[Dataset]"},
        {"name": "val", "type": "Optional[Dataset]"},
        {"name": "unsupervised", "type": "Optional[Dataset]"},
        {"name": "supervised_finetune", "type": "Optional[Dataset]"},
    ],
    "filling_rules": {
        "supervised": {...},
        "self_supervised": {...},
    }
}
```

## DSP-3：StageSpec IO 声明

每个 stage 通过 `@stage` 装饰器声明 `reads` / `writes`，Agent 可程序化查询 stage 依赖关系。

### StageSpec / FieldSpec 结构

| 结构 | 字段 | 类型 | 默认值 |
|------|------|------|--------|
| `FieldSpec` | `name` | str | (必填) |
| | `type` | str | `"Any"` |
| | `required` | bool | `True` |
| | `description` | str | `""` |
| `StageSpec` | `name` | str | (必填) |
| | `reads` | List[FieldSpec] | `[]` |
| | `writes` | List[FieldSpec] | `[]` |
| | `description` | str | `""` |

### 查询 API

```python
import senseframe as sf

# 列出全部 stage 名
sf.list_stages()
# ["validate", "preflight", "resolve", "load", "build", "train", "eval", "export"]

# 查询单个 stage 的 reads / writes（stage 名无 stage_ 前缀）
sf.stage_io("train")
# {"name": "train", "reads": [...], "writes": [...], "description": "..."}

# 查询全部 stage 的 IO 声明
sf.stage_io()  # {"stages": [spec_dict, ...]}

# Pipeline 字段依赖图（DAG）
sf.pipeline_graph()
# {"fields": {"config": {"producers": ["init"], "consumers": ["validate", "resolve", ...]}, ...}}
```

### 自定义 stage 声明

```python
from senseframe.engine.runner.pipeline import stage, Pipeline

@stage(
    name="my_eval",
    reads=["model", "datamodule", "trainer"],
    writes=["final_eval"],
    description="自定义评估 stage",
)
def my_eval(ctx):
    ...
    ctx.final_eval = metrics
    return ctx

pipeline = Pipeline.default()
pipeline.replace_stage("eval", my_eval)
pipeline.stages_with_spec()  # 含 my_eval 的 StageSpec
```

## 探索状态管理

闭合探索-反馈回路：训练结果 → 结构化反馈 → 探索历史 → 推荐下一步。

### PipelineContext 探索字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `trial_id` | `Optional[str]` | 当前试验 ID |
| `parent_trial_id` | `Optional[str]` | 父试验 ID（支持回溯） |
| `exploration_history` | `List[Dict]` | 探索历史列表 |
| `feedback` | `Optional[Dict]` | stage_eval 产出的结构化反馈（first-class 字段） |

### record_trial()

```python
ctx = sf.PipelineContext(config=my_config)
ctx.trial_id = "trial_001"
ctx.parent_trial_id = "trial_000"
ctx.record_trial(
    strategy={"loss": "focal", "lr": 0.001},
    result={"val_accuracy": 0.85},
)
```

写入 `exploration_history` 的记录结构：

```python
{
    "trial_id": "trial_001",
    "parent_trial_id": "trial_000",
    "strategy": {"loss": "focal", "lr": 0.001},
    "result": {"val_accuracy": 0.85},
    "timestamp": "2026-01-01T12:00:00",
    # 以下由 stage_eval 回写：
    "feedback": {"status": "converged", "diagnosis": "...", "suggestions": [...]},
    "status": "completed",
}
```

### stage_eval 结构化反馈

`stage_eval` 自动调用 `analyze_training_result` 产出 feedback，写入 `ctx.feedback`（first-class 字段）和 `exploration_history[-1]`：

| status | 触发条件 | suggestions 示例 |
|--------|----------|------------------|
| `numerical_instability` | 指标含 NaN/Inf | 降低 lr / 检查 loss 数值稳定性 / 启用梯度裁剪 |
| `underfitting` | `val_acc < 0.5`（分类） | 增大模型容量 / 增加 epochs / 降低 weight_decay |
| `overfitting` | `train_acc - val_acc > 0.15` | 增加数据增强 / 增大 weight_decay / 启用 dropout |
| `converged` | `early_stopped=True` | 更激进策略 / 不同特征工程 |
| `success` | 默认 | 记录到技能库 / 探索 recommend_next |

## ExplorationTracker

探索状态管理器，支持历史查询、推荐、持久化。

### 基础操作

```python
from senseframe.exploration import ExplorationTracker

tracker = ExplorationTracker(ctx.exploration_history)

# 添加试验（返回 trial_id，None 时自动生成 "trial_0001"）
tid = tracker.add_trial(
    strategy={"loss": "focal", "lr": 0.001},
    result={"val_accuracy": 0.85},
    feedback={"status": "success"},
)
# 语义化别名（仅传 strategy/result/feedback）
tid = tracker.submit_trial(strategy={...}, result={...})

# 查询
tracker.list_trials(status="completed")  # 按 status 过滤
tracker.get_trial("trial_0001")           # 按 ID 查询
tracker.best_trial(metric="val_accuracy", mode="max")  # 最优试验
tracker.explored_strategies()             # 去重后的策略列表
tracker.last_feedback()                   # 最近一次完成的 feedback
tracker.coverage()                        # 覆盖率统计
```

### coverage() 返回结构

```python
{
    "total_trials": 10,
    "completed": 8,
    "pending": 1,
    "failed": 1,
    "unique_strategies": 7,
}
```

### recommend_next() — 推荐下一步策略

```python
recommendations = tracker.recommend_next(task_type="classification", top_k=5)
```

每项返回：

```python
{
    "strategy": {"loss": "cross_entropy", "lr": 0.001, "batch_size": 64},
    "reason": "HPO 参数空间候选",
    "priority": "high",                   # 仅 feedback 感知推荐时存在
    "recommendation_id": "abc123",        # 用于 log_adoption 追溯
}
```

推荐来源（按优先级）：
1. **feedback 感知定向推荐**：基于 `last_feedback()` 的 status 生成对症策略
2. **兼容性矩阵候选**：`task_type` 指定时，从兼容的 loss/metric 组合生成
3. **CSI transform pipeline/augment 候选**：对 NTU-Fi_HAR / Widar / UT_HAR_data
4. **HPO 参数空间候选**：`learning_rate` × `batch_size` 网格

### 采纳追溯

```python
# 记录 Agent 采纳的推荐
tracker.log_adoption(
    recommendation_id="abc123",
    actual_strategy={"loss": "focal", "lr": 0.0005},
    reason="基于 overfitting feedback 调低 lr",
)

# 追溯链路：feedback → recommended → adopted
tracker.feedback_trace()
```

`feedback_trace()` 每项返回：

```python
{
    "feedback_status": "overfitting",
    "recommended_strategy": {"loss": "focal", "lr": 0.001},
    "adopted_strategy": {"loss": "focal", "lr": 0.0005},
    "adopted": True,
    "timestamp": "2026-01-01T12:00:00",
}
```

### 持久化

```python
# 保存（stage_export 自动调用）
tracker.save("runs/<实验目录>/exploration.json")

# 加载
tracker = ExplorationTracker.load("runs/<实验目录>/exploration.json")
```

保存的 JSON 结构：`{"history": [...], "action_log": [...]}`

## 技能库

将成功策略保存为可复用 Skill，下次 `search_skills` 复用。

### Skill 结构

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | (必填) | 技能名 |
| `description` | str | (必填) | 描述 |
| `code` | str | (必填) | 代码内容 |
| `tags` | List[str] | `[]` | 标签 |
| `version` | str | `"1.0.0"` | 版本号 |
| `created_at` | str | 当前时间 | 创建时间 ISO |
| `validated` | bool | `False` | 验证状态 |
| `validation_errors` | List[str] | `[]` | 验证错误 |
| `depends_on` | List[str] | `[]` | 依赖的其他技能名 |
| `source_path` | str | `""` | 来源扩展文件路径 |

### API

```python
import senseframe as sf

# 保存技能（返回 True 成功，False 验证失败）
sf.save_skill(
    name="wifi_csi_focal",
    code="register_loss('focal')(...)",
    description="WiFi CSI + Focal Loss",
    tags=["wifi_csi", "loss"],
)

# 加载技能（version=None 返回最新版）
skill = sf.load_skill("wifi_csi_focal", version=None)

# 全文检索（按相关度降序）
matches = sf.search_skills(query="WiFi CSI loss", top_k=5)

# 列出全部技能名
sf.list_skills()
```

技能存储路径默认 `~/.senseframe/skills/`，每技能保存为 `{name}.py` + `{name}.meta.json`。`search_skills` 默认用 hash-based 轻量嵌入；安装 sentence-transformers 后自动启用语义嵌入。

## 断点续跑

Pipeline 支持 stage 级断点续跑，失败 stage 后可从断点恢复，无需重跑已完成 stage。

### 失败时的提示

Pipeline 失败时输出 stderr 提示：

```
Pipeline failed at stage 'train'. To resume: Pipeline.resume('runs/<实验目录>')
```

### 续跑流程

```python
import senseframe as sf
from senseframe.engine.runner.pipeline import Pipeline

# 续跑：从 output_dir/pipeline_checkpoint.json 恢复
pipeline, completed = Pipeline.resume("runs/<实验目录>")
# completed = ["validate", "preflight", "resolve", "load", "build"]

# 构造新 ctx，传入 completed_stages 和 checkpoint 路径
ctx = sf.PipelineContext(config=my_config)
ctx.completed_stages = completed
ctx.stage_checkpoint_path = Path("runs/<实验目录>/pipeline_checkpoint.json")

result = pipeline.run(ctx)  # 跳过 completed_stages，从 "train" 开始
```

### checkpoint 文件结构

`{output_dir}/pipeline_checkpoint.json`：

```json
{
    "completed_stages": ["validate", "preflight", "resolve", "load", "build"],
    "trial_id": "trial_0001",
    "timestamp": "2026-01-01T12:00:00",
    "failed_stage": "train"
}
```

### PipelineContext 续跑字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `completed_stages` | `List[str]` | 已完成的 stage 名列表 |
| `stage_checkpoint_path` | `Optional[Path]` | pipeline_checkpoint.json 路径 |

`Pipeline.run` 启动时若 `stage_checkpoint_path` 存在且文件存在，自动加载并恢复 `completed_stages`，跳过已完成 stage。每 stage 成功后追加到 `completed_stages` 并写 checkpoint；失败时写 `failed_stage` 字段。

## DataProfile 字段契约

`DataProfiler.profile_bundle()` 输出的数据画像字段契约：

```python
sf.data_profile_schema()
```

返回结构：

```python
{
    "schema_version": "1.0.0",
    "fields": [
        {"name": "dtypes", "type": "Dict[str, str]"},
        {"name": "feature_names", "type": "List[str]"},
        {"name": "nullable", "type": "Dict[str, bool]"},
        {"name": "shapes", "type": "Dict[str, Tuple[int, ...]]"},
    ]
}
```

`DataProfile` 实例字段：`n_samples`, `input_shape`, `n_features`, `n_classes`, `class_distribution`, `missing_rate`, `value_range`, `mean`, `std`, `is_spatial`, `is_temporal`, `modality`, `recommended_task_type`, `recommended_loss`, `recommended_metrics`, `recommended_normalization`, `dataset_name`, `dtypes`, `feature_names`, `nullable`, `shapes`, `profile_source`, `imbalance_ratio`, `recommended_class_weights`。
