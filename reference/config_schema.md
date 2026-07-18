# ExperimentConfig Schema 字段参考

声明式实验配置的完整字段定义与校验规则。

P1 演进（2026-07-18）：6 个配置类已从 stdlib `dataclass` 迁移到 `pydantic v2 BaseModel`。
- 构造时（`__init__` / `from_dict` / `model_validate`）自动校验字段类型与约束，错误快速失败抛 `ValidationError`
- `ExperimentConfig.validate()` 保留为兼容方法，仅校验 pydantic 无法在构造时覆盖的延迟约束（如 `data_root` 由 CLI/env 后填充）
- `from_dict()` / `to_dict()` 保留为薄封装，委托 `model_validate()` / `model_dump()`
- JSON Schema 由 `ExperimentConfig.model_json_schema()` 自动生成（替代手写 schema）

## 顶层结构

```yaml
scene: SceneConfig              # 必需
input_features: List[InputFeature]   # 必需，非空
output_features: List[OutputFeature] # 必需，非空
trainer: TrainerConfig          # 可选，有默认值
hpo: HPOConfig                  # 可选，有默认值
output_dir: str = "runs"      # 可选
save_model: bool = true         # 可选
```

**命令式注入字段**（仅 Python 代码可设，不参与 `from_dict/to_dict` 序列化，也不出现在 `model_json_schema()` 生成的 JSON Schema 中）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `module_factory` | `Optional[Callable]` | `Callable(model, **kw) -> pl.LightningModule` |
| `datamodule_factory` | `Optional[Callable]` | `Callable(train_ds, test_ds, **kw) -> pl.LightningDataModule` |
| `trainer_factory` | `Optional[Callable]` | `Callable(**kw) -> pl.Trainer`（RFC-002 阶段 K） |
| `extra_callbacks` | `List[Any]` | `List[pl.Callback]` |

`trainer_factory` 为 None 时使用框架默认 `pl.Trainer` 构造逻辑。

P2 演进（2026-07-18）：这 4 个工厂字段已从 `ExperimentConfig` 拆分到独立的 `RuntimeInjections` dataclass（非 pydantic BaseModel）。
- `ExperimentConfig` 通过 `runtime: RuntimeInjections = Field(exclude=True)` 持有运行时注入对象
- 通过 `@property` 代理访问，`cfg.module_factory` 等价于 `cfg.runtime.module_factory`
- 构造时传入工厂字段会被 `model_validator(mode='before')` 提取并转发到 `runtime`：
  `ExperimentConfig(..., module_factory=...)` → `runtime=RuntimeInjections(module_factory=...)`
- `runtime` 字段用 `Field(exclude=True)` 排除 `model_dump()`，`model_json_schema()` 覆盖移除
- YAML 中不允许声明 `runtime` 或工厂字段（运行时对象不可序列化），`from_dict` 显式拒绝

访问方式（两种等价）：
```python
# 方式 1：兼容代理（推荐，旧代码无需修改）
cfg.module_factory = my_factory
cfg.extra_callbacks.append(my_callback)

# 方式 2：直接访问 runtime
cfg.runtime.module_factory = my_factory
cfg.runtime.extra_callbacks.append(my_callback)
```

## SceneConfig

场景配置：声明使用的场景容器、数据集、模型与学习范式。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | str | 是 | - | 场景名，如 `wifi_csi` |
| `dataset` | str | 是 | - | 数据集名 |
| `model_id` | str | 是 | - | 模型 ID |
| `learning_mode` | str | 否 | `supervised` | `supervised` / `self_supervised` |
| `data_root` | str | 否 | `""` | 数据根目录，必填（YAML/CLI/env 三选一），空字符串触发 env/CLI 回退 |
| `params` | SceneParams \| dict | 否 | `None` | 场景特定参数（escape hatch）；YAML 中的 dict 会在 from_dict 时自动转为 SceneParams 实例 |

**校验规则**：
- `name` / `dataset` / `model_id` 不能为空
- `learning_mode` 必须在 `("supervised", "self_supervised")` 中

**params 常用透传键**：
- `metrics`：评测指标列表，如 `[accuracy, macro_f1, micro_f1]`
- `average`：F1 平均方式，`macro` / `micro` / `weighted`
- `self_supervised_epochs`：自监督预训练轮数（仅 self_supervised）
- `gpu`：GPU 隔离，设置 `CUDA_VISIBLE_DEVICES`
- `resume`：checkpoint 恢复路径

## InputFeature

输入特征声明。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | str | 是 | - | 特征名 |
| `type` | str | 是 | - | 特征类型 |
| `shape` | list | 否 | `[]` | 形状 hint |

**支持的 type**：`csi` / `tabular` / `image` / `text` / `sequence`

**校验规则**：
- `name` 不能为空
- `type` 必须在支持列表中
- `shape` 必须是 list/tuple

## OutputFeature

输出特征声明。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `name` | str | 是 | - | 特征名 |
| `type` | str | 是 | - | 输出类型 |
| `num_classes` | int | 条件必填 | null | category/binary 时必填 |

**支持的 type**：`category` / `number` / `binary`

**校验规则**：
- `name` 不能为空
- `type` 必须在支持列表中
- `type` 为 `category`/`binary` 时 `num_classes` 必填
- `num_classes` 必须 >= 2

## TrainerConfig

训练器配置。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `epochs` | int | 否 | 100 | 训练轮数，>0 |
| `learning_rate` | float | 否 | null | 学习率，null=用模型默认值，>0 |
| `batch_size` | int | 否 | 64 | 批大小，>0 |
| `optimizer` | str | 否 | `adam` | 优化器 |
| `weight_decay` | float | 否 | `1e-4` | 权重衰减（L2 正则），>=0 |
| `early_stopping` | int | 否 | `5` | patience，null=不启用 |
| `early_stopping_min_delta` | float | 否 | `0.001` | val_loss 提升阈值，低于此值视为无提升 |
| `early_stopping_monitor` | str | 否 | `val_loss` | 早停监控指标名 |
| `deterministic` | bool | 否 | false | 确定性模式 |
| `max_time` | str | 否 | null | DD:HH:MM:SS 格式 |
| `seed` | int | 否 | 42 | 随机种子 |
| `enable_progress_bar` | bool | 否 | true | 进度条开关，后台进程可关闭 |
| `scheduler` | str | 否 | `cosine` | 学习率调度器：null/cosine/step |
| `gradient_clip_val` | float | 否 | null | 梯度裁剪阈值，null=不裁剪 |
| `gradient_clip_algorithm` | str | 否 | `norm` | 梯度裁剪算法：norm/value |
| `accumulate_grad_batches` | int | 否 | 1 | 梯度累积步数，>0 |
| `logger` | str | 否 | `csv` | 日志后端：csv/tensorboard/wandb/none |
| `self_supervised_epochs` | int | 否 | 100 | 自监督预训练轮数 |
| `metrics` | list[str] | 否 | `["accuracy", "macro_f1"]` | 评估指标列表 |
| `gpu` | int | 否 | null | 指定 GPU ID，null=自动 |
| `resume` | str | 否 | null | resume checkpoint 路径，null=不恢复 |
| `mixed_precision` | bool/str | 否 | null | 混合精度：True=16-mixed, False=32, 字符串直接透传（如 "bf16-mixed"）, null=自动 |
| `limit_train_batches` | int/float | 否 | null | 限制训练 batch 数（dry-run 用），null=不限制；1 或 1.0=只跑 1 batch |
| `limit_val_batches` | int/float | 否 | null | 限制验证 batch 数（dry-run 用），null=不限制 |
| `num_workers` | int | 否 | null | DataLoader 并行加载进程数，null=由 routing 按资源自动派生；Windows + Python 3.14 spawn 模式下 num_workers>0 需 `if __name__=='__main__'` 保护，可显式设为 0 规避 multiprocessing 错误 |
| `auto_lr_find` | bool | 否 | false | 自动 LR 标定（Lightning LR Range Test），true 时 stage_train 调 trainer.tune() |

**支持的 optimizer**：`adam` / `adamw` / `sgd` / `rmsprop`
**支持的 logger**：`csv`（默认，始终可用）/ `tensorboard`（需 `pip install tensorboard`）/ `wandb`（需 `pip install wandb`）/ `none`（关闭日志）

**校验规则**：
- `epochs` > 0
- `learning_rate` > 0
- `batch_size` > 0
- `weight_decay` >= 0
- `early_stopping` 为 null 或 > 0

## HPOConfig

超参搜索配置（基于 Optuna）。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `enabled` | bool | 否 | false | 是否启用 HPO |
| `n_trials` | int | 否 | 20 | trial 数量，>0 |
| `sampler` | str | 否 | `tpe` | 采样器 |
| `pruner` | str | 否 | `median` | 剪枝器 |
| `metric` | str | 否 | `val_loss` | 优化指标名 |
| `direction` | str | 否 | `minimize` | 优化方向 |
| `storage` | str | 否 | null | 持久化后端 URL（如 `sqlite:///runs/hpo.db`） |
| `study_name` | str | 否 | null | study 名称，配合 `load_if_exists` 断点续搜 |
| `load_if_exists` | bool | 否 | false | True 时恢复同名 study，续搜剩余 trial |
| `export_path` | str | 否 | null | 结果导出 JSON 路径（含 summary 摘要） |
| `timeout` | float | 否 | null | 搜索超时秒数，与 `n_trials` 任一满足即停止 |

**支持的 sampler**：`tpe` / `random` / `cmaes`
**支持的 pruner**：`median` / `none` / `hyperband`
**支持的 direction**：`minimize` / `maximize`

**校验规则**（P1 演进：pydantic 构造时即校验，不再依赖 `enabled` 短路）：
- `n_trials` > 0
- `sampler` / `pruner` / `direction` 必须在支持列表中
- `metric` 不能为空
- `load_if_exists: true` 时必须指定 `study_name`
- `timeout` 若非 null 必须 > 0

> 旧实现仅在 `enabled: true` 时校验，pydantic 版本在构造时即校验所有字段。
> 默认值均合法（`n_trials=20` / `sampler="tpe"` / ...），因此 `HPOConfig()` 默认构造不受影响。
> 若显式传入非法值（如 `sampler="garbage"`），即使 `enabled=false` 也会在构造时报错——这是更严格的契约，符合 fail-fast 原则。

### HPO 持久化与断点续搜

```yaml
hpo:
  enabled: true
  n_trials: 50
  storage: "sqlite:///runs/hpo_study.db"  # SQLite 持久化
  study_name: "exp1_resnet18"
  load_if_exists: true                     # 断点续搜
  export_path: "runs/hpo_result.json"      # 结果导出
  timeout: 3600.0                          # 1 小时超时
```

- **持久化**：`storage` 指定后端 URL，trial 历史持久化到磁盘，进程崩溃不丢失
- **断点续搜**：`load_if_exists: true` + `study_name` 恢复已有 study，仅执行剩余 `n_trials - len(existing_trials)` 个 trial
- **结果导出**：`export_path` 将 HPOOutput + summary 摘要写入 JSON，含 top-N trials、参数分布统计、完成率
- **超时控制**：`timeout` 秒数到达时停止搜索，与 `n_trials` 任一先满足即停止

### HPO 结果摘要

`HPOOutput.summary(top_n=5)` 返回统计摘要（无需可视化依赖）：
- `top_trials`：按 metric 排序的 top-N trial（minimize 升序 / maximize 降序）
- `param_distribution`：float 参数的 min/max/mean/count 分布
- `completion_rate` / `failure_rate` / `prune_rate`：完成/失败/剪枝比率

## 配置解析流程

1. `ExperimentConfig.from_dict(yaml_dict)` 解析嵌套 dict 为 pydantic BaseModel 实例（构造时自动校验字段类型与约束）
2. `config.validate()` 递归校验延迟约束（如 `data_root` 由 CLI/env 后填充，构造时为空字符串不报错）
3. 校验通过后传给 `run_experiment(config)`

```python
from senseframe.engine import ExperimentConfig, run_experiment
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)

config = ExperimentConfig.from_dict(config_dict)
config.validate()  # 显式校验
output = run_experiment(config)
```

## 字段透传机制

`scene.params` 中的键会透传到顶层配置 dict，允许覆盖 trainer 字段或传递 TrainerConfig 未定义的字段（escape hatch）：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  params:
    metrics: [accuracy, macro_f1]    # 透传到顶层
    self_supervised_epochs: 100      # 透传到顶层
    gpu: 0                           # 透传到顶层
```

已知透传键：`self_supervised_epochs`、`metrics`、`average`、`scheduler`、`gpu`、`resume`、`mixed_precision`。

### scheduler 配置说明

`scheduler` 已加入 TrainerConfig，推荐在 `trainer` 下直接配置：

```yaml
trainer:
  epochs: 200
  scheduler: cosine              # 直接在 trainer 下配置
```

支持值：`null`（不启用）、`cosine`（余弦退火）、`step`（阶梯衰减）。

- `cosine`：`T_max` 动态取 `max_epochs`（经路由 cap 后的实际 epochs），无 `max_epochs` 时回退 50
- `step`：`step_size` 动态取 `max(1, max_epochs // 3)`，无 `max_epochs` 时回退 30，`gamma=0.1`

**epochs 实际值来源**：`config.trainer.epochs` 与路由表 `RESOURCE_ROUTES[route_level]["max_epochs"]`（路由层内部字段名，非 TrainerConfig 字段）的较小值：
- `cpu_minimal`：cap 至 50 epochs
- `cpu_standard`：cap 至 200 epochs
- `gpu_*`：无上限（使用配置值）

**向后兼容**：通过 `scene.params.scheduler` 透传的旧配置仍可工作，但推荐迁移到 `trainer.scheduler`。

### logger 配置说明

`logger` 字段控制训练日志后端：

```yaml
trainer:
  logger: tensorboard          # 使用 TensorBoard
```

支持值与依赖：
- `csv`（默认）：CSV 文件，始终可用，输出到 `output_dir/metrics/`
- `tensorboard`：TensorBoard 事件文件，需 `pip install tensorboard`
- `wandb`：Weights & Biases，需 `pip install wandb` 并配置 `wandb login`
- `none`：关闭日志记录

未安装对应包时，runner 会抛出清晰的 `RuntimeError` 提示安装命令。

### 数据变换与缓存

数据管道增强能力：

- **TransformConfig 接口**：场景容器通过 `get_transforms(dataset)` 返回 `TransformConfig`，DataModule 在 `__getitem__` 后应用变换。WiFi CSI 场景因变换已内置于 Dataset，返回空配置；新场景可覆写此方法实现数据增强。
- **streaming 模式**：`GenericDataModule(streaming=True)` 支持 `IterableDataset`，适用于超大 CSV 数据集流式读取，避免全量加载。
- **cache_dir 缓存**：`GenericDataModule(cache_dir=...)` 支持数据集序列化缓存，命中时跳过数据加载，加速重复实验。

### 可观测性增强

- **confusion_matrix**：最终验证阶段自动计算混淆矩阵，包含在 `final_eval` 返回值的 `confusion_matrix` 字段（`List[List[int]]` 格式）。
- **learning_rate 日志**：每 epoch 结束自动 log 当前学习率到 `learning_rate` 指标，便于监控 scheduler 衰减曲线。
- **artifact_manifest.json**：`postprocess.py` 生成产物清单，含所有训练 + 后处理产物的路径、大小、SHA256 校验和，便于完整性校验与版本追溯。

### 分布式训练

支持多卡分布式训练，通过 YAML 顶层字段配置（透传到 Lightning Trainer）：

```yaml
# 多卡 DDP 示例
devices: 2                    # GPU 数量（int 或 "auto"）
strategy: "ddp"               # 分布式策略：ddp / ddp_spawn / fsdp
sync_batchnorm: true          # 同步 BatchNorm（分布式训练推荐开启）

# 多节点训练示例
devices: 4
strategy: "ddp"
num_nodes: 4                  # 节点数
sync_batchnorm: true
```

支持字段（YAML 顶层，ExperimentConfig 声明字段）：
- `devices`：GPU 数量，默认 1（单卡，向后兼容）；支持 int 或 `"auto"`（自动检测）
- `strategy`：分布式策略，默认 null（单设备）；支持 `"ddp"` / `"ddp_spawn"` / `"fsdp"` 等 Lightning 支持的策略
- `num_nodes`：多节点训练节点数，默认 1
- `sync_batchnorm`：是否启用同步 BatchNorm，默认 false
- `num_processes`：CPU 模式并行进程数，默认 1（仅 `device: cpu` 时生效）

> P2 修复（2026-07-18）：这 5 个字段原在文档中声明为"YAML 顶层字段"，但未在
> ExperimentConfig 中声明，被 `extra="ignore"` 静默丢弃，routing.py 永远读到默认值
> ——分布式训练 YAML 配置实际不生效。现已提升为声明字段，并通过
> `experiment_config_to_dict` 透传到 routing 层。

**混合精度**：通过 `mixed_precision` 字段配置，支持 `"16-mixed"` / `"bf16-mixed"` / `"32"`。

### 结构化错误码与状态摘要

`TrainOutput.error_code` 字段供 Agent 程序化分支决策。完整 14 项 error_code 枚举见 [troubleshooting.md](./troubleshooting.md)。

**TrainOutput.summary()**：生成机器可读的状态摘要：

```python
output = run_experiment(config)
summary = output.summary()
# success 时：{"status": "success", "key_metrics": {...}, "model_path": "...", "duration_s": ...}
# error 时：{"status": "error", "error_code": "OOM_ERROR", "error": "..."}
```

### 场景能力声明

`SceneMeta` 的 `supported_learning_modes` 字段显式声明场景支持的学习模式：

```python
meta = scene.meta()
# wifi_csi: ["supervised", "self_supervised"]
# generic: ["supervised"]（默认）
```

此字段与 `supported_tasks` 正交：
- `supported_tasks`：ML 任务类型（classification / regression）
- `supported_learning_modes`：训练范式（supervised / self_supervised）

### CLI 预检模式

`experiment --dry-run` 执行启动前检查不实际训练，详见 [resource_routing.md](./resource_routing.md) 的"预检模式"章节。

**list-scenes 增强**：输出含 `supported_learning_modes` 和 `input_shape_hint` 字段，供 Agent 程序化查询场景能力。
