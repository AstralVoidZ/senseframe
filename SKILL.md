---
name: "senseframe"
description: "AI Agent 驱动的 AutoML 训练框架（RFC-003 五层协议栈）。Invoke when user wants to train/evaluate/export models, register custom strategies (loss/metric/task_type/model/scene/normalization), profile data, compose stage pipelines, inject factories, load extensions, run HPO, use self-supervised paradigm, run NAS (DARTS/ENAS), AutoAugment search, meta-learning warm-start, orchestrate pipelines via OP (Orchestrator), run ε6 comparison experiments (Method/Baseline/ExperimentRunner), use multi-fidelity early pruning (ASHA/Hyperband), persist OP runs to store, introspect pipeline contracts, verify artifact integrity, release training resources, handle structured SenseFrameError exceptions via error_code, activate lazy scenes, resume failed pipelines, or manage exploration history and skills."
---

# SenseFrame

Agent 驱动的 AutoML 训练框架。提供可编程原语库 + 执行底座 + 安全护栏，Agent 持有训练流程控制权。

**RFC-003 五层协议栈**：DSP（数据交换）/ SP（搜索协议 Ask-Tell）/ IP（自省协议）/ OBP（观测性）/ OP（编排协议）。所有高级能力（HPO / NAS / AutoAugment / 元学习 / 对比实验）均通过 SP+OP 编排，可被 Agent 程序化驱动。

## 何时调用

- 训练模型（监督 / 自监督 / HPO）
- 注册自定义策略（loss / metric / task_type / model / scene / normalization）
- 探查数据特征，基于数据画像选择策略
- 编排自定义训练流程（Stage Pipeline）
- 注入自定义 LightningModule / DataModule / Callbacks / Trainer
- 通过 `load_extension` 加载扩展代码
- 编写 / 校验 YAML 实验配置
- 排查训练失败（基于 `error_code` 程序化决策，而非字符串匹配）
- 导出模型到 ONNX / TorchScript 等部署格式
- 校验训练产物完整性（`verify_artifacts`）
- 释放训练资源（`release_resources`，长任务/HPO 必用）
- 开发新场景容器
- 查询 pipeline / context / 数据契约（自省 API）
- 管理探索历史与技能库（策略复用）
- 从失败 stage 断点续跑 pipeline
- **P3 — 神经架构搜索**：DARTS（可微）/ ENAS（权重共享）/ Evolutionary
- **P3 — AutoAugment 数据增强搜索**：搜索最优增强策略组合
- **P3 — 元学习 warm-start**：用源数据集成功策略 warm-start 目标数据集 SP Study
- **P3 — Multi-fidelity 早停**：ASHA / Hyperband pruner 剪枝差 trial
- **P3 — OP 编排（Orchestrator）**：create_run + start + reconcile + complete/fail + 事件订阅 + 持久化
- **P3 — ε6 对比实验**：MethodRunner（SP 驱动）/ BaselineRunner（固定参数）/ ExperimentRunner（编排聚合）/ ComparisonReport

## 前置条件

- Python 3.x + PyTorch + PyTorch Lightning 已安装
- 数据集放置于 `resource/CSI_DATASETS/`（WiFi CSI 场景）或通过 `data_root` 指定路径
- 自监督模式仅支持 NTU-Fi_HAR 数据集，`num_classes` 声明 14
- 可复现场景：固定 `trainer.seed`，严格复现时启用 `trainer.deterministic: true`
- **查询注册表前必须显式激活**：调用 `sf.activate_lazy_scenes()` 后再查询模型/数据集（CQS 合规，查询不触发注册副作用）

## 核心工作流

**声明式路径** — 标准场景：
```
用户意图 → 生成/校验 YAML → [可选 dry-run 预检] → run_experiment → 产物溯源 → 后处理交付
```

**命令式路径** — 自定义流程：
```
activate_lazy_scenes → 数据画像 → 注册策略 → 组装 Pipeline / 注入工厂 / load_extension
→ 执行训练 → 结构化反馈 → release_resources → 探索状态更新 → 产物溯源 → 后处理交付
```

### 步骤 1：识别意图与路径

- **声明式**：标准场景，直接用 YAML + `run_experiment`
- **命令式**：自定义策略或流程，用 DataProfiler + register_* + Pipeline + load_extension

训练类型：
- 监督训练：`learning_mode: supervised`
- 自监督训练：`learning_mode: self_supervised`（仅 NTU-Fi_HAR）
- HPO 超参搜索：`hpo.enabled: true`

### 步骤 2：激活场景与数据画像

**入口点契约**（RFC-004 方案 B）：查询注册表前必须显式激活延迟场景。CLI 脚本（generate_config / validate_config / cli.py）已内置调用，命令式路径需手动调用。

```python
import senseframe as sf

# 必须先激活，再查询（CQS 合规：查询不触发注册副作用）
sf.activate_lazy_scenes()
sf.list_models()  # 此时才返回完整模型列表

# 数据画像（命令式路径推荐先做）
profiler = sf.DataProfiler(max_samples=500)
profile = profiler.profile_bundle(bundle, dataset_name="my_data")

print(f"推荐 task_type: {profile.recommended_task_type}")
print(f"推荐 loss: {profile.recommended_loss}")
print(f"推荐 metric: {profile.recommended_metrics}")
print(f"推荐 normalization: {profile.recommended_normalization}")

# 缓存与复用
profile.save("data_profile.json")
loaded = sf.DataProfile.load("data_profile.json")
```

画像字段：`n_samples`, `input_shape`, `n_features`, `n_classes`, `class_distribution`, `missing_rate`, `value_range`, `mean`, `std`, `is_spatial`, `is_temporal`, `modality`, `recommended_task_type`, `recommended_loss`, `recommended_metrics`, `recommended_normalization`, `dataset_name`, `dtypes`, `feature_names`, `nullable`, `shapes`, `profile_source`

### 步骤 3：注册自定义策略

内置策略不满足时，注册自定义策略而非修改框架源码：

```python
import senseframe as sf

sf.register_task_type("anomaly_detection", default_loss="bce_with_logits",
                      default_metrics=["accuracy"], description="异常检测")

@sf.register_loss("my_focal")
def _focal(alpha=0.25, gamma=2.0, **kw):
    import torch.nn as nn
    return nn.CrossEntropyLoss(**kw)

sf.list_task_types()
sf.has_task_type("anomaly_detection")
sf.get_task_type_default_loss("anomaly_detection")
sf.list_losses()
```

注册 API 一览：

| API | 说明 | 默认 overwrite |
|-----|------|---------------|
| `register_task_type(name, default_loss, default_metrics, ...)` | 注册任务类型 | True |
| `register_loss(name)` | 注册损失函数，装饰器 | True |
| `register_metric(name)` | 注册指标，装饰器 | True |
| `register_model(name, spec)` | 注册模型 | False |
| `register_dataset(name, spec)` | 注册数据集 | False |
| `register_normalization(name, strategy)` | 注册归一化策略 | False |
| `register_scene(name, container)` | 注册场景容器 | False |

查询：`list_task_types()` / `has_task_type()` / `get_task_type_default_loss()` / `list_losses()` / `has_loss()` / `get_loss()` 等。

### 步骤 4：组装训练流程

**方式 A：声明式 — YAML 配置**

```bash
python scripts/generate_config.py \
  --dataset UT_HAR_data --model ResNet18 --mode supervised --output configs/exp.yaml
python scripts/validate_config.py --config configs/exp.yaml
python -m senseframe.cli experiment --config configs/exp.yaml
```

参考：[配置 Schema](./reference/config_schema.md) | [配置模板](./reference/training_templates.md) | [数据集与模型](./reference/datasets_and_models.md)

**方式 B：命令式 — Stage Pipeline**

```python
import senseframe as sf

pipeline = sf.Pipeline.default()
# 8 个 stage：validate → preflight → resolve → load → build → train → eval → export

pipeline.replace_stage("eval", my_custom_eval)
pipeline.before("train", data_check_hook)
pipeline.after("train", log_metrics_hook)
pipeline.skip("export")

ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)
```

| API | 说明 |
|-----|------|
| `Pipeline.default()` | 默认 8 stage pipeline |
| `pipeline.replace_stage(name, fn)` | 替换 stage |
| `pipeline.before(name, hook)` | stage 前插入 hook |
| `pipeline.after(name, hook)` | stage 后插入 hook |
| `pipeline.skip(name)` | 跳过 stage |
| `pipeline.run(ctx)` | 执行 pipeline |
| `PipelineContext(config=...)` | stage 间共享上下文 |

**方式 C：命令式 — 工厂注入**

```python
from senseframe.engine.config import ExperimentConfig

config = ExperimentConfig(
    scene=...,
    input_features=...,
    output_features=...,
    module_factory=lambda model, **kw: MyLightningModule(model, **kw),
    datamodule_factory=lambda train_ds, test_ds, **kw: MyDataModule(train_ds, test_ds, **kw),
    trainer_factory=lambda **kw: pl.Trainer(**kw),
    extra_callbacks=[MyCallback()],
)
output = sf.run_experiment(config)
```

**方式 D：命令式 — load_extension**

```python
import senseframe as sf

# 扩展文件中可直接调用 register_* API，无需 import senseframe
mod = sf.load_extension("my_extension.py")
```

### 步骤 5：探测资源与选择模型

```bash
python -m senseframe.cli probe
python -m senseframe.cli recommend --dataset UT_HAR_data --priority balanced
```

参考：[资源路由与模型推荐](./reference/resource_routing.md)

### 步骤 6：预检模式

不实际训练，仅执行启动前检查：

```bash
python -m senseframe.cli experiment --config configs/exp.yaml --dry-run
```

检查项：配置校验、场景注册、数据集/模型/学习模式支持、资源探测、数据存在性、**文件扩展名声明一致性**（基于 DatasetSpec.file_format + layout 递归 glob，扩展名不匹配时 dry-run 阶段即报错）、显存、磁盘。

### 步骤 7：执行训练

```bash
# 基础训练
python -m senseframe.cli experiment --config configs/exp.yaml

# 启用自愈重试（OOM 自动降 batch_size）
python -m senseframe.cli experiment --config configs/exp.yaml --retry

# 训练后导出
python -m senseframe.cli experiment --config configs/exp.yaml \
    --export-formats onnx,torchscript,state_dict
```

### 步骤 8：产物溯源（RFC-004 方案 G）

每次训练自动生成 `artifact_manifest.json`，记录全部产物（model/metadata/config/metrics/feedback）的 SHA-256。Agent 可校验产物完整性：

```python
from senseframe import load_manifest, verify_artifacts

# 加载产物清单
manifest = load_manifest("runs/<exp>/artifact_manifest.json")
for name, desc in manifest.artifacts.items():
    print(f"{name}: {desc.path} sha256={desc.sha256[:16]}...")

# 校验产物完整性（检测缺失/篡改）— 返回 {产物名: hash 是否匹配}
result = verify_artifacts("runs/<exp>/")
tampered = [name for name, ok in result.items() if not ok]
if tampered:
    print(f"产物校验失败（缺失/篡改）: {tampered}")
```

### 步骤 9：释放资源（RFC-005）

长任务 / HPO / 串行多试验后，主动释放训练资源避免泄露（线程/句柄/pipe/GPU 显存）：

```python
ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)
ctx.release_resources()  # 主动清理 Trainer/DataLoader/Logger/GPU 显存
```

清理顺序：log_writer close → Logger finalize → Trainer `_teardown` → DataModule teardown → model `.cpu()` → 置 None → CUDA empty_cache → gc.collect。

注：`GenericDataModule` 已内置 persistent_workers pipe 泄露修复（模块级 patch `_shutdown_workers` 关闭 worker 进程 pipe）。HPO 路径自动调用 `release_resources`，命令式路径需手动调用。

### 步骤 10：后处理与导出

```bash
# 后处理：拷贝模型 + 生成推理脚本（所有产物均在 output_dir 内）
python scripts/postprocess.py --output-dir runs/<实验目录>

# 模型导出
python -m senseframe.cli export \
    --metadata runs/<实验目录>/metadata.json \
    --checkpoint runs/<实验目录>/model.pth \
    --formats onnx,torchscript,state_dict \
    --output-dir exports/
```

注：`postprocess.py` 仅接受 `--output-dir` 参数（P0-1.5 路径安全修复后，所有后处理产物均在 output_dir 内，manifest 存相对路径）。`metadata.config` 自动包含 `data_root`（stage_export 显式补录），推理脚本生成不依赖外部路径输入。

格式：`state_dict` / `torchscript` / `onnx` / `quantized_onnx`

## 输出格式

`TrainOutput` 关键字段（P5 dataclass 化后，training/env_snapshot/feedback 为类型化 dataclass）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `str` | `"success"` / `"error"` |
| `final_eval` | `Dict[str, Any]` | 最终验证指标 |
| `training` | `Optional[TrainingSummary]` | 训练摘要 dataclass（`.epochs_trained`/`.early_stopped`/`.duration_s`/`.best_val_loss`/`.log`/`.intermediate_values`） |
| `env_snapshot` | `Optional[EnvSnapshot]` | 环境快照 dataclass（`.torch`/`.pytorch_lightning`/`.cuda`/`.python`/`.deterministic`/`.seed`） |
| `feedback` | `Optional[FeedbackResult]` | 训练反馈 dataclass（`.status` 6 值枚举 / `.diagnosis` / `.suggestions` / `.test_metrics`） |
| `model_path` | `Optional[str]` | 最佳 checkpoint 路径 |
| `output_dir` | `Optional[str]` | 输出目录 |
| `error_code` | `Optional[str]` | 结构化错误码（Agent 可程序化分支） |
| `error` / `error_traceback` | `Optional[str]` | 失败时的错误信息 |

**序列化**：`TrainOutput.to_dict()` 内置多态序列化 helper，自动调用 dataclass 的 `to_dict()`；`summary()` 兼容 dict 和 dataclass 两种形态。

**类型校验**：构造点调用 `validate_feedback`/`validate_training_summary`/`validate_env_snapshot` 捕获类型污染（如 status 非法枚举值、epochs_trained 误传 str）。

```python
output = sf.run_experiment(config)
if output.status == "success":
    # 属性访问（P5 dataclass 化后推荐）
    print(f"epochs: {output.training.epochs_trained}")
    print(f"feedback: {output.feedback.status}")
    # to_dict() 序列化（跨进程/JSON 场景）
    import json
    print(json.dumps(output.to_dict(), indent=2))
```

## 错误处理（RFC-003 结构化异常）

所有训练错误抛出 `SenseFrameError` 子类，每个携带 `error_code` 类属性。基于 `error_code` 做程序化决策，而非字符串匹配。

| error_code | 异常类 | 建议动作 |
|------------|--------|---------|
| `CONFIG_VALIDATION_ERROR` | `ConfigValidationError` | 修正配置，不重试 |
| `SCENE_NOT_FOUND` | `SceneNotRegisteredError` | 检查 scene.name |
| `DATASET_NOT_SUPPORTED` | `DatasetNotSupportedError` | 检查 dataset 名 |
| `MODEL_NOT_SUPPORTED` | `ModelNotSupportedError` | 检查 model_id |
| `DATA_NOT_FOUND` | `DataNotFoundError` | 检查 data_root，不重试 |
| `DATA_LOAD_ERROR` | `DataCorruptedError` | 检查数据完整性/格式/权限（JSON/pickle 损坏、PermissionError） |
| `OOM_ERROR` | `OOMError` | 降 batch_size 重试 |
| `CHECKPOINT_ERROR` | `CheckpointError` | 检查 checkpoint 路径/版本兼容/完整性 |
| `PREFLIGHT_ERROR` | `PreflightError` | 升级硬件或换小模型 |
| `TRAINING_ERROR` | `TrainingError` | 查看 traceback |
| `MODEL_BUILD_ERROR` | `ModelBuildError` | 检查 model_id |
| `SAVE_ERROR` | `SaveError` | 检查磁盘空间/权限 |
| `UNKNOWN_ERROR` | `SenseFrameError`（基类） | 兜底分类，查看 traceback |

```python
from senseframe.engine.runner.errors import SenseFrameError, OOMError

try:
    output = sf.run_experiment(config)
except OOMError as e:
    # e.error_code == "OOM_ERROR"
    config.trainer.batch_size //= 2
    output = sf.run_experiment(config)
except SenseFrameError as e:
    print(f"[{e.error_code}] {e}")
```

参考：[错误排查指南](./reference/troubleshooting.md)

## 自省协议

查询 pipeline / context / 数据契约，无需读源码。组装 pipeline 前优先用自省 API 查询字段契约。

```python
import senseframe as sf

# PipelineContext 字段契约：每个字段首次填充的 stage
ctx = sf.PipelineContext(config=my_config)
ctx.schema()           # 字段 schema_version / type / fill_stage / has_default
ctx.filled_at("stage_load")  # ["bundle", "data_profile", "output_dir", "log_writer"]
ctx.completed_fields() # 当前已填充字段
ctx.describe()         # 运行时状态

# DatasetBundle 填充契约（按 learning_mode 校验 required/forbidden）
bundle = scene.load_dataset(...)
bundle.filling_rule("self_supervised")  # {"train": "forbidden", "unsupervised": "required", ...}
bundle.validate_filling("supervised")   # 返回错误列表，空列表表示通过

# Stage IO 声明（reads / writes）
pipeline = sf.Pipeline.default()
pipeline.stages_with_spec()  # List[StageSpec]，每项含 name / reads / writes / description

# 顶层自省 API
sf.context_schema()       # PipelineContext.schema() 的便捷入口
sf.stage_io("train") # 查询单个 stage 的 reads / writes（stage 名无 stage_ 前缀）
sf.list_stages()          # 列出全部 stage 名
sf.pipeline_graph()       # 返回 DAG 描述
sf.data_bundle_schema()   # DatasetBundle.schema()
sf.data_profile_schema()  # DataProfile 字段契约
```

## 探索状态与技能库

闭合探索-反馈回路：训练结果 → 结构化反馈 → 探索历史 → 推荐下一步。每次试验后记录 `record_trial`，成功策略 `save_skill` 落库复用。

```python
import senseframe as sf

# 探索状态（PipelineContext 内置）
ctx = sf.PipelineContext(config=my_config)
ctx.trial_id = "trial_001"
ctx.parent_trial_id = "trial_000"  # 支持回溯
ctx.record_trial(strategy={"loss": "focal", "lr": 0.001}, result={"val_accuracy": 0.85})

# stage_eval 自动产出结构化反馈（first-class 字段，非 ctx.extra）
feedback = ctx.feedback
# {"status": "overfitting"/"underfitting"/"converged"/"numerical_instability"/"success",
#  "diagnosis": "...", "suggestions": [...]}

# 持久化探索历史
from senseframe.exploration import ExplorationTracker
tracker = ExplorationTracker(ctx.exploration_history)
tracker.save(output_dir / "exploration.json")

# 技能库：将成功策略保存为可复用 Skill
sf.save_skill(name="wifi_csi_focal", description="WiFi CSI + Focal Loss",
              code="register_loss('focal')(...)", tags=["wifi_csi", "focal"])
skill = sf.load_skill("wifi_csi_focal")
matches = sf.search_skills(query="WiFi CSI")
all_skills = sf.list_skills()
```

| API | 说明 |
|-----|------|
| `PipelineContext.record_trial(strategy, result)` | 记录一次探索试验 |
| `PipelineContext.exploration_history` | 探索历史列表 |
| `PipelineContext.feedback` | 结构化反馈（first-class 字段，RFC-004） |
| `ExplorationTracker(history)` | 探索状态管理器，支持 save / recommend_next |
| `save_skill(name, description, code, tags, source_path)` | 保存技能 |
| `load_skill(name, version=None)` | 加载技能（None = 最新版） |
| `search_skills(query)` | 全文检索技能 |
| `list_skills()` | 列出全部技能名（返回 `List[str]`） |

## 断点续跑

Pipeline 支持 stage 级断点续跑，失败 stage 后可从断点恢复，无需重跑已完成 stage。

```python
import senseframe as sf
from senseframe.engine.runner.pipeline import Pipeline

# 首次运行（失败于 train stage）
pipeline = sf.Pipeline.default()
ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)
# 失败时输出：Pipeline failed at stage 'train'. To resume: Pipeline.resume('<output_dir>')

# 续跑：从失败 stage 恢复
pipeline, completed = Pipeline.resume(ctx.output_dir)
ctx2 = sf.PipelineContext(config=my_config)
ctx2.completed_stages = completed
ctx2.stage_checkpoint_path = ctx.output_dir / "pipeline_checkpoint.json"
result = pipeline.run(ctx2)
```

## RFC-003 SP：搜索协议（Ask-Tell 标准化接口）

所有搜索驱动能力（HPO / NAS / AutoAugment / ε6 Method / 元学习）均通过 SP ask/tell 接口驱动。对齐 Optuna Study/Trial/Sampler 范式 + Google Vizier RPC 思想。

```python
import senseframe as sf
from senseframe.search_protocol import (
    ParameterSpec, SearchSpace, StudyManager, get_study_manager,
    RandomSampler, GridSampler, register_sampler, list_samplers,
    register_pruner, list_pruners, Pruner,
)

# 1. 构造搜索空间
space = SearchSpace(parameters=[
    ParameterSpec(name="lr", type="float", low=1e-4, high=1e-2, log=True),
    ParameterSpec(name="batch_size", type="int", low=8, high=64, step=8),
    ParameterSpec(name="optimizer", type="categorical",
                  choices=["adam", "sgd", "adamw"]),
])

# 2. 创建 Study（可选指定 sampler / pruner / warm_start_from / history_store）
sm = StudyManager()  # 或 get_study_manager() 全局单例
study_id = sm.create_study(
    name="hpo_run", direction="maximize",
    search_space=space, sampler="random",
)

# 3. Ask-Tell 循环
for _ in range(n_trials):
    trial = sm.ask(study_id)               # SP 采样参数
    config = apply_params(base_config, trial.params)
    output = sf.run_experiment(config)
    val_acc = output.final_eval.get("val_accuracy", 0.0)
    sm.tell(trial.trial_id, value=val_acc)  # SP 记录结果

# 4. 查询最优
best = sm.best_trial(study_id)
trials = sm.list_trials(study_id)
```

| API | 说明 |
|-----|------|
| `StudyManager()` / `get_study_manager()` | Study 管理器（实例 / 全局单例） |
| `sm.create_study(name, direction, search_space, sampler, pruner?, warm_start_from?, history_store?)` | 创建 Study |
| `sm.ask(study_id)` | 采样一次 trial（返回 `TrialSpec`） |
| `sm.tell(trial_id, value, intermediate_values?, state?, feedback?)` | 报告 trial 结果 |
| `sm.best_trial(study_id)` | 查询最优 trial |
| `sm.list_trials(study_id)` | 列出全部 trial |
| `register_sampler(name, cls)` / `list_samplers()` | Sampler 注册表 |
| `register_pruner(name, cls)` / `list_pruners()` | Pruner 注册表 |

内置 Sampler（`senseframe.search_protocol` 注册表）：
- 默认可用：`random` / `grid` / `tpe`（random_fallback）/ `asha` / `hyperband`
- 显式 import 子模块后可用：`darts`（`from senseframe.nas import DARTSSampler`）/ `enas` / `evolutionary`（`from senseframe.nas.sampler import ENASSampler, EvolutionarySampler`）/ `autoaugment`（`from senseframe.autoaugment import AutoAugmentSampler`）

内置 Pruner：`asha` / `hyperband`（同时实现 Sampler + Pruner Protocol，可作 sampler 创建即获得早停能力）

## RFC-003 ε1：损失函数搜索

通过 SP ask/tell 驱动损失函数组合搜索（loss + label_smoothing），验证协议栈可行性。

```python
import senseframe as sf
from senseframe.automl import build_loss_search_space, run_loss_search, LossSearchResult

space = build_loss_search_space(include_label_smoothing=True)
result = run_loss_search(config, n_trials=10, direction="maximize",
                         metric="val_accuracy", sampler="random")
print(f"best: {result.best_params} -> {result.best_value}")
```

## RFC-003 ε6：对比实验（Method / Baseline / Experiment）

ε6 是 RFC-003 协议栈的"应用落地形态"：Method 组走 SP 驱动搜索，Baseline 组用固定参数作基准，ExperimentRunner 编排聚合 + 输出 ComparisonReport（DSP 合规）。

```python
import senseframe as sf
from senseframe.experiment import (
    MethodConfig, BaselineConfig, ExperimentDesign, ExperimentBudget,
    MethodRunner, BaselineRunner, ExperimentRunner, ComparisonReport,
    TrialGroup, TrialStatus,
)

# 1. Method 组：SP 驱动搜索
method_config = MethodConfig(
    name="hpo_method", base_config=base_config,
    search_space=space, metric="val_accuracy", direction="maximize",
)
sm = sf.StudyManager()
study_id = sm.create_study(name="exp", direction="maximize",
                           search_space=space, sampler="random")
method_runner = MethodRunner(config=method_config, study_id=study_id,
                             study_manager=sm, experiment_id="exp_001")

# 2. Baseline 组：固定参数（不走 SP）
baseline_config = BaselineConfig(
    name="paper_baseline", base_config=base_config,
    group=TrialGroup.BASELINE_PAPER, reported_metrics={"val_accuracy": 0.85},
)
baseline_runner = BaselineRunner(config=baseline_config, experiment_id="exp_001")

# 3. ExperimentRunner：编排聚合
design = ExperimentDesign(
    name="exp", datasets=["UT_HAR_data"], models=["ResNet18"],
    method=method_config, baselines=[baseline_config],
    budget=ExperimentBudget(max_trials_per_group=5, n_repeats=1),
)
exp_runner = ExperimentRunner(design=design, experiment_id="exp_001")
report = exp_runner.run()  # 自动驱动 Method + Baseline，聚合为 ComparisonReport
report.save("reports/exp_001.json")
```

| Runner | 驱动方式 | 用途 |
|--------|----------|------|
| `MethodRunner` | SP ask/tell | 搜索驱动试验（HPO / NAS / ε1 loss） |
| `BaselineRunner` | 固定参数 / 论文报告值 | 对比基准（BASELINE_REPRO / BASELINE_PAPER） |
| `ExperimentRunner` | 编排 Method + Baseline | 端到端对比实验，输出 ComparisonReport |

**P2.12 OP 迁移**：MethodRunner 支持 `use_op=True` 通过 Orchestrator 编排 run_pipeline（create_run + start + complete/fail + CloudEvent 发射），ExperimentRunner 可订阅 OP 事件实现事件驱动聚合。

## P3：神经架构搜索（NAS）

DARTS（可微 NAS）+ ENAS（权重共享）+ Evolutionary Sampler，全部注册到 SP Sampler 注册表。

```python
from senseframe.nas import DARTSSampler, DARTSPipelineRun, ENASSampler
from senseframe.nas.search_space import NASSearchSpace

# 方式 A：注册到 SP，用 ask/tell 驱动
sf.register_sampler("darts", DARTSSampler)  # 已内置注册

# 方式 B：直接运行 DARTSPipelineRun（含 supernet 训练 + arch 搜索）
# 修复（文档契约）：DARTSPipelineRun 构造参数是
#   sampler: DARTSSampler, builder: ArchitectureBuilder, search_space: SearchSpace,
#   input_shape: Tuple[int, ...], num_classes: int, n_epochs: int,
#   lr_w: float, lr_arch: float
# 旧文档用 supernet/train_loader/val_loader（不存在）+ lr_arch=3e-4（代码默认 1e-3）。
from senseframe.nas.darts import DARTSPipelineRun
run = DARTSPipelineRun(
    sampler=darts_sampler,         # DARTSSampler 实例（上面 register_sampler 的对象）
    builder=arch_builder,          # ArchitectureBuilder 实例
    search_space=nas_space,        # SP SearchSpace
    input_shape=(1, 250, 90),      # 模型输入形状（不含 batch 维）
    num_classes=7,
    n_epochs=50, lr_w=0.025, lr_arch=0.001,  # 代码默认值；可按需覆盖
)
result = run.run(train_loader, val_loader)
# result = {"best_arch": ..., "final_alpha": ..., "history": [...]}
```

**资源泄露修复（RFC-005）**：DARTSPipelineRun.run() 用 try/finally 显式释放 supernet / optimizer / iterator；DARTSSampler.update() 用 `.detach().clone()` 切断计算图引用 + `zero_grad(set_to_none=True)` 释放 grad tensor；`_InfiniteLoader` 实现 close() + context manager 关闭 DataLoader worker 进程。

## P3：AutoAugment 数据增强搜索

搜索最优数据增强策略组合（基于 SP ask/tell）。

```python
from senseframe.autoaugment import (
    AutoAugmentSampler, AutoAugmentPolicyBuilder, AugmentationSearchSpace,
    make_autoaugment_datamodule_factory,
)

# 1. 构造增强搜索空间
aug_space = AugmentationSearchSpace.build_default()

# 2. 创建 SP Study
sm = sf.StudyManager()
study_id = sm.create_study(name="autoaug", direction="maximize",
                           search_space=aug_space.to_sp_search_space(),
                           sampler="autoaugment")

# 3. Ask-Tell 循环（每次 trial 是一组增强策略）
trial = sm.ask(study_id)
policy = AutoAugmentPolicyBuilder().from_params(trial.params).build()
# 用 policy 训练，把 val_accuracy 回报给 sm.tell(...)
```

## P3：元学习 warm-start

用源数据集成功策略 warm-start 目标数据集 SP Study（迁移学习对搜索空间的偏向）。

```python
from senseframe.automl.meta_learner import MetaLearner

# 修复（文档契约）：MetaLearner 构造需要 study_manager + history_store 两个必填参数。
# 旧文档用 MetaLearner() 无参构造，与代码契约不符，会抛 TypeError。
learner = MetaLearner(
    study_manager=sm,         # StudyManager 实例（前面创建的）
    history_store=store,      # HistoryStore 实例（前面创建的）
)
# 从源数据集历史中提取成功策略
learner.warm_start(study_id=target_study_id,
                   source_dataset="UT_HAR_data",
                   success_threshold=0.7)

# 或在 create_study 时直接指定 warm_start_from
study_id = sm.create_study(
    name="target", direction="maximize", search_space=space,
    sampler="random",
    warm_start_from=source_history,  # List[Dict]：源数据集 trial 历史
)
```

**Sampler warm_start 契约**：`Sampler` Protocol 的 `warm_start(source_history)` 方法是可选的（`@runtime_checkable` 不强制实现）。`EvolutionarySampler` / `AutoAugmentSampler` 实现 warm_start 作为元学习受益示例（用源数据集成功策略作为初始 population 种子）。`RandomSampler` / `GridSampler` / `ASHASampler` / `HyperbandSampler` 的 warm_start 是 no-op（无状态采样器，不从 history 受益），保留方法仅为满足 Python 3.12+ `@runtime_checkable` Protocol 的 isinstance 检查。

## P3：Multi-fidelity 早停（ASHA / Hyperband）

```python
from senseframe.search_protocol import ASHASampler, HyperbandSampler

# ASHA 同时是 Sampler + Pruner（采样=随机，剪枝=Successive Halving）
pruner = ASHASampler(max_resource=81, eta=3, direction="maximize")

# 在 MethodRunner 中注入 pruner
runner = MethodRunner(config=method_config, study_id=study_id,
                      study_manager=sm, pruner=pruner, experiment_id="exp")
result = runner.run(dataset="UT_HAR_data", model_id="ResNet18", run_idx=0)
# 训练后 MethodRunner 自动检查 should_prune，True 则标记 trial 为 pruned
```

## P3：OP 编排（Orchestrator）

OP（Orchestration Protocol）提供 PipelineRun 状态机 + CloudEvent 发射 + 持久化 + 异步执行能力。

```python
from senseframe.orchestration import (
    Orchestrator, PipelineDef, PipelineRun, get_orchestrator,
    PHASE_PENDING, PHASE_RUNNING, PHASE_SUCCEEDED, PHASE_FAILED, PHASE_PAUSED,
    EVENT_PIPELINE_STARTED, EVENT_PIPELINE_SUCCEEDED, EVENT_PIPELINE_FAILED,
)

orch = Orchestrator()  # 或 get_orchestrator() 全局单例
try:
    pdef = PipelineDef(name="my_pipeline")
    pipeline_id = orch.create_pipeline(pdef)
    run_id = orch.create_run(pipeline_id, params={"lr": 0.001})

    # 同步路径：start → run_pipeline → complete/fail
    orch.start(run_id)  # → PHASE_RUNNING + emit EVENT_PIPELINE_STARTED
    try:
        output = sf.run_pipeline(config)
        orch.complete(run_id, output_uri=str(output.output_dir))
    except Exception as e:
        orch.fail(run_id, error=str(e))
        raise

    # 异步路径：start_and_execute → Future + wait_for_completion
    future = orch.start_and_execute(run_id, pipeline=my_pipeline)
    orch.wait_for_completion(run_id, timeout=600)
finally:
    orch.shutdown()  # 关闭 ThreadPoolExecutor + 置 None
```

| API | 说明 |
|-----|------|
| `Orchestrator()` / `get_orchestrator()` | 编排器（实例 / 全局单例） |
| `orch.create_pipeline(pdef)` | 创建 Pipeline 定义 |
| `orch.create_run(pipeline_id, params)` | 创建 PipelineRun（PHASE_PENDING） |
| `orch.start(run_id)` / `pause` / `resume` / `retry` / `stop` | 状态机转移 |
| `orch.complete(run_id, output_uri)` | 标记成功（PHASE_SUCCEEDED） |
| `orch.fail(run_id, error, stage_name)` | 标记失败（PHASE_FAILED） |
| `orch.reconcile(run_id, pipeline)` | 同步驱动 Pipeline 执行（含 stage 包装 + checkpoint + 事件） |
| `orch.start_and_execute(run_id, pipeline)` | 异步执行，返回 `Future` |
| `orch.wait_for_completion(run_id, timeout)` | 阻塞等待 run 完成 |
| `orch.subscribe(event_type, callback)` | 订阅 CloudEvent，返回 unsubscribe 函数 |
| `orch.list_runs()` / `orch.get_run(run_id)` | 查询 run |
| `orch.shutdown()` | 关闭 ThreadPoolExecutor（必调用，避免泄露） |

**P3.4 OP 持久化**：`Orchestrator(store=FileOrchestrationStore(...))` 把 PipelineRun + 事件持久化到磁盘，支持跨进程恢复（K8s Operator 适配）。`store=None` 时仅内存（默认）。

**事件订阅资源管理（RFC-005）**：`subscribe` 返回 unsubscribe 函数；ExperimentRunner 用 try/finally + `_unsubscribe_all()` 确保异常路径也释放订阅。`shutdown(wait=False)` 关闭 ThreadPool + 置 None，避免进程退出后线程残留。

## 自监督训练

两阶段训练：EntLoss 无监督预训练 + CrossEntropyLoss 监督微调。仅支持 NTU-Fi_HAR 数据集。

参考：[自监督训练范式](./reference/self_supervised_paradigm.md)

## 场景开发

继承 `SceneContainer`，实现 4 个抽象方法：

```python
from senseframe.scenes.base import SceneContainer, SceneMeta

class MyScene(SceneContainer):
    def meta(self) -> SceneMeta: ...
    def load_dataset(self, dataset_name, root, learning_mode="supervised") -> DatasetBundle: ...
    def build_model_for_dataset(self, model_id, dataset, num_classes, **kw) -> nn.Module: ...
    def get_dataset_info(self, dataset_name, **kw) -> dict: ...
    # 可选：get_transforms / get_search_space / get_model_info / get_feature_spec / get_task_spec
```

参考：[场景开发指南](./reference/scene_development.md)

## Agent 提示词（commands/）

`commands/` 目录下的 4 个 `.md` 文件是 **Agent 提示词**，不是 SenseFrame CLI 子命令。

| 文件 | slash 命令 | 场景 |
|------|-----------|------|
| `senseframe-train.md` | `/senseframe-train` | 简单：单模型监督训练 |
| `senseframe-hpo.md` | `/senseframe-hpo` | 中等：HPO 超参搜索 |
| `senseframe-full.md` | `/senseframe-full` | 困难：完整闭环 + 自监督 + 断点续跑 |
| `senseframe-auto.md` | `/senseframe-auto` | 多轮自主调优 |

**定位说明**：
- 这些 `.md` 文件部署到 `.opencode/.claude/.agents/commands/` 供 AI Agent CLI 工具（opencode/Claude Code）调用
- slash 命令（如 `/senseframe-train`）由 Agent CLI 工具解析，不是 SenseFrame CLI 子命令
- 文件内容是标准 Agent 提示词模板（Role/Objective/Context/Protocol），内部调用 SenseFrame CLI 子命令作为执行原语
- 部署逻辑见 `tests/pack_code.py` 的 `deploy_commands()` 函数

## CLI 命令

所有命令输出结构化 JSON。

| 命令 | 说明 | 示例 |
|------|------|------|
| `probe` | 探测硬件资源 | `python -m senseframe.cli probe` |
| `list-models` | 列出可用模型 | `python -m senseframe.cli list-models --dataset UT_HAR_data` |
| `list-datasets` | 列出可用数据集 | `python -m senseframe.cli list-datasets` |
| `list-scenes` | 列出场景容器 | `python -m senseframe.cli list-scenes` |
| `paradigms` | 列出 SOTA 范式 | `python -m senseframe.cli paradigms --category cnn` |
| `recommend` | 根据资源推荐模型 | `python -m senseframe.cli recommend --dataset UT_HAR_data` |
| `experiment` | YAML 配置驱动训练 | `python -m senseframe.cli experiment --config configs/exp.yaml` |
| `export` | 多格式模型导出 | `python -m senseframe.cli export --formats onnx` |

## 指导原则

- **入口点激活**：查询注册表前调 `activate_lazy_scenes()`，CQS 合规
- **路径选择**：标准场景用声明式，自定义流程用命令式
- **数据先行**：先用 DataProfiler 画像，基于数据特征选策略，不要盲猜
- **开放策略空间**：内置策略不满足时用 register_* 注册，不要修改框架源码
- **配置校验**：始终先 `validate_config.py` 再训练
- **预检优先**：长任务前用 `--dry-run` 验证可行性
- **资源适配**：CPU 选小模型，GPU 启用 mixed_precision
- **自愈重试**：无人值守训练启用 `--retry`
- **错误码驱动**：基于 `error_code` 做程序化决策，捕获 `SenseFrameError` 子类
- **资源释放**：长任务/HPO 后调 `release_resources()`，避免泄露
- **产物溯源**：训练后用 `verify_artifacts(output_dir)` 校验产物完整性（接受目录路径，返回 `{产物名: bool}`）
- **内省优先**：组装 pipeline 前用 `sf.context_schema()` / `sf.stage_io()` 查询字段契约，避免读源码
- **探索闭环**：每次试验后记录 `record_trial`（或 `add_trial`），成功策略 `save_skill` 落库复用
- **断点续跑**：长 pipeline 失败后用 `Pipeline.resume(output_dir)` 从失败 stage 恢复
- **SP 驱动搜索**：HPO/NAS/AutoAugment/ε6 Method 统一走 SP ask/tell，不要绕过协议栈
- **OP 编排必释放**：`Orchestrator` 用完必调 `shutdown()`，避免 ThreadPool 泄露；`subscribe` 返回的 unsubscribe 函数用 try/finally 确保调用
- **stage 名不带前缀**：调用 `sf.stage_io()` / `pipeline.check_readiness()` 时用 `"train"`/`"eval"`（无 `stage_` 前缀）；`ctx.filled_at()` 用带前缀名 `"stage_load"`
- **NAS 资源管理**：DARTSPipelineRun 用 try/finally 释放 supernet/optimizer/iterator；DARTSSampler 用 `.detach().clone()` 切断计算图引用
- **类型安全（P5）**：`TrainOutput.training/env_snapshot/feedback` 是 dataclass 实例，用属性访问（`.epochs_trained`/`.status`）；序列化用 `.to_dict()`；构造点有 `validate_*` 校验捕获类型污染
- **SceneParams 正交化（P5）**：`SceneConfig.params` 是 `Optional[SceneParams]`，提供 dict-like 兼容层（`[]/= /in/.get()/.items()`）；HPO 赋值前检查 `is None` 创建空实例
- **接口契约优先于消费者数量（P5）**：即使字段当前无消费者也定义清楚（SceneParams 10 个标准字段），为未来场景扩展铺路

## 参考资源

按需加载，不要一次性全部加载：

| 文档 | 用途 | 加载时机 |
|------|------|----------|
| [config_schema.md](./reference/config_schema.md) | 配置完整字段与校验规则 | 编写/校验 YAML 时 |
| [datasets_and_models.md](./reference/datasets_and_models.md) | 数据集与模型支持表 | 选择数据集/模型时 |
| [training_templates.md](./reference/training_templates.md) | YAML 配置模板 | 生成配置时 |
| [resource_routing.md](./reference/resource_routing.md) | 五级路由表 + 模型推荐 + 预检模式 | 资源探测/模型选择/预检时 |
| [self_supervised_paradigm.md](./reference/self_supervised_paradigm.md) | 自监督训练详解 | 自监督模式时 |
| [scene_development.md](./reference/scene_development.md) | 场景开发指南 | 新增场景时 |
| [troubleshooting.md](./reference/troubleshooting.md) | 错误排查指南 + error_code 枚举 | 训练失败时 |
| [introspect.md](./reference/introspect.md) | 自省协议 + 探索状态 + 技能库 + 断点续跑 API | 查询字段契约/管理探索/续跑时 |
