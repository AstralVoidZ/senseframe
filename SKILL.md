---
name: "senseframe"
description: "AI Agent 驱动的 AutoML 训练框架。Invoke when user wants to train/evaluate/export models, register custom strategies (loss/metric/task_type/model/scene/normalization), profile data, compose stage pipelines, inject factories, load extensions, run HPO, use self-supervised paradigm, introspect pipeline contracts, verify artifact integrity, release training resources, handle structured SenseFrameError exceptions via error_code, activate lazy scenes, resume failed pipelines, or manage exploration history and skills."
---

# SenseFrame

Agent 驱动的 AutoML 训练框架。提供可编程原语库 + 执行底座 + 安全护栏，Agent 持有训练流程控制权。

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

## 前置条件

- Python 3.x + PyTorch + PyTorch Lightning 已安装
- 数据集放置于 `CSI_DATASETS/`（WiFi CSI 场景）或通过 `data_root` 指定路径
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

画像字段：`n_samples`, `input_shape`, `n_classes`, `class_distribution`, `missing_rate`, `value_range`, `mean`, `std`, `is_spatial`, `is_temporal`, `modality`, `recommended_*`

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
| `register_model(name, spec)` | 注册模型 | True |
| `register_dataset(name, spec)` | 注册数据集 | True |
| `register_normalization(name, strategy)` | 注册归一化策略 | True |
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

检查项：配置校验、场景注册、数据集/模型/学习模式支持、资源探测、数据存在性、显存、磁盘。

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

# 校验产物完整性（检测缺失/篡改）
report = verify_artifacts("runs/<exp>/")
if report.dangling_refs:
    print(f"缺失产物: {report.dangling_refs}")
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
# 后处理：拷贝模型 + 生成推理脚本
python scripts/postprocess.py --output-dir runs/<实验目录> \
  --models-dir models --result-dir result

# 模型导出
python -m senseframe.cli export \
    --metadata runs/<实验目录>/metadata.json \
    --checkpoint runs/<实验目录>/model.pth \
    --formats onnx,torchscript,state_dict \
    --output-dir exports/
```

格式：`state_dict` / `torchscript` / `onnx` / `quantized_onnx`

## 输出格式

`TrainOutput` 关键字段：

| 字段 | 说明 |
|------|------|
| `status` | `"success"` / `"error"` |
| `final_eval` | 最终验证指标 |
| `training.epochs_trained` | 实际训练轮数 |
| `model_path` | 最佳 checkpoint 路径 |
| `output_dir` | 输出目录 |
| `error_code` | 结构化错误码（Agent 可程序化分支） |
| `error` / `error_traceback` | 失败时的错误信息 |

## 错误处理（RFC-003 结构化异常）

所有训练错误抛出 `SenseFrameError` 子类，每个携带 `error_code` 类属性。基于 `error_code` 做程序化决策，而非字符串匹配。

| error_code | 异常类 | 建议动作 |
|------------|--------|---------|
| `CONFIG_VALIDATION_ERROR` | `ConfigValidationError` | 修正配置，不重试 |
| `SCENE_NOT_FOUND` | `SceneNotRegisteredError` | 检查 scene.name |
| `DATASET_NOT_SUPPORTED` | `DatasetNotSupportedError` | 检查 dataset 名 |
| `MODEL_NOT_SUPPORTED` | `ModelNotSupportedError` | 检查 model_id |
| `DATA_NOT_FOUND` | `DataNotFoundError` | 检查 data_root，不重试 |
| `OOM_ERROR` | `OOMError` | 降 batch_size 重试 |
| `PREFLIGHT_ERROR` | `PreflightError` | 升级硬件或换小模型 |
| `TRAINING_ERROR` | `TrainingError` | 查看 traceback |
| `MODEL_BUILD_ERROR` | `ModelBuildError` | 检查 model_id |
| `SAVE_ERROR` | `SaveError` | 检查磁盘空间/权限 |

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
sf.stage_io("stage_train") # 查询单个 stage 的 reads / writes
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
              code="register_loss('focal')(...)", version="1.0.0")
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
| `save_skill(name, description, code, version)` | 保存技能 |
| `load_skill(name, version=None)` | 加载技能（None = 最新版） |
| `search_skills(query)` | 全文检索技能 |
| `list_skills()` | 列出全部技能 |

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
- **产物溯源**：训练后用 `verify_artifacts()` 校验产物完整性
- **内省优先**：组装 pipeline 前用 `sf.context_schema()` / `sf.stage_io()` 查询字段契约，避免读源码
- **探索闭环**：每次试验后记录 `record_trial`，成功策略 `save_skill` 落库复用
- **断点续跑**：长 pipeline 失败后用 `Pipeline.resume(output_dir)` 从失败 stage 恢复

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
