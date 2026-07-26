# SenseFrame

> AI Agent 驱动的 AutoML 训练框架 — 可编程原语库 + 执行底座 + 安全护栏

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SenseFrame 让 Agent 持有训练流程控制权：框架提供可组合的 Stage Pipeline、开放策略注册表、数据驱动的策略推荐，Agent 决定如何编排训练，而非框架替 Agent 决策。

> **声明**：个人实验性项目，用于学习训练框架设计，不保证生产可用性。

## 特性

- **开放策略空间** — `task_type` / `loss` / `metric` / `model` / `scene` / `normalization` 均可运行时注册
- **数据驱动** — `DataProfiler` 探查数据特征并推荐策略（task_type / loss / metric / normalization）
- **可编程训练流程** — 9 个 Stage 可替换、插入 hook、跳过、断点续跑
- **声明式 + 命令式** — YAML 快速启动，代码注入自定义逻辑
- **资源感知路由** — 自动探测 CPU/GPU/内存，7 级路由选择训练路线
- **自愈重试** — OOM 自动降 `batch_size` 重试
- **结构化异常** — `SenseFrameError` 基类 + 20 个错误码，每个携带 `error_code`，消除字符串匹配
- **产物溯源** — `ArtifactManifest` 记录全部产物 SHA-256，`verify_artifacts()` 校验完整性
- **资源安全** — `release_resources()` 主动清理 Trainer / DataLoader / Logger / GPU 显存
- **自省协议** — `context_schema()` / `stage_io()` / `pipeline_graph()` 查询字段契约，无需读源码
- **探索闭环** — `record_trial` 记录历史，`save_skill` 策略复用，`Pipeline.resume` 断点续跑
- **搜索协议（SP）** — HPO / NAS / AutoAugment / 元学习 / ε6 对比实验统一走 Ask-Tell 接口
- **自监督学习** — 两阶段训练（AutoFi 风格 EntLoss 预训练 + 监督微调）
- **多格式导出** — ONNX / TorchScript / state_dict / 量化 ONNX
- **推理服务** — KServe v2 兼容 HTTP API + OTel 指标
- **MCP 服务器** — 通过 Model Context Protocol 与 Agent 交互，提供 10+ 工具和 12 个自省资源
- **NAS** — 内置 DARTS/ENAS/Evolutionary 架构搜索 + 真实可微超网
- **AutoAugment** — 进化搜索数据增强策略，5种增强原语
- **AutoML/PEFT** — LoRA/Adapter/PrefixTuning/PromptTuning 参数高效微调 + SP 驱动搜索
- **Orchestration** — CloudEvents 1.0 事件流 + K8s Operator CR 适配

## 快速开始

### 安装

```bash
git clone <repo-url>
cd SenseFrame
pip install -e '.[eeg,radio,dev]'   # 核心依赖 + EEG/Radio/开发工具

# 可选 extras（按需安装）
pip install -e '.[onnx]'   # ONNX 导出
pip install -e '.[otel]'   # OpenTelemetry 可观测性
pip install -e '.[all]'    # 全部可选依赖
```

### 声明式训练

```bash
# 探测资源
python -m senseframe.cli probe

# 预检（不实际训练）
python -m senseframe.cli experiment --config configs/config.yaml --dry-run

# 训练
python -m senseframe.cli experiment --config configs/config.yaml
```

```python
from senseframe import run_experiment
from senseframe.engine.config import ExperimentConfig

config = ExperimentConfig.from_dict(yaml_dict)
config.validate()
output = run_experiment(config)

if output.status == "success":
    print(f"macro_f1: {output.final_eval['val_macro_f1']:.4f}")
```

### 命令式编排

```python
import senseframe as sf

# 激活场景（查询注册表前必调）
sf.activate_lazy_scenes()

# 数据画像 → 策略推荐
profile = sf.DataProfiler().profile_bundle(bundle, dataset_name="my_data")
print(f"推荐: task_type={profile.recommended_task_type}, loss={profile.recommended_loss}")

# 注册自定义策略
sf.register_task_type("anomaly_detection", default_loss="bce_with_logits",
                      default_metrics=["accuracy"], description="异常检测")

# 自定义 Pipeline
pipeline = sf.Pipeline.default()
pipeline.replace_stage("eval", my_custom_eval)
pipeline.before("train", log_data_profile_hook)
pipeline.skip("export")

ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)

# 加载扩展代码
sf.load_extension("my_extension.py")
```

## 架构

SenseFrame 采用 **通用训练框架 + 场景包** 分层架构：

- **通用训练框架** — 基于 PyTorch Lightning，通过 `SceneContainer` 接口与领域逻辑解耦
- **场景包** — WiFi CSI（4 数据集 / 11 模型）/ Detection / Generic / Custom / 模板

两条执行路径归一到 `Pipeline.run()`（单一真相源）：

- **声明式** — YAML → `run_experiment(config)` → 内部委托 `run_pipeline`
- **命令式** — `DataProfiler` + `register_*` + `Pipeline` + `load_extension`

## 项目结构

```
senseframe/
├── autoaugment/         # AutoAugment 数据增强搜索
├── automl/              # AutoML（loss_search/meta_learner/peft_builder/peft_search）
├── common/              # 通用工具（checkpoint/path_safe/paths/runtime_state/transforms）
├── core/                # 核心抽象（foundation_model/losses/metrics/task/validators）
├── engine/              # 训练引擎（config/datamodule/module/self_supervised/metadata/hpo）
│   ├── callbacks/       # Lightning Callbacks（psnr_early_stopping）
│   └── runner/          # Pipeline 运行器（orchestrator/pipeline stages/artifacts）
├── experiment/          # ε6 对比实验（baseline/method/report/runner）
├── mcp/                 # MCP 服务器（tools/views/resources/orchestration/pagination）
├── nas/                 # 神经架构搜索（darts/sampler/supernet/builder）
├── scenes/              # 场景容器（wifi_csi/eeg/radio/detection/generic/custom）
├── cli.py               # 命令行入口
├── exploration.py       # 探索闭环（ExplorationTracker）
├── introspect.py        # 自省协议
├── orchestration.py     # 编排协议（CloudEvent/K8s Operator）
├── routing.py           # 资源路由
├── schemas.py           # 错误码 schema
└── skills.py            # 技能库
```

## CLI

所有命令输出结构化 JSON。

| 命令 | 说明 | 示例 |
|------|------|------|
| `probe` | 探测硬件资源 | `python -m senseframe.cli probe` |
| `list-models` | 列出可用模型 | `python -m senseframe.cli list-models --dataset UT_HAR_data` |
| `list-datasets` | 列出可用数据集 | `python -m senseframe.cli list-datasets` |
| `list-scenes` | 列出场景容器 | `python -m senseframe.cli list-scenes` |
| `paradigms` | 列出 SOTA 范式 | `python -m senseframe.cli paradigms --category cnn` |
| `recommend` | 根据资源推荐模型 | `python -m senseframe.cli recommend --dataset UT_HAR_data` |
| `experiment` | YAML 配置驱动训练 | `python -m senseframe.cli experiment --config configs/config.yaml` |
| `export` | 多格式模型导出 | `python -m senseframe.cli export --formats onnx` |
| `compare` | 对比两份配置（A/B 实验） | `python -m senseframe.cli compare a.yaml b.yaml --repeats 3` |
| `predict` | 批量推理 | `python -m senseframe.cli predict --model model.pth --metadata metadata.json --samples samples.json` |
| `exploration` | 探索状态管理 | `python -m senseframe.cli exploration list` |
| `skills` | 技能库管理 | `python -m senseframe.cli skills list` |
| `catalog` | 技术目录查询 | `python -m senseframe.cli catalog list` |
| `monitor` | 训练实时监控 | `python -m senseframe.cli monitor output_dir` |
| `serve` | 启动推理服务 | `python -m senseframe.cli serve output_dir --port 8000` |
| `create-scene` | 创建场景脚手架 | `python -m senseframe.cli create-scene my_scene` |

## 配置

### 最小配置示例

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: MLP
  learning_mode: supervised

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: y
    type: category
    num_classes: 7

trainer:
  epochs: 200
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
  metrics: [accuracy, macro_f1]

output_dir: runs
save_model: true
```

参考：[配置 Schema](reference/reference_config_schema.md) | [配置模板](reference/reference_training_templates.md)

## 错误处理

基于 `error_code` 做程序化决策，而非字符串匹配。所有训练错误抛出 `SenseFrameError` 子类：

| error_code | 异常类 | 建议动作 |
|------------|--------|---------|
| `CONFIG_VALIDATION_ERROR` | `ConfigValidationError` | 修正配置，不重试 |
| `SCENE_NOT_FOUND` | `SceneNotRegisteredError` | 检查 scene.name |
| `DATASET_NOT_SUPPORTED` | `DatasetNotSupportedError` | 检查 dataset 名 |
| `MODEL_NOT_SUPPORTED` | `ModelNotSupportedError` | 检查 model_id |
| `DATA_NOT_FOUND` | `DataNotFoundError` | 检查 data_root，不重试 |
| `DATA_LOAD_ERROR` | `DataCorruptedError` | 检查数据完整性 / 格式 / 权限 |
| `OOM_ERROR` | `OOMError` | 降 batch_size 重试 |
| `CHECKPOINT_ERROR` | `CheckpointError` | 检查 checkpoint 路径 / 版本 / 完整性 |
| `PREFLIGHT_ERROR` | `PreflightError` | 升级硬件或换小模型 |
| `TRAINING_ERROR` | `TrainingError` | 查看 traceback |
| `MODEL_BUILD_ERROR` | `ModelBuildError` | 检查 model_id |
| `SAVE_ERROR` | `SaveError` | 检查磁盘空间 / 权限 |

```python
from senseframe.engine.runner.errors import SenseFrameError, OOMError

try:
    output = run_experiment(config)
except OOMError as e:
    # e.error_code == "OOM_ERROR"
    config.trainer.batch_size //= 2
    output = run_experiment(config)
except SenseFrameError as e:
    print(f"[{e.error_code}] {e}")
```

## 产物溯源

每次训练自动生成 `manifest.json`，记录全部产物的 SHA-256：

```python
from senseframe import load_manifest, verify_artifacts

manifest = load_manifest("runs/<exp>/manifest.json")
# manifest.artifacts: {name: ArtifactDescriptor(path, sha256, size)}

# 校验产物完整性（未被篡改 / 丢失）— 返回 {产物名: hash 是否匹配}
result = verify_artifacts("runs/<exp>/")
tampered = [name for name, ok in result.items() if not ok]
if tampered:
    print(f"产物校验失败: {tampered}")
```

## 资源安全

`PipelineContext.release_resources()` 主动释放训练资源，避免长任务 / HPO 中的资源泄露：

```python
ctx = sf.PipelineContext(config=my_config)
result = pipeline.run(ctx)
ctx.release_resources()  # 主动清理 Trainer / DataLoader / Logger / GPU 显存
```

HPO 路径自动调用；命令式路径需手动调用。

## 数据集与模型

参考：[数据集与模型支持表](reference/reference_datasets_models.md)

## 场景扩展

新增领域继承 `SceneContainer`，实现 4 个抽象方法：

```python
from senseframe.scenes.base import SceneContainer, SceneMeta

class TimeSeriesScene(SceneContainer):
    def meta(self) -> SceneMeta: ...
    def load_dataset(self, dataset_name, root, learning_mode="supervised") -> DatasetBundle: ...
    def build_model_for_dataset(self, model_id, dataset, num_classes, **kw) -> nn.Module: ...
    def get_dataset_info(self, dataset_name, **kw) -> dict: ...
```

参考：场景开发模板见 `senseframe/scenes/_template/` 目录

## 参考资源

- [配置 Schema](reference/reference_config_schema.md) — 完整字段与校验规则
- [数据集与模型](reference/reference_datasets_models.md) — 数据集与模型支持表
- [自省协议](reference/reference_introspect.md) — 字段契约 / 探索状态 / 技能库 / 断点续跑
- [资源路由](reference/reference_resource_routing.md) — 7 级路由表 + 模型推荐
- [配置模板](reference/reference_training_templates.md) — YAML 配置模板
- [MCP 子系统](reference/reference_mcp.md) — MCP 服务器工具与自省资源
- [NAS + AutoAugment](reference/reference_nas_autoaugment.md) — 神经架构搜索与数据增强搜索
- [AutoML/PEFT](reference/reference_automl_peft.md) — 参数高效微调与 SP 驱动搜索
- [Orchestration](reference/reference_orchestration.md) — CloudEvents 1.0 事件流与 K8s Operator

## License

MIT
