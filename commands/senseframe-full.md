---
description: "SenseFrame 困难场景：完整闭环 + 自监督 + 断点续跑 + 自省全量"
subtask: false
---

<!--

本文件是 Agent 提示词，部署到 .opencode/.claude/.agents/commands/ 供 AI Agent CLI 工具调用。
slash 命令 /senseframe-full 由 Agent CLI 工具（opencode/Claude Code）解析，不是 SenseFrame CLI 子命令。
SenseFrame CLI 子命令清单见 SKILL.md 或 `python -m senseframe.cli --help`。

command: /senseframe-full
场景: 困难 (30-90 min)
目的: 全面压力测试 SenseFrame 架构能力

覆盖:
  A. 数据画像驱动策略
  B. 自监督两阶段训练
  C. Pipeline 自定义编排
  D. 断点续跑
  E. 自省协议全量验证
  F. 多格式模型导出

用法:
  /senseframe-full                        # 默认 NTU-Fi_HAR
  /senseframe-full Widar3.0               # 自定义数据集

$1 = 数据集名 (默认 NTU-Fi_HAR)
-->

# SenseFrame 完整闭环压力测试

## Role

你是 AI Agent、机器学习、AutoML、WiFi CSI 信号处理四领域专家。
本测试是 SenseFrame 架构能力的全面压力测试，旨在发现**架构级问题**而非表面 bug。

## Objective

用 SenseFrame 命令式路径完成完整训练闭环：
A. 数据画像驱动策略选择
B. 自监督预训练 + 监督微调两阶段
C. Pipeline 自定义编排（替换/hook/跳过）
D. 模拟失败 → 断点续跑恢复
E. 自省协议全量验证（schema/stage_io/graph/readiness/validate_graph）
F. 多格式模型导出

**数据集**: `$1`（默认 `NTU-Fi_HAR`）

## Context

- 工作目录: `.`（已部署 SenseFrame）
- SKILL `senseframe` 已加载
- 数据集: `resource/CSI_DATASETS/`（如未部署，可改 `--data-root` 或软链；命令中所有路径均可替换为实际数据根）
- GPU: 8GB 显存，自监督 batch_size 建议 32
- **报告输出目录**: `report/`（不存在时自动创建）
- **报告命名格式**: `report/full_<dataset>_<YYYYMMDD_HHMMSS>.md`
  - 例：`report/full_NTU-Fi_HAR_20260706_143025.md`
  - 时间戳取测试开始时刻，避免重名覆盖

**架构重点审视项**（发现任一问题标记 [严重]）:
1. 统一执行路径：run_experiment / Pipeline.run / reconcile 是否真正统一？
2. extra 纪律化：框架代码是否向 ctx.extra 写入？
3. 异常层级：error_code 是否结构化？classify_error 是否 stage 独立？
4. CQS 合规：getter 是否有副作用？
5. DSP-3 就绪度：check_readiness / validate_graph 是否真正可用？
6. **输出契约分离（方案 D）**：dry-run/probe 输出是否区分内部表示（`route_config`）与输出契约（`lightning_params`）？用户层是否只见 `to_lightning_params()` 转换后的字段？
7. **资源生命周期（方案 F）**：Pipeline.run 结束后 `ctx.trainer/module/model` 是否为 None？WSL2 环境是否出现内存回收提示？
8. **产物溯源（方案 G）**：每次运行是否生成 `manifest.json`？`verify_artifacts()` 是否全部通过？

## Execution Protocol

### Step 0: 首次部署检查

**目的**：确保 SenseFi 代码库路径和数据集目录就绪，避免到训练阶段才暴露 ImportError 或 data_root 为空。

**Do**:
```bash
# 1. 检查 SenseFi 路径（wifi_csi 场景硬性依赖）
echo "SENSEFRAME_SENSEFI_PATH=${SENSEFRAME_SENSEFI_PATH:-NOT_SET}"

# 2. 检查数据集目录是否存在
ls resource/CSI_DATASETS/ 2>/dev/null || echo "CSI_DATASETS_NOT_FOUND"
```

**自动发现规则**：
- 若 `SENSEFRAME_SENSEFI_PATH` 已设置且路径存在 → 跳过，无需用户干预
- 若未设置，但当前目录下存在 `resource/SenseFi/` 或 `resource/WiFi-CSI-Sensing-Benchmark-main/` → **自动设为该路径**（`export SENSEFRAME_SENSEFI_PATH=$(pwd)/resource/<found_dir>`），继续执行
- 若未设置且找不到 → **停止并询问用户**：请提供 SenseFi 代码库路径

- 若 `resource/CSI_DATASETS/` 存在且含 `$1` 子目录 → 自动填入 `data_root: resource/CSI_DATASETS` 到配置
- 若不存在 → **停止并询问用户**：请提供数据集根路径

**Gate**: SenseFi 路径已设置 + data_root 非空 + 数据集目录存在

### 阶段 A: 数据画像驱动

**Do**:
```python
import senseframe as sf

# 方案 B 入口点契约：查询 registry 前必须显式激活延迟场景。
# get_scene("wifi_csi") 会触发延迟激活，但若先用 list_scenes()/list_models()
# 查询元数据，必须先调用 activate_lazy_scenes()，否则 wifi_csi 的模型/数据集
# 元数据不在 registry 中（CQS 合规：getter 不再有自动注册副作用）。
sf.activate_lazy_scenes()

scene = sf.get_scene("wifi_csi")
bundle = scene.load_dataset("$1", root="resource/CSI_DATASETS/...",
                            learning_mode="self_supervised")

profiler = sf.DataProfiler(max_samples=500)
profile = profiler.profile_bundle(bundle, dataset_name="$1")

print(f"task_type: {profile.recommended_task_type}")
print(f"loss: {profile.recommended_loss}")
print(f"normalization: {profile.recommended_normalization}")
print(f"input_shape: {profile.input_shape}")
print(f"n_classes: {profile.n_classes}")
print(f"class_distribution: {profile.class_distribution}")
```

**Output**: 数据画像报告
**Gate**: 画像字段完整，recommended_* 合理

**Introspect**:
- [ ] DataProfiler 是否正确识别 CSI 数据特征（时序、空间、多天线）？
- [ ] recommended_normalization 是否适合 CSI 信号分布？
- [ ] class_distribution 是否反映 HAR 类别不平衡？
- [ ] profile.save/load 是否可用？

### 阶段 B: 自监督两阶段训练

**Do**:
```python
from senseframe.engine.config import (
    ExperimentConfig, SceneConfig, InputFeature, OutputFeature, TrainerConfig
)

# B1: 自监督预训练
# 注意：trainer 仅覆盖 epochs/seed，其余字段（weight_decay/early_stopping/
# scheduler/early_stopping_min_delta）使用 TrainerConfig 默认值（方案 E 默认最佳实践）
# 修复（文档契约）：ExperimentConfig 推荐用 from_dict(yaml_dict) + validate() 构造，
# 自动解析嵌套 dict 为 dataclass 实例并做基本校验。
# 旧文档把 scene 字段平铺到顶层、漏 input_features/output_features、用 max_epochs
# 而非 epochs，全部与代码契约不符。
config_pretrain_dict = {
    "scene": {
        "name": "wifi_csi", "dataset": "$1", "model_id": "ResNet18",
        "learning_mode": "self_supervised",
        "data_root": "$SENSEFRAME_DATA_ROOT",  # 必填：YAML/CLI/env 三选一
    },
    "input_features": [{"name": "csi", "type": "csi", "shape": [1, 250, 90]}],
    "output_features": [{"name": "label", "type": "category", "num_classes": 7}],
    "trainer": {"epochs": 5, "seed": 42},
}
config_pretrain = ExperimentConfig.from_dict(config_pretrain_dict)
config_pretrain.validate()
output_pretrain = sf.run_experiment(config_pretrain)
# 字段契约（方案 C）：final_eval 使用 val_ 前缀
print(f"pretrain val_loss: {output_pretrain.final_eval.get('val_loss')}")

# B2: 监督微调（加载预训练权重）
config_finetune_dict = {
    "scene": {
        "name": "wifi_csi", "dataset": "$1", "model_id": "ResNet18",
        "learning_mode": "supervised",
        "data_root": "$SENSEFRAME_DATA_ROOT",  # 必填：YAML/CLI/env 三选一
    },
    "input_features": [{"name": "csi", "type": "csi", "shape": [1, 250, 90]}],
    "output_features": [{"name": "label", "type": "category", "num_classes": 7}],
    "trainer": {"epochs": 10, "seed": 42},
}
config_finetune = ExperimentConfig.from_dict(config_finetune_dict)
config_finetune.validate()
output_finetune = sf.run_experiment(config_finetune)
# 字段契约（方案 C）：读 val_accuracy 而非 accuracy
print(f"finetune val_accuracy: {output_finetune.final_eval.get('val_accuracy')}")
print(f"finetune val_macro_f1: {output_finetune.final_eval.get('val_macro_f1')}")
```

**Output**: 两阶段训练曲线 + 最终指标
**Gate**: 两阶段均完成，微调 `output.final_eval["val_accuracy"]` 非零

**Introspect**:
- [ ] EntLoss 无监督预训练是否正常执行？
- [ ] 预训练 → 微调权重传递是否正确？
- [ ] 自监督对 val_accuracy 提升是多少？（对比有/无预训练）
- [ ] 自监督数据契约（unsupervised/train/test 填充规则）是否被校验？
- [ ] 早停是否在两阶段都生效？
- [ ] **方案 E 验证**：两阶段是否都继承了默认的 `weight_decay=1e-4` / `early_stopping=5` / `scheduler="cosine"` / `early_stopping_min_delta=0.001`？（检查 `config.trainer` 字段）
- [ ] **方案 C 验证**：`final_eval` 是否含 `val_loss`/`val_accuracy`/`val_macro_f1` 等 `val_` 前缀字段？（而非旧的无前缀 `accuracy`/`macro_f1`）

### 阶段 C: Pipeline 自定义编排

**Do**: 用命令式 Pipeline 替代声明式，验证编排能力。

```python
pipeline = sf.Pipeline.default()

# C1: 查询 stage 契约
specs = pipeline.stages_with_spec()
for spec in specs:
    print(f"{spec.name}: reads={spec.reads}, writes={spec.writes}")

# C2: 替换 eval
def my_custom_eval(ctx):
    # 自定义评估逻辑
    ...
    return ctx
pipeline.replace_stage("eval", my_custom_eval)

# C3: 插入 hook
def data_check_hook(ctx):
    print(f"数据形状: {ctx.bundle.train_dataset[0][0].shape}")
    return ctx
pipeline.before("train", data_check_hook)

def log_metrics_hook(ctx):
    print(f"训练时长: {ctx.training_duration_s}s, 最佳: {ctx.best_model_path}")
    return ctx
pipeline.after("train", log_metrics_hook)

# C4: 跳过 export
pipeline.skip("export")

# C5: 执行
ctx = sf.PipelineContext(config=config_finetune)
result = pipeline.run(ctx)
```

**Output**: 自定义 pipeline 执行结果
**Gate**: replace/before/after/skip 全部生效

**Introspect**:
- [ ] replace_stage 是否真正替换？自定义 eval 是否执行？
- [ ] before/after hook 是否在正确位置执行？
- [ ] skip 是否真正跳过 stage？
- [ ] stages_with_spec 的 reads/writes 是否与实际一致？
- [ ] ctx 的 first-class 字段（training_duration_s / best_model_path）是否可读？
- [ ] stage 失败时 ctx.failed_stage / ctx.failed_error 是否正确填充？

### 阶段 D: 断点续跑

**Do**: 模拟 train stage 失败，然后用 Pipeline.resume 恢复。

**⚠️ 关键约束（RFC-004 方案 F）**：Pipeline.run() 的 finally 块会调用
`ctx.release_resources()`，把 `trainer`/`module`/`model`/`datamodule`/`bundle`
等大对象置 None。因此续跑时 **不能复用原 ctx**，必须新构造 ctx 并从
`load` stage 重建 bundle/datamodule/model，再走到 `train` stage。
`completed_stages` 仅记录"逻辑进度"，不保留对象引用。

```python
# D1: 构造会失败的 pipeline
pipeline = sf.Pipeline.default()

original_train = None
for name, fn in pipeline.stages:
    if name == "train":
        original_train = fn
        break

def failing_train(ctx):
    raise RuntimeError("模拟训练失败")
pipeline.replace_stage("train", failing_train)

ctx = sf.PipelineContext(config=config_finetune)
result = pipeline.run(ctx)  # 应在 train 失败

print(f"失败 stage: {ctx.failed_stage}")
print(f"失败原因: {ctx.failed_error}")

# 验证方案 F：release_resources 已清空大对象
print(f"trainer is None: {ctx.trainer is None}")  # 应为 True
print(f"bundle is None: {ctx.bundle is None}")    # 应为 True

# D2: 断点续跑 — 必须新构造 ctx，从 load stage 重建大对象
pipeline2, completed = sf.Pipeline.resume(ctx.output_dir)
print(f"已完成 stages: {completed}")

ctx2 = sf.PipelineContext(config=config_finetune)
ctx2.completed_stages = completed
# 修复（文档契约）：stage_checkpoint_path 字段类型是 Path，不是 str。
# 旧文档赋 str 会在 ctx.stage_checkpoint_path.exists() 调用时 AttributeError。
# 同时 output_dir 也是 Path 类型，应用 Path 拼接而非字符串拼接。
from pathlib import Path
ctx2.stage_checkpoint_path = Path(ctx.output_dir) / "pipeline_checkpoint.json"

# 关键：completed_stages 含 load/build 时，Pipeline 会跳过它们，
# 但 ctx2 的大对象为 None（新构造）。因此续跑前需手动重建，
# 或清除 completed_stages 中 load/build 让 Pipeline 重跑。
# 推荐做法：清除 load/build，仅保留 validate/preflight/resolve
# 修复（文档契约）：stage 名不带 "stage_" 前缀（实际是 "load"/"build" 而非
# "stage_load"/"stage_build"），旧文档用前缀过滤会失效，load/build 仍会被跳过。
ctx2.completed_stages = [s for s in completed
                         if s not in ("load", "build")]

# 恢复原始 train
for name, fn in pipeline2.stages:
    if name == "train":
        pipeline2.replace_stage("train", original_train)
        break

result2 = pipeline2.run(ctx2)  # 应从 load 重建，再到 train
```

**Output**: 失败信息 + 续跑结果
**Gate**: 失败时 ctx.failed_stage 正确，续跑跳过已完成 stage

**Introspect**:
- [ ] 失败时 ctx.failed_stage / ctx.failed_error 是否正确填充？
- [ ] **方案 F 验证**：失败后 ctx.trainer/module/model/bundle 是否为 None（release_resources 已执行）？
- [ ] Pipeline.resume 是否正确返回 completed_stages？
- [ ] 续跑是否真正跳过已完成 stage？（对比耗时）
- [ ] checkpoint 文件是否持久化？
- [ ] **续跑重建**：ctx2 的大对象是否从 load/build stage 重建（非复用原 ctx 的 None 引用）？

### 阶段 E: 自省协议全量验证

**Do**: 验证所有自省 API。

```python
import senseframe as sf

ctx = sf.PipelineContext(config=config_finetune)

# E1: schema
schema = ctx.schema()                    # 每个字段是否有 fill_stage / type / has_default

# E2: filled_at / completed_fields
filled = ctx.filled_at("stage_load")
completed = ctx.completed_fields()

# E3: describe
desc = ctx.describe()

# E4: stage_io
# 注意：stage 名不带 "stage_" 前缀（如 "train"），带前缀会返回 not found
io_train = sf.stage_io("train")    # reads / writes

# E5: pipeline_graph
graph = sf.pipeline_graph()              # DAG

# E6: check_readiness（DSP-3）
pipeline = sf.Pipeline.default()
report = pipeline.check_readiness(ctx, "train")
# ReadinessReport(available=bool, missing_reads=list)

# E7: validate_graph（编译期 dangling ref）
dangling = pipeline.validate_graph()
# List[DanglingRef]
```

**Output**: 自省 API 全量验证结果
**Gate**: 所有自省 API 可调用，输出结构化

**Introspect**:
- [ ] schema() 输出是否完整准确？
- [ ] filled_at / completed_fields 是否与实际一致？
- [ ] stage_io reads/writes 是否与 stage 实际行为匹配？
- [ ] check_readiness 是否正确检测缺失字段？（available=False 时不阻塞执行？）
- [ ] validate_graph 是否正确检测 dangling reference？
- [ ] Agent 能否仅凭自省 API 组装 pipeline 而不读源码？

### 阶段 F: 模型导出 + 产物溯源验证

**Do**:
```bash
python -m senseframe.cli export \
    --metadata runs/<实验目录>/metadata.json \
    --checkpoint runs/<实验目录>/model.pth \
    --formats onnx,torchscript,state_dict \
    --output-dir exports/
```

**Do**: 验证产物溯源（RFC-004 方案 G）

```python
import senseframe as sf
from pathlib import Path

# 1. 加载本次运行的 manifest
manifest = sf.load_manifest(Path("runs/<实验目录>"))
print(f"run_id: {manifest.run_id}")
print(f"artifacts: {len(manifest.artifacts)}")

# 2. 按类型列出产物
for kind in ("model", "metrics", "config", "log", "metadata"):
    items = manifest.list_by_kind(kind)
    print(f"  {kind}: {len(items)} 个")

# 3. 校验所有产物完整性（hash + 存在性）
# verify_artifacts 接受 output_dir（含 manifest.json），返回 {产物名: hash 是否匹配}
report = sf.verify_artifacts(Path("runs/<实验目录>"))
verified = sum(1 for ok in report.values() if ok)
total = len(report)
print(f"verified: {verified}/{total}")
missing_or_mismatch = [name for name, ok in report.items() if not ok]
if missing_or_mismatch:
    print(f"MISMATCH: {missing_or_mismatch}")
```

**Output**: 多格式导出产物 + manifest 校验报告
**Gate**: 所有声明格式导出成功 + manifest.json 存在 + verify_artifacts 全部通过

**Introspect**:
- [ ] 每种格式是否导出成功？
- [ ] ONNX 是否通过 onnx.checker？
- [ ] 导出产物是否可用于推理？
- [ ] **方案 G 验证**：manifest.json 是否生成？artifacts 列表是否完整？
- [ ] **方案 G 验证**：verify_artifacts 是否全部通过（无 missing / hash_mismatch）？
- [ ] **方案 G 验证**：每个产物是否含 producer_stage / content_hash / size_bytes？

## Introspection Protocol

四视角评分，每阶段必填：

1. **AI/Agent** [1-5]: API 可编程？error_code 结构化？自省 API 价值？failed_stage 可读？
2. **ML** [1-5]: 自监督有效？策略合理？过拟合检测？
3. **AutoML** [1-5]: 端到端自动化？断点续跑可靠？Pipeline 编排灵活？
4. **CSI** [1-5]: CSI 预处理正确？模型适配？自监督对 CSI 有意义？

**纪律**: 基于执行事实；问题不掩盖；评分有区分度；建议指向具体 API/字段/文档。
**架构级问题标记 [严重]**: 统一执行路径 / extra 纪律 / 异常层级 / CQS / DSP-3。

## Output Contract

执行完毕后，**必须**将报告写入 `report/full_<dataset>_<YYYYMMDD_HHMMSS>.md`，
并在 stdout 输出报告路径。报告内容必须包含以下章节，**禁止省略**任何章节。

```markdown
# SenseFrame 测试报告：完整闭环压力测试

## 报告元数据
- 报告路径: `report/full_<dataset>_<YYYYMMDD_HHMMSS>.md`
- 生成时间: <ISO8601>
- 测试命令: `/senseframe-full <dataset>`
- 框架版本: <senseframe.__version__>

## 执行摘要
- 环境: Python <version> | torch <version> | CUDA <available/version>
- 硬件: <CPU/GPU 型号 + 显存>
- 数据集: <$1>
- 状态: <成功/失败/部分成功> | 总耗时: <min>

## 阶段执行总览
| 阶段 | 状态 | 耗时 | 关键产出 |
|------|------|------|---------|
| A. 数据画像 | ok/fail | <s> | <profile 字段摘要> |
| B. 自监督 | ok/fail | <s> | <预训练 val_acc → 微调 val_acc> |
| C. Pipeline 编排 | ok/fail | <s> | <replace/hook/skip 验证> |
| D. 断点续跑 | ok/fail | <s> | <failed_stage → completed_stages> |
| E. 自省协议 | ok/fail | <s> | <7 项 API 验证结果> |
| F. 模型导出 | ok/fail | <s> | <onnx/torchscript/state_dict> |

## 阶段 A: 数据画像详情
- 数据集 bundle 字段: <列出 train/test/val 样本数 + 形状>
- DataProfiler 输出（必须列实际值）:
  - recommended_task_type: <值>
  - recommended_loss: <值>
  - recommended_normalization: <值>
  - input_shape: <值>
  - n_classes: <值>
  - class_distribution: <dict>
  - modality: <值>（如 image 须标记 [疑似误判]）
  - is_temporal: <值>
  - is_spatial: <值>
- profile.save/load 是否可用: <yes/no + 路径>

## 阶段 B: 自监督训练详情
### B1 预训练
- config_pretrain: epochs=<>, seed=<>, weight_decay=<>, early_stopping=<>, scheduler=<>
- output_pretrain.output_dir: <path>
- output_pretrain.final_eval: <列出全部 val_* 字段>
- 预训练耗时: <s>
- EntLoss 是否正常执行: <yes/no + 日志片段>

### B2 微调
- config_finetune: epochs=<>, seed=<>, weight_decay=<>, early_stopping=<>, scheduler=<>
- output_finetune.output_dir: <path>
- output_finetune.final_eval:
  - val_accuracy: <值>
  - val_loss: <值>
  - val_macro_f1: <值>
- 微调耗时: <s>

### 自监督收益分析
- 无预训练 val_accuracy: <值>（如执行了对比；未执行标 N/A）
- 有预训练 val_accuracy: <值>
- 提升: <+x.x%> 或 <N/A>
- 早停是否两阶段都生效: <yes/no + 详情>

## 阶段 C: Pipeline 自定义编排详情
### C1 stage 契约
- stages_with_spec 返回（全部 stage）:
  | stage | reads | writes |
  |-------|-------|--------|
  | validate | <list> | <list> |
  | preflight | | |
  | ... | | |

### C2-C4 自定义操作验证
| 操作 | 类型 | 是否生效 | 证据（日志片段） |
|------|------|---------|----------------|
| replace_stage("eval", my_custom_eval) | replace | yes/no | <日志输出证明自定义 eval 执行> |
| before("train", data_check_hook) | hook | yes/no | <日志输出 "数据形状: ..."> |
| after("train", log_metrics_hook) | hook | yes/no | <日志输出 "训练时长: ..."> |
| skip("export") | skip | yes/no | <日志证明 export 跳过> |

### C5 自定义 pipeline 执行
- pipeline.run 结果: <success/fail>
- ctx.first-class 字段可读性:
  - training_duration_s: <值>
  - best_model_path: <值>
- stage 失败时（如有）:
  - ctx.failed_stage: <值>
  - ctx.failed_error: <值>

## 阶段 D: 断点续跑详情
### D1 模拟失败
- 失败 stage: train
- ctx.failed_stage: <实际值>
- ctx.failed_error: <实际异常信息>
- 方案 F 验证:
  - ctx.trainer is None: <yes/no>
  - ctx.module is None: <yes/no>
  - ctx.model is None: <yes/no>
  - ctx.bundle is None: <yes/no>
- 失败时日志片段: <粘贴 3-5 行>

### D2 续跑
- Pipeline.resume 返回 completed_stages: <list>
- ctx2.completed_stages 调整后: <list>
- 续跑结果: <成功/失败>
- 续跑耗时: <s>（对比首次 D1 耗时）
- 是否真正跳过已完成 stage: <yes/no + 耗时对比证据>
- checkpoint 文件路径: <path>
- 续跑重建验证: ctx2.bundle is None: <yes/no（应为 no，已重建）>

## 阶段 E: 自省协议详情
| API | 调用成功 | 输出准确 | 与实际一致 | 备注 |
|-----|---------|---------|-----------|------|
| schema() | yes/no | yes/no | yes/no | <字段数 + 是否含 fill_stage> |
| filled_at("stage_load") | yes/no | yes/no | yes/no | <返回值> |
| completed_fields() | yes/no | yes/no | yes/no | <字段数> |
| stage_io("train") | yes/no | yes/no | yes/no | <reads/writes 列表> |
| pipeline_graph() | yes/no | yes/no | yes/no | <DAG 描述> |
| check_readiness(ctx, "train") | yes/no | yes/no | yes/no | <available + missing_reads> |
| validate_graph() | yes/no | yes/no | yes/no | <dangling list> |

- Agent 能否仅凭自省 API 组装 pipeline: <yes/no + 说明>

## 阶段 F: 模型导出 + 产物溯源详情
### F1 多格式导出
| 格式 | 导出成功 | 文件路径 | 大小 | 验证 |
|------|---------|---------|------|------|
| onnx | yes/no | <path> | <bytes> | <onnx.checker 通过/失败> |
| torchscript | yes/no | <path> | <bytes> | <加载测试> |
| state_dict | yes/no | <path> | <bytes> | <加载测试> |

### F2 产物溯源
- manifest.json 路径: <path>
- run_id: <uuid>
- artifacts 总数: <N>
- verify_artifacts 结果:
  | 产物名 | kind | producer_stage | content_hash(前8位) | 校验结果 |
  |--------|------|---------------|-------------------|---------|
  | model_weights | model | stage_export | <hash> | ✓/✗ |
  | ... | | | | |
  - verified/total: <N/M>
- 失败产物详情（如有）: <list + 原因>

## Pipeline DAG（文字版）
<stage 依赖关系，从 validate → export>

## 架构级问题检查
| 检查项 | 状态 | 证据（粘贴日志/字段值） |
|--------|------|----------------------|
| 统一执行路径 | pass/fail | <如"run_experiment 与 Pipeline.run 输出一致"> |
| extra 纪律化 | pass/fail | <如"框架代码向 ctx.extra 写入 0 次"> |
| 异常层级 | pass/fail | <如"error_code 结构化，含 stage + code"> |
| CQS 合规 | pass/fail | <如"getter 无副作用"> |
| DSP-3 就绪度 | pass/fail | <如"check_readiness 正确检测缺失字段"> |
| 输出契约分离（方案 D） | pass/fail | <如"dry-run 输出 route_config + lightning_params"> |
| 资源生命周期（方案 F） | pass/fail | <如"Pipeline.run 后 trainer is None"> |
| 产物溯源（方案 G） | pass/fail | <如"manifest.json 生成 + verify_artifacts 全通过"> |
| 入口点激活（方案 B） | pass/fail | <如"activate_lazy_scenes 后 list_models 含 wifi_csi"> |
| 字段契约对齐（方案 C） | pass/fail | <如"final_eval 含 val_accuracy 而非 accuracy"> |
| 默认最佳实践（方案 E） | pass/fail | <如"trainer 含 weight_decay/early_stopping/scheduler"> |

## 自省评分矩阵
| 阶段 | AI/Agent | ML | AutoML | CSI | 平均 | 关键扣分原因 |
|------|----------|----|--------|-----|------|------------|
| A. 画像 | x | x | x | x | x.x | <如"CSI 误判为 image"> |
| B. 自监督 | x | x | x | x | x.x | <如"权重传递未验证"> |
| C. 编排 | x | x | x | x | x.x | <如"skip 不彻底"> |
| D. 续跑 | x | x | x | x | x.x | <如"重建机制复杂"> |
| E. 自省 | x | x | x | x | x.x | <如"stage_io 不准"> |
| F. 导出 | x | x | x | x | x.x | <如"onnx 导出失败"> |

## 关键发现（按严重度排序）
每个发现必须包含：复现命令 + 实际输出 + 期望输出 + 影响范围 + 严重度

1. **[严重]** <问题标题>
   - 复现命令: `<完整命令>`
   - 实际输出: `<粘贴实际输出>`
   - 期望输出: `<应该是什么>`
   - 根因分析: <代码位置 + 逻辑错误>
   - 影响: <对哪些功能/用户/场景有影响>
   - 建议修复: <具体修复方向>

2. **[中等/轻微]** <问题标题>
   - ...（同上格式）

## 改进建议（按优先级排序）
每条建议必须含：优先级 + 具体修改点 + 影响文件/模块 + 预期收益

1. **[P0]** <建议>
   - 修改文件: `<file:line>`
   - 修改内容: <具体改什么>
   - 预期收益: <修复后效果>

2. **[P1/P2]** <建议>
   - ...

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
- 下一步建议: <如"修复架构级问题后重新压测">
```

## Constraints

- 禁止跳过任何阶段（A-F 必须全部执行）
- 禁止伪造失败（断点续跑的失败必须真实模拟）
- 禁止跳过自省 API（每阶段必须用自省 API 查询契约）
- 禁止掩盖问题（架构级问题必须标记 [严重]）
- 禁止全 5 分（评分必须有区分度，扣分必须填"关键扣分原因"）
- 禁止省略章节：报告必须含全部章节，无内容时填"无"并说明原因
- 报告必须落盘到 `report/` 目录，禁止仅输出到 stdout
