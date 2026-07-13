---
description: "SenseFrame 困难场景：完整闭环 + 自监督 + 断点续跑 + 自省全量"
subtask: false
---

<!--

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
- 数据集: `CSI_DATASETS/`（如未部署，可改 `--data-root` 或软链；命令中所有路径均可替换为实际数据根）
- GPU: 8GB 显存，自监督 batch_size 建议 32

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
bundle = scene.load_dataset("$1", root="CSI_DATASETS/...",
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
from senseframe.engine.config import ExperimentConfig

# B1: 自监督预训练
# 注意：trainer 仅覆盖 max_epochs/seed，其余字段（weight_decay/early_stopping/
# scheduler/early_stopping_min_delta）使用 TrainerConfig 默认值（方案 E 默认最佳实践）
config_pretrain = ExperimentConfig(
    scene="wifi_csi", dataset="$1", model_id="ResNet18",
    learning_mode="self_supervised",
    trainer={"max_epochs": 5, "seed": 42},
)
output_pretrain = sf.run_experiment(config_pretrain)
# 字段契约（方案 C）：final_eval 使用 val_ 前缀
print(f"pretrain val_loss: {output_pretrain.final_eval.get('val_loss')}")

# B2: 监督微调（加载预训练权重）
config_finetune = ExperimentConfig(
    scene="wifi_csi", dataset="$1", model_id="ResNet18",
    learning_mode="supervised",
    trainer={"max_epochs": 10, "seed": 42},
)
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

**Do**: 模拟 stage_train 失败，然后用 Pipeline.resume 恢复。

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
ctx2.stage_checkpoint_path = str(ctx.output_dir) + "/pipeline_checkpoint.json"

# 关键：completed_stages 含 load/build 时，Pipeline 会跳过它们，
# 但 ctx2 的大对象为 None（新构造）。因此续跑前需手动重建，
# 或清除 completed_stages 中 load/build 让 Pipeline 重跑。
# 推荐做法：清除 load/build，仅保留 validate/preflight/resolve
ctx2.completed_stages = [s for s in completed
                         if s not in ("stage_load", "stage_build")]

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

```markdown
# SenseFrame 测试报告：完整闭环压力测试

## 执行摘要
- 环境: <Python/torch/CUDA>
- 数据集: <$1>
- 状态: <成功/失败/部分成功> | 耗时: <min>

## 阶段执行表
| 阶段 | 状态 | 耗时 | 关键产出 |
|------|------|------|---------|
| A. 数据画像 | ok/fail | <s> | <profile 字段摘要> |
| B. 自监督 | ok/fail | <s> | <预训练 val_acc → 微调 val_acc> |
| C. Pipeline 编排 | ok/fail | <s> | <replace/hook/skip 验证> |
| D. 断点续跑 | ok/fail | <s> | <failed_stage → completed_stages> |
| E. 自省协议 | ok/fail | <s> | <7 项 API 验证结果> |
| F. 模型导出 | ok/fail | <s> | <onnx/torchscript/state_dict> |

## 自监督收益分析
- 无预训练 val_accuracy: <值>（如执行了对比）
- 有预训练 val_accuracy: <值>
- 提升: <+x.x%>

## 断点续跑验证
- 模拟失败 stage: <name>
- ctx.failed_stage: <值>
- ctx.failed_error: <值>
- resume 返回 completed_stages: <list>
- 续跑结果: <成功/失败>

## 自省协议验证表
| API | 调用成功 | 输出准确 | 与实际一致 |
|-----|---------|---------|-----------|
| schema() | yes/no | yes/no | yes/no |
| filled_at() | yes/no | yes/no | yes/no |
| completed_fields() | yes/no | yes/no | yes/no |
| stage_io() | yes/no | yes/no | yes/no |
| pipeline_graph() | yes/no | yes/no | yes/no |
| check_readiness() | yes/no | yes/no | yes/no |
| validate_graph() | yes/no | yes/no | yes/no |

## Pipeline DAG（文字版）
<stage 依赖关系>

## 自省评分矩阵
| 阶段 | AI/Agent | ML | AutoML | CSI | 平均 |
|------|----------|----|--------|-----|------|
| A. 画像 | x | x | x | x | x.x |
| B. 自监督 | x | x | x | x | x.x |
| C. 编排 | x | x | x | x | x.x |
| D. 续跑 | x | x | x | x | x.x |
| E. 自省 | x | x | x | x | x.x |
| F. 导出 | x | x | x | x | x.x |

## 架构级问题检查
| 检查项 | 状态 | 证据 |
|--------|------|------|
| 统一执行路径 | pass/fail | <证据> |
| extra 纪律化 | pass/fail | <证据> |
| 异常层级 | pass/fail | <证据> |
| CQS 合规 | pass/fail | <证据> |
| DSP-3 就绪度 | pass/fail | <证据> |
| 输出契约分离（方案 D） | pass/fail | <证据> |
| 资源生命周期（方案 F） | pass/fail | <证据> |
| 产物溯源（方案 G） | pass/fail | <证据> |
| 入口点激活（方案 B） | pass/fail | <证据> |
| 字段契约对齐（方案 C） | pass/fail | <证据> |
| 默认最佳实践（方案 E） | pass/fail | <证据> |

## 关键发现（按严重度排序）
1. [严重/中等/轻微] <问题 + 复现步骤 + 影响>

## 改进建议（按优先级排序）
1. [P0/P1/P2] <具体建议 + 影响范围 + 预期收益>

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
```

## Constraints

- 禁止跳过任何阶段（A-F 必须全部执行）
- 禁止伪造失败（断点续跑的失败必须真实模拟）
- 禁止跳过自省 API（每阶段必须用自省 API 查询契约）
- 禁止掩盖问题（架构级问题必须标记 [严重]）
- 禁止全 5 分（评分必须有区分度）
