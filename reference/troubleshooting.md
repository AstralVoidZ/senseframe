# 错误排查指南

## 错误分类

senseframe 的错误分为两类：

1. **配置错误（快速失败）**：抛 `ValueError`，不进入训练流程
2. **运行时错误（捕获）**：捕获到 `TrainOutput.status="error"`，保留 traceback

## 配置错误（ValueError）

配置错误在 `run_experiment` 入口处快速失败，不进入训练编排层。

| 错误信息 | 原因 | 修复 |
|----------|------|------|
| `Scene 'X' not registered` | `scene.name` 拼错 | 用 `list-scenes` 查可用场景 |
| `Dataset 'X' not supported by scene` | `dataset` 不在该场景支持列表 | 用 `list-datasets` 查可用数据集 |
| `Model 'X' not supported by scene` | `model_id` 不在该场景支持列表 | 用 `list-models` 查可用模型 |
| `Self-supervised mode only supports dataset 'NTU-Fi_HAR'` | 自监督模式 `dataset` 写错 | 改为 `NTU-Fi_HAR` |
| `InputFeature.type 'X' 不支持` | `type` 拼错 | 用 `csi`/`tabular`/`image`/`text`/`sequence` |
| `OutputFeature.type='category' 需要指定 num_classes` | 缺 `num_classes` | 补上 `num_classes` 字段 |
| `num_classes 必须 >= 2` | `num_classes` 为 0 或 1 | 改为 >= 2 的整数 |
| `epochs 必须 > 0` | `epochs` 写成 0 或负数 | 改为正整数 |
| `learning_rate 必须 > 0` | `learning_rate` 为 0 或负数 | 改为正浮点数 |
| `batch_size 必须 > 0` | `batch_size` 为 0 或负数 | 改为正整数 |
| `optimizer 'X' 不支持` | `optimizer` 拼错 | 用 `adam`/`adamw`/`sgd`/`rmsprop` |
| `hpo.direction 'X' 不支持` | `direction` 拼错 | 用 `minimize`/`maximize` |
| `input_features 必须是非空 list` | `input_features` 为空或非 list | 补上至少一个输入特征 |
| `output_features 必须是非空 list` | `output_features` 为空或非 list | 补上至少一个输出特征 |

## 运行时错误（TrainOutput.status="error"）

运行时错误在训练过程中发生，被捕获到 `TrainOutput`，不抛异常。

| 错误模式 | 原因 | 修复 |
|----------|------|------|
| `Dataset directory not found: X` | 数据集目录不存在 | 检查 `data_root` 路径，确保子目录存在（如 `UT_HAR/`） |
| `GPU free VRAM XMB < required YMB` | 显存不足 | 换小模型，或用 CPU 路由 |
| `Disk free space XMB < 1024MB` | 磁盘空间不足 | 清理 `output_dir` 所在磁盘 |
| 训练中途 OOM | `batch_size` 过大 | 降低 `batch_size` 或换小模型 |
| `loss=NaN` | 学习率过大或数据异常 | 降低 `learning_rate`，检查数据归一化 |
| 训练不收敛 | 学习率/epoch 不合适 | 调整超参，启用 `early_stopping` |

## 结构化错误码

`TrainOutput.error_code` 字段（定义在 `senseframe/schemas.py` 的 `ERROR_CODES` 字典，共 14 项）。Agent 基于此做程序化分支，无需字符串匹配 `error` 字段。

| error_code | 说明 | 触发场景 | 建议动作 |
|------------|------|----------|---------|
| `OK` | 成功 | 训练正常完成 | 进入后处理 |
| `CONFIG_VALIDATION_ERROR` | 配置校验失败 | ValueError 逃逸到 except 块 | 修正配置，不重试 |
| `SCENE_NOT_FOUND` | 场景未注册 | `scene.name` 未注册 | 用 `list-scenes` 查可用场景 |
| `DATASET_NOT_SUPPORTED` | 数据集不被场景支持 | `dataset` 不在场景支持列表 | 用 `list-datasets` 查可用数据集 |
| `MODEL_NOT_SUPPORTED` | 模型不被场景支持 | `model_id` 不在场景支持列表 | 用 `list-models` 查可用模型 |
| `DATA_NOT_FOUND` | 数据集文件未找到 | FileNotFoundError | 检查 `data_root`，不重试 |
| `DATA_LOAD_ERROR` | 数据加载失败 | 数据集加载异常 | 检查数据完整性 |
| `MODEL_BUILD_ERROR` | 模型构建失败 | KeyError / AttributeError | 检查 `model_id` |
| `TRAINING_ERROR` | 训练过程异常 | 其他 RuntimeError | 查看 traceback 定位 |
| `OOM_ERROR` | 显存/内存不足 | RuntimeError 含 "out of memory" | 降 `batch_size` 重试 |
| `CHECKPOINT_ERROR` | Checkpoint 加载/保存失败 | RuntimeError 含 "checkpoint" | 检查 checkpoint 路径 |
| `SAVE_ERROR` | 模型/元数据保存失败 | OSError 含 "model.pth" | 检查磁盘空间/权限 |
| `PREFLIGHT_ERROR` | 预检失败 | RuntimeError 含 "VRAM" / "Disk" | 升级硬件或换小模型 |
| `UNKNOWN_ERROR` | 未知错误 | 其他异常类型 | 查看 traceback 定位 |

**TrainOutput.summary()**：生成机器可读的状态摘要：

```python
output = run_experiment(config)
summary = output.summary()
# success 时：{"status": "success", "key_metrics": {...}, "model_path": "...", "duration_s": ...}
# error 时：{"status": "error", "error_code": "OOM_ERROR", "error": "..."}
```

## 排查步骤

### 步骤 1：查看 TrainOutput 状态

```python
output = run_experiment(config)
if output.status == "error":
    print(f"Error: {output.error}")
    print(f"Error code: {output.error_code}")
    # 查看 output.output_dir/error_traceback.txt 获取完整堆栈
```

### 步骤 2：查看 error_traceback.txt

失败时 `output_dir` 下会生成 `error_traceback.txt`，包含完整 Python 堆栈。

### 步骤 3：查看 training_log.jsonl

`output_dir/training_log.jsonl` 是增量日志，每 epoch 一行 JSON。检查：
- loss 是否发散或 NaN
- 学习率是否合适
- 哪个 epoch 开始出问题

### 步骤 4：查看 metadata.json

`output_dir/metadata.json` 保存完整实验元数据，包括：
- 实际使用的配置（含路由覆盖后的值）
- 环境快照（torch/pl/cuda/python 版本）
- 训练结果

### 步骤 5：查看 feedback.json（结构化训练反馈）

`output_dir/feedback.json` 由 `stage_eval` 自动产出，包含训练结果诊断：

```json
{
  "status": "overfitting",
  "diagnosis": "train_acc=0.95, val_acc=0.78, gap=0.17 > 0.15",
  "suggestions": [
    "增加数据增强",
    "增大 weight_decay",
    "启用 dropout",
    "减小模型容量",
    "启用 early_stopping"
  ]
}
```

5 种 status：
- `numerical_instability`：指标含 NaN/Inf
- `underfitting`：`val_acc < 0.5`（分类任务）
- `overfitting`：`train_acc - val_acc > 0.15`
- `converged`：`early_stopped=True`
- `success`：默认

### 步骤 6：运行 probe 命令

```bash
python -m senseframe.cli probe
```

确认资源探测结果符合预期，路由级别正确。

## 输出目录结构

```
runs/{model_id}_{dataset}_{timestamp}_{pid}/
├── metadata.json              # 完整实验元数据（config + env + results）
├── training_log.jsonl         # 增量训练日志（每 epoch 一行 JSON）
├── feedback.json              # 结构化训练反馈（status + diagnosis + suggestions）
├── exploration.json           # 探索历史（ExplorationTracker 持久化）
├── pipeline_checkpoint.json   # stage 断点续跑 checkpoint
├── checkpoints/
│   └── best-epoch={N}-val_loss={X}.ckpt   # 最佳模型
├── metrics/                   # CSVLogger 输出
│   └── metrics.csv
└── error_traceback.txt        # 仅失败时存在
```

## 断点续跑

Pipeline 失败后可从失败 stage 恢复，无需重跑已完成 stage：

```python
from senseframe.engine.runner.pipeline import Pipeline

# 续跑：从 output_dir/pipeline_checkpoint.json 恢复
pipeline, completed = Pipeline.resume("runs/<实验目录>")
# completed 为已完成 stage 名列表
```

checkpoint 文件结构：
```json
{
    "completed_stages": ["validate", "preflight", "resolve", "load", "build"],
    "trial_id": "trial_0001",
    "timestamp": "2026-01-01T12:00:00",
    "failed_stage": "train"
}
```

## 失败清理机制

- **配置错误**：不创建输出目录，无残留
- **运行时错误**：半成品目录会被自动清理，但保留 `error_traceback.txt` + `pipeline_checkpoint.json`（用于断点续跑）
- **成功**：完整保留所有输出

## 常见问题

### Q: 数据集目录应该放在哪里？

A: 默认 `data_root` 为 `CSI_DATASETS/Data`。需在该目录下创建对应子目录：
- `CSI_DATASETS/Data/UT_HAR/`
- `CSI_DATASETS/Data/NTU-Fi_HAR/`
- `CSI_DATASETS/Data/NTU-Fi-HumanID/`
- `CSI_DATASETS/Data/Widardata/`

也可通过 `scene.data_root` 指定自定义路径。

### Q: 如何恢复中断的训练？

A: 通过 `scene.params.resume` 指定 checkpoint 路径：

```yaml
scene:
  params:
    resume: runs/ResNet18_UT_HAR_data_20240101_120000_123/checkpoints/best-epoch=50-val_loss=0.5.ckpt
```

Pipeline stage 级断点续跑用 `Pipeline.resume(output_dir)`。

### Q: 自监督模式为什么 num_classes 要写 14？

A: 自监督微调阶段使用 NTU-Fi-HumanID 数据集（14 类），框架内部硬编码 `num_classes=14`。YAML 中声明 14 是为了 schema 校验通过，实际值由框架覆盖。

### Q: epochs_trained 为什么比配置的 epochs 少？

A: 可能原因：
1. `early_stopping` 触发，提前停止
2. `max_time` 兜底超时
3. 路由级别限制 `max_epochs`（如 `cpu_minimal` 限制 50，`cpu_standard` 限制 200）
4. 自监督模式只记录 Stage 2 的轮数
