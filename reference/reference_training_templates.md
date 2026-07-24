# YAML 配置模板

## 监督学习模板（UT_HAR + ResNet18）

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  learning_mode: supervised
  params:
    metrics: [accuracy, macro_f1, micro_f1]
    average: macro

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 200
  learning_rate: 0.001
  batch_size: 64
  optimizer: adam
  seed: 42

output_dir: runs
save_model: true
```

## 自监督学习模板（NTU-Fi_HAR + ResNet18）

```yaml
# 注意：自监督模式仅支持 dataset=NTU-Fi_HAR
# Stage 1: 在 NTU-Fi_HAR 上无监督预训练（EntLoss）
# Stage 2: 在 NTU-Fi-HumanID 上监督微调（14 类，CrossEntropyLoss）
scene:
  name: wifi_csi
  dataset: NTU-Fi_HAR
  model_id: ResNet18
  learning_mode: self_supervised
  params:
    metrics: [accuracy, macro_f1]
    average: macro
    self_supervised_epochs: 100    # Stage 1 轮数

input_features:
  - name: csi
    type: csi
    shape: [3, 114, 500]

output_features:
  - name: action
    type: category
    num_classes: 14                # NTU-Fi-HumanID 类别数（框架自动派生，详见 reference_datasets_models.md）

trainer:
  epochs: 30                       # Stage 2 监督微调轮数
  learning_rate: 0.001
  batch_size: 64
  optimizer: adam
  seed: 42

output_dir: runs
save_model: true
```

## HPO 超参搜索模板

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  learning_mode: supervised
  params:
    metrics: [accuracy, macro_f1]

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 50
  batch_size: 64

hpo:
  enabled: true
  n_trials: 20
  sampler: tpe
  pruner: median
  metric: val_loss
  direction: minimize

output_dir: runs
```

## 最小配置模板

仅包含必需字段，其余用默认值：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: MLP

input_features:
  - name: csi
    type: csi

output_features:
  - name: action
    type: category
    num_classes: 7
```

## CPU 轻量配置模板

适合 CPU 环境快速验证：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: MLP
  learning_mode: supervised
  params:
    metrics: [accuracy]

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping: 10
  max_time: "00:30:00:00"          # 30 分钟兜底

output_dir: runs
save_model: true
```

## GPU 高性能配置模板

适合高端 GPU 充分训练：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet101
  learning_mode: supervised
  params:
    metrics: [accuracy, macro_f1, micro_f1]
    average: macro
    gpu: 0                          # GPU 隔离

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 200
  batch_size: 256
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.01
  seed: 42

output_dir: runs
save_model: true
```

## Widar 数据集配置模板

Widar 数据集（22 类，输入形状 (22, 20, 20)）：

```yaml
scene:
  name: wifi_csi
  dataset: Widar
  model_id: ResNet18
  learning_mode: supervised
  params:
    metrics: [accuracy, macro_f1]
    average: macro

input_features:
  - name: csi
    type: csi
    shape: [22, 20, 20]

output_features:
  - name: action
    type: category
    num_classes: 22

trainer:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  optimizer: adam
  seed: 42

output_dir: runs
save_model: true
```

## 分布式训练模板

多卡 DDP 训练，适合大规模数据集或大模型：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet101
  learning_mode: supervised

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 200
  batch_size: 128                   # 单卡 batch_size，总 batch = 128 * devices
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.01
  gradient_clip_val: 1.0            # 梯度裁剪
  accumulate_grad_batches: 2        # 梯度累积（等效 batch=256）

# 分布式训练配置（YAML 顶层）
devices: 2                          # GPU 数量（或 "auto" 自动检测）
strategy: "ddp"                     # 分布式策略
sync_batchnorm: true                # 同步 BatchNorm

output_dir: runs
save_model: true
```

多节点训练示例：

```yaml
# 在每个节点上运行，通过 MASTER_ADDR/MASTER_PORT 环境变量协调
devices: 4                          # 每节点 GPU 数
strategy: "ddp"
num_nodes: 4                        # 节点数（总 GPU = 4 * 4 = 16）
sync_batchnorm: true
```

## HPO 持久化与断点续搜模板

长时间 HPO 搜索，支持断点续搜与结果导出：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  learning_mode: supervised

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 50                        # HPO 时用少 epochs 加速

hpo:
  enabled: true
  n_trials: 100                     # 总 trial 数
  sampler: tpe
  pruner: median
  metric: macro_f1
  direction: maximize
  # 持久化与断点续搜
  storage: "sqlite:///runs/hpo_study.db"   # SQLite 持久化
  study_name: "resnet18_ut_har_v1"          # study 名称
  load_if_exists: true                       # 断点续搜
  export_path: "runs/hpo_result.json"       # 结果导出（含 summary）
  timeout: 7200.0                           # 2 小时超时

output_dir: runs
save_model: true
```

## 混合精度训练模板

bf16-mixed 精度训练（需 Ampere+ GPU）：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  learning_mode: supervised

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 200
  batch_size: 128
  learning_rate: 0.001
  optimizer: adamw

# 混合精度配置
mixed_precision: "bf16-mixed"       # 或 "16-mixed" / "32"

output_dir: runs
save_model: true
```

## 学习率调度器模板

启用 cosine 余弦退火或 step 阶梯衰减：

```yaml
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: ResNet18
  learning_mode: supervised

input_features:
  - name: csi
    type: csi
    shape: [1, 250, 90]

output_features:
  - name: action
    type: category
    num_classes: 7

trainer:
  epochs: 200
  batch_size: 64
  learning_rate: 0.001
  scheduler: cosine                 # 或 step / null（不启用）
  # cosine: T_max 动态取 epochs（fallback 50）
  # step:   step_size 动态取 epochs//3（fallback 30），gamma=0.1

output_dir: runs
save_model: true
```
