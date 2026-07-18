# 自监督训练范式

## 两阶段训练流程

senseframe 自监督采用 AutoFi 风格的两阶段训练：

### Stage 1 — 自监督预训练

- **损失**：`EntLoss`（KL + EH + HE + KDE）
- **训练参数**：全部参数（encoder + projector）
- **数据**：NTU-Fi_HAR 无监督数据
- **轮数**：`scene.params.self_supervised_epochs`（默认 100）
- **优化器**：AdamW（全部参数）
- **验证**：不记录验证指标（`limit_val_batches=0`）

### Stage 2 — 监督微调

- **损失**：`CrossEntropyLoss`
- **训练参数**：只训练 classifier 参数（encoder 冻结）
- **数据**：NTU-Fi-HumanID 监督数据（14 类）
- **轮数**：`trainer.epochs`
- **优化器**：Adam（仅 classifier 参数）
- **验证**：记录验证指标（accuracy, macro_f1 等）

## EntLoss 损失函数

`EntLoss` 由四个组件构成：

| 组件 | 全称 | 作用 |
|------|------|------|
| KL | Kullback-Leibler 散度 | 对齐两个增强视图的预测分布 |
| EH | Entropy of Hypothesis | 最小化每个样本的预测熵（鼓励自信预测） |
| HE | Hypothesis Entropy | 最大化批次级平均分布的熵（鼓励类别均衡） |
| KDE | Kernel Density Estimation | 余弦相似度损失（特征级对齐） |

最终损失：`loss = KDE × 100 + (KL + (1 + lam1) × EH - lam2 × HE)`

### EntLoss 超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 1.0 | 温度参数（sharpening） |
| `eps` | 1e-5 | 数值稳定性常数 |
| `lam1` | 0.0 | EH 权重系数 |
| `lam2` | 0.5 | HE 权重系数 |

通常用默认值，高级用户可通过 `scene.params` 透传调整。

## SelfSupervisedModule 配置

通过 `scene.params` 透传参数：

```yaml
scene:
  name: wifi_csi
  dataset: NTU-Fi_HAR
  model_id: ResNet18
  learning_mode: self_supervised
  params:
    self_supervised_epochs: 100   # Stage 1 轮数
    metrics: [accuracy, macro_f1]
    # EntLoss 超参（可选，通常用默认值）
    # tau: 1.0
    # eps: 1e-5
    # lam1: 0.0
    # lam2: 0.5
```

## 数据加载

自监督模式下 `WiFiCSIContainer.load_dataset` 返回 `DatasetBundle`，填充 3 个字段：

```python
bundle = DatasetBundle(
    unsupervised=unsupervised_dataset,        # NTU-Fi_HAR 无监督数据（Stage 1 用）
    supervised_finetune=supervised_dataset,   # NTU-Fi-HumanID 训练集（Stage 2 用）
    test=test_dataset,                        # NTU-Fi-HumanID 测试集（Stage 2 验证用）
)
```

- `unsupervised`：来自 `NTU-Fi_HAR`（无监督预训练数据，input_shape=(3, 114, 500)）
- `supervised_finetune`：来自 `NTU-Fi-HumanID`（由 `DatasetSpec.supervised_source` 声明，14 类）
- `test`：来自 `NTU-Fi-HumanID` 测试集

**DatasetBundle 填充契约**（自监督模式）：

| 字段 | 填充规则 |
|------|----------|
| `train` | forbidden（不填） |
| `test` | required |
| `val` | optional |
| `unsupervised` | required |
| `supervised_finetune` | required |

可通过 `bundle.validate_filling("self_supervised")` 校验填充是否合规。

## 关键约束

1. **数据集约束**：`dataset` 必须为 `NTU-Fi_HAR`，否则抛 `ValueError`
   ```
   Self-supervised mode only supports dataset 'NTU-Fi_HAR', got 'X'.
   Self-supervised uses NTU-Fi_HAR for unsupervised pretraining
   and NTU-Fi-HumanID for supervised fine-tuning.
   ```

   `NTU-Fi_HAR` 的 `DatasetSpec.supervised_source="NTU-Fi-HumanID"`，Stage 2 自动加载该数据集进行微调。

2. **类别数约束**：`num_classes` 在 `output_features` 中声明 14（NTU-Fi-HumanID 类别数）
   - 框架内部硬编码 `num_classes=14`，忽略 YAML 中的声明值
   - 但 YAML 中仍需声明 `num_classes`（schema 校验要求 >= 2）

3. **input_shape**：`NTU-Fi_HAR` 与 `NTU-Fi-HumanID` 均为 `(3, 114, 500)`，`input_features.shape` 声明 `[3, 114, 500]`

4. **epochs_trained 含义**：`output.training.epochs_trained` 只记录 Stage 2 的轮数
   - Stage 1 不记录验证指标，不影响 `epochs_trained`
   - 若需确认 Stage 1 是否完成，查看 `training_log.jsonl` 中的日志

5. **优化器切换**：
   - Stage 1：AdamW（训练全部参数）
   - Stage 2：Adam（仅训练 classifier 参数）

6. **权重传递机制**：
   - **单次运行内（`learning_mode=self_supervised`）**：Stage 1 和 Stage 2 复用同一个 `ctx.model` 对象（`SelfSupervisedModule` 包装），通过 `phase` 属性切换行为。Stage 1 训练的 encoder 权重通过 Python 对象引用自然延续到 Stage 2，无需显式 save/load。这是框架支持的标准路径。
   - **跨运行加载（先 self_supervised 保存 checkpoint，再 supervised 加载）**：**不支持**。`_Parrallel` 模型与监督模型（`NTU_Fi_*`）的 forward 签名（`(x1, x2, flag)` vs `(x)`）和 state_dict key 结构完全不同，框架不提供 key 映射函数。如需跨运行迁移 encoder 权重，需在外部 SenseFi 项目层面实现。

## 数据增强

Stage 1 使用 `gaussian_noise` 数据增强生成两个增强视图：

```python
def gaussian_noise(x, std=0.1):
    """对输入添加高斯噪声，生成增强视图。"""
    return x + torch.randn_like(x) * std
```

两个增强视图分别通过 encoder + projector，输出特征用于计算 EntLoss。
