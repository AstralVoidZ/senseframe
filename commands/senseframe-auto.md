---
description: "SenseFrame 多轮自主调优：基于 feedback 的迭代训练与配置优化"
subtask: false
---

<!--

command: /senseframe-auto
场景: 中等 (15-30 min)
目的: 验证多轮训练中 Agent 主导的配置调优闭环

用法:
  /senseframe-auto                          # 默认 UT_HAR_data + ResNet18 + 3 轮
  /senseframe-auto NTU-Fi_HAR MLP 5         # 自定义数据集 + 模型 + 5 轮
  /senseframe-auto --auto-lr                # 启用自动 LR 标定

$1 = 数据集名 (默认 UT_HAR_data)
$2 = 模型名 (默认 ResNet18)
$3 = 轮数 (默认 3)
$ARGUMENTS = 原始参数字符串（可含 --auto-lr）
-->

# SenseFrame 多轮自主调优

## Role

你是 AI Agent、机器学习、AutoML、WiFi CSI 信号处理四领域专家。
以专家身份执行多轮迭代训练，每轮基于上一轮 feedback 调整配置，验证探索-反馈回路。

## Objective

用 SenseFrame 完成 N 轮迭代训练，每轮基于 `feedback.json` + `exploration recommend` 调整配置，
输出多轮对比报告，验证 Agent 主导的调优闭环。

**参数**:
- 数据集: `$1`（默认 `UT_HAR_data`）
- 模型: `$2`（默认 `ResNet18`）
- 轮数: `$3`（默认 `3`）
- 自动 LR: `$ARGUMENTS` 含 `--auto-lr` 时启用

## Context

- 工作目录: `.`（已部署 SenseFrame 的测试目录）
- SKILL `senseframe` 已加载，API 文档在 `.opencode/skills/senseframe/SKILL.md`
- 数据集根: `resource/CSI_DATASETS/`
- **报告输出目录**: `report/`
- **运行目录**: `runs/auto_<YYYYMMDD_HHMMSS>/`

## 设计哲学

SenseFrame 只专注单次训练做到最好。多轮调优是 Agent 职责，不耦合进框架。
本 command 用现有 `senseframe-train` + `senseframe exploration recommend` + Python 脚本串联，
框架零改动。

## Execution Protocol

执行以下步骤，**禁止跳过任何步骤，禁止用 mock/dry-run 充数**。

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

### Step 1: 环境探测 + 基础配置生成

**Do**:
```bash
python -c "import senseframe, torch; print(f'sf={senseframe.__version__}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python scripts/generate_config.py --dataset $1 --model $2 --mode supervised --output configs/auto_base.yaml
```

**Gate**: 配置生成成功
**Introspect**:
- [ ] 生成的 epochs 是否为动态预算（基于 n_samples）而非硬编码 200？
- [ ] 若 `--auto-lr` 启用，base config 中 `auto_lr_find: true`？

### Step 2: 第 1 轮训练（基线）

**Do**:
```bash
python -m senseframe.cli experiment --config configs/auto_base.yaml --output-dir runs/auto_<ts>/round_0
```

**Gate**: 训练完成，`metadata.json` + `feedback.json` + `training_log.jsonl` 生成
**Introspect**:
- [ ] `metadata.json` 含 `best_epoch` / `best_model_score` / `epoch_utilization`？
- [ ] `feedback.json` 的 status 是否合理（converged/overfitting/underfitting）？
- [ ] `epoch_utilization` 值是多少？若 <0.3 说明预算过大，>0.9 说明预算不足
- [ ] `training_log.jsonl` 含 epoch 0（train_only）和 epoch N+1（final_eval）行？
- [ ] metrics.csv 与 training_log.jsonl 的 epoch 范围一致？

### Step 3: 读取 feedback + 获取推荐

**Do**:
```bash
# 读取 feedback
python -c "import json; fb=json.load(open('runs/auto_<ts>/round_0/feedback.json')); print(json.dumps(fb, indent=2, ensure_ascii=False))"

# 获取推荐
python -m senseframe.cli exploration recommend --output-dir runs/auto_<ts>/round_0 --task-type classification --top-k 1
```

**Gate**: 推荐输出含 strategy + reason
**Introspect**:
- [ ] 推荐的 strategy 是否与 feedback status 对应？（overfitting→weight_decay/dropout, underfitting→lr_scale/epochs_scale）
- [ ] 推荐链路是否可追溯？

### Step 4: 应用推荐 + 生成下一轮配置

**Do**:
用 Python 脚本读取推荐，应用到 base config 生成 `round_<N>.yaml`：

```python
import yaml, copy
base = yaml.safe_load(open("configs/auto_base.yaml"))
# 从 recommendations.json 读取推荐策略
rec = yaml.safe_load(open("runs/auto_<ts>/round_0/recommendations.json"))
strategy = rec[0]["strategy"] if rec else {}

config = copy.deepcopy(base)
# 策略应用规则
if "lr_scale" in strategy:
    current_lr = config["trainer"].get("learning_rate", 1e-3)
    config["trainer"]["learning_rate"] = current_lr * strategy["lr_scale"]
if "weight_decay" in strategy:
    config["trainer"]["weight_decay"] = strategy["weight_decay"]
if "epochs_scale" in strategy:
    config["trainer"]["epochs"] = int(config["trainer"]["epochs"] * strategy["epochs_scale"])
if "gradient_clip_val" in strategy:
    config["trainer"]["gradient_clip_val"] = strategy["gradient_clip_val"]

yaml.safe_dump(config, open(f"configs/auto_round_1.yaml", "w"), allow_unicode=True)
print(f"Round 1 config: {config['trainer']}")
```

**Gate**: `configs/auto_round_1.yaml` 生成
**Introspect**:
- [ ] 策略应用是否正确？（lr_scale 乘法，weight_decay 替换）
- [ ] 配置变更是否记录到 exploration_history？

### Step 5: 第 2~N 轮训练（迭代）

对每一轮重复 Step 2-4：
```bash
python -m senseframe.cli experiment --config configs/auto_round_<N>.yaml --output-dir runs/auto_<ts>/round_<N>
# 读取 feedback + 推荐 → 生成 round_<N+1>.yaml
```

每轮记录以下数据到对比表：
- Round 号
- LR / Weight Decay / Epochs
- Val Accuracy / Val Loss
- Feedback status
- Best Epoch / Epoch Utilization
- Training Duration

**Gate**: 所有轮次完成
**Introspect**:
- [ ] 每轮 feedback 是否基于 best epoch（而非 final epoch）？
- [ ] config_hash 是否每轮不同（配置确实变更）？
- [ ] epoch_utilization 是否在合理范围（0.2~0.8）？

### Step 6: 生成对比报告

**Do**:
汇总所有轮次数据，生成 `report/auto_<dataset>_<model>_<YYYYMMDD_HHMMSS>.md`：

```markdown
# SenseFrame Auto-Tune Report

## 配置
- 数据集: $1
- 模型: $2
- 轮数: $3
- auto_lr: enabled/disabled

## 多轮对比

| Round | LR | Weight Decay | Epochs | Val Acc | Val Loss | Feedback | Best Epoch | Epoch Util | Duration |
|-------|-----|-------------|--------|---------|----------|----------|------------|-----------|----------|
| 0 | 1e-3 | 1e-4 | 37 | 0.911 | 0.239 | converged | 6 | 0.162 | 26.97s |
| 1 | 5e-4 | 1e-4 | 37 | 0.925 | 0.210 | converged | 8 | 0.216 | 28.12s |
| 2 | 2.5e-4 | 1e-4 | 37 | 0.931 | 0.198 | converged | 10 | 0.270 | 29.45s |

## 最佳配置
- Round X: val_accuracy=0.XXX
- config: runs/auto_<ts>/round_X/config.yaml

## 推荐链路
- R0 feedback: <status> → recommend: <strategy>
- R1 feedback: <status> → recommend: <strategy>

## 数据通路验证
- [ ] metadata.json 含 best_epoch/best_model_score/epoch_utilization
- [ ] training_log.jsonl 含 epoch 0 (train_only) + epoch N+1 (final_eval)
- [ ] metrics.csv 与 training_log.jsonl epoch 范围一致
- [ ] feedback 基于 best epoch（非 final epoch）

## 发现的问题
（记录测试中发现的真实问题）
```

**Gate**: 报告生成，含完整对比表 + 数据通路验证

## 自动 LR 标定（可选）

若 `$ARGUMENTS` 含 `--auto-lr`：
- 在 base config 中设置 `trainer.auto_lr_find: true`
- 每轮训练前 stage_train 会自动用独立 Trainer 跑 LR Range Test
- tune 结果（建议 LR）会写入 `metadata.config.learning_rate`
- 报告中记录每轮的 tune 建议 LR vs 实际使用 LR

## 注意事项

1. **框架不耦合调优逻辑**：本 command 的策略应用脚本由 Agent 编写，不是框架代码
2. **每轮独立 output_dir**：`runs/auto_<ts>/round_<N>/`，避免覆盖
3. **exploration.json 继承**：每轮训练的 exploration_history 应包含前轮记录
4. **epoch_utilization 分析**：若 <0.3，下轮可降低 epochs 预算；若 >0.9，下轮应提高
5. **不自动调整 patience**：patience 是用户决策，框架只提供数据供 Agent 分析

## 训练波动调优指导（CSI 数据特性）

CSI 数据训练存在固有的 val_accuracy 波动（batch 间分布差异大，CSI 信号受环境影响大）。
典型表现：epoch 5 val_accuracy 从 0.815 跌至 0.521，epoch 8 从 0.922 跌至 0.768。
这不是框架缺陷，而是 CSI 数据特性。Agent 在多轮调优中可通过以下策略缓解：

### 诊断信号

从 `training_log.jsonl` 读取 val_accuracy 序列，计算波动幅度：
```python
import json
logs = [json.loads(l) for l in open("training_log.jsonl")]
val_accs = [e["val_accuracy"] for e in logs if e.get("val_accuracy") is not None and e.get("phase", "train_val") != "final_eval"]
# 计算相邻 epoch 的 val_accuracy 差值
deltas = [abs(val_accs[i+1] - val_accs[i]) for i in range(len(val_accs)-1)]
max_delta = max(deltas) if deltas else 0
avg_delta = sum(deltas) / len(deltas) if deltas else 0
print(f"Max delta: {max_delta:.3f}, Avg delta: {avg_delta:.3f}")
```

### 调优策略

| 波动幅度 | 诊断 | 建议策略 | 配置调整 |
|---------|------|---------|---------|
| max_delta > 0.15 | 严重波动 | 启用梯度裁剪 | `gradient_clip_val: 1.0` |
| max_delta > 0.10 | 中等波动 | 增大 batch_size 平滑梯度 | `batch_size: 128`（若显存允许） |
| avg_delta > 0.05 | 持续抖动 | 降低学习率 | `learning_rate *= 0.5` |
| 波动可接受 | 正常 | 保持当前配置 | 无需调整 |

**重要**：波动不一定是坏事。Early Stopping 的 patience=5 已能容忍这种波动，最终在 best epoch 找到全局最优。
Agent 的职责是判断波动是否导致 Early Stopping 误触发（best_epoch 过早），而非消除所有波动。
