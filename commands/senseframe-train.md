---
description: "SenseFrame 简单场景：单模型监督训练 + 四视角自省测试"
subtask: false
---

<!--

command: /senseframe-train
场景: 简单 (5-15 min)
目的: 验证声明式训练路径 + 配置校验 + 预检 + 基础训练 + 错误码

用法:
  /senseframe-train                     # 默认 UT_HAR_data + ResNet18
  /senseframe-train NTU-Fi_HAR MLP      # 自定义数据集 + 模型

$1 = 数据集名 (默认 UT_HAR_data)
$2 = 模型名 (默认 ResNet18)
$ARGUMENTS = 原始参数字符串
-->

# SenseFrame 简单训练测试

## Role

你是 AI Agent、机器学习、AutoML、WiFi CSI 信号处理四领域专家。
以专家身份执行真实训练，边训练边审视框架，发现真实问题。

## Objective

用 SenseFrame 声明式路径完成一次 WiFi CSI 监督训练，验证 8-stage pipeline 基础功能，
产出结构化测试报告。

**参数**:
- 数据集: `$1`（默认 `UT_HAR_data`）
- 模型: `$2`（默认 `ResNet18`）

## Context

- 工作目录: `.`（已部署 SenseFrame 的测试目录）
- SKILL `senseframe` 已加载，API 文档在 `.opencode/skills/senseframe/SKILL.md`
- 数据集根: `CSI_DATASETS/`
- Python 环境: 已 `pip install -r requirements.txt`

## Execution Protocol

执行以下步骤，每步完成后填写对应的自省检查点。
**禁止跳过任何步骤，禁止用 mock/dry-run 充数。**

### Step 1: 环境探测

**Do**:
```bash
python -c "import senseframe, torch; print(f'sf={senseframe.__version__}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -m senseframe.cli probe
```

**Output**: 环境信息（Python/torch/CUDA/显存）
**Gate**: `import senseframe` 成功且 `__version__` 可读

### Step 2: 配置生成与校验

**Do**:
```bash
python scripts/generate_config.py --dataset $1 --model $2 --mode supervised --output configs/test.yaml
python scripts/validate_config.py --config configs/test.yaml
```

**Output**: 配置文件 + 校验结果
**Gate**: 配置生成成功，校验通过或有可读警告

**Introspect**:
- [ ] 默认 epochs/batch_size/lr 是否合理？
- [ ] 默认 loss/metric 是否匹配 task_type？
- [ ] 校验错误（如有）是否含 error_code？
- [ ] **方案 E 验证**：默认 `TrainerConfig` 是否含 `weight_decay=1e-4` / `early_stopping=5` / `scheduler="cosine"` / `early_stopping_min_delta=0.001`？（检查生成的 YAML 中 trainer 段）

### Step 3: 预检

**Do**:
```bash
python -m senseframe.cli experiment --config configs/test.yaml --dry-run
```

**Output**: 预检报告（资源路由级别、模型选择、显存评估、数据存在性）
**Gate**: 预检完成，输出结构化

**Introspect**:
- [ ] 资源路由是否真正基于硬件探测？（对比 CPU/GPU 下输出差异）
- [ ] 预检输出对 Agent 友好？（JSON vs 自由文本）
- [ ] 数据存在性校验是否覆盖数据集契约？

### Step 4: 执行训练

**Do**:
```bash
python -m senseframe.cli experiment --config configs/test.yaml
```

**Output**: 训练产物（model.pth + metadata.json + training_log + **manifest.json**）
**Gate**: 训练完成，`output.final_eval["val_accuracy"]` 非零

**Do**: 训练后立即验证产物溯源（RFC-004 方案 G）

```python
import senseframe as sf
from pathlib import Path

# 加载 manifest 并校验产物完整性
manifest = sf.load_manifest(Path("runs/<实验目录>/manifest.json"))
print(f"artifacts: {len(manifest.artifacts)}")
report = sf.verify_artifacts(manifest)
assert report["verified"] == report["total"], f"产物校验失败: {report}"
print(f"全部 {report['total']} 个产物校验通过")
```

**Introspect — stage by stage**:

| Stage | 检查点 |
|-------|--------|
| validate | 错误信息是否结构化（error_code）？ |
| load | CSI reshape/normalization 是否正确？数据画像字段是否完整？ |
| build | 模型输入形状是否匹配数据？ |
| train | loss 是否收敛？日志对 Agent 可读？ |
| eval | 指标是否充分？feedback（overfit/underfit/converged）是否准确？**方案 C**：`final_eval` 是否含 `val_accuracy`/`val_loss`/`val_macro_f1` 等 `val_` 前缀字段？ |
| export | 导出格式是否正确？metadata 是否完整？**方案 G**：`manifest.json` 是否生成？`verify_artifacts()` 是否通过？ |
| **方案 F** | 训练结束后 `ctx.trainer/module/model` 是否为 None（`release_resources` 已执行）？ |

### Step 5: 后处理

**Do**:
```bash
python scripts/postprocess.py --output-dir runs/<实验目录> --models-dir models --result-dir result --eval-script eval.py
```

**Output**: 最终交付物（模型权重 + 推理脚本 + manifest）
**Gate**: 产物齐全

## Introspection Protocol

每个检查点必须回答，不得留空或写空话：

1. **AI/Agent 视角** [1-5]: API 是否可程序化？错误信息是否含 error_code？
2. **ML 视角** [1-5]: 策略是否合理？训练是否收敛？评估是否充分？
3. **AutoML 视角** [1-5]: 端到端自动化程度？资源路由是否真实？
4. **CSI 视角** [1-5]: 数据预处理是否正确？模型是否适配 CSI 特征？

**纪律**: 评分有区分度，发现问题必须扣分；观察必须引用具体日志/字段，禁止"看起来不错"。

## Output Contract

执行完毕后，输出以下结构化报告（markdown 格式）：

```markdown
# SenseFrame 测试报告：简单训练

## 执行摘要
- 环境: <Python/torch/CUDA>
- 数据集: <$1> | 模型: <$2>
- 状态: <成功/失败> | 耗时: <min>

## 训练结果
- val_accuracy: <值，读自 `output.final_eval["val_accuracy"]`>
- val_loss: <值，读自 `output.final_eval["val_loss"]`>
- epochs: <actual/planned>
- 早停: <yes/no>
- 训练耗时: <s>
- manifest 校验: <verified/total>

## 自省评分矩阵
| Stage | AI/Agent | ML | AutoML | CSI | 平均 |
|-------|----------|----|--------|-----|------|
| validate | x | x | x | x | x.x |
| preflight | x | x | x | x | x.x |
| load | x | x | x | x | x.x |
| build | x | x | x | x | x.x |
| train | x | x | x | x | x.x |
| eval | x | x | x | x | x.x |
| export | x | x | x | x | x.x |

## 关键发现（按严重度排序）
1. [严重/中等/轻微] <问题 + 复现步骤 + 影响>
2. ...

## 改进建议（按优先级排序）
1. [P0/P1/P2] <具体建议 + 影响范围>

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
```

## Constraints

- 禁止伪造结果：训练必须真实执行
- 禁止掩盖问题：发现的问题必须记录
- 禁止空话：观察必须引用具体输出
- 禁止全 5 分：评分必须有区分度
