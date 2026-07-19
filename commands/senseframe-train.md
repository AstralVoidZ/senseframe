---
description: "SenseFrame 简单场景：单模型监督训练 + 四视角自省测试"
subtask: false
---

<!--

本文件是 Agent 提示词，部署到 .opencode/.claude/.agents/commands/ 供 AI Agent CLI 工具调用。
slash 命令 /senseframe-train 由 Agent CLI 工具（opencode/Claude Code）解析，不是 SenseFrame CLI 子命令。
SenseFrame CLI 子命令清单见 SKILL.md 或 `python -m senseframe.cli --help`。

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
- 数据集根: `resource/CSI_DATASETS/`（如未部署，可改 `--data-root` 或软链；命令中所有路径均可替换为实际数据根）
- Python 环境: 已 `pip install -e '.[eeg,radio,dev]'`
- **报告输出目录**: `report/`（不存在时自动创建）
- **报告命名格式**: `report/train_<dataset>_<model>_<YYYYMMDD_HHMMSS>.md`
  - 例：`report/train_UT_HAR_data_ResNet18_20260706_143025.md`
  - 时间戳取训练开始时刻，避免重名覆盖

## Execution Protocol

执行以下步骤，每步完成后填写对应的自省检查点。
**禁止跳过任何步骤，禁止用 mock/dry-run 充数。**

### Step 0: 首次部署检查（环境检查集中化）

**目的**：确保 SenseFi 代码库路径、数据集目录、Python 环境、GPU 状态、磁盘空间全部就绪，避免到训练阶段才暴露 ImportError / OOM / 磁盘满等问题。

**Do**:
```bash
# === 环境检查清单（P2-6 集中化）===

# 1. SenseFi 路径（wifi_csi 场景硬性依赖）
echo "SENSEFRAME_SENSEFI_PATH=${SENSEFRAME_SENSEFI_PATH:-NOT_SET}"
test -d "${SENSEFRAME_SENSEFI_PATH:-/nonexistent}" && echo "SENSEFI_PATH: OK" || echo "SENSEFI_PATH: MISSING"

# 2. 数据集目录
echo "SENSEFRAME_DATA_ROOT=${SENSEFRAME_DATA_ROOT:-NOT_SET}"
ls resource/CSI_DATASETS/ 2>/dev/null || echo "CSI_DATASETS_NOT_FOUND"

# 3. Python 环境（torch/pytorch_lightning/senseframe 版本）
python -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"
python -c "import pytorch_lightning as pl; print(f'pytorch_lightning={pl.__version__}')"
python -c "import senseframe; print(f'senseframe={senseframe.__version__}')"

# 4. GPU 状态（可 GPU 时输出型号 + 显存）
python -c "import torch; print(f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# 5. 磁盘空间（至少 5GB 可用）
df -h . | awk 'NR==2{print "disk_free="$4}'

# 6. 检查 generate_config 生成的 data_root 是否为空
python scripts/generate_config.py --dataset $1 --model $2 --mode supervised --output /tmp/_check.yaml
python -c "import yaml; c=yaml.safe_load(open('/tmp/_check.yaml')); print(f'data_root={c[\"scene\"].get(\"data_root\",\"EMPTY\")}')"
```

**自动发现规则**：
- 若 `SENSEFRAME_SENSEFI_PATH` 已设置且路径存在 → 跳过，无需用户干预
- 若未设置，但当前目录下存在 `resource/SenseFi/` 或 `resource/WiFi-CSI-Sensing-Benchmark-main/` → **自动设为该路径**（`export SENSEFRAME_SENSEFI_PATH=$(pwd)/resource/<found_dir>`），继续执行
- 若未设置且找不到 → **停止并询问用户**：请提供 SenseFi 代码库路径

- 若 `resource/CSI_DATASETS/` 存在且含 `$1` 子目录 → 自动填入 `data_root: resource/CSI_DATASETS` 到配置
- 若不存在 → **停止并询问用户**：请提供数据集根路径

**Gate**: SenseFi 路径已设置 + data_root 非空 + 数据集目录存在
**Introspect**:
- [ ] `test -d "$SENSEFRAME_SENSEFI_PATH"` 返回 0（路径存在）？
- [ ] `python -c "import os; print(bool(os.environ.get('SENSEFRAME_SENSEFI_PATH')))"` 输出 True？
- [ ] `test -d "resource/CSI_DATASETS/$1"` 返回 0（数据集目录存在）？
- [ ] 若任何一项缺失，是否已询问用户而非静默继续？

### Step 1: 环境探测

**Do**:
```bash
python -c "import senseframe, torch; print(f'sf={senseframe.__version__}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -m senseframe.cli probe
```

**Output**: 环境信息（Python/torch/CUDA/显存）
**Gate**: `import senseframe` 成功（退出码 0）且 `__version__` 非空字符串
**Introspect**:
- [ ] `python -c "import senseframe; print(senseframe.__version__)"` 退出码 0？
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 输出 True/False（非异常）？
- [ ] GPU 可用时 `python -m senseframe.cli probe` 输出 JSON 含 `gpu_name`/`gpu_free_vram_mb`？

### Step 2: 配置生成与校验

**Do**:
```bash
python scripts/generate_config.py --dataset $1 --model $2 --mode supervised --output configs/test.yaml
python scripts/validate_config.py --config configs/test.yaml
```

**Output**: 配置文件 + 校验结果
**Gate**: 配置生成成功（退出码 0）+ 校验通过（无 ValueError）
**Introspect**:
- [ ] `test -f configs/test.yaml` 返回 0（文件存在）？
- [ ] `python scripts/validate_config.py --config configs/test.yaml` 退出码 0？
- [ ] YAML 中 `trainer.weight_decay`/`early_stopping`/`scheduler`/`early_stopping_min_delta` 字段存在？（方案 E 默认最佳实践）
- [ ] 校验错误（如有）异常类型为 `ValueError` 且消息含字段名？

### Step 3: 预检

**Do**:
```bash
python -m senseframe.cli experiment --config configs/test.yaml --dry-run
```

**Output**: 预检报告（资源路由级别、模型选择、显存评估、数据存在性 + config_semantics + dynamic_validation + training_contract）
**Gate**:
- `--dry-run` 退出码 0
- 输出可被 `python -m json.tool` 解析
- `report["status"]` == "ok"
- `report["dynamic_validation"]["status"]` == "passed"（非 "skipped"）
- `report["config_semantics"]` 数组存在且无 error 级失败
- `report["training_contract"]` 数组存在（动态校验成功后）

**Introspect**:
- [ ] `python -m senseframe.cli experiment --config configs/test.yaml --dry-run | python -m json.tool` 退出码 0（输出是合法 JSON）？
- [ ] `report["status"]` == "ok"？
- [ ] `report["dynamic_validation"]["status"]` == "passed"？（非 "skipped"）
- [ ] `report["dynamic_validation"]["checks"]` 包含 `forward_pass`/`output_shape_match`/`backward_pass`/`param_count_reasonable` 四项？
- [ ] `report["config_semantics"]` 包含 `early_stopping_within_epochs`/`batch_size_within_dataset`/`scheduler_epochs_compatible`/`deterministic_cuda_available` 四项？
- [ ] `report["training_contract"]` 包含 `loss_task_match`/`metrics_task_match`/`early_stopping_within_epochs` 三项？
- [ ] `report["plan"]["device"]`/`accelerator`/`devices`/`precision` 字段存在？
- [ ] `--dry-run --static-only` 时 `dynamic_validation.status` == "skipped"？（显式跳过验证）

### Step 4: 执行训练

**Do**:
```bash
python -m senseframe.cli experiment --config configs/test.yaml
```

**Output**: 训练产物（model.pth + metadata.json + training_log + **manifest.json**）
**Gate**:
- 训练完成（退出码 0）
- `output.final_eval["val_accuracy"]` > 0
- `manifest.json` 存在且 `verify_artifacts()` 全部通过

**Do**: 训练后立即验证产物溯源（RFC-004 方案 G）

```python
import senseframe as sf
from pathlib import Path

# verify_artifacts 接受 output_dir（含 manifest.json），返回 {产物名: hash 是否匹配}
# 注意：传入目录路径，不是 manifest 对象
output_dir = Path("runs/<实验目录>")
report = sf.verify_artifacts(output_dir)
total = len(report)
verified = sum(1 for ok in report.values() if ok)
print(f"产物校验: {verified}/{total} verified")
assert verified == total, f"产物校验失败: {report}"
print(f"全部 {total} 个产物校验通过")
```

**Introspect — stage by stage**:

| Stage | 检查点（可程序化验证） |
|-------|----------------------|
| validate | `python scripts/validate_config.py --config configs/test.yaml` 退出码 0？异常含字段名？ |
| preflight | `--dry-run` 输出 JSON？`dynamic_validation.status == "passed"`？`config_semantics` 无 error？ |
| load | `data_profile.json` 生成？`n_samples > 0`？`modality == "temporal"`（CSI 场景）？ |
| build | `dynamic_validation.checks` 中 `forward_pass.ok == True`？`output_shape == (batch, num_classes)`？`param_count_reasonable.ok == True`？ |
| train | ep0 `train_loss` 在 [0.5, 3.0]？`training_log.jsonl` 首行可被 `json.tool` 解析？early_stopping 触发时 `best_epoch < epochs`？ |
| eval | `final_eval` 含 `val_loss`/`val_accuracy`/`val_macro_f1`（val_ 前缀）？`feedback.status` 为 `converged`/`overfitting`/`underfitting` 之一？ |
| export | `manifest.json` 存在？`verify_artifacts()` 返回全 True？每个 artifact 含 `producer_stage`/`content_hash`/`size_bytes`？ |
| **方案 F** | 训练结束后 `ctx.trainer is None` 且 `ctx.module is None`（`release_resources` 已执行）？ |

### Step 5: 后处理

**Do**:
```bash
# postprocess.py 仅接受 --output-dir（P0-1.5 路径安全修复后所有产物均在 output_dir 内）
# 自动生成 eval.py 推理脚本到 output_dir/eval.py，metadata.config 含 data_root 供推理脚本使用
python scripts/postprocess.py --output-dir runs/<实验目录>
```

**Output**: 最终交付物（模型权重 + 推理脚本 + manifest）
**Gate**: 产物齐全

## Introspection Protocol

每个检查点必须回答，不得留空或写空话：

1. **AI/Agent 视角** [1-5]: API 是否可程序化？错误信息是否含 error_code？
2. **ML 视角** [1-5]: 策略是否合理？训练是否收敛？评估是否充分？
3. **AutoML 视角** [1-5]: 端到端自动化程度？资源路由是否真实？
4. **CSI 视角** [1-5]: 数据预处理是否正确？模型是否适配 CSI 特征？

### 评分标准（量化 rubric）

**5 分（满分）**：
- 所有检查项通过
- 输出结构化 JSON，含 error_code 字段
- 性能/精度达到预期
- 无任何 warning

**4 分（良好但有改进空间）**：
- 核心功能正常，但存在 1 个以下问题：
  - 某些检查项未覆盖（如 dynamic_validation skipped）
  - 输出非完全结构化（如部分自由文本）
  - 有 warning 但不影响功能
- **必须填写扣分理由**（引用具体日志/字段）

**3 分（及格但有明显问题）**：
- 核心功能正常，但存在 2-3 个问题
- **必须填写扣分理由 + 修复建议**

**2 分以下（不及格）**：
- 核心功能异常
- **必须标记 [严重] 并填写根因分析**

### 禁止行为
- 禁止不填扣分理由给 4 分
- 禁止全 5 分（必须有区分度）
- 禁止全 4 分无理由（CSI 视角零区分度）
- 禁止用"看起来不错"代替具体证据

**纪律**: 评分有区分度，发现问题必须扣分；观察必须引用具体日志/字段，禁止"看起来不错"。

## Output Contract

执行完毕后，**必须**将报告写入 `report/train_<dataset>_<model>_<YYYYMMDD_HHMMSS>.md`，
并在 stdout 输出报告路径。报告内容必须包含以下章节，**禁止省略**任何章节，
**禁止用"正常/没问题"等空话代替具体证据**。

```markdown
# SenseFrame 测试报告：简单训练

## 报告元数据
- 报告路径: `report/train_<dataset>_<model>_<YYYYMMDD_HHMMSS>.md`
- 生成时间: <ISO8601>
- 测试命令: `/senseframe-train <dataset> <model>`
- 框架版本: <senseframe.__version__>

## 执行摘要
- 环境: Python <version> | torch <version> | CUDA <available/version>
- 硬件: <CPU/GPU 型号 + 显存>
- 数据集: <$1> | 模型: <$2>
- 状态: <成功/失败/部分成功> | 总耗时: <min>
- 训练耗时: <s>

## 配置详情（Step 2 产物）
- 配置文件路径: `configs/test.yaml`
- trainer 段关键字段（必须列出实际值，便于复现）:
  - epochs: <值>
  - learning_rate: <值>
  - batch_size: <值>
  - optimizer: <值>
  - weight_decay: <值>（方案 E 字段，缺失则标记 [缺失]）
  - early_stopping: <值>（方案 E 字段，缺失则标记 [缺失]）
  - early_stopping_min_delta: <值>（方案 E 字段，缺失则标记 [缺失]）
  - scheduler: <值>（方案 E 字段，缺失则标记 [缺失]）
- scene 段: name=<>, dataset=<>, model_id=<>, learning_mode=<>
- input_features: shape=<>
- output_features: num_classes=<>
- 校验结果: <通过/警告列表/错误列表>

## 预检详情（Step 3 产物）
- 资源路由级别: <L0-L4>
- 路由配置: <device/batch_size/num_workers/precision 等实际生效值>
- 模型选择: <model_id + 选择理由（如注册表命中）>
- 显存评估: <预估/实际>
- 数据存在性: <数据集路径 + 是否存在>
- 预检输出格式: <JSON/自由文本>

## 训练详情（Step 4 产物）
- output_dir: <实际训练输出目录路径>
- 训练日志关键片段（粘贴 3-5 行 epoch 末尾日志，含 val_* 指标）:
  ```
  <epoch 17: train_loss=0.012 val_loss=0.095 val_accuracy=0.971 val_macro_f1=0.964>
  <epoch 18: ...>
  <epoch 25: ...>
  <epoch 26: Early stopping triggered>（如有）
  ```
- 训练结果指标（必须读自 `output.final_eval`）:
  - val_accuracy: <值>（best epoch <N>: <值>）
  - val_loss: <值>（best: <值>）
  - val_macro_f1: <值>（best: <值>）
  - 其他 val_* 字段: <列出>
- epochs: <actual>/<planned>
- 早停: <yes/no>（patience=<>, best_epoch=<>, triggered_at_epoch=<>）
- 训练耗时: <s>

## 产物清单（Step 5 产物）
- 产物目录: `<output_dir>`
- manifest.json 路径: `<output_dir>/manifest.json`
- 产物校验结果（verify_artifacts 返回值）:
  | 产物名 | kind | 校验结果 | 备注 |
  |--------|------|---------|------|
  | model_weights | model | ✓/✗ | <如 ✗ 注明原因：missing/hash_mismatch> |
  | data_profile | profile | ✓/✗ | |
  | metrics | metrics | ✓/✗ | |
  | config | config | ✓/✗ | |
  | training_log | log | ✓/✗ | |
  | feedback | feedback | ✓/✗ | |
  | env_snapshot | log | ✓/✗ | |
  | model_metadata | metadata | ✓/✗ | |
  - verified/total: <N/M>
- 后处理产物: <models/ + eval.py + result/ 路径>

## Stage 执行日志（按 stage 粘贴关键输出，每 stage 5-10 行）
### validate
```
<实际日志输出>
```
### preflight
```
<实际日志输出>
```
### load
```
<实际日志输出>
```
### build
```
<实际日志输出>
```
### train
```
<实际日志输出（含 epoch 进度 + val_* 指标）>
```
### eval
```
<实际日志输出>
```
### export
```
<实际日志输出>
```

## 自省评分矩阵

> **填写规则**：
> - 每个单元格评分必须对应"关键扣分原因"列的具体理由
> - 若评 4 分必须填写扣分理由（引用具体日志/字段），否则视为违规
> - 若评 5 分必须确认该 stage 无任何 warning/error
> - CSI 视角不允许全 stage 相同评分（需有区分度）

| Stage | AI/Agent | ML | AutoML | CSI | 平均 | 关键扣分原因 |
|-------|----------|----|--------|-----|------|------------|
| validate | x | x | x | x | x.x | <如"error_code 缺失"> 或 **满分（无扣分）** |
| preflight | x | x | x | x | x.x | <如"dynamic_validation skipped"> 或 **满分（无扣分）** |
| load | x | x | x | x | x.x | <如"CSI 误判为 image"> 或 **满分（无扣分）** |
| build | x | x | x | x | x.x | <如"shape 不匹配未报错"> 或 **满分（无扣分）** |
| train | x | x | x | x | x.x | <如"日志格式混乱"> 或 **满分（无扣分）** |
| eval | x | x | x | x | x.x | <如"feedback 字段空"> 或 **满分（无扣分）** |
| export | x | x | x | x | x.x | <如"manifest 校验失败"> 或 **满分（无扣分）** |

## 关键发现（按严重度排序）
每个发现必须包含：复现命令 + 实际输出 + 期望输出 + 影响范围 + 严重度

1. **[严重]** <问题标题>
   - 复现命令: `<完整命令>`
   - 实际输出: `<粘贴实际输出>`
   - 期望输出: `<应该是什么>
   - 根因分析: <代码位置 + 逻辑错误>
   - 影响: <对哪些功能/用户/场景有影响>
   - 建议修复: <具体修复方向>

2. **[中等]** <问题标题>
   - 复现命令: `<完整命令>`
   - 实际输出: `<粘贴>`
   - 期望输出: `<应该是什么>`
   - 影响: <...>

3. **[轻微]** <问题标题>
   - ...（同上格式）

## 改进建议（按优先级排序）
每条建议必须含：优先级 + 具体修改点 + 影响文件/模块 + 预期收益

1. **[P0]** <建议>
   - 修改文件: `<file:line>`
   - 修改内容: <具体改什么>
   - 预期收益: <修复后效果>

2. **[P1]** <建议>
   - ...

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
- 下一步建议: <如"修复 P0 后重测 / 切换数据集验证泛化性">
```

---

## 常见失败模式与诊断（P2-5）

> Agent 在评分时遇到以下症状，必须按对应严重度标记并引用具体日志。

### F1: dynamic_validation skipped
- **症状**: `report["dynamic_validation"]["status"] == "skipped"`
- **根因**: 使用 `--dry-run --static-only`（显式跳过）或静态校验失败
- **修复**: 移除 `--static-only` 或修复静态校验错误后重跑
- **严重度**: [轻微]（显式跳过时）/ [中等]（静态失败导致跳过时）

### F2: 评分零区分度
- **症状**: 所有 stage 评 4 分，CSI 视角全 4 分，无扣分理由
- **根因**: 评分纪律未量化，Introspect 检查点过宽泛
- **修复**: 使用量化 rubric + 具体化检查点（每个 4 分必须填理由）
- **严重度**: [中等]（影响测试报告可信度）

### F3: preflight 检查项不全
- **症状**: `checks` 数组仅 7 项，缺模型契约/训练契约/数据契约
- **根因**: preflight 仅做存在性检查，缺语义校验
- **修复**: 实施 Preflight 增强方案（见 `docs/issues/preflight_enhancement_plan.md`）
- **严重度**: [中等]

### F4: config_semantics 有 error 级失败
- **症状**: `report["config_semantics"]` 中某项 `severity == "error"`
- **常见原因**:
  - `batch_size > n_samples`（CONFIG_BATCH_SIZE_TOO_LARGE）
  - `scheduler` 与 `epochs` 不兼容（CONFIG_SCHEDULER_INCOMPATIBLE）
- **修复**: 调整配置参数
- **严重度**: [严重]（训练会失败或无意义）

### F5: model_contract 前向失败
- **症状**: `report["dynamic_validation"]["checks"]` 中 `forward_pass.ok == False`
- **根因**: 模型架构与输入 shape 不匹配
- **修复**: 检查 `input_features` 配置和模型架构
- **严重度**: [严重]

### F6: data_contract 类别缺失
- **症状**: `report["data_contract"]` 中 `class_coverage.ok == False`
- **根因**: 训练集缺少某些类别的样本
- **修复**: 补充缺失类别数据或调整 `num_classes`
- **严重度**: [中等]（会导致评估指标偏差）

### F7: dependency_contract logger 缺失
- **症状**: `report["dependency_contract"]` 中 `logger_dependency.ok == False`
- **根因**: `logger=tensorboard` 但未安装 tensorboard 包
- **修复**: `pip install tensorboard` 或改用 `logger=csv`
- **严重度**: [严重]（训练会崩溃）

### F8: reproducibility seed 未设置
- **症状**: `report["reproducibility"]` 中 `seed_set.ok == False`
- **根因**: 配置中未设置 `trainer.seed`
- **修复**: 在配置中添加 `trainer.seed: 42`
- **严重度**: [轻微]（不影响训练，但影响可复现性）

### F9: resource_contract num_workers 过多
- **症状**: `report["resource_contract"]` 中 `num_workers_reasonable.ok == False`
- **根因**: `num_workers > cpu_count`
- **修复**: 降低 `num_workers` 到 `<= cpu_count`
- **严重度**: [轻微]（不影响正确性，但浪费资源）

### F10: training_contract loss 不匹配
- **症状**: `report["training_contract"]` 中 `loss_task_match.ok == False`
- **根因**: loss 函数与 task_type 不匹配（如 classification 用了 mse）
- **修复**: 使用与 task_type 匹配的 loss
- **严重度**: [中等]（训练可能不收敛）

## Constraints

- 禁止伪造结果：训练必须真实执行
- 禁止掩盖问题：发现的问题必须记录
- 禁止空话：观察必须引用具体日志/字段值，禁止"看起来不错"
- 禁止全 5 分：评分必须有区分度，扣分必须填"关键扣分原因"
- 禁止省略章节：报告必须含全部章节，无内容时填"无"并说明原因
- 报告必须落盘到 `report/` 目录，禁止仅输出到 stdout
