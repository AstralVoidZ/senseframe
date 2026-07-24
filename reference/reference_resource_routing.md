# 资源路由与模型推荐

## 五级路由表

框架自动探测硬件并路由到合适配置：

| 路由级别 | 条件 | device | max_params | max_epochs | batch_size | precision | num_workers |
|----------|------|--------|------------|------------|------------|-----------|-------------|
| `cpu_minimal` | 无GPU + 可用内存<2GB | cpu | 1.0M | 50 | 32 | 32 | 0 |
| `cpu_standard` | 无GPU + 可用内存≥2GB | cpu | 25.0M | 200 | 64 | 32 | 0 |
| `gpu_entry` | GPU显存<4GB | cuda | 50.0M | ∞ | 64 | 16-mixed | 4 |
| `gpu_standard` | GPU显存4-8GB | cuda | 100.0M | ∞ | 128 | 16-mixed | 4 |
| `gpu_high` | GPU显存≥8GB | cuda | ∞ | ∞ | 256 | 16-mixed | 8 |

**max_epochs cap**：`config.trainer.epochs` 与路由表 `max_epochs` 的较小值为实际训练 epochs。`gpu_*` 无上限（使用配置值）。此值同时作为 scheduler 的 `max_epochs` 参数（影响 cosine 的 `T_max` 与 step 的 `step_size`）。

## 路由规则详解

### CPU 路由
- 无 CUDA 可用时进入 CPU 路由
- 可用内存 < 2GB → `cpu_minimal`（只支持极轻量模型，max_params=1.0M）
- 可用内存 ≥ 2GB → `cpu_standard`（支持大部分模型，max_params=25.0M）
- `requires_gpu=True` 的模型在 CPU 路由下不可用

### GPU 路由
- 有 CUDA 可用时按显存分级
- 显存 < 4GB → `gpu_entry`
- 显存 4-8GB → `gpu_standard`
- 显存 ≥ 8GB → `gpu_high`（支持全部模型）

## 配置合并优先级

`resolve_config` 按以下优先级合并配置：

```
YAML 非 null 值 > 路由配置 > 模型默认值
```

具体字段：
- `device`：YAML > 路由（YAML 写 `auto` 时用路由）
- `batch_size`：YAML > 路由
- `learning_rate`：YAML > 模型默认（`default_lr`）
- `optimizer`：YAML > 默认 `adam`
- `precision`：YAML `mixed_precision` > 路由
- `num_workers`：路由（CPU=0），加上限 8 防止 fd 耗尽
- `pin_memory`：仅 GPU 路由启用
- `persistent_workers`：`num_workers > 0` 时启用

## CLI 探测命令

### 探测硬件资源

```bash
python -m senseframe.cli probe
```

输出：资源探测结果 + 路由级别 + 路由配置 + 可用模型列表

### 按数据集推荐模型

```bash
python -m senseframe.cli recommend --dataset UT_HAR_data --priority balanced
```

`--priority` 选项：
- `accuracy`：大模型优先（按参数量降序）
- `speed`：小模型优先（按参数量升序）
- `memory`：低显存优先（按显存升序）
- `balanced`：默认平衡

### 列出可用资源

```bash
# 列出所有模型（可按数据集/范式过滤）
python -m senseframe.cli list-models --dataset UT_HAR_data
python -m senseframe.cli list-models --paradigm cnn

# 列出所有数据集
python -m senseframe.cli list-datasets

# 列出已注册的场景容器
python -m senseframe.cli list-scenes

# 列出 SOTA 范式
python -m senseframe.cli paradigms
```

## 启动前资源预检

`run_experiment` 在训练前执行 `_preflight_check`，检查：

1. **数据集目录存在性**：`data_root` 下需有对应数据集子目录
   - `UT_HAR_data` → `UT_HAR/`
   - `NTU-Fi_HAR` → `NTU-Fi_HAR/`
   - `NTU-Fi-HumanID` → `NTU-Fi-HumanID/`
   - `Widar` → `Widardata/`
2. **GPU 显存预检**：模型 `estimated_vram_mb × 1.2` 需 ≤ 可用显存
3. **磁盘空间**：`output_dir` 所在磁盘至少 1GB 可用空间

预检失败抛 `FileNotFoundError` 或 `RuntimeError`，不进入训练流程。

## 预检模式（--dry-run）

`experiment` 命令支持 `--dry-run` 标志，不实际训练，仅执行启动前检查并输出报告：

```bash
python -m senseframe.cli experiment --config my.yaml --dry-run
```

输出包含 7 项检查结果与训练计划摘要：

| 序号 | 检查项 | 说明 |
|------|--------|------|
| 1 | `config_validation` | 配置 schema 校验 |
| 2 | `scene_registered` | 场景注册校验 |
| 3 | `dataset_supported` | 数据集支持校验 |
| 4 | `model_supported` | 模型支持校验 |
| 5 | `learning_mode_supported` | 学习模式支持校验 |
| 6 | `resource_probe` | 硬件资源探测 + 路由 |
| 7 | `preflight` | 数据存在性、显存、磁盘空间预检 |

`status: ok` 时退出码 0，`status: blocked` 时退出码 1。

### 增强的契约字段

除上表 7 项基础检查外，`--dry-run` 报告还附带以下契约字段（实现于
`senseframe/engine/runner/preflight.py` 的 `validate_*` 函数，由
`senseframe/cli.py` 的 `_run_preflight` 组装进 `report`；注意：pipeline stage
`pipeline/stages/preflight.py` 与 `validate.py` 只做资源探测/路由与 schema 校验，
不产出这些契约字段）。每项检查均为 `CheckResult` 结构：`name` / `ok` /
`severity`(info/warning/error) / `detail` / `error_code` / `remediation`。

| 字段 | 类型 | 说明 | 包含的检查项 |
|------|------|------|--------------|
| `config_semantics` | 数组 | 配置语义校验（跨字段逻辑约束），有 error 级失败则整体 blocked | `early_stopping_within_epochs` / `batch_size_within_dataset` / `scheduler_epochs_compatible` / `deterministic_cuda_available` |
| `dependency_contract` | 数组 | 依赖契约校验（logger / export / deterministic 依赖），有 error 级失败则整体 blocked | logger 依赖、export 格式依赖、deterministic 依赖等 |
| `reproducibility` | 数组 | 可复现性检查（seed / deterministic / 版本记录） | `seed_set` 等 |
| `resource_contract` | 数组 | 资源契约校验（显存 / num_workers / 训练规模估算）；dry-run 下 `vram_probe_result=None` | `vram_sufficient` / `num_workers_reasonable` 等 |
| `dynamic_validation` | 对象 | 轻量动态校验（CPU 上 forward + backward 1 step，不启动 Lightning Trainer）。`status`: `passed` / `skipped` / `failed`；`skipped` 触发条件：`--static-only` 或静态检查未通过 | `checks`: `forward_pass` / `output_shape_match` / `backward_pass` / `param_count_reasonable` |
| `model_contract` | — | 非独立顶层 key：其 checks 归入 `dynamic_validation.checks`（`cli.py` 在统一报告里将 model_contract 类别映射到 `dynamic_validation`） | 同 `dynamic_validation.checks` |
| `training_contract` | 数组 | 训练契约校验（loss / metrics 与 task_type 一致性）；仅当 `dynamic_validation.status == "passed"` 且拿到 `num_classes` 后才生成 | `loss_task_match` / `metrics_task_match` / `early_stopping_within_epochs` |
| `data_contract` | 数组 | 数据契约校验（基于 `DataProfile`，如类别覆盖、不平衡）；仅当动态校验成功后且 `detail` 含 `class_distribution` 时才生成 | `class_coverage` 等 |

说明：
- `config_semantics` 会在动态校验成功后用真实 `n_samples` 重跑一次（首次 `n_samples=0` 时
  `batch_size_within_dataset` 检查被禁用），使 batch_size 与数据集规模约束真正生效。
- `dynamic_validation` 失败时会更新顶层 `status` 为 `blocked`（旧逻辑仅 `config_semantics` /
  `dependency_contract` 失败才置 blocked，已修复）。
- `--static-only` 标志可显式跳过动态校验（`dynamic_validation.status == "skipped"`）。
