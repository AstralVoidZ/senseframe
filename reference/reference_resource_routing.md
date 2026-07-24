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
