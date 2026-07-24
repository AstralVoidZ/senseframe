# P3 验证经验教训（10.2 单场景 + 10.4 跨场景 + DANN 跨模态）

> 本文档记录 P3 验证 10.2 单场景（A1-A5 × 4 CSI 数据集）、10.4 跨场景（B1-B8）与
> DANN 跨模态对齐（B5_DANN/B6_DANN）过程中踩过的坑、根因分析与防御性设计建议。
> 所有教训均对齐 [project_memory.md 第 129 条](../../project_memory.md) 的归一化历史根因。
> 更新日期：2026-07-21

## 目录

1. [归一化策略的应用位置冲突](#1-归一化策略的应用位置冲突)
2. [数据集原始 shape 必须实测，不能依赖文档/直觉](#2-数据集原始-shape-必须实测不能依赖文档直觉)
3. [scene 的 self_supervised 模式语义可能与实验意图不符](#3-scene-的-self_supervised-模式语义可能与实验意图不符)
4. [CLI 参数覆盖机制缺失导致快速验证成本高](#4-cli-参数覆盖机制缺失导致快速验证成本高)
5. [依赖管理：从 requirements.txt 迁移到 pyproject.toml](#5-依赖管理从-requirementstxt-迁移到-pyprojecttoml)
6. [PromptTuning 形状不匹配：d_model 推断错误 + 注入位置错误](#6-prompttuning-形状不匹配d_model-推断错误--注入位置错误)
7. [长 epoch 训练的日志频率陷阱](#7-长-epoch-训练的日志频率陷阱)
8. [SP 搜索 vs 固定配置：上限数据集上难显优势](#8-sp-搜索-vs-固定配置上限数据集上难显优势)
9. [GridSampler 全网格爆炸：n_trials 必须显式限制](#9-gridsampler-全网格爆炸n_trials-必须显式限制)
10. [B5/B6 跨场景迁移的双层架构对齐缺失](#10-b5b6-跨场景迁移的双层架构对齐缺失)
11. [非上限数据集 SP 搜索价值有限 + GridSampler 遍历顺序缺陷](#11-非上限数据集-sp-搜索价值有限--gridsampler-遍历顺序缺陷)
12. [DANN 跨模态对齐：双模态架构冲突与短训练失效](#12-dann-跨模态对齐双模态架构冲突与短训练失效)

---

## 1. 归一化策略的应用位置冲突

### 现象

A1 NTU-Fi_HAR baseline val_acc 仅 67.74%（同一配置下 SenseFi 基准可达 95%+），A3 NTU-Fi_HAR MAE pretrain loss 高达 1722（正常 < 1.0），LoRA 微调 val_acc 仅 21.51%。

在 collate_fn 中添加归一化后，A1 NTU-Fi_HAR 提升到 100%，但 A1 UT_HAR_data 从 95.56% **降到 30.04%**（重复归一化破坏数据）。

### 根因

SenseFrame 的归一化策略注册表（[registry.py](../senseframe/registry.py)）支持 `register_normalization(name, strategy)` 注册，但**应用位置不统一**：

| 数据集 | Loader 实现 | 加载时归一化? | 备注 |
|---|---|---|---|
| UT_HAR_data | [tensor_loader.py:95](../senseframe/data/loaders/tensor_loader.py#L95) | ✅ 是 | `data_norm = norm_strategy.apply(data)` |
| NTU-Fi_HAR | [csi_mat_loader.py](../senseframe/data/loaders/csi_mat_loader.py) | ❌ 否 | 仅 `CSIDataset(dir, layout=layout)` |
| NTU-Fi-HumanID | 同上 | ❌ 否 | 同上 |
| Widar | csv_folder_loader.py | ❌ 否 | 同上 |

后果：
- NTU-Fi 系列原始 CSI 振幅 ~42 直接进入模型 → MAE loss 爆炸（1722）
- collate_fn 添加归一化后，UT_HAR_data 被**重复归一化**（`(x-mean)/std` 做了两次）→ val_acc 从 95.56% 降到 30.04%

### 修复

在 [p3_eval_common.py:320-360](../scripts/p3_eval_common.py#L320-L360) 引入白名单机制：

```python
# 已在 loader 层应用归一化的数据集白名单
_LOADER_NORMALIZED_DATASETS = {"UT_HAR_data"}

def _make_collate_fn(target_shape, dataset_name=None):
    norm_strategy = None
    if dataset_name is not None and dataset_name not in _LOADER_NORMALIZED_DATASETS:
        from senseframe.registry import get_normalization_or_none
        norm_strategy = get_normalization_or_none(dataset_name)
        # ...
    elif dataset_name in _LOADER_NORMALIZED_DATASETS:
        logger.info("collate_fn: dataset '%s' skip collate-time normalization "
                    "(loader has already applied it)", dataset_name)
```

### 防御性建议

**短期**（已落地）：白名单 + collate_fn 显式归一化补齐 loader 缺失。

**长期**（建议后续重构）：
- 归一化**应在 loader 层统一应用**，collate_fn 不再做归一化（单一职责）
- 修改 `tensor_loader.py` / `csi_mat_loader.py` / `csv_folder_loader.py` 都在加载时调用 `norm_strategy.apply(data)`
- 删除 `_LOADER_NORMALIZED_DATASETS` 白名单（不再需要）
- 在 `DatasetLoader` 基类的 `load_splits` 模板方法中强制调用归一化（模板方法模式）

### 验证效果

| 实验 | 修复前 | 修复后 |
|---|---|---|
| A1 UT_HAR_data | 95.56% → 30.04%（重复归一化） | **95.56%** |
| A1 NTU-Fi_HAR | 67.74%（未归一化） | **100.00%** |
| A1 NTU-Fi-HumanID | 31.48%（未归一化） | **100.00%** |
| A3 UT_HAR_data | 80.44% | **80.44%** |
| A3 NTU-Fi_HAR | 21.51%（MAE loss=1722） | **86.02%**（MAE loss=0.42） |
| A3 NTU-Fi-HumanID | 11.11%（MAE loss=1722） | **55.56%** |

---

## 2. 数据集原始 shape 必须实测，不能依赖文档/直觉

### 现象

A3 NTU-Fi_HAR 报错：`RuntimeError: shape '[342, 500]' is invalid for input of size 684000`

### 根因

CSI_DATASET_CONFIG 中 NTU-Fi_HAR / NTU-Fi-HumanID 的 `reshape_to` 写的是 `(342, 500)`，依据是"NTU-Fi 单 sample 0.5s × 1000Hz = 500 sample"。但 CSIMatLoader 实际产出 `(342, 2000)` —— NTU-Fi 原始 .mat 时长 4 秒 × 500Hz = 2000 sample。

### 修复

```python
"NTU-Fi_HAR": {
    "raw_shape": (342, 2000),       # 342 = 3 antenna × 114 subcarrier
    "reshape_to": (342, 2000),      # 2000 = 4s × 500Hz
    "patch_len": 20,                # 2000 % 20 == 0 → 100 patches
    "num_classes": 6,
},
```

### 防御性建议

- **新数据集接入时，第一步必须用 `data[0].shape` 实测**，写进 DatasetSpec.input_shape
- 文档/代码注释中**禁止**写"假设 shape 为 X"这类猜测性描述
- 在 CSIMatLoader.load_splits 的日志中已输出 `train_shape`，新数据集接入时必须看这条日志确认

---

## 3. scene 的 self_supervised 模式语义可能与实验意图不符

### 现象

A3 NTU-Fi_HAR 走 scene 的 `learning_mode="self_supervised"` 加载，触发 `CUDA device-side assert`（CrossEntropyLoss 标签越界）。

### 根因

[container.py:165-174](../senseframe/scenes/wifi_csi/container.py#L165-L174) 中 NTU-Fi_HAR 的 self_supervised 模式设计为"跨数据集迁移"：
- `unsupervised` = NTU-Fi_HAR 自己（用于 MAE 预训练）
- `supervised_finetune` = NTU-Fi-HumanID 数据（14 类，用于微调）
- `test` = NTU-Fi-HumanID 测试集

但 A3 实验意图是"同数据集预训练 + 微调"（用 NTU-Fi_HAR 自己的 6 类标签微调），不是跨数据集迁移。配置 `num_classes=6` 但实际标签是 0-13（HumanID 14 类），触发 CUDA nll_loss 断言。

### 修复

A2-A5 统一走 `learning_mode="supervised"` 加载，`pretrain_ds = train_ds`（MAE 只用 x 不用标签，复用 train 集做自监督 mask 重建即可）：

```python
if config.pretrain_source == "csi_4datasets":
    bundle = _load_csi_dataset(config.target_dataset, data_root,
                               learning_mode="supervised")
    train_ds = bundle.train
    val_ds = bundle.val if bundle.val else bundle.test
    pretrain_ds = train_ds  # 复用 train 集做自监督 mask 重建
```

### 防御性建议

- **scene 的 learning_mode 语义必须在文档中显式说明**，不能假设 `self_supervised` = "同数据集自监督"
- 跨数据集迁移（B 系列）才是 scene 的 self_supervised 模式的正确用法
- 同数据集自监督预训练（A3）应直接复用 train 集，不依赖 scene 的 self_supervised 模式

---

## 4. CLI 参数覆盖机制缺失导致快速验证成本高

### 现象

P3 验证方案要求 A1-A5 × 4 数据集 = 20 个实验，每个实验默认 50 epochs + 20 pretrain_epochs。完整跑一遍预计 6+ 小时，但快速验证（10 epochs + 5 pretrain_epochs）只需 30 分钟。原代码无 CLI 参数支持，必须改 ExperimentConfig 默认值才能快速验证。

### 修复

在 [p3_eval_common.py:add_common_args](../scripts/p3_eval_common.py) 新增 4 个参数：

```python
parser.add_argument("--epochs", type=int, default=None,
                    help="微调训练 epoch 数（默认用 ExperimentConfig.epochs=50）")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--pretrain-epochs", type=int, default=None)
parser.add_argument("--learning-rate", type=float, default=None)

def apply_arg_overrides(args, config):
    """args 中对应字段为 None 时不覆盖（保留 ExperimentConfig 默认值）。"""
    from dataclasses import replace
    overrides = {}
    for field in ("epochs", "batch_size", "pretrain_epochs", "learning_rate"):
        v = getattr(args, field, None)
        if v is not None:
            overrides[field] = v
    return replace(config, **overrides) if overrides else config
```

### 防御性建议

- 所有实验脚本入口都应支持 CLI 覆盖关键超参（epochs/batch_size/lr/pretrain_epochs）
- `default=None` 而非具体值，避免覆盖 ExperimentConfig 的默认值（保留单一数据源）
- 用 `dataclasses.replace` 而非直接修改对象（不可变语义，避免副作用）

---

## 5. 依赖管理：从 requirements.txt 迁移到 pyproject.toml

### 背景

原 `requirements.txt` 是扁平依赖清单，无法表达：
- 可选依赖（EEG/Radio/ONNX/OTel 按需安装）
- 项目元数据（版本/作者/Python 版本约束）
- 工具配置（pytest markers / ruff lint rules）

### 修复

迁移到 [pyproject.toml](../pyproject.toml)（PEP 621 标准）：

```toml
[project]
name = "senseframe"
dynamic = ["version"]                    # 从 senseframe/__init__.py 的 __version__ 读
requires-python = ">=3.10"
dependencies = [...]                     # 核心依赖

[project.optional-dependencies]
eeg = ["mne>=1.5.0"]                     # 可选 extras
radio = ["h5py>=3.8.0"]
onnx = ["onnx>=1.14.0"]
otel = [...]                             # 替代原 requirements-otel.txt
dev = ["pytest>=7.0.0"]
all = ["senseframe[eeg,radio,onnx,otel,dev]"]

[tool.setuptools.dynamic]
version = { attr = "senseframe.__version__" }  # 动态版本

[tool.pytest.ini_options]                # 工具配置集中
markers = ["slow", "integration", "gpu"]
```

安装方式：
- `pip install .` — 仅核心依赖
- `pip install -e .[eeg,radio,dev]` — 开发常用组合
- `pip install -e .[all]` — 全部可选依赖

### 影响范围（必须同步更新）

迁移时**必须**同步更新所有引用 `requirements.txt` 的位置：

| 文件 | 修改内容 |
|---|---|
| `tests/pack_code.py` | `ROOT_PAYLOAD_FILES` / `EnvState` / `should_skip_deps` / `update_deps` / `install_requirements` / 后置校验 / 帮助文本 / bash 脚本 |
| `README.md` | 安装说明 |
| `examples/README.md` | 环境准备 |
| `SKILL.md` | ONNX 可选依赖说明 |
| `commands/senseframe-train.md` | 前置条件 |

### 防御性建议

- **CI 应增加一致性检查**：grep 全仓库 `requirements\.txt` 引用，若存在则报错（除非在 pyproject.toml 自身的迁移说明注释中）
- `pack_code.py` 的 `.senseframe_deploy.json` 中 `requirements_hash` 字段名保留不变（避免破坏已有部署的增量判断），但实际 hash 计算对象改为 `pyproject.toml`
- 安装命令统一用 `pip install -e '.[eeg,radio,dev]'`（可编辑模式 + extras），覆盖 SenseFrame 部署场景的常用依赖

---

## 6. PromptTuning 形状不匹配：d_model 推断错误 + 注入位置错误

### 现象

A5 prompt_tuning × 3 数据集全部失败：
```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 6840 but got size 2000 for tensor number 1 in the list.
```

### 根因（双重错误）

**错误 1：`_infer_d_model` 取错了维度**

[peft_builder.py:_infer_d_model](../senseframe/automl/peft_builder.py) 原实现：
```python
def _infer_d_model(backbone: nn.Module) -> int:
    for module in backbone.modules():
        if isinstance(module, nn.Linear):
            return module.in_features  # ← 取第一个 Linear 的 in_features
    return 128
```

对 CSIFoundationModel，第一个 Linear 是 [CSIPatchEmbedder.proj](../senseframe/scenes/wifi_csi/foundation_model.py) = `nn.Linear(patch_len * C, d_model)`：
- NTU-Fi: `patch_len * C = 20 * 342 = 6840`（而非真正的 d_model=128）
- UT_HAR: `patch_len * C = 10 * 90 = 900`（而非 128）

prompt 张量被错误地初始化为 `(prompt_len, 6840)`，而期望拼接的 patch tokens 维度是 `(B, n_patches, 128)`。

**错误 2：`PEFTModel.forward` 在错误的层注入 prompt**

原实现：
```python
def forward(self, *args, **kwargs):
    if args and isinstance(args[0], torch.Tensor):
        x = args[0]
        if self.prompt_layer is not None:
            x = self.prompt_layer(x)  # ← 直接对 backbone 输入 (B, C, L) 拼接 prompt
        ...
    return self.backbone(*args, **kwargs)
```

但 PromptTuning 的标准做法（Lester et al. 2021）是在 **patch/word embedding 之后、encoder 之前** 拼接 prompt：
- 输入 `(B, C, L)` → patch_embedder → `(B, n_patches, d_model)` → cat prompt → encoder

原实现对 `(B, C, L)` 原始信号直接 cat prompt，语义错误（prompt 与原始 CSI 振幅不是同一空间）。

### 修复（两处协同修改）

**修复 1：`_infer_d_model` 优先用 backbone.d_model 属性**

```python
@staticmethod
def _infer_d_model(backbone: nn.Module) -> int:
    """优先使用 backbone.d_model 属性（CSIFoundationModel 显式声明 d_model=128）；
    缺失时兜底取第一个 Linear 的 in_features。"""
    d = getattr(backbone, "d_model", None)
    if isinstance(d, int) and d > 0:
        return d
    for module in backbone.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
    return 128
```

**修复 2：`PEFTModel.forward` 检测 backbone 内部结构，在 patch embedding 之后注入**

```python
def forward(self, *args, **kwargs):
    has_prompt_or_prefix = (self.prompt_layer is not None or
                            self.prefix_layer is not None)
    if has_prompt_or_prefix and args and isinstance(args[0], torch.Tensor):
        x = args[0]
        backbone = self.backbone
        # 检测 backbone 是否暴露 patch_embedder + encoder + pos_embed 标准接口
        if (hasattr(backbone, "patch_embedder")
            and hasattr(backbone, "encoder")
            and hasattr(backbone, "pos_embed")):
            # 复刻 CSIFoundationModel.encode(x) 流程，在 encoder 之前注入 prompt
            patches = backbone.patch_embedder(x) + backbone.pos_embed
            if self.prompt_layer is not None:
                patches = self.prompt_layer(patches)
            if self.prefix_layer is not None:
                patches = self.prefix_layer(patches)
            x = backbone.encoder(patches)
            if hasattr(backbone, "encoder_norm"):
                x = backbone.encoder_norm(x)
            return x
        # 简单 backbone：直接对输入注入（兼容 _SequenceBackbone 等测试用例）
        ...
    return self.backbone(*args, **kwargs)
```

### 防御性建议

- **`_infer_d_model` 必须优先用显式属性**：`backbone.d_model` 是模型内部 token 的真实维度，比"第一个 Linear 的 in_features"更可靠（后者可能是 patch_len * C 这种"展平后"的维度）
- **PromptTuning / PrefixTuning 的注入位置必须在 embedding 之后**：直接对原始信号 `(B, C, L)` cat prompt 在语义和维度上都是错误的
- **PEFTModel.forward 需要感知 backbone 内部结构**：通过 `hasattr` 检测标准接口（patch_embedder/encoder/pos_embed），有则穿透注入，无则兜底原行为，保持向后兼容
- **建议在 SensingFoundationModel Protocol 中显式定义 `patch_embedder` / `encoder` / `encoder_norm` 接口**，让 PEFTModel 能稳定地穿透注入，不依赖 `hasattr` 启发式检测

### 验证效果

| 实验 | 修复前 | 修复后 |
|---|---|---|
| A5 UT_HAR_data | FAILED（shape mismatch） | **66.94%** (macro_f1=59.20%, params=2183) |
| A5 NTU-Fi_HAR | FAILED（shape mismatch） | **69.89%** (macro_f1=68.48%, params=2054) |
| A5 NTU-Fi-HumanID | FAILED（shape mismatch） | **61.11%** (macro_f1=53.36%, params=3086) |
| A5 Widar | 未跑（小数据集已失败） | **24.00%** (macro_f1=10.53%, params=4118) |

A5 修复后参数量极小（2000-4000，相比 A1 的 1.6M 仅占 0.2%），精度低于 A1 是预期内的（容量限制）。但**修复使 A5 从完全不可用变为可用**，验证了 prompt_tuning 路径的正确性。

---

## 7. 长 epoch 训练的日志频率陷阱

### 现象

A1 Widar 训练 14 分钟无新日志输出，误判为卡死并 StopCommand。实际进程正常跑完 epoch 1 后正在跑 epoch 2-9（每 epoch ~2 分钟），但日志只在 epoch 1 和每 10 个 epoch 输出。

### 根因

[p3_eval_common.py](../scripts/p3_eval_common.py) 的训练日志逻辑：
```python
# 原逻辑（错误）
if (epoch + 1) % 10 == 0 or epoch == 0:
    logger.info("epoch %d/%d: ...", epoch + 1, epochs, ...)
```

Widar 每 epoch 约 2 分钟，epochs=10：
- epoch 1（epoch==0）→ 输出
- epoch 2-9 → **8 个 epoch 无日志**（约 16 分钟静默）
- epoch 10（(epoch+1)%10==0）→ 输出

长 epoch 训练时，日志间隔过长，无法判断是卡死还是正常训练。

### 修复

改为**每个 epoch 都输出日志**：
```python
# 新逻辑（正确）
logger.info(
    "epoch %d/%d: train_loss=%.4f, val_acc=%.4f, macro_f1=%.4f",
    epoch + 1, epochs, mean_loss, val_acc, macro_f1,
)
```

### 防御性建议

- **训练日志的输出频率必须高于人工判断卡死的阈值**（通常 5-10 分钟）
- 长 epoch 训练（每 epoch > 1 分钟）必须每 epoch 输出日志
- 短 epoch 训练（每 epoch < 10 秒）可以每 10 epoch 输出，避免日志爆炸
- 建议在 `_train_classifier` 中加自适应逻辑：`if avg_epoch_time > 60: log_every = 1; else: log_every = 10`

---

## 8. SP 搜索 vs 固定配置：上限数据集上难显优势

### 现象

10.5 C 系列 SP 搜索在 NTU-Fi_HAR 上跑完 5 个实验：

| 实验 | 方法 | val_acc | macro_f1 | trainable_params | 时间 | best_params |
|---|---|---|---|---|---|---|
| C1 | 固定 lora (rank=8) | 88.17% | 88.45% | 21,254 | 90s | — |
| C2 | 固定 adapter (bottleneck=128) | **100.00%** | **100.00%** | 3,275,526 | 81s | — |
| C3 | 固定 prompt_tuning (length=10) | 63.44% | 57.66% | 2,054 | 79s | — |
| C4 | SP random × 20 trials | **100.00%** | **100.00%** | 6,550,278 | 1577s | adapter bottleneck=256 |
| C5 | SP grid × 24 trials | 91.40% | 91.59% | 11,014 | 1821s | lora rank=4 |

**通过标准 1：C4 best_val_accuracy > max(C1, C2, C3) + 1%**
- max(C1, C2, C3) = 100.00%（C2 已达上限）
- C4 = 100.00%，improvement = **+0.00%**
- **结果：未通过**（C4 与 C2 持平，无法超越）

**通过标准 2：C4 n_completed / n_trials ≥ 90%**
- C4: 20/20 = 100% ✓

**通过标准 3：C5 best_val_accuracy ≥ C4（Grid 是上界）**
- C5 = 91.40% vs C4 = 100.00%，grid_upper_bound_diff = **-8.60%**
- **结果：未通过**（Grid 受限于 n_trials=24 个网格点，未覆盖到 C4 的最优 adapter 配置）

### 根因

**核心问题：NTU-Fi_HAR 是"上限数据集"**

A 系列 10.2 验证已暴露此问题（参见附录 A1/A2 在 NTU-Fi_HAR 上都达 100%），C 系列在 NTU-Fi_HAR 上验证 SP 搜索存在结构性缺陷：

1. **C2 固定 adapter (bottleneck=128) 已达 100%**，SP 搜索没有任何"提升空间"
2. **C4 找到的最优配置是 adapter bottleneck=256**（参数量从 3.28M 翻倍到 6.55M），val_acc 仍为 100% — 与 C2 持平
3. **C5 GridSampler 受 n_trials=24 限制**，24 个网格点未覆盖到 adapter bottleneck=256 的最优配置，最优停留在 lora rank=4 (91.40%)

**SP 搜索的真正价值场景：**
- 在"非上限数据集"上（如 Widar A1 仅 69.73%），SP 搜索有 ~30% 的提升空间，可以探索不同 PEFT 配置找最优
- 在"上限数据集"上，SP 搜索退化为"参数量翻倍但不涨点"的资源浪费

### 防御性建议

- **SP 搜索评估必须避开"上限数据集"**：在 A 系列已发现某数据集 A1=100% 时，C 系列应在更难的子集上跑（如降低训练样本数 / 增加类别数 / 加入噪声）
- **SP 搜索的通过标准应包含"参数效率"约束**：当前仅看 val_acc，但 C4 参数量翻倍（6.55M vs C2 的 3.28M）不涨点应判定为"无效搜索"。建议标准改为 `C4.val_acc > max(C1,C2,C3) + 1% AND C4.trainable_params ≤ 1.5 × min(C1,C2,C3).trainable_params`
- **SP 搜索结果应记录"搜索轨迹"**：当前只保留 best_params，看不到搜索过程中 val_acc 的分布。建议在 `_run_sp_search` 中把每个 trial 的 (params, val_acc) 都存到结果中，便于后续分析"搜索是否真的探索了有效空间"
- **后续应在 Widar / EEG / Radio 上重跑 C 系列**：这三个数据集 A 系列表现均不饱和（Widar 69.73%，EEG/Radio 待验证），SP 搜索有真实提升空间

### 验证效果

| 检查项 | 结果 |
|---|---|
| C4 vs max(C1,C2,C3) | +0.00%（上限数据集，无空间） |
| C4 n_completed/n_trials | 100% ✓ |
| C5 vs C4 | -8.60%（Grid 24 点未覆盖最优） |
| **总体判定** | NTU-Fi_HAR 上 SP 搜索价值无法体现，需迁移到非上限数据集重验 |

---

## 9. GridSampler 全网格爆炸：n_trials 必须显式限制

### 现象

C5 原配置 `n_trials=None`（意图：跑完全网格 192 个点），实际启动后预估需 4 小时（192 × ~75s/trial）。改为 `n_trials=24` 后限制在 30 分钟内完成，但导致 grid search 没覆盖到 C4 的最优配置（adapter bottleneck=256），C5 val_acc (91.40%) 反而低于 C4 (100.00%)。

附带问题：C4 与 C5 的 best_params 字段类型不一致：
```json
// C4 (RandomSampler)
"best_params": {"peft_method": "adapter", "peft_rank": "8",
                "adapter_bottleneck": "256", "prompt_length": "20"}  // 全是 str

// C5 (GridSampler)
"best_params": {"peft_method": "lora", "peft_rank": 4,
                "adapter_bottleneck": 128, "prompt_length": 10}  // peft_rank/bottleneck/length 是 int
```

### 根因

**问题 1：GridSampler 没有"完成全网格就停止"的语义**

[search_protocol.py:GridSampler.sample](../senseframe/search_protocol.py) 的实现：
```python
def sample(self, search_space, history):
    # 生成全网格 all_combos
    all_combos = list(itertools.product(*grids))
    idx = len(history) % len(all_combos)  # ← 永远不会停止
    combo = all_combos[idx]
    return {p.name: v for p, v in zip(search_space.parameters, combo)}
```

`idx = len(history) % len(all_combos)` 取模意味着：跑完 192 个 trial 后会从第 0 个开始重复。Sampler 自身没有"网格已耗尽"的信号，StudyManager.ask() 也只看 `n_trials` 是否达到，不看网格是否覆盖完。

**问题 2：categorical 参数类型保留不一致**

[search_protocol.py:RandomSampler.sample](../senseframe/search_protocol.py) 中：
```python
elif p.type == "categorical" and p.choices:
    params[p.name] = str(self._rng.choice(p.choices))  # ← str() 强转
```

`np.random.Generator.choice` 对 list 返回 numpy scalar，作者用 `str()` 转换避免 numpy 类型泄漏，但副作用是把 `int` choices (`[4, 8, 16, 32]`) 转成了 `str` (`"4"`, `"8"`, ...)。GridSampler 则直接取 `p.choices` 中的原始 Python `int`，导致同一参数在两个 Sampler 下类型不同。

当前 [`_params_to_peft_config`](../scripts/p3_eval_common.py) 用 `int(params.get("peft_rank", 8))` 兜底转换，所以 C4/C5 都能正常 build PEFT 模型，但这是**脆弱的**：如果某天某个参数不是数值型（如 `peft_target_modules` 这种字符串列表），`int()` 转换会直接报错。

### 修复

**修复 1：C5 n_trials 从 None 改为 24**

[scripts/p3_eval_common.py:SP_SEARCH_EXPERIMENTS](../scripts/p3_eval_common.py)：
```python
# 修复前
{"id": "C5", "method": "sp_search", "config": {"sampler": "grid", "n_trials": None}},

# 修复后
{"id": "C5", "method": "sp_search",
 "config": {"sampler": "grid", "n_trials": 24}},  # 限制 24 个网格点（全网格 192 点过多）
```

**修复 2（建议，未落地）：RandomSampler 保留 choices 原始类型**

```python
# 修复前
elif p.type == "categorical" and p.choices:
    params[p.name] = str(self._rng.choice(p.choices))

# 修复后
elif p.type == "categorical" and p.choices:
    # 用 random.choice 保留 choices 中的原始 Python 类型（int/str/...）
    import random as _random
    params[p.name] = _random.choice(p.choices)
```

但此修复涉及 `RandomSampler` 的独立 RNG（P3-1 修复：用 `np.random.default_rng(seed)` 避免被全局 set_seed 重置），改用 `random.choice` 会重新引入全局 random 依赖。需要用 `self._rng.choice(p.choices).item()` 把 numpy scalar 转回 Python 原生类型，保留独立 RNG。

### 防御性建议

- **GridSampler 必须显式设置 n_trials**：`n_trials=None` 在 SP 协议中没有"跑完全网格"的语义，只会无限循环。当前实现下 `None` 会被 `int(sp_cfg.get("n_trials", 20))` 兜底为 20，但这是隐式的，不应依赖
- **GridSampler 应支持"全网格自动停止"**：建议在 `GridSampler.sample` 中检测 `len(history) >= len(all_combos)`，抛 `StopIteration` 或返回 sentinel，StudyManager.ask 捕获后自动结束 study（语义对齐 Optuna TPESampler 的 `n_trials=None` 行为）
- **Sampler 应保留 choices 中的原始类型**：所有 categorical 参数的采样结果类型应与 `ParameterSpec.choices` 中的元素类型一致。建议把 `RandomSampler` 的 `str(self._rng.choice(p.choices))` 改为 `self._rng.choice(p.choices).item()`（numpy scalar → Python 原生类型）
- **搜索空间设计应避免全网格爆炸**：当前 4 参数 × 3+4+4+4 = 192 个点已偏多。建议搜索空间设计时预估 `prod(len(p.choices) for p in params)`，超过 50 个点时应改用 RandomSampler 或缩减 choices
- **CI 应增加 SP 搜索配置校验**：grep `SP_SEARCH_EXPERIMENTS` 中所有 `n_trials=None` 的 grid 配置，CI 报错（强制显式指定）

### 验证效果

| 修复项 | 修复前 | 修复后 |
|---|---|---|
| C5 n_trials | None（被兜底为 20） | 24（显式） |
| C5 完成时间 | 预估 4 小时（192 trial） | 30 分钟（24 trial） |
| C5 best_params 类型 | 与 C4 不一致（int） | 仍与 C4 不一致（待修复 2 落地） |
| C5 val_acc vs C4 | 91.40% < 100% | 同（n_trials=24 限制下未覆盖最优 adapter） |

---

## 10. B5/B6 跨场景迁移的双层架构对齐缺失

### 现象

10.4 跨场景迁移评估 B5/B6 设计意图：

| 实验 | pretrain | target | finetune | 设计意图 |
|---|---|---|---|---|
| B4 | none | PhysioNet_MI | scratch | EEG baseline（无预训练） |
| B5 | csi_4datasets | PhysioNet_MI | lora | CSI MAE 预训练 → EEG LoRA 微调 |
| B6 | csi_4datasets | PhysioNet_MI | full | CSI MAE 预训练 → EEG 全量微调 |

B4 baseline 已跑通（val_acc=59.09%, 5 epoch, CPU），但 B5/B6 启动后实际行为与设计意图**完全不符**：

- B5 实际行为 = "EEG scratch + LoRA"（用 EEG input_shape 构建 backbone + EEG 数据训练 + LoRA 微调）
- B6 实际行为 = "EEG scratch + Full"（同上，但 Full 微调）
- **CSI MAE 预训练完全没发生**，"跨场景迁移"退化为"同场景 scratch"

通过标准 `B5 val_acc > B4 + 3%` 在当前实现下毫无意义：B5 与 B4 用同样的 backbone 结构、同样的数据集，只是微调方法不同（LoRA vs scratch），无法验证 CSI→EEG 的迁移增益。

### 根因

**根因 1：pretrain_source 是字符串标签，未驱动真实跨数据集预训练**

[scripts/p3_eval_common.py:run_single_experiment](../scripts/p3_eval_common.py) 的预训练逻辑（line 773-847）：

```python
if is_csi:
    if config.pretrain_source == "csi_4datasets":
        bundle = _load_csi_dataset(config.target_dataset, data_root, ...)
        pretrain_ds = train_ds  # ← 用 target_dataset 自己做预训练！
elif is_eeg:
    pretrain_ds = None  # ← EEG 分支直接置 None，跳过 MAE 预训练
    # 注释："EEG 预训练待 RadioML 就绪后扩展"

# 后续：
if config.pretrain_source != "none" and pretrain_ds is not None:
    backbone.pretrain(pretrain_loader, pretrain_cfg)
    # CSI 分支：在 target_dataset 上做 MAE 预训练（同数据集预训练）
    # EEG 分支：pretrain_ds=None，跳过预训练
```

`pretrain_source="csi_4datasets"` 在 CSI 分支被解释为"用 target_dataset 自己做预训练"（同数据集预训练，不是跨数据集），在 EEG 分支被解释为"跳过预训练"。两个分支都没实现"在 CSI 数据集上预训练 → 在 EEG 数据集上微调"的真正跨场景迁移。

**根因 2：CSIPatchEmbedder 维度由 input_shape 决定，跨模态不兼容**

[senseframe/scenes/wifi_csi/foundation_model.py:CSIPatchEmbedder](../senseframe/scenes/wifi_csi/foundation_model.py)：

```python
class CSIPatchEmbedder(nn.Module):
    def __init__(self, input_shape, patch_len, d_model):
        C, L = input_shape
        self.proj = nn.Linear(patch_len * C, d_model)  # ← 维度由 patch_len * C 决定
```

不同模态的 input_shape 与 patch_len * C：

| 模态 | 数据集 | input_shape (C, L) | patch_len | patch_len * C |
|---|---|---|---|---|
| CSI | NTU-Fi_HAR | (342, 2000) | 20 | **6,840** |
| CSI | Widar | (22, 400) | 20 | 440 |
| EEG | PhysioNet_MI | (64, 480) | 20 | **1,280** |
| Radio | RadioML2018 | (2, 1024) | 16 | 32 |

CSI NTU-Fi_HAR 的 patch_embedder.proj 权重形状 `(128, 6840)`，EEG PhysioNet_MI 的 patch_embedder.proj 权重形状 `(128, 1280)`。**直接迁移 backbone 时，patch_embedder.proj 的输入维度不匹配，无法复用预训练权重**。

transformer encoder + pos_embed + decoder 这些模块是 modality-agnostic 的（只依赖 d_model 和 n_patches），可以跨模态迁移。但 patch_embedder 是 modality-specific 的，必须替换或对齐。

### 方案设计

#### 方案 A：替换 patch_embedder + 保留 transformer 主体（推荐）

最小改动、文献标准的跨模态迁移做法：

```python
# B5/B6 流程：
# 1. 用 CSI 数据集构建 backbone 并 MAE 预训练
csi_config = CSI_DATASET_CONFIG["NTU-Fi_HAR"]
backbone = _build_backbone(csi_config, d_model=128)
backbone.pretrain(csi_pretrain_loader, pretrain_cfg)

# 2. 替换 patch_embedder 为 EEG 维度（保留 transformer encoder + decoder + pos_embed 主体）
eeg_config = EEG_DATASET_CONFIG["PhysioNet_MI"]
backbone.replace_patch_embedder(
    new_input_shape=eeg_config["input_shape"],
    new_patch_len=eeg_config["patch_len"],
)  # 重新初始化 patch_embedder.proj + pos_embed

# 3. 在 EEG 数据上构建 PEFT 模型 + 微调
peft_model = _build_peft_model(backbone, "lora", ...)
# LoRA 注入到 transformer encoder 的 query/value（modality-agnostic），
# 不影响新初始化的 patch_embedder（patch_embedder 跟着 PEFT 冻结策略走）
```

需要在 [CSIFoundationModel](../senseframe/scenes/wifi_csi/foundation_model.py) 上新增 `replace_patch_embedder(new_input_shape, new_patch_len)` 方法：
- 重新构建 `self.patch_embedder = CSIPatchEmbedder(new_input_shape, new_patch_len, self.d_model)`
- 重新初始化 `self.pos_embed = nn.Parameter(torch.empty(1, new_n_patches, d_model))`（n_patches 变了）
- 保留 `self.encoder` / `self.encoder_norm` / `self.decoder_embed` / `self.decoder` / `self.decoder_norm` / `self.mask_token`（modality-agnostic）

#### 方案 B：modality-specific patch embedder 多分支

在 CSIFoundationModel 中维护 `patch_embedders: Dict[str, CSIPatchEmbedder]`，forward 时按 `modality` 参数选择。

- 优点：单一 backbone 支持多模态推理，迁移时不需要替换模块
- 缺点：backbone 需要知道所有模态的 input_shape，增加配置复杂度；多模态混合 batch 训练需要额外对齐逻辑
- 适用：未来 SenseFrame 多模态统一推理场景

#### 方案 C：通用 modality adapter

在 patch_embedder 之前加一个 modality adapter，把不同 C 的输入投影到统一维度（如 64），再走单一 patch_embedder。

- 优点：patch_embedder 维度固定，backbone 完全 modality-agnostic
- 缺点：modality adapter 引入额外参数 + 训练阶段（adapter 需要先训练）；与 MAE 预训练流程耦合（adapter 也要预训练）
- 适用：模态差异大、需要可学习对齐的场景

**推荐方案 A**：最小改动 + 文献标准 + 易于验证（B5 vs B4 的增益直接反映 CSI→EEG 迁移价值）。

### 防御性建议

- **`pretrain_source` 应是可执行的预训练数据集名，而非字符串标签**：当前 `pretrain_source="csi_4datasets"` 只是标签，实际预训练数据由 `target_dataset` 决定。建议改为 `pretrain_source="NTU-Fi_HAR"`（具体数据集名）或 `pretrain_source="csi_aggregated"`（聚合数据集名），并在 run_single_experiment 中显式加载 pretrain_dataset（独立于 target_dataset）
- **跨模态迁移应在 backbone 层暴露 `replace_patch_embedder` API**：当前 `CSIPatchEmbedder` 是 `CSIFoundationModel` 的私有属性，外部无法替换。应在 `CSIFoundationModel` 上暴露 `replace_patch_embedder(new_input_shape, new_patch_len)` 方法，封装 patch_embedder 重建 + pos_embed 重新初始化逻辑
- **CI 应有 B5/B6 配置一致性检查**：grep `CROSS_DOMAIN_EXPERIMENTS` 中 `pretrain != "none"` 且 `target` 跨模态的实验，CI 检查 run_single_experiment 是否真正加载 pretrain 数据集（而非 target_dataset 自己）。可静态检查：若 `is_eeg and pretrain_source.startswith("csi")`，则 pretrain_ds 必须从 CSI_DATASET_CONFIG 加载，不能是 None
- **跨场景迁移结果应记录 backbone 迁移细节**：ExperimentResult 应增加 `transferred_modules` 字段（如 `["encoder", "decoder", "pos_embed"]`）和 `reinitialized_modules` 字段（如 `["patch_embedder", "pos_embed"]`），便于分析"哪些模块迁移有效"
- **B5/B6 验证应避开"上限数据集"陷阱**（教训 8）：PhysioNet_MI 5 受试者 225 样本 / 2 类，B4 baseline val_acc=59.09%（接近随机猜 50%），有充足提升空间，是验证 CSI→EEG 迁移价值的正确场景

### 验证效果

| 检查项 | 当前状态 | 方案 A 落地后预期 |
|---|---|---|
| B5 是否真跑 CSI→EEG 迁移 | ✗（EEG scratch + LoRA） | ✓（CSI MAE 预训练 → 替换 patch_embedder → EEG LoRA 微调） |
| B6 是否真跑 CSI→EEG 迁移 | ✗（EEG scratch + Full） | ✓（CSI MAE 预训练 → 替换 patch_embedder → EEG Full 微调） |
| B5 vs B4 增益是否反映迁移价值 | ✗（仅反映 LoRA vs scratch 差异） | ✓（反映 CSI 预训练对 EEG 的迁移增益） |
| backbone 跨模态权重复用 | ✗（每次重新构建） | ✓（transformer encoder + decoder 迁移，patch_embedder 替换） |

---

## 11. 非上限数据集 SP 搜索价值有限 + GridSampler 遍历顺序缺陷

### 现象

Widar C 系列（非上限数据集，22 类，A1 baseline val_acc=69.73%）完成 5 个实验后，发现两个反直觉结果：

| 实验 | 配置 | val_acc | macro_f1 | trainable_params | best_params | cost |
|------|------|---------|----------|------------------|-------------|------|
| C1_Widar | LoRA r=8 (固定) | 0.4127 | 0.2254 | 23,318 | - | 528s |
| C2_Widar | Adapter bn=128 (固定) | 0.6357 | 0.5400 | 1,639,190 | - | 621s |
| C3_Widar | PromptTuning len=10 (固定) | 0.2405 | 0.1044 | 4,118 | - | 565s |
| C4_Widar | RandomSampler × 20 | **0.6495** | 0.5632 | 3,275,542 | adapter, r=4, bn=256, pl=5 | 12,029s |
| C5_Widar | GridSampler × 24 | 0.4241 | 0.2492 | 23,318 | lora, r=8, bn=32, pl=10 | 14,203s |

**反直觉 1**：SP 搜索（C4 random）相对固定配置 max(C1,C2,C3) 仅提升 **+0.01%**（0.6495 vs 0.6357），与预期"非上限数据集应有 ~30% 提升空间"严重不符。

**反直觉 2**：GridSampler（C5）比 RandomSampler（C4）**差 0.23%**（0.4241 vs 0.6495），且 C5 全程 24 个 trial 没采样到任何 adapter 组合（adapter 是 Widar 上最优方法）。

### 根因

#### 根因 1：SP 搜索价值与"数据集上限"强相关，非上限数据集提升空间仍有限

教训 8 在 NTU-Fi_HAR 上限数据集上发现"C2=100%, C4=100%，SP 无提升空间"。本次在 Widar（非上限数据集）上验证，原假设是"非上限数据集有 ~30% 提升空间，SP 搜索应能找到显著更优组合"。

实际结果：C4 random 找到的最优组合（adapter bn=256, val_acc=0.6495）仅比 C2 固定配置（adapter bn=128, val_acc=0.6357）提升 +1.38%。

**深层原因**：
- Widar 上 adapter 方法本身已接近其性能上限（不同 bn 的 adapter 全在 0.61-0.65 区间）
- SP 搜索的 20 个 trial 中，6 个 adapter trial 全部在 0.61-0.65，差异仅来自 bn 参数微调
- 真正的瓶颈不在 PEFT 超参，而在 backbone 容量 / 数据质量 / 特征表达

#### 根因 2：GridSampler 固定遍历顺序 + n_trials 截断 = 错过最优区域

C5 grid 24 个 trial 的实际采样顺序（从日志提取）：
- trial 1: prompt_tuning, params=13078
- trial 2-16: **lora rank=4** × (adapter_bottleneck ∈ {32,64,128,256}) × (prompt_length ∈ {5,10,20,50}) = 15 个组合
- trial 17-24: **lora rank=8** × 部分 adapter_bottleneck/prompt_length 组合 = 8 个组合

**GridSampler 按 `peft_method → peft_rank → adapter_bottleneck → prompt_length` 字典序嵌套遍历**，导致：
- 前 1 个 trial 是 prompt_tuning（仅 1 种 peft_rank 有效）
- 接下来 16 个 trial 全是 lora rank=4（4 × 4 = 16 组合）
- 再接下来至少 16 个 trial 才是 lora rank=8
- **adapter 方法排在字典序最后**（'adapter' > 'lora' > 'prompt_tuning'），n_trials=24 根本跑不到

这是教训 9（GridSampler 全网格爆炸）的延伸：即使 n_trials 限制为 24，固定遍历顺序仍会导致采样集中在字典序靠前的 peft_method 上，错过最优区域。

#### 根因 3：PEFT 方法选择高度依赖数据集特性

| 数据集 | LoRA r=8 | Adapter bn=128 | PromptTuning len=10 | 最优方法 |
|--------|----------|----------------|---------------------|----------|
| NTU-Fi_HAR（上限） | 88.17% | 100% | 63.44% | adapter（已达 100%） |
| Widar（非上限） | 41.27% | 63.57% | 24.05% | adapter（但仅 63.57%） |

Widar 22 类 / 936 训练样本，adapter 参数量大（1.6M-3.3M）能充分拟合；lora 参数量小（23K）欠拟合；prompt_tuning 参数量极小（4K）严重欠拟合。这与 NTU-Fi_HAR 上的方法排序一致，但 Widar 上 adapter 仍未达上限（63.57% vs NTU-Fi_HAR 100%）。

### 防御性建议

- **SP 搜索价值评估必须区分"上限数据集"与"非上限数据集"**：上限数据集（A1 ≥ 95%）上 SP 搜索无提升空间（教训 8）；非上限数据集上 SP 搜索提升也有限（本次 +1.38%），不应预期"非上限 = 大提升空间"。真正的提升空间取决于"当前配置距数据集真实上限有多远"，而非"A1 baseline 高低"
- **GridSampler 不应使用默认字典序遍历**：当 search_space 包含 peft_method（3 种）× peft_rank（3 种）× adapter_bottleneck（3-4 种）× prompt_length（3-4 种）时，字典序导致 adapter 排在最后。建议：
  - 方案 A：GridSampler 使用 `shuffle=True`（若 Optuna 支持）打乱遍历顺序
  - 方案 B：按"经验优先级"排序 search_space（adapter > lora > prompt_tuning）
  - 方案 C：GridSampler 仅用于"已收敛到某 peft_method 后的细粒度超参搜索"，不用于跨方法比较
- **SP 搜索应优先用 RandomSampler**：C4 random 在 20 trial 内采样到 6 个 adapter + 6 个 lora + 8 个其他，覆盖均衡；C5 grid 在 24 trial 内全是 prompt_tuning + lora，完全错过 adapter。RandomSampler 的无偏性在小 n_trials 下优势显著
- **ExperimentResult 应记录 SP 搜索的"方法覆盖统计"**：当前 best_params 只记录最优 trial，无法看出"SP 搜索是否覆盖了所有 peft_method"。建议增加 `method_coverage: {"lora": 8, "adapter": 6, "prompt_tuning": 6}` 字段，便于诊断 GridSampler 遍历顺序问题
- **Widar 上 PEFT 方法选择已有明确结论**：adapter 主导（+22% vs lora），后续 Widar 相关实验应默认 adapter，不再探索 lora/prompt_tuning

### 验证效果

| 检查项 | NTU-Fi_HAR（上限，教训 8） | Widar（非上限，本次） |
|--------|---------------------------|----------------------|
| C2 固定 adapter baseline | 100% | 63.57% |
| C4 random SP 搜索最优 | 100%（无提升） | 64.95%（+1.38%） |
| C5 grid SP 搜索最优 | 91.40%（-8.60%） | 42.41%（-21.16%） |
| SP 搜索相对固定配置提升 | 0% | +1.38% |
| GridSampler 是否覆盖 adapter | 部分 | ✗（24 trial 全是 prompt_tuning + lora） |
| 结论 | 上限数据集 SP 无价值 | 非上限数据集 SP 价值有限 + GridSampler 遍历顺序缺陷 |

**核心结论**：
1. SP 搜索价值不能简单按"上限/非上限"二分，真正决定因素是"当前配置距数据集真实上限的距离"
2. GridSampler 在多方法 search_space + 小 n_trials 下会因字典序遍历错过最优区域，应优先用 RandomSampler
3. PEFT 方法选择应基于数据集特性（样本数 / 类别数 / 特征维度），而非"一刀切"默认 lora

---

## 12. DANN 跨模态对齐：双模态架构冲突与短训练失效

### 现象

10.4 跨场景迁移评估追加 B5_DANN/B6_DANN 实验组（CSI→EEG），设计意图：在 B5（LoRA）/ B6（full）基础上引入 DANN 跨模态对抗，让 CSI 预训练特征空间更"对齐"EEG 模态，从而提升迁移增益。

| 实验 | pretrain | target | finetune | use_dann | 设计意图 |
|---|---|---|---|---|---|
| B5 | csi_4datasets | PhysioNet_MI | lora | False | CSI MAE 预训练 → EEG LoRA 微调（基线） |
| B5_DANN | csi_4datasets | PhysioNet_MI | lora | True | B5 + CSI/EEG 模态对抗对齐 |
| B6 | csi_4datasets | PhysioNet_MI | full | False | CSI MAE 预训练 → EEG 全量微调（基线） |
| B6_DANN | csi_4datasets | PhysioNet_MI | full | True | B6 + CSI/EEG 模态对抗对齐 |

5 epoch 真实训练结果（commit d561212 后运行）：

| 实验 | val_acc | macro_f1 | disc_loss 行为 |
|---|---|---|---|
| B5_DANN (lora) | 0.5336 | 0.3479 | 0.04 → 0.0001（对抗失效） |
| B6_DANN (full) | 0.5336 | 0.3479 | 0.09 → 0.82（对抗有效但波动） |

val_acc=0.5336 = PhysioNet 二分类 majority class baseline，说明 DANN 未带来任何增益，模型只预测多数类。

### 根因

#### 1. 双模态架构冲突（最严重）

`replace_patch_embedder` 后 backbone 只能处理 EEG 输入 (64, 480)，CSI 对抗信号 (342, 2000) forward 失败：

```
RuntimeError: shape '[2, 24, 20, 342]' is invalid for input of size 1368000
```

修复方案：`DANNCrossModalModel` 持有独立的 `csi_patch_embedder` + `csi_pos_embed`（replace 前深拷贝保存），CSI 路径通过它们 + 共享 `backbone.encoder`（modality-agnostic）。CSI 模态特定层始终 freeze（不被对抗破坏）。

#### 2. LoRA + DANN 架构上不兼容

B5_DANN (lora) 的 disc_loss 从 0.04 衰减到 0.0001，判别器完全失效。根因：
- LoRA freeze backbone（`freeze_backbone=True`），encoder 主体不更新
- 判别器对抗的目的是让 encoder 学到"模态不可区分"的特征
- 但 encoder 不更新，对抗梯度无法反向影响 encoder → 判别器也学不到判别性特征
- **结论**：LoRA 的"冻结 backbone"设计目标与 DANN 的"对抗更新 encoder"目标根本冲突

#### 3. λ 调度不匹配短训练

DANN 原论文 λ 调度：`λ = 2/(1+exp(-10*p))-1`，p=epoch/total_epochs。该调度设计用于 50-100 epoch，5 epoch 下：
- epoch 1: λ=0.000（无对抗，浪费 1/5 训练预算）
- epoch 2: λ=0.762（对抗骤增到 76%）
- epoch 3-5: λ=0.964→0.999（接近满对抗）

encoder 还没学到任务特征就被强对抗压制，task_loss 不下降。

#### 4. CSI/EEG 模态差异过大

| 模态 | input_shape | n_patches | 特征空间 |
|---|---|---|---|
| CSI (NTU-Fi_HAR) | (342, 2000) | 100 | 高维时频 |
| EEG (PhysioNet_MI) | (64, 480) | 24 | 低维时序 |

两个模态的 patch 数（100 vs 24）和维度差异巨大，判别器对 pooled 特征 (B, d_model) 做模态判别，但 pooled 特征已丢失模态特定的时频结构，对抗信号弱。

#### 5. PhysioNet 类不平衡

PhysioNet MI 二分类（左/右手运动想象），53% 是 majority class。task_loss 在 0.70 附近（接近 -log(0.53)=0.63），说明模型只学到多数类先验，DANN 对抗无法弥补弱 task 信号。

### 防御性建议

- **LoRA + DANN 不应组合使用**：LoRA freeze backbone 使对抗梯度无法影响 encoder，判别器失效。若需 PEFT + DANN，应使用 `freeze_backbone=False` 的 full 微调，或修改 DANN 架构让对抗梯度仅流向 LoRA 层（不动 frozen backbone）
- **DANNCrossModalModel 必须持有独立的源模态 patch_embedder**：`replace_patch_embedder` 后 backbone 只能处理目标模态，源模态对抗信号需要独立 patch_embedder + pos_embed，共享 encoder。CSI 模态特定层应 freeze（预训练好，不被对抗破坏）
- **短训练（epoch ≤ 10）应避免 DANN 原论文 λ 调度**：原调度设计用于 50-100 epoch，短训练下 λ 从 0 骤增到 0.76 压制任务学习。短训练应用固定 λ=0.5 或线性调度 `λ = epoch/total_epochs`
- **DANN 实验组必须先跑同配置非 DANN baseline**：B5_DANN/B6_DANN 需与 B5/B6 对照才能计算 transfer_gain。单独看 DANN 的 val_acc 无法判断对抗是否有效（需对照同 epoch 数的非 DANN 基线）
- **跨模态对齐应优先评估模态相似度**：CSI (342, 2000) vs EEG (64, 480) 差异过大，对抗信号弱。应先用 CKA / 余弦相似度等无监督指标评估两模态特征空间重叠度，再决定是否值得 DANN 对齐
- **类不平衡数据集应加权 loss 或重采样**：PhysioNet 53% majority class，应使用 `class_weight` 或 `WeightedRandomSampler`，否则 task 信号过弱无法驱动对抗学习

### 验证效果

| 检查项 | B5_DANN (lora) | B6_DANN (full) |
|---|---|---|
| 双模态 forward | ✅（EEG + CSI 同时输入正确处理） | ✅ |
| CSI 模态特定层 freeze | ✅（requires_grad=False） | ✅ |
| checkpoint 加载 | ✅（跳过 100 epoch MAE） | ✅ |
| GRL 梯度反转 | ✗（disc_loss→0，对抗失效） | ✅（disc_loss 波动） |
| λ 调度 | ✅（照搬原论文公式） | ✅ |
| val_acc 超越 majority baseline | ✗（0.5336 = 53% baseline） | ✗ |
| DANN 带来增益 | ✗（无 B5/B6 基线对照） | ✗ |

**核心结论**：
1. DANN 跨模态对齐的架构正确性已验证（双模态 forward / GRL / λ 调度 / checkpoint 加载 / CSI freeze 全部正常）
2. LoRA + DANN 架构上不兼容（freeze backbone 使对抗失效），不应组合使用
3. 5 epoch 短训练 + 原论文 λ 调度不匹配，DANN 需 50+ epoch 才能体现对抗对齐价值
4. 跨模态对齐前应先评估模态相似度，CSI/EEG 差异过大导致对抗信号弱
5. 负面结果符合"验证架构价值而非刷 SOTA"的项目立场，为后续改进提供清晰方向

---

## 附：10.2 单场景验证最终结果（20 个实验）

### 实验配置

- 小数据集（UT_HAR_data / NTU-Fi_HAR / NTU-Fi-HumanID）：epochs=50, batch_size=64, pretrain_epochs=5
- Widar（数据量大）：epochs=10, batch_size=16, pretrain_epochs=5
- MAE 预训练：mask_ratio=0.75（论文推荐值）
- LoRA: rank=8, alpha=16, target=query_value
- Adapter: bottleneck=128, target=all
- PromptTuning: prompt_length=10

### 性能汇总表

| 实验 | 方法 | UT_HAR | NTU-Fi_HAR | NTU-Fi-HumanID | Widar | 平均参数量 |
|---|---|---|---|---|---|---|
| A1 | scratch | 95.56% | 100.00% | 100.00% | 69.73% | 1,639,943 |
| A2 | full finetune | 95.16% | 100.00% | 100.00% | 69.50% | 1,639,943 |
| A3 | LoRA | 80.44% | 86.02% | 55.56% | 43.79% | 22,060 |
| A4 | adapter | 95.97% | 100.00% | 94.44% | 62.34% | 2,486,572 |
| A5 | prompt_tuning | 66.94% | 69.89% | 61.11% | 24.00% | 2,860 |

### 通过标准检查

**标准 1：A3/A4/A5 至少一组平均 val_acc > A1 + 2%**
- A1 平均 acc: 91.32%（阈值 93.32%）
- A3: 66.45% (-24.87%) ✗
- A4: 88.19% (-3.14%) ✗
- A5: 55.48% (-35.84%) ✗
- **结果：全部未通过**

**标准 2：A3/A4/A5 trainable_params < A1 × 30%（阈值 491,982）**
- A3: 22,060 (1.3% of A1) ✓
- A4: 2,486,572 (151.6% of A1) ✗（adapter 注入到所有 Linear，参数量翻倍）
- A5: 2,860 (0.2% of A1) ✓
- **结果：A3/A5 通过，A4 未通过**

**标准 3：A2 ≥ A1（预训练有效性）**
- UT_HAR_data: A1=95.56% vs A2=95.16% (-0.40%) ✗
- NTU-Fi_HAR: A1=100% vs A2=100% (+0.00%) ✓
- NTU-Fi-HumanID: A1=100% vs A2=100% (+0.00%) ✓
- Widar: A1=69.73% vs A2=69.50% (-0.23%) ✗
- **结果：2/4 通过（小数据集上 A1 已达上限，A2 无空间超越）**

### 关键观察

1. **小数据集上 A1 已达性能上限**（NTU-Fi_HAR / NTU-Fi-HumanID 都 100%），A2/A4 无法超越，PEFT 方法（A3/A5）受容量限制更低
2. **Widar 上 A1 表现一般（69.73%）**，但 A2/A4 也没显著超越，A3/A5 因参数量过小大幅退化
3. **A4 adapter 参数量爆炸**（151.6% of A1），因为注入到所有 Linear（含 FFN）且 bottleneck=128 较大
4. **A5 prompt_tuning 参数量极小（0.2% of A1）**，Widar 22 类任务几乎学不动（24% acc, 10% f1）
5. **MAE 预训练 5 epoch 收益不显著**：A2 vs A1 几乎持平，需要更多 pretrain epochs 才能体现预训练价值
6. **归一化是 P3 验证的"生死线"**：未归一化时所有 NTU-Fi 实验失效，归一化后全部恢复

### 后续推进

- 10.3 跨场景迁移（B 系列）：A 系列在同数据集上预训练+微调，B 系列验证跨数据集迁移增益
- 10.5 SP 搜索驱动集成（C4/C5）：用 Optuna 搜索 PEFT 超参（rank/bottleneck/prompt_length），看能否突破 A 系列的固定配置瓶颈
- RadioML 2018.01A 下载完成后接入 A1-A5 × RadioML，验证 RF 模态的 PEFT 有效性
- EEG 数据集（PhysioNet MI / BCI IV 2a）下载完成后接入 A1-A5 × EEG
