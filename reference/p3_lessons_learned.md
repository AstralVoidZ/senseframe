# P3 验证经验教训（10.2 单场景验证）

> 本文档记录 P3 验证 10.2 单场景性能验证（A1-A5 × 4 CSI 数据集）过程中踩过的坑、根因分析与防御性设计建议。
> 所有教训均对齐 [project_memory.md 第 129 条](../../project_memory.md) 的归一化历史根因。
> 更新日期：2026-07-19

## 目录

1. [归一化策略的应用位置冲突](#1-归一化策略的应用位置冲突)
2. [数据集原始 shape 必须实测，不能依赖文档/直觉](#2-数据集原始-shape-必须实测不能依赖文档直觉)
3. [scene 的 self_supervised 模式语义可能与实验意图不符](#3-scene-的-self_supervised-模式语义可能与实验意图不符)
4. [CLI 参数覆盖机制缺失导致快速验证成本高](#4-cli-参数覆盖机制缺失导致快速验证成本高)
5. [依赖管理：从 requirements.txt 迁移到 pyproject.toml](#5-依赖管理从-requirementstxt-迁移到-pyprojecttoml)
6. [PromptTuning 形状不匹配：d_model 推断错误 + 注入位置错误](#6-prompttuning-形状不匹配d_model-推断错误--注入位置错误)
7. [长 epoch 训练的日志频率陷阱](#7-长-epoch-训练的日志频率陷阱)

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
