# NAS 与 AutoAugment 参考

> 锚点：SenseFrame 神经架构搜索（NAS）与数据增强搜索（AutoAugment）实现

## 概述

SenseFrame 通过 SP（Search Protocol）协议统一驱动两类搜索任务：

- **NAS（神经架构搜索）**：搜索模型架构参数（cell_type、n_layers、hidden_dim、
  n_heads 等），用 `ArchitectureBuilder` 将参数翻译为 `nn.Module`，通过
  `module_factory` 注入 Pipeline，替换 scene 默认模型。
- **AutoAugment（数据增强搜索）**：搜索数据增强策略（op、magnitude、
  probability 的组合），用 `AutoAugmentPolicyBuilder` 将参数翻译为 transform
  函数，通过 `datamodule_factory` 注入 Pipeline，替换 scene 默认 transform。

两者均满足 SP Sampler Protocol（`name` + `sample` + `warm_start`），注册到 SP
采样器注册表后可被 `StudyManager` 的 ask/tell 循环统一调度。搜索空间通过
`to_sp_search_space()` 转换为标准 SP `SearchSpace`，复用 SP 全部采样基础设施。

## NAS 神经架构搜索

### 搜索空间

`ArchitectureSearchSpace`（`senseframe/nas/search_space.py`）是 NAS 搜索空间的
DSP 合规 dataclass。

**模块级常量**：

| 常量 | 取值 |
|---|---|
| `SUPPORTED_CELL_TYPES` | `["conv1d", "rnn", "attention"]` |
| `SUPPORTED_ACTIVATIONS` | `["relu", "gelu", "tanh", "elu"]` |
| `SUPPORTED_RNN_TYPES` | `["lstm", "gru"]` |
| `SUPPORTED_ATTENTION_ACTIVATIONS` | `["gelu", "relu"]` |
| `SUPPORTED_N_HEADS` | `[2, 4, 8]` |

> 注：`hybrid` 是 `cell_type` 的合法取值（conv1d + rnn 级联），但未列入
> `SUPPORTED_CELL_TYPES`，由 `ArchitectureSearchSpace._build_default_parameters`
> 单独处理。

**`ArchitectureParameterSpec` 字段**（单个架构参数规格）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 参数名（如 `"cell_type"` / `"n_layers"`） |
| `type` | `str` | `"categorical"` / `"int"` / `"float"` |
| `choices` | `Optional[List[Any]]` | categorical 可选值 |
| `low` / `high` | `Optional[float]` | int/float 上下界 |
| `log` | `bool` | 是否对数采样 |
| `step` | `Optional[float]` | int 步长 |
| `default` | `Optional[Any]` | 默认值（用于初始化） |

**`ArchitectureSearchSpace` 字段**：

| 字段 | 类型 | 默认值 |
|---|---|---|
| `schema_version` | `str` | `"1.0.0"` |
| `cell_types` | `List[str]` | `["conv1d", "rnn"]` |
| `parameters` | `List[ArchitectureParameterSpec]` | `[]`（自动生成） |
| `custom_params` | `Dict[str, ArchitectureParameterSpec]` | `{}` |

`__post_init__` 在 `parameters` 为空时调用 `_build_default_parameters`：按
`cell_types` 拼接对应默认参数集（`_default_conv1d_params` /
`_default_rnn_params` / `_default_hybrid_params` / `_default_attention_params`），
合并同名 categorical choices，最后由 `custom_params` 覆盖。

**默认参数集摘要**：

- `conv1d`：`n_layers`(1-8), `hidden_dim`(16-512, log), `activation`,
  `kernel_size`({3,5,7}), `dropout`(0-0.5)
- `rnn`：`n_layers`(1-8), `hidden_dim`(16-512, log), `activation`({tanh,relu}),
  `rnn_type`({lstm,gru}), `bidirectional`({True,False}), `dropout`(0-0.5)
- `hybrid`：conv1d 参数 + `rnn_type` + `bidirectional`
- `attention`：`n_layers`(1-8), `d_model`(16-512, log), `n_heads`({2,4,8}),
  `dropout`(0-0.5), `activation`({gelu,relu})

**DSP 方法**：`schema()`（JSON Schema 自省）、`describe()`、`to_dict()` /
`from_dict()`、`to_sp_search_space()`（转 SP `SearchSpace`）、
`get_param(name)`、`validate_params(params)`。

### 架构构建器

`ArchitectureBuilder`（`senseframe/nas/builder.py`）的 `build` 方法按
`cell_type` 分发到四个具体网络类：

```python
def build(self, arch_params, input_shape, num_classes) -> nn.Module
```

| `cell_type` | 实现类 | 输入约定 |
|---|---|---|
| `conv1d` | `Conv1dNet` | `(channels, length)`，堆叠 Conv1d+BN+Act+Dropout，AdaptiveAvgPool1d+Linear |
| `rnn` | `RNNNet` | 转置为 `(length, channels)`，LSTM/GRU 取最后时刻 hidden |
| `hybrid` | `HybridNet` | Conv1d 特征 → 转置 → RNN 时序建模 |
| `attention` | `AttentionNet` | 转置为 `(length, channels)` 作 token 序列，TransformerEncoder + mean pool |

`AttentionNet` 在 `in_channels != d_model` 时插入 `nn.Linear` 投影层；
`dim_feedforward = d_model * 4`；分类头对序列做 mean pooling 后接 `Linear`。

辅助函数：`_activation_module(name)` 构造激活模块，
`_rnn_cell(...)` 构造 LSTM/GRU 单元（`batch_first=True`，单层时强制
`dropout=0`）。

### 采样器

三种 NAS 采样器均满足 SP Sampler Protocol，通过 `register_sampler` 注册：

| 采样器类 | 注册名 | 算法 | 关键参数 |
|---|---|---|---|
| `EvolutionarySampler` | `"evolutionary"` | 进化策略（锦标赛选择 + 变异） | `population_size=20`, `mutation_rate=0.3`, `tournament_size=3`, `direction="maximize"`, `seed=None` |
| `ENASSampler` | `"enas"` | 权重共享 NAS（controller LSTM 采样） | `controller=None`, `shared_weights=None`, `controller_hidden=64`, `seed=None` |
| `DARTSSampler` | `"darts"` | 可微架构搜索（双优化 α ↔ w） | `arch_alpha=None`, `lr_arch=3e-4`, `seed=None` |

**EvolutionarySampler 算法**：
1. 初始化阶段（`len(population) < population_size`）：`_random_init` 随机生成个体。
2. 进化阶段：`_tournament_select` 从已评估个体中选 `tournament_size` 个取最优作父代，
   `_mutate` 对每个参数以 `mutation_rate` 概率扰动（categorical 换另一个 choice，
   int/float 在 ±10% 范围内扰动）。
3. `_sync_population_from_history` 从 SP history 同步 fitness，population 自动修剪到
   `population_size * 2`。
4. `warm_start(source_history)` 将源数据集历史注入 population（ε4 元学习受益）。

**ENASSampler 算法**：
- `controller` 为 `None` 时走 `_random_sample` 随机 fallback。
- `controller` 非空时走 `_controller_sample`：用 dummy input 做 forward，
  对 categorical 参数取 logits → softmax → multinomial 采样；
  对 int/float 参数用 sigmoid 映射到 `[low, high]`。
- `warm_start` 为 no-op（ENAS 通过 controller 训练，不从历史偏向）。
- 简化决策：controller 一次性输出所有架构决策（非 autoregressive LSTM）。

### DARTS 可微搜索

`DARTSSampler`（`senseframe/nas/darts.py`）实现可微架构搜索的核心机制：

**双优化**：
- `arch_alpha: Dict[str, torch.Tensor]` 是架构参数（`requires_grad=True`），
  每个 categorical 参数对应一个 α 向量（长度 = `len(choices)`）。
- `sample()`：`softmax(alpha)` → `argmax` 选离散架构，再由
  `_discretize_to_arch_params` 翻译回 `arch_params` dict。
- `update(gradient)`：用 Adam 优化器（`lr=lr_arch`）更新 α。梯度经
  `g.detach().clone()` 切断计算图引用，避免资源泄露。
- `cleanup()`：释放 α tensor 和 optimizer 引用。

**真实超网集成**（P1.3）：
- `attach_supernet(supernet)`：将 `DARTSSupernet` 实例附到 sampler，
  使 `sample()` / `discretize()` 基于真实超网 α。
- `detach_supernet()`：解除附着。

**BatchNorm 双优化修复**（P2.3-3）：α 更新阶段临时把所有 BN 的 momentum 设为 0.0
（BN 仍用 batch stats 归一化，但 running stats 不再更新），更新后恢复原 momentum。
辅助函数 `_set_bn_momentum` / `_restore_bn_momentum`。

**`DARTSPipelineRun`**：DARTS 专用 PipelineRun（不继承标准 Pipeline，因双优化不符合
stage-based 流程）。构造参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `sampler` | — | `DARTSSampler` 实例 |
| `builder` | — | `ArchitectureBuilder`（`use_real_supernet=True` 时可传 `None`） |
| `search_space` | — | SP `SearchSpace` |
| `input_shape` | — | 模型输入形状（不含 batch 维） |
| `num_classes` | — | 输出类别数 |
| `n_epochs` | `50` | 训练 epoch 数 |
| `lr_w` | `0.025` | w 学习率（SGD + momentum=0.9 + weight_decay=3e-4） |
| `lr_arch` | `3e-4` | α 学习率（Adam + betas=(0.5,0.999) + weight_decay=1e-3） |
| `use_real_supernet` | `False` | 是否走真实可微超网路径 |
| `supernet_kwargs` | `None` | 传递给 `DARTSSupernet` 的额外参数 |

`run(train_loader, val_loader)` 按 `use_real_supernet` 分发：

- `_run_simplified`（默认）：用 `ArchitectureBuilder` 构造单个架构作超网近似，
  α 用 `randn_like * 0.01` 近似梯度。返回 `{"best_arch", "final_alpha", "history"}`。
- `_run_real_supernet`（P1.3）：构造 `DARTSSupernet`，α 真实参与 forward（softmax
  加权），通过验证集 backward 真实可微更新。返回 `{"best_arch", "final_alpha",
  "history", "supernet_arch"}`。

两条路径均用 `_InfiniteLoader` 循环迭代 loader，并在 `finally` 中显式释放
iterator / optimizer / supernet 引用（含 `gc.collect()` 和
`torch.cuda.empty_cache()`）。`_InfiniteLoader` 含空 loader 防御（连续 2 次
`StopIteration` 视为空 loader，raise 清晰错误）。

### 真实超网

`DARTSSupernet`（`senseframe/nas/supernet.py`）实现真正的可微超网。

**候选 op 列表**（`OP_NAMES`）：

```python
OP_NAMES = ["conv3", "conv5", "avgpool", "maxpool", "identity"]
```

所有 op 输入输出形状一致 `(B, C, L) → (B, C, L)`（pool 用 `stride=1 + padding`，
`identity` 在 `c_in != c_out` 时退化为 1x1 Conv1d 对齐 channel）。

**`DARTSCell`**：可微 cell，M 个候选 op 并行计算，α softmax 加权混合。

```python
forward(x):
    weights = softmax(alpha, dim=-1)   # (M,)
    outputs = [op(x) for op in ops]    # M 个 (B, c_out, L)
    return sum(w_i * out_i)            # 加权求和（autograd 跟踪到 α）
```

- `alpha: nn.Parameter(torch.randn(M) * 0.001)`（P2.3-2 修复：小随机初始化打破对称性）。
- `discretize()`：`argmax(alpha)` → op 名。

**`DARTSSupernet` 字段**：

| 字段 | 类型 | 默认值 |
|---|---|---|
| `input_shape` | `Tuple[int, ...]` | — |
| `num_classes` | `int` | — |
| `n_cells` | `int` | `3`（必须 ≥ 1） |
| `c_stem` | `int` | `32` |
| `c_cell` | `int` | `64` |
| `op_names` | `List[str]` | `OP_NAMES` |

结构：stem（Conv1d+BN+ReLU）→ N 个 `DARTSCell` 串联 →
`AdaptiveAvgPool1d(1)` + `Linear(c_cell, num_classes)`。

**参数分离**（双优化基础）：
- `w_parameters()`：除 α 外的模型权重（stem + cells.ops + classifier），用 SGD 更新。
  精确匹配 `cells.<idx>.alpha` 命名路径过滤（P2.3-1 修复）。
- `alpha_parameters()`：`cells.alpha`，用 Adam 更新。
- `alpha_dict()`：返回 `{cell_idx_str: alpha_tensor}`，与 `DARTSSampler.arch_alpha` 对齐。

**离散化**：
- `discretize()`：每个 cell `argmax α` → op 名，返回 `{"cell_0": "conv3", ...}`。
- `build_discrete_model()`：返回 `_DiscreteSupernet`，每个 cell 只保留 argmax 选中的 op。
  **参数共享**：直接引用原 supernet 权重（不做深拷贝），离散化后继续微调时复用超网
  已训练权重；需独立权重请用 `copy.deepcopy`。

### NAS 模块工厂

`make_nas_module_factory`（`senseframe/nas/__init__.py`）构造 NAS
`module_factory`，注入 `config.module_factory`：

```python
def make_nas_module_factory(
    arch_params: Dict[str, Any],
    input_shape: Tuple[int, ...],
    builder: Optional[ArchitectureBuilder] = None,
):
```

返回的 `module_factory(model=None, **kwargs)` 符合 Pipeline `stage_build` 契约：
1. 忽略 scene 构造的默认 `model`（NAS 用自己的 `arch_params` 构造）。
2. 强制要求 `kwargs["num_classes"]`（框架不猜测数据集类别数，由
   `DatasetSpec.num_classes` 派生）。
3. `ArchitectureBuilder.build(arch_params, input_shape, num_classes)` 构造 `nn.Module`。
4. 用 `GenericLightningModule` 包装（复用训练/验证/指标逻辑）。

## AutoAugment 数据增强搜索

### 增强搜索空间

`AugmentationSearchSpace`（`senseframe/autoaugment/search_space.py`）是增强搜索
空间的 DSP 合规 dataclass。

**模块级常量**：

```python
SUPPORTED_AUGMENT_OPS = ["time_jitter", "freq_masking", "noise", "cutout", "none"]
DEFAULT_N_OPS_RANGE = (1, 3)
DEFAULT_MAGNITUDE_RANGE = (0.0, 1.0)
DEFAULT_PROBABILITY_RANGE = (0.0, 1.0)
```

**`AugmentationParameterSpec` 字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 参数名（如 `"op_0"` / `"magnitude_0"` / `"probability_0"`） |
| `type` | `str` | `"categorical"` / `"float"` |
| `choices` | `Optional[List[Any]]` | categorical 可选值（op_i 用） |
| `low` / `high` | `Optional[float]` | float 上下界（magnitude_i / probability_i 用） |
| `default` | `Optional[Any]` | 默认值 |

**`AugmentationSearchSpace` 字段**：

| 字段 | 类型 | 默认值 |
|---|---|---|
| `schema_version` | `str` | `"1.0.0"` |
| `ops` | `List[str]` | `list(SUPPORTED_AUGMENT_OPS)` |
| `n_ops` | `int` | `2`（约束 1-5） |
| `magnitude_range` | `Tuple[float, float]` | `(0.0, 1.0)` |
| `probability_range` | `Tuple[float, float]` | `(0.0, 1.0)` |

`__post_init__` 校验：`n_ops` ∈ [1, 5]，`ops` 非空，
`magnitude_range` / `probability_range` ∈ [0, 1] 且 low ≤ high。

`to_sp_search_space()` 为每个槽位 `i` 生成 3 个 SP `ParameterSpec`：
`op_i`（categorical）、`magnitude_i`（float）、`probability_i`（float），
共 `n_ops * 3` 个参数。

便捷工厂 `build_default_search_space(n_ops=2)` 返回默认搜索空间。

### 策略构建器

`AutoAugmentPolicyBuilder`（`senseframe/autoaugment/policy_builder.py`）将策略参数
dict 翻译为 transform 函数 `fn(x, y) -> (x, y)`。

```python
class AutoAugmentPolicyBuilder:
    def __init__(self, search_space: Optional[AugmentationSearchSpace] = None)
    def build(self, policy_params: Dict[str, Any]) -> Callable
    def build_eval_transform(self, policy_params: Dict[str, Any]) -> Callable
```

`build()` 流程：
1. 若提供 `search_space`，调用 `validate_params` 校验参数范围。
2. 解析策略：按 `i = 0, 1, ...` 提取 `(op_i, magnitude_i, probability_i)`，
   直到 `op_i` key 不存在。
3. 空策略返回 `IdentityTransform()`。
4. 非空策略返回 `AutoAugmentTransform(ops_chain)`。

`build_eval_transform()` 始终返回 `IdentityTransform()`（评估阶段不增强）。

**`AutoAugmentTransform`**（module-level callable 类，P5 P1-A 修复，可 pickle）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ops_chain` | `list` | `[(op_name, magnitude, probability), ...]` |
| `_rng` | `np.random.Generator` | 随机数生成器 |

`__call__(x, y=None)`：将输入转 numpy → 按顺序应用每个原语（按 `probability_i`
概率应用，`self._rng.random() > probability` 时跳过）→ 转回原类型返回 `(x, y)`。

**`IdentityTransform`**：module-level callable 类，`__call__(x, y=None) -> (x, y)`。
替代旧 `lambda x, y: (x, y)` 闭包，确保 DataLoader `num_workers>0` 时可序列化。

**增强算子注册表** `_AUGMENT_OPS`：`{op_name: callable(x, magnitude) -> x}`，
含 `"none": lambda x, m: x`。工具函数 `list_augment_ops()` /
`get_augment_op(name)`。便捷工厂 `make_policy_from_params(policy_params,
search_space=None)`。

### 增强原语

4 种增强原语（适配 WiFi CSI 时序信号），均操作 `np.ndarray`：

| 原语 | 函数 | 行为 |
|---|---|---|
| `time_jitter` | `_time_jitter(x, magnitude)` | 时序抖动：在时间轴（`axis=-1`）上随机偏移，`max_shift = max(1, int(length * magnitude * 0.1))`，用 `np.roll` 实现 |
| `freq_masking` | `_freq_masking(x, magnitude)` | 频域掩码：沿 channels 轴掩码连续片段（要求 `x.ndim >= 2` 且 `n_channels >= 2`），`n_mask = max(1, int(n_channels * magnitude * 0.3))`，掩码片段置 0 |
| `noise` | `_noise(x, magnitude)` | 高斯噪声：`noise_std = magnitude * 0.1 * (data_std + 1e-8)`，添加零均值高斯噪声 |
| `cutout` | `_cutout(x, magnitude)` | 随机遮挡：时序片段置零，`cutout_len = max(1, int(length * magnitude * 0.2))`，在 `[..., start:start+cutout_len]` 处置 0 |

`"none"` 为 no-op（`lambda x, m: x`），允许搜索"不增强"。

### 进化采样器

`AutoAugmentSampler`（`senseframe/autoaugment/sampler.py`）注册名
`"autoaugment"`，算法与 `EvolutionarySampler` 相同（进化策略），但默认
`mutation_rate=0.4`（更激进，增强策略搜索特性）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `population_size` | `20` | 种群大小 |
| `mutation_rate` | `0.4` | 变异概率（高于 NAS 的 0.3） |
| `tournament_size` | `3` | 锦标赛选择大小 |
| `direction` | `"maximize"` | 优化方向 |
| `seed` | `None` | 随机种子 |

方法与 `EvolutionarySampler` 一致：`sample()`、`warm_start(source_history)`、
`_sync_population_from_history`、`_tournament_select`、`_mutate`、
`population_size_actual()`、`evaluated_count()`。

### AutoAugment 模块工厂

`make_autoaugment_datamodule_factory`（`senseframe/autoaugment/__init__.py`）构造
AutoAugment `datamodule_factory`，注入 `config.datamodule_factory`：

```python
def make_autoaugment_datamodule_factory(
    policy_params: Dict[str, Any],
    base_train_transform: Optional[Callable] = None,
    base_eval_transform: Optional[Callable] = None,
    builder: Optional[AutoAugmentPolicyBuilder] = None,
    search_space: Optional[AugmentationSearchSpace] = None,
):
```

流程：
1. `builder.build(policy_params)` 构造 `aug_train_transform`。
2. `eval_transform = base_eval_transform or IdentityTransform()`。
3. 若提供 `base_train_transform`，用 `ChainedTransform([base, aug])` 组合
   （P5 P1-A：module-level callable 类，可 pickle）。
4. 返回 `datamodule_factory(train_dataset, test_dataset, **kwargs)`，
   从 `kwargs` 移除 scene 的 `train_transform` / `eval_transform`（避免重复），
   构造 `GenericDataModule`。

与 NAS `make_nas_module_factory` 的对称性：
- NAS：`arch_params → ArchitectureBuilder.build() → nn.Module → GenericLightningModule`
- AutoAugment：`policy_params → AutoAugmentPolicyBuilder.build() → transform fn → GenericDataModule`

## SP 协议集成

NAS 和 AutoAugment 均通过 SP 协议统一调度：

**搜索空间转换**：
- `ArchitectureSearchSpace.to_sp_search_space()` → SP `SearchSpace`
  （参数为 `cell_type` / `n_layers` / `hidden_dim` 等）。
- `AugmentationSearchSpace.to_sp_search_space()` → SP `SearchSpace`
  （参数为 `op_i` / `magnitude_i` / `probability_i`，共 `n_ops * 3` 个）。

**采样器注册**：
- `register_sampler("evolutionary", EvolutionarySampler)`
- `register_sampler("enas", ENASSampler)`
- `register_sampler("darts", DARTSSampler)`
- `register_sampler("autoaugment", AutoAugmentSampler)`

**ask/tell 循环**：
1. `StudyManager` 用转换后的 SP `SearchSpace` 注册 study。
2. `sampler.sample(search_space, history)` 采样参数。
3. NAS：参数经 `make_nas_module_factory` 注入 `config.module_factory`；
   AutoAugment：参数经 `make_autoaugment_datamodule_factory` 注入
   `config.datamodule_factory`。
4. `run_pipeline(config)` 执行训练，产出 `val_accuracy` 等指标。
5. `sampler` 通过 `_sync_population_from_history` 从 SP `history` 同步 fitness，
   下一轮 `sample` 基于更新后的 population 进化。

**warm_start**（ε4 元学习）：`EvolutionarySampler` 和 `AutoAugmentSampler` 的
`warm_start(source_history)` 将源数据集成功策略注入 population，跳过初始化阶段
直接进入进化阶段。`ENASSampler` 和 `DARTSSampler` 的 `warm_start` 为 no-op
（前者通过 controller 训练，后者通过梯度更新 α）。

## 使用示例

### NAS 搜索流程（进化算法）

```python
from senseframe.nas import (
    ArchitectureSearchSpace, ArchitectureBuilder, EvolutionarySampler,
    make_nas_module_factory,
)
from senseframe.search_protocol import StudyManager

# 1. 构造搜索空间（conv1d + rnn + attention）
arch_ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn", "attention"])
sp_ss = arch_ss.to_sp_search_space()

# 2. 构造采样器
sampler = EvolutionarySampler(population_size=20, mutation_rate=0.3, seed=42)

# 3. ask/tell 循环
manager = StudyManager(search_space=sp_ss, sampler=sampler, direction="maximize")
for trial_idx in range(20):
    params = manager.ask()
    config.module_factory = make_nas_module_factory(
        arch_params=params, input_shape=(30, 100),
    )
    # 调用方需通过 kwargs 传入 num_classes
    result = run_pipeline(config)  # 训练并返回 val_accuracy
    manager.tell(params, value=result["val_accuracy"])

best = manager.best_trial()
```

### DARTS 真实超网搜索流程

```python
from senseframe.nas import DARTSSampler, DARTSPipelineRun, ArchitectureSearchSpace

arch_ss = ArchitectureSearchSpace(cell_types=["conv1d"])
sp_ss = arch_ss.to_sp_search_space()

sampler = DARTSSampler(lr_arch=3e-4, seed=42)
run = DARTSPipelineRun(
    sampler=sampler,
    builder=None,                    # use_real_supernet=True 时可传 None
    search_space=sp_ss,
    input_shape=(30, 100),
    num_classes=7,
    n_epochs=50,
    lr_w=0.025,
    use_real_supernet=True,          # 走真实可微超网
    supernet_kwargs={"n_cells": 3, "c_stem": 32, "c_cell": 64},
)
result = run.run(train_loader, val_loader)
# result = {"best_arch": {"cell_0": "conv3", ...}, "final_alpha": ..., "history": ...}
```

### AutoAugment 搜索流程

```python
from senseframe.autoaugment import (
    AugmentationSearchSpace, AutoAugmentSampler,
    make_autoaugment_datamodule_factory,
)
from senseframe.search_protocol import StudyManager

# 1. 构造搜索空间（2 个 op 槽位）
aug_ss = AugmentationSearchSpace(n_ops=2)
sp_ss = aug_ss.to_sp_search_space()   # 6 个参数（op_0/mag_0/prob_0/op_1/mag_1/prob_1）

# 2. 构造采样器
sampler = AutoAugmentSampler(population_size=20, mutation_rate=0.4, seed=42)

# 3. ask/tell 循环
manager = StudyManager(search_space=sp_ss, sampler=sampler, direction="maximize")
for trial_idx in range(20):
    params = manager.ask()
    config.datamodule_factory = make_autoaugment_datamodule_factory(
        policy_params=params,
        base_train_transform=cfg.train_transform,
    )
    result = run_pipeline(config)
    manager.tell(params, value=result["val_accuracy"])

best = manager.best_trial()
```
