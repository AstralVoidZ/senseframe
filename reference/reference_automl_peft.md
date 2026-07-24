# AutoML 与 PEFT 参考

> 锚点：SenseFrame AutoML 能力（损失搜索、元学习、PEFT 参数高效微调）

## 概述

SenseFrame AutoML 子系统建立在搜索协议（SP，`search_protocol`）的 `ask/tell` 循环之上，提供四类自动化能力：

| 模块 | 文件 | 作用 |
|------|------|------|
| `loss_search` | `senseframe/automl/loss_search.py` | 损失函数组合搜索（ε1） |
| `meta_learner` | `senseframe/automl/meta_learner.py` | 跨数据集元学习（ε4） |
| `peft_builder` | `senseframe/automl/peft_builder.py` | 4 种 PEFT 方法 + Full 微调构建器 |
| `peft_search` | `senseframe/automl/peft_search.py` | PEFT 微调策略搜索（SP 驱动） |

设计要点：
- `loss_search` 与 `peft_search` 对称：前者搜损失，后者搜微调策略，均通过 SP `ask/tell` 驱动
- `meta_learner` 不直接干预采样，而是将源数据集成功策略注入目标 Study 的 `tracker.history`，让 Sampler 自然读取作为采样偏向
- `peft_builder` 提供 LoRA / Adapter / PrefixTuning / PromptTuning / Full 五种微调方法，由 `PEFTBuilder.build` 分发
- 与 HPO（Optuna 路径）并存：HPO 搜数值超参，AutoML 搜策略组合，互不替换

## 损失函数搜索（loss_search）

### 搜索空间

`build_loss_search_space` 从损失注册表动态构造 SP `SearchSpace`：

```python
from senseframe.automl import build_loss_search_space

ss = build_loss_search_space(
    include_label_smoothing=True,   # 是否包含 label_smoothing 浮点参数
    extra_losses=None,              # 额外追加的 loss 名称
    include_self_supervised=False,  # 是否包含自监督损失（如 ent_loss）
)
```

参数说明：
- `include_label_smoothing: bool = True`：是否加入 `label_smoothing` 浮点参数（范围 0.0–0.3）
- `extra_losses: Optional[List[str]] = None`：额外追加的自定义 loss 名（与注册表取并集去重）
- `include_self_supervised: bool = False`：默认排除自监督损失。自监督 loss（如 `EntLoss`）forward 返回 dict，签名与监督任务 `(logits, y_long)` 不兼容，采样后必然失败

返回的 `SearchSpace` 包含：
- `loss`（categorical）：choices 来自 `list_supervised_losses()`（默认）或 `list_losses()`（`include_self_supervised=True`）
- `label_smoothing`（float，可选）：`low=0.0, high=0.3`

### 搜索流程

`run_loss_search` 通过 SP `ask/tell` 驱动搜索：

```python
from senseframe.automl import run_loss_search

result = run_loss_search(
    config=config,                  # ExperimentConfig 实例（不会被修改）
    n_trials=10,                    # 试验次数
    direction="maximize",           # "maximize" / "minimize"
    metric="val_accuracy",          # 评估指标名
    sampler="random",               # SP Sampler 名（"random" / "grid"）
    include_label_smoothing=True,   # 搜索空间是否含 label_smoothing
    study_manager=None,             # 可选 StudyManager（None 用全局单例）
)
```

执行流程：
1. 调用 `build_loss_search_space(include_label_smoothing=...)` 构造搜索空间
2. `sm.create_study(name="loss_search", direction=..., search_space=..., sampler=...)` 创建 Study
3. 循环 `n_trials` 次：
   - `trial = sm.ask(study_id)` 采样参数
   - `modified_config = _apply_loss_params(config, trial.params)` 应用参数
   - `result = run_pipeline(modified_config)` 执行一次完整训练
   - `sm.tell(trial.trial_id, value, state, feedback=...)` 上报结果
4. 提取 `sm.best_trial(study_id)`，封装为 `LossSearchResult` 返回

`_apply_loss_params` 在 `apply_params` 基础上扩展：
- `loss` → `scene.params["loss"]`
- `label_smoothing` → `scene.params["loss_kwargs"]["label_smoothing"]`（合并已有 `loss_kwargs`，不覆盖其他 key）

### 结果结构

`LossSearchResult` dataclass 字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `study_id` | `str` | — | SP Study ID（供后续 `list_trials` / `best_trial` 查询） |
| `best_params` | `Dict[str, Any]` | `{}` | 最优 trial 的参数 |
| `best_value` | `Optional[float]` | `None` | 最优 trial 的指标值 |
| `n_trials` | `int` | `0` | 总试验数 |
| `n_completed` | `int` | `0` | 成功完成的试验数 |
| `n_failed` | `int` | `0` | 失败试验数 |
| `trials` | `List[TrialResult]` | `[]` | 全部 trial 列表（SP TrialResult） |
| `direction` | `str` | `"maximize"` | 优化方向 |
| `metric` | `str` | `"val_accuracy"` | 评估指标名 |

提供 `to_dict()` 方法序列化为字典。与 HPO `HPOOutput` 的区别：不含 Optuna 特定字段（`trial_number` / `n_pruned`），`trials` 使用 SP 的 `TrialResult`（schema_version + 自省友好）。

## 元学习（meta_learner）

### MetaLearner

`MetaLearner` 实现跨数据集迁移搜索经验（P3.2.3）：

```python
from senseframe.automl import MetaLearner, HistoryStore
from senseframe.search_protocol import StudyManager

sm = StudyManager()
store = HistoryStore(base_dir=Path("/path/to/sf_history"))

learner = MetaLearner(study_manager=sm, history_store=store)
n_injected = learner.warm_start(
    study_id="target_study_id",
    source_dataset="UT_HAR_data",
    success_threshold=0.7,
)
```

`MetaLearner.__init__(study_manager, history_store)`：
- `study_manager: StudyManager`：用于访问 `_trackers` 字典
- `history_store: HistoryStore`：用于加载源数据集历史

`warm_start(study_id, source_dataset, success_threshold=0.7) -> int`：
- 加载源数据集历史 `store.load_history(dataset=source_dataset)`
- 过滤成功策略：优先用 `result.val_accuracy > success_threshold`，fallback 到 `result.value > success_threshold`
- 将成功策略 `tracker.history.extend(successful)` 注入目标 Study 的 tracker
- 可选优化：若 Sampler 类实现 `warm_start` 方法（如 `EvolutionarySampler`），创建临时实例调用一次（用于可能的类级状态副作用）
- 返回注入的历史条目数（0 表示源数据集不存在或无成功策略）；`study_id` 不存在时抛 `KeyError`

**关键设计决策**：因 `StudyManager.ask()` 每次 ask 都创建新的 sampler 实例（`sampler = sampler_cls()`），实例级 `warm_start` 状态无法跨 ask 保留。所以核心机制是将源数据集成功策略**注入 `tracker.history`**，让后续 `sampler.sample(search_space, history)` 自然从扩展后的 history 中读取作为采样偏向。

### 与 Exploration 集成

`MetaLearner` 与 `ExplorationTracker` 的协作链路：

1. **历史持久化**：`HistoryStore.save_history(dataset, tracker)` 将 `ExplorationTracker.history` 序列化为 `{base_dir}/{dataset}/history.json`
2. **历史加载**：`HistoryStore.load_history(dataset)` 读取并返回 `List[Dict]`（不存在时返回空 list）
3. **warm-start 注入**：`MetaLearner.warm_start` 把源数据集成功策略 extend 到目标 Study 的 `tracker.history`
4. **采样偏向**：`StudyManager.ask()` 调用 `sampler.sample(search_space, history)`，sampler 从扩展后的 history 读取成功策略作为采样偏向
5. **推荐查询**：`ExplorationTracker.recommend_next(task_type, top_k=5)` 基于 feedback 感知排序返回推荐策略（不直接被 MetaLearner 调用，但共享同一份 `tracker.history`）

`ExplorationTracker.recommend_next` 的 feedback 感知排序（RFC-002 阶段 R）：若最近试验有 feedback，按其 `status` 生成定向推荐并置于列表前列：
- `numerical_instability` → 推荐稳定 loss + 降低 lr + 梯度裁剪
- `underfitting` → 推荐更强 loss + 提高 lr + 增加训练轮数
- `overfitting` → 推荐数据增强 + 增大 weight_decay + dropout
- `converged` → 推荐未探索的新 pipeline；`success` → 微调 lr（×0.5 / ×2.0）

`ExplorationTracker` 关键 API：`add_trial`（别名 `record_trial`）/ `update_trial` / `best_trial`（仅从 `status="completed"` 选 best）/ `last_feedback` / `log_adoption`（闭合 `feedback → recommended → adopted` 链路）/ `feedback_trace`（返回追溯链路）。

## 参数高效微调（PEFT）

### PEFTConfig

`PEFTConfig` dataclass（`senseframe/core/foundation_model.py`）与 PEFT 搜索空间参数对齐：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `peft_method` | `str` | `"lora"` | PEFT 方法：`lora` / `adapter` / `prefix_tuning` / `prompt_tuning` / `full` |
| `peft_rank` | `int` | `8` | LoRA 秩 |
| `peft_alpha` | `int` | `1` | LoRA 缩放系数 |
| `peft_dropout` | `float` | `0.0` | LoRA / Adapter dropout |
| `peft_target_modules` | `str` | `"query_value"` | 注入目标：`query` / `value` / `query_value` / `all` |
| `adapter_bottleneck` | `int` | `128` | Adapter bottleneck 维度 |
| `prompt_length` | `int` | `10` | Prefix / Prompt 长度 |
| `freeze_backbone` | `bool` | `True` | 是否冻结 backbone 原始参数 |

### PEFT 方法

#### LoRA（`LoRALayer`）

低秩分解旁路：冻结原始 Linear，添加 A/B 低秩旁路。

构造参数：`original: nn.Linear`、`rank: int`（来自 `peft_rank`，默认 8）、`alpha: float`（来自 `peft_alpha`，默认 1）、`dropout: float = 0.0`（来自 `peft_dropout`）。

实现细节：
- `A: (rank, in_features)` — `kaiming_uniform_(a=sqrt(5))` 初始化
- `B: (out_features, rank)` — `zeros` 初始化（保证初始时 LoRA 输出为 0，不破坏预训练权重）
- `scaling = alpha / rank`；forward：`out = original(x) + B(A(dropout(x))) * scaling`
- dropout 放在 A 之前（对 LoRA 输入做 dropout），符合 LoRA 原论文（Hu et al. 2021）

#### Adapter（`AdapterLayer`）

瓶颈残差结构：冻结原始 Linear，添加 bottleneck 残差旁路。

构造参数：`original: nn.Linear`、`bottleneck: int`（来自 `adapter_bottleneck`，默认 128）、`dropout: float = 0.0`。

实现细节：
- `adapter_down: Linear(in_features, bottleneck, bias=False)` / `adapter_up: Linear(bottleneck, out_features, bias=False)`（`zeros` 初始化，初始输出为 0）/ `activation = ReLU()`
- forward：`out = original(x) + dropout(adapter_up(activation(adapter_down(x))))`

#### Prefix Tuning（`PrefixTuningLayer`）

可学习 prefix 张量拼接到输入序列前。

构造参数：`d_model: int`（由 `_infer_d_model` 推断）、`prefix_len: int = 10`（来自 `prompt_length`）。

实现细节：
- `prefix: (prefix_len, d_model)` — `normal_(std=0.02)` 初始化
- forward：输入 `(B, seq, d_model)` → `cat([prefix.expand(B, -1, -1), x], dim=1)`
- 2D 输入时（非序列）发 warning 并跳过 prefix 注入（用 `_warned_2d` 标记避免重复 log）

#### Prompt Tuning（`PromptTuningLayer`）

可学习 prompt 张量拼接到输入 embedding 前。

构造参数：`d_model: int`、`prompt_len: int = 10`（来自 `prompt_length`）。

实现细节：`prompt: (prompt_len, d_model)` — `normal_(std=0.02)` 初始化；forward：输入 `(B, seq, d_model)` → `cat([prompt.expand(B, -1, -1), x], dim=1)`；2D 输入时发 warning 并跳过。

### PEFTBuilder

`PEFTBuilder.build(backbone, peft_params)` 静态方法根据 `peft_method` 分发构建：

```python
from senseframe.automl import PEFTBuilder

peft_model = PEFTBuilder.build(
    backbone=foundation_model,
    peft_params={
        "peft_method": "lora",
        "peft_rank": 8,
        "peft_alpha": 1,
        "peft_dropout": 0.0,
        "peft_target_modules": "query_value",
        "freeze_backbone": True,
    },
)
```

`peft_params` 值可以是 `str`（SP 采样）或原生类型（直接调用），由 `_coerce_int` / `_coerce_float` / `_coerce_bool` 统一转换。

构建流程：
1. **重复 build 检查**：若 `backbone` 已是 `PEFTModel` 或内部已含 `LoRALayer` / `AdapterLayer` / `PrefixTuningLayer` / `PromptTuningLayer`，raise `ValueError`
2. 创建 `PEFTModel(backbone)` 容器，设置 `peft_method`
3. 按 `method` 分发：`_build_lora` / `_build_adapter` / `_build_prefix_tuning` / `_build_prompt_tuning` / `_build_full`
4. 统一调用 `_apply_freeze_backbone(peft_model, method, freeze_backbone)` 处理冻结语义

`_should_inject_lora(name, module, target)` 判断是否对某 Linear 注入 LoRA/Adapter：
- `target="all"`：所有 Linear
- `target="query"`：name 含 "query"
- `target="value"`：name 含 "value"
- `target="query_value"`：name 含 query 或 value

`_infer_d_model(backbone)` 推断 d_model：优先用 `backbone.d_model` 属性（如 `CSIFoundationModel.d_model=128`），缺失时兜底取第一个 Linear 的 `in_features`，再兜底 128。

### PEFTModel

`PEFTModel` 包装基础模型，注入 PEFT 模块后统一管理：

```python
class PEFTModel(nn.Module):
    backbone: nn.Module
    lora_modules: nn.ModuleList       # 所有 LoRALayer
    adapter_modules: nn.ModuleList    # 所有 AdapterLayer
    prefix_layer: Optional[PrefixTuningLayer]
    prompt_layer: Optional[PromptTuningLayer]
    peft_method: str
```

forward 行为：
- LoRA/Adapter 模块在构建时已通过 `_set_submodule` 替换 backbone 中的 Linear（in-place），forward 时自动生效
- Prefix/Prompt 层作为独立模块，在 forward 时拼接到输入序列前
- **CSI 模型穿透注入**（P3-P2-11 修复）：当 backbone 暴露 `patch_embedder` + `encoder` + `pos_embed` 标准接口（如 `CSIFoundationModel`）时，输入是 `(B, C, L)` 原始信号，prompt/prefix 应在 patch embedding 之后注入。复刻 `encode(x)` 流程：`patches = patch_embedder(x) + pos_embed` → 注入 prompt/prefix → `encoder(patches)` → `encoder_norm`
- **简单 backbone**：直接对输入注入（兼容直接接受 `(B, seq, d_model)` 的简单 backbone）

`encode_features(x)` 方法：`forward` 的语义别名，供 DANN 等下游模块调用以表达"供下游用"的意图。`DANNCrossModalModel.forward` 调用 `backbone.encode_features(x)` 提取特征，`PEFTModel` 必须暴露此方法否则 `AttributeError`。

`freeze_backbone` 语义（P3-2 修复后统一）：
- `freeze_backbone=True`（默认）：冻结所有非 PEFT 参数（backbone 原始权重 `requires_grad=False`），PEFT 模块参数（LoRA A/B / Adapter bottleneck / Prefix / Prompt）保持 `requires_grad=True`
- `freeze_backbone=False`：解冻 backbone 所有参数（含 PEFT 模块），适用于"PEFT + 全量微调混合"场景
- `method="full" + freeze_backbone=True`：语义矛盾（Full 无 PEFT 模块可冻结，模型完全冻结无意义），自动转 `False` 并发 warning

## PEFT 搜索（peft_search）

`peft_search` 通过 SP `ask/tell` 驱动 PEFT 微调策略搜索，验证"搜索微调策略而非架构"的新 AutoML 范式。

### 搜索空间

`build_peft_search_space(include_methods=None)` 构造含 9 个参数的 `SearchSpace`：

| 参数名 | 类型 | 取值 |
|--------|------|------|
| `peft_method` | categorical | `lora` / `adapter` / `prefix_tuning` / `prompt_tuning` / `full`（默认全部 5 种） |
| `peft_rank` | categorical | `4` / `8` / `16` / `32` / `64` |
| `peft_alpha` | categorical | `1` / `2` / `4` |
| `peft_dropout` | float | `0.0` – `0.3`，step `0.05` |
| `peft_target_modules` | categorical | `query` / `value` / `query_value` / `all` |
| `learning_rate` | float | `1e-5` – `1e-3`，`log=True` |
| `adapter_bottleneck` | categorical | `64` / `128` / `256` |
| `prompt_length` | categorical | `5` / `10` / `20` / `50` |
| `freeze_backbone` | categorical | `True` / `False` |

`include_methods` 可限定 PEFT 方法子集（如 `["lora", "adapter"]`）。

### ask/tell 循环

`run_peft_search` 主入口：

```python
from senseframe.automl import run_peft_search

result = run_peft_search(
    config=config,                # ExperimentConfig 实例
    foundation_model=fm,          # 基础模型（实现 SensingFoundationModel Protocol 时取 model_id 用作 study name）
    n_trials=10,
    direction="maximize",
    metric="val_accuracy",
    sampler="random",
    study_manager=None,
)
```

执行流程：
1. `sm.create_study(name=f"peft_search_{model_id}", direction=..., search_space=build_peft_search_space(), sampler=...)`
2. `try/finally` 包裹 ask/tell 循环（P3-1 修复：确保异常时仍 `stop_study` 释放 SP Study 资源）
3. 每次 trial：
   - `trial = sm.ask(study_id)`
   - `modified_config = _apply_peft_params(config, foundation_model, trial.params)`
   - `result = run_pipeline(modified_config)`
   - `extract_metric(result, metric)` 返回 `None` 时用 0.0 兜底（P3-P2-6 修复）
   - `sm.tell(trial.trial_id, value, state, feedback=...)`
4. `finally` 块调用 `sm.stop_study(study_id)`（防御性 `hasattr` 检查，兼容旧版 SP）
5. 提取 `sm.best_trial(study_id)`，封装为 `PEFTSearchResult` 返回

### 评估流程

`_apply_peft_params(config, foundation_model, params)`：
1. `copy.deepcopy(config)` 生成新 config
2. `copy.deepcopy(foundation_model)` 在深拷贝上构建 PEFT（避免污染原模型）— 大模型 deepcopy 会 log 耗时（P3-P2-5 修复）
3. `PEFTBuilder.build(foundation_copy, params)` 构建 PEFT 模型
4. PEFT 参数写入 `scene.params`（场景容器可见 + 可追溯）
5. `learning_rate` 写入 `trainer.learning_rate`
6. 通过 `module_factory` 注入 PEFT 模型（覆盖场景默认模型）：包装原 `module_factory`，忽略传入的 `model` 参数，使用 `peft_model`

### 结果结构

`PEFTSearchResult` dataclass 与 `LossSearchResult` 对称：`study_id` / `best_params` / `best_value` / `n_trials` / `n_completed` / `n_failed` / `trials` / `direction` / `metric`，提供 `to_dict()` 方法。

## 基础模型协议

### SensingFoundationModel Protocol

`SensingFoundationModel`（`senseframe/core/foundation_model.py`）是 `@runtime_checkable` Protocol，定义感知基础模型抽象：

```python
@runtime_checkable
class SensingFoundationModel(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def modality(self) -> str: ...
    def pretrain(self, unlabeled_data: Any, config: PretrainConfig) -> None: ...
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def get_peft_module(self, peft_config: PEFTConfig) -> nn.Module: ...
```

实现者需提供：
- `model_id: str`：基础模型 ID（如 `'csi-mae-base'`）
- `modality: str`：模态（`csi` / `radio` / `eeg` / `acoustic`）
- `pretrain(unlabeled_data, config)`：自监督预训练
- `encode(x)`：特征提取
- `get_peft_module(peft_config)`：基于 PEFT 配置构建微调模块

`PretrainConfig` dataclass 字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | `str` | `"mae"` | 预训练方法：`mae` / `simclr` / `byol` |
| `epochs` | `int` | `100` | 训练轮数 |
| `batch_size` | `int` | `32` | 批大小 |
| `learning_rate` | `float` | `1e-3` | 学习率 |
| `mask_ratio` | `float` | `0.75` | MAE mask 比例（论文推荐 0.75，P3-P2-9 修复） |
| `augmentations` | `List[str]` | `[]` | 增强方法列表 |

### CSIFoundationModel

`CSIFoundationModel`（`senseframe/scenes/wifi_csi/foundation_model.py`）实现 `SensingFoundationModel` Protocol，采用 MAE 自监督预训练。

构造函数：

```python
CSIFoundationModel(
    input_shape=(C, L),       # 必填，CSI 输入形状
    d_model=128,              # encoder 隐藏维度
    n_heads=4,                # 注意力头数
    n_encoder_layers=4,       # encoder 层数
    n_decoder_layers=2,       # decoder 层数
    patch_len=16,             # patch 长度（L 必须能被 patch_len 整除）
    decoder_dim=64,           # decoder 隐藏维度
)
```

校验：`n_heads >= 1`，`d_model % n_heads == 0`，`L % patch_len == 0`。

模块结构：
- `CSIPatchEmbedder`：`(B, C, L)` → `(B, n_patches, d_model)`，沿 L 切 patch，`proj = Linear(patch_len * C, d_model)`
- `pos_embed: (1, n_patches, d_model)` — `trunc_normal_(std=0.02)`
- `encoder: Sequential[CSITransformerEncoderLayer]` — Pre-LN transformer，`CSIAttention` 含显式 `query` / `key` / `value` / `out` Linear 命名（便于 `peft_target_modules='query_value'` 匹配注入 LoRA/Adapter）
- `encoder_norm` / `decoder_embed: Linear(d_model, decoder_dim)` / `mask_token: (1, 1, decoder_dim)`（`trunc_normal_`）/ `decoder_pos_embed` / `decoder`（仅 self-attention）/ `decoder_proj: Linear(decoder_dim, patch_len * C)`

#### MAE 预训练

`pretrain(unlabeled_data, config)` 流程：
1. 写入 `self._mask_ratio = config.mask_ratio`（供 `SelfSupervisedModule.validation_step` 读取对齐训练口径）
2. `optimizer = AdamW(self.parameters(), lr=config.learning_rate)`
3. 每 epoch：取 `batch[0]`（若 batch 是 list/tuple）→ `_mae_forward_loss(x, mask_ratio)` → `zero_grad` / `backward` / `step`
4. 累积 epoch loss，按 epoch 打印 mean loss（P3-P2-8 修复：原实现无任何日志，训练过程不可观测）

`_mae_forward_loss(x, mask_ratio)`：
- `target = patch_embedder.to_patches(x)` — 重建目标（不含投影 + pos_embed）
- `patches = patch_embedder.proj(target) + pos_embed`
- `random_masking(patches, mask_ratio)` → `(x_visible, mask, ids_restore)`
- `enc_out = encoder(x_visible)` + `encoder_norm`
- `recon = decoder(...)` 还原全部 patches 顺序
- loss = `((recon - target) ** 2).mean(dim=-1)`，仅在 masked patches 上计算：`(loss * mask).sum() / mask.sum()`

#### mae_reconstruct

`mae_reconstruct(x, mask_ratio) -> (recon, target, mask)` 是 MAE 重建的 public API（消除 `self_supervised.py` / `_mae_forward_loss` / `p0_pretrain_with_psnr.py` 三处重复）。返回：
- `recon: (B, n_patches, patch_len * C)` 重建张量
- `target: (B, n_patches, patch_len * C)` 原始 patches
- `mask: (B, n_patches)` float，`1 = masked` / `0 = visible`

`random_masking(x, mask_ratio)` 内部：`len_keep = int(N * (1 - mask_ratio))`，`argsort(rand)` 打乱，保留前 `len_keep` 个 patch，`ids_restore` 用于 decoder 还原顺序。

#### 特征提取与 PEFT

- `encode(x)`：`(B, C, L)` → `patch_embedder(x) + pos_embed` → `encoder` → `encoder_norm` → `(B, n_patches, d_model)`
- `encode_features(x)`：`encode` 的语义别名，供 DANN 等下游模块调用以表达"供下游用"意图
- `get_peft_module(peft_config)`：`copy.deepcopy(self)` 后调用 `PEFTBuilder.build(foundation_copy, peft_config.__dict__)`，深拷贝避免污染预训练权重
- `forward(x)`：调用 `encode(x)`，供 `PEFTBuilder` 注入 LoRA 等模块时使用

#### 跨模态迁移

`replace_patch_embedder(new_input_shape, new_patch_len)`：替换 modality-specific 模块（`patch_embedder` / `pos_embed` / `decoder_pos_embed` / `decoder_proj`），保留 modality-agnostic 模块（`encoder` / `encoder_norm` / `decoder_embed` / `mask_token` / `decoder` / `decoder_norm`）。用于 B5/B6 跨场景迁移：CSI 上预训练后替换 patch_embedder 为目标模态（如 EEG）的维度。

## 使用示例

### 示例 1：PEFT 微调流程

```python
import torch
from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
from senseframe.core.foundation_model import PEFTConfig

# 1. 加载预训练的 CSI 基础模型
fm = CSIFoundationModel(input_shape=(342, 2000), d_model=128, patch_len=16)
# fm.pretrain(unlabeled_data, PretrainConfig(method="mae", mask_ratio=0.75, epochs=100))

# 2. 配置 PEFT 并构建（深拷贝避免污染预训练权重）
peft_config = PEFTConfig(
    peft_method="lora", peft_rank=8, peft_alpha=1,
    peft_target_modules="query_value", freeze_backbone=True,
)
peft_model = fm.get_peft_module(peft_config)  # 等价于 PEFTBuilder.build(copy.deepcopy(fm), peft_config.__dict__)

# 3. 验证可训练参数
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"trainable: {trainable}/{total} ({100*trainable/total:.2f}%)")

# 4. 前向（PEFTModel 自动应用 LoRA + 穿透 patch_embedder 注入）
x = torch.randn(4, 342, 2000)
features = peft_model(x)                  # (B, n_patches, d_model)
features = peft_model.encode_features(x)  # 语义别名，供 DANN 下游用
```

### 示例 2：PEFT 搜索流程

```python
from pathlib import Path
from senseframe.automl import run_peft_search, HistoryStore
from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel

fm = CSIFoundationModel(input_shape=(342, 2000), d_model=128, patch_len=16)
config = ...  # ExperimentConfig，需配置 module_factory

# （可选）元学习 warm-start：先保存源数据集历史到 HistoryStore
# store = HistoryStore(base_dir=Path("/path/to/sf_history"))
# store.save_history("UT_HAR_data", source_tracker)

result = run_peft_search(
    config=config, foundation_model=fm,
    n_trials=20, direction="maximize", metric="val_accuracy", sampler="random",
)
print(f"best_params: {result.best_params}")
print(f"best_value: {result.best_value}, completed: {result.n_completed}, failed: {result.n_failed}")
# 持久化目标 study 历史供后续 warm-start：
#   sm._trackers[result.study_id] → store.save_history("target_dataset", tracker)
```

`result.best_params` 形如 `{"peft_method": "lora", "peft_rank": 16, "peft_alpha": 2, "peft_dropout": 0.05, "peft_target_modules": "query_value", "learning_rate": 5e-4, "adapter_bottleneck": 128, "prompt_length": 10, "freeze_backbone": True}`，可直接传给 `PEFTBuilder.build` 或构造 `PEFTConfig` 复现最优配置。
