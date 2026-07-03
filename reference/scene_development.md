# 场景开发指南

本指南介绍如何为 SenseFrame 开发新场景容器（SceneContainer），使框架支持新的领域（如图像、时序、NLP、检测、分割）。

## 一、场景容器职责

场景容器（`SceneContainer`）是领域逻辑与通用训练流程的解耦层。引擎通过此接口与场景交互，新增场景只需实现抽象方法 + 可选覆写新任务相关方法，无需修改引擎代码。

### 必须实现的抽象方法（4 个）

| 方法 | 职责 | 返回值 |
|------|------|--------|
| `meta()` | 声明场景能力（支持的任务、模型、数据集、学习模式） | `SceneMeta` |
| `load_dataset()` | 加载训练/测试集 | `DatasetBundle`（监督填 `train/test`，自监督填 `unsupervised/supervised_finetune/test`） |
| `build_model_for_dataset()` | 按 (model, dataset) 构建模型实例 | `nn.Module` |
| `get_dataset_info()` | 返回数据集元信息 | `dict`（含 `num_classes`, `input_shape`, `modality` 等） |

### 可选覆写的方法

| 方法 | 默认行为 | 何时覆写 |
|------|---------|---------|
| `get_task_spec()` | 从 `get_dataset_info` 派生分类 TaskSpec | 需支持回归/检测/分割等非分类任务时 |
| `get_feature_spec()` | 从 `get_dataset_info` 派生 FeatureSpec | 需提供场景特定的特征元信息时 |
| `get_scene_params()` | 返回空 SceneParams | 需提供场景特定参数时 |
| `postprocess()` | 直接返回输入（无后处理） | 需 NMS / CRF / 阈值化等后处理时 |
| `get_default_config()` | 返回训练默认配置 | 需调优超参默认值时 |
| `get_search_space()` | 返回空 HPO 空间 | 需支持 HPO 搜索时 |
| `get_transforms()` | 返回空配置 | 需逐样本数据变换/增强时 |
| `get_model_info()` | 返回空字典 | 需提供模型属性（VRAM 估算等）时 |
| `normalize()` | 不做归一化 | 需场景特定归一化策略时 |

> **设计原则**（LSP/ISP/OCP）：子类实现 4 抽象方法即可工作；其余方法按需覆写。
> 不要偷偷给 `get_dataset_info(root=None)` 加额外位置参数，这会破坏 LSP。

## 二、四类核心抽象

新增场景时，建议按以下顺序考虑四个抽象：

### 2.1 `TaskType` 与 `TaskSpec` — 决定 loss / metrics / 输出激活

```python
from senseframe.core.task import TaskType, TaskSpec

# 分类任务（最常见）
spec = TaskSpec.classification(num_classes=10)
# → task_type=CLASSIFICATION, loss="cross_entropy", metrics=["accuracy", "macro_f1"]

# 回归任务
spec = TaskSpec.regression(loss="mse", metrics=["mse", "mae"])
# → task_type=REGRESSION, loss="mse", num_classes=None

# 检测任务
spec = TaskSpec(
    task_type=TaskType.DETECTION,
    num_classes=80,
    loss="bce_with_logits",
    output_activation="sigmoid",
    metrics=["map"],
)
```

**子类覆写 `get_task_spec()`** 来自定义默认任务规格：

```python
def get_task_spec(self, dataset_name: str, model_id: str = "", **kwargs) -> TaskSpec:
    info = self.get_dataset_info(dataset_name)
    if info["task_type"] == "regression":
        return TaskSpec.regression(loss="mse")
    if info["task_type"] == "detection":
        return TaskSpec(
            task_type=TaskType.DETECTION,
            num_classes=info["num_classes"],
            loss="bce_with_logits",
            output_activation="sigmoid",
        )
    return TaskSpec.classification(num_classes=info["num_classes"])
```

### 2.2 `FeatureSpec` — 描述输入特征规格

```python
from senseframe.core.features import FeatureSpec

# 从 dataset_info 自动派生（默认实现已覆盖）
spec = FeatureSpec.from_dataset_info(
    {"input_shape": [3, 224, 224], "n_features": 150528},
    modality="image",
)
# → input_shape=(3,224,224), num_channels=3, sequence_length=50176, feature_dim=150528

# 子类可覆写 get_feature_spec() 补充场景特定元信息
def get_feature_spec(self, dataset_name: str, **kwargs) -> FeatureSpec:
    spec = super().get_feature_spec(dataset_name, **kwargs)
    return spec.with_overrides(
        extra={"antenna_pairs": 3, "subcarrier_count": 30},
    )
```

### 2.3 `SceneParams` — 场景参数正交化

`SceneParams` 替代了原 `params: dict`，提供标准字段 + 自定义扩展：

```python
from senseframe.core.params import SceneParams

params = SceneParams(
    target_frames=128,
    window_size=64,
    overlap_ratio=0.5,
    sampling_rate=1000,
    task_type="classification",
    loss="focal",
    metrics=["accuracy", "macro_f1"],
    extra={"custom_field": 42},   # 场景特定扩展
)

# 标准字段直接读
print(params.target_frames)   # 128
print(params.get("n_antennas", 3))  # 从 extra 读，缺省返回 3

# 通过 SceneContext 自动升级（dict → SceneParams）
ctx = SceneContext(params={"target_frames": 200})  # 自动转为 SceneParams
```

**子类覆写 `get_scene_params()`** 标准化场景参数：

```python
def get_scene_params(self, dataset_name: str, **kwargs) -> SceneParams:
    return SceneParams(
        target_frames=200,
        window_size=128,
        overlap_ratio=0.5,
        task_type="classification",
        loss="cross_entropy",
        extra={"my_scene_specific_param": 1.0},
    )
```

### 2.4 Loss 工厂 — `@register_loss`

Loss 不再硬编码，可通过注册表替换：

```python
from senseframe.core.losses import register_loss, get_loss, list_losses

# 内置 losses
print(list_losses())
# ['cross_entropy', 'cross_entropy_weighted', 'mse', 'mae',
#  'smooth_l1', 'bce_with_logits', 'focal', 'ent_loss']

# 用户自定义 loss
@register_loss("my_focal_v2")
def _my_focal_v2(alpha=0.5, gamma=2.0, eps=1e-6):
    class FocalLossV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.eps = eps
        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce)
            return (self.alpha * (1 - pt + self.eps) ** self.gamma * ce).mean()
    return FocalLossV2()

# 查询
loss = get_loss("my_focal_v2", alpha=0.3)
```

**HPO 集成**：将 `loss` 加入 `get_search_space()` 后，HPO 试验自动按 `loss` 名查询工厂：

```python
def get_search_space(self, model_id: str, dataset_name: str, **kwargs) -> SearchSpace:
    space = SearchSpace()
    space.params["loss"] = {
        "type": "categorical",
        "values": ["cross_entropy", "focal", "my_focal_v2"],
    }
    return space
```

## 三、最小场景示例：时序分类

以下从零实现一个 `TimeSeriesScene`，展示四类抽象如何配合。

### 3.1 目录结构

```
senseframe/scenes/time_series/
├── __init__.py          # 导出 TimeSeriesContainer
├── container.py         # 场景容器实现（含 4 抽象 + 4 可选覆写）
└── losses.py            # 场景特定 loss（可选）
```

### 3.2 实现容器

```python
# senseframe/scenes/time_series/container.py
"""时序分类场景容器。"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from ..base import (
    DefaultConfig, SceneContainer, SceneMeta, SearchSpace, TransformConfig,
    DatasetBundle, SceneContext,
)
from ...core.task import TaskType, TaskSpec
from ...core.features import FeatureSpec
from ...core.params import SceneParams


class TimeSeriesContainer(SceneContainer):
    """时序分类场景：支持自定义时序数据集。"""

    _DATASETS = {
        "har_activity": {
            "num_classes": 6,
            "input_shape": [1, 128],  # (channel, seq_len)
            "modality": "timeseries",
        },
    }

    # --------------------------------------------------------
    # 必须实现的 4 抽象方法
    # --------------------------------------------------------
    def meta(self) -> SceneMeta:
        return SceneMeta(
            name="time_series",
            supported_tasks=["classification"],
            supported_models=["TCN", "LSTM"],
            supported_datasets=list(self._DATASETS.keys()),
            input_shape_hint=[1, 128],
            supported_learning_modes=["supervised"],
        )

    def load_dataset(
        self, dataset_name: str, root: Optional[str] = None,
        learning_mode: str = "supervised", **kwargs,
    ) -> DatasetBundle:
        info = self._DATASETS[dataset_name]
        x_train = torch.randn(64, *info["input_shape"])
        y_train = torch.randint(0, info["num_classes"], (64,))
        x_test = torch.randn(16, *info["input_shape"])
        y_test = torch.randint(0, info["num_classes"], (16,))
        return DatasetBundle(
            train=TensorDataset(x_train, y_train),
            test=TensorDataset(x_test, y_test),
        )

    def build_model_for_dataset(
        self, model_id: str, dataset_name: str,
        num_classes: Optional[int] = None,
        learning_mode: str = "supervised", **kwargs,
    ) -> nn.Module:
        info = self._DATASETS[dataset_name]
        nc = num_classes or info["num_classes"]
        if model_id == "TCN":
            return self._build_tcn(nc)
        if model_id == "LSTM":
            return self._build_lstm(nc)
        raise ValueError(f"Unknown model: {model_id}")

    def get_dataset_info(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        return dict(self._DATASETS[dataset_name])

    # --------------------------------------------------------
    # 可选覆写 get_task_spec
    # 默认实现已自动从 dataset_info 派生分类 TaskSpec
    # 这里演示如何覆写为多任务类型
    # --------------------------------------------------------
    def get_task_spec(
        self, dataset_name: str, model_id: str = "", **kwargs,
    ) -> TaskSpec:
        info = self.get_dataset_info(dataset_name)
        # 自定义默认 loss 为 focal 而非 cross_entropy
        return TaskSpec.classification(
            num_classes=info["num_classes"],
            loss="focal",
        )

    # --------------------------------------------------------
    # 可选覆写 get_feature_spec
    # --------------------------------------------------------
    def get_feature_spec(
        self, dataset_name: str, **kwargs,
    ) -> FeatureSpec:
        info = self.get_dataset_info(dataset_name)
        return FeatureSpec(
            input_shape=tuple(info["input_shape"]),
            num_channels=info["input_shape"][0],
            sequence_length=info["input_shape"][1],
            modality=info.get("modality", "timeseries"),
        )

    # --------------------------------------------------------
    # 可选覆写 get_scene_params
    # --------------------------------------------------------
    def get_scene_params(self, dataset_name: str, **kwargs) -> SceneParams:
        return SceneParams(
            target_frames=128,
            window_size=64,
            overlap_ratio=0.5,
            sampling_rate=50,
            task_type="classification",
            loss="focal",
            metrics=["accuracy", "macro_f1"],
        )

    # --------------------------------------------------------
    # 其他可选覆写
    # --------------------------------------------------------
    def get_default_config(
        self, model_id: str, dataset_name: str, **kwargs,
    ) -> DefaultConfig:
        return DefaultConfig(epochs=50, learning_rate=1e-3, batch_size=64)

    def get_search_space(
        self, model_id: str, dataset_name: str, **kwargs,
    ) -> SearchSpace:
        space = SearchSpace()
        space.params["learning_rate"] = {
            "type": "float", "low": 1e-5, "high": 1e-2, "log": True,
        }
        # 注入 task / loss 搜索空间
        from ...engine.hpo import get_task_search_space_extension
        space.params.update(get_task_search_space_extension())
        return space

    def get_transforms(self, dataset_name: str) -> TransformConfig:
        def train_transform(x, y):
            x = x + torch.randn_like(x) * 0.01  # 训练时加噪声
            return x, y
        return TransformConfig(train_transform=train_transform)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        return {"estimated_vram_mb": 512, "paradigm": "cnn"}

    # ----- 内部模型构建 -----
    def _build_tcn(self, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def _build_lstm(self, num_classes: int) -> nn.Module:
        class LstmWrap(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc = nn.Linear(32, nc)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        return LstmWrap(num_classes)
```

### 3.3 注册场景

```python
# senseframe/scenes/time_series/__init__.py
from .container import TimeSeriesContainer

__all__ = ["TimeSeriesContainer"]
```

```python
# senseframe/scenes/__init__.py 中追加
from .time_series import TimeSeriesContainer
register_scene("time_series", TimeSeriesContainer())
```

### 3.4 YAML 配置（含 task_spec 字段）

```yaml
# time_series.yaml
scene:
  name: time_series
  dataset: har_activity
  model_id: TCN
  task_spec:                          # 显式声明任务规格
    task_type: classification
    num_classes: 6
    loss: focal                       # 显式指定 loss
    metrics: [accuracy, macro_f1]
    output_activation: softmax        # ONNX 导出时附加
input_features:
  - name: x
    type: timeseries
    shape: [1, 128]
output_features:
  - name: y
    type: category
    num_classes: 6
trainer:
  epochs: 50
  batch_size: 64
```

`validate_config.py` 会自动校验：
- `task_type` 必须在 `TaskType` 枚举中
- `loss` 必须在 `@register_loss` 注册表中
- `output_activation` 必须在 `none/softmax/sigmoid/tanh/relu` 中

### 3.5 使用场景

```bash
# 列出场景（自动包含新场景）
python -m senseframe.cli list-scenes

# 校验配置（含 task_spec 校验）
python scripts/validate_config.py --config time_series.yaml

# 训练
python -m senseframe.cli experiment --config time_series.yaml

# 导出 ONNX 时附加 softmax
python -m senseframe.cli export \
    --metadata runs/<exp>/metadata.json \
    --checkpoint runs/<exp>/model.pth \
    --formats onnx \
    --output-activation softmax
```

## 四、非分类任务接入示例：检测

参考 `senseframe/scenes/detection/container.py`：

```python
class DetectionContainer(SceneContainer):
    def meta(self) -> SceneMeta:
        return SceneMeta(
            name="detection",
            supported_tasks=["detection"],
            supported_models=["SimpleDetector"],
            supported_datasets=["dummy_box", "tiny_coco"],
        )

    def get_task_spec(self, dataset_name, model_id="", **kwargs) -> TaskSpec:
        info = self.get_dataset_info(dataset_name)
        return TaskSpec(
            task_type=TaskType.DETECTION,
            num_classes=info["num_classes"],
            loss="bce_with_logits",
            metrics=["map"],
            output_activation="sigmoid",
        )

    def get_feature_spec(self, dataset_name, **kwargs) -> FeatureSpec:
        info = self.get_dataset_info(dataset_name)
        return FeatureSpec(
            input_shape=tuple(info["input_shape"]),
            modality="image",
        )
```

最小可用端到端：

```python
from senseframe.scenes import get_scene
scene = get_scene("detection")
print(scene.get_task_spec("dummy_box"))        # TaskType.DETECTION
print(scene.get_feature_spec("dummy_box"))     # FeatureSpec(input_shape=(3,64,64), ...)
model = scene.build_model_for_dataset("SimpleDetector", "dummy_box")
```

## 五、关键设计点

### 5.1 DatasetBundle

`load_dataset()` 不再返回元组，统一返回 `DatasetBundle`：

```python
# 监督模式
return DatasetBundle(train=train_ds, test=test_ds)

# 自监督模式
return DatasetBundle(
    unsupervised=unsup_ds,    # 预训练用
    supervised_finetune=sup_ds,  # 微调用
    test=test_ds,
)
```

**填充契约校验**（DSP-2）：`DatasetBundle` 提供按 `learning_mode` 校验填充合规性的方法：

```python
# 查询填充规则
DatasetBundle.filling_rule("self_supervised")
# {"train": "forbidden", "test": "required", "val": "optional",
#  "unsupervised": "required", "supervised_finetune": "required"}

# 校验当前 bundle 填充是否合规（返回错误列表，空列表表示通过）
errors = bundle.validate_filling("self_supervised")

# 自省：运行时状态
bundle.describe("self_supervised")
# {"filled_fields": [...], "learning_mode": "self_supervised", "validation_errors": []}
```

场景开发者应在 `load_dataset` 实现后用 `validate_filling` 自测填充合规性。

### 5.2 归一化策略

归一化不应在 `Dataset.__getitem__` 中实现。通过 `register_normalization()` 注入策略：

```python
from senseframe.registry import register_normalization, ZScoreStrategy

register_normalization("har_activity", ZScoreStrategy(mean=0.0, std=1.0))
```

### 5.3 归一化与变换分离

- **归一化**：数据驱动的统计量修正（z-score / min-max），通过 `register_normalization` 注入
- **变换**：训练时的样本级增强（噪声 / 翻转 / 裁剪），通过 `get_transforms()` 返回

```python
# 归一化（数据统计）
register_normalization("har_activity", ZScoreStrategy(mean=-0.5, std=2.1))

# 变换（增强）
def get_transforms(self, dataset_name):
    return TransformConfig(train_transform=lambda x, y: (x + torch.randn_like(x) * 0.01, y))
```

### 5.4 HPO 搜索空间

`get_search_space()` 返回的 spec 会被 `engine.hpo.apply_params()` 解析：

- **标准字段**（learning_rate / batch_size 等）→ 直接更新 `trainer.*`
- **task 字段**（loss / task_type / output_activation）→ 写入 `scene.params` 供场景容器解析
- **场景特定字段** → 写入 `scene.params` 透传

### 5.5 ONNX 输出激活

`task_spec.output_activation` 决定 ONNX 导出时是否在模型末端串接激活：

| output_activation | 串接模块 | 适用 |
|---|---|---|
| `none` / `None` | 不串接 | 默认；模型输出原始 logits |
| `softmax` | `nn.Softmax(dim=-1)` | 分类概率 |
| `sigmoid` | `nn.Sigmoid()` | 多标签 / 目标检测置信度 |
| `tanh` | `nn.Tanh()` | 回归到 [-1, 1] |
| `relu` | `nn.ReLU()` | 回归到 [0, ∞) |

**注意**：`state_dict` 导出不受 `output_activation` 影响（保持原模型权重）。

### 5.6 自监督模式支持

```python
def load_dataset(self, dataset_name, root=None, learning_mode="supervised", **kwargs):
    if learning_mode == "self_supervised":
        return DatasetBundle(
            unsupervised=unsup_ds,
            supervised_finetune=sup_ds,
            test=test_ds,
        )
    return DatasetBundle(train=train_ds, test=test_ds)
```

同时 `meta().supported_learning_modes` 需包含 `"self_supervised"`。

## 六、测试场景

### 6.1 单元测试模板

```python
# tests/test_time_series_scene.py
import pytest
from senseframe.scenes import get_scene
from senseframe.core.task import TaskType


class TestTimeSeriesScene:
    def setup_method(self):
        self.scene = get_scene("time_series")

    def test_meta(self):
        meta = self.scene.meta()
        assert meta.name == "time_series"
        assert "har_activity" in meta.supported_datasets

    def test_load_dataset_returns_bundle(self):
        bundle = self.scene.load_dataset("har_activity")
        assert bundle.train is not None
        assert bundle.test is not None

    def test_get_task_spec(self):
        spec = self.scene.get_task_spec("har_activity")
        assert spec.task_type == TaskType.CLASSIFICATION
        assert spec.loss == "focal"  # 场景特定默认

    def test_get_feature_spec(self):
        spec = self.scene.get_feature_spec("har_activity")
        assert spec.input_shape == (1, 128)
        assert spec.modality == "timeseries"

    def test_get_scene_params(self):
        params = self.scene.get_scene_params("har_activity")
        assert params.target_frames == 128
        assert params.loss == "focal"
```

### 6.2 端到端测试（不跑真实训练）

用 `get_scene` + `build_model_for_dataset` + 假数据 + 1 step 验证：

```python
def test_end_to_end_model_forward(self):
    model = self.scene.build_model_for_dataset("TCN", "har_activity")
    x = torch.randn(2, 1, 128)
    out = model(x)
    assert out.shape == (2, 6)
```

## 七、常见陷阱

1. **变换双重应用**：`get_transforms()` 返回非空变换后，`Dataset.__getitem__` 中不应再做相同变换，否则会双重处理。
2. **归一化循环导入**：归一化通过 `register_normalization` 注入，不要从 `data/legacy.py` 反向导入。
3. **input_shape 不含 batch 维**：`input_shape=[1, 128]` 表示 `(channel, seq_len)`，runner 会自动补 batch 维。
4. **learning_mode 未声明**：若支持自监督但未在 `meta().supported_learning_modes` 中声明，`--dry-run` 预检会报 `learning_mode_supported` 失败。
5. **build_model_for_dataset 签名**：必须含 `**kwargs`，以兼容 runner 透传的额外参数。
6. **TaskType.REGRESSION 时 num_classes 必须为 None**：否则会与 `TaskSpec.regression()` 工厂冲突。
7. **loss 名唯一性**：`@register_loss` 不允许重复注册同名 loss。
8. **SceneParams 字段命名**：`target_frames` / `window_size` 等是标准字段，自定义参数请放 `extra`。
9. **DatasetBundle 不含 extra**：场景特定元信息请挂载到 `bundle.属性名`，不要传 `extra` 字段。

## 八、TaskSpec 端到端接入

TaskSpec / TaskType / Loss 工厂抽象**接入训练流**，让非分类任务（检测/回归/分割）能端到端运行。

### 8.1 TaskSpec 解析优先级

`engine/runner/orchestrator.py` 中 `_resolve_task_spec()` 按以下优先级解析最终 TaskSpec：

1. **YAML 显式声明**（`scene.task_spec`）→ 转换为 `TaskSpec`
2. **场景容器默认**（`scene.get_task_spec()`）→ 由场景决定

```yaml
# YAML 中显式声明 task_spec（优先级最高）
scene:
  name: wifi_csi
  dataset: UT_HAR_data
  model_id: MLP
  task_spec:
    task_type: classification
    num_classes: 7
    loss: focal
    output_activation: softmax
```

若 YAML 未声明 `task_spec`，runner 调用 `scene.get_task_spec(dataset, model_id)` 获取场景默认值。

### 8.2 场景能力校验

`_validate_scene_capabilities()` 在训练编排前快速失败：

- `task_spec.task_type.value` 必须在 `scene.meta().supported_tasks` 中
- `learning_mode` 必须在 `scene.meta().supported_learning_modes` 中

```python
# detection 场景声明 supported_tasks=["detection"]
# 若用户配置 task_type="classification"，会立即报错：
# ValueError: Scene 'detection' does not support task_type 'classification'.
```

### 8.3 metadata.json 记录 task_spec

训练完成后，`metadata.json` 新增 `task_spec` 字段，供推理/导出复用：

```json
{
  "task_spec": {
    "task_type": "detection",
    "num_classes": 3,
    "loss": "bce_with_logits",
    "metrics": ["map"],
    "output_activation": "sigmoid",
    "extra": {}
  }
}
```

### 8.4 Detection NMS 后处理

`DetectionContainer.postprocess()` 对模型输出应用 NMS：

```python
from senseframe.scenes import get_scene

scene = get_scene("detection")
model = scene.build_model_for_dataset("SimpleDetector", "dummy_box")
x = torch.randn(1, 3, 64, 64)
raw_output = model(x)  # {"bboxes": (1,4), "logits": (1,3)}

# NMS 后处理：score 过滤 + IoU 去重
result = scene.postprocess(raw_output)
# {"bboxes": (K,4), "scores": (K,), "labels": (K,)}
```

后处理参数从 `SceneParams.extra` 读取：
- `bbox_format`：`cxcywh` / `xyxy` / `xywh`（默认 `cxcywh`）
- `nms_threshold`：IoU 阈值（默认 0.5）
- `score_threshold`：置信度阈值（默认 0.05）

NMS 实现为纯 torch（无 torchvision 依赖），支持 GPU 张量。

### 8.5 新场景接入清单

新增非分类场景时，除 4 个抽象方法外，还需：

1. 覆写 `get_task_spec()` 返回正确的 `TaskType`
2. 在 `meta().supported_tasks` 中声明支持的任务类型
3. 在 `meta().supported_learning_modes` 中声明学习模式
4. 按需覆写 `postprocess()` 实现任务特定后处理
