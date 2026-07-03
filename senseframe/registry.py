"""
模型与数据集注册中心（单一实现）。

Phase 10：动态注册中心，替代硬编码的 MODEL_TABLE / DATASET_INFO / NORMALIZATION_CONSTANTS。
Phase 14.1.1：合并 registry.py + registry_v2.py 为单一实现。
Phase R2（架构重构）：registry_v2.py 已降级为薄 re-export 层，本文件为唯一实现。

核心 API：
- @register_model(model_id, **metadata)      注册模型元数据
- bind_model_factory(model_id, dataset, factory, learning_mode)  绑定工厂
- register_dataset(name, **spec)              注册数据集
- register_normalization(name, strategy)      注册归一化策略
- get_model_spec / get_dataset_spec / get_normalization 查询
- unregister_model / unregister_dataset / unregister_normalization 注销（级联清理）

向后兼容 API（派生自注册表）：
- MODEL_TABLE / DATASET_INFO / EPOCHS_TABLE / NORMALIZATION_CONSTANTS
- get_model / get_default_epochs / list_models / list_datasets / get_model_info
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================
# 数据类
# ============================================================
@dataclass(frozen=True)
class ModelSpec:
    """模型规格：元数据 + 工厂绑定表。

    Attributes:
        model_id: 模型 ID（如 "ResNet18"）
        paradigm: 范式（"cnn" / "rnn" / "transformer" / "hybrid" / "traditional_ml"）
        enabled: 是否启用
        requires_gpu: 是否需要 GPU
        estimated_vram_mb: 估算显存占用 (MB)
        estimated_params_m: 估算参数量 (M)
        default_lr: 默认学习率
    """

    model_id: str
    paradigm: str
    enabled: bool = True
    requires_gpu: bool = False
    estimated_vram_mb: int = 0
    estimated_params_m: float = 0.0
    default_lr: float = 1e-3
    version: str = "1.0.0"  # P3: 模型版本管理

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            "paradigm": self.paradigm,
            "enabled": self.enabled,
            "requires_gpu": self.requires_gpu,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_params_m": self.estimated_params_m,
            "default_lr": self.default_lr,
            "version": self.version,
        }


@dataclass(frozen=True)
class DatasetSpec:
    """数据集规格（向后兼容别名，等同于 DatasetDescriptor）。"""
    name: str
    num_classes: int
    input_shape: Tuple[int, ...]
    # 加载描述
    dir_names: Tuple[str, ...] = ()        # 可能的目录名（如 ("Widardata", "Widar")）
    file_format: str = "auto"              # auto / csv / npy / mat / image
    loader_type: str = "auto"              # auto / tensor / csi_mat / csv_folder / image_folder / numpy / streaming_csv
    # 自监督模式
    unsupervised_source: str = ""          # 自监督预训练数据集名
    supervised_source: str = ""            # 监督微调数据集名

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "classes": self.num_classes,
            "dir_names": list(self.dir_names),
            "file_format": self.file_format,
            "loader_type": self.loader_type,
        }


# ============================================================
# 归一化策略
# ============================================================
class NormalizationStrategy:
    """归一化策略抽象基类。"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def is_noop(self) -> bool:
        return False


class ZScoreStrategy(NormalizationStrategy):
    """Z-Score 归一化：(x - mean) / std。

    RFC Phase B：支持 per-channel 归一化。
    - mean/std 为标量时：全局归一化（向后兼容）
    - mean/std 为向量时：按通道归一化（广播到最后一维）
    """

    def __init__(self, mean, std):
        # RFC Phase B：支持标量（float）和向量（list/np.ndarray）
        self.mean = np.asarray(mean, dtype=np.float64) if not isinstance(mean, (int, float)) else float(mean)
        self.std = np.asarray(std, dtype=np.float64) if not isinstance(std, (int, float)) else float(std)
        if isinstance(self.std, np.ndarray):
            if np.any(self.std == 0):
                raise ValueError(f"ZScoreStrategy std contains zero values")
        elif self.std == 0:
            raise ValueError(f"ZScoreStrategy std must be non-zero, got {self.std}")

    def apply(self, x: np.ndarray) -> np.ndarray:
        # RFC Phase B：向量 mean/std 广播到对应维度
        if isinstance(self.mean, np.ndarray):
            # per-channel：mean/std shape (C,)，广播到 x 的通道维（axis 0 或指定 axis）
            # 默认广播到 axis 0（假设 x shape 为 (C, ...) 或 (N, C, ...)）
            if x.ndim == 1:
                return (x - self.mean) / self.std
            # 尝试广播到第一维
            try:
                return (x - self.mean[:, np.newaxis]) / self.std[:, np.newaxis]
            except ValueError:
                # 回退到标量广播
                return (x - self.mean.mean()) / self.std.mean()
        return (x - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        # RFC Phase B：向量时转为 list 便于 JSON 序列化
        mean_val = self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean
        std_val = self.std.tolist() if isinstance(self.std, np.ndarray) else self.std
        return {"type": "ZScoreStrategy", "mean": mean_val, "std": std_val}


class IdentityStrategy(NormalizationStrategy):
    """恒等归一化（不做任何处理）。"""

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "IdentityStrategy"}

    def is_noop(self) -> bool:
        return True


# ============================================================
# 注册表存储（模块级单例）
# ============================================================
# 注册表写操作锁。保护 unregister_* 的多步清理原子性。
# register_* 本身因 overwrite=False 幂等性，竞态为良性，无需加锁。
_REGISTRY_LOCK = threading.RLock()

_MODEL_REGISTRY: Dict[str, ModelSpec] = {}
_DATASET_REGISTRY: Dict[str, DatasetSpec] = {}
_NORMALIZATION_REGISTRY: Dict[str, NormalizationStrategy] = {}

# P3: 模型版本管理（类似 skills.py 的版本管理）
# model_id -> [历史版本 ModelSpec 列表]，当前版本仍在 _MODEL_REGISTRY
_MODEL_VERSIONS: Dict[str, List[ModelSpec]] = {}

# 全局工厂绑定表：{(model_id, dataset, learning_mode): factory}
_FACTORY_BINDINGS: Dict[Tuple[str, str, str], Callable] = {}

# 场景级工厂绑定表：{scene_name: {(model_id, dataset, learning_mode): factory}}
_SCENE_FACTORIES: Dict[str, Dict[Tuple[str, str, str], Callable]] = {}

# 全局 Epoch 表：{(model_id, dataset): default_epochs}
_EPOCHS_TABLE: Dict[Tuple[str, str], int] = {}

# 场景级 Epoch 表：{scene_name: {(model_id, dataset): default_epochs}}
_SCENE_EPOCHS: Dict[str, Dict[Tuple[str, str], int]] = {}


# ============================================================
# 模型注册 API
# ============================================================
def register_model(
    model_id: str,
    *,
    paradigm: str = "unknown",
    enabled: bool = True,
    requires_gpu: bool = False,
    estimated_vram_mb: int = 0,
    estimated_params_m: float = 0.0,
    default_lr: float = 1e-3,
    version: str = "1.0.0",  # P3: 模型版本管理
    overwrite: bool = False,
) -> ModelSpec:
    """注册模型元数据（不绑定工厂）。

    工厂绑定通过 bind_model_factory() 单独完成。

    P3: 支持版本管理。同 model_id 不同 version 时，旧版本存入 _MODEL_VERSIONS
    历史记录，可通过 get_model_spec(model_id, version=...) 回溯。同版本重复
    注册保持幂等（向后兼容）。
    """
    if model_id in _MODEL_REGISTRY:
        existing = _MODEL_REGISTRY[model_id]
        if not overwrite and existing.version == version:
            # 同版本：幂等返回（向后兼容）
            return existing
        # 不同版本或 overwrite=True：将旧版本存入历史
        _MODEL_VERSIONS.setdefault(model_id, []).append(existing)
    spec = ModelSpec(
        model_id=model_id,
        paradigm=paradigm,
        enabled=enabled,
        requires_gpu=requires_gpu,
        estimated_vram_mb=estimated_vram_mb,
        estimated_params_m=estimated_params_m,
        default_lr=default_lr,
        version=version,
    )
    _MODEL_REGISTRY[model_id] = spec
    return spec


def bind_model_factory(
    model_id: str,
    dataset: str,
    factory: Callable,
    *,
    learning_mode: str = "supervised",
    default_epochs: Optional[int] = None,
) -> None:
    """绑定 (model, dataset, learning_mode) → factory（全局级）。

    Args:
        model_id: 模型 ID
        dataset: 数据集名（"*" 表示匹配任意数据集，需用户自行保证）
        factory: 无参（或仅位置/关键字参数）可调用对象，返回 nn.Module
        learning_mode: "supervised" / "self_supervised" / "*"
        default_epochs: 该 (model, dataset) 组合的默认训练轮数
    """
    key = (model_id, dataset, learning_mode)
    _FACTORY_BINDINGS[key] = factory
    if default_epochs is not None:
        _EPOCHS_TABLE[(model_id, dataset)] = default_epochs


def bind_scene_factory(
    scene_name: str,
    model_id: str,
    dataset: str,
    factory: Callable,
    *,
    learning_mode: str = "supervised",
    default_epochs: Optional[int] = None,
) -> None:
    """绑定场景级工厂：(scene, model, dataset, learning_mode) → factory。

    场景级工厂优先于全局级。场景应使用此 API 注册自己的工厂，
    避免污染全局命名空间。

    Args:
        scene_name: 场景名（如 "wifi_csi"）
        model_id: 模型 ID
        dataset: 数据集名
        factory: 可调用对象，返回 nn.Module
        learning_mode: "supervised" / "self_supervised" / "*"
        default_epochs: 该 (model, dataset) 组合的默认训练轮数
    """
    if scene_name not in _SCENE_FACTORIES:
        _SCENE_FACTORIES[scene_name] = {}
    key = (model_id, dataset, learning_mode)
    _SCENE_FACTORIES[scene_name][key] = factory
    if default_epochs is not None:
        set_scene_epochs(scene_name, model_id, dataset, default_epochs)


def set_default_epochs(model_id: str, dataset: str, epochs: int) -> None:
    """单独设置默认 epoch（全局级，不绑定工厂）。"""
    _EPOCHS_TABLE[(model_id, dataset)] = epochs


def set_scene_epochs(scene_name: str, model_id: str, dataset: str, epochs: int) -> None:
    """设置场景级默认 epoch。

    Args:
        scene_name: 场景名
        model_id: 模型 ID
        dataset: 数据集名
        epochs: 默认训练轮数
    """
    if scene_name not in _SCENE_EPOCHS:
        _SCENE_EPOCHS[scene_name] = {}
    _SCENE_EPOCHS[scene_name][(model_id, dataset)] = epochs


def get_model_spec(model_id: str, version: Optional[str] = None) -> Optional[ModelSpec]:
    """获取模型 spec（P3: 支持版本回退）。

    Args:
        model_id: 模型 ID
        version: 指定版本号。None（默认）返回当前版本；指定版本时从历史
            记录中查找，未命中返回 None。

    Returns:
        ModelSpec；version 指定但未找到时返回 None。

    Raises:
        KeyError: version 为 None 且 model_id 未注册（向后兼容行为）。
    """
    if version is None:
        # 向后兼容：未注册时抛 KeyError
        if model_id not in _MODEL_REGISTRY:
            raise KeyError(f"Model '{model_id}' is not registered")
        return _MODEL_REGISTRY[model_id]
    # P3: 查找历史版本
    for spec in _MODEL_VERSIONS.get(model_id, []):
        if spec.version == version:
            return spec
    current = _MODEL_REGISTRY.get(model_id)
    if current is not None and current.version == version:
        return current
    return None


def is_model_registered(model_id: str) -> bool:
    return model_id in _MODEL_REGISTRY


def unregister_model(model_id: str) -> bool:
    """注销模型注册，级联清理关联的工厂绑定与 epoch 条目。

    清理范围：
    - _MODEL_REGISTRY 中的当前版本 spec
    - _MODEL_VERSIONS 中的历史版本记录
    - _FACTORY_BINDINGS / _SCENE_FACTORIES 中该 model_id 的所有绑定
    - _EPOCHS_TABLE / _SCENE_EPOCHS 中该 model_id 的所有 epoch 条目

    Args:
        model_id: 模型 ID

    Returns:
        True 如果模型已注册并被移除；False 如果模型未注册
    """
    with _REGISTRY_LOCK:
        if model_id not in _MODEL_REGISTRY:
            return False
        del _MODEL_REGISTRY[model_id]
        _MODEL_VERSIONS.pop(model_id, None)
        # 级联清理工厂绑定（全局级 + 场景级）
        for k in [k for k in _FACTORY_BINDINGS if k[0] == model_id]:
            del _FACTORY_BINDINGS[k]
        for bindings in _SCENE_FACTORIES.values():
            for k in [k for k in bindings if k[0] == model_id]:
                del bindings[k]
        # 级联清理 epoch 条目（全局级 + 场景级）
        for k in [k for k in _EPOCHS_TABLE if k[0] == model_id]:
            del _EPOCHS_TABLE[k]
        for epochs_map in _SCENE_EPOCHS.values():
            for k in [k for k in epochs_map if k[0] == model_id]:
                del epochs_map[k]
        return True


# ============================================================
# 数据集注册 API
# ============================================================
def register_dataset(
    name: str,
    *,
    num_classes: int,
    input_shape: Tuple[int, ...],
    dir_names: Tuple[str, ...] = (),
    file_format: str = "auto",
    loader_type: str = "auto",
    unsupervised_source: str = "",
    supervised_source: str = "",
    overwrite: bool = False,
) -> DatasetSpec:
    if not overwrite and name in _DATASET_REGISTRY:
        return _DATASET_REGISTRY[name]
    spec = DatasetSpec(
        name=name,
        num_classes=num_classes,
        input_shape=tuple(input_shape),
        dir_names=tuple(dir_names),
        file_format=file_format,
        loader_type=loader_type,
        unsupervised_source=unsupervised_source,
        supervised_source=supervised_source,
    )
    _DATASET_REGISTRY[name] = spec
    return spec


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered")
    return _DATASET_REGISTRY[name]


def is_dataset_registered(name: str) -> bool:
    return name in _DATASET_REGISTRY


def unregister_dataset(name: str) -> bool:
    """注销数据集注册，级联清理关联的工厂绑定与 epoch 条目。

    清理范围：
    - _DATASET_REGISTRY 中的数据集 spec
    - _FACTORY_BINDINGS / _SCENE_FACTORIES 中该 dataset 的所有绑定
      （不含 dataset="*" 通配绑定）
    - _EPOCHS_TABLE / _SCENE_EPOCHS 中该 dataset 的所有 epoch 条目

    Args:
        name: 数据集名

    Returns:
        True 如果数据集已注册并被移除；False 如果数据集未注册
    """
    with _REGISTRY_LOCK:
        if name not in _DATASET_REGISTRY:
            return False
        del _DATASET_REGISTRY[name]
        # 级联清理工厂绑定（全局级 + 场景级），跳过 "*" 通配
        for k in [k for k in _FACTORY_BINDINGS if k[1] == name and k[1] != "*"]:
            del _FACTORY_BINDINGS[k]
        for bindings in _SCENE_FACTORIES.values():
            for k in [k for k in bindings if k[1] == name and k[1] != "*"]:
                del bindings[k]
        # 级联清理 epoch 条目（全局级 + 场景级）
        for k in [k for k in _EPOCHS_TABLE if k[1] == name]:
            del _EPOCHS_TABLE[k]
        for epochs_map in _SCENE_EPOCHS.values():
            for k in [k for k in epochs_map if k[1] == name]:
                del epochs_map[k]
        return True


# ============================================================
# 归一化策略注册 API
# ============================================================
def register_normalization(name: str, strategy: NormalizationStrategy, overwrite: bool = False) -> None:
    """注册归一化策略。

    Args:
        name: 策略名
        strategy: NormalizationStrategy 实例
        overwrite: 若 True，覆盖已注册的同名策略；若 False（默认），已存在时跳过并告警。
                   与 register_dataset 语义一致，避免静默覆盖。
    """
    import logging
    _log = logging.getLogger(__name__)
    if name in _NORMALIZATION_REGISTRY and not overwrite:
        _log.warning(
            f"Normalization '{name}' already registered, skipping "
            f"(use overwrite=True to replace)"
        )
        return
    _NORMALIZATION_REGISTRY[name] = strategy


def get_normalization(name: str) -> NormalizationStrategy:
    """获取归一化策略；若未注册则回退到 IdentityStrategy。"""
    return _NORMALIZATION_REGISTRY.get(name, IdentityStrategy())


def has_normalization(name: str) -> bool:
    return name in _NORMALIZATION_REGISTRY


def unregister_normalization(name: str) -> bool:
    """注销归一化策略。

    Args:
        name: 策略名

    Returns:
        True 如果策略已注册并被移除；False 如果策略未注册
    """
    with _REGISTRY_LOCK:
        if name not in _NORMALIZATION_REGISTRY:
            return False
        del _NORMALIZATION_REGISTRY[name]
        return True


# ============================================================
# 工厂解析
# ============================================================
def resolve_factory(
    model_id: str,
    dataset: str,
    learning_mode: str = "supervised",
    *,
    scene_name: Optional[str] = None,
) -> Callable:
    """解析 (model, dataset, learning_mode) 对应的工厂。

    查找顺序：
    1. 场景级（如果 scene_name 提供）：精确 → 通配 dataset → 通配 mode → 全通配
    2. 全局级：精确 → 通配 dataset → 通配 mode → 全通配
    3. 其他场景级（scene_name=None 时回退搜索所有场景）

    全部未命中则 raise KeyError。
    """
    if model_id not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{model_id}' is not registered")

    candidates = [
        (model_id, dataset, learning_mode),
        (model_id, dataset, "*"),
        (model_id, "*", learning_mode),
        (model_id, "*", "*"),
    ]

    # 1) 场景级（如果指定了 scene_name）
    if scene_name and scene_name in _SCENE_FACTORIES:
        scene_bindings = _SCENE_FACTORIES[scene_name]
        for key in candidates:
            if key in scene_bindings:
                return scene_bindings[key]

    # 2) 全局级
    for key in candidates:
        if key in _FACTORY_BINDINGS:
            return _FACTORY_BINDINGS[key]

    # 3) 其他场景级（scene_name=None 时回退）
    if scene_name is None:
        for sn, bindings in _SCENE_FACTORIES.items():
            for key in candidates:
                if key in bindings:
                    return bindings[key]

    raise KeyError(
        f"No factory bound for ({model_id}, {dataset}, {learning_mode}). "
        f"Use bind_model_factory() or bind_scene_factory() first."
    )


# ============================================================
# 视图 API（供旧代码派生表）
# ============================================================
def iter_model_specs() -> List[Tuple[str, ModelSpec]]:
    return list(_MODEL_REGISTRY.items())


def iter_dataset_specs() -> List[Tuple[str, DatasetSpec]]:
    return list(_DATASET_REGISTRY.items())


def iter_normalizations() -> List[Tuple[str, NormalizationStrategy]]:
    return list(_NORMALIZATION_REGISTRY.items())


def iter_epoch_entries(scene_name: Optional[str] = None) -> List[Tuple[Tuple[str, str], int]]:
    """迭代 epoch 条目。

    Args:
        scene_name: 如果提供，合并场景级 epoch 表（场景级覆盖全局级）。
    """
    merged = dict(_EPOCHS_TABLE)
    if scene_name and scene_name in _SCENE_EPOCHS:
        merged.update(_SCENE_EPOCHS[scene_name])
    elif scene_name is None:
        # 无场景名时合并所有场景级（后注册的覆盖先注册的）
        for sn, epochs_map in _SCENE_EPOCHS.items():
            merged.update(epochs_map)
    return list(merged.items())


# ============================================================
# 内部工具：仅供测试使用
# ============================================================
def _reset_for_test() -> None:
    """清空所有注册表（仅供单测使用）。"""
    _MODEL_REGISTRY.clear()
    _DATASET_REGISTRY.clear()
    _NORMALIZATION_REGISTRY.clear()
    _FACTORY_BINDINGS.clear()
    _EPOCHS_TABLE.clear()
    _SCENE_FACTORIES.clear()
    _SCENE_EPOCHS.clear()
    _MODEL_VERSIONS.clear()
    # 重置 WiFi CSI 场景注册标志，使 wifi_csi_register() 可重新执行
    try:
        import senseframe.scenes.wifi_csi._register as _mod
        _mod._wifi_csi_registered = False
    except ImportError:
        pass


# ============================================================
# 向后兼容视图 + 公开 API
# ============================================================
def _build_model_table() -> Dict[str, Dict[str, Any]]:
    """派生旧 MODEL_TABLE（向后兼容 routing/cli/inference/data）。"""
    return {model_id: spec.to_dict() for model_id, spec in iter_model_specs()}


def _build_dataset_info() -> Dict[str, Dict[str, Any]]:
    """派生旧 DATASET_INFO（向后兼容）。"""
    return {name: spec.to_dict() for name, spec in iter_dataset_specs()}


def _build_normalization_constants() -> Dict[str, Dict[str, float]]:
    """从归一化策略派生 NORMALIZATION_CONSTANTS（向后兼容 data.py）。"""
    out: Dict[str, Dict[str, float]] = {}
    for name, strategy in iter_normalizations():
        info = strategy.to_dict()
        if info.get("type") == "ZScoreStrategy":
            out[name] = {"mean": info["mean"], "std": info["std"]}
    return out


class _EpochsTableView:
    """EPOCHS_TABLE 视图：每次访问重建，反映注册表当前状态。"""

    def _table(self):
        return dict(iter_epoch_entries())

    def __getitem__(self, key):
        table = self._table()
        if key not in table:
            raise KeyError(key)
        return table[key]

    def __contains__(self, key) -> bool:
        return key in self._table()

    def __len__(self) -> int:
        return len(self._table())

    def __iter__(self):
        return iter(self._table())

    def items(self):
        return self._table().items()

    def keys(self):
        return self._table().keys()

    def values(self):
        return self._table().values()

    def get(self, key, default=None):
        return self._table().get(key, default)

    def __repr__(self) -> str:
        return repr(self._table())


class _LazyView(dict):
    """延迟视图：每次访问从注册表派生，反映最新状态。"""

    def __init__(self, builder_fn):
        self._builder = builder_fn

    def _refresh(self):
        return self._builder()

    def __getitem__(self, key):
        return self._refresh()[key]

    def __contains__(self, key) -> bool:
        return key in self._refresh()

    def __len__(self) -> int:
        return len(self._refresh())

    def __iter__(self):
        return iter(self._refresh())

    def __bool__(self) -> bool:
        return bool(self._refresh())

    def items(self):
        return self._refresh().items()

    def keys(self):
        return self._refresh().keys()

    def values(self):
        return self._refresh().values()

    def get(self, key, default=None):
        return self._refresh().get(key, default)

    def __repr__(self) -> str:
        return repr(self._refresh())


# 惰性视图（每次访问反映注册表当前状态）
MODEL_TABLE = _LazyView(_build_model_table)
DATASET_INFO = _LazyView(_build_dataset_info)
NORMALIZATION_CONSTANTS = _LazyView(_build_normalization_constants)
EPOCHS_TABLE = _EpochsTableView()


def get_model(
    model_id: str,
    dataset: str,
    num_classes: Optional[int] = None,
    learning_mode: str = "supervised",
    *,
    scene_name: Optional[str] = None,
):
    """查表构建模型实例（向后兼容）。

    工厂签名统一为 factory(num_classes=None, **kwargs)。
    绑定时通过 functools.partial 固定差异（如 UT_HAR 无参、ViT 关键字参数）。
    """
    import torch.nn as nn
    if not is_model_registered(model_id):
        raise ValueError(
            f"Unknown model_id: {model_id}. Available: {list(MODEL_TABLE.keys())}"
        )
    try:
        factory = resolve_factory(model_id, dataset, learning_mode, scene_name=scene_name)
    except KeyError as e:
        if learning_mode == "self_supervised":
            raise ValueError(f"Model {model_id} not available for self_supervised mode")
        if not is_dataset_registered(dataset):
            raise ValueError(
                f"Unknown dataset: {dataset}. Available: {list(DATASET_INFO.keys())}"
            )
        raise ValueError(f"Model {model_id} not available for dataset {dataset}")
    return factory(num_classes=num_classes)


def get_default_epochs(model_id: str, dataset: str, *, scene_name: Optional[str] = None) -> int:
    """获取模型在指定数据集上的默认 epoch 数。

    查找顺序：场景级（如果 scene_name 提供）→ 全局级 → 其他场景级。
    """
    key = (model_id, dataset)
    # 1) 场景级
    if scene_name and scene_name in _SCENE_EPOCHS:
        if key in _SCENE_EPOCHS[scene_name]:
            return _SCENE_EPOCHS[scene_name][key]
    # 2) 全局级
    if key in _EPOCHS_TABLE:
        return _EPOCHS_TABLE[key]
    # 3) 其他场景级（scene_name=None 时回退）
    if scene_name is None:
        for sn, epochs_map in _SCENE_EPOCHS.items():
            if key in epochs_map:
                return epochs_map[key]
    raise ValueError(f"No default epochs for ({model_id}, {dataset})")


def list_models(
    dataset: Optional[str] = None,
    paradigm: Optional[str] = None,
    enabled_only: bool = True,
) -> List[Dict[str, Any]]:
    """列出模型，可按数据集/范式过滤。"""
    results: List[Dict[str, Any]] = []
    for model_id, spec in iter_model_specs():
        if enabled_only and not spec.enabled:
            continue
        if paradigm and spec.paradigm != paradigm:
            continue
        if dataset and dataset not in DATASET_INFO:
            continue
        if dataset:
            try:
                resolve_factory(model_id, dataset, "supervised")
            except KeyError:
                continue
        entry = {"model_id": model_id, **spec.to_dict()}
        if dataset:
            try:
                entry["default_epochs"] = get_default_epochs(model_id, dataset)
            except ValueError:
                pass
        results.append(entry)
    return results


def get_model_info(model_id: str, dataset: Optional[str] = None) -> Dict[str, Any]:
    """查询单个模型详情。"""
    if not is_model_registered(model_id):
        raise ValueError(f"Unknown model_id: {model_id}")
    info = {"model_id": model_id, **get_model_spec(model_id).to_dict()}
    if dataset:
        info["default_epochs"] = get_default_epochs(model_id, dataset)
    return info


def list_datasets() -> List[Dict[str, Any]]:
    """列出所有可用数据集。"""
    return [{"name": k, **v} for k, v in DATASET_INFO.items()]


# ============================================================
# 向后兼容视图（延迟重建）
# ============================================================
# 视图在首次访问时从注册表派生，不再在模块加载时固定。
# 场景注册后视图自动反映最新状态。
