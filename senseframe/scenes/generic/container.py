"""
通用表格场景容器：支持 CSV / NumPy 数据，内置通用 MLP 模型。

设计目标：
- 让 senseframe 能处理任意表格数据集，不局限于 WiFi CSI
- 数据加载支持两种格式：
  - CSV: {root}/{dataset_name}.csv，最后一列为标签（或通过 label_column 指定）
  - NumPy: {root}/{dataset_name}/{x_train,y_train,x_test,y_test}.npy
- 模型：内置 GenericMLP，支持可配置隐藏层
- 归一化：默认标准化（z-score），可关闭
- 与 WiFi CSI 场景并存，通过场景注册表统一访问

不依赖 SenseFi 基准库，纯 PyTorch 实现。
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ..base import (
    DatasetBundle,
    DefaultConfig,
    SceneContainer,
    SceneMeta,
    SearchSpace,
    TransformConfig,
)
from ...common.transforms import ChainedTransform


# ============================================================
# 内置通用 MLP 模型
# ============================================================
class GenericMLP(nn.Module):
    """
    通用 MLP 模型，适用于表格分类任务。

    结构：input → [Linear+ReLU+Dropout] × n_layers → Linear(output)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持 (batch, features) 和 (batch, 1, features) 两种输入
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.network(x)


# ============================================================
# 数据加载
# ============================================================
def _load_csv(
    csv_path: Path,
    label_column: Optional[str] = None,
    skip_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    加载 CSV 文件，分离特征与标签。

    Args:
        csv_path: CSV 文件路径
        label_column: 标签列名。若为 None，使用最后一列
        skip_header: 是否跳过首行（列名）

    Returns:
        (features, labels, feature_names)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    if skip_header:
        header = rows[0]
        data_rows = rows[1:]
    else:
        header = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    if not data_rows:
        raise ValueError(f"CSV file has no data rows: {csv_path}")

    # 确定标签列索引
    if label_column is not None:
        if label_column not in header:
            raise ValueError(
                f"label_column '{label_column}' not found in CSV header: {header}"
            )
        label_idx = header.index(label_column)
    else:
        label_idx = len(header) - 1

    feature_names = [h for i, h in enumerate(header) if i != label_idx]

    # 解析数值
    features = []
    labels = []
    for row in data_rows:
        if len(row) != len(header):
            continue  # 跳过不规则行
        try:
            feat = [float(row[i]) for i in range(len(row)) if i != label_idx]
            label = float(row[label_idx])
            features.append(feat)
            labels.append(label)
        except ValueError:
            continue  # 跳过非数值行

    if not features:
        raise ValueError(f"No valid numeric rows in CSV: {csv_path}")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), feature_names


def _load_numpy(
    npy_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 NumPy 格式数据集。

    期望目录结构：
        {npy_dir}/x_train.npy
        {npy_dir}/y_train.npy
        {npy_dir}/x_test.npy
        {npy_dir}/y_test.npy

    Returns:
        (x_train, y_train, x_test, y_test)
    """
    required = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    for name in required:
        if not (npy_dir / name).exists():
            raise FileNotFoundError(
                f"Required NumPy file not found: {npy_dir / name}"
            )

    x_train = np.load(npy_dir / "x_train.npy")
    y_train = np.load(npy_dir / "y_train.npy")
    x_test = np.load(npy_dir / "x_test.npy")
    y_test = np.load(npy_dir / "y_test.npy")

    return x_train, y_train, x_test, y_test


def _train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """简单 train/test 分割（带随机种子）。"""
    rng = np.random.RandomState(seed)
    n = len(features)
    indices = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return (
        features[train_idx], labels[train_idx],
        features[test_idx], labels[test_idx],
    )


# ============================================================
# 数据集信息缓存（避免重复扫描文件）
# ============================================================
_DATASET_INFO_CACHE: Dict[str, Dict[str, Any]] = {}


def _infer_dataset_info(
    dataset_name: str,
    root: str,
    label_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    推断数据集信息（num_classes, input_shape, n_features）。

    缓存结果以避免重复 IO。
    """
    cache_key = f"{dataset_name}@{root}"
    if cache_key in _DATASET_INFO_CACHE:
        return _DATASET_INFO_CACHE[cache_key]

    root_path = Path(root)
    csv_path = root_path / f"{dataset_name}.csv"
    npy_dir = root_path / dataset_name

    if csv_path.exists():
        # CSV 模式
        features, labels, feature_names = _load_csv(csv_path, label_column)
        num_classes = int(labels.max()) + 1
        input_shape = [features.shape[1]]
        info = {
            "num_classes": num_classes,
            "input_shape": input_shape,
            "n_features": features.shape[1],
            "n_samples": features.shape[0],
            "feature_names": feature_names,
            "format": "csv",
        }
    elif npy_dir.exists() and (npy_dir / "x_train.npy").exists():
        # NumPy 模式
        x_train = np.load(npy_dir / "x_train.npy")
        y_train = np.load(npy_dir / "y_train.npy")
        num_classes = int(y_train.max()) + 1
        input_shape = list(x_train.shape[1:])
        info = {
            "num_classes": num_classes,
            "input_shape": input_shape,
            "n_features": int(np.prod(x_train.shape[1:])),
            "n_samples": x_train.shape[0],
            "feature_names": [],
            "format": "numpy",
        }
    else:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in '{root}'. "
            f"Expected either {csv_path} or {npy_dir}/x_train.npy"
        )

    _DATASET_INFO_CACHE[cache_key] = info
    return info


# ============================================================
# 通用场景容器
# ============================================================
class GenericContainer(SceneContainer):
    """
    通用表格场景容器：支持 CSV / NumPy 数据。

    使用方式：
        # CSV 数据
        container = GenericContainer()
        bundle = container.load_dataset("iris", "/data")
        train_ds, test_ds = bundle.train, bundle.test

        # NumPy 数据（/data/mydata/x_train.npy 等）
        bundle = container.load_dataset("mydata", "/data")
        train_ds, test_ds = bundle.train, bundle.test

    支持的模型：GenericMLP（内置）
    """

    def meta(self) -> SceneMeta:
        return SceneMeta(
            name="generic",
            supported_tasks=["classification"],
            supported_models=["GenericMLP"],
            supported_datasets=[],  # 动态：由数据目录决定
            input_shape_hint=None,
            requires_custom_dataloader=False,
            is_dynamic_dataset=True,
            modality="tabular",  # P5 P1-D：显式声明模态，消除 shape-based fallback
        )

    def load_dataset(
        self,
        dataset_name: str,
        root: str,
        learning_mode: str = "supervised",
        label_column: Optional[str] = None,
        test_ratio: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> DatasetBundle:
        """
        加载表格数据集。

        Phase 9.1：统一返回 DatasetBundle（监督模式填 train/test）。

        自动检测格式：
        - {root}/{dataset_name}.csv 存在 → CSV 模式
        - {root}/{dataset_name}/x_train.npy 存在 → NumPy 模式

        Args:
            dataset_name: 数据集名
            root: 数据根目录
            learning_mode: 仅支持 "supervised"
            label_column: CSV 模式下的标签列名（None=最后一列）
            test_ratio: CSV 模式下的测试集比例
            seed: 随机种子（CSV 分割用）

        Returns:
            DatasetBundle: train/test 已填充
        """
        if learning_mode != "supervised":
            raise ValueError(
                f"GenericContainer only supports supervised mode, got '{learning_mode}'"
            )

        root_path = Path(root)
        csv_path = root_path / f"{dataset_name}.csv"
        npy_dir = root_path / dataset_name

        if csv_path.exists():
            # CSV 模式：加载全量后分割
            features, labels, _ = _load_csv(csv_path, label_column)
            x_train, y_train, x_test, y_test = _train_test_split(
                features, labels, test_ratio=test_ratio, seed=seed
            )
        elif npy_dir.exists() and (npy_dir / "x_train.npy").exists():
            # NumPy 模式：直接加载预分割数据
            x_train, y_train, x_test, y_test = _load_numpy(npy_dir)
        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in '{root}'. "
                f"Expected {csv_path} or {npy_dir}/x_train.npy"
            )

        # 转为 TensorDataset
        train_ds = TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_test, dtype=torch.long),
        )
        return DatasetBundle(train=train_ds, test=test_ds)

    def build_model_for_dataset(
        self,
        model_id: str,
        dataset: str,
        num_classes: int,
        learning_mode: str = "supervised",
        data_root: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> nn.Module:
        """
        Phase 9.3：构建通用模型（唯一模型构建入口）。

        优先使用显式传入的 input_dim；否则从 data_root 推断。
        旧的 build_model 抽象方法已删除，统一走此方法。
        """
        if model_id != "GenericMLP":
            raise ValueError(
                f"GenericContainer only supports 'GenericMLP', got '{model_id}'"
            )
        if learning_mode != "supervised":
            raise ValueError(
                f"GenericContainer only supports supervised mode, got '{learning_mode}'"
            )
        if input_dim is None:
            if data_root is None:
                raise ValueError(
                    "GenericMLP requires either input_dim or data_root "
                    "to infer input dimension."
                )
            info = _infer_dataset_info(dataset, data_root)
            input_dim = info["n_features"]
        return GenericMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def normalize(self, x, dataset_name: str):
        """通用场景默认不做归一化（数据加载时已处理）。"""
        return x

    def get_dataset_info(
        self,
        dataset_name: str,
        root: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """返回数据集信息。

        Phase 9.2：root 通过显式参数或 **kwargs 传递，符合 LSP。
        """
        if root is None:
            raise ValueError(
                "GenericContainer.get_dataset_info requires 'root' argument "
                "to locate the dataset files."
            )
        info = _infer_dataset_info(dataset_name, root)
        # 转换为与 WiFi CSI 一致的格式
        return {
            "num_classes": info["num_classes"],
            "input_shape": tuple(info["input_shape"]),
            "classes": info["num_classes"],
            "n_features": info["n_features"],
            "n_samples": info["n_samples"],
            "format": info["format"],
        }

    def get_default_config(
        self,
        model_id: str,
        dataset_name: str,
        root: Optional[str] = None,
        **kwargs,
    ) -> DefaultConfig:
        """返回默认训练配置。

        Phase 9.2：root 通过显式参数或 **kwargs 传递，符合 LSP。
        """
        return DefaultConfig(
            epochs=50,
            learning_rate=1e-3,
            batch_size=32,
            extra={
                "paradigm": "traditional_ml",
                "estimated_vram_mb": 256,
                "estimated_params_m": 0.1,
                "requires_gpu": False,
            },
        )

    def get_search_space(
        self,
        model_id: str,
        dataset_name: str,
        root: Optional[str] = None,
        **kwargs,
    ) -> SearchSpace:
        """通用场景的 HPO 搜索空间。

        Phase 9.2：root 通过显式参数或 **kwargs 传递，符合 LSP。
        """
        return SearchSpace(params={
            "learning_rate": {
                "type": "float", "low": 1e-5, "high": 1e-1, "log": True,
            },
            "batch_size": {
                "type": "categorical", "values": [16, 32, 64, 128],
            },
            "weight_decay": {
                "type": "float", "low": 1e-6, "high": 1e-2, "log": True,
            },
        })

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        """返回数据集的变换配置（RFC-002 阶段 U：接入 generic transforms 原语）。

        Agent 可通过 params.transform.pipeline 配置原语序列，
        通过 params.transform.augment 注入数据增强（仅 train）。

        支持的 params.transform 字段：
        - pipeline: 原语名列表，如 ["rolling_stats", "fft_features"]
        - augment: 增强原语名列表，如 ["jitter", "scaling"]（仅 train）
        - pipeline_params: 原语参数，如 {"rolling_stats": {"window": 7}}
        """
        params = kwargs.get("params") or kwargs.get("scene_params") or {}
        transform_cfg_params = params.get("transform", {}) if isinstance(params, dict) else {}
        pipeline = transform_cfg_params.get("pipeline")
        augment = transform_cfg_params.get("augment")
        pipeline_params = transform_cfg_params.get("pipeline_params", {})
        # P3 上策：从配置读取 seed，传递给 compose_transforms 创建独立 Generator
        transform_seed = transform_cfg_params.get("seed")

        train_transform = None
        eval_transform = None

        if pipeline:
            from .transforms import compose_transforms
            pipeline_fn = compose_transforms(pipeline, seed=transform_seed, **pipeline_params)
            train_transform = pipeline_fn
            eval_transform = pipeline_fn

        if augment:
            from .transforms import compose_transforms
            augment_seed = None if transform_seed is None else transform_seed + 1
            augment_fn = compose_transforms(augment, seed=augment_seed, **pipeline_params)
            if train_transform is not None:
                train_transform = ChainedTransform([train_transform, augment_fn])
            else:
                train_transform = augment_fn

        return TransformConfig(train_transform=train_transform, eval_transform=eval_transform)

    def list_datasets(self, root: str) -> List[str]:
        """列出数据根目录下所有可用数据集。"""
        root_path = Path(root)
        if not root_path.exists():
            return []
        datasets = set()
        # CSV 文件
        for p in root_path.glob("*.csv"):
            datasets.add(p.stem)
        # NumPy 目录（含 x_train.npy）
        for p in root_path.iterdir():
            if p.is_dir() and (p / "x_train.npy").exists():
                datasets.add(p.name)
        return sorted(datasets)

    def get_catalog(self):
        """返回 generic 场景的技术目录（RFC-002 阶段 U）。"""
        from .catalog import CATALOG
        return CATALOG


__all__ = [
    "GenericContainer",
    "GenericMLP",
]
