"""
通用时序场景技术目录：枚举可探索的时序特征工程与增强方向。

设计理念（RFC-002 阶段 U：场景包深度扩展）：
- 探索状态可见，Agent 看见完整搜索空间而非盲选
- 技术目录引导 Agent 探索领域方向
- 每项含名称、描述、适用场景、参数空间

generic 场景是动态数据集场景，applicable 用 ["*"] 表示适用于所有数据集。
Agent 可通过此目录了解通用时序处理有哪些技术可探索，
选择用场景包内置的实现，或通过 load_extension 生成自定义。
"""

from __future__ import annotations

from typing import Any, Dict, List


# 技术目录条目格式
CATALOG: List[Dict[str, Any]] = [
    # 特征工程
    {
        "name": "rolling_stats",
        "category": "feature_engineering",
        "description": "滑动窗口统计：对时间维做滚动 mean/std/min/max 统计",
        "applicable": ["*"],
        "params": {"window": [3, 5, 7, 9], "stat": ["mean", "std", "min", "max"]},
        "implemented": True,
    },
    {
        "name": "fft_features",
        "category": "feature_engineering",
        "description": "FFT 频域特征：提取频谱幅值特征",
        "applicable": ["*"],
        "params": {"n_fft": [None, 64, 128, 256]},
        "implemented": True,
    },
    {
        "name": "wavelet_decomp",
        "category": "feature_engineering",
        "description": "小波分解：haar 小波多层分解，提取多尺度时频特征",
        "applicable": ["*"],
        "params": {"wavelet": ["haar"], "level": [1, 2, 3, 4]},
        "implemented": True,
    },
    {
        "name": "seasonal_decompose",
        "category": "feature_engineering",
        "description": "季节性分解：移动平均提取趋势，返回去趋势残差",
        "applicable": ["*"],
        "params": {"period": [6, 12, 24, 48]},
        "implemented": True,
    },
    # 数据增强
    {
        "name": "jitter",
        "category": "augmentation",
        "description": "时域抖动：添加高斯噪声增强泛化能力",
        "applicable": ["*"],
        "params": {"sigma": [0.001, 0.01, 0.05, 0.1]},
        "implemented": True,
    },
    {
        "name": "scaling",
        "category": "augmentation",
        "description": "幅度缩放：乘以 (1+高斯噪声) 改变整体幅度",
        "applicable": ["*"],
        "params": {"sigma": [0.05, 0.1, 0.2, 0.3]},
        "implemented": True,
    },
    {
        "name": "window_warp",
        "category": "augmentation",
        "description": "窗口切片：随机裁剪一段，插值放缩回原长度",
        "applicable": ["*"],
        "params": {"ratio": [0.1, 0.2, 0.3, 0.5]},
        "implemented": True,
    },
    {
        "name": "magnitude_warp",
        "category": "augmentation",
        "description": "幅度扭曲：样条插值生成非线性幅度扭曲曲线",
        "applicable": ["*"],
        "params": {"sigma": [0.05, 0.1, 0.2], "knot": [2, 4, 8]},
        "implemented": True,
    },
]


def list_techniques() -> List[str]:
    """列出所有技术名。"""
    return [t["name"] for t in CATALOG]


def get_technique(name: str) -> Dict[str, Any]:
    """获取技术详情。"""
    for t in CATALOG:
        if t["name"] == name:
            return t
    raise KeyError(f"Technique '{name}' not found in catalog")


def list_by_category(category: str) -> List[Dict[str, Any]]:
    """按类别列出技术。"""
    return [t for t in CATALOG if t["category"] == category]


def list_categories() -> List[str]:
    """列出所有技术类别。"""
    return sorted(set(t["category"] for t in CATALOG))


def get_applicable_techniques(dataset: str) -> List[Dict[str, Any]]:
    """获取适用于指定数据集的技术。

    generic 场景 applicable=["*"]，对所有数据集都适用，故总是返回全部。
    """
    return [
        t for t in CATALOG
        if "*" in t["applicable"] or dataset in t["applicable"]
    ]


def is_augment(name: str) -> bool:
    """判断技术是否属于数据增强类（用于区分 pipeline vs augment）。"""
    for t in CATALOG:
        if t["name"] == name:
            return t["category"] == "augmentation"
    return False


def suggest_pipeline(dataset: str, categories: List[str] = None) -> List[str]:
    """基于目录推荐 pipeline 原语序列（排除增强类）。

    Agent 可据此构建 params.transform.pipeline 配置，也可自行调整。

    Args:
        dataset: 数据集名（generic 场景总是返回全部非增强类）
        categories: 限定类别（None 时包含所有非增强类）

    Returns:
        推荐的原语名列表（按目录顺序）
    """
    result = []
    for t in CATALOG:
        if t["category"] == "augmentation":
            continue
        if "*" not in t["applicable"] and dataset not in t["applicable"]:
            continue
        if categories is not None and t["category"] not in categories:
            continue
        if not t.get("implemented", False):
            continue
        result.append(t["name"])
    return result


def suggest_augment(dataset: str) -> List[str]:
    """基于目录推荐数据增强原语序列。

    Agent 可据此构建 params.transform.augment 配置。

    Args:
        dataset: 数据集名（generic 场景总是返回全部增强类）

    Returns:
        推荐的增强原语名列表
    """
    result = []
    for t in CATALOG:
        if t["category"] != "augmentation":
            continue
        if "*" not in t["applicable"] and dataset not in t["applicable"]:
            continue
        if not t.get("implemented", False):
            continue
        result.append(t["name"])
    return result


__all__ = [
    "CATALOG",
    "list_techniques",
    "get_technique",
    "list_by_category",
    "list_categories",
    "get_applicable_techniques",
    "is_augment",
    "suggest_pipeline",
    "suggest_augment",
]
