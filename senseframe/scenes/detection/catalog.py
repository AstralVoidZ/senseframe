"""
目标检测场景技术目录：枚举可探索的图像增强与 bbox 处理方向。

设计理念（RFC-002 阶段 U：场景包深度扩展）：
- 探索状态可见，Agent 看见完整搜索空间而非盲选
- 技术目录引导 Agent 探索领域方向
- 每项含名称、描述、适用场景、参数空间

detection 场景的内置数据集为 dummy_box / tiny_coco，
applicable 字段列出适用数据集。
Agent 可通过此目录了解检测领域有哪些技术可探索，
选择用场景包内置的实现，或通过 load_extension 生成自定义。
"""

from __future__ import annotations

from typing import Any, Dict, List


# 技术目录条目格式
CATALOG: List[Dict[str, Any]] = [
    # 图像增强
    {
        "name": "hsv_jitter",
        "category": "augmentation",
        "description": "HSV 空间抖动：转 HSV 后对 H/S/V 通道加噪声再转回 RGB",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {
            "hue": [0.05, 0.1, 0.15],
            "saturation": [0.05, 0.1, 0.15],
            "brightness": [0.05, 0.1, 0.15],
        },
        "implemented": True,
    },
    {
        "name": "cutout",
        "category": "augmentation",
        "description": "随机遮挡：在图像上随机选矩形区域置零",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {"n_holes": [1, 2, 3], "length": [8, 16, 32]},
        "implemented": True,
    },
    {
        "name": "mixup",
        "category": "augmentation",
        "description": "MixUp 增强：对样本及 one-hot 标签做线性插值混合",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {"alpha": [0.1, 0.2, 0.4]},
        "implemented": True,
        "note": "batch 级变换，需在 collate 后应用",
    },
    {
        "name": "random_erasing",
        "category": "augmentation",
        "description": "随机擦除：随机选择面积与长宽比随机的区域置零",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {
            "area_ratio": [[0.02, 0.1], [0.02, 0.2], [0.1, 0.3]],
            "min_aspect": [0.3, 0.5],
        },
        "implemented": True,
    },
    # bbox 处理
    {
        "name": "bbox_clip",
        "category": "bbox_processing",
        "description": "bbox 裁剪：将边界框裁剪到图像边界内",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {},
        "implemented": True,
    },
    {
        "name": "bbox_flip",
        "category": "bbox_processing",
        "description": "bbox 翻转：水平/垂直翻转边界框坐标",
        "applicable": ["dummy_box", "tiny_coco"],
        "params": {"flip_type": ["horizontal", "vertical"]},
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
    """获取适用于指定数据集的技术。"""
    return [t for t in CATALOG if dataset in t["applicable"]]


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
        dataset: 数据集名
        categories: 限定类别（None 时包含所有非增强类）

    Returns:
        推荐的原语名列表（按目录顺序）
    """
    result = []
    for t in CATALOG:
        if t["category"] == "augmentation":
            continue
        if dataset not in t["applicable"]:
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
        dataset: 数据集名

    Returns:
        推荐的增强原语名列表
    """
    result = []
    for t in CATALOG:
        if t["category"] != "augmentation":
            continue
        if dataset not in t["applicable"]:
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
