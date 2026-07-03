"""
WiFi CSI 技术目录：枚举可探索的信号处理方向。

设计理念（RFC-002 原则 5）：
- 探索状态可见，Agent 看见完整搜索空间而非盲选
- 技术目录引导 Agent 探索领域方向
- 每项含名称、描述、适用场景、参数空间

Agent 可通过此目录了解 CSI 领域有哪些技术可探索，
选择用场景包内置的实现，或通过 load_extension 生成自定义。
"""

from __future__ import annotations

from typing import Any, Dict, List


# 技术目录条目格式
CATALOG: List[Dict[str, Any]] = [
    # 去噪
    {
        "name": "hampel",
        "category": "denoising",
        "description": "Hampel 滤波器：基于滑动窗口中位数检测并替换离群点",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"window": [3, 5, 7], "threshold": [2.0, 3.0, 4.0]},
        "implemented": True,
    },
    {
        "name": "moving_average",
        "category": "denoising",
        "description": "滑动平均平滑：简单时域平滑",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"window": [3, 5, 7, 9]},
        "implemented": True,
    },
    # 相位处理
    {
        "name": "phase_unwrap",
        "category": "phase",
        "description": "相位解卷绕：将跳变相位转换为连续相位",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID"],
        "params": {},
        "implemented": True,
        "note": "需要加载 CSIphase 模态，当前默认加载 CSIamp",
    },
    {
        "name": "linear_phase_calibration",
        "category": "phase",
        "description": "线性相位校准：去除 CFO 引起的线性相位偏移",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID"],
        "params": {},
        "implemented": True,
        "note": "需要相位数据",
    },
    # 时频变换
    {
        "name": "stft",
        "category": "time_frequency",
        "description": "STFT 多普勒频谱：提取时频特征，捕捉人体运动多普勒效应",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"],
        "params": {"n_fft": [32, 64, 128, 256], "hop_length": [None, 16, 32]},
        "implemented": True,
    },
    {
        "name": "cwt",
        "category": "time_frequency",
        "description": "连续小波变换：多尺度时频分析",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID"],
        "params": {"scales": [None, "auto"], "wavelet": ["morlet"]},
        "implemented": True,
    },
    {
        "name": "fft",
        "category": "time_frequency",
        "description": "FFT 频域特征：提取频谱特征",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"n_fft": [None, 64, 128, 256]},
        "implemented": True,
    },
    # 特征工程
    {
        "name": "select_subcarriers",
        "category": "feature_engineering",
        "description": "子载波选择：选择敏感子载波，降低维度",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"],
        "params": {"indices": "list of int"},
        "implemented": True,
    },
    {
        "name": "differential",
        "category": "feature_engineering",
        "description": "差分特征：一阶/二阶差分，捕捉时序变化",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"axis": [-1, -2]},
        "implemented": True,
    },
    {
        "name": "bvp",
        "category": "feature_engineering",
        "description": "体速度剖面（BVP）：基于菲涅尔区估计人体运动速度分布",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID"],
        "params": {"freq_range": [[0.5, 5.0], [0.1, 10.0]], "fs": [1000.0, 2000.0]},
        "implemented": True,
    },
    # 数据增强
    {
        "name": "time_jitter",
        "category": "augmentation",
        "description": "时域抖动：添加高斯噪声增强泛化能力",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"sigma": [0.001, 0.01, 0.05, 0.1]},
        "implemented": True,
    },
    {
        "name": "freq_masking",
        "category": "augmentation",
        "description": "频域掩码：随机屏蔽频率分量，类似 SpecAugment",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar", "UT_HAR_data"],
        "params": {"mask_ratio": [0.05, 0.1, 0.15, 0.2]},
        "implemented": True,
    },
    {
        "name": "amplitude_rotation",
        "category": "augmentation",
        "description": "幅度旋转：对 CSI 幅值做随机旋转",
        "applicable": ["NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"],
        "params": {"angle_range": [1.0, 5.0, 10.0]},
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
