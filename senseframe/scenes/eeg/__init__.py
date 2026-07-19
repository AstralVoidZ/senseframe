"""脑电 EEG 信号场景包：BCI Competition / PhysioNet MI。

支持：
- 数据集：BCI Competition IV-2a / PhysioNet MI
- 模型：EEGNet / DeepConvNet / TransformerEEG
- 学习模式：supervised + self_supervised
- 变换：CSP / 时频分析 / 通道标准化

场景作为独立子包，通过延迟注册暴露给框架核心层。
"""
from .container import EEGContainer

__all__ = ["EEGContainer"]
