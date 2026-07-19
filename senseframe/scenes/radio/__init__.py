"""无线电信号场景包：RadioML 调制识别。

支持：
- 数据集：RadioML 2016A / 2018（24 类调制方式）
- 模型：ResNet1D / CNN1D / Transformer1D（1D 信号）
- 学习模式：supervised
- 变换：IQ → 复数谱图 / 时频图

场景作为独立子包，通过延迟注册暴露给框架核心层。
"""
from .container import RadioContainer

__all__ = ["RadioContainer"]
