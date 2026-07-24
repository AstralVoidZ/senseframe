"""L3 conftest：算法行为测试 fixtures。

L3 测试锚定论文/官方实现，使用真实小规模模型实例 + Stub 模型替代 MagicMock，
验证算法行为（前向/梯度/采样）符合论文定义。
"""
from __future__ import annotations

import pytest


@pytest.fixture
def mae_paper_default_mask_ratio() -> float:
    """MAE 论文默认 mask_ratio=0.75。

    Anchor: He et al., 2022, Table 1c — mask_ratio=0.75 为最优重建质量。
    """
    return 0.75
