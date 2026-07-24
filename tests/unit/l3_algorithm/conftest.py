"""L3 conftest：算法行为测试 fixtures。

L3 测试锚定论文/官方实现，使用真实小规模模型实例 + Stub 模型替代 MagicMock，
验证算法行为（前向/梯度/采样）符合论文定义。
"""
from __future__ import annotations

import pytest
import torch

from tests.fakes.stub_models import StubDannModel, StubMaeModel


@pytest.fixture
def stub_dann_model() -> StubDannModel:
    """StubDannModel：DANN 模型的可预测替身。

    用于 DANN 梯度反转 / decoder freeze 测试，返回可预测的 (logits, disc_loss)。
    """
    torch.manual_seed(0)
    return StubDannModel(in_features=10, num_classes=7)


@pytest.fixture
def stub_mae_model() -> StubMaeModel:
    """StubMaeModel：MAE 模型的可预测替身。

    用于 MAE 前向行为测试，mae_reconstruct 返回可预测的 (recon, target, mask)。
    mask_ratio 默认 0.75（He 2022 论文默认值）。
    """
    return StubMaeModel(mask_ratio=0.75)


@pytest.fixture
def mae_paper_default_mask_ratio() -> float:
    """MAE 论文默认 mask_ratio=0.75。

    Anchor: He et al., 2022, Table 1c — mask_ratio=0.75 为最优重建质量。
    """
    return 0.75
