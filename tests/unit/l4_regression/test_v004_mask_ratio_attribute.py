"""V004: CSIFoundationModel _mask_ratio 属性（validation_step 读取）。

Anchor: bug 编号 I10 + 修复 commit 07318bf。
原始问题: _mask_ratio 是幻影属性，CSIFoundationModel 从未设置，getattr 永远返回 0.75。
         若训练用非 0.75 的 mask_ratio，PSNR 验证口径不一致（训练用 0.5，验证仍读 0.75）。
修复方式: 构造时初始化 _mask_ratio=0.75；pretrain 时写入 config.mask_ratio，
         validation_step 读取 self._mask_ratio 对齐训练口径。

如果此测试失败，说明 V004 修复被回退。
"""
from __future__ import annotations

from typing import Tuple

import pytest

from senseframe.core.foundation_model import PretrainConfig
from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel


def _make_small_model(
    input_shape: Tuple[int, int] = (3, 64),
    **kwargs,
) -> CSIFoundationModel:
    """构造小模型用于快速测试。"""
    defaults = dict(
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        patch_len=8,
        decoder_dim=16,
    )
    defaults.update(kwargs)
    return CSIFoundationModel(input_shape=input_shape, **defaults)


@pytest.mark.l4_regression
class TestV004MaskRatioAttribute:
    """锁定 V004 修复：_mask_ratio 属性正确初始化与 pretrain 写入。"""

    def test_mask_ratio_initial_and_after_pretrain(self):
        """V004 anchor: 初始 _mask_ratio==0.75；pretrain 后 _mask_ratio==config.mask_ratio (0.5)。"""
        model = _make_small_model()

        # 初始 _mask_ratio 应为默认值 0.75
        assert model._mask_ratio == 0.75, (
            "如果此断言失败，V004 修复被回退：初始 _mask_ratio 应为默认值 0.75"
        )

        # pretrain 时应写入 config.mask_ratio
        config = PretrainConfig(epochs=1, learning_rate=0.001, mask_ratio=0.5)
        # 构造一个空 DataLoader（pretrain 会遍历，空则跳过）
        model.pretrain([], config)

        assert model._mask_ratio == 0.5, (
            "如果此断言失败，V004 修复被回退：pretrain 后 _mask_ratio 应为 config.mask_ratio (0.5)"
        )
