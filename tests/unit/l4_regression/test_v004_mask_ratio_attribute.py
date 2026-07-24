"""V004: CSIFoundationModel _mask_ratio 属性（validation_step 读取）。

Anchor: bug 编号 I10 + 修复 commit 07318bf。
原始问题: _mask_ratio 是幻影属性，CSIFoundationModel 从未设置，getattr 永远返回 0.75。
         若训练用非 0.75 的 mask_ratio，PSNR 验证口径不一致（训练用 0.5，验证仍读 0.75）。
修复方式: 构造时初始化 _mask_ratio=0.75；pretrain 时写入 config.mask_ratio，
         validation_step 读取 self._mask_ratio 对齐训练口径。

回归测试策略（行为断言，消除自证）：
- 不直接断言 _mask_ratio == 0.75（镜像源码默认值，自证断言）
- 而是验证 pretrain(config.mask_ratio=X) 后 _mask_ratio == X（行为：pretrain 写入 config.mask_ratio）
- 再验证两次 pretrain 用不同 mask_ratio 后 _mask_ratio 跟随最后一次（证明属性是动态更新的，非硬编码）

回滚验证：删除 foundation_model.py 中 pretrain 的 self._mask_ratio = config.mask_ratio 赋值，
        测试会因 _mask_ratio 未更新而失败。
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
    """锁定 V004 修复：pretrain 写入 _mask_ratio，validation_step 读取对齐训练口径。"""

    def test_pretrain_updates_mask_ratio(self):
        """V004 anchor: pretrain(mask_ratio=0.5) 后 _mask_ratio == 0.5。

        行为断言：验证 pretrain 的副作用——写入 config.mask_ratio 到 _mask_ratio 属性。
        不断言初始默认值（0.75 是源码常量，镜像即自证），
        只断言 pretrain 后值被更新为 config.mask_ratio。

        回滚验证：删除 pretrain 中 self._mask_ratio = config.mask_ratio，
        _mask_ratio 保持默认值 0.75 != 0.5，测试失败。
        """
        model = _make_small_model()
        config = PretrainConfig(epochs=1, learning_rate=0.001, mask_ratio=0.5)
        model.pretrain([], config)

        assert model._mask_ratio == 0.5, (
            "V004 修复被回退：pretrain 后 _mask_ratio 应为 config.mask_ratio (0.5)，"
            f"实际 {model._mask_ratio}"
        )

    def test_pretrain_mask_ratio_tracks_last_config(self):
        """V004 anchor: 连续两次 pretrain 用不同 mask_ratio，_mask_ratio 跟随最后一次。

        行为断言：证明 _mask_ratio 是动态更新的（pretrain 写入），而非硬编码常量。
        如果 _mask_ratio 是硬编码（如永远 0.75），两次 pretrain 后值不会变。

        回滚验证：删除 pretrain 中 self._mask_ratio = config.mask_ratio，
        _mask_ratio 保持默认值，第二次 pretrain 后仍 != 0.3，测试失败。
        """
        model = _make_small_model()

        # 第一次 pretrain: mask_ratio=0.5
        config1 = PretrainConfig(epochs=1, learning_rate=0.001, mask_ratio=0.5)
        model.pretrain([], config1)
        assert model._mask_ratio == 0.5

        # 第二次 pretrain: mask_ratio=0.3
        config2 = PretrainConfig(epochs=1, learning_rate=0.001, mask_ratio=0.3)
        model.pretrain([], config2)
        assert model._mask_ratio == 0.3, (
            "V004 修复被回退：第二次 pretrain 后 _mask_ratio 应为 0.3，"
            f"实际 {model._mask_ratio}（未跟随 config.mask_ratio 更新）"
        )
