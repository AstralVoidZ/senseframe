"""V021: I22 CSIClassifier 迁移（从 scripts 迁入 senseframe 包）。

Anchor: bug 编号 V021 + 修复 commit 3ed29a0。
原始问题: CSIClassifier 定义在 scripts/p3_eval_common.py 中，
  tests 直接依赖 scripts 模块（非包内），导致测试耦合外部脚本路径，
  且 senseframe.scenes.wifi_csi 域无法自治（分类头跨包引用 scripts）。
修复方式: I22 将 CSIClassifier 从 scripts 迁入
  senseframe/scenes/wifi_csi/classifier.py，让域内自治。

如果此测试失败，说明 V021 修复被回退（CSIClassifier 迁回 scripts 或导入路径失效）。
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.l4_regression
class TestV021CsiClassifierMigration:
    """锁定 V021 修复：CSIClassifier 迁入 senseframe 包后导入路径可用。"""

    def test_csi_classifier_importable_and_constructible(self):
        """V021 anchor: from senseframe.scenes.wifi_csi.classifier import CSIClassifier 可导入且可构造。

        如果此断言失败，V021 修复被回退。
        """
        # V021 关键断言：导入路径存在（I22 迁移后的正确路径）
        from senseframe.scenes.wifi_csi.classifier import CSIClassifier

        assert CSIClassifier is not None, (
            "如果此断言失败，V021 修复被回退：CSIClassifier 无法从 "
            "senseframe.scenes.wifi_csi.classifier 导入"
        )

        # V021 关键断言：CSIClassifier 可构造（backbone + d_model + num_classes）
        # 使用最小 mock backbone（duck-typed forward 返回 3D 张量供 mean pooling）
        class MockBackbone(torch.nn.Module):
            def forward(self, x):
                # 返回 (B, n_patches, d_model) 供 mean pooling
                return torch.randn(2, 4, 8)

        backbone = MockBackbone()
        classifier = CSIClassifier(backbone=backbone, d_model=8, num_classes=3)
        assert classifier is not None, (
            "如果此断言失败，V021 修复被回退：CSIClassifier 构造失败"
        )

        # 验证 forward 可运行（backbone → mean pool → Linear → logits）
        x = torch.randn(2, 3, 64)
        logits = classifier(x)
        assert logits.shape == (2, 3), (
            f"如果此断言失败，V021 修复被回退：CSIClassifier forward 输出形状 "
            f"应为 (2, 3)，实际 {logits.shape}"
        )
