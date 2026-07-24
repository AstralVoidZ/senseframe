"""V008: csi_data_root 参数。

Anchor: bug 编号 I6 + 修复 commit 7f92cbb。
原始问题: 跨模态场景下 EEG 的 data_root 传给 WiFiCSIContainer 会加载失败，
         CSI 对抗信号从错误目录加载。
修复方式: csi_data_root 独立配置，从 config.scene.params.csi_data_root 读取，
         未配置时 fallback 到 target data_root（向后兼容）。

如果此测试失败，说明 V008 修复被回退。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_load_ctx(tmp_path, params=None):
    """构造 stage_load 测试用 ctx mock。"""
    ctx = MagicMock()
    ctx.dry_run = True
    ctx.config.scene.data_root = str(tmp_path)
    ctx.config.scene.params = params
    ctx.config.scene.dataset = "PhysioNet_MI"
    ctx.config.scene.model_id = "ResNet18"
    ctx.config.output_dir = str(tmp_path)
    ctx.config.save_model = True
    ctx.config.trainer.batch_size = 32
    ctx.dataset = "PhysioNet_MI"
    ctx.learning_mode = "supervised"
    ctx.scene = MagicMock()
    ctx.scene.load_dataset.return_value = MagicMock(
        train=None, test=None, val=None, unsupervised=None, supervised_finetune=None)
    ctx.output = MagicMock()
    return ctx


@pytest.mark.l4_regression
class TestV008CsiDataRootParam:
    """锁定 V008 修复：csi_data_root 参数传入而非 target data_root。"""

    def test_csi_data_root_from_params(self, tmp_path):
        """V008 anchor: csi_data_root 参数传入 _build_csi_adversarial_loader 而非 target data_root。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        custom_csi_root = str(tmp_path / "custom_csi_data")
        params = SceneParams(extra={
            "use_dann": True,
            "pretrain_source": "csi_4datasets",
            "csi_data_root": custom_csi_root,
        })
        ctx = _make_load_ctx(tmp_path, params)

        mock_loader = MagicMock()
        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_load_pretrain_checkpoint", return_value="/fake/ckpt.pt"), \
             patch.object(load_module, "_build_csi_adversarial_loader", return_value=mock_loader) as mock_build:
            load_module.stage_load(ctx)
            mock_build.assert_called_once()
            # 第二个位置参数应为 csi_data_root（custom_csi_root），而非 target data_root
            assert mock_build.call_args.args[1] == custom_csi_root, (
                "如果此断言失败，V008 修复被回退：应传入 csi_data_root 而非 target data_root"
            )

    def test_csi_data_root_falls_back_to_target_data_root(self, tmp_path):
        """V008 anchor: 未配置 csi_data_root 时 fallback 到 target data_root（向后兼容）。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        params = SceneParams(extra={
            "use_dann": True,
            "pretrain_source": "csi_4datasets",
        })
        ctx = _make_load_ctx(tmp_path, params)
        target_data_root = ctx.config.scene.data_root

        mock_loader = MagicMock()
        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_load_pretrain_checkpoint", return_value="/fake/ckpt.pt"), \
             patch.object(load_module, "_build_csi_adversarial_loader", return_value=mock_loader) as mock_build:
            load_module.stage_load(ctx)
            mock_build.assert_called_once()
            # 未配置 csi_data_root → fallback 到 target data_root
            assert mock_build.call_args.args[1] == target_data_root, (
                "如果此断言失败，V008 修复被回退：未配置 csi_data_root 时应 fallback 到 target data_root"
            )
