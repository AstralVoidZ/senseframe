"""stage_load csi_loader 注入测试。

验证 use_dann=True + pretrain_source 为 CSI 时 stage_load 注入 csi_loader（HIGH 2 修复）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_load_ctx(tmp_path, params=None):
    """构造 stage_load 测试用 ctx mock（Minor 3：消除 4 个测试的重复 setup）。

    Args:
        tmp_path: pytest fixture 提供的临时目录
        params: SceneParams 实例；None 表示无 params（向后兼容场景）

    Returns:
        配置好的 MagicMock ctx，可直接传给 stage_load
    """
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


class TestCsiLoaderInjection:
    """验证 csi_loader 注入逻辑。"""

    def test_no_dann_no_csi_loader(self, tmp_path):
        """use_dann=False 时，scene_kwargs 不含 csi_loader（向后兼容）。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        ctx = _make_load_ctx(tmp_path, SceneParams())  # 无 use_dann

        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_load_pretrain_checkpoint", return_value=None), \
             patch.object(load_module, "_build_csi_adversarial_loader") as mock_build:
            load_module.stage_load(ctx)
            mock_build.assert_not_called()

        assert "csi_loader" not in (ctx.scene_kwargs or {})

    def test_dann_with_csi_pretrain_injects_loader(self, tmp_path):
        """use_dann=True + pretrain_source=csi_4datasets 时，scene_kwargs 含 csi_loader。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        params = SceneParams(extra={"use_dann": True, "pretrain_source": "csi_4datasets"})
        ctx = _make_load_ctx(tmp_path, params)

        mock_loader = MagicMock()
        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_load_pretrain_checkpoint", return_value="/fake/ckpt.pt"), \
             patch.object(load_module, "_build_csi_adversarial_loader", return_value=mock_loader) as mock_build:
            load_module.stage_load(ctx)
            mock_build.assert_called_once()

        assert ctx.scene_kwargs.get("csi_loader") is mock_loader

    def test_dann_without_csi_pretrain_no_loader(self, tmp_path):
        """use_dann=True 但 pretrain_source=none 时，不注入 csi_loader。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        params = SceneParams(extra={"use_dann": True, "pretrain_source": "none"})
        ctx = _make_load_ctx(tmp_path, params)

        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_build_csi_adversarial_loader") as mock_build:
            load_module.stage_load(ctx)
            mock_build.assert_not_called()

        assert "csi_loader" not in (ctx.scene_kwargs or {})

    def test_dann_csi_loader_build_returns_none_skips_injection(self, tmp_path):
        """_build_csi_adversarial_loader 返回 None（加载失败）时，不注入 csi_loader 但不报错。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        params = SceneParams(extra={"use_dann": True, "pretrain_source": "csi_4datasets"})
        ctx = _make_load_ctx(tmp_path, params)

        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_load_pretrain_checkpoint", return_value="/fake/ckpt.pt"), \
             patch.object(load_module, "_build_csi_adversarial_loader", return_value=None):
            load_module.stage_load(ctx)  # 不应抛异常

        assert "csi_loader" not in (ctx.scene_kwargs or {})
