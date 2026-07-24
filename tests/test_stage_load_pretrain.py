"""stage_load pretrain_source 加载测试。

验证 scene.params.pretrain_source 配置时 stage_load 能解析预训练数据集
并加载 pretrain checkpoint（v2 差距 3 修复）。
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestPretrainSourceResolution:
    """验证 _resolve_pretrain_source 函数。"""

    def test_none_returns_none(self):
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("none", "UT_HAR_data") is None

    def test_csi_4datasets_cross_modal(self):
        """csi_4datasets + EEG target → NTU-Fi_HAR。"""
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("csi_4datasets", "PhysioNet_MI") == "NTU-Fi_HAR"

    def test_csi_4datasets_same_modal(self):
        """csi_4datasets + CSI target → target 自己。"""
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("csi_4datasets", "UT_HAR_data") == "UT_HAR_data"

    def test_explicit_dataset_name(self):
        """显式数据集名直接返回。"""
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("NTU-Fi_HAR", "UT_HAR_data") == "NTU-Fi_HAR"

    def test_unknown_falls_back_to_none(self):
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("unknown_source", "UT_HAR_data") is None

    def test_radioml_resolves(self):
        """radioml → RadioML2018（与 target 无关）。"""
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("radioml", "UT_HAR_data") == "RadioML2018"

    def test_eegmmidb_resolves(self):
        """eegmmidb → PhysioNet_MI（与 target 无关）。"""
        from senseframe.engine.runner.pipeline.stages.load import _resolve_pretrain_source
        assert _resolve_pretrain_source("eegmmidb", "UT_HAR_data") == "PhysioNet_MI"


class TestLoadPretrainCheckpoint:
    """验证 _load_pretrain_checkpoint 函数。"""

    def test_dir_not_found_returns_none(self, tmp_path):
        """output_dir/runs 不存在 → None。"""
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint
        result = _load_pretrain_checkpoint("NTU-Fi_HAR", str(tmp_path))
        assert result is None

    def test_no_candidates_returns_none(self, tmp_path):
        """runs/ 存在但无匹配 .ckpt → None。"""
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint
        (tmp_path / "runs").mkdir()
        result = _load_pretrain_checkpoint("NTU-Fi_HAR", str(tmp_path))
        assert result is None

    def test_multiple_candidates_returns_latest(self, tmp_path):
        """多个候选 .ckpt → 返回 mtime 最新的。"""
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        old_ckpt = runs_dir / "ckpt_old_NTU-Fi_HAR.ckpt"
        new_ckpt = runs_dir / "ckpt_new_NTU-Fi_HAR.ckpt"
        old_ckpt.write_text("old")
        new_ckpt.write_text("new")
        # 显式设置 mtime：old 更早，new 更晚
        os.utime(old_ckpt, (1000, 1000))
        os.utime(new_ckpt, (2000, 2000))
        result = _load_pretrain_checkpoint("NTU-Fi_HAR", str(tmp_path))
        assert result is not None
        assert os.path.basename(result) == "ckpt_new_NTU-Fi_HAR.ckpt"


class TestStageLoadPretrainIntegration:
    """验证 stage_load 集成 pretrain_source。"""

    def test_pretrain_source_triggers_checkpoint_load(self, tmp_path):
        """scene.params.pretrain_source 设置时应加载 pretrain checkpoint。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        ctx = MagicMock()
        ctx.dry_run = True
        ctx.config.scene.data_root = str(tmp_path)
        ctx.config.scene.params = SceneParams(extra={"pretrain_source": "csi_4datasets"})
        ctx.config.scene.dataset = "PhysioNet_MI"
        ctx.config.scene.model_id = "ResNet18"
        ctx.config.output_dir = str(tmp_path)
        ctx.config.save_model = True
        ctx.dataset = "PhysioNet_MI"
        ctx.learning_mode = "supervised"
        ctx.scene = MagicMock()
        ctx.scene.load_dataset.return_value = MagicMock(train=None, test=None)
        ctx.output = MagicMock()

        with patch.object(load_module, "_load_pretrain_checkpoint") as mock_load:
            mock_load.return_value = "/fake/pretrain.ckpt"
            load_module.stage_load(ctx)
            mock_load.assert_called_once()
            # csi_4datasets + PhysioNet_MI（EEG）→ NTU-Fi_HAR（跨模态默认）
            assert mock_load.call_args.args[0] == "NTU-Fi_HAR"
            # ctx.pretrain_checkpoint 应被写入 mock 的返回值
            assert ctx.pretrain_checkpoint == "/fake/pretrain.ckpt"
