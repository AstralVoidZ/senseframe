"""V007: .pt/.ckpt checkpoint 搜索。

Anchor: bug 编号 I5 + 修复 commit 7f92cbb。
原始问题: producer（scripts/p0_pretrain_with_psnr.py）产出 .pt 文件，
         loader 旧实现仅搜索 *.ckpt 导致 pretrain 静默失效；
         且 producer 直接产出在 output_dir（非 runs/），旧 loader 仅搜索 output_dir/runs/。
修复方式: _load_pretrain_checkpoint 同时搜索 .pt 和 .ckpt 扩展名，
         且搜索 output_dir 直接目录与 output_dir/runs/。

如果此测试失败，说明 V007 修复被回退。
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.l4_regression
class TestV007PtCkptCheckpointSearch:
    """锁定 V007 修复：_load_pretrain_checkpoint 支持 .pt 扩展名。"""

    def test_pt_extension_found_in_runs_dir(self, tmp_path):
        """V007 anchor: runs/ 下的 .pt checkpoint 能被找到。"""
        import torch
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # 构造 producer 实际产出的 .pt 文件（backbone_state_dict 格式）
        pt_file = runs_dir / "ntu_pretrain_test_psnr30p0.pt"
        model = torch.nn.Linear(10, 5)
        torch.save({"backbone_state_dict": model.state_dict()}, pt_file)

        result = _load_pretrain_checkpoint("test", str(tmp_path))
        assert result is not None, (
            "如果此断言失败，V007 修复被回退：.pt 扩展名 checkpoint 应能被找到"
        )
        assert str(result).endswith(".pt"), (
            "如果此断言失败，V007 修复被回退：返回路径应以 .pt 结尾"
        )

    def test_pt_in_output_dir_directly_found(self, tmp_path):
        """V007 anchor: output_dir 直接目录下的 .pt 也能被找到。"""
        import torch
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint

        # 直接在 output_dir 下放 .pt 文件（模拟 producer 真实产出位置）
        pt_file = tmp_path / "ntu_pretrain_NTU-Fi_HAR_psnr35p0.pt"
        model = torch.nn.Linear(10, 5)
        torch.save({"backbone_state_dict": model.state_dict()}, pt_file)

        result = _load_pretrain_checkpoint("NTU-Fi_HAR", str(tmp_path))
        assert result is not None, (
            "如果此断言失败，V007 修复被回退：output_dir 直接目录下的 .pt 应被找到"
        )
        assert str(result).endswith(".pt")

    def test_pt_and_ckpt_returns_latest(self, tmp_path):
        """V007 anchor: .pt 和 .ckpt 共存时返回 mtime 最新的。"""
        import torch
        from senseframe.engine.runner.pipeline.stages.load import _load_pretrain_checkpoint

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        ckpt_file = runs_dir / "ntu_pretrain_test_old.ckpt"
        pt_file = runs_dir / "ntu_pretrain_test_new.pt"
        ckpt_file.write_text("old")
        torch.save(
            {"backbone_state_dict": torch.nn.Linear(2, 2).state_dict()}, pt_file
        )
        os.utime(ckpt_file, (1000, 1000))
        os.utime(pt_file, (2000, 2000))

        result = _load_pretrain_checkpoint("test", str(tmp_path))
        assert result is not None
        assert str(result).endswith(".pt"), (
            "如果此断言失败，V007 修复被回退：mtime 最新的 .pt 应被返回"
        )
