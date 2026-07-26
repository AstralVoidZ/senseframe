"""L4 conftest：回归测试 fixtures。

L4 测试与 L1 共享基础 fixture（契约常量），但 docstring 强制引用违反编号 +
修复 commit。L4 是叠加层，不删除旧测试中的行为测试。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock


class _DummyDannModel(nn.Module):
    """模拟 DANN 模型：forward(x_eeg, x_csi, lambda_) -> (logits, disc_loss)。"""

    def __init__(self, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x_eeg, x_csi=None, lambda_=0.0):
        logits = self.fc(x_eeg)
        disc_loss = torch.tensor(0.0, requires_grad=True)
        return logits, disc_loss


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
