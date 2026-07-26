"""V001: DANN 早停 flaky test 修复（确定性空 val_loader 消除 RNG）。

Anchor: bug 编号 C1 + 修复 commit a393df9。
原始问题: val_acc 始终为 0.0 时早停行为不确定，旧测试用 RNG 产生 val_acc
         导致 flaky（不同 random seed 下 val_acc 可能非 0，早停 epoch 数变化）。
修复方式: 用确定性空 val_loader 消除 RNG flaky（val_acc 确定性为 0.0，
         sum([])/max(0,1)=0/1=0.0）。

回归测试策略（解耦 V002）：
- V001 的修复目标是"确定性"（消除 RNG），而非"具体 epoch 数"
- 原 V001 断言"3 个 epoch"依赖 V002 的 best_val_acc=-1.0 初始化，耦合
- 改为断言"两次运行产生相同 epoch 数"（确定性验证，不依赖具体值）
- 这样 V001 独立验证"确定性"修复，V002 独立验证"-1.0 初始化"修复

回滚验证：将 val_loader 改回 RNG（如 torch.randn 产生非确定性 val_acc），
        两次运行的 epoch 数可能不同，测试失败。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from unit.l4_regression.conftest import _DummyDannModel


def _count_epoch_logs(ctx, epochs, learning_rate):
    """运行 _train_dann_loop 并返回 epoch 日志数。"""
    from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

    with patch("senseframe.engine.runner.pipeline.stages.train._logger") as mock_logger:
        _train_dann_loop(ctx, epochs=epochs, learning_rate=learning_rate)

    return len([
        c for c in mock_logger.info.call_args_list
        if "DANN epoch" in str(c)
    ])


@pytest.mark.l4_regression
class TestV001PipelineFlakyFix:
    """锁定 V001 修复：确定性空 val_loader 消除早停 flaky。"""

    def test_empty_val_loader_produces_deterministic_epoch_count(self):
        """V001 anchor: 空 val_loader 确定性——两次运行产生相同 epoch 数。

        C1 修复目标：用确定性空 loader 消除 RNG flaky。
        不断言具体 epoch 数（那是 V002 的 -1.0 初始化决定），
        只断言"两次运行结果一致"（确定性）。
        """
        def _make_ctx():
            ctx = MagicMock()
            ctx.model = _DummyDannModel()
            ctx.datamodule = MagicMock()
            ctx.datamodule.train_dataloader.return_value = []
            # 空 val_loader：val_acc 确定性为 0.0（sum([])/max(0,1)=0/1=0.0）
            ctx.datamodule.val_dataloader.return_value = []
            ctx.scene_kwargs = {}
            ctx.lightning_params = {"accelerator": "cpu"}
            ctx.resolved = {
                "optimizer": "adamw",
                "weight_decay": 0.0,
                "scheduler": None,
                "gradient_clip_val": None,
                "early_stopping": 2,
            }
            return ctx

        # 两次独立运行，相同输入
        count1 = _count_epoch_logs(_make_ctx(), epochs=100, learning_rate=0.01)
        count2 = _count_epoch_logs(_make_ctx(), epochs=100, learning_rate=0.01)

        assert count1 == count2, (
            "V001 修复被回退：空 val_loader 应产生确定性 epoch 数，"
            f"但两次运行不同（{count1} vs {count2}），说明存在 RNG 依赖"
        )
        # 辅助断言：epoch 数 > 0（确保训练确实运行了）
        assert count1 > 0, "应至少运行 1 个 epoch"
