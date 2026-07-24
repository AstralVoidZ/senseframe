"""V012: I12 except 改 warning（原 except Exception 静默吞掉异常）。

Anchor: bug 编号 V012 + 修复 commit 43387a1。
原始问题: SelfSupervisedModule.validation_step 中 MAE 重建失败时，
  ``except Exception`` 静默吞掉异常（仅 debug 日志或无日志），导致 PSNR
  缓存失败不可观测，PSNREarlyStoppingCallback 读到 None 缓存无法定位根因。
修复方式: ``except Exception as e`` 后调用 ``_logger.warning``，消息含
  "PSNR reconstruction cache failed"，确保缓存失败可观测。

如果此测试失败，说明 V012 修复被回退（except 回到静默吞异常）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.l4_regression
class TestV012ExceptWarning:
    """锁定 V012 修复：MAE 重建失败时 logger.warning 被调用。"""

    def test_except_logs_warning_on_mae_failure(self):
        """V012 anchor: mae_reconstruct 抛异常时 logger.warning 被调用，消息含 PSNR/cache/failed。

        如果此断言失败，V012 修复被回退。
        """
        from senseframe.engine.self_supervised import SelfSupervisedModule

        # 构造 mock MAE model：mae_reconstruct 抛异常，forward 返回二元组张量
        model = MagicMock()
        model.mae_reconstruct.side_effect = RuntimeError("fake error")
        # forward 返回二元组张量，供 ce_criterion 消费（避免 unpack 失败）
        model.return_value = (torch.randn(2, 7), torch.randn(2, 7))

        module = SelfSupervisedModule(model=model, num_classes=7)
        batch = (torch.randn(2, 3, 32), torch.tensor([0, 1]))

        with patch.object(module, "log"):
            with patch("senseframe.engine.self_supervised._logger") as mock_logger:
                module.validation_step(batch, 0)

        # V012 关键断言：warning 被调用（非 debug / 非静默）
        mock_logger.warning.assert_called_once(), (
            "如果此断言失败，V012 修复被回退：mae_reconstruct 抛异常时应调用 "
            "logger.warning（非 debug / 非静默吞异常）"
        )
        log_msg = str(mock_logger.warning.call_args)
        assert "PSNR" in log_msg or "cache" in log_msg or "failed" in log_msg, (
            f"如果此断言失败，V012 修复被回退：warning 消息应含 PSNR/cache/failed，"
            f"实际: {log_msg!r}"
        )
