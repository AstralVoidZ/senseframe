"""V010: use_dann warning。

Anchor: bug 编号 I8 + 修复 commit 7f92cbb。
原始问题: use_dann=True 但 pretrain_source 未配置或非 CSI 时，对抗训练静默关闭，
         用户期望对抗训练生效但实际未生效，无任何提示。
修复方式: 检测到该情况时输出 warning 日志提示用户检查配置。

如果此测试失败，说明 V010 修复被回退。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from unit.l4_regression.conftest import _make_load_ctx


@pytest.mark.l4_regression
class TestV010UseDannWarning:
    """锁定 V010 修复：use_dann=True 但条件不满足时输出 warning。"""

    def test_use_dann_without_csi_logs_warning(self, tmp_path):
        """V010 anchor: use_dann=True + pretrain_source=none 时输出 warning。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        # pretrain_source=none → pretrain_dataset=None → 非 CSI
        params = SceneParams(extra={"use_dann": True, "pretrain_source": "none"})
        ctx = _make_load_ctx(tmp_path, params)

        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_build_csi_adversarial_loader") as mock_build, \
             patch.object(load_module._logger, "warning") as mock_warning:
            load_module.stage_load(ctx)
            mock_build.assert_not_called()
            # 验证 use_dann 相关 warning 被调用
            warning_calls = mock_warning.call_args_list
            warning_msgs = [str(c.args) + str(c.kwargs) for c in warning_calls]
            assert any("use_dann" in msg for msg in warning_msgs), (
                "如果此断言失败，V010 修复被回退：应输出 use_dann 相关 warning"
            )

    def test_use_dann_without_pretrain_source_logs_warning(self, tmp_path):
        """V010 anchor: use_dann=True 但未配置 pretrain_source 时输出 warning。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.core.params import SceneParams

        params = SceneParams(extra={"use_dann": True})  # 无 pretrain_source
        ctx = _make_load_ctx(tmp_path, params)

        with patch.object(load_module, "_compute_data_hash", return_value="hash123"), \
             patch.object(load_module, "_build_csi_adversarial_loader") as mock_build, \
             patch.object(load_module._logger, "warning") as mock_warning:
            load_module.stage_load(ctx)
            mock_build.assert_not_called()
            warning_calls = mock_warning.call_args_list
            warning_msgs = [str(c.args) + str(c.kwargs) for c in warning_calls]
            assert any("use_dann" in msg for msg in warning_msgs), (
                "如果此断言失败，V010 修复被回退：应输出 use_dann 相关 warning"
            )
