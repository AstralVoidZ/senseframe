"""V009: except 收窄（FileNotFoundError/ValueError）。

Anchor: bug 编号 I7 + 修复 commit 7f92cbb。
原始问题: _build_csi_adversarial_loader 用 except Exception 吞掉所有异常，
         ImportError 等环境/依赖问题被静默掩盖，配置错误无法暴露。
修复方式: except 收窄为 (FileNotFoundError, ValueError)，仅捕获数据相关降级异常，
         ImportError/RuntimeError 等向上抛出暴露配置/环境错误。

如果此测试失败，说明 V009 修复被回退。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.l4_regression
class TestV009ExceptNarrowing:
    """锁定 V009 修复：ImportError 不被 (FileNotFoundError, ValueError) except 吞掉。"""

    def test_file_not_found_returns_none(self, tmp_path):
        """V009 anchor: FileNotFoundError 被捕获返回 None（降级不中断）。"""
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = FileNotFoundError("csi data not found")
            result = _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
        assert result is None, (
            "如果此断言失败，V009 修复被回退：FileNotFoundError 应被捕获返回 None"
        )

    def test_value_error_returns_none(self, tmp_path):
        """V009 anchor: ValueError 被捕获返回 None（降级不中断）。"""
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = ValueError("bad dataset name")
            result = _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
        assert result is None, (
            "如果此断言失败，V009 修复被回退：ValueError 应被捕获返回 None"
        )

    def test_import_error_not_swallowed(self, tmp_path):
        """V009 anchor: ImportError 不被 except 吞掉（暴露环境/依赖问题）。"""
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = ImportError("missing optional dep")
            try:
                _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
            except ImportError:
                pass
            else:
                pytest.fail(
                    "如果此断言失败，V009 修复被回退：ImportError 应向上抛出而非被吞掉"
                )

    def test_runtime_error_not_swallowed(self, tmp_path):
        """V009 anchor: RuntimeError 等非 FileNotFoundError/ValueError 异常也不被吞掉。"""
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = RuntimeError("unexpected")
            try:
                _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
            except RuntimeError:
                pass
            else:
                pytest.fail(
                    "如果此断言失败，V009 修复被回退：RuntimeError 应向上抛出而非被吞掉"
                )
