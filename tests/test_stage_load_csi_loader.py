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

    def test_csi_data_root_from_params(self, tmp_path):
        """I6 修复：CSI data_root 应从 config.scene.params.csi_data_root 读取。

        跨模态场景下 EEG 的 data_root 传给 WiFiCSIContainer 会加载失败，
        csi_data_root 独立配置让 CSI 对抗信号从正确目录加载。
        """
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
            assert mock_build.call_args.args[1] == custom_csi_root, \
                f"expected csi_data_root={custom_csi_root}, got {mock_build.call_args.args[1]}"

    def test_csi_data_root_falls_back_to_target_data_root(self, tmp_path):
        """I6 修复：未配置 csi_data_root 时 fallback 到 target data_root（向后兼容）。"""
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
            assert mock_build.call_args.args[1] == target_data_root

    def test_use_dann_without_csi_logs_warning(self, tmp_path):
        """I8 修复：use_dann=True 但 pretrain_source 未配置或非 CSI 时输出 warning。

        用户期望对抗训练生效但实际静默关闭，应显式 warning 提示。
        """
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
            assert any("use_dann" in msg for msg in warning_msgs), \
                f"expected use_dann warning, got: {warning_msgs}"

    def test_use_dann_without_pretrain_source_logs_warning(self, tmp_path):
        """I8 修复：use_dann=True 但未配置 pretrain_source 时输出 warning。"""
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
            assert any("use_dann" in msg for msg in warning_msgs), \
                f"expected use_dann warning, got: {warning_msgs}"


class TestBuildCsiAdversarialLoaderExceptNarrowed:
    """I7 修复：except Exception 收窄为 (FileNotFoundError, ValueError)。

    ImportError 等环境问题不应被吞掉，应向上抛出暴露配置错误。
    """

    def test_file_not_found_returns_none(self, tmp_path):
        """FileNotFoundError 被捕获，返回 None（降级不中断）。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = FileNotFoundError("csi data not found")
            result = _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
        assert result is None

    def test_value_error_returns_none(self, tmp_path):
        """ValueError 被捕获，返回 None（降级不中断）。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = ValueError("bad dataset name")
            result = _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
        assert result is None

    def test_import_error_not_swallowed(self, tmp_path):
        """ImportError 不被 except 吞掉（暴露环境/依赖问题）。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = ImportError("missing optional dep")
            with pytest.raises(ImportError):
                _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)

    def test_runtime_error_not_swallowed(self, tmp_path):
        """其他非 FileNotFoundError/ValueError 异常也不被吞掉。"""
        from senseframe.engine.runner.pipeline.stages import load as load_module
        from senseframe.engine.runner.pipeline.stages.load import _build_csi_adversarial_loader

        with patch("senseframe.scenes.wifi_csi.container.WiFiCSIContainer") as mock_container:
            mock_container.return_value.load_dataset.side_effect = RuntimeError("unexpected")
            with pytest.raises(RuntimeError):
                _build_csi_adversarial_loader("NTU-Fi_HAR", str(tmp_path), 32)
