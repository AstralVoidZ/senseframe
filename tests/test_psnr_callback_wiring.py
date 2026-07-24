"""PSNR Callback 全链路接线测试。

验证 self_supervised + psnr metric 时 stage_build 接线 PSNREarlyStoppingCallback（MEDIUM 3），
且 SelfSupervisedModule.validation_step 产生 _psnr_reconstruction/_psnr_target（MEDIUM 4）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest


class TestPsnrCallbackWiring:
    """验证 stage_build 接线 PSNREarlyStoppingCallback。"""

    def test_psnr_metric_wires_psnr_callback(self):
        """is_self_supervised + pretrain_early_stop_metric=psnr → 接线 PSNREarlyStoppingCallback。"""
        from senseframe.engine.runner.pipeline.stages import build as build_module
        from senseframe.engine.callbacks.psnr_early_stopping import PSNREarlyStoppingCallback
        from senseframe.core.params import SceneParams

        ctx = MagicMock()
        ctx.learning_mode = "self_supervised"
        ctx.config.scene.data_root = "/tmp/data"
        ctx.config.scene.task_spec = None
        ctx.config.scene.params = SceneParams(extra={"pretrain_early_stop_metric": "psnr"})
        ctx.config.trainer.early_stopping = 10
        ctx.config.trainer.early_stopping_monitor = "val_loss"
        ctx.config.trainer.early_stopping_min_delta = 0.1
        ctx.config.extra_callbacks = None
        ctx.config.module_factory = None
        ctx.config.datamodule_factory = None
        ctx.model_id = "CSIFoundationModel"
        ctx.dataset = "UT_HAR_data"
        ctx.num_classes = 7
        ctx.feature_spec = MagicMock(feature_dim=10)
        ctx.scene_info = {"input_shape": [1, 250, 90], "n_features": 10}
        ctx.route_config = {"max_epochs": 50}
        ctx.resolved = {"learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
                        "scheduler": None, "batch_size": 32, "num_workers": 0,
                        "metrics": ["accuracy", "macro_f1"], "logger": "csv",
                        "pin_memory": False, "persistent_workers": False}
        ctx.bundle = MagicMock(
            unsupervised=MagicMock(), supervised_finetune=MagicMock(),
            val=MagicMock(), test=MagicMock())
        ctx.scene = MagicMock()
        ctx.scene.get_transforms.return_value = MagicMock(
            train_transform=None, eval_transform=None, supervised_transform=None)
        ctx.scene.build_model_for_dataset.return_value = MagicMock(spec=nn.Module)
        ctx.scene_kwargs = {}
        ctx.data_profile = None
        ctx.output_dir = MagicMock()
        ctx.log_writer = MagicMock()
        ctx.extra = None
        ctx.pruner = None
        ctx.trial_id = None
        ctx.pretrain_checkpoint = None
        ctx.intermediate_values = {}

        # build.py 中 GenericDataModule / SelfSupervisedModule 是函数级 import，
        # 必须在源模块 patch（参考 Task 1 _make_build_ctx 模式）。
        with patch("senseframe.engine.datamodule.GenericDataModule", MagicMock()), \
             patch("senseframe.engine.self_supervised.SelfSupervisedModule", MagicMock()), \
             patch.object(build_module, "build_logger", MagicMock()):
            build_module.stage_build(ctx)

        # 应含 PSNREarlyStoppingCallback，不含 Lightning EarlyStopping
        psnr_cbs = [c for c in ctx.callbacks if isinstance(c, PSNREarlyStoppingCallback)]
        assert len(psnr_cbs) == 1
        from pytorch_lightning.callbacks import EarlyStopping
        es_cbs = [c for c in ctx.callbacks if isinstance(c, EarlyStopping)]
        assert len(es_cbs) == 0

    def test_default_wires_lightning_early_stopping(self):
        """默认配置（无 pretrain_early_stop_metric）→ 接线 Lightning EarlyStopping。"""
        from senseframe.engine.runner.pipeline.stages import build as build_module
        from pytorch_lightning.callbacks import EarlyStopping
        from senseframe.engine.callbacks.psnr_early_stopping import PSNREarlyStoppingCallback

        ctx = MagicMock()
        ctx.learning_mode = "supervised"
        ctx.config.scene.data_root = "/tmp/data"
        ctx.config.scene.task_spec = None
        ctx.config.scene.params = None
        ctx.config.trainer.early_stopping = 10
        ctx.config.trainer.early_stopping_monitor = "val_loss"
        ctx.config.trainer.early_stopping_min_delta = 0.0
        ctx.config.extra_callbacks = None
        ctx.config.module_factory = None
        ctx.config.datamodule_factory = None
        ctx.model_id = "MLP"
        ctx.dataset = "UT_HAR_data"
        ctx.num_classes = 7
        ctx.feature_spec = MagicMock(feature_dim=10)
        ctx.scene_info = {"input_shape": [1, 250, 90], "n_features": 10}
        ctx.route_config = {"max_epochs": 50}
        ctx.resolved = {"learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
                        "scheduler": None, "batch_size": 32, "num_workers": 0,
                        "metrics": ["accuracy", "macro_f1"], "logger": "csv",
                        "pin_memory": False, "persistent_workers": False}
        ctx.bundle = MagicMock(
            train=MagicMock(), val=MagicMock(), test=MagicMock())
        ctx.scene = MagicMock()
        ctx.scene.get_transforms.return_value = MagicMock(
            train_transform=None, eval_transform=None)
        ctx.scene.build_model_for_dataset.return_value = MagicMock(spec=nn.Module)
        ctx.scene_kwargs = {}
        ctx.data_profile = None
        ctx.output_dir = MagicMock()
        ctx.log_writer = MagicMock()
        ctx.extra = None
        ctx.pruner = None
        ctx.trial_id = None
        ctx.pretrain_checkpoint = None
        ctx.intermediate_values = {}
        # supervised 分支读取 ctx.config.trainer.epochs
        ctx.config.trainer.epochs = 50

        with patch("senseframe.engine.datamodule.GenericDataModule", MagicMock()), \
             patch("senseframe.engine.module.GenericLightningModule", MagicMock()), \
             patch.object(build_module, "build_logger", MagicMock()):
            build_module.stage_build(ctx)

        es_cbs = [c for c in ctx.callbacks if isinstance(c, EarlyStopping)]
        assert len(es_cbs) == 1
        psnr_cbs = [c for c in ctx.callbacks if isinstance(c, PSNREarlyStoppingCallback)]
        assert len(psnr_cbs) == 0


class TestSelfSupervisedPsnrCache:
    """验证 SelfSupervisedModule.validation_step 产生 PSNR 缓存。"""

    def test_mae_model_caches_psnr_tensors(self):
        """model 有 _mae_forward_loss 方法时，validation 缓存 _psnr_reconstruction/_psnr_target。"""
        from senseframe.engine.self_supervised import SelfSupervisedModule

        # 构造 mock MAE model（duck-typed _mae_forward_loss + patch_embedder）
        class MockMaeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embedder = MagicMock()
                self.patch_embedder.to_patches.return_value = torch.randn(2, 10, 4)
                self.patch_embedder.proj = MagicMock(return_value=torch.randn(2, 10, 8))
                self.pos_embed = torch.zeros(1, 10, 8)
                self._mae_forward_loss = MagicMock(return_value=torch.tensor(0.5))

            def random_masking(self, patches, mask_ratio):
                return patches, torch.ones(2, 10), MagicMock()

            def _forward_encoder(self, x):
                return x

            def _forward_decoder(self, enc, ids):
                return torch.randn(2, 10, 4)

            def forward(self, x1, x2=None, flag=None):
                return torch.randn(2, 7), torch.randn(2, 7)

        model = MockMaeModel()
        module = SelfSupervisedModule(
            model=model, learning_rate=0.001, weight_decay=0.0,
            metrics=["accuracy"], num_classes=7, incremental_log_writer=MagicMock(),
        )
        module._psnr_reconstruction = None
        module._psnr_target = None

        batch = (torch.randn(2, 1, 250), torch.tensor([0, 1]))
        with patch.object(module, "log"):
            module.validation_step(batch, 0)

        assert module._psnr_reconstruction is not None
        assert module._psnr_target is not None


class TestPSNRCacheMinor:
    """Task 5 残留 Minor：PSNR cache 日志 + mask_ratio 读取。"""

    def test_except_logs_debug(self):
        """MAE 重建失败时应记录 debug 日志（不静默吞异常）。"""
        import logging
        from unittest.mock import patch, MagicMock
        from senseframe.engine.self_supervised import SelfSupervisedModule

        # 构造一个 model：有 _mae_forward_loss + patch_embedder，但 to_patches 抛异常
        model = MagicMock()
        model._mae_forward_loss = lambda x, r: x
        # patch_embedder.to_patches 抛异常触发 except 分支
        model.patch_embedder.to_patches.side_effect = RuntimeError("fake error")
        # forward 返回二元组张量，供 ce_criterion 消费（避免 unpack 失败）
        model.return_value = (torch.randn(2, 7), torch.randn(2, 7))

        module = SelfSupervisedModule(model=model, num_classes=7)
        batch = (torch.randn(2, 3, 32), torch.tensor([0, 1]))

        with patch.object(module, "log"):
            with patch("senseframe.engine.self_supervised._logger") as mock_logger:
                module.validation_step(batch, 0)

        # 验证 debug 日志被调用
        mock_logger.debug.assert_called_once()
        # 日志消息应含 "PSNR" 或 "cache" 或 "failed"
        log_msg = str(mock_logger.debug.call_args)
        assert "PSNR" in log_msg or "cache" in log_msg or "failed" in log_msg

    def test_mask_ratio_from_model_attribute(self):
        """model 有 _mask_ratio 属性时，validation_step 应使用该值而非硬编码 0.75。"""
        from unittest.mock import patch, MagicMock
        from senseframe.engine.self_supervised import SelfSupervisedModule

        model = MagicMock()
        model._mae_forward_loss = lambda x, r: x
        model._mask_ratio = 0.5  # 自定义 mask_ratio
        # patch_embedder.to_patches 返回有效张量
        model.patch_embedder.to_patches.return_value = torch.randn(2, 4, 16)
        model.patch_embedder.proj.return_value = torch.randn(2, 4, 16)
        model.pos_embed = torch.zeros(1, 4, 16)
        model.random_masking.return_value = (
            torch.randn(2, 2, 16),  # x_visible
            torch.ones(2, 4),        # mask
            torch.randint(0, 4, (2, 4)),  # ids_restore
        )
        model._forward_encoder.return_value = torch.randn(2, 2, 16)
        model._forward_decoder.return_value = torch.randn(2, 4, 16)
        # forward 返回二元组张量，供 ce_criterion 消费（避免 unpack 失败）
        model.return_value = (torch.randn(2, 7), torch.randn(2, 7))

        module = SelfSupervisedModule(model=model, num_classes=7)
        batch = (torch.randn(2, 3, 32), torch.tensor([0, 1]))

        with patch.object(module, "log"):
            module.validation_step(batch, 0)

        # 验证 random_masking 被调用时 mask_ratio=0.5（而非 0.75）
        call_args = model.random_masking.call_args
        # random_masking(patches, mask_ratio=...) 第二个参数
        assert call_args.kwargs.get("mask_ratio") == 0.5 or \
               (len(call_args.args) >= 2 and call_args.args[1] == 0.5)
