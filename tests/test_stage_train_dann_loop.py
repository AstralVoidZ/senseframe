"""_train_dann_loop 搜索空间消费测试。

验证 DANN 路径读取 ctx.resolved 的 optimizer/scheduler/gradient_clip_val（MEDIUM 5 修复）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest


class _DummyDannModel(nn.Module):
    """模拟 DANN 模型：forward(x_eeg, x_csi, lambda_) -> (logits, disc_loss)。"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x_eeg, x_csi=None, lambda_=0.0):
        logits = self.fc(x_eeg)
        disc_loss = torch.tensor(0.0, requires_grad=True)
        return logits, disc_loss


class TestDannLoopOptimizerConfig:
    """验证 DANN 路径从 ctx.resolved 读取 optimizer 配置。"""

    def test_sgd_optimizer_used(self):
        """ctx.resolved['optimizer']='sgd' 时，DANN 用 SGD 而非 AdamW。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "sgd",
            "weight_decay": 0.01,
            "scheduler": None,
            "gradient_clip_val": None,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.optim.SGD", wraps=torch.optim.SGD) as mock_sgd, \
             patch("torch.optim.AdamW", wraps=torch.optim.AdamW) as mock_adamw:
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        mock_sgd.assert_called_once()
        mock_adamw.assert_not_called()

    def test_gradient_clip_applied(self):
        """ctx.resolved['gradient_clip_val']=1.0 时，梯度被裁剪。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        # 非空 train_dataloader，让训练循环执行
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": None,
            "gradient_clip_val": 1.0,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as mock_clip:
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        mock_clip.assert_called_once()
        # 验证 max_norm 参数 == 1.0
        _, kwargs = mock_clip.call_args
        args = mock_clip.call_args[0]
        # clip_grad_norm_(parameters, max_norm, ...) 第二个位置参数是 max_norm
        assert args[1] == 1.0 or kwargs.get("max_norm") == 1.0

    def test_cosine_scheduler_steps(self):
        """ctx.resolved['scheduler']='cosine' 时，scheduler.step() 每 epoch 调用。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "gradient_clip_val": None,
            "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"), \
             patch("torch.optim.lr_scheduler.CosineAnnealingLR") as mock_cosine_cls:
            _train_dann_loop(ctx, epochs=3, learning_rate=0.01)
            # scheduler 被实例化
            mock_cosine_cls.assert_called_once()
            # 拿到 mock 实例，验证 step() 被调用 3 次（每 epoch 一次）
            mock_scheduler = mock_cosine_cls.return_value
            assert mock_scheduler.step.call_count == 3

    def test_early_stopping_breaks_loop(self):
        """ctx.resolved['early_stopping']=2 时，连续 2 epoch 无提升则 break。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
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

        with patch("senseframe.engine.runner.pipeline.stages.train._logger") as mock_logger:
            _train_dann_loop(ctx, epochs=100, learning_rate=0.01)

        # patience=2，val_acc 始终 0.0（空 val_loader）
        # epoch 1: best_val_acc 0.0→0.0 (no improve, count=0→0? 不，初始 best=0.0，val_acc=0.0 不 > 0.0，所以 count=1)
        # epoch 2: count=2 >= patience=2, break
        # 所以 "DANN epoch" 日志应出现 2 次（epoch 1 + epoch 2），early stopping 日志 1 次
        epoch_logs = [c for c in mock_logger.info.call_args_list
                      if "DANN epoch" in str(c)]
        early_stop_logs = [c for c in mock_logger.info.call_args_list
                           if "early stopping" in str(c)]
        assert len(epoch_logs) == 2  # 只跑了 2 个 epoch 就 break
        assert len(early_stop_logs) == 1  # 早停日志出现 1 次


class TestDannLoopTrainingLog:
    """验证 DANN 路径写入 training_log（MEDIUM 6 修复）。"""

    def test_training_log_populated(self):
        """DANN 训练后 ctx.training_log 应非空，长度 == epochs。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=3, learning_rate=0.01)

        assert len(ctx.training_log) == 3
        entry = ctx.training_log[0]
        assert "epoch" in entry
        assert "train_loss" in entry
        assert "val_loss" in entry
        assert "val_accuracy" in entry
        assert "val_macro_f1" in entry
        assert "lr" in entry

    def test_training_log_entry_types(self):
        """training_log entry 类型应符合 schema（epoch=int, losses=float）。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        entry = ctx.training_log[0]
        assert isinstance(entry["epoch"], int)
        assert isinstance(entry["train_loss"], (int, float))
        assert isinstance(entry["val_loss"], (int, float))
        assert isinstance(entry["val_accuracy"], (int, float))
        assert isinstance(entry["val_macro_f1"], (int, float))

    def test_training_log_values_rounded(self):
        """training_log entry 的 loss/accuracy 数值应 round 到 6 位小数（与 Lightning 路径对齐）。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        # 非空 train_dataloader + val_dataloader，产生非整数 loss
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        ctx.datamodule.val_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        entry = ctx.training_log[0]
        # 验证 round 精度：round(x, 6) == x 成立则说明已 round（或本就是有限精度）
        # 用更严格的断言：数值的小数位数不超过 6
        for key in ("train_loss", "val_loss", "val_accuracy", "val_macro_f1"):
            val = entry[key]
            # round 到 6 位后应与原值相等
            assert round(val, 6) == val, f"{key}={val!r} 未 round 到 6 位"


class TestDannLoopFinalEval:
    """验证 DANN 路径 final_eval 完整化（LOW 7 修复）。"""

    def test_final_eval_contains_val_loss_and_macro_f1(self):
        """DANN wrapper 的 final_eval 应含 val_loss + val_macro_f1。"""
        from senseframe.engine.runner.pipeline.stages import train as train_module

        ctx = MagicMock()
        ctx.dry_run = False
        ctx.config.scene.params = MagicMock()
        ctx.config.scene.params.get = MagicMock(side_effect=lambda k, d=None: True if k == "use_dann" else d)
        ctx.resolved = {"epochs": 3, "learning_rate": 0.01, "optimizer": "adamw",
                        "weight_decay": 0.0, "scheduler": None,
                        "gradient_clip_val": None, "early_stopping": None}
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.intermediate_values = {}
        ctx.dry_run = False

        # 直接调用 _train_dann_loop，然后模拟 wrapper 写 final_eval
        with patch.object(train_module, "_should_use_dann", return_value=True), \
             patch.object(train_module, "_train_dann_loop") as mock_loop:
            # 模拟 _train_dann_loop 写入 best_model_score + training_log +
            # _dann_best_val_loss/_dann_best_val_macro_f1（与真实实现写回 ctx 的字段对齐）
            def fake_loop(ctx, epochs, learning_rate):
                ctx.best_model_score = 0.85
                ctx._dann_best_val_loss = 0.6
                ctx._dann_best_val_macro_f1 = 0.82
                ctx.training_log = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6,
                                      "val_accuracy": 0.85, "val_macro_f1": 0.82, "lr": 0.01}]
            mock_loop.side_effect = fake_loop
            train_module.stage_train(ctx)

        # final_eval 应含 val_accuracy + val_loss + val_macro_f1
        assert "val_accuracy" in ctx.final_eval
        assert "val_loss" in ctx.final_eval
        assert "val_macro_f1" in ctx.final_eval

    def test_best_epoch_written_back(self):
        """DANN 路径应回写 ctx.best_epoch（best_val_acc 对应的 epoch，1-based）。"""
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        # 非空 val_dataloader 让 val_acc 非零
        # 构造标签匹配模型预测，确保 val_acc > 0（避免 RNG 导致 val_acc=0.0 时
        # best_epoch 不被设置的 flaky 问题）
        x_val = torch.randn(2, 10)
        with torch.no_grad():
            preds = ctx.model(x_val, None, 0.0)[0].argmax(dim=-1)
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = [(x_val, preds)]
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=3, learning_rate=0.01)

        # val_acc > 0（标签匹配模型预测），best_epoch 应被回写
        assert ctx.best_epoch is not None
        assert isinstance(ctx.best_epoch, int)
        assert 1 <= ctx.best_epoch <= 3  # 1-based，在 epochs 范围内

    def test_final_eval_values_rounded(self):
        """final_eval 的数值应 round 到 6 位小数。"""
        from senseframe.engine.runner.pipeline.stages import train as train_module

        ctx = MagicMock()
        ctx.dry_run = False
        ctx.config.scene.params = MagicMock()
        ctx.config.scene.params.get = MagicMock(side_effect=lambda k, d=None: True if k == "use_dann" else d)
        ctx.resolved = {"epochs": 3, "learning_rate": 0.01, "optimizer": "adamw",
                        "weight_decay": 0.0, "scheduler": None,
                        "gradient_clip_val": None, "early_stopping": None}
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.intermediate_values = {}
        ctx.dry_run = False

        with patch.object(train_module, "_should_use_dann", return_value=True), \
             patch.object(train_module, "_train_dann_loop") as mock_loop:
            # 模拟 _train_dann_loop 写入未 round 的原始数值
            def fake_loop(ctx, epochs, learning_rate):
                ctx.best_model_score = 0.85123456789
                ctx._dann_best_val_loss = 0.6123456789
                ctx._dann_best_val_macro_f1 = 0.82123456789
                ctx.best_epoch = 2
                ctx.training_log = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6,
                                      "val_accuracy": 0.85, "val_macro_f1": 0.82, "lr": 0.01}]
            mock_loop.side_effect = fake_loop
            train_module.stage_train(ctx)

        # final_eval 数值应被 round 到 6 位
        assert round(ctx.final_eval["val_accuracy"], 6) == ctx.final_eval["val_accuracy"]
        assert round(ctx.final_eval["val_loss"], 6) == ctx.final_eval["val_loss"]
        assert round(ctx.final_eval["val_macro_f1"], 6) == ctx.final_eval["val_macro_f1"]
