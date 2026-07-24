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
        # I1 修复后 best_val_acc 初始 -1.0：
        # epoch 1: val_acc=0.0 > -1.0 → True（best 更新），count=0
        # epoch 2: val_acc=0.0 > 0.0 → False，count=1
        # epoch 3: val_acc=0.0 > 0.0 → False，count=2 >= patience=2, break
        # 所以 "DANN epoch" 日志应出现 3 次（epoch 1+2+3），early stopping 日志 1 次
        epoch_logs = [c for c in mock_logger.info.call_args_list
                      if "DANN epoch" in str(c)]
        early_stop_logs = [c for c in mock_logger.info.call_args_list
                           if "early stopping" in str(c)]
        assert len(epoch_logs) == 3  # 跑了 3 个 epoch 才 break
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
            # 模拟 _train_dann_loop 返回 DannTrainResult（与真实实现返回值对齐）
            def fake_loop(ctx, epochs, learning_rate):
                from senseframe.engine.runner.pipeline.stages.train import DannTrainResult
                return DannTrainResult(
                    best_score=0.85, best_epoch=1,
                    best_val_loss=0.6, best_val_macro_f1=0.82,
                    best_state=None,
                )
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
                from senseframe.engine.runner.pipeline.stages.train import DannTrainResult
                return DannTrainResult(
                    best_score=0.85123456789, best_epoch=2,
                    best_val_loss=0.6123456789, best_val_macro_f1=0.82123456789,
                    best_state=None,
                )
            mock_loop.side_effect = fake_loop
            train_module.stage_train(ctx)

        # final_eval 数值应被 round 到 6 位
        assert round(ctx.final_eval["val_accuracy"], 6) == ctx.final_eval["val_accuracy"]
        assert round(ctx.final_eval["val_loss"], 6) == ctx.final_eval["val_loss"]
        assert round(ctx.final_eval["val_macro_f1"], 6) == ctx.final_eval["val_macro_f1"]


class TestDannLoopBestMetrics:
    """验证 best_val_acc 初始化修复（I1）：val_acc=0 时 best_epoch 仍被记录。"""

    def test_best_epoch_recorded_when_val_acc_zero(self):
        """val_acc=0 时 best_epoch 仍应被记录（I1 修复：best_val_acc 初始化 -1.0）。

        背景：原 best_val_acc=0.0 初始化与合法值 0.0 冲突，
        val_acc=0 时 0.0 > 0.0 为 False，best_epoch 始终 None。
        用空 val_loader 确定性产生 val_acc=0.0（sum([])/max(0,1)=0/1=0.0），
        避免 RNG 导致的非确定性（与 test_early_stopping_breaks_loop 同策略）。
        """
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        ctx = MagicMock()
        ctx.model = _DummyDannModel()
        ctx.datamodule = MagicMock()
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(2, 10), torch.tensor([0, 1]))
        ]
        # 空 val_loader：val_acc 确定性为 0.0
        ctx.datamodule.val_dataloader.return_value = []
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            _train_dann_loop(ctx, epochs=2, learning_rate=0.01)

        # 验证 best_epoch 被记录（非 None）
        assert ctx.best_epoch is not None, "val_acc=0 时 best_epoch 仍应被记录（best_val_acc=-1.0）"
        assert isinstance(ctx.best_epoch, int)
        assert 1 <= ctx.best_epoch <= 2


class TestDannLoopBestModelAndDataclass:
    """I2 + I3 修复验证：best model 权重保存 + dataclass 返回。"""

    def test_dann_loop_saves_best_model_weights(self):
        """I2 修复：训练结束后 ctx.model 应持有 best epoch 的权重（非末轮）。

        构造 2 epoch 训练：epoch 1 val_acc=1.0（best），epoch 2 val_acc=0.0。
        模型用大 bias 固定预测为 class 0，但 linear 权重仍被 optimizer 更新，
        确保 best_state（epoch 1 权重）!= 末轮权重（epoch 2）。
        循环结束后应加载 best_state 回 model。
        """
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop, DannTrainResult

        class _WeightTrackingModel(nn.Module):
            """预测固定为 class 0（大 bias），但 linear 权重仍被训练更新。"""
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, x, x_csi=None, lambda_=0.0):
                # 缩小 linear 贡献，加大 class 0 bias，确保 argmax 始终为 class 0
                hidden = self.linear(x) * 0.001
                bias = torch.tensor([10.0, 0.0, 0.0], device=x.device)
                logits = hidden + bias
                disc_loss = torch.tensor(0.0, requires_grad=True)
                return logits, disc_loss

        class _PerEpochValLoader:
            """每次 __iter__ 返回不同 epoch 的 val 数据。"""
            def __init__(self, epoch_data_list):
                self.epoch_data_list = epoch_data_list
                self.epoch_idx = 0

            def __iter__(self):
                data = self.epoch_data_list[self.epoch_idx % len(self.epoch_data_list)]
                self.epoch_idx += 1
                return iter([data])

        torch.manual_seed(42)
        model = _WeightTrackingModel()

        ctx = MagicMock()
        ctx.model = model
        ctx.datamodule = MagicMock()
        # 训练数据：随机 x，标签为 class 0（与预测一致，产生非零梯度更新 linear）
        ctx.datamodule.train_dataloader.return_value = [
            (torch.randn(4, 10), torch.tensor([0, 0, 0, 0]))
        ]
        # 验证数据：epoch 1 → 标签 class 0（val_acc=1.0），epoch 2 → 标签 class 1（val_acc=0.0）
        x_val = torch.randn(4, 10)
        ctx.datamodule.val_dataloader.return_value = _PerEpochValLoader([
            (x_val, torch.tensor([0, 0, 0, 0])),  # epoch 1: val_acc=1.0
            (x_val, torch.tensor([1, 1, 1, 1])),  # epoch 2: val_acc=0.0
        ])
        ctx.scene_kwargs = {}
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.training_log = []
        ctx.resolved = {
            "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
            "gradient_clip_val": None, "early_stopping": None,
        }

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            result = _train_dann_loop(ctx, epochs=2, learning_rate=0.01)

        # 验证返回 DannTrainResult
        assert isinstance(result, DannTrainResult)
        # best_epoch 应是 1（val_acc=1.0 > epoch 2 的 0.0）
        assert result.best_epoch == 1
        # best_state 非 None
        assert result.best_state is not None

        # I2 核心断言：model 当前权重 == best_state（已加载回 best epoch 权重）
        current_state = model.state_dict()
        for k in current_state:
            assert torch.equal(current_state[k], result.best_state[k]), \
                f"权重 {k} 未恢复到 best epoch（I2 修复失败）"

    def test_dann_loop_returns_dataclass(self):
        """I3 修复：_train_dann_loop 应返回 DannTrainResult 而非通过 ctx 私有属性传递。

        用 SimpleNamespace（非 MagicMock）以便 hasattr 正确检测私有属性是否存在。
        """
        from types import SimpleNamespace
        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop, DannTrainResult

        ctx = SimpleNamespace(
            model=_DummyDannModel(),
            datamodule=MagicMock(),
            scene_kwargs={},
            lightning_params={"accelerator": "cpu"},
            training_log=[],
            resolved={
                "optimizer": "adamw", "weight_decay": 0.0, "scheduler": None,
                "gradient_clip_val": None, "early_stopping": None,
            },
        )
        ctx.datamodule.train_dataloader.return_value = []
        ctx.datamodule.val_dataloader.return_value = []

        with patch("senseframe.engine.runner.pipeline.stages.train._logger"):
            result = _train_dann_loop(ctx, epochs=1, learning_rate=0.01)

        assert isinstance(result, DannTrainResult)
        assert hasattr(result, "best_score")
        assert hasattr(result, "best_epoch")
        assert hasattr(result, "best_val_loss")
        assert hasattr(result, "best_val_macro_f1")
        assert hasattr(result, "best_state")
        # 验证 ctx 不再有 _dann_best_* 私有属性
        assert not hasattr(ctx, "_dann_best_val_loss"), \
            "ctx 不应再有 _dann_best_val_loss 私有属性"
        assert not hasattr(ctx, "_dann_best_val_macro_f1"), \
            "ctx 不应再有 _dann_best_val_macro_f1 私有属性"
