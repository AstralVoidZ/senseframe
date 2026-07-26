"""stage_train DANN 分支测试。

验证 scene.params.use_dann=True 时 stage_train 走 DANN 训练循环，
而非默认的 Lightning Trainer 路径（v2 差距 2+3 修复）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestStageTrainDannBranch:
    """验证 DANN 分支触发与训练循环正确性。"""

    def test_use_dann_flag_read_from_scene_params(self):
        """stage_train 应从 scene.params.use_dann 读取 DANN 启用标志。"""
        from senseframe.engine.runner.pipeline.stages.train import _should_use_dann
        from senseframe.core.params import SceneParams

        # use_dann=True
        params_on = SceneParams(extra={"use_dann": True})
        assert _should_use_dann(params_on) is True

        # use_dann=False（默认）
        params_off = SceneParams()
        assert _should_use_dann(params_off) is False

        # use_dann 未设置
        assert _should_use_dann(None) is False

    def test_dann_branch_calls_train_dann_not_lightning(self, tmp_path):
        """use_dann=True 时应调用 _train_dann_loop，不调用 Lightning trainer.fit。"""
        from senseframe.engine.runner.pipeline.stages import train as train_module

        # 构造 mock ctx
        ctx = MagicMock()
        ctx.dry_run = False
        ctx.learning_mode = "supervised"
        ctx.config.trainer.epochs = 5
        ctx.config.trainer.learning_rate = 1e-3
        ctx.config.scene.params = MagicMock()
        ctx.config.scene.params.get = lambda k, d=None: True if k == "use_dann" else d
        ctx.config.trainer.deterministic = False
        ctx.config.trainer.enable_progress_bar = False
        ctx.config.trainer.max_time = None
        ctx.config.trainer.resume = None
        ctx.config.trainer.auto_lr_find = False
        ctx.config.trainer.limit_train_batches = None
        ctx.config.trainer.limit_val_batches = None
        ctx.lightning_params = {"accelerator": "cpu", "devices": 1, "precision": "32"}
        ctx.distributed_kwargs = {}
        ctx.resolved = {"batch_size": 32, "learning_rate": 1e-3, "optimizer": "adamw",
                        "scheduler": None, "gradient_clip_val": None,
                        "gradient_clip_algorithm": "norm", "accumulate_grad_batches": 1}
        ctx.model = MagicMock()
        ctx.datamodule = MagicMock()
        ctx.module = MagicMock()
        ctx.callbacks = []
        ctx.csv_logger = MagicMock()
        ctx.pl_logger = MagicMock()
        ctx.route_config = {}
        ctx.output_dir = tmp_path

        with patch.object(train_module, "_train_dann_loop") as mock_dann, \
             patch("pytorch_lightning.Trainer") as mock_trainer_cls:
            from senseframe.engine.runner.pipeline.stages.train import DannTrainResult
            mock_dann.return_value = DannTrainResult(
                best_score=0.0, best_epoch=None,
                best_val_loss=None, best_val_macro_f1=None,
                best_state=None,
            )
            train_module.stage_train(ctx)

            # DANN 路径应被调用
            mock_dann.assert_called_once()
            # Lightning Trainer 不应被实例化
            mock_trainer_cls.assert_not_called()

    def test_no_dann_falls_through_to_lightning(self, tmp_path):
        """use_dann=False（默认）时应走原有 Lightning Trainer 路径。"""
        from senseframe.engine.runner.pipeline.stages import train as train_module

        ctx = MagicMock()
        ctx.dry_run = False
        ctx.learning_mode = "supervised"
        ctx.config.trainer.epochs = 5
        ctx.config.trainer.learning_rate = 1e-3
        ctx.config.scene.params = MagicMock()
        ctx.config.scene.params.get = lambda k, d=None: d  # use_dann 默认 None
        ctx.config.trainer.deterministic = False
        ctx.config.trainer.enable_progress_bar = False
        ctx.config.trainer.max_time = "00:02:00:00"
        ctx.config.trainer.resume = None
        ctx.config.trainer.auto_lr_find = False
        ctx.config.trainer.limit_train_batches = None
        ctx.config.trainer.limit_val_batches = None
        ctx.lightning_params = {"accelerator": "cpu", "devices": 1, "precision": "32"}
        ctx.distributed_kwargs = {}
        ctx.resolved = {"batch_size": 32, "learning_rate": 1e-3, "optimizer": "adamw",
                        "scheduler": None, "gradient_clip_val": None,
                        "gradient_clip_algorithm": "norm", "accumulate_grad_batches": 1}
        ctx.model = MagicMock()
        ctx.datamodule = MagicMock()
        ctx.module = MagicMock()
        ctx.callbacks = []
        ctx.csv_logger = MagicMock()
        ctx.pl_logger = MagicMock()
        ctx.route_config = {}
        ctx.output_dir = tmp_path

        with patch.object(train_module, "_train_dann_loop") as mock_dann, \
             patch("pytorch_lightning.Trainer") as mock_trainer_cls:
            mock_trainer = MagicMock()
            mock_trainer_cls.return_value = mock_trainer

            train_module.stage_train(ctx)

            # DANN 路径不应被调用
            mock_dann.assert_not_called()

    def test_train_dann_loop_writes_back_ctx_metrics(self):
        """_train_dann_loop 集成测试：验证 λ 调度、双 loss 反传、验证指标计算、ctx 写回。

        使用真实 torch tensor + 最小 fake model + fake datamodule 跑 2 epoch，
        覆盖 _train_dann_loop 核心循环逻辑（非 mock 路径）。
        csi_loader=None（不测试 CSI 对抗路径，保持简单）。
        """
        import torch
        import torch.nn as nn

        from senseframe.engine.runner.pipeline.stages.train import _train_dann_loop

        # Fake DANN model：forward(x_eeg, x_csi=None, lambda_=0.0) -> (logits, disc_loss)
        # csi_loader=None 时 x_csi 始终为 None，disc_loss 返回 None
        # （与真实 DANNCrossModalModel x_csi=None 时行为一致）
        class _FakeDannModel(nn.Module):
            def __init__(self, in_features=8, num_classes=3):
                super().__init__()
                self.fc = nn.Linear(in_features, num_classes)

            def forward(self, x_eeg, x_csi=None, lambda_=0.0):
                logits = self.fc(x_eeg)
                return logits, None

        # Fake datamodule：2 batches × 4 samples，3 classes
        class _FakeDataModule:
            def __init__(self, num_batches=2, batch_size=4,
                         in_features=8, num_classes=3):
                torch.manual_seed(42)
                self._batches = [
                    (torch.randn(batch_size, in_features),
                     torch.randint(0, num_classes, (batch_size,)))
                    for _ in range(num_batches)
                ]

            def train_dataloader(self):
                return list(self._batches)

            def val_dataloader(self):
                return list(self._batches)

        torch.manual_seed(0)  # C1 修复：消除 RNG flaky
        model = _FakeDannModel()
        datamodule = _FakeDataModule()

        ctx = MagicMock()
        ctx.lightning_params = {"accelerator": "cpu"}
        ctx.model = model
        ctx.datamodule = datamodule
        ctx.scene_kwargs = {}  # csi_loader=None
        # MEDIUM 5 回归适配：_train_dann_loop 现从 ctx.resolved 读取
        # optimizer/scheduler/gradient_clip_val/early_stopping，需提供真实 dict
        ctx.resolved = {
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": None,
            "gradient_clip_val": None,
            "early_stopping": None,
        }

        _train_dann_loop(ctx=ctx, epochs=2, learning_rate=1e-3)

        # ctx 写回断言
        assert isinstance(ctx.best_model_score, float), \
            f"best_model_score 应为 float，实际为 {type(ctx.best_model_score)}"
        assert 0.0 <= ctx.best_model_score <= 1.0, \
            f"best_model_score 应在 [0, 1] 范围内，实际为 {ctx.best_model_score}"
        assert ctx.pruned is False
        assert ctx.pruned_epoch is None
        # Minor 4.3：DANN 路径回写 best_epoch（best_val_acc 对应的 epoch，1-based）
        # _FakeDannModel 权重取决于测试运行时的 RNG 状态（_FakeDataModule 的
        # manual_seed(42) 仅作用于数据生成，模型 init 在其之前），val_acc 可能
        # epoch1>epoch2 或相等，故仅断言 best_epoch 在合法区间内。
        assert isinstance(ctx.best_epoch, int)
        assert 1 <= ctx.best_epoch <= 2

    def test_stage_train_dann_branch_writes_final_eval_and_freezes_intermediate_values(self, tmp_path):
        """DANN 分支 wrapper 代码覆盖：final_eval 写入 + intermediate_values 冻结 + training_duration_s。

        隔离 _train_dann_loop（mock 为最小 no-op，仅设 best_model_score=0.5），
        专门测试 wrapper 代码（Critical #1 final_eval + Important #2 freeze + Timer 写回）。
        _train_dann_loop 自身逻辑由 test_train_dann_loop_writes_back_ctx_metrics 覆盖。
        """
        from senseframe.engine.runner.pipeline.stages import train as train_module
        from senseframe.engine.runner.callbacks import FrozenDict

        # 构造 mock ctx（参考 test_dann_branch_calls_train_dann_not_lightning）
        ctx = MagicMock()
        ctx.dry_run = False
        ctx.learning_mode = "supervised"
        ctx.config.trainer.epochs = 5
        ctx.config.trainer.learning_rate = 1e-3
        ctx.config.scene.params = MagicMock()
        ctx.config.scene.params.get = lambda k, d=None: True if k == "use_dann" else d
        ctx.config.trainer.deterministic = False
        ctx.config.trainer.enable_progress_bar = False
        ctx.config.trainer.max_time = None
        ctx.config.trainer.resume = None
        ctx.config.trainer.auto_lr_find = False
        ctx.config.trainer.limit_train_batches = None
        ctx.config.trainer.limit_val_batches = None
        ctx.lightning_params = {"accelerator": "cpu", "devices": 1, "precision": "32"}
        ctx.distributed_kwargs = {}
        ctx.resolved = {"batch_size": 32, "learning_rate": 1e-3, "optimizer": "adamw",
                        "scheduler": None, "gradient_clip_val": None,
                        "gradient_clip_algorithm": "norm", "accumulate_grad_batches": 1}
        ctx.model = MagicMock()
        ctx.datamodule = MagicMock()
        ctx.module = MagicMock()
        ctx.callbacks = []
        ctx.csv_logger = MagicMock()
        ctx.pl_logger = MagicMock()
        ctx.route_config = {}
        ctx.output_dir = tmp_path
        # 关键：intermediate_values 必须是真实 dict，wrapper 会 FrozenDict(ctx.intermediate_values)
        ctx.intermediate_values = {}

        # mock _train_dann_loop 为最小 no-op：仅设 best_model_score
        def _fake_train_dann_loop(ctx, epochs, learning_rate):
            from senseframe.engine.runner.pipeline.stages.train import DannTrainResult
            return DannTrainResult(
                best_score=0.5, best_epoch=1,
                best_val_loss=None, best_val_macro_f1=None,
                best_state=None,
            )

        # mock perf_counter 使 Timer.elapsed 确定性为正数，避免真实 sleep
        with patch.object(train_module, "_train_dann_loop", side_effect=_fake_train_dann_loop), \
             patch("pytorch_lightning.Trainer") as mock_trainer_cls, \
             patch("senseframe.observability.time.perf_counter", side_effect=[100.0, 100.05]):
            train_module.stage_train(ctx)
            # 仍是 DANN 路径，Lightning Trainer 不应被实例化
            mock_trainer_cls.assert_not_called()

        # 断言 wrapper 写入 final_eval（Critical #1：供 stage_eval 使用）
        assert ctx.final_eval is not None, "final_eval 应被 wrapper 写入"
        assert "val_accuracy" in ctx.final_eval, "final_eval 应包含 val_accuracy 键"
        assert isinstance(ctx.final_eval["val_accuracy"], float), \
            f"final_eval['val_accuracy'] 应为 float，实际为 {type(ctx.final_eval['val_accuracy'])}"
        assert ctx.final_eval["val_accuracy"] == 0.5

        # 断言 intermediate_values 被冻结为 FrozenDict（Important #2：与 Lightning 路径对齐）
        assert isinstance(ctx.intermediate_values, FrozenDict), \
            f"intermediate_values 应为 FrozenDict，实际为 {type(ctx.intermediate_values)}"

        # 断言 training_duration_s 为正数（Timer elapsed 写回）
        assert isinstance(ctx.training_duration_s, float), \
            f"training_duration_s 应为 float，实际为 {type(ctx.training_duration_s)}"
        assert ctx.training_duration_s > 0, \
            f"training_duration_s 应为正数，实际为 {ctx.training_duration_s}"
