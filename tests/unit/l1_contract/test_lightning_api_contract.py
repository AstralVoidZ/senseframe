"""L1 契约测试：PyTorch Lightning API 契约。

锚点来源：PyTorch Lightning 官方 API（Callback / Trainer / LightningModule）。
- Lightning Callback: https://lightning.ai/docs/pytorch/stable/common/callbacks.html
  - 继承 pytorch_lightning.Callback
  - 生命周期 hooks: on_fit_start / on_validation_epoch_end / on_train_epoch_end 等
- Lightning Trainer: https://lightning.ai/docs/pytorch/stable/common/trainer.html
  - fit(module, datamodule=..., ckpt_path=...) 签名
  - 属性: sanity_checking / should_stop / current_epoch / callback_metrics / callbacks / max_epochs

验证策略：
- 不 mock Lightning，而是用 FakeTrainer（tests/fakes/fake_trainer.py）验证
  "Trainer 隐式协议"——FakeTrainer 实现的属性正是 Lightning Trainer 官方 API 要求的。
- PSNREarlyStoppingCallback 继承 Callback（Lightning 官方 API 要求）。
"""
from __future__ import annotations

import inspect

import pytest


@pytest.mark.l1_contract
class TestLightningApiContract:
    """验证 PSNREarlyStoppingCallback 与 FakeTrainer 符合 Lightning 官方 API 契约。"""

    # ============================================================
    # Callback 契约
    # ============================================================

    def test_psnr_callback_extends_lightning_callback(self):
        """L1 anchor: PSNREarlyStoppingCallback 继承 pytorch_lightning.Callback，锚点：Lightning 官方 API。

        Lightning 文档: 自定义 Callback 必须继承 pytorch_lightning.Callback 基类，
        否则 Trainer 无法识别和注册该 Callback。
        断言: PSNREarlyStoppingCallback 是 Callback 的子类。
        """
        from senseframe.engine.callbacks.psnr_early_stopping import (
            PSNREarlyStoppingCallback,
        )

        # 兼容 pytorch_lightning 和 lightning 包
        try:
            from pytorch_lightning import Callback
        except ImportError:
            from lightning import Callback

        assert issubclass(PSNREarlyStoppingCallback, Callback), (
            "PSNREarlyStoppingCallback 必须继承 pytorch_lightning.Callback"
            "（Lightning 官方 API 要求）"
        )

    def test_callback_base_class_has_lifecycle_hooks(self):
        """L1 anchor: Callback 基类含 on_fit_start / on_validation_epoch_end 等 hooks。

        锚点：Lightning 官方 API
        (https://lightning.ai/docs/pytorch/stable/common/callbacks.html#hooks)。
        Callback 基类定义了完整的生命周期 hooks，子类可按需 override。
        """
        try:
            from pytorch_lightning import Callback
        except ImportError:
            from lightning import Callback

        # Lightning 官方定义的 Callback hooks（子类可 override）
        _REQUIRED_HOOKS = (
            "on_fit_start",
            "on_fit_end",
            "on_train_start",
            "on_validation_start",
            "on_validation_epoch_end",
            "on_train_epoch_end",
        )
        for hook in _REQUIRED_HOOKS:
            assert hasattr(Callback, hook), (
                f"Lightning Callback 基类必须有 {hook} hook"
                f"（Lightning 官方 API 定义）"
            )

    def test_psnr_callback_overrides_on_validation_epoch_end(self):
        """L1 anchor: PSNREarlyStoppingCallback override on_validation_epoch_end hook。

        锚点：Lightning Callback 生命周期。
        PSNREarlyStoppingCallback 在 validation epoch 结束时计算 PSNR，
        必须实现 on_validation_epoch_end(trainer, pl_module) hook。
        """
        from senseframe.engine.callbacks.psnr_early_stopping import (
            PSNREarlyStoppingCallback,
        )

        assert hasattr(PSNREarlyStoppingCallback, "on_validation_epoch_end"), (
            "PSNREarlyStoppingCallback 必须实现 on_validation_epoch_end hook"
        )
        # 验证是 override 而非仅继承（方法在子类中定义）
        own_methods = PSNREarlyStoppingCallback.__dict__
        assert "on_validation_epoch_end" in own_methods, (
            "PSNREarlyStoppingCallback 必须在自身定义 on_validation_epoch_end"
            "（override 基类 hook）"
        )

    def test_psnr_callback_on_validation_epoch_end_signature(self):
        """L1 anchor: on_validation_epoch_end 签名含 (trainer, pl_module)，锚点：Lightning hook 签名。

        Lightning 文档: on_validation_epoch_end(self, trainer, pl_module) 接收
        trainer 和 pl_module 两个参数。PSNREarlyStoppingCallback 通过 trainer
        访问 sanity_checking / should_stop，通过 pl_module 访问缓存的张量。
        """
        from senseframe.engine.callbacks.psnr_early_stopping import (
            PSNREarlyStoppingCallback,
        )

        sig = inspect.signature(PSNREarlyStoppingCallback.on_validation_epoch_end)
        params = list(sig.parameters.keys())
        # 期望参数: self, trainer, pl_module
        assert "trainer" in params, (
            f"on_validation_epoch_end 必须有 trainer 参数，"
            f"实际参数: {params}"
        )
        assert "pl_module" in params, (
            f"on_validation_epoch_end 必须有 pl_module 参数，"
            f"实际参数: {params}"
        )

    # ============================================================
    # FakeTrainer → Lightning Trainer 隐式协议
    # ============================================================

    def test_fake_trainer_has_sanity_checking_attribute(self):
        """L1 anchor: FakeTrainer.sanity_checking 对应 Lightning Trainer.sanity_checking。

        锚点：Lightning Trainer 官方 API
        (https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-sanity-checking)。
        Trainer.sanity_checking 是 bool 属性，标识当前是否在 sanity check 阶段。
        PSNREarlyStoppingCallback.on_validation_epoch_end 依赖此属性跳过 sanity check。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert hasattr(trainer, "sanity_checking"), (
            "FakeTrainer 必须实现 sanity_checking 属性"
            "（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.sanity_checking, bool), (
            f"FakeTrainer.sanity_checking 必须是 bool，"
            f"实际 {type(trainer.sanity_checking).__name__}"
        )

    def test_fake_trainer_has_should_stop_attribute(self):
        """L1 anchor: FakeTrainer.should_stop 对应 Lightning Trainer.should_stop。

        锚点：Lightning Trainer 官方 API。
        Trainer.should_stop 是 bool 属性，Callback 可设为 True 触发早停。
        PSNREarlyStoppingCallback 在 PSNR 连续无提升时设置 trainer.should_stop = True。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert hasattr(trainer, "should_stop"), (
            "FakeTrainer 必须实现 should_stop 属性"
            "（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.should_stop, bool), (
            f"FakeTrainer.should_stop 必须是 bool，"
            f"实际 {type(trainer.should_stop).__name__}"
        )

    def test_fake_trainer_has_current_epoch_attribute(self):
        """L1 anchor: FakeTrainer.current_epoch 对应 Lightning Trainer.current_epoch。

        锚点：Lightning Trainer 官方 API。
        Trainer.current_epoch 是 int 属性，标识当前训练 epoch（0-based）。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert hasattr(trainer, "current_epoch"), (
            "FakeTrainer 必须实现 current_epoch 属性"
            "（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.current_epoch, int), (
            f"FakeTrainer.current_epoch 必须是 int，"
            f"实际 {type(trainer.current_epoch).__name__}"
        )

    def test_fake_trainer_has_callback_metrics_attribute(self):
        """L1 anchor: FakeTrainer.callback_metrics 对应 Lightning Trainer.callback_metrics。

        锚点：Lightning Trainer 官方 API。
        Trainer.callback_metrics 是 dict 属性，缓存训练过程中的指标（如 val_loss）。
        编排层（Orchestrator）读取此属性获取训练结果。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert hasattr(trainer, "callback_metrics"), (
            "FakeTrainer 必须实现 callback_metrics 属性"
            "（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.callback_metrics, dict), (
            f"FakeTrainer.callback_metrics 必须是 dict，"
            f"实际 {type(trainer.callback_metrics).__name__}"
        )

    def test_fake_trainer_has_callbacks_and_max_epochs_attributes(self):
        """L1 anchor: FakeTrainer.callbacks / max_epochs 对应 Lightning Trainer 属性。

        锚点：Lightning Trainer 官方 API。
        - Trainer.callbacks: Callback 实例列表
        - Trainer.max_epochs: 最大训练 epoch 数
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert hasattr(trainer, "callbacks"), (
            "FakeTrainer 必须实现 callbacks 属性（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.callbacks, list), (
            f"FakeTrainer.callbacks 必须是 list，"
            f"实际 {type(trainer.callbacks).__name__}"
        )
        assert hasattr(trainer, "max_epochs"), (
            "FakeTrainer 必须实现 max_epochs 属性（Lightning Trainer 官方 API）"
        )
        assert isinstance(trainer.max_epochs, int), (
            f"FakeTrainer.max_epochs 必须是 int，"
            f"实际 {type(trainer.max_epochs).__name__}"
        )

    def test_fake_trainer_fit_signature_matches_lightning_trainer(self):
        """L1 anchor: FakeTrainer.fit() 签名兼容 Lightning Trainer.fit()。

        锚点：Lightning Trainer.fit() 官方签名
        (https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit)。
        Trainer.fit(model, datamodule=None, ckpt_path=None) 接收 module + 
        可选 datamodule + 可选 ckpt_path。
        FakeTrainer.fit 必须接受相同的关键字参数，确保 Callback 测试时可替换。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        sig = inspect.signature(FakeTrainer.fit)
        params = sig.parameters

        # Lightning Trainer.fit 接受 module/model 作为第一个位置参数
        has_module_param = any(
            name in ("module", "model")
            for name in params
        )
        assert has_module_param, (
            f"FakeTrainer.fit 必须有 module/model 参数，"
            f"实际参数: {list(params.keys())}"
        )

        # Lightning Trainer.fit 接受 datamodule 关键字参数
        assert "datamodule" in params, (
            f"FakeTrainer.fit 必须有 datamodule 参数（Lightning Trainer.fit 签名），"
            f"实际参数: {list(params.keys())}"
        )

        # Lightning Trainer.fit 接受 ckpt_path 关键字参数
        assert "ckpt_path" in params, (
            f"FakeTrainer.fit 必须有 ckpt_path 参数（Lightning Trainer.fit 签名），"
            f"实际参数: {list(params.keys())}"
        )

    def test_fake_trainer_has_tune_validate_test_methods(self):
        """L1 anchor: FakeTrainer 实现 tune/validate/test 方法，锚点：Lightning Trainer API。

        锚点：Lightning Trainer 官方 API
        (https://lightning.ai/docs/pytorch/stable/common/trainer.html)。
        - Trainer.tune(): LR 标定
        - Trainer.validate(): 独立验证
        - Trainer.test(): 独立测试
        FakeTrainer 实现这些方法，确保编排测试无需真实 Lightning。
        """
        from tests.fakes.fake_trainer import FakeTrainer

        trainer = FakeTrainer()
        assert callable(getattr(trainer, "tune", None)), (
            "FakeTrainer 必须实现 tune() 方法（Lightning Trainer 官方 API）"
        )
        assert callable(getattr(trainer, "validate", None)), (
            "FakeTrainer 必须实现 validate() 方法（Lightning Trainer 官方 API）"
        )
        assert callable(getattr(trainer, "test", None)), (
            "FakeTrainer 必须实现 test() 方法（Lightning Trainer 官方 API）"
        )


