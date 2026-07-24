"""Fake 替身库自身的契约测试。

化用 ppsspp-dfx FakeTransport 模式：每个 Fake 必须有契约测试，
验证其满足被替身对象的隐式协议。Fake 行为偏离真实实现时即发现。

契约来源：
- FakeTrainer ↔ pytorch_lightning.Trainer 官方 API
- FakeLightningModule ↔ pytorch_lightning.LightningModule 官方 API
- FakeSampler ↔ senseframe.search_protocol.Sampler Protocol
- FakePruner ↔ senseframe.search_protocol.Pruner Protocol
- StubDannModel ↔ senseframe.scenes.wifi_csi.dann.DANNCrossModalModel
- StubMaeModel ↔ senseframe.scenes.wifi_csi.foundation_model.CSIFoundationModel
"""
from __future__ import annotations

import inspect

import pytest
import torch

from tests.fakes.fake_trainer import FakeTrainer
from tests.fakes.fake_lightning_module import FakeLightningModule
from tests.fakes.fake_sampler import FakeSampler
from tests.fakes.fake_pruner import FakePruner
from tests.fakes.stub_models import StubDannModel, StubMaeModel


# ============================================================
# FakeTrainer 契约测试
# ============================================================

class TestFakeTrainerContract:
    """FakeTrainer 必须满足 Lightning Trainer 的隐式协议。

    协议来源：pytorch_lightning.Trainer 官方 API + SenseFrame 编排层实际访问的属性。
    """

    def test_sanity_checking_attribute_exists_and_is_bool(self):
        """Trainer.sanity_checking 是 bool 属性（PSNR Callback I9 修复依赖）。"""
        trainer = FakeTrainer(sanity_checking=False)
        assert isinstance(trainer.sanity_checking, bool)
        trainer = FakeTrainer(sanity_checking=True)
        assert trainer.sanity_checking is True

    def test_should_stop_writable(self):
        """Trainer.should_stop 可被 Callback 设置为 True（早停依赖）。"""
        trainer = FakeTrainer()
        assert trainer.should_stop is False
        trainer.should_stop = True
        assert trainer.should_stop is True

    def test_current_epoch_is_int(self):
        """Trainer.current_epoch 是 int（Orchestrator 读取 epoch 依赖）。"""
        trainer = FakeTrainer(current_epoch=5)
        assert isinstance(trainer.current_epoch, int)
        assert trainer.current_epoch == 5

    def test_callback_metrics_is_dict(self):
        """Trainer.callback_metrics 是 dict（Orchestrator 读取 val_loss 依赖）。"""
        trainer = FakeTrainer(callback_metrics={"val_loss": 0.5})
        assert isinstance(trainer.callback_metrics, dict)
        assert trainer.callback_metrics["val_loss"] == 0.5

    def test_callbacks_list_appendable(self):
        """Trainer.callbacks 是 list，可 append Callback。"""
        trainer = FakeTrainer()
        trainer.callbacks.append("some_callback")
        assert len(trainer.callbacks) == 1

    def test_fit_method_exists(self):
        """Trainer.fit() 方法存在且可调用。"""
        trainer = FakeTrainer()
        trainer.fit(module=None, datamodule=None)
        assert trainer._fit_called is True

    def test_tune_returns_lr_find_dict(self):
        """Trainer.tune() 返回 {"lr_find": {"suggestion": float}} 格式。"""
        trainer = FakeTrainer()
        result = trainer.tune(module=None)
        assert "lr_find" in result
        assert "suggestion" in result["lr_find"]
        assert isinstance(result["lr_find"]["suggestion"], float)

    def test_teardown_method_exists(self):
        """Trainer._teardown() 方法存在（_fit_with_oom_fallback 依赖）。"""
        trainer = FakeTrainer()
        trainer._teardown()
        assert trainer._teardown_called is True


# ============================================================
# FakeLightningModule 契约测试
# ============================================================

class TestFakeLightningModuleContract:
    """FakeLightningModule 必须满足 LightningModule 的隐式协议。

    协议来源：pytorch_lightning.LightningModule 官方 API + PSNR Callback 实际访问的属性。
    """

    def test_log_method_records_calls(self):
        """Module.log(name, value) 记录指标（PSNR Callback L93 依赖）。"""
        module = FakeLightningModule()
        module.log("val_psnr", 12.5, prog_bar=True)
        assert module._logs["val_psnr"] == 12.5

    def test_psnr_reconstruction_attribute_exists(self):
        """Module._psnr_reconstruction 属性存在（PSNR Callback L87 getattr 依赖）。"""
        module = FakeLightningModule()
        assert module._psnr_reconstruction is None  # 默认 None

    def test_psnr_target_attribute_exists(self):
        """Module._psnr_target 属性存在（PSNR Callback L88 getattr 依赖）。"""
        module = FakeLightningModule()
        assert module._psnr_target is None

    def test_state_dict_and_load_state_dict_roundtrip(self):
        """Module.state_dict()/load_state_dict() 往返一致（_train_dann_loop 依赖）。"""
        module = FakeLightningModule()
        module._state_dict = {"weight": torch.tensor([1.0, 2.0])}
        state = module.state_dict()
        module2 = FakeLightningModule()
        module2.load_state_dict(state)
        assert torch.equal(module2._state_dict["weight"], torch.tensor([1.0, 2.0]))

    def test_train_eval_mode_switch(self):
        """Module.train()/eval() 切换模式（stage_train 依赖）。"""
        module = FakeLightningModule()
        module.eval()
        assert module._training_mode is False
        module.train()
        assert module._training_mode is True

    def test_training_log_is_list(self):
        """Module.training_log 是 list（stage_train 写入训练日志依赖）。"""
        module = FakeLightningModule()
        assert isinstance(module.training_log, list)
        module.training_log.append({"epoch": 1, "loss": 0.5})
        assert len(module.training_log) == 1

    def test_trainer_attribute_exists_and_default_none(self):
        """Module.trainer 属性存在且默认 None（engine/module.py 访问 self.trainer.sanity_checking）。"""
        module = FakeLightningModule()
        assert hasattr(module, "trainer")
        assert module.trainer is None  # 默认 None，测试可按需设置为 FakeTrainer

    def test_parameters_returns_non_empty_iterable(self):
        """Module.parameters() 返回非空可迭代对象（torch.optim.Adam 构造依赖）。"""
        module = FakeLightningModule()
        params = list(module.parameters())
        assert len(params) > 0, (
            "parameters() 返回空，torch.optim.Adam(module.parameters()) 会抛 ValueError"
        )
        # 验证返回的参数可被 optimizer 消费（不抛异常）
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        assert optimizer is not None


# ============================================================
# FakeSampler 契约测试
# ============================================================

class TestFakeSamplerContract:
    """FakeSampler 必须满足 SP Sampler Protocol。

    协议来源：senseframe.search_protocol.Sampler（@runtime_checkable Protocol）。
    注意：是 SenseFrame 自有 Protocol，非 Optuna BaseSampler。
    """

    def test_has_name_class_attribute(self):
        """Sampler.name 类属性存在（Protocol L166 要求）。"""
        assert hasattr(FakeSampler, "name")
        assert isinstance(FakeSampler.name, str)

    def test_init_accepts_seed(self):
        """Sampler.__init__ 接受 seed 参数（StudyManager 通过 sampler_cls(seed=...) 实例化）。"""
        sampler = FakeSampler(seed=42)
        assert sampler.seed == 42

    def test_sample_method_signature(self):
        """Sampler.sample(search_space, history) -> Dict 签名（Protocol L168 要求）。"""
        sig = inspect.signature(FakeSampler.sample)
        params = list(sig.parameters.keys())
        assert "search_space" in params
        assert "history" in params

    def test_sample_returns_dict(self):
        """Sampler.sample() 返回 dict（StudyManager.ask 依赖）。"""
        from types import SimpleNamespace
        # 构造最小 SearchSpace
        spec = SimpleNamespace(low=0, high=10)
        search_space = SimpleNamespace(params={"lr": spec})
        sampler = FakeSampler()
        result = sampler.sample(search_space, history=[])
        assert isinstance(result, dict)
        assert result["lr"] == 0  # 返回下界

    def test_isinstance_sampler_protocol(self):
        """FakeSampler 通过 isinstance(x, Sampler) runtime_checkable 检查。"""
        from senseframe.search_protocol import Sampler
        assert isinstance(FakeSampler(), Sampler)

    def test_warm_start_method_exists(self):
        """Sampler.warm_start() 方法存在（Protocol L175 声明，元学习用）。"""
        sampler = FakeSampler()
        sampler.warm_start([])
        assert sampler._warm_started is True


# ============================================================
# FakePruner 契约测试
# ============================================================

class TestFakePrunerContract:
    """FakePruner 必须满足 SP Pruner Protocol。

    协议来源：senseframe.search_protocol.Pruner（@runtime_checkable Protocol）。
    注意：是 SenseFrame 自有 Protocol，非 Optuna BasePruner。
    """

    def test_has_name_class_attribute(self):
        """Pruner.name 类属性存在（Protocol L323 要求）。"""
        assert hasattr(FakePruner, "name")
        assert isinstance(FakePruner.name, str)

    def test_should_prune_method_signature(self):
        """Pruner.should_prune(trial_id, intermediate_values, rung) -> bool 签名（Protocol L325 要求）。"""
        sig = inspect.signature(FakePruner.should_prune)
        params = list(sig.parameters.keys())
        assert "trial_id" in params
        assert "intermediate_values" in params
        assert "rung" in params

    def test_should_prune_returns_bool(self):
        """Pruner.should_prune() 返回 bool（IntermediateMetricLogger 依赖）。"""
        pruner = FakePruner(should_prune=False)
        result = pruner.should_prune("trial_1", {1: 0.5}, rung=1)
        assert isinstance(result, bool)
        assert result is False

    def test_should_prune_records_call(self):
        """Pruner.should_prune() 记录调用参数（测试断言依赖）。"""
        pruner = FakePruner(should_prune=True)
        pruner.should_prune("trial_1", {1: 0.5}, rung=2)
        assert pruner.call_count == 1
        assert pruner.last_trial_id == "trial_1"
        assert pruner.last_rung == 2

    def test_isinstance_pruner_protocol(self):
        """FakePruner 通过 isinstance(x, Pruner) runtime_checkable 检查。"""
        from senseframe.search_protocol import Pruner
        assert isinstance(FakePruner(), Pruner)


# ============================================================
# StubDannModel 契约测试
# ============================================================

class TestStubDannModelContract:
    """StubDannModel 必须满足 DANNCrossModalModel 的前向协议。

    协议来源：senseframe.scenes.wifi_csi.dann.DANNCrossModalModel.forward。
    """

    def test_forward_returns_tuple_of_two(self):
        """forward 返回 (logits, disc_loss) 二元组。"""
        model = StubDannModel(in_features=10, num_classes=7)
        model.train()  # 训练模式
        x_eeg = torch.randn(2, 10)
        x_csi = torch.randn(2, 10)
        result = model(x_eeg, x_csi, lambda_=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_logits_shape(self):
        """logits 形状为 (B, num_classes)。"""
        model = StubDannModel(in_features=10, num_classes=7)
        model.train()
        x_eeg = torch.randn(4, 10)
        x_csi = torch.randn(4, 10)
        logits, _ = model(x_eeg, x_csi, lambda_=0.0)
        assert logits.shape == (4, 7)

    def test_forward_disc_loss_is_scalar_when_training_with_csi(self):
        """训练模式 + x_csi 提供时，disc_loss 是可微标量。"""
        model = StubDannModel(in_features=10, num_classes=7)
        model.train()
        x_eeg = torch.randn(2, 10)
        x_csi = torch.randn(2, 10)
        _, disc_loss = model(x_eeg, x_csi, lambda_=0.5)
        assert disc_loss is not None
        assert disc_loss.requires_grad

    def test_forward_disc_loss_none_when_eval(self):
        """eval 模式时 disc_loss 为 None。"""
        model = StubDannModel(in_features=10, num_classes=7)
        model.eval()
        x_eeg = torch.randn(2, 10)
        logits, disc_loss = model(x_eeg, x_csi=None, lambda_=0.0)
        assert disc_loss is None

    def test_has_decoder_params(self):
        """Stub 有 decoder 参数（decoder freeze 测试依赖）。"""
        model = StubDannModel(in_features=10, num_classes=7)
        decoder_params = [n for n, _ in model.named_parameters() if "decoder" in n]
        assert len(decoder_params) > 0

    def test_has_get_inner_backbone_method(self):
        """_get_inner_backbone() 方法存在（PEFTModel 穿透测试依赖）。"""
        model = StubDannModel(in_features=10, num_classes=7)
        backbone = model._get_inner_backbone()
        assert backbone is model


# ============================================================
# StubMaeModel 契约测试
# ============================================================

class TestStubMaeModelContract:
    """StubMaeModel 必须满足 CSIFoundationModel.mae_reconstruct 协议。

    协议来源：senseframe.scenes.wifi_csi.foundation_model.CSIFoundationModel.mae_reconstruct。
    """

    def test_mae_reconstruct_returns_tuple_of_three(self):
        """mae_reconstruct 返回 (recon, target, mask) 三元组。"""
        model = StubMaeModel(mask_ratio=0.75, n_patches=10, patch_dim=8)
        x = torch.randn(2, 3, 32)
        recon, target, mask = model.mae_reconstruct(x, 0.75)
        assert isinstance(recon, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_mask_ratio_attribute_exists(self):
        """_mask_ratio 属性存在（SelfSupervisedModule.validation_step 读取）。"""
        model = StubMaeModel(mask_ratio=0.75)
        assert hasattr(model, "_mask_ratio")
        assert model._mask_ratio == 0.75

    def test_mask_values_are_zero_or_one(self):
        """mask 值为 0 或 1（1=masked, 0=visible）。"""
        model = StubMaeModel(mask_ratio=0.75, n_patches=10, patch_dim=8)
        x = torch.randn(2, 3, 32)
        _, _, mask = model.mae_reconstruct(x, 0.75)
        unique_vals = set(mask.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_mask_ratio_proportion(self):
        """mask 中 1 的比例约等于 mask_ratio（MAE 论文语义）。"""
        model = StubMaeModel(mask_ratio=0.75, n_patches=10, patch_dim=8)
        x = torch.randn(2, 3, 32)
        _, _, mask = model.mae_reconstruct(x, 0.75)
        masked_ratio = mask.mean().item()
        assert abs(masked_ratio - 0.75) < 0.15  # 允许小数误差

    def test_recon_supports_gradient(self):
        """recon 支持梯度流（decoder freeze / 梯度反转测试依赖）。"""
        model = StubMaeModel(mask_ratio=0.75, n_patches=10, patch_dim=8)
        x = torch.randn(2, 3, 32)
        recon, _, _ = model.mae_reconstruct(x, 0.75)
        # recon 通过 self.proj(target) 产生，应有 grad_fn
        assert recon.requires_grad

    def test_forward_returns_tuple_of_two(self):
        """forward 返回 (logits, aux_logits) 二元组（ce_criterion 消费）。"""
        model = StubMaeModel(mask_ratio=0.75, n_patches=10, patch_dim=8)
        x1 = torch.randn(2, 3, 32)
        logits, aux = model(x1)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(aux, torch.Tensor)
