"""Fake 替身库：满足隐式协议的测试替身。

设计原则（化用 ppsspp-dfx FakeTransport 模式）：
- Fake 不是 Mock：实现协议语义，有状态，可验证端到端链路
- 每个 Fake 有自身契约测试（test_fakes.py），确保行为与真实实现对齐
- 命名规范：Fake*（有状态，实现协议）/ Stub*（无状态，返回固定值）

协议来源：
- FakeTrainer: pytorch_lightning.Trainer 官方 API
- FakeLightningModule: pytorch_lightning.LightningModule 官方 API
- FakeSampler: senseframe.search_protocol.Sampler Protocol（非 Optuna）
- FakePruner: senseframe.search_protocol.Pruner Protocol（非 Optuna）
- StubDannModel: senseframe.scenes.wifi_csi.dann.DANNCrossModalModel
- StubMaeModel: senseframe.scenes.wifi_csi.foundation_model.CSIFoundationModel
"""
from tests.fakes.fake_trainer import FakeTrainer
from tests.fakes.fake_lightning_module import FakeLightningModule
from tests.fakes.fake_sampler import FakeSampler
from tests.fakes.fake_pruner import FakePruner
from tests.fakes.stub_models import StubDannModel, StubMaeModel

__all__ = [
    "FakeTrainer",
    "FakeLightningModule",
    "FakeSampler",
    "FakePruner",
    "StubDannModel",
    "StubMaeModel",
]
