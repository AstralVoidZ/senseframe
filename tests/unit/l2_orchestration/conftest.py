"""L2 conftest：编排 spec 契约测试 fixtures。

L2 测试锚定项目内 RFC/设计文档，使用 Fake 替身（FakeTrainer / FakeSampler）
替代 MagicMock，验证编排逻辑符合 spec。
"""
from __future__ import annotations

import pytest

from tests.fakes.fake_trainer import FakeTrainer
from tests.fakes.fake_lightning_module import FakeLightningModule
from tests.fakes.fake_sampler import FakeSampler
from tests.fakes.fake_pruner import FakePruner


@pytest.fixture
def fake_trainer() -> FakeTrainer:
    """FakeTrainer：Lightning Trainer 的测试替身。

    实现 Trainer 隐式协议（sanity_checking / should_stop / current_epoch /
    callback_metrics / callbacks），让编排测试无需真实 Trainer.fit() 开销。
    """
    return FakeTrainer()


@pytest.fixture
def fake_trainer_sanity() -> FakeTrainer:
    """FakeTrainer 处于 sanity_checking 阶段。"""
    return FakeTrainer(sanity_checking=True)


@pytest.fixture
def fake_module() -> FakeLightningModule:
    """FakeLightningModule：LightningModule 的测试替身。"""
    return FakeLightningModule()


@pytest.fixture
def fake_sampler() -> FakeSampler:
    """FakeSampler：SP Sampler Protocol 的测试替身。"""
    return FakeSampler(seed=42)


@pytest.fixture
def fake_pruner() -> FakePruner:
    """FakePruner：SP Pruner Protocol 的测试替身（默认不剪枝）。"""
    return FakePruner(should_prune=False)


@pytest.fixture
def fake_pruner_always() -> FakePruner:
    """FakePruner 总是剪枝（用于早停编排测试）。"""
    return FakePruner(should_prune=True)
