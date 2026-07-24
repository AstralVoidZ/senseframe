"""L2 conftest：编排 spec 契约测试 fixtures（脚手架）。

L2 测试锚定项目内 RFC/设计文档，使用 Fake 替身（FakeTrainer / FakeSampler）
替代 MagicMock，验证编排逻辑符合 spec。

当前 L2 测试目录尚未实装测试文件。未来添加测试时，直接从 tests/fakes/
导入所需 Fake 类实例化即可（如 `from tests.fakes.fake_trainer import FakeTrainer`），
无需通过 fixture 注入。
"""
from __future__ import annotations
