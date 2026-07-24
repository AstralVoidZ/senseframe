"""FakeTrainer：Lightning Trainer 的测试替身。

实现 Trainer 的隐式协议，让 Callback / 编排测试无需真实 Trainer.fit() 开销。
协议来源：pytorch_lightning.Trainer 官方 API + SenseFrame 编排层实际访问的属性。

关键属性（测试与 Callback 实际依赖）：
- sanity_checking: bool — sanity check 阶段标志（PSNR Callback I9 修复依赖）
- should_stop: bool — 早停标志（可被 Callback 覆写为 True）
- current_epoch: int — 当前 epoch（0-based）
- callback_metrics: Dict[str, Any] — 指标缓存（Orchestrator 读取 val_loss 等）
- callbacks: list — 回调列表
- max_epochs: int — 最大 epoch 数

关键方法：
- fit(module, datamodule=..., ckpt_path=...) — 训练入口
- tune(module, datamodule=...) — LR 标定，返回 {"lr_find": {"suggestion": float}}
- validate() / test() — 独立验证/测试
- _teardown() — 资源清理（_fit_with_oom_fallback 依赖）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeTrainer:
    """Lightning Trainer 的测试替身。

    实现 Trainer 隐式协议，让编排测试（L2）无需真实 Trainer.fit() 开销。
    所有属性可读可写，方法记录调用次数供断言。

    Attributes:
        sanity_checking: sanity check 阶段标志（默认 False）
        should_stop: 早停标志（默认 False，可被 Callback 覆写）
        current_epoch: 当前 epoch（默认 0）
        callback_metrics: 指标缓存（默认空 dict）
        callbacks: 回调列表
        max_epochs: 最大 epoch 数（默认 1）
    """

    def __init__(
        self,
        *,
        sanity_checking: bool = False,
        should_stop: bool = False,
        current_epoch: int = 0,
        callback_metrics: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1,
    ) -> None:
        self.sanity_checking: bool = sanity_checking
        self.should_stop: bool = should_stop
        self.current_epoch: int = current_epoch
        self.callback_metrics: Dict[str, Any] = dict(callback_metrics or {})
        self.callbacks: List[Any] = []
        self.max_epochs: int = max_epochs
        self._fit_called: bool = False
        self._tune_called: bool = False
        self._validate_called: bool = False
        self._test_called: bool = False
        self._teardown_called: bool = False
        self._tune_result: Dict[str, Any] = {"lr_find": {"suggestion": 1e-3}}

    def fit(self, module: Any, datamodule: Any = None, ckpt_path: Optional[str] = None) -> None:
        """训练入口（记录调用，不真正训练）。"""
        self._fit_called = True

    def tune(self, module: Any, datamodule: Any = None) -> Dict[str, Any]:
        """LR 标定（返回预设结果）。"""
        self._tune_called = True
        return self._tune_result

    def validate(self, module: Any = None, datamodule: Any = None) -> List[Dict[str, Any]]:
        """独立验证（返回空列表）。"""
        self._validate_called = True
        return []

    def test(self, module: Any = None, datamodule: Any = None) -> List[Dict[str, Any]]:
        """独立测试（返回空列表）。"""
        self._test_called = True
        return []

    def _teardown(self) -> None:
        """资源清理（_fit_with_oom_fallback 依赖）。"""
        self._teardown_called = True
