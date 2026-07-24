"""FakeLightningModule：LightningModule 的测试替身。

实现 LightningModule 的隐式协议，让 Callback / 编排测试无需真实 Module。
协议来源：pytorch_lightning.LightningModule 官方 API + SenseFrame 编排层实际访问的属性。

关键属性（PSNR Callback + stage_train 实际依赖）：
- log(name, value, **kwargs): 指标记录方法
- _psnr_reconstruction: Optional[torch.Tensor] — PSNR 重建缓存
- _psnr_target: Optional[torch.Tensor] — PSNR 目标缓存
- training_log: List[Dict] — 训练日志（stage_train 写入）
- phase: str — 训练阶段（"self_supervised" / "supervised"）
- learning_rate: float — 学习率（auto_lr_find 覆写）

关键方法：
- state_dict() / load_state_dict() — 权重快照/加载（_train_dann_loop 依赖）
- parameters() — 参数迭代（optimizer 构造依赖）
- train() / eval() — 模式切换
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeLightningModule:
    """LightningModule 的测试替身。

    实现 LightningModule 隐式协议，让编排测试无需真实 nn.Module。
    log 调用记录到 _logs 字典供断言。

    Attributes:
        _psnr_reconstruction: PSNR 重建缓存（默认 None）
        _psnr_target: PSNR 目标缓存（默认 None）
        training_log: 训练日志列表
        phase: 训练阶段（默认 "supervised"）
        learning_rate: 学习率（默认 1e-3）
    """

    def __init__(self) -> None:
        self._psnr_reconstruction: Optional[Any] = None
        self._psnr_target: Optional[Any] = None
        self.training_log: List[Dict[str, Any]] = []
        self.phase: str = "supervised"
        self.learning_rate: float = 1e-3
        self._logs: Dict[str, Any] = {}
        self._training_mode: bool = True
        self._state_dict: Dict[str, Any] = {}
        self._current_epoch_loss: float = 0.0
        self._current_epoch_steps: int = 0
        self._current_val_epoch_loss: float = 0.0
        self._current_val_epoch_steps: int = 0
        self._has_validation_run: bool = False

    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        """指标记录（记录到 _logs 字典）。"""
        self._logs[name] = value

    def state_dict(self) -> Dict[str, Any]:
        """权重快照（返回内部状态副本）。"""
        return dict(self._state_dict)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """权重加载。"""
        self._state_dict = dict(state)

    def parameters(self) -> List[Any]:
        """参数迭代（返回空列表，Stub 模型应继承 nn.Module 提供真实参数）。"""
        return []

    def train(self, mode: bool = True) -> "FakeLightningModule":
        """切换到训练模式。"""
        self._training_mode = True
        return self

    def eval(self) -> "FakeLightningModule":
        """切换到评估模式。"""
        self._training_mode = False
        return self

    def to(self, device: Any) -> "FakeLightningModule":
        """设备迁移（no-op，Fake 不绑定设备）。"""
        return self
