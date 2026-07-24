"""PSNR early stopping Lightning Callback（框架级）。

v2 差距 3 修复：从 scripts/p0_pretrain_with_psnr.py 迁移为框架级 Callback，
让自监督预训练可通过 scene.params.pretrain_early_stop_metric="psnr" 启用。

设计：
- compute_psnr：纯函数，迁移自 scripts/p0_pretrain_with_psnr.py:compute_psnr
- PSNREarlyStoppingCallback：Lightning Callback，on_validation_epoch_end 计算 PSNR
- best_psnr：跟踪历史最优 PSNR（供 Agent 评估预训练质量）
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from pytorch_lightning import Callback
except ImportError:
    from lightning import Callback


def compute_psnr(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 5.0,
) -> float:
    """计算 PSNR（峰值信噪比）。

    迁移自 scripts/p0_pretrain_with_psnr.py:compute_psnr。

    Args:
        reconstructed: 重建张量
        target: 目标张量
        max_value: 信号最大值（CSI 归一化后 5σ 边界，默认 5.0）

    Returns:
        PSNR 值（dB），完美重建返回 100.0
    """
    mse = torch.mean((reconstructed - target) ** 2)
    if mse.item() < 1e-10:
        return 100.0
    return float(10 * torch.log10(max_value ** 2 / mse))


class PSNREarlyStoppingCallback(Callback):
    """基于 PSNR 的早停 Callback。

    在自监督预训练（MAE 等重建任务）中，PSNR 是比 loss 更直观的质量指标。
    当 PSNR 连续 patience 个 epoch 无提升（< min_delta）时触发停止。

    Attributes:
        patience: 容忍 epoch 数
        min_delta: PSNR 提升阈值
        best_psnr: 历史最优 PSNR（供 Agent 评估）
        counter: 当前无提升计数
        should_stop: 是否触发停止
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.1):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_psnr: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _update_psnr(self, psnr: float) -> None:
        """更新 PSNR 状态机（供测试直接调用 + on_validation_epoch_end 内部使用）。"""
        if self.best_psnr is None or psnr > self.best_psnr + self.min_delta:
            self.best_psnr = psnr
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """validation 结束后计算 PSNR 并更新状态机。

        从 pl_module 缓存的 reconstruction_batch / target_batch 取数据。
        若模块未缓存，跳过（no-op）。
        """
        # I9 修复：sanity_check 阶段不污染状态机（与其他 on_validation_epoch_end 对齐）
        if trainer.sanity_checking:
            return
        recon = getattr(pl_module, "_psnr_reconstruction", None)
        target = getattr(pl_module, "_psnr_target", None)
        if recon is None or target is None:
            return
        psnr = compute_psnr(recon, target)
        self._update_psnr(psnr)
        pl_module.log("val_psnr", psnr, prog_bar=True)
        if self.should_stop:
            trainer.should_stop = True
