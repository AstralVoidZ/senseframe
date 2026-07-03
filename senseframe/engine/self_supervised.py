"""
自监督学习 LightningModule：两阶段训练（自监督预训练 + 监督微调）。

从原 senseframe/self_supervised.py 移入 engine 层，去除 CSI 特定耦合：
- METRIC_MAP 从 engine.module 导入（不再经过 senseframe.module 别名）
- 逻辑保持不变，仅迁移位置

两阶段训练：
- Phase 1 (self_supervised): EntLoss (KL+EH+HE+KDE)，训练全部参数
- Phase 2 (supervised): CrossEntropyLoss，只训练 classifier 参数
"""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from torchmetrics import ConfusionMatrix

from ..core.metrics import get_metric, has_metric
from ..core.ent_loss import EntLoss


# ============================================================
# EntLoss 已移至 core/ent_loss.py（消除 core→engine 循环依赖）
# ============================================================


# ============================================================
# 高斯噪声增强
# ============================================================
def gaussian_noise(csi: torch.Tensor, epsilon: float, device: torch.device) -> torch.Tensor:
    """对输入添加高斯噪声（使用 device 参数，不硬编码 .cuda()）。"""
    noise = torch.normal(1, 2, size=csi.shape, device=device)
    return csi + epsilon * noise


# ============================================================
# 自监督 LightningModule
# ============================================================
class SelfSupervisedModule(pl.LightningModule):
    """
    自监督学习 LightningModule（两阶段训练）。

    通过 phase 属性控制阶段：
    - phase='self_supervised': EntLoss 训练全部参数
    - phase='supervised': CrossEntropyLoss 只训练 classifier

    Runner 负责分两次调用 trainer.fit() 完成两阶段训练。
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-6,
        metrics: Optional[List[str]] = None,
        num_classes: int = 14,
        tau: float = 1.0,
        eps: float = 1e-5,
        lam1: float = 0.0,
        lam2: float = 0.5,
        incremental_log_writer: Optional[Any] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # EntLoss 参数
        self.criterion = EntLoss(tau=tau, eps=eps, lam1=lam1, lam2=lam2)
        self.ce_criterion = nn.CrossEntropyLoss()

        # 训练阶段
        self.phase = "self_supervised"  # "self_supervised" / "supervised"

        # 指标（监督阶段使用，通过 core.metrics 注册表获取）
        if metrics is None:
            metrics = ["accuracy", "macro_f1"]
        self.metric_names = metrics
        self.val_metrics = nn.ModuleDict()
        for name in metrics:
            if has_metric(name):
                m = get_metric(name, num_classes)
                if m is not None:
                    self.val_metrics[name] = m
        # Phase 2.2b：验证阶段 confusion_matrix（仅最终验证时计算并输出）
        self._val_confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes,
        )
        self._final_confusion_matrix: Optional[List[List[int]]] = None

        # 训练日志
        self.training_log: List[Dict[str, Any]] = []
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0
        self._last_val_metrics: Dict[str, float] = {}
        self._is_final_validation: bool = False
        # A3: 增量日志写入器
        self._log_writer = incremental_log_writer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.phase == "self_supervised":
            return self._self_supervised_step(batch, batch_idx)
        else:
            return self._supervised_step(batch, batch_idx)

    def _self_supervised_step(self, batch, batch_idx):
        """自监督阶段：使用 EntLoss 训练编码器。"""
        x, _ = batch
        device = x.device

        x1 = gaussian_noise(x, random.uniform(0, 2.0), device)
        x2 = gaussian_noise(x, random.uniform(0.1, 2.0), device)

        feat_x1, feat_x2 = self.model(x1, x2)
        loss_dict = self.criterion(feat_x1, feat_x2)
        loss = loss_dict["final-kde"]

        self._current_epoch_loss += loss.item()
        self._current_epoch_steps += 1

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _supervised_step(self, batch, batch_idx):
        """监督微调阶段：使用 CrossEntropyLoss 只训练分类器。"""
        x, y = batch
        y = y.long()

        y1, y2 = self.model(x, x, flag="supervised")
        loss = self.ce_criterion(y1, y) + self.ce_criterion(y2, y)

        self._current_epoch_loss += loss.item()
        self._current_epoch_steps += 1

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证阶段：使用监督模式评估。"""
        x, y = batch
        y = y.long()

        y1, y2 = self.model(x, x, flag="supervised")
        loss = self.ce_criterion(y1, y) + self.ce_criterion(y2, y)

        preds = torch.argmax(y1, dim=1)
        for name, metric in self.val_metrics.items():
            metric(preds, y)
        # Phase 2.2b：累积 confusion_matrix
        self._val_confusion_matrix(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.val_metrics:
            self.log(f"val_{name}", self.val_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """Phase 2.2c：每 epoch 结束 log 当前学习率，便于监控 scheduler 衰减。"""
        # sanity_check 阶段无 optimizer，跳过
        if self.trainer.sanity_checking:
            return
        try:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups") and len(opt.param_groups) > 0:
                current_lr = opt.param_groups[0].get("lr")
                if current_lr is not None:
                    self.log("learning_rate", float(current_lr),
                             prog_bar=False, on_step=False, on_epoch=True)
        except Exception:
            pass

    def on_validation_epoch_end(self):
        """收集每轮验证指标到训练日志。"""
        # 跳过 sanity check
        if self.trainer.sanity_checking:
            return

        # 自监督预训练阶段不做验证指标收集（limit_val_batches=0，无验证数据）
        if self.phase == "self_supervised":
            self._current_epoch_loss = 0.0
            self._current_epoch_steps = 0
            return

        # 独立验证（trainer.validate()）：存储指标供 final_eval 使用
        if self._is_final_validation:
            self._last_val_metrics = {}
            for name, metric in self.val_metrics.items():
                self._last_val_metrics[name] = round(metric.compute().item(), 6)
                metric.reset()
            # Phase 2.2b：计算并存储最终 confusion_matrix
            cm = self._val_confusion_matrix.compute()
            self._final_confusion_matrix = cm.long().tolist()
            self._val_confusion_matrix.reset()
            return

        # 训练中的验证：添加到训练日志
        if self._current_epoch_steps == 0:
            return

        epoch_entry = {"epoch": self.current_epoch + 1, "phase": self.phase}
        epoch_entry["loss"] = round(self._current_epoch_loss / max(self._current_epoch_steps, 1), 6)

        for name in self.val_metrics:
            val = self.val_metrics[name].compute().item()
            epoch_entry[name] = round(val, 6)
            self.val_metrics[name].reset()
        # 训练中验证也重置 confusion_matrix（仅最终验证保留）
        self._val_confusion_matrix.reset()

        self.training_log.append(epoch_entry)
        # A3: 增量持久化，防崩溃丢失
        if self._log_writer is not None:
            self._log_writer.write(epoch_entry)
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0

    def get_final_metrics(self) -> Dict[str, Any]:
        """获取最终评测指标（含 Phase 2.2b confusion_matrix）。"""
        if self._last_val_metrics:
            results = dict(self._last_val_metrics)
        else:
            # Fallback: 从当前状态计算
            results = {}
            for name, metric in self.val_metrics.items():
                results[name] = round(metric.compute().item(), 6)
                metric.reset()
        # Phase 2.2b：附加 confusion_matrix（如有）
        if self._final_confusion_matrix is not None:
            results["confusion_matrix"] = self._final_confusion_matrix
        return results

    def configure_optimizers(self):
        """根据训练阶段返回不同的优化器。"""
        if self.phase == "self_supervised":
            # 自监督阶段：AdamW 训练全部参数
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            # 监督微调阶段：Adam 只训练 classifier
            optimizer = torch.optim.Adam(
                self.model.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5,
            )
        return optimizer


__all__ = ["EntLoss", "SelfSupervisedModule", "gaussian_noise"]
