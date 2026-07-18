"""
自监督学习 LightningModule：两阶段训练（自监督预训练 + 监督微调）。

从原 senseframe/self_supervised.py 移入 engine 层，去除 CSI 特定耦合：
- METRIC_MAP 从 engine.module 导入（不再经过 senseframe.module 别名）
- 逻辑保持不变，仅迁移位置

两阶段训练：
- Phase 1 (self_supervised): EntLoss (KL+EH+HE+KDE)，训练全部参数
- Phase 2 (supervised): CrossEntropyLoss，只训练 classifier 参数
"""

import logging
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

_logger = logging.getLogger(__name__)


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
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-6,
        metrics: Optional[List[str]] = None,
        tau: float = 1.0,
        eps: float = 1e-5,
        lam1: float = 0.0,
        lam2: float = 0.5,
        incremental_log_writer: Optional[Any] = None,
        scheduler: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # 修复（2.12）：与 GenericLightningModule 对齐，支持 scheduler
        self.scheduler_type = scheduler
        self.max_epochs = max_epochs
        # num_classes 必填：框架不猜测数据集类别数。
        # Pipeline 应从 ctx.num_classes（由 DatasetSpec.num_classes 派生）传入。
        if num_classes is None or num_classes <= 0:
            raise ValueError(
                "SelfSupervisedModule: num_classes 必填且必须 > 0。"
                "Pipeline 应从 ctx.num_classes 传入（由 DatasetSpec.num_classes 派生），"
                "直接实例化时必须显式提供。"
            )
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
        # test 阶段指标（与 val 对称）
        self.test_metrics = nn.ModuleDict()
        for name in metrics:
            if has_metric(name):
                m = get_metric(name, num_classes)
                if m is not None:
                    self.test_metrics[name] = m
        self._test_confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes,
        )
        self._final_test_confusion_matrix: Optional[List[List[int]]] = None
        self._last_test_metrics: Dict[str, float] = {}
        self._is_final_test: bool = False

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

    def test_step(self, batch, batch_idx):
        """测试阶段：与 validation_step 对称，使用 test_metrics 和 test_ 前缀。"""
        x, y = batch
        y = y.long()

        y1, y2 = self.model(x, x, flag="supervised")
        loss = self.ce_criterion(y1, y) + self.ce_criterion(y2, y)

        preds = torch.argmax(y1, dim=1)
        for name, metric in self.test_metrics.items():
            metric(preds, y)
        self._test_confusion_matrix(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.test_metrics:
            self.log(f"test_{name}", self.test_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """收集 test 指标，与 on_validation_epoch_end 的 final_eval 路径对称。"""
        if self.trainer.sanity_checking:
            return

        self._last_test_metrics = {}
        cb_metrics = self.trainer.callback_metrics if self.trainer else {}
        test_loss_cb = cb_metrics.get("test_loss")
        if test_loss_cb is not None:
            self._last_test_metrics["test_loss"] = round(
                float(test_loss_cb.item() if hasattr(test_loss_cb, "item") else test_loss_cb), 6
            )
        for name in self.test_metrics:
            key = f"test_{name}"
            val = cb_metrics.get(key)
            if val is not None:
                self._last_test_metrics[key] = round(float(val.item()
                                                   if hasattr(val, "item") else val), 6)
        if self._test_confusion_matrix is not None:
            cm = self._test_confusion_matrix.compute()
            self._final_test_confusion_matrix = cm.long().tolist()
            self._test_confusion_matrix.reset()

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
            # 修复（双重 compute 陷阱）：val_metrics 已交给 self.log，Lightning 自动
            # compute + reset，手动 compute 返回 0。改为从 callback_metrics 读取。
            cb_metrics = self.trainer.callback_metrics if self.trainer else {}
            for name in self.val_metrics:
                # self.log 键名为 f"val_{name}"（见 validation_step），从 callback_metrics
                # 读取对应值。旧逻辑直接用 name 作键（无 val_ 前缀），与 training_log
                # 字段契约不一致（修复 2.6 字段命名）。
                key = f"val_{name}"
                val = cb_metrics.get(key)
                if val is not None:
                    self._last_val_metrics[key] = round(float(val.item()
                                                       if hasattr(val, "item") else val), 6)
            # Phase 2.2b：计算并存储最终 confusion_matrix
            # confusion_matrix 未交给 self.log，需手动 compute（不受双重 compute 影响）
            cm = self._val_confusion_matrix.compute()
            self._final_confusion_matrix = cm.long().tolist()
            self._val_confusion_matrix.reset()
            return

        # 训练中的验证：添加到训练日志
        if self._current_epoch_steps == 0:
            return

        epoch_entry = {"epoch": self.current_epoch + 1, "phase": self.phase}
        # 修复（2.6 字段命名）：loss → train_loss，与 _TRAINING_LOG_ENTRY_SCHEMA 一致
        epoch_entry["train_loss"] = round(self._current_epoch_loss / max(self._current_epoch_steps, 1), 6)

        # 修复（双重 compute 陷阱）：从 callback_metrics 读取，不手动 compute。
        # 旧逻辑 val_metrics[name].compute() 在 Lightning 已 reset 后返回 0。
        cb_metrics = self.trainer.callback_metrics if self.trainer else {}
        for name in self.val_metrics:
            key = f"val_{name}"
            val = cb_metrics.get(key)
            if val is not None:
                epoch_entry[key] = round(float(val.item()
                                         if hasattr(val, "item") else val), 6)
            else:
                epoch_entry[key] = None
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
            # Fallback: 从 callback_metrics 读取
            # 修复（双重 compute 陷阱）：旧逻辑 metric.compute() 在 Lightning 已 reset 后返回 0。
            results = {}
            cb_metrics = self.trainer.callback_metrics if self.trainer else {}
            for name in self.val_metrics:
                key = f"val_{name}"
                val = cb_metrics.get(key)
                if val is not None:
                    results[key] = round(float(val.item()
                                         if hasattr(val, "item") else val), 6)
        # Phase 2.2b：附加 confusion_matrix（如有）
        if self._final_confusion_matrix is not None:
            results["confusion_matrix"] = self._final_confusion_matrix
        # 合并 test 指标（与 GenericLightningModule 对称）
        if self._last_test_metrics:
            results.update(self._last_test_metrics)
            if self._final_test_confusion_matrix is not None:
                results["test_confusion_matrix"] = self._final_test_confusion_matrix
        return results

    def configure_optimizers(self):
        """根据训练阶段返回不同的优化器（+ 可选 scheduler）。"""
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
        # 修复（2.12）：与 GenericLightningModule 对齐，支持 scheduler
        if self.scheduler_type is None:
            return optimizer
        elif self.scheduler_type == "cosine":
            t_max = self.max_epochs if self.max_epochs else 50
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif self.scheduler_type == "step":
            step_size = max(1, self.max_epochs // 3) if self.max_epochs else 30
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


__all__ = ["EntLoss", "SelfSupervisedModule", "gaussian_noise"]
