"""
通用 LightningModule：将任意 nn.Module 包装为 LightningModule。

不依赖 CSI 特定逻辑，支持任意场景容器返回的 nn.Module。

功能：
- 支持 accuracy / macro_f1 / micro_f1 / weighted_f1 / macro_precision / macro_recall
- Phase 14.2.1：支持回归指标 mse / mae / rmse + 检测指标 map
- Phase 14.2.2：支持 Detection 任务（模型输出 dict，含 logits/bboxes）
- Phase 2.2b：最终验证阶段计算 confusion_matrix
- Phase 2.2c：每 epoch log 当前学习率
- NaN/Inf 守卫（B1）
- num_classes 一致性校验（B1）
- 增量日志写入（A3）
- 支持 adam / adamw / sgd 优化器
- 支持 cosine / step 学习率调度器
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MeanAbsoluteError, MeanSquaredError, Precision, Recall

from ..core.task import TaskSpec
from ..core.losses import loss_from_spec


class GenericLightningModule(pl.LightningModule):
    """
    通用监督学习 LightningModule，包装任意 nn.Module。

    Args:
        model: 模型实例（nn.Module）
        learning_rate: 学习率
        metrics: 指标名称列表（如 ["accuracy", "macro_f1"]）
        num_classes: 类别数
        optimizer: 优化器类型（adam / adamw / sgd）
        weight_decay: 权重衰减
        scheduler: 调度器类型（None / cosine / step）
        incremental_log_writer: 增量日志写入器（A3）
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        metrics: Optional[List[str]] = None,
        num_classes: int = 7,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,
        max_epochs: Optional[int] = None,
        incremental_log_writer: Optional[Any] = None,
        task_spec: Optional[TaskSpec] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        # 优化 4：max_epochs 用于动态计算 scheduler 的 T_max（cosine 衰减周期）
        # 避免硬编码 T_max=50 导致不同 epochs 任务学习率衰减不匹配
        self.max_epochs = max_epochs

        # Phase 11.1：TaskSpec 决定 loss / metrics
        if task_spec is None:
            task_spec = TaskSpec.classification(num_classes=num_classes)
        self.task_spec = task_spec

        # 指标：先按 task_spec 决定默认，再允许 metrics 覆盖
        if metrics is None:
            metrics = task_spec.effective_metrics
        # Phase 11 向后兼容：分类任务下若未指定 metrics 沿用 [accuracy, macro_f1]
        if metrics is None and task_spec.task_type == "classification":
            metrics = ["accuracy", "macro_f1"]
        self.metric_names = metrics

        # Phase 11.2：根据 task_spec 构造可配置 loss
        # R-fix：loss_kwargs 已是 TaskSpec 直接字段
        loss_kwargs = task_spec.loss_kwargs or {}
        self.criterion = loss_from_spec(task_spec.effective_loss, loss_kwargs)

        # 初始化 torchmetrics
        # R-fix：通过 core.metrics 注册表获取指标，替代硬编码 METRIC_MAP
        from ..core.metrics import has_metric, get_metric
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        for name in metrics:
            if not has_metric(name):
                continue
            m = get_metric(name, num_classes)
            if m is not None:
                self.train_metrics[name] = m
                self.val_metrics[name] = get_metric(name, num_classes)
        # Phase 2.2b：验证阶段 confusion_matrix（仅分类任务且 num_classes>=2）
        if task_spec.task_type == "classification" and num_classes >= 2:
            self._val_confusion_matrix = ConfusionMatrix(
                task="multiclass", num_classes=num_classes,
            )
        else:
            self._val_confusion_matrix = None
        self._final_confusion_matrix: Optional[List[List[int]]] = None

        # 训练日志收集
        self.training_log: List[Dict[str, Any]] = []
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0
        self._current_lr: Optional[float] = None
        self._last_val_metrics: Dict[str, float] = {}
        self._is_final_validation: bool = False
        # 增量日志写入器（A3）：每 epoch 追加落盘，防崩溃丢失
        self._log_writer = incremental_log_writer

    def _check_finite(self, t: torch.Tensor, name: str, batch_idx: int):
        """守卫：检测 NaN/Inf，触发时抛出含上下文的 RuntimeError（B1）。"""
        if not torch.isfinite(t).all():
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            raise RuntimeError(
                f"Non-finite values in {name}: nan={n_nan}, inf={n_inf}, "
                f"epoch={self.current_epoch}, batch_idx={batch_idx}"
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Detection 任务 batch 可能含 3 元素 (x, bboxes, cls_labels)
        if len(batch) == 3:
            x, bboxes, cls_labels = batch
            y = (bboxes, cls_labels)
        else:
            x, y = batch
        output = self(x)
        # Phase 14.2.2：Detection 任务模型输出 dict（含 logits/bboxes）
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output
        # B1: NaN/Inf 守卫
        self._check_finite(logits, "train_logits", batch_idx)
        # Phase 11.2：根据 task_spec 选择 loss
        loss = self._compute_loss(logits, y, batch_idx=batch_idx, model_output=output if isinstance(output, dict) else None)
        self._check_finite(loss, "train_loss", batch_idx)

        # 记录训练指标
        preds = self._compute_preds(logits, output if isinstance(output, dict) else None)
        self._update_metrics(self.train_metrics, preds, y)

        self._current_epoch_loss += loss.item()
        self._current_epoch_steps += 1

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.train_metrics:
            self.log(f"train_{name}", self.train_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, bboxes, cls_labels = batch
            y = (bboxes, cls_labels)
        else:
            x, y = batch
        output = self(x)
        # Phase 14.2.2：Detection 任务模型输出 dict
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output
        self._check_finite(logits, "val_logits", batch_idx)
        loss = self._compute_loss(logits, y, batch_idx=batch_idx, model_output=output if isinstance(output, dict) else None)
        self._check_finite(loss, "val_loss", batch_idx)

        preds = self._compute_preds(logits, output if isinstance(output, dict) else None)
        self._update_metrics(self.val_metrics, preds, y)
        # Phase 2.2b：累积 confusion_matrix（仅分类任务）
        if self.task_spec.task_type == "classification" and self._val_confusion_matrix is not None:
            self._val_confusion_matrix(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.val_metrics:
            self.log(f"val_{name}", self.val_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

        # 累积 val_loss（与 train_loss 对称，供 epoch_entry 使用）
        self._current_val_epoch_loss += loss.item()
        self._current_val_epoch_steps += 1

        return loss

    def _update_metrics(self, metric_dict: nn.ModuleDict, preds, y):
        """Phase 14.2.2：按 task_type 更新指标（检测指标需特殊格式）。"""
        for name, metric in metric_dict.items():
            if name == "map":
                # MeanAveragePrecision 需要 list[dict] 格式
                # preds: list of {"boxes", "scores", "labels"}
                # y: list of {"boxes", "labels"}
                if isinstance(preds, dict) and "preds_list" in preds:
                    metric.update(preds["preds_list"], y if isinstance(y, list) else [y])
                # else: 检测指标格式不匹配，跳过（不阻断训练）
            else:
                metric(preds, y)

    def _compute_loss(self, logits, y, batch_idx: int, model_output=None) -> torch.Tensor:
        """按 task_spec 计算 loss（Phase 11.2 + Phase 14.2.2）。

        分类任务：logits dim 与 num_classes 一致性校验 + CrossEntropy 家族
        回归任务：MSELoss / L1Loss / SmoothL1Loss（logits 与 y shape 一致）
        检测任务：从 model_output dict 提取 bboxes，传入 loss 函数
        其他任务：直接调用 self.criterion
        """
        task_type = self.task_spec.task_type
        if task_type == "classification":
            if logits.shape[-1] != self.num_classes:
                raise RuntimeError(
                    f"logits dim {logits.shape[-1]} != num_classes {self.num_classes} "
                    f"(epoch={self.current_epoch}, batch_idx={batch_idx})"
                )
            return self.criterion(logits, y.long())
        if task_type == "regression":
            return self.criterion(logits, y.float())
        if task_type == "detection":
            # Phase 14.2.2：检测任务 loss 使用 logits + y（bboxes 在 y 中）
            # model_output 含完整 dict（bboxes/logits），loss 函数按需提取
            return self.criterion(logits, y)
        return self.criterion(logits, y)

    def _compute_preds(self, logits, model_output=None):
        """按 task_spec 计算预测值（Phase 14.2.2 增加 Detection 分支）。"""
        if self.task_spec.task_type == "classification":
            return torch.argmax(logits, dim=1)
        if self.task_spec.task_type == "regression":
            return logits.detach()
        if self.task_spec.task_type == "detection":
            # Phase 14.2.2：检测任务返回 dict（含 bboxes/scores/labels 供 map 指标使用）
            if model_output is not None and "bboxes" in model_output:
                return {
                    "preds_list": [{
                        "boxes": model_output["bboxes"],
                        "scores": torch.sigmoid(logits).max(dim=-1).values
                                  if logits.dim() > 1 else torch.sigmoid(logits),
                        "labels": torch.argmax(logits, dim=-1) if logits.dim() > 1 else torch.zeros_like(logits),
                    }]
                }
            return logits.detach()
        return logits.detach()

    def on_train_epoch_end(self):
        """Phase 2.2c：每 epoch 结束 log 当前学习率，便于监控 scheduler 衰减。

        字段契约（RFC-004 方案 C）：epoch_entry 必须含 train_loss / train_<metric>，
        与 val_loss / val_<metric> 对称，供 analyze_training_result 过拟合检测使用。
        """
        # sanity_check 阶段无 optimizer，跳过
        if self.trainer.sanity_checking:
            return
        try:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups") and len(opt.param_groups) > 0:
                current_lr = opt.param_groups[0].get("lr")
                if current_lr is not None:
                    self._current_lr = float(current_lr)
                    # log 为标量，logger 后端（csv/tensorboard/wandb）均可记录
                    self.log("learning_rate", float(current_lr),
                             prog_bar=False, on_step=False, on_epoch=True)
        except Exception:
            # 优化器未就绪等异常情况，静默跳过（不阻断训练）
            pass

    def on_validation_epoch_end(self):
        """收集每轮验证指标到训练日志（字段契约对齐 RFC-004 方案 C）。

        epoch_entry 字段结构（与 analyze_training_result 消费者对齐）：
            epoch: int
            lr: Optional[float]
            train_loss: float
            train_<metric>: Optional[float]  # 如 train_accuracy
            val_loss: float
            val_<metric>: Optional[float]    # 如 val_accuracy
        """
        # 跳过 sanity check
        if self.trainer.sanity_checking:
            return

        # 独立验证（trainer.validate()）：存储指标供 final_eval 使用
        if self._is_final_validation:
            self._last_val_metrics = {}
            # val_loss 与 epoch_entry 字段契约对齐（RFC-004 方案 C）
            if self._current_val_epoch_steps > 0:
                self._last_val_metrics["val_loss"] = round(
                    self._current_val_epoch_loss / max(self._current_val_epoch_steps, 1), 6
                )
            for name, metric in self.val_metrics.items():
                # final_eval 字段也加 val_ 前缀，与 training_log 对齐
                self._last_val_metrics[f"val_{name}"] = round(metric.compute().item(), 6)
                metric.reset()
            # Phase 2.2b：计算并存储最终 confusion_matrix（仅分类任务）
            if self._val_confusion_matrix is not None:
                cm = self._val_confusion_matrix.compute()
                self._final_confusion_matrix = cm.long().tolist()
                self._val_confusion_matrix.reset()
            # 重置累积器（与训练中验证路径对称）
            self._current_val_epoch_loss = 0.0
            self._current_val_epoch_steps = 0
            return

        # 训练中的验证：添加到训练日志（字段命名契约：train_/val_ 前缀对齐）
        epoch_entry = {"epoch": self.current_epoch + 1}
        epoch_entry["lr"] = getattr(self, "_current_lr", None)
        epoch_entry["train_loss"] = round(
            self._current_epoch_loss / max(self._current_epoch_steps, 1), 6
        )

        # train 指标（与 val 对称）
        for name, metric in self.train_metrics.items():
            val = metric.compute().item()
            epoch_entry[f"train_{name}"] = round(val, 6)
            metric.reset()

        # val 指标（统一 val_ 前缀）
        epoch_entry["val_loss"] = round(
            self._current_val_epoch_loss / max(self._current_val_epoch_steps, 1), 6
        )
        for name in self.val_metrics:
            val = self.val_metrics[name].compute().item()
            epoch_entry[f"val_{name}"] = round(val, 6)
            self.val_metrics[name].reset()
        # 训练中验证也重置 confusion_matrix（仅最终验证保留）
        if self._val_confusion_matrix is not None:
            self._val_confusion_matrix.reset()

        self.training_log.append(epoch_entry)
        # A3: 增量持久化，防崩溃丢失
        if self._log_writer is not None:
            self._log_writer.write(epoch_entry)
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0

    def get_final_metrics(self) -> Dict[str, Any]:
        """获取最终评测指标（含 Phase 2.2b confusion_matrix）。

        字段契约（RFC-004 方案 C）：返回 val_ 前缀字段（val_accuracy / val_loss 等），
        与 training_log 的 epoch_entry 字段命名对齐，供 analyze_training_result 消费。
        """
        if self._last_val_metrics:
            results = dict(self._last_val_metrics)
        else:
            # Fallback: 从当前状态计算（同样使用 val_ 前缀，保持字段契约一致）
            results = {}
            for name, metric in self.val_metrics.items():
                results[f"val_{name}"] = round(metric.compute().item(), 6)
                metric.reset()
        # Phase 2.2b：附加 confusion_matrix（如有）
        if self._final_confusion_matrix is not None:
            results["confusion_matrix"] = self._final_confusion_matrix
        return results

    def configure_optimizers(self):
        """根据配置返回 optimizer (+ scheduler)。"""
        params = self.model.parameters()

        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        if self.scheduler_type is None:
            return optimizer
        elif self.scheduler_type == "cosine":
            # 优化 4：T_max 动态化，匹配实际训练 epochs（fallback 50 保持向后兼容）
            t_max = self.max_epochs if self.max_epochs else 50
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif self.scheduler_type == "step":
            # step_size 按 max_epochs 的 1/3 计算（fallback 30 保持向后兼容）
            step_size = max(1, self.max_epochs // 3) if self.max_epochs else 30
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


__all__ = ["GenericLightningModule"]
