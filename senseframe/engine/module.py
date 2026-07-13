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

import logging
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

# 修复（5.4 / 5.8）：模块级 logger + OTel 埋点常量导入
# 旧逻辑：ML_TRAIN_LOSS / ML_LEARNING_RATE 定义但从未被 record_training_metric 调用；
# LR 调度 / 显存 / 梯度裁剪 均无日志。
_logger = logging.getLogger(__name__)
try:
    from ..observability_otel import (
        record_training_metric,
        ML_TRAIN_LOSS,
        ML_LEARNING_RATE,
    )
except ImportError:
    # OTel 未安装时降级为 no-op
    def record_training_metric(*args, **kwargs):
        pass

    ML_TRAIN_LOSS = "ml.train.loss"
    ML_LEARNING_RATE = "ml.learning_rate"


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
        num_classes: int,
        learning_rate: float = 1e-3,
        metrics: Optional[List[str]] = None,
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
        # num_classes 必填：框架不猜测数据集类别数。
        # Pipeline 应从 ctx.num_classes（由 DatasetSpec.num_classes 派生）传入。
        if num_classes is None or num_classes <= 0:
            raise ValueError(
                "GenericLightningModule: num_classes 必填且必须 > 0。"
                "Pipeline 应从 ctx.num_classes 传入（由 DatasetSpec.num_classes 派生），"
                "直接实例化时必须显式提供。"
            )
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
        # 对称性修复：新增 test_metrics，与 val_metrics 对称
        self.test_metrics = nn.ModuleDict()
        for name in metrics:
            if not has_metric(name):
                continue
            m = get_metric(name, num_classes)
            if m is not None:
                self.train_metrics[name] = m
                self.val_metrics[name] = get_metric(name, num_classes)
                self.test_metrics[name] = get_metric(name, num_classes)
        # Phase 2.2b：验证阶段 confusion_matrix（仅分类任务且 num_classes>=2）
        if task_spec.task_type == "classification" and num_classes >= 2:
            self._val_confusion_matrix = ConfusionMatrix(
                task="multiclass", num_classes=num_classes,
            )
            # 对称性修复：测试阶段 confusion_matrix，与 val 对称
            self._test_confusion_matrix = ConfusionMatrix(
                task="multiclass", num_classes=num_classes,
            )
        else:
            self._val_confusion_matrix = None
            self._test_confusion_matrix = None
        self._final_confusion_matrix: Optional[List[List[int]]] = None
        # 对称性修复：测试阶段最终 confusion_matrix
        self._final_test_confusion_matrix: Optional[List[List[int]]] = None

        # 训练日志收集
        self.training_log: List[Dict[str, Any]] = []
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0
        self._current_lr: Optional[float] = None
        self._last_val_metrics: Dict[str, float] = {}
        # 对称性修复：测试阶段最终指标存储，与 _last_val_metrics 对称
        self._last_test_metrics: Dict[str, float] = {}
        self._is_final_validation: bool = False
        # 对称性修复：标记最终测试阶段，与 _is_final_validation 对称
        self._is_final_test: bool = False
        # Part 3：标记是否已有 validation 运行过（用于判断 epoch 0 是否需要写 train-only entry）
        self._has_validation_run: bool = False
        # 增量日志写入器（A3）：每 epoch 追加落盘，防崩溃丢失
        self._log_writer = incremental_log_writer
        # 修复（hook 顺序竞争）：累积器重置时机固定在 on_train_epoch_start，
        # 不在 on_train_epoch_end / on_validation_epoch_end 重置。
        # 根因：Lightning 2.x 中 on_validation_epoch_end 可能在 on_train_epoch_end
        # 之前执行（数据反推确认），两个 hook 共享 _current_epoch_loss / _steps
        # 且都重置，形成竞争：on_val_end 先重置 steps=0 → on_train_end 看到
        # steps=0 → _epoch_train_loss=None → epoch 2 train_loss=null。
        # 方案：重置时机固定在 on_train_epoch_start（train loop 之前，确定性的），
        # on_val_end 从累积器算 train_loss（此时累积器有当前 epoch 的值），
        # on_train_end 不重置、不保存实例变量。

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

    # 对称性修复：新增 test_step，与 validation_step 完全对称
    # P2-3 修复后 val/test 分离，test 集需要独立评估以报告泛化能力
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, bboxes, cls_labels = batch
            y = (bboxes, cls_labels)
        else:
            x, y = batch
        output = self(x)
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output
        self._check_finite(logits, "test_logits", batch_idx)
        loss = self._compute_loss(logits, y, batch_idx=batch_idx, model_output=output if isinstance(output, dict) else None)
        self._check_finite(loss, "test_loss", batch_idx)

        preds = self._compute_preds(logits, output if isinstance(output, dict) else None)
        self._update_metrics(self.test_metrics, preds, y)
        # 对称性修复：累积 test confusion_matrix（仅分类任务）
        if self.task_spec.task_type == "classification" and self._test_confusion_matrix is not None:
            self._test_confusion_matrix(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.test_metrics:
            self.log(f"test_{name}", self.test_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

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

    def on_train_epoch_start(self):
        """每 epoch 训练开始时重置累积器。

        修复（hook 顺序竞争）：累积器重置时机固定在此处。
        根因：Lightning 2.x 中 on_validation_epoch_end 可能在 on_train_epoch_end
        之前执行，两个 hook 共享 _current_epoch_loss / _steps 且都重置，形成竞争。
        方案：重置固定在 on_train_epoch_start（train loop 之前，确定性的），
        on_val_end / on_train_end 都不重置，从累积器读当前 epoch 的值。
        """
        # sanity_check 阶段不重置（无 training_step 累积）
        if self.trainer.sanity_checking:
            return
        self._current_epoch_loss = 0.0
        self._current_epoch_steps = 0

    def on_train_epoch_end(self):
        """Phase 2.2c：每 epoch 结束 log 当前学习率，便于监控 scheduler 衰减。

        字段契约（RFC-004 方案 C）：epoch_entry 必须含 train_loss / train_<metric>，
        与 val_loss / val_<metric> 对称，供 analyze_training_result 过拟合检测使用。

        修复（5.4 / 5.8）：
        - OTel 埋点 ML_TRAIN_LOSS / ML_LEARNING_RATE（OTel 未初始化时 no-op）
        - LR 调度 INFO 日志（epoch 末尾 log 当前 lr）
        - 梯度裁剪 DEBUG 日志（每 epoch log gradient_clip_val）
        - 显存使用 DEBUG 日志（每 epoch log GPU memory）
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
                    # 修复（5.4）：ML_LEARNING_RATE OTel 埋点（OTel 未初始化时 no-op）
                    record_training_metric(
                        ML_LEARNING_RATE, value=float(current_lr),
                        stage="train", epoch=int(self.current_epoch),
                    )
                    # 修复（5.8）：LR 调度 INFO 日志（epoch 末尾 log 当前 lr）
                    _logger.info(
                        "LR schedule: epoch=%d, lr=%.6e",
                        self.current_epoch, float(current_lr),
                    )
        except Exception:
            # 优化器未就绪等异常情况，静默跳过（不阻断训练）
            pass

        # 修复（5.4）：ML_TRAIN_LOSS OTel 埋点（OTel 未初始化时 no-op）
        # 在 epoch 末尾记录本轮平均 train_loss
        try:
            if self._current_epoch_steps > 0:
                avg_train_loss = self._current_epoch_loss / max(self._current_epoch_steps, 1)
                record_training_metric(
                    ML_TRAIN_LOSS, value=float(avg_train_loss),
                    stage="train", epoch=int(self.current_epoch),
                )
        except Exception:
            pass

        # 修复（5.8）：梯度裁剪 DEBUG 日志（每 epoch log gradient_clip_val）
        # 高频路径用 DEBUG 不用 INFO
        try:
            grad_clip_val = getattr(self.trainer, "gradient_clip_val", None)
            if grad_clip_val is not None:
                _logger.debug(
                    "gradient clip: epoch=%d, gradient_clip_val=%s",
                    self.current_epoch, grad_clip_val,
                )
        except Exception:
            pass

        # 修复（5.8）：显存使用 DEBUG 日志（每 epoch log GPU memory）
        # 高频路径用 DEBUG 不用 INFO
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                _logger.debug(
                    "GPU memory: epoch=%d, allocated=%.3fGB, reserved=%.3fGB",
                    self.current_epoch, allocated, reserved,
                )
        except Exception:
            pass

        # 修复（hook 顺序竞争）：on_train_epoch_end 不重置累积器、不保存实例变量。
        # 根因：Lightning 2.x 中 on_validation_epoch_end 可能在 on_train_epoch_end
        # 之前执行，两个 hook 共享 _current_epoch_loss / _steps 且都重置，形成竞争。
        # 方案：累积器重置固定在 on_train_epoch_start，on_val_end 从累积器算
        # train_loss（此时累积器有当前 epoch 的值），on_train_end 不触碰累积器。
        # train_accuracy 从 cb_metrics 读取（Lightning 固有偏移 1 epoch，可接受）。

        # Part 3（风险推演 R1）：补齐 epoch 0 的 train-only entry。
        # 根因：on_validation_epoch_end 在首 epoch 不触发（首 epoch 无 validation），
        # 导致 training_log 缺失 epoch 0 的 train 数据。metrics.csv 有此行（CSVLogger 自动写）。
        # 向 metrics.csv 看齐：首 epoch 也写入 training_log，val 字段为 None。
        # 用 phase 字段区分语义，避免 analyze_training_result 误消费。
        if self.current_epoch == 0 and not self._has_validation_run:
            # 从累积器算 train_loss（on_train_epoch_start 已重置，training_step 已累积）
            if self._current_epoch_steps > 0:
                tl_loss = round(
                    self._current_epoch_loss / self._current_epoch_steps, 6)
            else:
                tl_loss = None
            # train_accuracy 从 cb_metrics 读取（Lightning 固有偏移，epoch 0 无前序可偏移）
            cb_metrics_e0 = self.trainer.callback_metrics if self.trainer else {}
            epoch_entry = {
                "epoch": 0,
                "lr": getattr(self, "_current_lr", None),
                "train_loss": tl_loss,
                "phase": "train_only",  # 首 epoch 训练快照，无 validation
            }
            for name in self.train_metrics:
                key = f"train_{name}"
                val = cb_metrics_e0.get(key)
                if val is not None:
                    epoch_entry[key] = round(
                        float(val.item() if hasattr(val, "item") else val), 6)
                else:
                    epoch_entry[key] = None
            # val 字段全为 None（首 epoch 无 validation）
            epoch_entry["val_loss"] = None
            for name in self.val_metrics:
                epoch_entry[f"val_{name}"] = None
            self.training_log.append(epoch_entry)
            if self._log_writer is not None:
                self._log_writer.write(epoch_entry)

    def on_train_start(self) -> None:
        """P1-3: 训练开始时打印首 batch 摘要，验证数据通路。

        在 on_train_start 钩子中尝试获取首 batch 并打印 shape/dtype/range/classes，
        与 DataProfile 做一致性校验。失败时降级为 warning，绝不中断训练。

        注意：
        - Lightning 钩子 on_train_start 不接受 trainer 参数（与 pl.Callback 不同），
          通过 self.trainer 访问当前 trainer。
        - data_profile 字段由 P2-4 注入，这里用 getattr 安全访问。
        """
        try:
            # Lightning 钩子中通过 self.trainer 访问当前 trainer
            trainer = getattr(self, "trainer", None)
            if trainer is None:
                return
            dl = getattr(trainer, "train_dataloader", None)
            if dl is None:
                return
            # dl 可能是 CombinedLoader / DataLoader / dict-like，统一用 iter() 取首 batch
            first_batch = next(iter(dl))
            # first_batch 可能是 (X, y) / (X, bboxes, cls_labels) / dict
            if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 2:
                X, y = first_batch[0], first_batch[1]
                # 检测任务 batch 含 3 元素 (x, bboxes, cls_labels)，y 取 cls_labels
                if len(first_batch) >= 3:
                    y = first_batch[2]
            else:
                _logger.info("on_train_start: first batch type=%s, skip shape summary",
                             type(first_batch).__name__)
                return
            # X / y 应为 torch.Tensor；非 tensor 跳过详细摘要
            if not hasattr(X, "shape") or not hasattr(y, "shape"):
                _logger.info("on_train_start: first batch X type=%s, y type=%s",
                             type(X).__name__, type(y).__name__)
                return
            # 类别列表（仅 y 为整数型 tensor 时有意义）
            try:
                y_classes = torch.unique(y).tolist() if hasattr(y, "tolist") else None
            except Exception:
                y_classes = None
            _logger.info(
                "on_train_start: first batch X.shape=%s, X.dtype=%s, "
                "X.range=[%.4f, %.4f], y.shape=%s, y.dtype=%s, y.classes=%s",
                tuple(X.shape), X.dtype,
                float(X.min()), float(X.max()),
                tuple(y.shape), y.dtype,
                y_classes,
            )
            # P1-3: 与 DataProfile 一致性校验（data_profile 由 P2-4 注入，可能不存在）
            data_profile = getattr(self, "data_profile", None)
            if data_profile is not None:
                expected_shape = getattr(data_profile, "input_shape", None)
                actual_shape = tuple(X.shape[1:])  # 去 batch 维
                if expected_shape and list(actual_shape) != list(expected_shape):
                    _logger.warning(
                        "Batch shape mismatch: DataProfile.input_shape=%s, "
                        "actual batch shape=%s",
                        expected_shape, actual_shape,
                    )
        except Exception as e:
            # 日志失败不能中断训练，降级为 warning
            _logger.warning("on_train_start batch summary failed: %s", e)

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

        # Part 3：标记已有 validation 运行（供 on_train_epoch_end 判断 epoch 0 是否需要 train-only entry）
        if not self._is_final_validation:
            self._has_validation_run = True

        # 独立验证（trainer.validate()）：存储指标供 final_eval 使用
        if self._is_final_validation:
            self._last_val_metrics = {}
            cb_metrics = self.trainer.callback_metrics if self.trainer else {}
            # P1-2 修复：val_loss 优先从 callback_metrics 读取（与 ModelCheckpoint 同源），
            # 避免手动简单平均与 Lightning 加权平均的计算方式差异导致
            # final_eval.val_loss ≠ best_model_score（如 0.0814 vs 0.0829）
            val_loss_cb = cb_metrics.get("val_loss")
            if val_loss_cb is not None:
                self._last_val_metrics["val_loss"] = round(
                    float(val_loss_cb.item() if hasattr(val_loss_cb, "item") else val_loss_cb), 6
                )
            elif self._current_val_epoch_steps > 0:
                self._last_val_metrics["val_loss"] = round(
                    self._current_val_epoch_loss / max(self._current_val_epoch_steps, 1), 6
                )
            # 修复（双重 compute 陷阱）：Metric 对象交给 self.log 后 Lightning 自动
            # compute + reset，此处手动 compute 会得到 0（metric 已被 reset）。
            # 改为从 trainer.callback_metrics 读取 Lightning 已 compute 的值。
            # 旧逻辑：for name, metric in self.val_metrics.items():
            #     self._last_val_metrics[f"val_{name}"] = round(metric.compute().item(), 6)
            #     metric.reset()
            # 影响：final_eval 全 0 → HPO extract_metric 返回 0 → HPO 完全失效。
            for name in self.val_metrics:
                key = f"val_{name}"
                val = cb_metrics.get(key)
                if val is not None:
                    self._last_val_metrics[key] = round(float(val.item()
                                                   if hasattr(val, "item") else val), 6)
            # Phase 2.2b：计算并存储最终 confusion_matrix（仅分类任务）
            # confusion_matrix 未交给 self.log，需手动 compute（不受双重 compute 影响）
            if self._val_confusion_matrix is not None:
                cm = self._val_confusion_matrix.compute()
                self._final_confusion_matrix = cm.long().tolist()
                self._val_confusion_matrix.reset()
            # Part 3（风险推演 R1）：补齐 final validation entry。
            # trainer.validate() 产出的指标也写入 training_log，与 metrics.csv 的
            # epoch 行对齐。train 字段为 None（final validation 无训练）。
            # 用 phase="final_eval" 区分语义，analyze_training_result 会过滤此行。
            # 修复：去掉 +1，与 metrics.csv 的 CSVLogger epoch 编号对齐
            # （CSVLogger 用 current_epoch，final_eval 时 current_epoch 已是 early stop 后的值）
            final_epoch = self.current_epoch
            final_entry = {
                "epoch": final_epoch,
                "lr": None,
                "train_loss": None,
                "phase": "final_eval",
            }
            for name in self.train_metrics:
                final_entry[f"train_{name}"] = None
            final_entry["val_loss"] = self._last_val_metrics.get("val_loss")
            for name in self.val_metrics:
                key = f"val_{name}"
                final_entry[key] = self._last_val_metrics.get(key)
            self.training_log.append(final_entry)
            if self._log_writer is not None:
                self._log_writer.write(final_entry)
            # 重置累积器（与训练中验证路径对称）
            self._current_val_epoch_loss = 0.0
            self._current_val_epoch_steps = 0
            return

        # 训练中的验证：添加到训练日志（字段命名契约：train_/val_ 前缀对齐）
        # 修复（任务1 / P0）：Lightning 2.x 中 on_validation_epoch_end 触发时，
        # current_epoch 已是递增后的值（训练 epoch 0 结束后 current_epoch 即为 1）。
        # 旧逻辑 `+1` 导致 epoch 从 2 开始，丢失 epoch 1 数据（training_log.jsonl 与
        # metrics.csv 均从 epoch=2 起步）。去掉 +1，直接使用 current_epoch。
        epoch_entry = {"epoch": self.current_epoch}
        epoch_entry["lr"] = getattr(self, "_current_lr", None)

        # 修复（hook 顺序竞争）：train_loss 从累积器算，不受 hook 顺序影响。
        # 根因：on_validation_epoch_end 可能在 on_train_epoch_end 之前执行，
        # 此时 training_step 已累积当前 epoch 的 loss/steps，从累积器算正确。
        # 旧方案从 _epoch_train_loss 实例变量读，但该变量在 on_train_epoch_end
        # 才保存，hook 顺序不确定时读到旧值或 None。
        # 累积器在 on_train_epoch_start 重置（确定性时机），training_step 累积，
        # on_validation_epoch_end 读取（无论在 on_train_epoch_end 前后都正确）。
        if self._current_epoch_steps > 0:
            epoch_entry["train_loss"] = round(
                self._current_epoch_loss / self._current_epoch_steps, 6)
        else:
            epoch_entry["train_loss"] = None

        # train_accuracy 从 callback_metrics 读取（Lightning 固有行为：train_*
        # 在 on_train_epoch_end 之后 compute，on_val_end 时可能是上一个 epoch 的值，
        # 偏移 1 epoch 可接受，metrics.csv 有准确值）
        cb_metrics = self.trainer.callback_metrics if self.trainer else {}
        for name in self.train_metrics:
            key = f"train_{name}"
            val = cb_metrics.get(key)
            if val is not None:
                epoch_entry[key] = round(
                    float(val.item() if hasattr(val, "item") else val), 6)
            else:
                epoch_entry[key] = None

        # val 指标（统一 val_ 前缀）
        # val 指标在 validation loop 中刚 compute，callback_metrics 的 val_* 是当前 epoch 的值
        _vl = cb_metrics.get("val_loss")
        epoch_entry["val_loss"] = round(float(_vl.item() if hasattr(_vl, "item") else _vl), 6) if _vl is not None else None
        for name in self.val_metrics:
            key = f"val_{name}"
            val = cb_metrics.get(key)
            if val is not None:
                epoch_entry[key] = round(float(val.item()
                                         if hasattr(val, "item") else val), 6)
            else:
                epoch_entry[key] = None
        # 训练中验证也重置 confusion_matrix（仅最终验证保留）
        if self._val_confusion_matrix is not None:
            self._val_confusion_matrix.reset()

        self.training_log.append(epoch_entry)
        # A3: 增量持久化，防崩溃丢失
        if self._log_writer is not None:
            self._log_writer.write(epoch_entry)
        # 修复（hook 顺序竞争）：不重置 train 累积器（重置固定在 on_train_epoch_start）
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0

    # 对称性修复：新增 on_test_epoch_end，与 on_validation_epoch_end 的 final_eval 路径对称
    # trainer.test() 调用后触发，存储 test 指标供 get_final_metrics 消费
    def on_test_epoch_end(self):
        """对称性修复：收集 test 指标，与 on_validation_epoch_end 的 final_eval 路径对称。

        trainer.test() 调用后触发，存储 _last_test_metrics 供 get_final_metrics 消费。
        """
        # 跳过 sanity check
        if self.trainer.sanity_checking:
            return

        self._last_test_metrics = {}
        cb_metrics = self.trainer.callback_metrics if self.trainer else {}
        # test_loss 优先从 callback_metrics 读取（与 val_loss 同源逻辑）
        test_loss_cb = cb_metrics.get("test_loss")
        if test_loss_cb is not None:
            self._last_test_metrics["test_loss"] = round(
                float(test_loss_cb.item() if hasattr(test_loss_cb, "item") else test_loss_cb), 6
            )
        # 从 callback_metrics 读取 test 指标（避免双重 compute 陷阱，与 val 路径一致）
        for name in self.test_metrics:
            key = f"test_{name}"
            val = cb_metrics.get(key)
            if val is not None:
                self._last_test_metrics[key] = round(float(val.item()
                                                     if hasattr(val, "item") else val), 6)
        # 对称性修复：计算并存储最终 test confusion_matrix（仅分类任务）
        if self._test_confusion_matrix is not None:
            cm = self._test_confusion_matrix.compute()
            self._final_test_confusion_matrix = cm.long().tolist()
            self._test_confusion_matrix.reset()

    def get_final_metrics(self) -> Dict[str, Any]:
        """获取最终评测指标（含 Phase 2.2b confusion_matrix）。

        字段契约（RFC-004 方案 C）：返回 val_ 前缀字段（val_accuracy / val_loss 等），
        与 training_log 的 epoch_entry 字段命名对齐，供 analyze_training_result 消费。

        对称性修复：同时返回 test_ 前缀字段（test_accuracy / test_loss 等），
        供 analyze_training_result 做 val-test gap 泛化分析。
        """
        if self._last_val_metrics:
            results = dict(self._last_val_metrics)
        else:
            # Fallback: 从 trainer.callback_metrics 读取（同样使用 val_ 前缀，保持字段契约一致）
            # 修复（双重 compute 陷阱）：旧逻辑 metric.compute() 在 Lightning 已 reset 后返回 0。
            # 改为从 callback_metrics 读取，与 on_validation_epoch_end 一致。
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
        # 对称性修复：附加 test_ 前缀指标（如有）
        if self._last_test_metrics:
            results.update(self._last_test_metrics)
            # 附加 test confusion_matrix（如有）
            if self._final_test_confusion_matrix is not None:
                results["test_confusion_matrix"] = self._final_test_confusion_matrix
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

        # 修复（任务2 / P1）：明确 scheduler 配置，避免 Lightning 默认在 epoch 开始时
        # 调用 scheduler.step()（可能在 optimizer.step() 之前），触发
        # "Detected call of lr_scheduler.step() before optimizer.step()" 警告，
        # 并导致 cosine scheduler 跳过初始 lr。
        # cosine（CosineAnnealingLR）与 step（StepLR）均为 epoch 级调度器，
        # 故 interval="epoch" + frequency=1，确保 scheduler 在 epoch 结束后
        # （该 epoch 内 optimizer.step() 已多次调用）才 step。
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": "lr_scheduler",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


__all__ = ["GenericLightningModule"]
