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
    # M3 修复：原 torch.normal(1, 2, ...) 采样自 N(1, 4)，噪声期望为 1 而非 0，
    # 在输入上叠加 epsilon * 1 的恒定均值偏移，与"零均值扰动"的增强语义相悖。
    # 改为零均值高斯噪声，std=2.0，保留 epsilon 缩放。
    noise = torch.randn(size=csi.shape, device=device) * 2.0
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
        # MEDIUM 4 修复：PSNR 重建缓存（供 PSNREarlyStoppingCallback 消费）
        # validation_step 在 MAE model 上缓存重建/目标张量，
        # PSNREarlyStoppingCallback.on_validation_epoch_end 读取计算 PSNR。
        self._psnr_reconstruction = None
        self._psnr_target = None
        # P1-4：val 阶段 loss 累加器（与 GenericLightningModule 对称），
        # 用于 on_validation_epoch_end final validation 分支的 val_loss fallback。
        # 主路径从 callback_metrics 读取，fallback 从累加器计算。
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0

    def forward(self, x):
        # P0-2 修复：_Parrallel.forward 签名为 (x1, x2, flag=...)，单参数调用会 TypeError。
        # Lightning 不走 forward（走 training_step），但外部调用 module(x)（如 inference）
        # 需要单输入监督路径。包装为 (x, x, flag='supervised') 取 y1。
        if hasattr(self.model, "forward") and callable(getattr(self.model, "forward", None)):
            import inspect
            sig = inspect.signature(self.model.forward)
            required = [p for p in sig.parameters.values()
                        if p.name != "self" and p.default is inspect.Parameter.empty]
            if len(required) >= 2:
                y1, _y2 = self.model(x, x, flag="supervised")
                return y1
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
        """验证阶段：使用监督模式评估。

        MEDIUM 4 修复：当 model 支持 MAE 重建（duck-typed _mae_forward_loss）时，
        缓存 _psnr_reconstruction/_psnr_target 供 PSNREarlyStoppingCallback 消费。
        """
        x, y = batch
        y = y.long()

        # MEDIUM 4：MAE 重建缓存（供 PSNREarlyStoppingCallback）
        # Tradeoff：每个 batch 覆盖缓存，仅保留最后一个 batch 的重建。
        # 这避免 cat 累积所有 batch 导致显存膨胀，PSNR 用单 batch 作代表已足够指示趋势。
        # 若需更精确的 epoch 级 PSNR，应改为 accumulate + on_validation_epoch_end 聚合。
        # M14：mae_reconstruct 内部随机采样 mask，此处接受 mask 随机性，不固定 seed——
        # PSNR 仅作为重建质量趋势的指示指标（早停判断用），单 batch 随机 mask 的波动
        # 在 epoch 级别可接受；若需复现严格 PSNR 数值，应在 mae_reconstruct 内固定 generator。
        # I11 修复：改用 mae_reconstruct 公共方法，消除 _forward_encoder/_forward_decoder 私有方法外调
        # I12 修复：except Exception as e + _logger.warning，避免静默吞异常
        # I13 修复：缓存张量 .detach().cpu()，避免 GPU 显存泄漏
        # 安全修复：先暂存到局部变量，model() 调用成功后才写入实例属性，
        # 防止 model() 失败后残留缓存导致 PSNREarlyStoppingCallback 基于无效数据决策
        _psnr_recon = None
        _psnr_tgt = None
        if hasattr(self.model, "mae_reconstruct"):
            try:
                mask_ratio = getattr(self.model, "_mask_ratio", 0.75)
                recon, target, mask = self.model.mae_reconstruct(x, mask_ratio)
                mask_bool = mask.bool()  # M13 修复：简化死分支
                _psnr_recon = recon[mask_bool].detach().cpu()  # I13: 加 .cpu()
                _psnr_tgt = target[mask_bool].detach().cpu()
            except Exception as e:
                _logger.warning("PSNR reconstruction cache failed: %s", e, exc_info=True)  # I12: debug→warning

        y1, y2 = self.model(x, x, flag="supervised")
        loss = self.ce_criterion(y1, y) + self.ce_criterion(y2, y)

        # model() 调用成功，将暂存的 PSNR 缓存写入实例属性
        self._psnr_reconstruction = _psnr_recon
        self._psnr_target = _psnr_tgt

        preds = torch.argmax(y1, dim=1)
        for name, metric in self.val_metrics.items():
            metric(preds, y)
        # Phase 2.2b：累积 confusion_matrix
        self._val_confusion_matrix(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name in self.val_metrics:
            self.log(f"val_{name}", self.val_metrics[name], prog_bar=True, on_step=False, on_epoch=True)

        # P1-4：累加 val_loss 供 on_validation_epoch_end fallback 使用
        # （主路径从 callback_metrics 读取，此为备用路径）
        self._current_val_epoch_loss += loss.item()
        self._current_val_epoch_steps += 1

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
        """每 epoch 结束：log 学习率 + SS 阶段写 training_log。

        修复：SS 阶段 training_log 写入从 on_validation_epoch_end 移到此处。
        原逻辑在 on_validation_epoch_end 写，但 SS 阶段 limit_val_batches=0 时
        on_validation_epoch_end 不触发，导致 SS 阶段无 training_log 记录，
        Agent 无法判断预训练收敛，HPO 缺少预训练曲线。
        """
        # sanity_check 阶段无 optimizer，跳过
        if self.trainer.sanity_checking:
            return
        current_lr = None
        try:
            opt = self.optimizers()
            if opt is not None and hasattr(opt, "param_groups") and len(opt.param_groups) > 0:
                current_lr = opt.param_groups[0].get("lr")
                if current_lr is not None:
                    self.log("learning_rate", float(current_lr),
                             prog_bar=False, on_step=False, on_epoch=True)
        except Exception as e:
            _logger.debug("learning_rate log skipped: %s", e)
        # 修复：SS 阶段在 on_train_epoch_end 写 training_log（不依赖 validation 触发）
        if self.phase == "self_supervised":
            if self._current_epoch_steps > 0:
                epoch_entry = {
                    "epoch": self.current_epoch,
                    "phase": "self_supervised",
                    "train_loss": round(self._current_epoch_loss / max(self._current_epoch_steps, 1), 6),
                }
                if current_lr is not None:
                    epoch_entry["lr"] = round(float(current_lr), 6)
                else:
                    epoch_entry["lr"] = None
                epoch_entry["train_accuracy"] = None
                self.training_log.append(epoch_entry)
                if self._log_writer is not None:
                    self._log_writer.write(epoch_entry)
            self._current_epoch_loss = 0.0
            self._current_epoch_steps = 0

    def on_validation_epoch_end(self):
        """收集每轮验证指标到训练日志。"""
        # 跳过 sanity check
        if self.trainer.sanity_checking:
            return

        # 修复：SS 阶段 training_log 已在 on_train_epoch_end 写入（不依赖 validation 触发）。
        # 原逻辑在 on_validation_epoch_end 写，但 limit_val_batches=0 时不触发，
        # 导致 SS 阶段无 training_log 记录。现 on_train_epoch_end 负责写入，此处仅重置累加器
        # （on_train_epoch_end 已重置，此处冗余但安全，防止 limit_val_batches>0 时重复写入）。
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
            # P1-4 修复：补 val_loss 读取，与 GenericLightningModule（module.py L563-571）对齐。
            # validation_step 已 self.log("val_loss", loss, ...)，cb_metrics 必有 val_loss 键。
            # 旧逻辑只迭代 val_metrics（accuracy/macro_f1），遗漏 val_loss，导致 final_eval
            # 缺 val_loss，下游 hpo.py:432 result.final_eval.get("val_loss") 拿到 None。
            val_loss_cb = cb_metrics.get("val_loss")
            if val_loss_cb is not None:
                self._last_val_metrics["val_loss"] = round(
                    float(val_loss_cb.item() if hasattr(val_loss_cb, "item") else val_loss_cb), 6
                )
            elif self._current_val_epoch_steps > 0:
                # Fallback：callback_metrics 未命中时从累加器计算（与 GenericLightningModule 对称）
                self._last_val_metrics["val_loss"] = round(
                    self._current_val_epoch_loss / max(self._current_val_epoch_steps, 1), 6
                )
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
            # P1-4：重置 val 累加器（与 train 累加器对称，防止跨 final validation 重复累积）
            self._current_val_epoch_loss = 0.0
            self._current_val_epoch_steps = 0
            return

        # 训练中的验证：添加到训练日志
        if self._current_epoch_steps == 0:
            return

        # I18 修复：去掉 +1，与 GenericLightningModule（module.py L623）对齐。
        # Lightning 2.x on_validation_epoch_end 触发时 current_epoch 已是递增后的值
        # （训练 epoch 0 结束后 current_epoch 即为 1），旧逻辑 +1 导致 epoch 从 2 开始，
        # 跨阶段对比错位（self_supervised 与 supervised 阶段日志错位 1 epoch）。
        epoch_entry = {"epoch": self.current_epoch, "phase": self.phase}
        # 修复（2.6 字段命名）：loss → train_loss，与 _TRAINING_LOG_ENTRY_SCHEMA 一致
        epoch_entry["train_loss"] = round(self._current_epoch_loss / max(self._current_epoch_steps, 1), 6)

        # I19 修复：补 lr + train_accuracy 字段，与 GenericLightningModule（module.py L624/L650）
        # 和 DANN 路径对齐（schemas.TrainingLogEntry 契约要求 lr/train_accuracy 等字段）。
        # lr 从 callback_metrics['learning_rate'] 读取（on_train_epoch_end L280 已 log）。
        cb_metrics = self.trainer.callback_metrics if self.trainer else {}
        lr_val = cb_metrics.get("learning_rate")
        if lr_val is not None:
            epoch_entry["lr"] = round(float(lr_val.item() if hasattr(lr_val, "item") else lr_val), 6)
        else:
            epoch_entry["lr"] = None
        # SelfSupervisedModule 无 train_metrics（仅 val_metrics），train_accuracy 恒为 None。
        # 补此字段使 analyze_training_result 的 train-val gap 检测不会因缺字段被跳过。
        epoch_entry["train_accuracy"] = None

        # 修复（双重 compute 陷阱）：从 callback_metrics 读取，不手动 compute。
        # 旧逻辑 val_metrics[name].compute() 在 Lightning 已 reset 后返回 0。
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
        # P1-4 修复：重置 val 累加器，与 GenericLightningModule（module.py L673-674）对称。
        # 旧逻辑遗漏此重置，导致跨 epoch 持续累加，final validation fallback 路径
        # 计算的是多 epoch 平均值而非单 epoch 值。
        self._current_val_epoch_loss = 0.0
        self._current_val_epoch_steps = 0

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
