"""
训练运行器：EpochLogCallback + run_experiment 薄适配器。

执行逻辑统一由 Pipeline.run() 提供（OOM 回退、stage checkpoint、OBP 指标、
feedback 分析、exploration_history、skill 自动沉淀）。
run_experiment 保留为向后兼容入口，委托 run_pipeline。
"""

from typing import Any

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

from ..config import ExperimentConfig
from ...observability import setup_logging
from ...schemas import TrainOutput

logger = setup_logging()


class EpochLogCallback(pl.Callback):
    """每 N epochs 打印一行训练进度日志（替代进度条）。

    P2: 集成 TrainingMonitor，每个 validation epoch end 写入实时指标，
    让 Agent 和用户在训练过程中看到实时指标曲线。
    """

    def __init__(self, log_every_n: int = 10, monitor: Any = None):
        super().__init__()
        self.log_every_n = log_every_n
        self.monitor = monitor  # P2: TrainingMonitor
        self._epoch_metrics = {}

    def _collect_metrics(self, trainer):
        """收集当前 epoch 的指标。"""
        metrics = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        return metrics

    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_metrics.update(self._collect_metrics(trainer))
        self._epoch_metrics["epoch"] = trainer.current_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        self._epoch_metrics.update(self._collect_metrics(trainer))

        # P2: 写入 TrainingMonitor
        if self.monitor is not None and self._epoch_metrics:
            self.monitor.on_epoch_end(self._epoch_metrics.copy())

        # 原有日志逻辑
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n == 0 or epoch == 0:
            train_loss = self._epoch_metrics.get("train_loss")
            val_loss = self._epoch_metrics.get("val_loss")
            val_macro_f1 = self._epoch_metrics.get("val_macro_f1")
            parts = [f"Epoch {epoch}/{trainer.max_epochs - 1}:"]
            parts.append(f"train_loss={train_loss:.4f}" if train_loss is not None else "train_loss=N/A")
            parts.append(f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=N/A")
            parts.append(f"val_macro_f1={val_macro_f1:.4f}" if val_macro_f1 is not None else "val_macro_f1=N/A")
            logger.info(" ".join(parts))

        self._epoch_metrics = {}  # 重置


def run_experiment(config: ExperimentConfig) -> TrainOutput:
    """声明式实验入口（薄适配器，委托 run_pipeline）。

    向后兼容：保留原入口签名，内部委托 Pipeline.run()。
    所有执行逻辑统一由 Pipeline.run() 提供，消除双执行路径的 feature parity 风险。
    """
    from .pipeline import run_pipeline
    return run_pipeline(config)
