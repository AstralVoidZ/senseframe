"""
训练运行器：EpochLogCallback + run_experiment 薄适配器。

执行逻辑统一由 Pipeline.run() 提供（OOM 回退、stage checkpoint、OBP 指标、
feedback 分析、exploration_history、skill 自动沉淀）。
run_experiment 保留为向后兼容入口，委托 run_pipeline。
"""

from typing import Any, Dict, Optional

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


class IntermediateMetricLogger(pl.Callback):
    """P2.3: 捕获 epoch 级中间值到 intermediate_values dict（ε5 Multi-fidelity）。

    每 validation epoch end 将目标指标写入 intermediate_values[epoch]，
    供 MethodRunner 早停检查（P2.4）与 SP Pruner should_prune 使用。

    设计原则（RFC-003 原则 4）：
    - 通过 Lightning Callback 注入，不修改 stage_train 主逻辑
    - 写入外部 dict（由 PipelineContext.intermediate_values 提供），
      回调本身无状态，便于测试与复用

    Args:
        metric: 目标指标名（trainer.callback_metrics 的 key），默认 "val_accuracy"
        intermediate_values: 外部 dict 引用，回调写入 {epoch: value}；
                             None 时回调 no-op（便于无条件注入）
    """

    def __init__(
        self,
        metric: str = "val_accuracy",
        intermediate_values: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.metric = metric
        self.intermediate_values = intermediate_values  # 外部 dict 引用；None 则 no-op

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.intermediate_values is None:
            return
        value = trainer.callback_metrics.get(self.metric)
        if value is None:
            return
        # Lightning callback_metrics 可能返回 Tensor 或 float（版本/后端相关）
        if hasattr(value, "item"):
            value = float(value.item())
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            return
        self.intermediate_values[trainer.current_epoch] = value


def run_experiment(config: ExperimentConfig) -> TrainOutput:
    """声明式实验入口（薄适配器，委托 run_pipeline）。

    向后兼容：保留原入口签名，内部委托 Pipeline.run()。
    所有执行逻辑统一由 Pipeline.run() 提供，消除双执行路径的 feature parity 风险。
    """
    from .pipeline import run_pipeline
    return run_pipeline(config)
