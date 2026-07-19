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

from .callbacks import StageAwareCallback
from ..config import ExperimentConfig
from ...observability import setup_logging
from ...schemas import TrainOutput

logger = setup_logging()


class EpochLogCallback(StageAwareCallback):
    """每 N epochs 打印一行训练进度日志（替代进度条）。

    P2: 集成 TrainingMonitor，每个 validation epoch end 写入实时指标，
    让 Agent 和用户在训练过程中看到实时指标曲线。
    """

    active_stages = {"train"}  # P0-1: 仅在 stage_train 激活

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
        # 修复（5.20）：Callback 异常不应中断训练
        # P0-2：不再写入 self._epoch_metrics（避免跨 hook 状态污染），
        # on_validation_epoch_end 会重新调用 _collect_metrics 获取完整指标
        try:
            # 临时变量收集，仅用于异常检测，结果丢弃
            _metrics = self._collect_metrics(trainer)
        except Exception as e:
            logger.warning("EpochLogCallback.on_train_epoch_end failed: %s", e, exc_info=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 修复（5.20）：Callback 异常不应中断训练
        try:
            # P0-1: 仅在 stage_train 激活
            if not self.is_active():
                return
            # P0-2: 跳过 sanity_check 阶段
            if trainer.sanity_checking:
                self._epoch_metrics = {}
                return
            # P0-2: 重新调用 _collect_metrics 获取完整指标（此时 train + val 都已 compute）
            # 不再依赖 on_train_epoch_end 的隐式状态传递
            metrics = self._collect_metrics(trainer)

            # P2: 写入 TrainingMonitor
            if self.monitor is not None and metrics:
                self.monitor.on_epoch_end(metrics.copy())

            # 原有日志逻辑
            # 修复（5.7）：旧逻辑 log_every_n=10，前 1-8 epoch 完全静默，
            # 用户无法追踪训练启动初期是否正常收敛。
            # 改为：前 5 epoch 每 epoch 打印；之后保持 log_every_n 间隔。
            epoch = trainer.current_epoch
            should_log = (
                epoch == 0
                or epoch < 5
                or (epoch + 1) % self.log_every_n == 0
            )
            if should_log:
                train_loss = metrics.get("train_loss")
                val_loss = metrics.get("val_loss")
                val_macro_f1 = metrics.get("val_macro_f1")
                parts = [f"Epoch {epoch}/{trainer.max_epochs - 1}:"]
                parts.append(f"train_loss={train_loss:.4f}" if train_loss is not None else "train_loss=N/A")
                parts.append(f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=N/A")
                parts.append(f"val_macro_f1={val_macro_f1:.4f}" if val_macro_f1 is not None else "val_macro_f1=N/A")
                logger.info(" ".join(parts))
        except Exception as e:
            logger.warning("EpochLogCallback.on_validation_epoch_end failed: %s", e, exc_info=True)


class IntermediateMetricLogger(StageAwareCallback):
    """P2.3: 捕获 epoch 级中间值到 intermediate_values dict（ε5 Multi-fidelity）。

    每 validation epoch end 将目标指标写入 intermediate_values[epoch]，
    供 MethodRunner 早停检查（P2.4）与 SP Pruner should_prune 使用。

    P1.1 Multi-fidelity 实时早停修复：
    - 若 pruner 注入，每个 epoch end 写入 intermediate_values 后立即调
      pruner.should_prune(trial_id, intermediate_values, rung=current_epoch)
    - should_prune=True 时设 trainer.should_stop=True，Lightning 提前终止训练
    - 通过 on_pruned 回调通知 stage_train 写 ctx.pruned/pruned_epoch
    - 替代 MethodRunner 事后剪枝（训练完整跑完才检查），节省无效训练算力

    设计原则（RFC-003 原则 4）：
    - 通过 Lightning Callback 注入，不修改 stage_train 主逻辑
    - 写入外部 dict（由 PipelineContext.intermediate_values 提供），
      回调本身无状态（除 _pruned_this_session 幂等标志），便于测试与复用

    epoch 索引方案（P1 修复）：
    - 使用 1-indexed（trainer.current_epoch + 1），与 training_log/CSV 的 epoch 对齐
    - 旧逻辑用 0-indexed（trainer.current_epoch），导致 intermediate_values key "1"
      对应 CSV epoch 2，Agent 跨系统读取时 off-by-one
    - OptunaReportingCallback 保持 0-indexed（Optuna step 惯例，独立索引空间）

    Args:
        metric: 目标指标名（trainer.callback_metrics 的 key），默认 "val_accuracy"
        intermediate_values: 外部 dict 引用，回调写入 {epoch: value}（1-indexed）；
                             None 时回调 no-op（便于无条件注入）
        pruner: Pruner Protocol 实例（None 时退化为旧路径，仅写 intermediate_values 不剪枝）
        trial_id: 当前 trial ID（传给 pruner.should_prune 用于跨 trial 比对）
        on_pruned: 剪枝回调 `(epoch: int) -> None`，stage_train 注入 lambda 写 ctx.pruned/pruned_epoch；
                   None 时仅设 trainer.should_stop，不通知外部
    """

    active_stages = {"train"}  # P0-1: 仅在 stage_train 激活

    def __init__(
        self,
        metric: str = "val_accuracy",
        intermediate_values: Optional[Dict[int, float]] = None,
        pruner: Any = None,
        trial_id: str = "",
        on_pruned: Any = None,
    ):
        super().__init__()
        self.metric = metric
        self.intermediate_values = intermediate_values  # 外部 dict 引用；None 则 no-op
        # P1.1: 实时早停注入
        self.pruner = pruner
        self.trial_id = trial_id
        self.on_pruned = on_pruned  # Optional[Callable[[int], None]]
        # 幂等标志：单次 fit() 内一旦剪枝就不再重复检查
        # （Lightning 可能在 prune 后再触发 on_validation_epoch_end，重复检查无意义且可能误判）
        self._pruned_this_session = False

    def on_validation_epoch_end(self, trainer, pl_module):
        # P1-5.20: 异常保护，Callback 异常不应中断训练
        try:
            self._on_validation_epoch_end_impl(trainer, pl_module)
        except Exception as e:
            logger.warning(
                "IntermediateMetricLogger.on_validation_epoch_end failed "
                "(metric=%s, epoch=%d): %s",
                self.metric, getattr(trainer, "current_epoch", -1), e,
            )

    def _on_validation_epoch_end_impl(self, trainer, pl_module):
        # P0-1: 仅在 stage_train 激活
        if not self.is_active():
            return
        # P0-3: 跳过 sanity_check 阶段
        if trainer.sanity_checking:
            return
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
        # P1 修复：1-indexed，与 training_log/CSV 的 epoch 对齐
        epoch_1indexed = trainer.current_epoch + 1
        self.intermediate_values[epoch_1indexed] = value

        # P1.1 Multi-fidelity 实时早停检查
        # 已剪枝则跳过（幂等）；pruner 未注入则跳过（向后兼容旧路径）
        if self._pruned_this_session or self.pruner is None:
            return

        # 方案 A：每个 epoch 调 should_prune，rung 用当前 epoch（1-indexed）
        # 与 intermediate_values key 对齐，pruner 内部按 rung 分桶比对
        try:
            should_prune = self.pruner.should_prune(
                self.trial_id,
                dict(self.intermediate_values),  # 浅拷贝避免 pruner 修改外部 dict
                rung=epoch_1indexed,
            )
        except Exception as e:
            # Pruner 异常不应阻断训练，降级为不剪枝
            logger.warning(
                "IntermediateMetricLogger: pruner.should_prune raised "
                "(trial_id=%s, epoch=%d), skipping prune: %s",
                self.trial_id, epoch_1indexed, e,
            )
            return

        if should_prune:
            # 设 trainer.should_stop=True 让 Lightning 提前终止
            # Lightning 2.x Trainer 公开属性，下一轮 epoch 开始前检查并退出 fit loop
            try:
                trainer.should_stop = True
            except Exception as e:
                logger.warning(
                    "IntermediateMetricLogger: failed to set trainer.should_stop "
                    "(epoch=%d): %s", epoch_1indexed, e,
                )
                return
            self._pruned_this_session = True
            logger.info(
                "IntermediateMetricLogger: trial pruned at epoch=%d "
                "(trial_id=%s, metric=%s, value=%.4f)",
                epoch_1indexed, self.trial_id, self.metric, value,
            )
            # 通知外部（stage_train 注入的回调）写 ctx.pruned/pruned_epoch
            if self.on_pruned is not None:
                try:
                    self.on_pruned(epoch_1indexed)
                except Exception as e:
                    logger.warning(
                        "IntermediateMetricLogger: on_pruned callback raised "
                        "(epoch=%d): %s", epoch_1indexed, e,
                    )


class OptunaReportingCallback(StageAwareCallback):
    """修复（2.8）：桥接 Lightning 中间指标到 Optuna trial.report()，启用 HPO 剪枝。

    在 on_validation_epoch_end 调用 trial.report(intermediate_value, step)，
    让 Optuna Pruner 基于中间 epoch 的指标值决定是否剪枝。

    现状（修复前）：IntermediateMetricLogger 仅写入 intermediate_values dict，
    pruner 永远收不到 intermediate values，HPO 永不剪枝。

    设计原则（RFC-003 原则 4）：
    - 通过 Lightning Callback 注入，不修改 stage_train 主逻辑
    - trial 通过构造函数传入（duck-typed，需有 report(value, step) 方法）

    Args:
        trial: optuna.trial.Trial 实例（None 时回调 no-op，便于无条件注入）
        metric: 目标指标名（trainer.callback_metrics 的 key），默认 "val_accuracy"
    """

    active_stages = {"train"}  # P0-1: 仅在 stage_train 激活

    def __init__(self, trial: Any = None, metric: str = "val_accuracy"):
        super().__init__()
        self.trial = trial
        self.metric = metric

    def on_validation_epoch_end(self, trainer, pl_module):
        # P0-1: 仅在 stage_train 激活
        if not self.is_active():
            return
        if self.trial is None:
            return
        # 跳过 sanity check（sanity check 阶段无有效指标）
        if trainer.sanity_checking:
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
        try:
            # P5 P3-14：1-indexed step，与 training_log/CSV/IntermediateMetricLogger 的 epoch 对齐
            self.trial.report(value, step=trainer.current_epoch + 1)
        except Exception as e:
            # trial.report 异常不应阻断训练，降级为 warning
            logger.warning(
                "OptunaReportingCallback: trial.report failed for "
                "metric='%s' at epoch=%d: %s",
                self.metric, trainer.current_epoch + 1, e,
            )


def run_experiment(config: ExperimentConfig) -> TrainOutput:
    """声明式实验入口（薄适配器，委托 run_pipeline）。

    向后兼容：保留原入口签名，内部委托 Pipeline.run()。
    所有执行逻辑统一由 Pipeline.run() 提供，消除双执行路径的 feature parity 风险。
    """
    from .pipeline import run_pipeline
    return run_pipeline(config)
