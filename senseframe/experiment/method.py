"""RFC-003 ε6 MethodRunner — Method 组运行器（SP 驱动）。

Method 组通过 SP（search_protocol）的 ask/tell 驱动搜索：
1. sm.ask(study_id) → trial（SP 采样参数）
2. apply_params(config, trial.params) → modified_config
3. run_pipeline(modified_config) → TrainOutput
4. sm.tell(trial.trial_id, value, state) → 报告结果
5. 构造 TrialResult（DSP 合规）

过渡形态：直接调 run_pipeline，不走向 OP create_run（OP 完整迁移推迟到 P2）。

P2.4：支持 Multi-fidelity 早停（ε5）— 可选 Pruner 注入，
训练后检查 should_prune，若 True 则标记 trial 为 pruned。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..engine.hpo import apply_params, extract_metric
from ..engine.runner.pipeline import run_pipeline
from ..search_protocol import Pruner, SearchSpace, StudyManager, get_study_manager
from .design import MethodConfig
from .types import TrialGroup, TrialResult, TrialStatus

logger = logging.getLogger(__name__)


def _extract_metrics_from_train_output(train_output) -> Dict[str, float]:
    """从 TrainOutput 提取指标字典（用于 TrialResult.metrics）。

    优先取 final_eval 中的数值字段，回退到 training dict。
    """
    metrics: Dict[str, float] = {}
    for k, v in (train_output.final_eval or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics[k] = float(v)
    # 补充 training dict 中的数值字段（如 best_val_loss）
    for k, v in (train_output.training or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in metrics:
            metrics[k] = float(v)
    return metrics


def _extract_efficiency_from_train_output(train_output) -> Dict[str, Any]:
    """从 TrainOutput 提取效率指标（wall_time_s / n_epochs_trained）。"""
    training = train_output.training or {}
    return {
        "wall_time_s": float(training.get("duration_s", 0.0)),
        "n_epochs_trained": int(training.get("epochs_trained", 0)),
    }


def _build_config_snapshot(base_config, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造配置快照（用于 TrialResult.config_snapshot）。

    保留可序列化的关键字段，避免完整 ExperimentConfig 序列化（含工厂对象）。
    """
    snapshot: Dict[str, Any] = {
        "scene": {
            "name": base_config.scene.name,
            "model_id": base_config.scene.model_id,
            "dataset": base_config.scene.dataset,
            "learning_mode": base_config.scene.learning_mode,
        },
        "trainer": {
            "max_epochs": getattr(base_config.trainer, "max_epochs", None),
            "learning_rate": getattr(base_config.trainer, "learning_rate", None),
            "batch_size": getattr(base_config.trainer, "batch_size", None),
            "optimizer": getattr(base_config.trainer, "optimizer", None),
        },
    }
    if params:
        snapshot["applied_params"] = params
    return snapshot


class MethodRunner:
    """Method 组运行器：通过 SP ask/tell 驱动搜索（ε6）。

    每次调用 run() 执行一次 SP 试验：
    ask → apply_params → run_pipeline → tell → TrialResult

    Attributes:
        config: MethodConfig 实例
        study_id: SP Study ID
        study_manager: StudyManager 实例（None 时用全局单例）
        experiment_id: 所属实验 ID（用于 TrialResult.experiment_id）
    """

    def __init__(
        self,
        config: MethodConfig,
        study_id: str,
        study_manager: Optional[StudyManager] = None,
        experiment_id: str = "",
        pruner: Optional[Pruner] = None,
        use_op: bool = False,
        orchestrator: Optional[Any] = None,
    ):
        self.config = config
        self.study_id = study_id
        self._sm = study_manager
        self.experiment_id = experiment_id
        self._agent_decisions = 0  # SP ask 次数计数
        # P2.4: ε5 Multi-fidelity 早停 — 可选 Pruner 注入
        # 训练后检查 should_prune，若 True 则标记 trial 为 pruned
        self.pruner = pruner
        # P2.12: OP 迁移 — 可选通过 Orchestrator 编排 run_pipeline
        # use_op=False（默认）直接调 run_pipeline（P1 行为，向后兼容）
        # use_op=True 时通过 OP create_run + start + reconcile + complete/fail 包装
        self.use_op = use_op
        self._orchestrator = orchestrator  # None 时用全局单例

    @property
    def sm(self) -> StudyManager:
        return self._sm or get_study_manager()

    @property
    def orchestrator(self):
        """获取 Orchestrator 实例（None 时用全局单例，P2.12）。"""
        if self._orchestrator is not None:
            return self._orchestrator
        from ..orchestration import get_orchestrator
        return get_orchestrator()

    def _run_pipeline_via_op(
        self,
        modified_config,
        trial_params: Dict[str, Any],
    ):
        """通过 OP 编排 run_pipeline（P2.12，OP 迁移路径）。

        与直接 run_pipeline 的区别：
        - create_run + start + complete/fail 包装，发射 CloudEvent
        - ExperimentRunner 可订阅 OP 事件实现事件驱动聚合（P2.13）
        - 失败时 run.error 持久化到 PipelineRun

        P2.12 渐进式迁移：run_pipeline 仍在主线程执行（同步语义），
        OP 仅作状态跟踪 + 事件发射层。完全异步执行推迟到 P3（需解决 GIL + 资源隔离）。

        Args:
            modified_config: 应用 trial.params 后的 ExperimentConfig
            trial_params: SP Trial 参数（用于 OP create_run.params 持久化）

        Returns:
            TrainOutput（run_pipeline 的返回值）
        """
        from ..orchestration import PipelineDef, PHASE_SUCCEEDED, PHASE_FAILED

        orch = self.orchestrator
        pdef = PipelineDef(name=f"method_{self.experiment_id or 'default'}")
        pipeline_id = orch.create_pipeline(pdef)
        run_id = orch.create_run(pipeline_id, params=trial_params)

        # start 触发 PHASE_RUNNING + emit EVENT_PIPELINE_STARTED
        orch.start(run_id)

        train_output = None
        error: Optional[Exception] = None
        try:
            train_output = run_pipeline(modified_config)
        except Exception as e:
            error = e

        # 根据 train_output 状态回写 OP 状态
        if train_output is not None and train_output.status == "success" and error is None:
            orch.complete(run_id, output_uri=str(train_output.output_dir or ""))
        else:
            err_msg = str(error) if error is not None else (
                train_output.error if train_output is not None else "unknown error"
            )
            orch.fail(run_id, error=err_msg)

        # 若有异常，重新抛出（保持与直接 run_pipeline 一致的异常语义）
        if error is not None:
            raise error
        return train_output

    def run(self, dataset: str, model_id: str, run_idx: int) -> TrialResult:
        """执行一次 Method 试验（SP 驱动）。

        Args:
            dataset: 数据集名（覆盖 base_config.scene.dataset）
            model_id: 模型 ID（覆盖 base_config.scene.model_id）
            run_idx: 试验序号（用于 TrialResult.run_index）

        Returns:
            TrialResult（DSP 合规）
        """
        # 1. SP ask
        trial = self.sm.ask(self.study_id)
        self._agent_decisions += 1

        # 2. 应用参数 + 覆盖 dataset/model_id
        modified_config = apply_params(self.config.base_config, trial.params)
        modified_config.scene.dataset = dataset
        modified_config.scene.model_id = model_id

        # 3. 训练
        try:
            if self.use_op:
                # P2.12: OP 迁移路径 — create_run + start + complete/fail 包装
                train_output = self._run_pipeline_via_op(modified_config, trial.params)
            else:
                # P1 路径：直接调 run_pipeline（向后兼容）
                train_output = run_pipeline(modified_config)
        except Exception as e:
            # 训练异常 → SP tell failed + 返回失败 TrialResult
            self.sm.tell(
                trial.trial_id,
                value=0.0,
                state="failed",
                feedback={"error": str(e)},
            )
            logger.warning(
                "Method trial exception: loss=%s, error=%s",
                trial.params.get("loss"), e,
            )
            return TrialResult(
                experiment_id=self.experiment_id,
                group=TrialGroup.METHOD,
                method_name=self.config.name,
                dataset=dataset,
                model_id=model_id,
                run_index=run_idx,
                sp_trial_id=trial.trial_id,
                status=TrialStatus.FAILED,
                error_msg=str(e),
                agent_decisions=self._agent_decisions,
                config_snapshot=_build_config_snapshot(modified_config, trial.params),
            )

        # 4. SP tell
        if train_output.status == "success":
            try:
                value = extract_metric(train_output, self.config.metric)
            except ValueError as e:
                # 指标缺失 → 标记失败
                self.sm.tell(
                    trial.trial_id,
                    value=0.0,
                    state="failed",
                    feedback={"error": f"metric extraction failed: {e}"},
                )
                return TrialResult(
                    experiment_id=self.experiment_id,
                    group=TrialGroup.METHOD,
                    method_name=self.config.name,
                    dataset=dataset,
                    model_id=model_id,
                    run_index=run_idx,
                    sp_trial_id=trial.trial_id,
                    status=TrialStatus.FAILED,
                    error_msg=f"metric extraction failed: {e}",
                    agent_decisions=self._agent_decisions,
                )

            # P2.4: ε5 Multi-fidelity 早停检查
            # 从 TrainOutput.training 提取 epoch 级中间值
            intermediate_values: Dict[int, float] = (
                train_output.training.get("intermediate_values", {})
                if train_output.training
                else {}
            )

            should_prune = False
            if self.pruner is not None and intermediate_values:
                # rung = 最后一个 epoch（已收集到的最高 fidelity）
                rung = max(intermediate_values.keys())
                try:
                    should_prune = self.pruner.should_prune(
                        trial.trial_id, intermediate_values, rung,
                    )
                except Exception as e:
                    # Pruner 异常不应阻断试验完成，降级为不剪枝
                    logger.warning("Pruner should_prune raised, skipping prune: %s", e)
                    should_prune = False

            if should_prune:
                # 剪枝：标记 trial 为 pruned，value 仍上报（供 SP 记录）
                self.sm.tell(
                    trial.trial_id,
                    value=value,
                    intermediate_values=intermediate_values,
                    state="pruned",
                    feedback={
                        "model_path": train_output.model_path,
                        "final_eval": train_output.final_eval,
                        "pruned": True,
                    },
                )
                status = TrialStatus.PRUNED
                error_msg = None
            else:
                self.sm.tell(
                    trial.trial_id,
                    value=value,
                    intermediate_values=intermediate_values if intermediate_values else None,
                    state="completed",
                    feedback={
                        "model_path": train_output.model_path,
                        "final_eval": train_output.final_eval,
                    },
                )
                status = TrialStatus.SUCCESS
                error_msg = None
        else:
            self.sm.tell(
                trial.trial_id,
                value=0.0,
                state="failed",
                feedback={
                    "error": train_output.error,
                    "error_code": train_output.error_code,
                },
            )
            status = TrialStatus.FAILED
            error_msg = train_output.error

        # 5. 构造 TrialResult（DSP 合规）
        metrics = _extract_metrics_from_train_output(train_output)
        efficiency = _extract_efficiency_from_train_output(train_output)

        return TrialResult(
            experiment_id=self.experiment_id,
            group=TrialGroup.METHOD,
            method_name=self.config.name,
            dataset=dataset,
            model_id=model_id,
            run_index=run_idx,
            sp_trial_id=trial.trial_id,
            metrics=metrics,
            best_model_path=train_output.model_path,
            wall_time_s=efficiency["wall_time_s"],
            n_epochs_trained=efficiency["n_epochs_trained"],
            agent_decisions=self._agent_decisions,
            config_snapshot=_build_config_snapshot(modified_config, trial.params),
            artifact_manifest_path=(
                f"{train_output.output_dir}/artifact_manifest.json"
                if train_output.output_dir else None
            ),
            status=status,
            error_msg=error_msg,
        )


__all__ = ["MethodRunner"]
