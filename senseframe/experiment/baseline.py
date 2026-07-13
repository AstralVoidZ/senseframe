"""RFC-003 ε6 BaselineRunner — Baseline 组运行器（固定参数，不走 SP）。

Baseline 组是固定参数的对比基准：
- BASELINE_REPRO：用官方代码 + 固定参数跑 run_pipeline
- BASELINE_PAPER：直接引用论文报告的指标，不跑训练

与 MethodRunner 的区别：
- 不走 SP ask/tell（无搜索）
- manual_tunes 记录人工调参次数（对比 Method 的 agent_decisions）
- BASELINE_PAPER 跳过训练，直接用 reported_metrics
"""
from __future__ import annotations

import logging
from typing import Optional

from ..engine.runner.pipeline import run_pipeline
from .design import BaselineConfig
from .method import (
    _build_config_snapshot,
    _extract_efficiency_from_train_output,
    _extract_metrics_from_train_output,
)
from .types import TrialGroup, TrialResult, TrialStatus

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Baseline 组运行器（ε6）。

    每次调用 run() 执行一次 Baseline 试验：
    - BASELINE_REPRO: run_pipeline(固定 config) → TrainOutput → TrialResult
    - BASELINE_PAPER: 直接用 reported_metrics 构造 TrialResult（跳过训练）

    Attributes:
        config: BaselineConfig 实例
        experiment_id: 所属实验 ID
    """

    def __init__(self, config: BaselineConfig, experiment_id: str = ""):
        self.config = config
        self.experiment_id = experiment_id

    def run(self, dataset: str, model_id: str, run_idx: int) -> TrialResult:
        """执行一次 Baseline 试验。

        Args:
            dataset: 数据集名（覆盖 base_config.scene.dataset）
            model_id: 模型 ID（覆盖 base_config.scene.model_id）
            run_idx: 试验序号

        Returns:
            TrialResult（DSP 合规）
        """
        # BASELINE_PAPER: 直接用论文报告的指标，不跑训练
        if self.config.group == TrialGroup.BASELINE_PAPER:
            return TrialResult(
                experiment_id=self.experiment_id,
                group=TrialGroup.BASELINE_PAPER,
                method_name=self.config.name,
                dataset=dataset,
                model_id=model_id,
                run_index=run_idx,
                metrics=dict(self.config.reported_metrics),
                manual_tunes=self.config.manual_tunes,
                config_snapshot=_build_config_snapshot(self.config.base_config),
                status=TrialStatus.SUCCESS,
            )

        # BASELINE_REPRO: 跑 run_pipeline（固定参数，不走 SP）
        # 资源泄露修复：深拷贝 config，避免多个 baseline run 共享同一 base_config
        # 导致后一个 run 看到前一个 run 修改的 dataset/model_id（跨试验污染）
        # 对齐 MethodRunner.run() 用 apply_params 做 deepcopy 的语义
        import copy as _copy
        modified_config = _copy.deepcopy(self.config.base_config)
        modified_config.scene.dataset = dataset
        modified_config.scene.model_id = model_id

        try:
            train_output = run_pipeline(modified_config)
        except Exception as e:
            logger.warning(
                "Baseline trial exception: name=%s, error=%s",
                self.config.name, e,
            )
            return TrialResult(
                experiment_id=self.experiment_id,
                group=TrialGroup.BASELINE_REPRO,
                method_name=self.config.name,
                dataset=dataset,
                model_id=model_id,
                run_index=run_idx,
                status=TrialStatus.FAILED,
                error_msg=str(e),
                manual_tunes=self.config.manual_tunes,
                config_snapshot=_build_config_snapshot(modified_config),
            )

        if train_output.status == "success":
            status = TrialStatus.SUCCESS
            error_msg = None
        else:
            status = TrialStatus.FAILED
            error_msg = train_output.error

        metrics = _extract_metrics_from_train_output(train_output)
        efficiency = _extract_efficiency_from_train_output(train_output)

        return TrialResult(
            experiment_id=self.experiment_id,
            group=TrialGroup.BASELINE_REPRO,
            method_name=self.config.name,
            dataset=dataset,
            model_id=model_id,
            run_index=run_idx,
            metrics=metrics,
            best_model_path=train_output.model_path,
            wall_time_s=efficiency["wall_time_s"],
            n_epochs_trained=efficiency["n_epochs_trained"],
            manual_tunes=self.config.manual_tunes,
            config_snapshot=_build_config_snapshot(modified_config),
            artifact_manifest_path=(
                f"{train_output.output_dir}/artifact_manifest.json"
                if train_output.output_dir else None
            ),
            status=status,
            error_msg=error_msg,
        )


__all__ = ["BaselineRunner"]
