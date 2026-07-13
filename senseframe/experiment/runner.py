"""RFC-003 ε6 ExperimentRunner — 对比实验编排器（L5 OP 之上的应用层）。

编排流程：
1. 创建 SP Study（Method 组的搜索空间）
2. 串行执行所有试验（过渡形态；P2 改异步 + 事件驱动）
   - Method 组：SP ask → run_pipeline → SP tell
   - Baseline 组：run_pipeline（固定参数）/ reported_metrics（论文）
3. 聚合结果 → ComparisonReport（含显著性检验）

过渡形态边界：
- 数据结构先合规（TrialResult DSP 合规）
- Method 走 SP ask/tell（P0 SP 已可用）
- Baseline 走 run_pipeline（OP create_run 作为状态记录推迟到 P2）
- 串行执行（异步 + 事件驱动推迟到 P2）
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..automl.loss_search import build_loss_search_space
from ..search_protocol import SearchSpace, StudyManager, get_study_manager
from .baseline import BaselineRunner
from .design import ExperimentDesign
from .method import MethodRunner
from .report import ComparisonReport
from .types import TrialGroup

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """对比实验编排器（ε6）。

    Attributes:
        design: ExperimentDesign 实例
        study_manager: StudyManager 实例（None 时用全局单例）
        experiment_id: 实验 ID（自动生成或显式指定）
    """

    def __init__(
        self,
        design: ExperimentDesign,
        study_manager: Optional[StudyManager] = None,
        experiment_id: Optional[str] = None,
    ):
        design.validate()
        self.design = design
        self._sm = study_manager
        self.experiment_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"

    @property
    def sm(self) -> StudyManager:
        return self._sm or get_study_manager()

    def _create_study(self) -> str:
        """创建 SP Study（Method 组的搜索空间）。"""
        search_space = self.design.method.search_space
        if search_space is None:
            # 默认用损失搜索空间（ε1）
            search_space = build_loss_search_space()

        return self.sm.create_study(
            name=f"{self.design.name}_method",
            direction=self.design.method.direction,
            search_space=search_space,
            sampler=self.design.method.sampler,
        )

    def run(self, output_path: Optional[str] = None) -> ComparisonReport:
        """执行对比实验。

        Args:
            output_path: 报告保存路径（None 时不保存）

        Returns:
            ComparisonReport
        """
        logger.info(
            "Experiment %s started: design=%s, datasets=%s, models=%s, "
            "method=%s, baselines=%d, budget=%s",
            self.experiment_id, self.design.name,
            self.design.datasets, self.design.models,
            self.design.method.name, len(self.design.baselines),
            self.design.budget,
        )

        # 1. 创建 SP Study
        study_id = self._create_study()

        # 2. 创建 MethodRunner
        method_runner = MethodRunner(
            config=self.design.method,
            study_id=study_id,
            study_manager=self._sm,
            experiment_id=self.experiment_id,
        )

        # 3. 创建 BaselineRunner 列表
        baseline_runners = [
            BaselineRunner(config=b, experiment_id=self.experiment_id)
            for b in self.design.baselines
        ]

        # 4. 串行执行所有试验
        # Method 组：max_trials_per_group 控制 SP ask 次数（搜索预算）
        # Baseline 组：n_repeats 控制固定参数重复次数（用于显著性检验）
        results = []
        total_runs = (
            len(self.design.datasets) * len(self.design.models)
            * (self.design.budget.max_trials_per_group
               + len(self.design.baselines) * self.design.budget.n_repeats)
        )
        run_idx = 0

        for dataset in self.design.datasets:
            for model_id in self.design.models:
                # Method 组：max_trials_per_group 次 SP ask
                for method_idx in range(self.design.budget.max_trials_per_group):
                    results.append(method_runner.run(dataset, model_id, run_idx))
                    run_idx += 1

                # Baseline 组：每个 baseline 跑 n_repeats 次（固定参数重复）
                for baseline_runner in baseline_runners:
                    for repeat_idx in range(self.design.budget.n_repeats):
                        results.append(baseline_runner.run(dataset, model_id, run_idx))
                        run_idx += 1

        logger.info(
            "Experiment %s completed: %d/%d runs collected",
            self.experiment_id, len(results), total_runs,
        )

        # 5. 聚合 + 显著性检验
        report = ComparisonReport(
            experiment_id=self.experiment_id,
            design_name=self.design.name,
            results=results,
            generated_at=datetime.now().isoformat(),
        )
        report.summary = report.build_summary(results)
        report.significance = report.build_significance(
            results, metric=self.design.method.metric,
        )

        # 6. 保存报告（可选）
        if output_path:
            from pathlib import Path
            report.save(Path(output_path))

        return report


__all__ = ["ExperimentRunner"]
