"""RFC-003 ε6 ExperimentRunner — 对比实验编排器（L5 OP 之上的应用层）。

编排流程：
1. 创建 SP Study（Method 组的搜索空间）
2. 串行执行所有试验（P2.13 事件驱动过渡形态）
   - Method 组：SP ask → run_pipeline → SP tell（P2.12 可选 OP 包装）
   - Baseline 组：run_pipeline（固定参数）/ reported_metrics（论文）
3. 聚合结果 → ComparisonReport（含显著性检验）

P2.13 事件驱动聚合：
- use_op=True 时订阅 OP CloudEvent（pipeline.succeeded / pipeline.failed）
- 事件回调收集 trial 完成事件（用于监控/日志/未来异步聚合）
- 主流程仍用 MethodRunner.run() 返回值聚合（同步语义）
- 完全异步事件驱动推迟到 P3（需解决 GIL + 资源隔离）

过渡形态边界：
- 数据结构先合规（TrialResult DSP 合规）
- Method 走 SP ask/tell（P0 SP 已可用）
- Baseline 走 run_pipeline（OP create_run 可选包装）
- 串行执行（异步 + 事件驱动推迟到 P3）
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        use_op: 是否通过 OP 编排试验（P2.13，默认 False 向后兼容）
        orchestrator: 可选 Orchestrator 实例（None 时用全局单例）
    """

    def __init__(
        self,
        design: ExperimentDesign,
        study_manager: Optional[StudyManager] = None,
        experiment_id: Optional[str] = None,
        use_op: bool = False,
        orchestrator: Optional[Any] = None,
    ):
        design.validate()
        self.design = design
        self._sm = study_manager
        self.experiment_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        # P2.13: 事件驱动聚合支持
        self.use_op = use_op
        self._orchestrator = orchestrator
        # 事件收集器（线程安全，用于事件驱动聚合）
        self._event_log: List[Dict[str, Any]] = []
        self._event_lock = threading.Lock()
        self._unsubscribe_callbacks: List[Any] = []

    @property
    def sm(self) -> StudyManager:
        return self._sm or get_study_manager()

    @property
    def orchestrator(self):
        """获取 Orchestrator 实例（None 时用全局单例，P2.13）。"""
        if self._orchestrator is not None:
            return self._orchestrator
        from ..orchestration import get_orchestrator
        return get_orchestrator()

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

    def _subscribe_op_events(self) -> None:
        """订阅 OP CloudEvent（P2.13，事件驱动聚合）。

        订阅 pipeline.succeeded / pipeline.failed 事件，
        在回调中收集事件到 _event_log（用于监控/日志/未来异步聚合）。

        主流程仍用 MethodRunner.run() 返回值聚合（同步语义），
        事件订阅不影响主流程，仅作监控层。
        """
        from ..orchestration import (
            EVENT_PIPELINE_SUCCEEDED,
            EVENT_PIPELINE_FAILED,
        )

        orch = self.orchestrator

        def _extract_run_id(event) -> str:
            """从 CloudEvent.source 提取 run_id（格式：/senseframe/pipeline/{run_id}）。"""
            source = getattr(event, "source", "") or ""
            return source.rsplit("/", 1)[-1] if source else ""

        def on_pipeline_succeeded(event):
            run_id = _extract_run_id(event)
            with self._event_lock:
                self._event_log.append({
                    "event_type": EVENT_PIPELINE_SUCCEEDED,
                    "run_id": run_id,
                    "timestamp": event.time,
                    "data": event.data,
                })
            logger.debug(
                "Experiment %s: pipeline.succeeded run_id=%s",
                self.experiment_id, run_id,
            )

        def on_pipeline_failed(event):
            run_id = _extract_run_id(event)
            with self._event_lock:
                self._event_log.append({
                    "event_type": EVENT_PIPELINE_FAILED,
                    "run_id": run_id,
                    "timestamp": event.time,
                    "data": event.data,
                })
            logger.debug(
                "Experiment %s: pipeline.failed run_id=%s error=%s",
                self.experiment_id, run_id,
                event.data.get("error", ""),
            )

        # 订阅并保存取消订阅函数
        self._unsubscribe_callbacks.append(
            orch.subscribe(EVENT_PIPELINE_SUCCEEDED, on_pipeline_succeeded)
        )
        self._unsubscribe_callbacks.append(
            orch.subscribe(EVENT_PIPELINE_FAILED, on_pipeline_failed)
        )

    def _unsubscribe_all(self) -> None:
        """取消所有 OP 事件订阅（在 run() 结束时调用）。"""
        for unsub in self._unsubscribe_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe_callbacks.clear()

    def get_event_log(self) -> List[Dict[str, Any]]:
        """获取收集的 OP 事件日志（P2.13，用于监控/调试）。"""
        with self._event_lock:
            return list(self._event_log)

    def run(self, output_path: Optional[str] = None) -> ComparisonReport:
        """执行对比实验。

        Args:
            output_path: 报告保存路径（None 时不保存）

        Returns:
            ComparisonReport
        """
        logger.info(
            "Experiment %s started: design=%s, datasets=%s, models=%s, "
            "method=%s, baselines=%d, budget=%s, use_op=%s",
            self.experiment_id, self.design.name,
            self.design.datasets, self.design.models,
            self.design.method.name, len(self.design.baselines),
            self.design.budget, self.use_op,
        )

        # P2.13: 若 use_op=True，订阅 OP 事件
        if self.use_op:
            self._subscribe_op_events()

        try:
            # 1. 创建 SP Study
            study_id = self._create_study()

            # 2. 创建 MethodRunner（P2.13: use_op 透传）
            method_runner = MethodRunner(
                config=self.design.method,
                study_id=study_id,
                study_manager=self._sm,
                experiment_id=self.experiment_id,
                use_op=self.use_op,
                orchestrator=self._orchestrator,
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
                "Experiment %s completed: %d/%d runs collected, events=%d",
                self.experiment_id, len(results), total_runs,
                len(self._event_log),
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
        finally:
            # P2.13: 无论成功失败，取消 OP 事件订阅
            if self.use_op:
                self._unsubscribe_all()


__all__ = ["ExperimentRunner"]
