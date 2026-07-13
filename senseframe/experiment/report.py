"""RFC-003 ε6 ComparisonReport — 对比实验报告 + 显著性检验。

聚合 TrialResult 列表，生成对比报告：
- GroupSummary: 每组的聚合统计（mean/std/n_trials）
- SignificanceTest: Method vs Baseline 的显著性检验（t-test / bootstrap）
- ComparisonReport: 完整报告（含序列化）

显著性检验策略：
- scipy 可用时：用 scipy.stats.ttest_ind（参数化）+ Cohen's d（效应量）
- scipy 不可用时：降级为 bootstrap（非参数化）
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import TrialGroup, TrialResult, TrialStatus

logger = logging.getLogger(__name__)

# scipy 可用性检测
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ============================================================
# 聚合统计
# ============================================================
@dataclass
class GroupSummary:
    """单组聚合统计。"""
    group: str  # TrialGroup.value
    method_name: str
    n_trials: int = 0
    n_success: int = 0
    n_failed: int = 0
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    mean_wall_time_s: float = 0.0
    mean_agent_decisions: int = 0
    mean_manual_tunes: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _cohen_d(a: List[float], b: List[float]) -> float:
    """Cohen's d 效应量。

    d = (mean_a - mean_b) / pooled_std
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = _mean(a), _mean(b)
    var_a = sum((v - mean_a) ** 2 for v in a) / (len(a) - 1)
    var_b = sum((v - mean_b) ** 2 for v in b) / (len(b) - 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def _bootstrap_pvalue(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """Bootstrap 置换检验（scipy 不可用时降级用）。

    H0: a 和 b 来自同一分布
    统计量: |mean_a - mean_b|
    """
    if len(a) < 2 or len(b) < 2:
        return 1.0
    observed = abs(_mean(a) - _mean(b))
    combined = a + b
    n_a = len(a)
    rng = random.Random(seed)

    count_extreme = 0
    for _ in range(n_bootstrap):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        perm_diff = abs(_mean(perm_a) - _mean(perm_b))
        if perm_diff >= observed:
            count_extreme += 1

    return count_extreme / n_bootstrap


# ============================================================
# 显著性检验
# ============================================================
@dataclass
class SignificanceTest:
    """两组显著性检验结果。"""
    pair: str  # 如 "method_vs_baseline_repro"
    metric: str
    method_values: List[float] = field(default_factory=list)
    baseline_values: List[float] = field(default_factory=list)
    p_value: float = 1.0
    effect_size: float = 0.0  # Cohen's d
    significant: bool = False  # p < 0.05
    test_method: str = "none"  # "ttest" / "bootstrap"
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_significance_test(
    pair: str,
    metric: str,
    method_values: List[float],
    baseline_values: List[float],
    alpha: float = 0.05,
) -> SignificanceTest:
    """执行显著性检验。

    优先用 scipy.stats.ttest_ind，scipy 不可用时降级为 bootstrap。
    """
    if len(method_values) < 2 or len(baseline_values) < 2:
        return SignificanceTest(
            pair=pair, metric=metric,
            method_values=method_values, baseline_values=baseline_values,
            test_method="insufficient_samples",
        )

    effect_size = _cohen_d(method_values, baseline_values)

    if _SCIPY_AVAILABLE:
        try:
            _, p_value = _scipy_stats.ttest_ind(method_values, baseline_values)
            p_value = float(p_value)
            test_method = "ttest"
        except Exception:
            p_value = _bootstrap_pvalue(method_values, baseline_values)
            test_method = "bootstrap"
    else:
        p_value = _bootstrap_pvalue(method_values, baseline_values)
        test_method = "bootstrap"

    return SignificanceTest(
        pair=pair,
        metric=metric,
        method_values=method_values,
        baseline_values=baseline_values,
        p_value=p_value,
        effect_size=effect_size,
        significant=p_value < alpha,
        test_method=test_method,
        alpha=alpha,
    )


# ============================================================
# ComparisonReport
# ============================================================
@dataclass
class ComparisonReport:
    """对比实验报告。"""
    experiment_id: str
    design_name: str
    results: List[TrialResult] = field(default_factory=list)
    summary: Dict[str, GroupSummary] = field(default_factory=dict)  # key: group_value
    significance: Dict[str, SignificanceTest] = field(default_factory=dict)  # key: pair_name
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "design_name": self.design_name,
            "results": [r.to_dict() for r in self.results],
            "summary": {k: v.to_dict() for k, v in self.summary.items()},
            "significance": {k: v.to_dict() for k, v in self.significance.items()},
            "generated_at": self.generated_at,
        }

    def save(self, path: Path) -> None:
        """保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("ComparisonReport saved to %s", path)

    def build_summary(self, results: List[TrialResult]) -> Dict[str, GroupSummary]:
        """从 results 聚合 GroupSummary（按 group 分组）。"""
        groups: Dict[str, List[TrialResult]] = {}
        for r in results:
            groups.setdefault(r.group.value, []).append(r)

        summary: Dict[str, GroupSummary] = {}
        for group_value, group_results in groups.items():
            success_results = [r for r in group_results if r.status == TrialStatus.SUCCESS]
            n_trials = len(group_results)
            n_success = len(success_results)
            n_failed = n_trials - n_success

            # 聚合 metrics（取所有 success 结果的均值/标准差）
            all_metric_names = set()
            for r in success_results:
                all_metric_names.update(r.metrics.keys())

            mean_metrics = {}
            std_metrics = {}
            for metric_name in all_metric_names:
                values = [r.metrics[metric_name] for r in success_results if metric_name in r.metrics]
                if values:
                    mean_metrics[metric_name] = _mean(values)
                    std_metrics[metric_name] = _std(values)

            wall_times = [r.wall_time_s for r in success_results]
            agent_decisions = [r.agent_decisions for r in success_results]
            manual_tunes_list = [r.manual_tunes for r in success_results if r.manual_tunes is not None]

            summary[group_value] = GroupSummary(
                group=group_value,
                method_name=success_results[0].method_name if success_results else "",
                n_trials=n_trials,
                n_success=n_success,
                n_failed=n_failed,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                mean_wall_time_s=_mean(wall_times),
                mean_agent_decisions=int(_mean(agent_decisions)) if agent_decisions else 0,
                mean_manual_tunes=_mean(manual_tunes_list) if manual_tunes_list else None,
            )

        return summary

    def build_significance(
        self,
        results: List[TrialResult],
        metric: str,
        alpha: float = 0.05,
    ) -> Dict[str, SignificanceTest]:
        """对 Method vs 每个 Baseline 做显著性检验。"""
        method_results = [r for r in results
                          if r.group == TrialGroup.METHOD
                          and r.status == TrialStatus.SUCCESS
                          and metric in r.metrics]
        if not method_results:
            return {}

        method_values = [r.metrics[metric] for r in method_results]
        significance: Dict[str, SignificanceTest] = {}

        for baseline_group in (TrialGroup.BASELINE_PAPER, TrialGroup.BASELINE_REPRO):
            baseline_results = [r for r in results
                                if r.group == baseline_group
                                and r.status == TrialStatus.SUCCESS
                                and metric in r.metrics]
            if not baseline_results:
                continue
            baseline_values = [r.metrics[metric] for r in baseline_results]

            pair_name = f"method_vs_{baseline_group.value}"
            significance[pair_name] = run_significance_test(
                pair=pair_name,
                metric=metric,
                method_values=method_values,
                baseline_values=baseline_values,
                alpha=alpha,
            )

        return significance


__all__ = [
    "GroupSummary",
    "SignificanceTest",
    "ComparisonReport",
    "run_significance_test",
]
