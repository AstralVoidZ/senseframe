"""RFC-003 ε6 experiment 模块设计定义。

定义对比实验的设计时数据结构：
- ExperimentDesign: 实验整体设计（数据集 × 模型 × Method/Baseline 组 × 预算）
- MethodConfig: Method 组配置（SP 驱动搜索）
- BaselineConfig: Baseline 组配置（固定参数）
- ExperimentBudget: 实验预算（试验次数 / 重复次数）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..engine.config import ExperimentConfig
from ..search_protocol import SearchSpace
from .types import TrialGroup


# ============================================================
# 预算
# ============================================================
@dataclass
class ExperimentBudget:
    """实验预算。

    Args:
        max_trials_per_group: 每组最大试验次数（Method 组 = SP ask 次数）
        n_repeats: 每个配置重复次数（用于显著性检验，>=3 时可做 t-test）
    """
    max_trials_per_group: int = 10
    n_repeats: int = 3

    def validate(self) -> None:
        if self.max_trials_per_group <= 0:
            raise ValueError(f"max_trials_per_group must be > 0, got {self.max_trials_per_group}")
        if self.n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {self.n_repeats}")


# ============================================================
# Method 组配置
# ============================================================
@dataclass
class MethodConfig:
    """Method 组配置（SP 驱动搜索）。

    Method 组通过 SP ask/tell 驱动搜索：
    - search_space 决定搜什么（None 时用 build_loss_search_space()）
    - sampler 决定怎么搜（SP Sampler 名）
    - metric + direction 决定评估什么

    Attributes:
        name: Method 名称（如 "senseframe_loss_search"）
        base_config: 基础 ExperimentConfig（SP 参数会覆盖到 scene.params）
        metric: 评估指标名（如 "val_accuracy"）
        direction: "maximize" / "minimize"
        search_space: SP SearchSpace（None 时由 MethodRunner 自动构造）
        sampler: SP Sampler 名（"random" / "grid"）
    """
    name: str
    base_config: ExperimentConfig
    metric: str = "val_accuracy"
    direction: str = "maximize"
    search_space: Optional[SearchSpace] = None
    sampler: str = "random"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("MethodConfig.name must be non-empty")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize'/'minimize', got {self.direction}")


# ============================================================
# Baseline 组配置
# ============================================================
@dataclass
class BaselineConfig:
    """Baseline 组配置（固定参数，不走 SP）。

    Baseline 组是固定参数的对比基准：
    - 论文报告（BASELINE_PAPER）：直接引用论文数字
    - 代码复现（BASELINE_REPRO）：用官方代码 + 固定参数跑 run_pipeline

    Attributes:
        name: Baseline 名称（如 "sensefi_mlp" / "paper_mlp"）
        base_config: 固定的 ExperimentConfig（不走 SP，不修改）
        manual_tunes: 人工调参次数（对比 Method 的 agent_decisions）
        group: BASELINE_PAPER / BASELINE_REPRO
        reported_metrics: 论文报告的指标（BASELINE_PAPER 时用，跳过实际训练）
    """
    name: str
    base_config: ExperimentConfig
    manual_tunes: int = 0
    group: TrialGroup = TrialGroup.BASELINE_REPRO
    reported_metrics: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("BaselineConfig.name must be non-empty")
        if self.group == TrialGroup.METHOD:
            raise ValueError("BaselineConfig.group cannot be METHOD")


# ============================================================
# ExperimentDesign
# ============================================================
@dataclass
class ExperimentDesign:
    """对比实验设计（应用层）。

    定义对比实验的完整设计：
    - datasets × models：实验矩阵
    - method：Method 组配置（SP 驱动搜索）
    - baselines：Baseline 组列表（固定参数）
    - budget：试验预算

    Example:
        design = ExperimentDesign(
            name="loss_search_vs_sensefi",
            datasets=["UT_HAR_data", "NTU-Fi-HAR"],
            models=["mlp", "cnn1d"],
            method=MethodConfig(...),
            baselines=[BaselineConfig(...)],
            budget=ExperimentBudget(max_trials_per_group=10, n_repeats=3),
        )
    """
    name: str
    datasets: List[str]
    models: List[str]
    method: MethodConfig
    baselines: List[BaselineConfig] = field(default_factory=list)
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("ExperimentDesign.name must be non-empty")
        if not self.datasets:
            raise ValueError("ExperimentDesign.datasets must be non-empty")
        if not self.models:
            raise ValueError("ExperimentDesign.models must be non-empty")
        self.method.validate()
        for b in self.baselines:
            b.validate()
        self.budget.validate()


__all__ = [
    "ExperimentBudget",
    "MethodConfig",
    "BaselineConfig",
    "ExperimentDesign",
]
