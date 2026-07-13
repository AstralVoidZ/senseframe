"""RFC-003 ε6 experiment 模块 — 对比实验基础设施（过渡形态，DSP 合规）。

定位：L5 OP 之上的应用层，通过 SP + OP 驱动对比实验。

核心组件：
- TrialResult（DSP 合规）：schema_version + schema() + describe()
- ExperimentDesign：对比实验设计（datasets × models × Method/Baseline）
- MethodRunner：Method 组运行器（SP ask/tell 驱动）
- BaselineRunner：Baseline 组运行器（固定参数）
- ExperimentRunner：对比实验编排器
- ComparisonReport：对比报告（含显著性检验）

过渡形态边界（RFC-003 融合审查第 11.4 节）：
- 数据结构先合规（TrialResult DSP 合规）✅
- Method 走 SP ask/tell（P0 SP 已可用）✅
- Baseline 走 run_pipeline（OP create_run 推迟到 P2）
- 串行执行（异步 + 事件驱动推迟到 P2）
- 效率指标从 TrainOutput 提取（从 OBP 查询推迟到 P2）
"""
from .baseline import BaselineRunner
from .design import (
    BaselineConfig,
    ExperimentBudget,
    ExperimentDesign,
    MethodConfig,
)
from .method import MethodRunner
from .report import (
    ComparisonReport,
    GroupSummary,
    SignificanceTest,
    run_significance_test,
)
from .runner import ExperimentRunner
from .types import TrialGroup, TrialResult, TrialStatus

__all__ = [
    # types
    "TrialGroup",
    "TrialStatus",
    "TrialResult",
    # design
    "ExperimentBudget",
    "MethodConfig",
    "BaselineConfig",
    "ExperimentDesign",
    # runners
    "MethodRunner",
    "BaselineRunner",
    "ExperimentRunner",
    # report
    "ComparisonReport",
    "GroupSummary",
    "SignificanceTest",
    "run_significance_test",
]
