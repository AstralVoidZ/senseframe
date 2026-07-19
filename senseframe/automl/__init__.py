"""RFC-003 ε1-ε6：AutoML 机制应用层（基于 SP/OP 协议栈）。

P1 已实现：
- ε1 损失函数搜索（SP 驱动）：loss_search 模块

P3 新增：
- ε4 元学习（SP + skills 整合）：meta_learner 模块（MetaLearner + HistoryStore）
- P3 阶段 8 PEFT 微调策略搜索（SP + PEFTBuilder）：peft_builder / peft_search 模块

P2~P3 规划：
- ε2 NAS（SP + ArchitectureBuilder）
- ε3 AutoAugment（SP + 增强搜索空间）
- ε5 Multi-fidelity 早停（SP Sampler 扩展）
"""
from .loss_search import (
    build_loss_search_space,
    run_loss_search,
    LossSearchResult,
)
from .meta_learner import MetaLearner
from .peft_builder import (
    PEFTBuilder,
    PEFTModel,
    LoRALayer,
    AdapterLayer,
    PrefixTuningLayer,
    PromptTuningLayer,
)
from .peft_search import (
    build_peft_search_space,
    run_peft_search,
    PEFTSearchResult,
)
from ..exploration import HistoryStore

__all__ = [
    "build_loss_search_space",
    "run_loss_search",
    "LossSearchResult",
    # P3.2: ε4 元学习
    "MetaLearner",
    "HistoryStore",
    # P3 阶段 8: PEFT 微调策略搜索
    "PEFTBuilder",
    "PEFTModel",
    "LoRALayer",
    "AdapterLayer",
    "PrefixTuningLayer",
    "PromptTuningLayer",
    "build_peft_search_space",
    "run_peft_search",
    "PEFTSearchResult",
]
