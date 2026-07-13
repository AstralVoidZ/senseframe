"""RFC-003 ε4 元学习：跨数据集迁移搜索经验（P3.2.3）。

P3.2.3 MetaLearner 设计核心：
基于 ExplorationTracker 历史加载 + Sampler warm_start。

机制：
1. 从 HistoryStore 加载源数据集历史（如 UT_HAR_data 的成功策略）
2. 按 success_threshold 过滤出成功策略（val_accuracy > 阈值，
   或 result.value > 阈值作为 fallback）
3. 将成功策略注入目标 Study 的 tracker.history
4. 后续 StudyManager.ask() 调用 sampler.sample(search_space, history)
   时，sampler 自然从扩展后的 history 中读取成功策略作为采样偏向

关键设计决策（与规划文档 4.2 节伪代码的差异）：
因为 StudyManager.ask() 每次 ask 都创建新的 sampler 实例
（`sampler = sampler_cls()`），实例级 warm_start 状态无法跨 ask 保留。
所以 MetaLearner.warm_start 的核心机制是**将源数据集成功策略
注入 tracker.history**，让后续 sampler.sample() 调用自然从扩展后的
history 中读取。这是最简单且与现有架构对齐的设计。

可选优化：若 Sampler 类实现 warm_start 方法（如 EvolutionarySampler
和 AutoAugmentSampler），则额外调用一次——但因为是临时实例，
仅用于触发可能的副作用（如类级状态）。主要机制仍是 history 注入。
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..exploration import ExplorationTracker, HistoryStore
from ..search_protocol import StudyManager, get_sampler


class MetaLearner:
    """元学习：跨数据集迁移搜索经验（P3.2.3）。

    基于 ExplorationTracker 历史加载 + Sampler warm_start。

    Usage:
        from senseframe.automl import MetaLearner, HistoryStore
        from senseframe.search_protocol import StudyManager

        sm = StudyManager()
        store = HistoryStore(base_dir=Path(tempfile.gettempdir()) / "sf_history")

        # 创建新 study 并从源数据集 warm-start
        study_id = sm.create_study(
            name="target",
            sampler="evolutionary",
            search_space=ss,
            warm_start_from="UT_HAR_data",
            history_store=store,
        )
        # 此时 tracker.history 已被注入源数据集成功策略
    """

    def __init__(self, study_manager: StudyManager, history_store: HistoryStore):
        """初始化 MetaLearner。

        Args:
            study_manager: StudyManager 实例（用于访问 _trackers）
            history_store: HistoryStore 实例（用于加载源数据集历史）
        """
        self.sm = study_manager
        self.store = history_store

    def warm_start(
        self,
        study_id: str,
        source_dataset: str,
        success_threshold: float = 0.7,
    ) -> int:
        """从源数据集历史 warm-start 新 study。

        将源数据集中满足 success_threshold 的成功策略注入到目标 study
        的 tracker.history，让后续 sampler.sample() 自然从扩展后的
        history 中读取作为采样偏向。

        Args:
            study_id: 目标 study ID
            source_dataset: 源数据集名（如 "UT_HAR_data"）
            success_threshold: 成功策略的 val_accuracy 阈值（默认 0.7）。
                若 trial.result 含 val_accuracy 则用之；否则 fallback 到
                result.value 与同一阈值比较。

        Returns:
            注入的历史条目数（0 表示源数据集不存在或无成功策略）

        Raises:
            KeyError: study_id 不存在
        """
        source_history = self.store.load_history(dataset=source_dataset)
        if not source_history:
            return 0

        # 提取成功策略作为初始采样偏向
        # 优先用 val_accuracy，fallback 到 value（SP Tell 上报的指标）
        successful: List[Dict[str, Any]] = []
        for trial in source_history:
            result = trial.get("result") or {}
            if not isinstance(result, dict):
                continue
            val_acc = result.get("val_accuracy")
            if val_acc is None:
                value = result.get("value")
                if value is None:
                    continue
                if float(value) > success_threshold:
                    successful.append(trial)
            else:
                if float(val_acc) > success_threshold:
                    successful.append(trial)

        if not successful:
            return 0

        # 注入到 tracker（作为采样偏向，扩展 history）
        tracker = self.sm._trackers.get(study_id)
        if tracker is None:
            raise KeyError(f"Study '{study_id}' not found")
        tracker.history.extend(successful)

        # 可选优化：调用 Sampler 类的 warm_start（如存在）
        # 注：StudyManager.ask 每次创建新 sampler 实例，warm_start 注入
        # tracker.history 即已足够（sampler 会从 history 中读取成功策略）。
        # 但若 sampler 类自身维护跨实例状态（如类变量），可在此调用类
        # 方法 warm_start。实例级 warm_start 在 ask() 内部由
        # sample(search_space, history) 自然完成。
        study = self.sm.get_study(study_id)
        if study is not None:
            sampler_cls = get_sampler(study.sampler)
            if sampler_cls is not None and hasattr(sampler_cls, "warm_start"):
                # 创建临时实例调用 warm_start（用于可能的类级状态副作用）
                try:
                    temp_sampler = sampler_cls()
                    if hasattr(temp_sampler, "warm_start"):
                        temp_sampler.warm_start(successful)
                except Exception:
                    # warm_start 是可选优化，失败不影响主流程
                    pass

        return len(successful)


__all__ = ["MetaLearner"]
