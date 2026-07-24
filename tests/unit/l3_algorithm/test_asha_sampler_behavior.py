"""L3 算法行为测试：ASHA（Asynchronous Successive Halving Algorithm）。

锚点来源：Li et al., 2018《Massively Parallel Hyperparameter Tuning》
(https://arxiv.org/abs/1810.09734)

论文关键结论（作为测试锚点）：
- §3: ASHA 是异步连续减半算法，用于 Multi-fidelity 早停
- §3: 每个 rung 保留 top 1/η 的试验（η 是减半率，默认 3）
- §3: 试验在 rung r 达到阈值后才被提升到 rung r+1
- §3: 被剪枝的试验（未达到阈值）停止资源分配
"""
from __future__ import annotations

import pytest

from senseframe.search_protocol import (
    ASHASampler,
    ParameterSpec,
    Sampler,
    SearchSpace,
)


def _make_search_space() -> SearchSpace:
    """构造测试用搜索空间。"""
    return SearchSpace(parameters=[
        ParameterSpec(name="lr", type="float", low=0.001, high=0.1, log=True),
        ParameterSpec(name="hidden", type="int", low=16, high=128),
        ParameterSpec(name="opt", type="categorical", choices=["adam", "sgd"]),
    ])


@pytest.mark.l3_algorithm
class TestASHASamplerBehavior:
    """验证 ASHA 行为符合 Li et al., 2018。"""

    def test_satisfies_sp_sampler_protocol(self):
        """L3 anchor: ASHASampler 实现 SP Sampler Protocol。

        锚点：Li et al., 2018 §3 — ASHA 是一种采样策略，应满足
        搜索协议的 Sampler 契约（name + sample 方法），使其可被
        StudyManager 统一调度。
        """
        asha = ASHASampler()
        assert isinstance(asha, Sampler), \
            "ASHASampler 应满足 SP Sampler Protocol（统一采样调度前提）"
        assert asha.name == "asha"

    def test_sample_returns_valid_params(self):
        """L3 anchor: sample() 返回搜索空间内的有效参数字典。

        锚点：Li et al., 2018 §3 — ASHA 的采样阶段与 RandomSampler
        等价（随机采样），区别仅在 should_prune 的早停逻辑。
        采样结果应在搜索空间约束范围内。
        """
        asha = ASHASampler(seed=42)
        ss = _make_search_space()
        params = asha.sample(ss, [])
        assert isinstance(params, dict), "sample() 应返回 dict"
        # 应包含搜索空间中所有参数
        assert "lr" in params
        assert "hidden" in params
        assert "opt" in params
        # 参数值应在搜索空间范围内
        assert 0.001 <= params["lr"] <= 0.1, "lr 应在 [0.001, 0.1] 范围内"
        assert 16 <= params["hidden"] <= 128, "hidden 应在 [16, 128] 范围内"
        assert params["opt"] in ["adam", "sgd"], "opt 应在 choices 内"

    def test_should_prune_low_performance_returns_true(self):
        """L3 anchor: should_prune() 在低性能试验上返回 True（停止资源分配）。

        锚点：Li et al., 2018 §3 — 被剪枝的试验（性能低于 rung 阈值）
        停止资源分配。当 rung 内试验数 >= η 时，非 top 1/η 的试验
        应被剪枝（should_prune 返回 True）。
        """
        eta = 3
        asha = ASHASampler(eta=eta, direction="maximize")
        # 前 η 个试验填满 rung 0（数据不足时不剪枝）
        asha.should_prune("t1", {0: 0.5}, 0)  # False（不足 η）
        asha.should_prune("t2", {0: 0.6}, 0)  # False（不足 η）
        asha.should_prune("t3", {0: 0.7}, 0)  # False（刚好达到 η）
        # 第 4 个试验：低性能 → 应剪枝
        result = asha.should_prune("t4_low", {0: 0.4}, 0)
        assert result is True, \
            "低性能试验应被剪枝（非 top 1/η，论文 §3 停止资源分配）"

    def test_should_prune_high_performance_returns_false(self):
        """L3 anchor: should_prune() 在高性能试验上返回 False（继续训练）。

        锚点：Li et al., 2018 §3 — 达到 rung 阈值的高性能试验
        被提升到下一 rung，继续分配资源训练。
        """
        eta = 3
        asha = ASHASampler(eta=eta, direction="maximize")
        # 前 η 个试验填满 rung 0
        asha.should_prune("t1", {0: 0.5}, 0)
        asha.should_prune("t2", {0: 0.6}, 0)
        asha.should_prune("t3", {0: 0.7}, 0)
        # 第 4 个试验：高性能（高于所有已有）→ 应保留
        result = asha.should_prune("t4_high", {0: 0.9}, 0)
        assert result is False, \
            "高性能试验应保留（top 1/η，论文 §3 提升到下一 rung）"

    def test_rung_promotion_keeps_top_fraction(self):
        """L3 anchor: 每个 rung 保留 top 1/η 的试验，其余被剪枝。

        锚点：Li et al., 2018 §3 — ASHA 在每个 rung 将试验按性能排序，
        保留 top 1/η（η 是减半率，默认 3），剪枝其余试验。
        n_keep = len(rung) // η，只有 top n_keep 的试验继续训练。
        """
        eta = 3
        asha = ASHASampler(eta=eta, direction="maximize")
        # 填满 rung 0：3 个中等性能试验（都不剪枝，数据不足）
        for tid, val in [("t1", 0.5), ("t2", 0.6), ("t3", 0.7)]:
            assert asha.should_prune(tid, {0: val}, 0) is False, \
                f"前 η 个试验不应剪枝（数据不足，{tid}）"
        # 第 4 个试验：高性能（0.95）→ top 1，应保留
        assert asha.should_prune("t4_best", {0: 0.95}, 0) is False, \
            "top 1/η 的高性能试验应保留（论文 §3）"
        # 第 5 个试验：低性能（0.1）→ 非 top，应剪枝
        assert asha.should_prune("t5_worst", {0: 0.1}, 0) is True, \
            "非 top 1/η 的低性能试验应剪枝（论文 §3）"

    def test_data_insufficient_no_pruning(self):
        """L3 anchor: 数据点不足 η 时不剪枝（避免过早判断）。

        锚点：Li et al., 2018 §3 — ASHA 需要 rung 内有足够试验
        （至少 η 个）才能做可靠的 top 1/η 选择。不足 η 时不剪枝。
        """
        eta = 3
        asha = ASHASampler(eta=eta, direction="maximize")
        # 1 个试验（不足 η=3）
        assert asha.should_prune("t1", {0: 0.5}, 0) is False, \
            "试验数 < η 时不应剪枝（数据不足，论文 §3）"
        # 2 个试验（仍不足 η=3）
        assert asha.should_prune("t2", {0: 0.6}, 0) is False, \
            "试验数 < η 时不应剪枝（数据不足，论文 §3）"

    def test_rung_not_in_values_no_pruning(self):
        """L3 anchor: 试验未达到当前 rung 时不剪枝（尚无评估数据）。

        锚点：Li et al., 2018 §3 — 试验在 rung r 达到阈值后才被
        提升到 rung r+1。若 rung 不在 intermediate_values 中，
        说明试验尚未在该 rung 被评估，不应剪枝。
        """
        asha = ASHASampler(eta=3, direction="maximize")
        # rung=1 不在 intermediate_values 中（只有 rung 0 的数据）
        result = asha.should_prune("t1", {0: 0.5}, 1)
        assert result is False, \
            "rung 不在 intermediate_values 时不应剪枝（试验未达到该 rung，论文 §3）"
