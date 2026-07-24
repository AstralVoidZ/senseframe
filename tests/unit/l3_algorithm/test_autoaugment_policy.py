"""L3 算法行为测试：AutoAugment（学习数据增强策略）。

锚点来源：Cubuk et al., 2019《AutoAugment: Learning Augmentation Strategies from Data》
(https://arxiv.org/abs/1805.09501)

论文关键结论（作为测试锚点）：
- §3: AutoAugment 搜索增强策略，每个策略含 N 个子策略，每个子策略含 2 个操作
- §3: 每个操作有 type（增强类型）、magnitude（强度 [0, 10] 离散等级）、probability（概率 [0, 1]）
- §3: 增强操作只修改数据值，不改变数据形状/维度
"""
from __future__ import annotations

import numpy as np
import pytest

from senseframe.autoaugment import (
    AugmentationSearchSpace,
    AutoAugmentPolicyBuilder,
    AutoAugmentSampler,
    SUPPORTED_AUGMENT_OPS,
)
from senseframe.search_protocol import Sampler


@pytest.mark.l3_algorithm
class TestAutoAugmentPolicyBehavior:
    """验证 AutoAugment 行为符合 Cubuk et al., 2019。"""

    def test_policy_constructible_with_n_ops(self):
        """L3 anchor: AutoAugment 策略可构造（N 个操作槽位）。

        锚点：Cubuk et al., 2019 §3 — AutoAugment 搜索增强策略，
        策略由若干操作槽位组成。SenseFrame 用 n_ops 参数控制槽位数
        （默认 2，对应论文每个子策略含 2 个操作）。
        """
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        params = sampler.sample(sp_ss, [])
        # 应成功构造含 n_ops 个操作槽位的策略
        assert isinstance(params, dict)
        assert "op_0" in params
        assert "op_1" in params
        # 每个 op 应在候选池中
        assert params["op_0"] in SUPPORTED_AUGMENT_OPS
        assert params["op_1"] in SUPPORTED_AUGMENT_OPS

    def test_subpolicy_contains_two_operations(self):
        """L3 anchor: 每个子策略含 2 个操作（论文默认配置）。

        锚点：Cubuk et al., 2019 §3 — 每个子策略由 2 个顺序应用的
        增强操作组成。n_ops=2 时，构造的 transform 应含 2 个操作。
        """
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        params = sampler.sample(sp_ss, [])
        builder = AutoAugmentPolicyBuilder()
        transform = builder.build(params)
        # transform 应是 AutoAugmentTransform（非 IdentityTransform）
        assert hasattr(transform, "ops_chain"), \
            "n_ops=2 时 transform 应是 AutoAugmentTransform（含操作链）"
        # 操作链长度应 = 2（论文每个子策略 2 个操作）
        assert len(transform.ops_chain) == 2, \
            "子策略应含 2 个操作（论文 §3 默认配置）"
        # 每个操作三元组 (op_name, magnitude, probability) 结构完整
        for op_entry in transform.ops_chain:
            assert len(op_entry) == 3, \
                "每个操作应为 (op_name, magnitude, probability) 三元组"

    def test_magnitude_in_bounded_range(self):
        """L3 anchor: 操作的 magnitude 在有界范围内。

        锚点：Cubuk et al., 2019 §3 — 每个操作有 magnitude 参数控制
        增强强度，论文中取值范围为 [0, 10]（10 个离散等级）。
        SenseFrame 将 magnitude 归一化到 [0, 1] 连续范围，作为论文
        [0, 10] 离散等级的连续近似（magnitude=0.5 对应论文 magnitude=5）。
        本测试验证 magnitude 在有界范围内的行为契约。
        """
        ss = AugmentationSearchSpace(n_ops=2)  # 默认 magnitude_range=(0.0, 1.0)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        # 多次采样验证 magnitude 始终在范围内
        for _ in range(10):
            params = sampler.sample(sp_ss, [])
            for i in range(ss.n_ops):
                mag = params[f"magnitude_{i}"]
                # 实现归一化到 [0, 1]（论文原始范围 [0, 10] 的连续近似）
                assert 0.0 <= mag <= 1.0, \
                    f"magnitude_{i}={mag} 应在 [0, 1] 范围内（论文 [0, 10] 的归一化）"

    def test_probability_in_unit_range(self):
        """L3 anchor: 操作的 probability 在 [0, 1] 范围。

        锚点：Cubuk et al., 2019 §3 — 每个操作有 probability 参数
        控制增强应用概率，取值范围为 [0, 1]（0=从不应用，1=总是应用）。
        """
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        for _ in range(10):
            params = sampler.sample(sp_ss, [])
            for i in range(ss.n_ops):
                prob = params[f"probability_{i}"]
                assert 0.0 <= prob <= 1.0, \
                    f"probability_{i}={prob} 应在 [0, 1] 范围内（论文 §3）"

    def test_transform_preserves_data_shape(self):
        """L3 anchor: 策略应用后数据形状不变（增强不改变维度）。

        锚点：Cubuk et al., 2019 §3 — 增强操作只修改数据值（如添加噪声、
        遮挡片段），不改变数据的形状/维度，保持与下游模型输入兼容。
        """
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        params = sampler.sample(sp_ss, [])
        builder = AutoAugmentPolicyBuilder()
        transform = builder.build(params)
        # 构造 2D 输入（channels, length），模拟 WiFi CSI 时序信号
        x = np.random.randn(3, 100).astype(np.float32)
        y = 1
        xx, yy = transform(x, y)
        # 形状应不变（增强不改维度）
        assert xx.shape == x.shape, \
            f"增强后数据形状应不变：{xx.shape} != {x.shape}（论文 §3）"
        # 标签应不变
        assert yy == y, "增强不应修改标签"

    def test_transform_preserves_shape_1d_input(self):
        """L3 anchor: 1D 输入增强后形状也不变。

        锚点：Cubuk et al., 2019 §3 — 增强操作对任意维度的输入
        都不应改变其形状。
        """
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        sampler = AutoAugmentSampler(seed=42)
        params = sampler.sample(sp_ss, [])
        builder = AutoAugmentPolicyBuilder()
        transform = builder.build(params)
        # 1D 输入
        x = np.random.randn(100).astype(np.float32)
        xx, _ = transform(x, 0)
        assert xx.shape == x.shape, \
            f"1D 输入增强后形状应不变：{xx.shape} != {x.shape}"
