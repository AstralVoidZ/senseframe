"""L3 算法行为测试：DARTS（Differentiable Architecture Search）。

锚点来源：Liu et al., 2019《DARTS: Differentiable Architecture Search》
(https://arxiv.org/abs/1806.09055)

论文关键结论（作为测试锚点）：
- §2.2: DARTS 通过 softmax 权重连续化架构参数 α，使离散选择可微
- §2.3: 架构参数 α 通过梯度下降优化
- §2.3: 搜索结束后取 argmax 离散化得到最终架构
- §2.3.1: 双层优化——α 用验证集梯度更新，w 用训练集梯度更新
"""
from __future__ import annotations

import pytest

# torch 可选（如未安装则 skip 整个文件）
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from senseframe.nas import DARTSSampler, DARTSSupernet
from senseframe.nas.search_space import ArchitectureSearchSpace


@pytest.mark.l3_algorithm
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestDARTSSearchBehavior:
    """验证 DARTS 行为符合 Liu et al., 2019。"""

    def test_architecture_params_constructible(self):
        """L3 anchor: DARTS 模型可构造，架构参数 α 初始化为可微 tensor。

        锚点：Liu et al., 2019 §2.2 — DARTS 引入架构参数 α，
        用连续 softmax 加权混合所有候选操作，使架构搜索可微。
        α 必须是 requires_grad=True 的叶子 tensor 才能通过梯度下降优化。
        """
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        sampler._init_arch_alpha_from_search_space(ss)
        # α 应非空（每个 categorical 参数一个 α 向量）
        assert len(sampler.arch_alpha) > 0
        for name, alpha in sampler.arch_alpha.items():
            assert isinstance(alpha, torch.Tensor), \
                f"α[{name}] 应是 torch.Tensor"
            assert alpha.requires_grad, \
                f"α[{name}] 应可训练（requires_grad=True，论文 §2.2 可微要求）"
            assert alpha.is_leaf, \
                f"α[{name}] 应是叶子 tensor（梯度下降优化前提）"

    def test_softmax_produces_probability_distribution(self):
        """L3 anchor: softmax(α) 产生合法概率分布（和为 1，非负）。

        锚点：Liu et al., 2019 §2.2 — 用 softmax(α) 将架构参数连续化，
        得到各操作的概率权重，所有候选操作权重和为 1（概率分布）。
        """
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"]).to_sp_search_space()
        sampler._init_arch_alpha_from_search_space(ss)
        for name, alpha in sampler.arch_alpha.items():
            probs = torch.softmax(alpha, dim=-1)
            # 概率和应接近 1（浮点容差）
            assert abs(probs.sum().item() - 1.0) < 1e-6, \
                f"softmax(α[{name}]) 和应 = 1（概率分布），got {probs.sum().item()}"
            # 所有概率非负
            assert (probs >= 0).all(), \
                f"softmax(α[{name}]) 应非负（概率分布性质）"

    def test_argmax_selects_optimal_operation(self):
        """L3 anchor: 搜索结束后取 argmax(α) 选择最优操作。

        锚点：Liu et al., 2019 §2.3 — 搜索收敛后，对每个节点的 α 取 argmax
        离散化得到最终架构（保留贡献最大的操作）。softmax 单调，故
        argmax(softmax(α)) = argmax(α)。
        """
        # 构造 ArchitectureSearchSpace 提供合法 choices
        ss_arch = ArchitectureSearchSpace(cell_types=["conv1d", "rnn"])
        sp_ss = ss_arch.to_sp_search_space()
        cell_type_param = ss_arch.get_param("cell_type")
        n_choices = len(cell_type_param.choices)
        # 构造 α 使最后一个 choice 的 logit 远高于其他（argmax 明确）
        alpha_data = torch.zeros(n_choices)
        alpha_data[-1] = 10.0
        alpha = alpha_data.clone().requires_grad_(True)
        sampler = DARTSSampler(arch_alpha={"cell_type": alpha})
        # sample() 内部做 softmax + argmax
        params = sampler.sample(sp_ss, [])
        # 应选中 α 值最大的操作（最后一个 choice）
        assert params["cell_type"] == cell_type_param.choices[-1], \
            "argmax(α) 应选择 α 值最大的操作（论文 §2.3 离散化）"

    def test_architecture_params_trainable_via_gradient(self):
        """L3 anchor: 架构参数 α 可通过梯度下降更新（可训练）。

        锚点：Liu et al., 2019 §2.3 — α 通过梯度下降优化，
        梯度来源于验证集 loss 对 α 的偏导（双层优化的架构参数侧）。
        update(gradient) 应使 α 发生变化。
        """
        sampler = DARTSSampler()
        ss = ArchitectureSearchSpace(cell_types=["conv1d"]).to_sp_search_space()
        sampler.sample(ss, [])
        # 记录 α 初始值
        alpha_before = {k: v.clone() for k, v in sampler.arch_alpha.items()}
        # 构造梯度并更新（模拟验证集梯度）
        gradient = {k: torch.ones_like(v) for k, v in sampler.arch_alpha.items()}
        sampler.update(gradient)
        # α 应发生变化（Adam 优化器应用了梯度）
        changed = any(
            not torch.allclose(alpha_before[k], sampler.arch_alpha[k])
            for k in sampler.arch_alpha
        )
        assert changed, "update(gradient) 应使 α 发生变化（梯度下降生效，论文 §2.3）"

    def test_supernet_alpha_participates_in_forward(self):
        """L3 anchor: α 通过 softmax 加权真实参与前向传播（可微基础）。

        锚点：Liu et al., 2019 §2.2 — 超网中每个节点的输出是
        所有候选 op 的 softmax(α) 加权求和，使 α 参与计算图，
        从而可通过 autograd 反传梯度（非近似梯度）。
        """
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=1)
        x = torch.randn(2, 30, 100)
        # 前向传播
        logits = sn(x)
        # 用 loss 反传
        loss = logits.sum()
        loss.backward()
        # α 应有非 None 梯度（说明 α 通过 softmax 加权参与了计算图）
        for cell in sn.cells:
            assert cell.alpha.grad is not None, \
                "α 应有梯度（通过 softmax 加权参与前向计算图，论文 §2.2）"
            assert cell.alpha.grad.abs().sum().item() > 0, \
                "α 梯度应非全 0（autograd 真实反传，非近似）"

    def test_double_optimization_separates_alpha_and_weights(self):
        """L3 anchor: 双层优化——α 与 w 参数分离，各自独立更新。

        锚点：Liu et al., 2019 §2.3.1 — 双层优化：架构参数 α 用验证集
        梯度更新，模型权重 w 用训练集梯度更新，两者交替优化。
        实现应将 α 和 w 参数分离（互斥），各自有独立优化器。
        """
        sn = DARTSSupernet(input_shape=(30, 100), num_classes=7, n_cells=2)
        w_params = list(sn.w_parameters())
        alpha_params = list(sn.alpha_parameters())
        # α 和 w 应互斥（不应有重叠的参数）
        alpha_ptrs = {p.data_ptr() for p in alpha_params}
        w_ptrs = {p.data_ptr() for p in w_params}
        assert not (alpha_ptrs & w_ptrs), \
            "α 参数与 w 参数应互斥（双层优化分离前提，论文 §2.3.1）"
        # α 参数数量 = n_cells（每个 cell 一个 α）
        assert len(alpha_params) == 2, \
            "α 参数数量应 = n_cells（每个 cell 一个 α）"
        # w 参数应远多于 α（w 含 stem + cells.ops + classifier 权重）
        assert len(w_params) > len(alpha_params), \
            "w 参数应远多于 α（w 是模型权重，α 仅是架构参数）"
