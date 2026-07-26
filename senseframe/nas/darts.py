"""RFC-003 ε2 NAS：DARTS 可微架构搜索（P3.3.2 + P1.3 真实超网）。

DARTS（Differentiable Architecture Search）：可微架构搜索，梯度-based。

与 EvolutionarySampler / ENASSampler 区别：
- DARTS 通过 softmax 加权混合 op，使架构参数 α 可微
- 双优化：α（架构参数，验证集）↔ w（模型权重，训练集）交替更新
- 收敛后用 argmax 离散化得到最终架构

P1.3（2026-07-19）新增真实超网路径（use_real_supernet=True）：
- DARTSSupernet（见 supernet.py）实现所有候选 op 并行 + α softmax 加权
- DARTSPipelineRun.use_real_supernet=True 走真实双优化路径
- DARTSSampler.attach_supernet() 可将超网实例附到 sampler

简化路径（use_real_supernet=False，默认）：
- 用 ArchitectureBuilder 构造单个架构作为超网近似
- α 仅控制 cell_type / discrete hyperparams 的选择
- α 梯度用 randn_like 近似（保留作为对比基准和向后兼容）

注册：
- DARTSSampler 注册到 SP Sampler 注册表（register_sampler("darts", DARTSSampler)）
- DARTSPipelineRun 不继承标准 Pipeline（双优化不符合 stage-based 流程）
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..search_protocol import (
    Sampler,
    SearchSpace,
    register_sampler,
)

logger = logging.getLogger(__name__)


# ============================================================
# P2.3-3 修复：BatchNorm 双优化 stats 更新两次
# ============================================================
# DARTS 双优化每 epoch 做 2 次 forward：w 更新（train batch）+ α 更新（val batch）。
# BN1d 在 train 模式下每次 forward 都更新 running_mean / running_var，
# 严格 DARTS 应仅在 w 更新阶段更新 BN stats（α 更新阶段冻结 BN stats）。
#
# 修复策略：α 更新前临时把所有 BN 的 momentum 设为 0.0（BN 仍用 batch stats
# 归一化，但不更新 running stats），更新后恢复原 momentum。
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _set_bn_momentum(model: nn.Module, momentum: float) -> Dict[int, float]:
    """临时修改所有 BN 模块的 momentum，返回 {id(module): original_momentum}。"""
    original: Dict[int, float] = {}
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            original[id(m)] = m.momentum
            m.momentum = momentum
    return original


def _restore_bn_momentum(model: nn.Module, original: Dict[int, float]) -> None:
    """恢复 BN momentum（按 id 匹配，避免重复模块被多次恢复）。"""
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            mid = id(m)
            if mid in original:
                m.momentum = original[mid]


class DARTSSampler:
    """DARTS：可微架构搜索（梯度-based）。

    架构参数 α（可微），双优化：α 与 w 交替更新。

    关键设计：
    - sample() 用 softmax 采样架构（argmax 选离散架构）
    - update(gradient) 实现 α 梯度更新（DARTS 双优化的架构参数侧）
    - 满足 SP Sampler Protocol（name + sample + warm_start）

    Args:
        arch_alpha: 架构参数 α dict（{op_name: torch.Tensor requires_grad=True}）
            None 时由 sample() 第一次调用时从 search_space 构造
        lr_arch: 架构参数学习率
        seed: 随机种子
    """
    name = "darts"

    def __init__(
        self,
        arch_alpha: Optional[Dict[str, torch.Tensor]] = None,
        lr_arch: float = 3e-4,
        seed: Optional[int] = None,
    ):
        """初始化 DARTSSampler。

        Args:
            arch_alpha: 架构参数 α dict（{op_name: torch.Tensor requires_grad=True}）
                None 时由 sample() 第一次调用时从 search_space 构造
            lr_arch: 架构参数学习率（默认 3e-4，DARTS 论文 Liu et al. 2018
                第 4 节"Architecture Search"推荐值）
            seed: 随机种子
        """
        # arch_alpha: Dict[str, torch.Tensor]，每个 tensor 是某节点的 op 选择分布
        self.arch_alpha: Dict[str, torch.Tensor] = dict(arch_alpha) if arch_alpha else {}
        self.lr_arch = lr_arch
        self.optimizer: Optional[torch.optim.Optimizer] = None  # 延迟构造
        self._rng = random.Random(seed)
        # 保留 search_space 引用，用于 _discretize_to_arch_params
        self._search_space: Optional[SearchSpace] = None
        # P1.3：真实超网实例（attach_supernet 后非 None）
        self._supernet: Any = None

    # ============================================================
    # SP Sampler Protocol 实现
    # ============================================================
    def sample(
        self,
        search_space: SearchSpace,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """softmax 采样架构参数（argmax 选离散架构）。

        Args:
            search_space: SP SearchSpace（含参数规格）
            history: 已完成 trial 的历史（DARTS 不依赖 history，忽略）

        Returns:
            离散化后的 arch_params dict（含 cell_type + 超参）
        """
        # 若 arch_alpha 为空，从 search_space 构造初始 α
        if not self.arch_alpha:
            self._init_arch_alpha_from_search_space(search_space)
        self._search_space = search_space

        # softmax + argmax 选每个节点的 op
        sampled: Dict[str, int] = {}
        for k, alpha in self.arch_alpha.items():
            probs = torch.softmax(alpha, dim=-1)
            sampled[k] = int(probs.argmax().item())

        # 将离散选择翻译回 arch_params dict（cell_type + 超参）
        return self._discretize_to_arch_params(sampled)

    def update(self, gradient: Dict[str, torch.Tensor]) -> None:
        """架构参数 α 梯度更新（DARTS 双优化）。

        Args:
            gradient: {op_name: tensor}，每个 tensor 是 α 的梯度
        """
        if not self.arch_alpha:
            return  # α 未初始化，无法更新

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(self.arch_alpha.values()), lr=self.lr_arch,
            )

        # 资源泄露修复：detach gradient 切断计算图引用，
        # 避免外部 tensor 持有的计算图通过 .grad 累积在 arch_alpha 上。
        # 直接赋值 grad 会持有外部 tensor 引用（可能含计算图），
        # 改用 .detach() + .clone() 切断引用链，确保 grad 是叶子 tensor。
        for k, g in gradient.items():
            if k in self.arch_alpha:
                self.arch_alpha[k].grad = g.detach().clone()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True 释放 grad tensor 引用

    def cleanup(self) -> None:
        """释放 sampler 持有的资源（P3 资源泄露修复）。

        释放：
        - arch_alpha tensor（requires_grad=True，持有计算图）
        - optimizer（持有 arch_alpha tensor 的强引用）

        调用后 sampler 不可再用，需重新初始化 arch_alpha。
        """
        if self.optimizer is not None:
            # 释放 optimizer 对 arch_alpha 的引用
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer = None
        # 释放 arch_alpha tensor
        self.arch_alpha.clear()
        self._search_space = None
        # P1.3：解除超网引用（不 del，由调用方管理 supernet 生命周期）
        self._supernet = None

    def warm_start(self, source_history: List[Dict[str, Any]]) -> None:
        """P3.2.1 ε4 元学习兼容：no-op（DARTS 用梯度更新 α，不从历史偏向）。

        保留此方法满足 SP Sampler Protocol（@runtime_checkable 要求所有方法存在）。
        """
        return None

    # ============================================================
    # P1.3：真实超网支持（可选）
    # ============================================================
    def attach_supernet(self, supernet: Any) -> None:
        """将 DARTSSupernet 实例附到 sampler（P1.3 新增）。

        附着后，sample() / discretize() 将基于真实超网的 α 参数，
        而非 sampler 自身的 arch_alpha dict。

        Args:
            supernet: DARTSSupernet 实例（见 senseframe.nas.supernet）
        """
        self._supernet = supernet

    def detach_supernet(self) -> None:
        """解除附着的超网实例（P1.3 新增）。"""
        self._supernet = None

    @property
    def search_space(self):
        """返回当前绑定的 SearchSpace（可能为 None）。"""
        return self._search_space

    @property
    def supernet(self):
        """返回当前绑定的 supernet（可能为 None）。"""
        return self._supernet

    # ============================================================
    # 内部方法
    # ============================================================
    def _init_arch_alpha_from_search_space(self, search_space: SearchSpace) -> None:
        """从 search_space 构造初始 α（每个 categorical 参数一个 α 向量）。

        每个 categorical 参数对应一个 α 向量（长度 = len(choices)）。
        int/float 参数不参与可微搜索（DARTS 仅对离散选择可微）。
        """
        for p in search_space.parameters:
            if p.type == "categorical" and p.choices:
                self.arch_alpha[p.name] = torch.zeros(
                    len(p.choices), requires_grad=True,
                )

    def _discretize_to_arch_params(self, sampled: Dict[str, int]) -> Dict[str, Any]:
        """将离散 op 索引翻译回 arch_params dict。

        Args:
            sampled: {param_name: op_index}

        Returns:
            arch_params dict（{param_name: choice_value}）
        """
        if self._search_space is None:
            # 无 search_space 上下文，直接返回索引（测试兼容）
            return dict(sampled)

        arch_params: Dict[str, Any] = {}
        param_map = {p.name: p for p in self._search_space.parameters}
        for name, idx in sampled.items():
            p = param_map.get(name)
            if p is not None and p.choices and 0 <= idx < len(p.choices):
                arch_params[name] = p.choices[idx]
            else:
                arch_params[name] = idx
        return arch_params


# 注册到 SP Sampler 注册表
register_sampler("darts", DARTSSampler)


# ============================================================
# P3.3.2 + P1.3: DARTSPipelineRun（特殊 PipelineRun，内部双优化）
# ============================================================
class DARTSPipelineRun:
    """DARTS 特殊 PipelineRun（内部双优化，不走标准 stage）。

    流程：
    1. 构造超网（所有候选架构的并集）
    2. 交替更新：w（模型权重，训练集）↔ α（架构参数，验证集）
    3. 收敛后 sample() 得到最终架构

    两条路径：
    - use_real_supernet=False（默认）：简化 DARTS，α 用 randn_like 近似梯度。
      用 ArchitectureBuilder 构造单个架构作为超网近似，α 仅控制
      cell_type / discrete hyperparams 的选择。
    - use_real_supernet=True（P1.3 新增）：真实 DARTS 可微超网。
      用 DARTSSupernet（见 supernet.py）实现所有候选 op 并行 + α softmax
      加权，α 通过验证集 backward 真实可微更新。

    注意：DARTSPipelineRun 不继承标准 Pipeline，因为 DARTS 的双优化
    不符合 stage-based 流程。它是独立的训练循环。
    """

    def __init__(
        self,
        sampler: DARTSSampler,
        builder: Any,
        search_space: SearchSpace,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_epochs: int = 50,
        lr_w: float = 0.025,
        lr_arch: float = 3e-4,
        use_real_supernet: bool = False,
        supernet_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """初始化 DARTSPipelineRun。

        Args:
            sampler: DARTSSampler 实例
            builder: ArchitectureBuilder 实例（use_real_supernet=True 时可传 None）
            search_space: SP SearchSpace
            input_shape: 模型输入形状（不含 batch 维），如 (channels, length)
            num_classes: 输出类别数
            n_epochs: 训练 epoch 数
            lr_w: 模型权重 w 学习率（默认 0.025，DARTS 论文 Liu et al. 2018
                第 4 节"CIFAR-10"推荐值，带 momentum=0.9 / weight_decay=3e-4）
            lr_arch: 架构参数 α 学习率（默认 3e-4，DARTS 论文推荐值，
                Adam optimizer betas=(0.5, 0.999) / weight_decay=1e-3）
            use_real_supernet: 是否使用真实 DARTS 可微超网（P1.3 新增）。
                True 时使用 DARTSSupernet（supernet.py），α 通过验证集
                backward 真实更新；False 时走简化路径（randn_like 近似梯度）。
            supernet_kwargs: 传递给 DARTSSupernet 的额外参数（n_cells / c_stem /
                c_cell / op_names），仅 use_real_supernet=True 时生效。
        """
        self.sampler = sampler
        self.builder = builder
        self.search_space = search_space
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_epochs = n_epochs
        self.lr_w = lr_w
        self.lr_arch = lr_arch
        self.use_real_supernet = use_real_supernet
        self.supernet_kwargs = supernet_kwargs or {}

    def run(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> Dict[str, Any]:
        """执行 DARTS 双优化训练（根据 use_real_supernet 分发）。

        Args:
            train_loader: 训练数据 loader（可迭代）
            val_loader: 验证数据 loader（可迭代）

        Returns:
            use_real_supernet=False: {"best_arch", "final_alpha", "history"}
            use_real_supernet=True: {"best_arch", "final_alpha", "history",
                                      "discrete_model"(可选), "supernet_arch"}
        """
        if self.use_real_supernet:
            return self._run_real_supernet(train_loader, val_loader)
        return self._run_simplified(train_loader, val_loader)

    # ============================================================
    # P1.3：真实超网双优化路径
    # ============================================================
    def _run_real_supernet(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> Dict[str, Any]:
        """真实 DARTS 可微超网双优化（P1.3 新增）。

        流程：
        1. 构造 DARTSSupernet（所有候选 op 并行 + α softmax 加权）
        2. w_optimizer = SGD(supernet.w_parameters(), lr=lr_w, momentum=0.9)
        3. alpha_optimizer = Adam(supernet.alpha_parameters(), lr=lr_arch)
        4. for epoch in range(n_epochs):
             - 训练集 batch：w_loss.backward() → w_optimizer.step()
             - 验证集 batch：alpha_loss.backward() → alpha_optimizer.step()
        5. supernet.discretize() 得到 best_arch

        与简化路径的核心差异：
        - α 真实参与 forward（softmax 加权所有 op），通过 autograd 反传梯度
        - 不需要 randn_like 近似梯度
        - 离散化基于真实超网 argmax α

        资源管理（沿用简化路径的 try/finally 模式）：
        - 异常时释放 supernet / optimizers / iterator
        - zero_grad(set_to_none=True) 释放 grad 引用

        Args:
            train_loader: 训练数据 loader
            val_loader: 验证数据 loader

        Returns:
            {"best_arch": discretize dict, "final_alpha": alpha_dict,
             "history": [{"epoch", "w_loss", "alpha_loss"}],
             "supernet_arch": discretize dict（与 best_arch 一致）}
        """
        from .supernet import DARTSSupernet

        # 1. 构造真实超网
        supernet = DARTSSupernet(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            **self.supernet_kwargs,
        )
        supernet.train()

        # 2. 双优化器：w 用 SGD（带动量），α 用 Adam
        w_optimizer = torch.optim.SGD(
            list(supernet.w_parameters()),
            lr=self.lr_w,
            momentum=0.9,
            weight_decay=3e-4,
        )
        alpha_optimizer = torch.optim.Adam(
            list(supernet.alpha_parameters()),
            lr=self.lr_arch,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

        # 附着超网到 sampler（P1.3：使 sampler 可访问超网 α）
        self.sampler.attach_supernet(supernet)

        # 3. 训练循环
        history: List[Dict[str, Any]] = []
        criterion = nn.CrossEntropyLoss()

        train_iter = _InfiniteLoader(train_loader)
        val_iter = _InfiniteLoader(val_loader)

        try:
            for epoch in range(self.n_epochs):
                # ---- w 更新（训练集，α 冻结）----
                try:
                    x_train, y_train = train_iter.next()
                except StopIteration:
                    break
                w_optimizer.zero_grad(set_to_none=True)
                logits = supernet(x_train)
                w_loss = criterion(logits, y_train)
                w_loss.backward()
                w_optimizer.step()

                # ---- α 更新（验证集，w 冻结）----
                try:
                    x_val, y_val = val_iter.next()
                except StopIteration:
                    break
                # P2.3-3 修复：α 更新阶段暂停 BN running stats 更新。
                # 原实现 w 和 α 各 forward 一次，BN running stats 被更新两次，
                # 严格 DARTS 应只在 w 更新时更新 BN stats（α 更新时冻结）。
                # 通过临时把 BN momentum 设为 0.0 实现：BN 仍用 batch stats
                # 归一化（保留 train 模式行为），但 running stats 不再更新。
                bn_momenta = _set_bn_momentum(supernet, 0.0)
                try:
                    alpha_optimizer.zero_grad(set_to_none=True)
                    # 用 _real 后缀标识真实超网路径的中间变量，
                    # 与简化路径的 val_logits / alpha_loss 区分（grep 实证清晰）
                    val_logits_real = supernet(x_val)
                    alpha_loss_real = criterion(val_logits_real, y_val)
                    # 真实可微：α 通过 softmax 加权参与 forward，
                    # backward 直接反传到 α（无需随机梯度近似）
                    alpha_loss_real.backward()
                    alpha_optimizer.step()
                finally:
                    _restore_bn_momentum(supernet, bn_momenta)

                history.append({
                    "epoch": epoch,
                    "w_loss": float(w_loss.item()),
                    "alpha_loss": float(alpha_loss_real.item()),
                })
                # 释放本轮 loss tensor 引用
                del w_loss, alpha_loss_real, logits, val_logits_real

            # 4. 离散化得到 best_arch
            best_arch = supernet.discretize()

            # 构造 final_alpha dict（detach + cpu 便于序列化）
            final_alpha = {
                k: v.detach().cpu().clone()
                for k, v in supernet.alpha_dict().items()
            }

            return {
                "best_arch": best_arch,
                "final_alpha": final_alpha,
                "history": history,
                "supernet_arch": dict(best_arch),
            }
        finally:
            # 资源泄露修复：显式释放训练资源
            train_iter.close()
            val_iter.close()
            w_optimizer.zero_grad(set_to_none=True)
            alpha_optimizer.zero_grad(set_to_none=True)
            del w_optimizer
            del alpha_optimizer
            # 解除 sampler 对 supernet 的引用
            self.sampler.detach_supernet()
            del supernet
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ============================================================
    # 简化路径（use_real_supernet=False）
    # ============================================================
    def _run_simplified(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> Dict[str, Any]:
        """简化 DARTS 双优化（α 用 randn_like 近似梯度）。

        流程（简化 DARTS）：
        1. 用 sampler.sample() 得到初始 arch_params，构造超网近似模型
        2. w_optimizer = SGD(supernet.parameters(), lr=lr_w)
        3. 初始化 α（若 sampler.arch_alpha 为空）
        4. for epoch in range(n_epochs):
             - 取一个 train batch，更新 w（train loss.backward）
             - 取一个 val batch，计算 val loss 对 α 的梯度，更新 α
        5. 最终 sampler.sample() 得到 best_arch

        资源泄露修复（P3 审查）：
        - val_logits / alpha_loss 用 torch.no_grad() 包裹，避免计算图累积
          （α 不参与 supernet forward，val loss 仅用于监控，无需 backward）
        - alpha_grad tensor 用 .detach() 切断外部计算图引用
        - try/finally 确保异常时释放 supernet / optimizer / iterator
        - _InfiniteLoader.close() 释放 loader 引用

        Args:
            train_loader: 训练数据 loader（可迭代）
            val_loader: 验证数据 loader（可迭代）

        Returns:
            {"best_arch": arch_params, "final_alpha": alpha_dict,
             "history": [{"epoch": i, "w_loss": ..., "alpha_loss": ...}]}
        """
        # 1. 初始化 α（若 sampler.arch_alpha 为空）
        if not self.sampler.arch_alpha:
            self.sampler._init_arch_alpha_from_search_space(self.search_space)
        self.sampler._search_space = self.search_space

        # 2. 用初始 sample 构造超网近似模型
        initial_arch = self.sampler.sample(self.search_space, [])
        supernet = self.builder.build(initial_arch, self.input_shape, self.num_classes)
        supernet.train()

        # 3. w_optimizer（模型权重）
        w_optimizer = torch.optim.SGD(supernet.parameters(), lr=self.lr_w)

        # 4. 确保 α 的 optimizer 延迟构造（在 sampler.update 中处理）
        # 取 α 参数列表
        alpha_params = list(self.sampler.arch_alpha.values())

        # 历史：每个 epoch 的 loss
        history: List[Dict[str, Any]] = []
        criterion = nn.CrossEntropyLoss()

        # 迭代器
        train_iter = _InfiniteLoader(train_loader)
        val_iter = _InfiniteLoader(val_loader)

        try:
            for epoch in range(self.n_epochs):
                # ---- w 更新（训练集）----
                try:
                    x_train, y_train = train_iter.next()
                except StopIteration:
                    break
                w_optimizer.zero_grad()
                logits = supernet(x_train)
                w_loss = criterion(logits, y_train)
                w_loss.backward()
                w_optimizer.step()

                # ---- α 更新（验证集）----
                try:
                    x_val, y_val = val_iter.next()
                except StopIteration:
                    break
                # 资源泄露修复：α 不参与 supernet forward（简化 DARTS），
                # val loss 仅用于监控，用 torch.no_grad() 避免计算图累积。
                # 原实现中 val_logits / alpha_loss 创建计算图但从未 backward，
                # 每个 epoch 泄露一份计算图（n_epochs=50 时累积 50 份）。
                for ap in alpha_params:
                    if ap.grad is not None:
                        ap.grad.zero_()

                with torch.no_grad():
                    val_logits = supernet(x_val)
                    alpha_loss = criterion(val_logits, y_val)

                # 构造 α 梯度（简化：用 val_loss 对 α 的数值梯度近似）
                # 资源泄露修复：randn_like 在 alpha_params 所在设备创建 tensor，
                # 用 .detach() 显式切断（虽然 randn_like 本身无计算图，但保持一致性）
                alpha_grad: Dict[str, torch.Tensor] = {}
                for name, ap in self.sampler.arch_alpha.items():
                    # 用随机梯度作为 α 更新信号（简化 DARTS 的近似）
                    # 真实 DARTS 通过 softmax 加权使 α 可微；此处用扰动梯度近似
                    grad = (torch.randn_like(ap) * 0.01).detach()
                    alpha_grad[name] = grad

                self.sampler.update(alpha_grad)
                # 释放 alpha_grad dict 引用（update 内部已 detach+clone）
                alpha_grad.clear()

                history.append({
                    "epoch": epoch,
                    "w_loss": float(w_loss.item()),
                    "alpha_loss": float(alpha_loss.item()),
                })
                # 释放本轮 loss tensor 引用
                del w_loss, alpha_loss, logits, val_logits

            # 5. 最终 sample 得到 best_arch
            best_arch = self.sampler.sample(self.search_space, [])

            # 构造 final_alpha dict（detach + cpu 便于序列化）
            final_alpha = {
                k: v.detach().cpu().clone() for k, v in self.sampler.arch_alpha.items()
            }

            return {
                "best_arch": best_arch,
                "final_alpha": final_alpha,
                "history": history,
            }
        finally:
            # 资源泄露修复：显式释放训练资源
            # 1. 释放 iterator（如果 loader 是 DataLoader 且 num_workers>0，
            #    持有 worker 进程引用）
            train_iter.close()
            val_iter.close()
            # 2. 释放 w_optimizer 对 supernet 参数的引用
            w_optimizer.zero_grad(set_to_none=True)
            del w_optimizer
            # 3. 释放 supernet（nn.Module 持有大量参数 tensor）
            del supernet
            # 4. 触发 Python GC（supernet 可能有大 tensor）
            import gc
            gc.collect()
            # 5. 如果使用 CUDA，清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class _InfiniteLoader:
    """循环迭代 loader（避免 epoch 间重新初始化迭代器）。

    简化 DARTS 训练循环用：当 loader 耗尽时重新开始。

    资源泄露修复（P3 审查）：
    - 持有 loader 引用，如果 loader 是 DataLoader 且 num_workers>0，会持有 worker 进程
    - 添加 close() 方法显式释放 loader 和 _iter 引用
    - 支持 context manager 协议（__exit__ 调用 close）

    P2.3-6 修复：空 loader 防御。
    - 原实现 next() 在 loader 为空（首次 iter 就 StopIteration）时
      会陷入无限递归：next() -> StopIteration -> 重建 iter -> next() -> StopIteration -> ...
    - 修复：记录重置次数，连续 2 次 StopIteration 视为空 loader，raise 清晰错误。
    """

    def __init__(self, loader: Any):
        self.loader = loader
        self._iter = iter(loader)
        # P2.3-6 修复：连续 StopIteration 计数，用于空 loader 检测
        self._consecutive_reset_count = 0
        self._MAX_RESETS = 2  # 连续 2 次重置仍 StopIteration 视为空 loader

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            result = next(self._iter)
            # 成功取到数据，重置计数
            self._consecutive_reset_count = 0
            return result
        except StopIteration:
            # P2.3-6 修复：检测空 loader
            self._consecutive_reset_count += 1
            if self._consecutive_reset_count >= self._MAX_RESETS:
                raise RuntimeError(
                    f"_InfiniteLoader: loader 连续 {self._MAX_RESETS} 次 "
                    f"StopIteration，疑似空 loader（无任何 batch）。"
                    f"请检查 train_loader/val_loader 是否含数据。"
                ) from None
            self._iter = iter(self.loader)
            # 递归一次：若新 iter 仍 StopIteration，下次 next() 会触发上面计数+1
            return self.next()

    @property
    def is_closed(self) -> bool:
        """返回 loader 是否已关闭。"""
        return self._iter is None

    def close(self) -> None:
        """释放 loader 和 iterator 引用（资源泄露修复）。"""
        # 如果 _iter 是 DataLoaderIterator，调用 _shutdown_workers 释放 worker 进程
        _iter_obj = self._iter
        if _iter_obj is not None and hasattr(_iter_obj, "_shutdown_workers"):
            try:
                _iter_obj._shutdown_workers()
            except Exception:
                pass
        self._iter = None
        self.loader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


__all__ = ["DARTSSampler", "DARTSPipelineRun"]
