"""P1.3 DARTS 真实超网（2026-07-19）。

替代 darts.py 中"简化 DARTS"的随机梯度近似（见 _run_simplified 方法），
实现真正的可微超网：

- 所有候选操作并行计算，α softmax 加权求和
- 双优化：w 用 SGD（训练集），α 用 Adam（验证集）
- 离散化：argmax α 选择每个 cell 的最佳 op

结构（cell-based DARTS 简化版）：
- stem: Conv1d + BN + ReLU
- cells: N 个 DARTSCell 串联，每个 cell 内 M 个候选 op 并行
- classifier: AdaptiveAvgPool1d(1) + Linear

候选 op（每个 cell，对齐 DARTS 原论文简化版）：
- conv3 / conv5 / avgpool / maxpool / identity
- 所有 op 保持 (B, C, L) 形状一致（pool 用 stride=1 + padding）
- identity 在 c_in != c_out 时退化为 1x1 Conv1d 对齐 channel

参数分离（双优化基础）：
- w_parameters(): stem + cells.ops + classifier（用 SGD 更新）
- alpha_parameters(): cells.alpha（用 Adam 更新）
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 候选 op 名称（与 DARTSCell.alpha 索引对齐）
OP_NAMES: List[str] = ["conv3", "conv5", "avgpool", "maxpool", "identity"]


def _build_op(op_name: str, c_in: int, c_out: int) -> nn.Module:
    """构造单个候选 op。

    所有 op 输入输出形状一致：(B, c_in, L) → (B, c_out, L)
    （length 保持不变，pool 用 stride=1 + padding）

    Args:
        op_name: OP_NAMES 之一
        c_in: 输入通道数
        c_out: 输出通道数

    Returns:
        nn.Module 实例
    """
    if op_name == "conv3":
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
    if op_name == "conv5":
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
    if op_name == "avgpool":
        # AvgPool 保持 length（stride=1, padding=1, kernel=3）
        # channel 对齐：若 c_in != c_out，加 1x1 Conv
        layers: List[nn.Module] = [
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        ]
        if c_in != c_out:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(c_out))
        return nn.Sequential(*layers)
    if op_name == "maxpool":
        layers = [nn.MaxPool1d(kernel_size=3, stride=1, padding=1)]
        if c_in != c_out:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(c_out))
        return nn.Sequential(*layers)
    if op_name == "identity":
        if c_in == c_out:
            return nn.Identity()
        # channel 不匹配时用 1x1 conv 对齐（DARTS 原论文用 factorized reduction，
        # 此处简化为 1x1 conv）
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
        )
    raise ValueError(f"Unknown op '{op_name}' (supported: {OP_NAMES})")


class DARTSCell(nn.Module):
    """DARTS 可微 cell：M 个候选 op 并行计算，α softmax 加权混合。

    每个 cell 持有：
    - ops: ModuleList（M 个候选 op）
    - alpha: Parameter(M,) — 可学习架构参数

    forward(x):
        weights = softmax(alpha, dim=-1)  # (M,)
        outputs = [op(x) for op in ops]   # M 个 (B, c_out, L)
        return sum(w_i * out_i)           # 加权求和

    离散化：argmax(alpha) → op 索引 → op 名
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        op_names: List[str] = None,
    ):
        super().__init__()
        op_names = list(op_names) if op_names else list(OP_NAMES)
        self.op_names = op_names
        self.ops = nn.ModuleList(
            [_build_op(name, c_in, c_out) for name in self.op_names]
        )
        # P2.3-2 修复：α 初始化为小随机值（而非 zeros）。
        # 原实现 zeros 使 softmax 后等概率分布，早期训练 α 梯度信号弱
        # （所有 op 权重相同，argmax 无区分度），可能减缓架构搜索收敛。
        # 改用 0.001 标准差的小随机初始化，打破对称性同时保持近等概率。
        self.alpha = nn.Parameter(torch.randn(len(self.op_names)) * 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=-1)
        # 并行计算所有 op（autograd 跟踪到 α）
        outputs = [op(x) for op in self.ops]
        # 加权求和（显式循环，避免堆叠增加内存峰值）
        out = outputs[0] * weights[0]
        for i in range(1, len(outputs)):
            out = out + outputs[i] * weights[i]
        return out

    def discretize(self) -> str:
        """离散化：argmax α → op 名。"""
        idx = int(self.alpha.argmax().item())
        return self.op_names[idx]


class DARTSSupernet(nn.Module):
    """DARTS 可微超网（P1.3 新增）。

    结构：
    - stem: Conv1d(in_channels → c_stem) + BN + ReLU
    - cells: N 个 DARTSCell 串联（第一个 c_in=c_stem，其余 c_in=c_cell）
    - classifier: AdaptiveAvgPool1d(1) + Linear(c_cell → num_classes)

    参数分离（双优化基础）：
    - w_parameters(): stem + cells.ops + classifier（用 SGD 更新）
    - alpha_parameters(): cells.alpha（用 Adam 更新）

    离散化：每个 cell argmax α → op 名，返回 dict
        {"cell_0": "conv3", "cell_1": "maxpool", ...}

    Args:
        input_shape: 模型输入形状（不含 batch 维），如 (channels, length)
        num_classes: 输出类别数
        n_cells: cell 数量（默认 3，必须 ≥ 1）
        c_stem: stem 输出通道数（默认 32）
        c_cell: cell 输出通道数（默认 64）
        op_names: 候选 op 名称列表（默认 OP_NAMES）

    Examples:
        >>> supernet = DARTSSupernet(input_shape=(30, 100), num_classes=7)
        >>> x = torch.randn(4, 30, 100)
        >>> logits = supernet(x)  # (4, 7)
        >>> arch = supernet.discretize()  # {"cell_0": "conv3", ...}
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        n_cells: int = 3,
        c_stem: int = 32,
        c_cell: int = 64,
        op_names: List[str] = None,
    ):
        super().__init__()
        # P1.3-1 修复：n_cells=0 会导致 ModuleList 为空，forward 时 pool 输入维度
        # 不匹配 classifier 输入，引发 shape mismatch。显式校验阻断早期失败。
        if n_cells < 1:
            raise ValueError(
                f"n_cells must be >= 1, got {n_cells}. "
                f"DARTSSupernet requires at least one DARTSCell."
            )
        if len(input_shape) < 1:
            raise ValueError(f"input_shape too short: {input_shape}")
        in_channels = int(input_shape[0])

        # stem: Conv1d + BN + ReLU
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c_stem, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c_stem),
            nn.ReLU(inplace=True),
        )

        # cells: N 个 DARTSCell 串联
        self.cells = nn.ModuleList()
        prev_c = c_stem
        for _ in range(n_cells):
            self.cells.append(DARTSCell(prev_c, c_cell, op_names=op_names))
            prev_c = c_cell

        # classifier: AdaptiveAvgPool1d + Linear
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(c_cell, num_classes)

        self.n_cells = n_cells
        self.c_stem = c_stem
        self.c_cell = c_cell
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入可能是 (B, C, L) 或 (B, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.pool(x).squeeze(-1)  # (B, c_cell)
        return self.classifier(x)

    def w_parameters(self):
        """w 参数（除 alpha 外的模型权重）— 用 SGD 更新。

        P2.3-1 修复：原实现 name.endswith("alpha") 会误过滤任何以 alpha 结尾的
        参数（如某 op 内部有 alpha 命名的 buffer）。改为精确匹配：
        - "alpha"（顶层，不会出现）
        - "cells.<idx>.alpha"（DARTSCell.alpha 的命名路径）
        """
        for name, p in self.named_parameters():
            # 精确匹配 DARTSCell.alpha 的命名路径
            # named_parameters() 返回 "cells.0.alpha" / "cells.1.alpha" 等
            if name == "alpha" or name.endswith(".alpha"):
                # 进一步验证：父模块必须是 DARTSCell（避免误过滤其他 alpha 命名）
                # 简化：DARTSCell.alpha 是唯一的 alpha 参数，且命名必含 "cells."
                if "cells." in name or name == "alpha":
                    continue
            yield p

    def alpha_parameters(self):
        """α 参数（架构参数）— 用 Adam 更新。"""
        for cell in self.cells:
            yield cell.alpha

    def alpha_dict(self) -> Dict[str, torch.Tensor]:
        """返回 α 参数 dict（{cell_idx_str: alpha_tensor}）。

        用于与 DARTSSampler 接口对齐（sampler.arch_alpha 是 dict）。
        """
        return {f"cell_{i}": cell.alpha for i, cell in enumerate(self.cells)}

    def discretize(self) -> Dict[str, Any]:
        """离散化：每个 cell argmax α → op 名。

        Returns:
            {"cell_0": "conv3", "cell_1": "maxpool", ...}
        """
        arch: Dict[str, Any] = {}
        for i, cell in enumerate(self.cells):
            arch[f"cell_{i}"] = cell.discretize()
        return arch

    def build_discrete_model(self) -> nn.Module:
        """根据 discretize() 结果构建离散化模型。

        每个 cell 只保留 argmax 选中的 op，丢弃其余 op。
        返回的模型与 supernet forward 行为一致（但无 α 加权）。

        用于训练完成后部署：去除 softmax 加权，提升推理效率。
        """
        return _DiscreteSupernet(self)


class _DiscreteCell(nn.Module):
    """离散化 cell：只保留 argmax 选中的 op。"""

    def __init__(self, cell: DARTSCell, op_name: str):
        super().__init__()
        if op_name not in cell.op_names:
            raise ValueError(
                f"op_name '{op_name}' not in cell.op_names {cell.op_names}"
            )
        idx = cell.op_names.index(op_name)
        # 直接引用原 op 的参数（共享权重）
        self.op = cell.ops[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _DiscreteSupernet(nn.Module):
    """从 DARTSSupernet 离散化得到的模型（无 α 加权）。

    每个 cell 只保留 argmax 选中的 op。其余结构与 supernet 一致。

    警示：参数共享（P1.3-3 修复）
    -------------------------------
    本类**直接引用** supernet 的 stem / cells / pool / classifier 引用，
    不做深拷贝。这意味着：

    1. _DiscreteSupernet 与原 supernet 共享同一份权重 tensor
    2. 对 _DiscreteSupernet 的训练会更新原 supernet 的权重（反之亦然）
    3. _DiscreteCell.op 同样直接引用原 cell.ops[idx] 的参数

    这是有意设计：离散化后继续微调时，复用超网已训练的权重而非从零开始。
    如需独立权重，请用 copy.deepcopy(_DiscreteSupernet(supernet)) 显式拷贝。
    """

    def __init__(self, supernet: DARTSSupernet):
        super().__init__()
        self.stem = supernet.stem
        self.cells = nn.ModuleList(
            [
                _DiscreteCell(cell, cell.discretize())
                for cell in supernet.cells
            ]
        )
        self.pool = supernet.pool
        self.classifier = supernet.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


__all__ = [
    "OP_NAMES",
    "DARTSCell",
    "DARTSSupernet",
]
