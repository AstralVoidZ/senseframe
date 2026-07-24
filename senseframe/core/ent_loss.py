"""
自监督 EntLoss（KL + EH + HE + KDE）。

从 engine/self_supervised.py 提取到 core 层，消除 core.losses → engine 的循环依赖。
EntLoss 是纯 nn.Module，仅依赖 torch，无 engine 层依赖。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# KDE loss 缩放因子（AutoFi 论文 §3.2：KDE 项量级远小于 KL/EH/HE，需放大 100 倍
# 才能在 final-kde 总损失中起到有效正则作用）
_KDE_LOSS_SCALE = 100


class EntLoss(nn.Module):
    """AutoFi 自监督损失：KL + EH + HE + KDE。"""

    def __init__(self, tau: float = 1.0, eps: float = 1e-5, lam1: float = 0.0, lam2: float = 0.5):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.lam1 = lam1
        self.lam2 = lam2

    def forward(self, feat1, feat2):
        probs1 = F.softmax(feat1, dim=-1)
        probs2 = F.softmax(feat2, dim=-1)

        loss = {}
        loss["kl"] = 0.5 * (self._kl(probs1, probs2) + self._kl(probs2, probs1))

        sharpened_probs1 = F.softmax(feat1 / self.tau, dim=-1)
        sharpened_probs2 = F.softmax(feat2 / self.tau, dim=-1)
        loss["eh"] = 0.5 * (self._eh(sharpened_probs1) + self._eh(sharpened_probs2))
        loss["he"] = 0.5 * (self._he(sharpened_probs1) + self._he(sharpened_probs2))

        loss["final"] = loss["kl"] + ((1 + self.lam1) * loss["eh"] - self.lam2 * loss["he"])

        loss["kde"] = self._cosine_similarity_loss(feat1, feat2)
        loss["final-kde"] = loss["kde"] * _KDE_LOSS_SCALE + loss["final"]

        return loss

    def _kl(self, probs1, probs2):
        kl = (probs1 * (probs1 + self.eps).log() - probs1 * (probs2 + self.eps).log()).sum(dim=1)
        return kl.mean()

    def _he(self, probs):
        mean = probs.mean(dim=0)
        ent = -(mean * (mean + self.eps).log()).sum()
        return ent

    def _eh(self, probs):
        ent = -(probs * (probs + self.eps).log()).sum(dim=1)
        return ent.mean()

    def _cosine_similarity_loss(self, output_net, target_net, eps=1e-7):
        output_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_norm + eps)
        output_net[output_net != output_net] = 0

        target_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_norm + eps)
        target_net[target_net != target_net] = 0

        model_sim = torch.mm(output_net, output_net.transpose(0, 1))
        target_sim = torch.mm(target_net, target_net.transpose(0, 1))

        model_sim = (model_sim + 1.0) / 2.0
        target_sim = (target_sim + 1.0) / 2.0

        model_sim = model_sim / torch.sum(model_sim, dim=1, keepdim=True)
        target_sim = target_sim / torch.sum(target_sim, dim=1, keepdim=True)

        loss = torch.mean(target_sim * torch.log((target_sim + eps) / (model_sim + eps)))
        return loss
