"""P3 阶段 8：PEFT 微调模块构建器。

实现 LoRA / Adapter / PrefixTuning / PromptTuning / Full 五种微调策略，
将基础模型（backbone）包装为可训练的 PEFTModel。

设计要点：
- LoRA: 冻结原始 Linear，添加 A/B 旁路（B 初始化为 zeros，初始输出为 0）
- Adapter: 冻结原始 Linear，添加 bottleneck 残差旁路
- PrefixTuning: 可学习 prefix 张量，拼接到输入序列前
- PromptTuning: 可学习 prompt 张量，拼接到输入序列前
- Full: 不注入任何 PEFT 模块，backbone 全部参数可训练

freeze_backbone 语义（P3-2 修复后统一）：
- freeze_backbone=True（默认）：冻结所有非 PEFT 参数（backbone 原始权重）。
  无论哪种 PEFT 方法，backbone 原始参数 requires_grad=False。
  LoRA/Adapter：注入模块的 A/B/bottleneck 参数仍 requires_grad=True。
  Prefix/Prompt：prefix/prompt 张量 requires_grad=True。
  Full：与 freeze_backbone=True 矛盾，会 raise（Full 必须不冻结）。
- freeze_backbone=False：解冻 backbone 所有参数，PEFT 模块参数也可训练。
  适用于"PEFT + 全量微调混合"场景（少用）。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================
# PEFT 层实现
# ============================================================
class LoRALayer(nn.Module):
    """LoRA 旁路层：包装 Linear，添加 A/B 低秩旁路。

    A: (rank, in_features) — kaiming_uniform 初始化
    B: (out_features, rank) — zeros 初始化（保证初始时 LoRA 输出为 0）
    forward: out = original(x) + B(A(dropout(x))) * (alpha / rank)

    P3-P2-1 修复：dropout 放置遵循 LoRA 原论文（Hu et al. 2021），
    对 LoRA 旁路的**输入**做 dropout（即 dropout(x) 后再进 A/B），
    而非对旁路输出做 dropout。原实现 dropout(B(A(x))) 偏离标准。
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        # 冻结原始权重（P3-2：统一由 _freeze_non_peft_params 处理，但此处保留
        # 以保证 LoRALayer 独立使用时也能正确冻结）
        for p in self.original.parameters():
            p.requires_grad = False
        in_features = original.in_features
        out_features = original.out_features
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P3-P2-1 修复：dropout 放在 A 之前（对 LoRA 输入做 dropout），
        # 符合 LoRA 原论文：B(A(dropout(x))) * scaling
        # F.linear(x, A) = x @ A.T  -> (..., rank)
        # F.linear(z, B) = z @ B.T  -> (..., out_features)
        lora_out = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.original(x) + lora_out * self.scaling


class AdapterLayer(nn.Module):
    """Adapter bottleneck 层：冻结原始 Linear，添加残差 bottleneck 旁路。

    结构：Linear(d, bottleneck) -> ReLU -> Linear(bottleneck, d)
    forward: out = original(x) + adapter(x)
    """

    def __init__(
        self,
        original: nn.Linear,
        bottleneck: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False
        in_features = original.in_features
        out_features = original.out_features
        self.adapter_down = nn.Linear(in_features, bottleneck, bias=False)
        self.adapter_up = nn.Linear(bottleneck, out_features, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bottleneck = bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.adapter_down(x)
        h = self.activation(h)
        h = self.adapter_up(h)
        return self.original(x) + self.dropout(h)


class PrefixTuningLayer(nn.Module):
    """Prefix Tuning 层：可学习 prefix 张量拼接到输入序列前。

    prefix: (prefix_len, d_model) — normal 初始化
    forward: x (batch, seq, d_model) -> cat([prefix, x], dim=1)

    P3-P2-3 修复：2D 输入时（非序列）静默 no-op 改为 warning，
    避免用户误以为 prefix 已拼接但实际未生效。
    """

    def __init__(self, d_model: int, prefix_len: int = 10):
        super().__init__()
        self.prefix = nn.Parameter(torch.zeros(prefix_len, d_model))
        nn.init.normal_(self.prefix, std=0.02)
        self.prefix_len = prefix_len
        self.d_model = d_model
        # 标记是否已对 2D 输入发过 warning（避免每个 batch 重复 log）
        self._warned_2d = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 期望 3D 输入 (batch, seq, d_model)
        if x.dim() == 2:
            if not self._warned_2d:
                logger.warning(
                    "PrefixTuningLayer: 2D 输入 (shape=%s) 无法拼接 prefix "
                    "（需 3D (B, seq, d_model)），跳过 prefix 注入。"
                    "若 backbone 输出是 2D，prefix_tuning 可能无效果。",
                    tuple(x.shape),
                )
                self._warned_2d = True
            return x  # 非序列输入不拼接
        batch_size = x.shape[0]
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prefix, x], dim=1)


class PromptTuningLayer(nn.Module):
    """Prompt Tuning 层：可学习 prompt 张量拼接到输入 embedding 前。

    prompt: (prompt_len, d_model) — normal 初始化
    forward: x (batch, seq, d_model) -> cat([prompt, x], dim=1)

    P3-P2-3 修复：2D 输入时（非序列）静默 no-op 改为 warning。
    """

    def __init__(self, d_model: int, prompt_len: int = 10):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(prompt_len, d_model))
        nn.init.normal_(self.prompt, std=0.02)
        self.prompt_len = prompt_len
        self.d_model = d_model
        self._warned_2d = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if not self._warned_2d:
                logger.warning(
                    "PromptTuningLayer: 2D 输入 (shape=%s) 无法拼接 prompt "
                    "（需 3D (B, seq, d_model)），跳过 prompt 注入。",
                    tuple(x.shape),
                )
                self._warned_2d = True
            return x
        batch_size = x.shape[0]
        prompt = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt, x], dim=1)


# ============================================================
# PEFTModel 包装器
# ============================================================
class PEFTModel(nn.Module):
    """PEFT 模型包装器：backbone + 注入的 PEFT 模块。

    LoRA/Adapter 模块在构建时直接替换 backbone 中的 Linear 层（in-place），
    forward 时自动生效。Prefix/Prompt 层作为独立模块，在 forward 时
    拼接到输入序列前。
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.lora_modules: nn.ModuleList = nn.ModuleList()
        self.adapter_modules: nn.ModuleList = nn.ModuleList()
        self.prefix_layer: Optional[PrefixTuningLayer] = None
        self.prompt_layer: Optional[PromptTuningLayer] = None
        self.peft_method: str = ""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # 对第一个位置参数（通常是输入张量）应用 prefix/prompt
        has_prompt_or_prefix = (
            self.prompt_layer is not None or self.prefix_layer is not None
        )
        if has_prompt_or_prefix and args and isinstance(args[0], torch.Tensor):
            x = args[0]
            backbone = self.backbone
            # P3-P2-11 修复：当 backbone 是 CSIFoundationModel 等带 patch_embedder
            # 的模型时，输入是 (B, C, L) 原始信号，prompt/prefix 应在 patch
            # embedding 之后注入（与 (B, n_patches, d_model) 拼接）。
            # 原实现直接对 (B, C, L) 拼接 prompt 会因 d_model 维度不匹配报错：
            #   prompt (B, prompt_len, d_model=128) vs x (B, C=342, L=2000)
            # 修复：检测 backbone 是否暴露 patch_embedder + encoder + pos_embed
            # 标准接口，是则穿透注入；否则保持原行为（兼容 _SequenceBackbone
            # 等直接接受 (B, seq, d_model) 的简单 backbone）。
            if (
                hasattr(backbone, "patch_embedder")
                and hasattr(backbone, "encoder")
                and hasattr(backbone, "pos_embed")
            ):
                # 复刻 CSIFoundationModel.encode(x) 流程，在 encoder 之前注入 prompt
                patches = backbone.patch_embedder(x) + backbone.pos_embed
                if self.prompt_layer is not None:
                    patches = self.prompt_layer(patches)
                if self.prefix_layer is not None:
                    patches = self.prefix_layer(patches)
                x = backbone.encoder(patches)
                if hasattr(backbone, "encoder_norm"):
                    x = backbone.encoder_norm(x)
                return x
            # 简单 backbone：直接对输入注入
            if self.prompt_layer is not None:
                x = self.prompt_layer(x)
            if self.prefix_layer is not None:
                x = self.prefix_layer(x)
            args = (x,) + args[1:]
        return self.backbone(*args, **kwargs)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征序列（供 DANN 等下游模块用）。

        与 forward 等价：应用 PEFT 模块（LoRA/Adapter/Prompt/Prefix）后
        返回 backbone 的特征输出 (B, n_patches, d_model)。

        语义上强调"供下游用"，与 CSIFoundationModel.encode_features 对齐。
        DANN 的 DANNCrossModalModel.forward 调用 backbone.encode_features(x)
        提取特征，PEFTModel 必须暴露此方法否则 AttributeError。

        Args:
            x: 输入张量 (B, C, L)

        Returns:
            特征序列 (B, n_patches, d_model)
        """
        return self.forward(x)


# ============================================================
# PEFTBuilder 构建器
# ============================================================
class PEFTBuilder:
    """PEFT 构建器：按 peft_method 分发到对应构建逻辑。"""

    @staticmethod
    def build(backbone: nn.Module, peft_params: Dict[str, Any]) -> nn.Module:
        """构建 PEFT 模型。

        Args:
            backbone: 基础模型（nn.Module）
            peft_params: PEFT 参数字典（含 peft_method / peft_rank / 等）
                        值可能是 str（SP 采样）或原生类型（直接调用）

        Returns:
            PEFTModel 实例

        Raises:
            ValueError: backbone 已含 PEFT 模块（重复 build 会静默失败）
                       / 未知 peft_method
                       / full + freeze_backbone=True（语义矛盾）
        """
        # P3-3 修复：检查 backbone 是否已含 PEFT 模块，避免重复 build 静默失败。
        # _set_submodule 替换已含 PEFT 的 Linear 会破坏原有 PEFT 结构。
        if isinstance(backbone, PEFTModel):
            raise ValueError(
                f"backbone 已是 PEFTModel（peft_method={backbone.peft_method}），"
                f"不能重复 build。请用原始 backbone 而非 PEFTModel 作为输入。"
            )
        # 检查 backbone 内部是否已含 LoRALayer / AdapterLayer（被注入过）
        for name, module in backbone.named_modules():
            if isinstance(module, (LoRALayer, AdapterLayer,
                                   PrefixTuningLayer, PromptTuningLayer)):
                raise ValueError(
                    f"backbone 已含 PEFT 模块（{name}: {type(module).__name__}），"
                    f"不能重复 build。请用原始 backbone。"
                )

        method = str(peft_params.get("peft_method", "lora")).lower()
        peft_model = PEFTModel(backbone)
        peft_model.peft_method = method

        # P3-2 修复：统一 freeze_backbone 语义。
        # 先构建 PEFT 模块（各 _build_* 方法不再单独处理 freeze），
        # 最后统一调用 _freeze_non_peft_params 处理冻结逻辑。
        if method == "lora":
            PEFTBuilder._build_lora(peft_model, peft_params)
        elif method == "adapter":
            PEFTBuilder._build_adapter(peft_model, peft_params)
        elif method == "prefix_tuning":
            PEFTBuilder._build_prefix_tuning(peft_model, peft_params)
        elif method == "prompt_tuning":
            PEFTBuilder._build_prompt_tuning(peft_model, peft_params)
        elif method == "full":
            PEFTBuilder._build_full(peft_model, peft_params)
        else:
            raise ValueError(
                f"Unknown peft_method: '{method}'. "
                f"Supported: lora / adapter / prefix_tuning / prompt_tuning / full"
            )

        # P3-2 修复：统一冻结语义（在所有 PEFT 模块构建后执行）
        freeze_backbone = PEFTBuilder._coerce_bool(
            peft_params.get("freeze_backbone"), True
        )
        PEFTBuilder._apply_freeze_backbone(peft_model, method, freeze_backbone)

        return peft_model

    @staticmethod
    def _apply_freeze_backbone(
        peft_model: "PEFTModel", method: str, freeze_backbone: bool
    ) -> None:
        """统一处理 freeze_backbone 语义（P3-2 修复）。

        语义：
        - freeze_backbone=True：冻结所有 backbone 原始参数（非 PEFT 模块参数）。
          PEFT 模块（LoRA A/B / Adapter bottleneck / Prefix / Prompt）仍可训练。
          Full + freeze_backbone=True 自动转为 False（warning）— full 无 PEFT
          模块可冻结，backbone 必须可训练，否则模型完全不可训练（无意义配置）。
        - freeze_backbone=False：解冻 backbone 所有参数（含 PEFT 模块）。
        """
        if method == "full" and freeze_backbone:
            # P3-2 修复（向后兼容调整）：full + freeze_backbone=True 不再 raise，
            # 改为 warning + 自动转 False。原因：
            # 1. PEFTConfig 默认 freeze_backbone=True（为 LoRA 等设计），
            #    full 方法用默认配置就 raise 会破坏所有 full 的默认使用
            # 2. full 无 PEFT 模块可冻结，freeze_backbone=True 语义上等于
            #    "模型完全冻结"，这是无意义配置，自动转 False 更友好
            logger.warning(
                "peft_method='full' 与 freeze_backbone=True 语义矛盾："
                "Full 微调必须解冻所有参数。自动设 freeze_backbone=False。"
                "若需冻结 backbone，请改用 lora/adapter 等 PEFT 方法。"
            )
            freeze_backbone = False

        # 遍历 backbone 所有参数，识别哪些是"原始参数"（非 PEFT 模块持有）
        # PEFT 模块挂在 backbone 内部（_set_submodule 替换）或 PEFTModel 顶层
        peft_param_ids = set()
        # LoRA/Adapter：A/B/bottleneck 等参数在 PEFTModel.lora_modules/adapter_modules
        # 注意：LoRALayer.original 是 backbone 原参数，不应加入 peft_param_ids
        for lora in peft_model.lora_modules:
            for p in lora.parameters():
                if p is not lora.original.weight and p is not lora.original.bias:
                    peft_param_ids.add(id(p))
        for adapter in peft_model.adapter_modules:
            for p in adapter.parameters():
                if p is not adapter.original.weight and p is not adapter.original.bias:
                    peft_param_ids.add(id(p))
        # Prefix/Prompt：prefix/prompt 张量
        if peft_model.prefix_layer is not None:
            for p in peft_model.prefix_layer.parameters():
                peft_param_ids.add(id(p))
        if peft_model.prompt_layer is not None:
            for p in peft_model.prompt_layer.parameters():
                peft_param_ids.add(id(p))

        # 遍历 backbone 所有参数，冻结非 PEFT 参数
        # P3-2 修复（bug 修正）：原实现 for p in backbone.parameters(): p.requires_grad=False
        # 会误冻结 LoRA A/B（因为 LoRALayer 通过 _set_submodule 挂在 backbone 内部，
        # backbone.parameters() 会遍历到 A/B）。现用 peft_param_ids 过滤。
        for p in peft_model.backbone.parameters():
            if freeze_backbone:
                if id(p) in peft_param_ids:
                    # PEFT 模块参数（A/B/bottleneck/prefix/prompt）保持可训练
                    p.requires_grad = True
                else:
                    # backbone 原始参数冻结
                    p.requires_grad = False
            else:
                p.requires_grad = True

    # ------------------------------------------------------------
    # 类型强制转换辅助（兼容 SP str 采样与原生类型）
    # ------------------------------------------------------------
    @staticmethod
    def _coerce_int(val: Any, default: int) -> int:
        if val is None:
            return default
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float)):
            return int(val)
        return int(str(val))

    @staticmethod
    def _coerce_float(val: Any, default: float) -> float:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val))

    @staticmethod
    def _coerce_bool(val: Any, default: bool) -> bool:
        """将任意值转为 bool（兼容 SP str 采样与原生类型）。

        P3-P2-2 修复：原实现仅识别 "true" 字符串，不识别 "1"/"0"/"yes"/"no"。
        现支持：
        - bool / int / float：直接 bool()
        - str："true"/"1"/"yes" -> True；"false"/"0"/"no" -> False
        - None：返回 default
        """
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        s = str(val).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        # 兜底：非空字符串视为 True（与 Python bool(str) 语义一致）
        return bool(s)

    @staticmethod
    def _infer_d_model(backbone: nn.Module) -> int:
        """从 backbone 推断 d_model（prompt/prefix 张量的维度）。

        优先使用 backbone.d_model 属性（CSIFoundationModel 显式声明 d_model=128，
        是 patch_embedder 输出后 token 的真实维度）；缺失时兜底取第一个 Linear
        的 in_features（适用于 _SequenceBackbone 等直接对 (B, seq, d_model)
        输入做投影的简单 backbone）。

        P3-P2-11 修复：原实现只取第一个 Linear 的 in_features，对 CSIFoundationModel
        会取到 CSIPatchEmbedder.proj.in_features = patch_len * C（如 NTU-Fi 20*342=6840），
        而 PromptTuning 的 prompt 应在 patch embedding 之后注入，d_model 应为 128。
        导致 prompt (10, 6840) 与 patch tokens (B, n_patches, 128) cat 时 dim=2 不匹配。
        """
        # 优先使用显式 d_model 属性
        d = getattr(backbone, "d_model", None)
        if isinstance(d, int) and d > 0:
            return d
        # 兜底：第一个 Linear 的 in_features
        for module in backbone.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 128

    @staticmethod
    def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
        """通过点分名称替换子模块。"""
        parts = dotted_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    @staticmethod
    def _should_inject_lora(name: str, module: nn.Module, target: str) -> bool:
        """判断是否对该模块注入 LoRA/Adapter。

        target:
        - "query" -> name 含 "query"
        - "value" -> name 含 "value"
        - "query_value" -> name 含 query 或 value
        - "all" -> 所有 Linear
        """
        if not isinstance(module, nn.Linear):
            return False
        target_lower = str(target).lower()
        name_lower = name.lower()
        if target_lower == "all":
            return True
        if target_lower == "query":
            return "query" in name_lower
        if target_lower == "value":
            return "value" in name_lower
        if target_lower == "query_value":
            return "query" in name_lower or "value" in name_lower
        return False

    # ------------------------------------------------------------
    # 各 PEFT 方法构建逻辑
    # ------------------------------------------------------------
    @staticmethod
    def _check_required_params(
        method: str, params: Dict[str, Any], required_keys: List[str]
    ) -> None:
        """P3-P2-4 修复：检查必需参数是否存在，缺失时 log warning。

        不 raise（保持向后兼容，用 _coerce_* 的 default 兜底），
        但 warning 帮助用户诊断 SP 采样参数不完整或拼写错误。

        Args:
            method: peft_method 名（lora/adapter/...）
            params: PEFT 参数字典
            required_keys: 该方法所需的参数名列表
        """
        missing = [k for k in required_keys if k not in params or params[k] is None]
        if missing:
            logger.warning(
                "PEFTBuilder._build_%s: 缺失参数 %s，将使用默认值。"
                "若来自 SP 采样，请检查 search_space 是否包含这些参数。",
                method, missing,
            )

    @staticmethod
    def _build_lora(peft_model: PEFTModel, params: Dict[str, Any]) -> None:
        # P3-P2-4 修复：缺失参数 warning（不阻断，用 default 兜底）
        PEFTBuilder._check_required_params(
            "lora", params,
            ["peft_rank", "peft_alpha", "peft_dropout", "peft_target_modules"],
        )
        rank = PEFTBuilder._coerce_int(params.get("peft_rank"), 8)
        alpha = PEFTBuilder._coerce_int(params.get("peft_alpha"), 1)
        dropout = PEFTBuilder._coerce_float(params.get("peft_dropout"), 0.0)
        target = str(params.get("peft_target_modules", "query_value"))
        # P3-2 修复：freeze_backbone 由 _apply_freeze_backbone 统一处理

        backbone = peft_model.backbone
        # 快照命名模块，避免迭代中修改
        linear_specs = [
            (name, mod) for name, mod in backbone.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        for name, module in linear_specs:
            if not PEFTBuilder._should_inject_lora(name, module, target):
                continue
            lora = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
            PEFTBuilder._set_submodule(backbone, name, lora)
            peft_model.lora_modules.append(lora)

    @staticmethod
    def _build_adapter(peft_model: PEFTModel, params: Dict[str, Any]) -> None:
        # P3-P2-4 修复：缺失参数 warning
        PEFTBuilder._check_required_params(
            "adapter", params,
            ["adapter_bottleneck", "peft_dropout", "peft_target_modules"],
        )
        bottleneck = PEFTBuilder._coerce_int(params.get("adapter_bottleneck"), 128)
        dropout = PEFTBuilder._coerce_float(params.get("peft_dropout"), 0.0)
        target = str(params.get("peft_target_modules", "all"))
        # P3-2 修复：freeze_backbone 由 _apply_freeze_backbone 统一处理

        backbone = peft_model.backbone
        linear_specs = [
            (name, mod) for name, mod in backbone.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        for name, module in linear_specs:
            if not PEFTBuilder._should_inject_lora(name, module, target):
                continue
            adapter = AdapterLayer(module, bottleneck=bottleneck, dropout=dropout)
            PEFTBuilder._set_submodule(backbone, name, adapter)
            peft_model.adapter_modules.append(adapter)

    @staticmethod
    def _build_prefix_tuning(peft_model: PEFTModel, params: Dict[str, Any]) -> None:
        # P3-P2-4 修复：缺失参数 warning
        PEFTBuilder._check_required_params("prefix_tuning", params, ["prompt_length"])
        prefix_len = PEFTBuilder._coerce_int(params.get("prompt_length"), 10)
        # P3-2 修复：freeze_backbone 由 _apply_freeze_backbone 统一处理
        d_model = PEFTBuilder._infer_d_model(peft_model.backbone)

        peft_model.prefix_layer = PrefixTuningLayer(d_model=d_model, prefix_len=prefix_len)

    @staticmethod
    def _build_prompt_tuning(peft_model: PEFTModel, params: Dict[str, Any]) -> None:
        # P3-P2-4 修复：缺失参数 warning
        PEFTBuilder._check_required_params("prompt_tuning", params, ["prompt_length"])
        prompt_len = PEFTBuilder._coerce_int(params.get("prompt_length"), 10)
        # P3-2 修复：freeze_backbone 由 _apply_freeze_backbone 统一处理
        d_model = PEFTBuilder._infer_d_model(peft_model.backbone)

        peft_model.prompt_layer = PromptTuningLayer(d_model=d_model, prompt_len=prompt_len)

    @staticmethod
    def _build_full(peft_model: PEFTModel, params: Dict[str, Any]) -> None:
        # Full fine-tuning: 不注入任何 PEFT 模块。
        # P3-2 修复：参数解冻由 _apply_freeze_backbone 统一处理
        # （Full + freeze_backbone=True 会在那里 raise）
        pass


__all__ = [
    "LoRALayer",
    "AdapterLayer",
    "PrefixTuningLayer",
    "PromptTuningLayer",
    "PEFTModel",
    "PEFTBuilder",
]
