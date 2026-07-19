"""P3 验证评估共享基础设施（10.2-10.6 通用）。

提供：
- ExperimentConfig dataclass：单次实验配置（实验组 ID / 预训练 / 微调 / 数据集 / 超参）
- ExperimentResult dataclass：单次实验结果（指标 + 训练时间 + 可训练参数量）
- run_single_experiment()：执行单次实验的统一入口（10.2-10.5 共用）
- aggregate_results()：汇总多组实验结果到 DataFrame / CSV
- compute_transfer_gain()：跨场景迁移增益计算（10.4 用）
- compute_search_effectiveness()：SP 搜索相对固定配置的提升（10.5 用）

设计原则：
- 所有实验脚本（exp_10_*.py）通过 import本模块复用基础设施
- 实验配置与结果序列化为 JSON，便于跨脚本传递与结果归档
- CSI 训练流程用简化 torch 循环（不走 Lightning），透明可控易调试
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根加入 sys.path（从 scripts/p3_eval_common.py 向上两级），
# 让 `from senseframe...` / `from scripts...` 在直接 python scripts/xxx.py 运行时也能工作
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ============================================================
# SenseFi 路径自动配置（CSI 场景必需）
# ============================================================
# 多候选路径：Windows 开发环境 + WSL 部署环境
_SENSEFI_CANDIDATES = [
    r"<WIFI_CSI_BENCHMARK_ROOT>",  # Windows
    str(Path.home() / "projects" / "thepot" / "CSI_DATASETS" / "WiFi-CSI-Sensing-Benchmark"),  # WSL
]
if "SENSEFRAME_SENSEFI_PATH" not in os.environ:
    for _cand in _SENSEFI_CANDIDATES:
        if Path(_cand).exists():
            os.environ["SENSEFRAME_SENSEFI_PATH"] = _cand
            break

# CSI 数据集根目录（多候选探测）
_CSI_DATA_CANDIDATES = [
    Path("data/wifi_csi").absolute(),  # Windows 相对路径
    Path.home() / "projects" / "thepot" / "CSI_DATASETS",  # WSL
]
for _cand in _CSI_DATA_CANDIDATES:
    if _cand.exists():
        _CSI_DATA_ROOT = str(_cand)
        break
else:
    _CSI_DATA_ROOT = str(_CSI_DATA_CANDIDATES[0])  # 默认值，加载时报错

# EEG 数据集根目录（多候选探测）
_EEG_DATA_CANDIDATES = {
    "physionet": [
        Path("data/eeg/physionet/eegmmidb").absolute(),  # Windows
        Path.home() / "projects" / "thepot" / "data" / "eeg" / "physionet" / "eegmmidb",  # WSL
    ],
    "bci_iv_2a": [
        Path("data/eeg/bci_iv_2a").absolute(),  # Windows
        Path.home() / "projects" / "thepot" / "data" / "eeg" / "bci_iv_2a",  # WSL
    ],
}


# ============================================================
# CSI 数据集元数据（input_shape → CSIFoundationModel 配置）
# ============================================================
# 每个数据集的原始 input_shape 和适配到 (C, L) 的 reshape 方式 + patch_len
# 原始 shape 来自 senseframe/scenes/wifi_csi/_register.py 的 DATASET_INFO
CSI_DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "UT_HAR_data": {
        "raw_shape": (1, 250, 90),       # (antenna, subcarrier, time)
        "reshape_to": (90, 250),          # (C=90, L=250) — flatten antenna*subcarrier
        "patch_len": 10,                  # 250 % 10 == 0
        "num_classes": 7,
    },
    "NTU-Fi_HAR": {
        # SenseFrame CSIMatLoader 实际产出 (342, 2000)：342 = 3 antenna × 114 subcarrier，
        # 2000 = 4s × 500Hz（NTU-Fi 原始 .mat 时长 4 秒，500 sample/s）
        "raw_shape": (342, 2000),
        "reshape_to": (342, 2000),         # 已是 (C, L)，无需 reshape
        "patch_len": 20,                   # 2000 % 20 == 0 → 100 patches
        "num_classes": 6,
    },
    "NTU-Fi-HumanID": {
        "raw_shape": (342, 2000),
        "reshape_to": (342, 2000),
        "patch_len": 20,
        "num_classes": 14,
    },
    "Widar": {
        "raw_shape": (22, 20, 20),        # (channel, h, w) BVP
        "reshape_to": (22, 400),          # (C=22, L=400) — flatten h*w
        "patch_len": 20,                  # 400 % 20 == 0
        "num_classes": 22,
    },
}

# EEG 数据集元数据
EEG_DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "PhysioNet_MI": {
        "input_shape": (64, 480),         # (channels, time)
        "patch_len": 20,                  # 480 % 20 == 0
        "num_classes": 2,
        "data_root": str(next((p for p in _EEG_DATA_CANDIDATES["physionet"] if p.exists()), _EEG_DATA_CANDIDATES["physionet"][0])),
    },
    "BCI_Competition_IV_2a": {
        "input_shape": (22, 1000),        # (channels, time)
        "patch_len": 20,                  # 1000 % 20 == 0
        "num_classes": 4,
        "data_root": str(next((p for p in _EEG_DATA_CANDIDATES["bci_iv_2a"] if p.exists()), _EEG_DATA_CANDIDATES["bci_iv_2a"][0])),
    },
}

# RadioML 数据集元数据（待 RadioML 2018.01A 下载完成后启用）
RADIO_DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "RadioML2018": {
        "input_shape": (2, 1024),         # (IQ channels, time)
        "patch_len": 16,                  # 1024 % 16 == 0
        "num_classes": 24,
        "data_root": "data/radio/radioml_2018_01a",
    },
}


def _get_dataset_config(dataset_name: str) -> Optional[Dict[str, Any]]:
    """根据数据集名查找配置（CSI / EEG / Radio 三类）。"""
    for cfg_dict in (CSI_DATASET_CONFIG, EEG_DATASET_CONFIG, RADIO_DATASET_CONFIG):
        if dataset_name in cfg_dict:
            return cfg_dict[dataset_name]
    return None


def _reshape_sample(x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """把样本 reshape 到 (C, L)。

    根据输入 shape 与 target_shape 的维度关系，选择正确的 reshape/permute 方式，
    避免直接 flatten 导致数据维度语义错乱。

    支持的输入 shape：
    - 2D (C, L) → 直接返回（若已匹配）
    - 3D (a, b, c) → 根据 target_shape 选择：
        * (a*b, c)：flatten 前两维（NTU-Fi: (3,114,500)→(342,500)）
        * (a, b*c)：flatten 后两维（Widar: (22,20,20)→(22,400)）
        * (c, b)   当 a=1：squeeze + transpose（UT_HAR: (1,250,90)→(90,250)）
        * (a*c, b)：permute(0,2,1) + flatten 前两维
    """
    if x.dim() == 2:
        if x.shape == target_shape:
            return x.float()
        if x.shape[0] == target_shape[1] and x.shape[1] == target_shape[0]:
            return x.t().contiguous().float()
        return x.reshape(target_shape).float()

    if x.dim() == 3:
        a, b, c = x.shape
        target_C, target_L = target_shape
        # (a, b, c) → (a*b, c)：flatten 前两维
        if a * b == target_C and c == target_L:
            return x.reshape(a * b, c).float()
        # (a, b, c) → (a, b*c)：flatten 后两维
        if a == target_C and b * c == target_L:
            return x.reshape(a, b * c).float()
        # (1, b, c) → (c, b)：squeeze + transpose
        if a == 1 and b == target_L and c == target_C:
            return x.squeeze(0).t().contiguous().float()
        # (a, b, c) → (a*c, b)：permute + flatten 前两维
        if a * c == target_C and b == target_L:
            return x.permute(0, 2, 1).reshape(a * c, b).contiguous().float()

    # 兜底：直接 reshape
    return x.reshape(target_shape).float()


# ============================================================
# CSI 分类头（基础模型 backbone + mean pooling + Linear 分类）
# ============================================================
class CSIClassifier(nn.Module):
    """CSI 分类器：backbone + mean pooling + Linear 分类头。

    backbone 可以是：
    - CSIFoundationModel（scratch / full 微调）
    - PEFTModel（LoRA / Adapter / Prefix / Prompt 微调，包装 backbone）

    forward 流程：
    1. backbone(x) → (B, n_patches, d_model) 特征序列
    2. mean pooling → (B, d_model) 全局特征
    3. Linear(d_model, num_classes) → (B, num_classes) logits
    """

    def __init__(self, backbone: nn.Module, d_model: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone 返回 (B, n_patches, d_model)
        features = self.backbone(x)
        # mean pooling over patches
        pooled = features.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)  # (B, num_classes)


# ============================================================
# 实验配置 / 结果数据结构
# ============================================================
@dataclass
class ExperimentConfig:
    """单次实验配置（10.2-10.5 共用）。

    字段对应 P3 验证方案中的实验组（A1-A5 / B1-B8 / C1-C5）。
    """
    experiment_id: str          # 如 "A1", "B2", "C4"
    experiment_group: str       # "single_scene" / "cross_domain" / "sp_search"
    pretrain_source: str        # "none" / "csi_4datasets" / "radioml" / "eegmmidb"
    finetune_method: str        # "scratch" / "full" / "lora" / "adapter" / "prompt_tuning"
    target_dataset: str         # "UT_HAR_data" / "NTU-Fi_HAR" / "RadioML2018" / "PhysioNet_MI" / ...
    finetune_params: Dict[str, Any] = field(default_factory=dict)
    # SP 搜索相关（仅 C4/C5 用）
    sp_search: Optional[Dict[str, Any]] = None  # {"sampler": "random", "n_trials": 20}
    # 输出目录
    output_dir: str = "results/p3_validation"
    # 随机种子
    seed: int = 42
    # 训练超参（默认值，可被 finetune_params 覆盖）
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    # MAE 预训练超参（仅 pretrain_source != "none" 时用）
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    pretrain_mask_ratio: float = 0.75


@dataclass
class ExperimentResult:
    """单次实验结果（10.2-10.5 共用）。

    所有指标均为最终 epoch 的值（非最佳 epoch）。
    """
    experiment_id: str
    status: str = "pending"     # "pending" / "running" / "success" / "failed"
    val_accuracy: float = 0.0
    macro_f1: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    total_epochs: int = 0
    training_time_seconds: float = 0.0
    # SP 搜索特有（仅 C4/C5）
    best_params: Optional[Dict[str, Any]] = None
    n_completed: int = 0
    n_trials: int = 0
    search_cost_seconds: float = 0.0
    # 错误信息（status=failed 时）
    error: Optional[str] = None
    # 元数据
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# 数据加载
# ============================================================
def _load_csi_dataset(dataset_name: str, data_root: str, learning_mode: str = "supervised"):
    """加载 CSI 数据集（通过 SenseFrame scene API）。"""
    from senseframe.scenes import activate_lazy_scenes, get_scene
    activate_lazy_scenes()
    scene = get_scene("wifi_csi")
    return scene.load_dataset(dataset_name, root=data_root, learning_mode=learning_mode)


def _load_eeg_dataset(dataset_name: str, data_root: str):
    """加载 EEG 数据集（直接用 Dataset 类，不走 scene API）。"""
    from senseframe.scenes.eeg.datasets import (
        PhysioNetEegmmidbDataset,
        BCICompetitionIV2aDataset,
    )
    if dataset_name == "PhysioNet_MI":
        full_ds = PhysioNetEegmmidbDataset(root=data_root, subjects=list(range(1, 11)))
    elif dataset_name == "BCI_Competition_IV_2a":
        full_ds = BCICompetitionIV2aDataset(root=data_root)
    else:
        raise ValueError(f"Unknown EEG dataset: {dataset_name}")

    n = len(full_ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    return torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )


# 已在 loader 层应用归一化的数据集白名单（collate_fn 跳过归一化避免重复）
# - UT_HAR_data: tensor_loader.py 第 95 行 `data_norm = norm_strategy.apply(data)` 已做 ZScore
# - NTU-Fi_HAR / NTU-Fi-HumanID / Widar: CSIMatLoader/CSVFolderLoader 未在加载时归一化，
#   原始幅值进入模型，collate_fn 必须显式 apply（见 project_memory.md 第 129 条教训）
_LOADER_NORMALIZED_DATASETS = {"UT_HAR_data"}


def _make_collate_fn(target_shape: Tuple[int, int], dataset_name: Optional[str] = None):
    """构造 collate_fn：reshape 样本到 (C, L) + 应用注册的归一化策略（若 loader 未做）。

    归一化修复（对齐 project_memory.md 第 129 条教训）：
    NTU-Fi_HAR 原始 CSI 振幅 ~42，未归一化时 MAE loss=1722，LoRA 训练不收敛。
    项目已有 register_normalization 机制（NTU-Fi: ZScore(42.32, 4.98)，
    UT_HAR: ZScore(17.65, 5.90)）。但 UT_HAR_data 的 TensorLoader 已在加载时归一化，
    collate_fn 再 apply 会重复归一化破坏数据。

    Args:
        target_shape: (C, L) 目标形状
        dataset_name: 数据集名（用于查归一化策略）；None 表示不归一化
    """
    # 提前解析归一化策略（避免每个 batch 都查注册表）
    norm_strategy = None
    if dataset_name is not None and dataset_name not in _LOADER_NORMALIZED_DATASETS:
        from senseframe.registry import get_normalization_or_none
        norm_strategy = get_normalization_or_none(dataset_name)
        if norm_strategy is None:
            logger.warning(
                "collate_fn: dataset '%s' has no registered normalization; "
                "raw amplitudes will enter the model (may cause MAE loss explosion)",
                dataset_name,
            )
        elif not norm_strategy.is_noop():
            logger.info(
                "collate_fn: dataset '%s' applying normalization %s",
                dataset_name, norm_strategy.to_dict(),
            )
    elif dataset_name in _LOADER_NORMALIZED_DATASETS:
        logger.info(
            "collate_fn: dataset '%s' skip collate-time normalization "
            "(loader has already applied it)", dataset_name,
        )

    def collate(batch):
        xs, ys = zip(*batch)
        xs_tensors = []
        for x in xs:
            t = _reshape_sample(torch.as_tensor(x), target_shape)
            if norm_strategy is not None and not norm_strategy.is_noop():
                # NormalizationStrategy.apply 接 numpy，转回去再转回来
                t = torch.as_tensor(
                    norm_strategy.apply(t.cpu().numpy()),
                    dtype=t.dtype,
                ).to(t.device)
            xs_tensors.append(t)
        xs_tensor = torch.stack(xs_tensors)
        ys_tensor = torch.tensor([int(y) for y in ys], dtype=torch.long)
        return xs_tensor, ys_tensor
    return collate


# ============================================================
# 模型构建
# ============================================================
def _build_backbone(
    dataset_config: Dict[str, Any],
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
) -> nn.Module:
    """构建 CSIFoundationModel backbone。"""
    from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel

    input_shape = dataset_config["reshape_to"] if "reshape_to" in dataset_config else dataset_config["input_shape"]
    patch_len = dataset_config["patch_len"]

    return CSIFoundationModel(
        input_shape=input_shape,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=2,
        patch_len=patch_len,
        decoder_dim=64,
    )


def _build_peft_model(backbone: nn.Module, finetune_method: str, finetune_params: Dict[str, Any]) -> nn.Module:
    """根据微调方法构建 PEFT 模型。"""
    from senseframe.automl.peft_builder import PEFTBuilder

    if finetune_method == "scratch":
        return backbone  # 不加 PEFT，全量训练
    elif finetune_method == "full":
        return PEFTBuilder.build(backbone, {"peft_method": "full", "freeze_backbone": False})
    elif finetune_method == "lora":
        params = {
            "peft_method": "lora",
            "peft_rank": finetune_params.get("peft_rank", 8),
            "peft_alpha": finetune_params.get("peft_alpha", 16),
            "peft_target_modules": "query_value",
            "freeze_backbone": True,
        }
        return PEFTBuilder.build(backbone, params)
    elif finetune_method == "adapter":
        params = {
            "peft_method": "adapter",
            "adapter_bottleneck": finetune_params.get("adapter_bottleneck", 128),
            "peft_target_modules": "all",
            "freeze_backbone": True,
        }
        return PEFTBuilder.build(backbone, params)
    elif finetune_method == "prompt_tuning":
        params = {
            "peft_method": "prompt_tuning",
            "prompt_length": finetune_params.get("prompt_length", 10),
            "freeze_backbone": True,
        }
        return PEFTBuilder.build(backbone, params)
    else:
        raise ValueError(f"Unknown finetune_method: {finetune_method}")


# ============================================================
# 训练循环
# ============================================================
def _train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    d_model: int = 128,
) -> Tuple[float, float, int]:
    """训练分类器，返回 (val_accuracy, macro_f1, trainable_params)。

    model 是 CSIClassifier（backbone + 分类头）。
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_acc = 0.0
    best_macro_f1 = 0.0

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).long()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        # ---- 验证 ----
        model.eval()
        all_preds = []
        all_labels = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).long()
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / max(val_total, 1)
        # 计算 macro F1
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_macro_f1 = macro_f1

        # 每个 epoch 都输出日志：便于观察训练曲线 + 及时发现卡死
        # （原逻辑仅在 epoch 1 和每 10 个 epoch 输出，长 epoch 训练时易误判卡死）
        mean_loss = sum(train_losses) / max(len(train_losses), 1)
        logger.info(
            "epoch %d/%d: train_loss=%.4f, val_acc=%.4f, macro_f1=%.4f",
            epoch + 1, epochs, mean_loss, val_acc, macro_f1,
        )

    return best_val_acc, best_macro_f1, trainable_params


# ============================================================
# 实验执行入口
# ============================================================
def run_single_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
) -> ExperimentResult:
    """执行单次 P3 验证实验。

    流程：
    1. 根据 target_dataset 加载数据集（CSI / EEG / Radio）
    2. 根据 pretrain_source 决定是否 MAE 预训练 backbone
    3. 根据 finetune_method 构建 PEFT 模型（或 scratch 全量训练）
    4. 加分类头，训练 + 评估
    5. 收集指标到 ExperimentResult

    Args:
        config: 实验配置
        dry_run: True 时跳过真实训练，仅验证配置 + 返回 pending 状态结果

    Returns:
        ExperimentResult
    """
    result = ExperimentResult(
        experiment_id=config.experiment_id,
        status="pending",
        config_snapshot=asdict(config),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    if dry_run:
        logger.info("[dry_run] experiment %s config validated", config.experiment_id)
        result.status = "pending"
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return result

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("experiment %s: device=%s", config.experiment_id, device)

    try:
        result.status = "running"

        # ---- 1. 查找数据集配置 ----
        dataset_config = _get_dataset_config(config.target_dataset)
        if dataset_config is None:
            raise ValueError(f"Unknown target_dataset: {config.target_dataset}")

        is_csi = config.target_dataset in CSI_DATASET_CONFIG
        is_eeg = config.target_dataset in EEG_DATASET_CONFIG
        is_radio = config.target_dataset in RADIO_DATASET_CONFIG

        # ---- 2. 加载数据集 ----
        logger.info("experiment %s: loading dataset %s",
                    config.experiment_id, config.target_dataset)

        if is_csi:
            data_root = _CSI_DATA_ROOT
            if config.pretrain_source == "csi_4datasets":
                # P3 验证 A2-A5：在 target_dataset 上做 MAE 预训练 + PEFT 微调
                # 设计说明：不使用 scene 的 self_supervised 模式加载，原因：
                # 1. UT_HAR_data 是 .npy 张量，没有 unsupervised/supervised_finetune 划分
                # 2. NTU-Fi_HAR 的 self_supervised 模式设计为"用 NTU-Fi-HumanID 做监督微调"
                #    （跨数据集迁移），但 A3 期望"用 NTU-Fi_HAR 自己的 6 类标签微调"，
                #    会导致标签越界（HumanID 14 类 vs HAR 6 类）触发 CUDA nll_loss 断言
                # 3. P3 验证目标是"同数据集预训练 + 微调"，不是跨数据集迁移（那是 B 系列的事）
                # 因此统一走 supervised 模式加载，pretrain_ds = train_ds（MAE 只用 x，不用标签）
                bundle = _load_csi_dataset(config.target_dataset, data_root,
                                           learning_mode="supervised")
                train_ds = bundle.train
                val_ds = bundle.val if bundle.val else bundle.test
                pretrain_ds = train_ds  # 复用 train 集做自监督 mask 重建
            else:
                bundle = _load_csi_dataset(config.target_dataset, data_root,
                                           learning_mode="supervised")
                train_ds = bundle.train
                val_ds = bundle.val if bundle.val else bundle.test
                pretrain_ds = None
        elif is_eeg:
            data_root = dataset_config["data_root"]
            train_ds, val_ds, _ = _load_eeg_dataset(config.target_dataset, data_root)
            pretrain_ds = None  # EEG 预训练待 RadioML 就绪后扩展
        elif is_radio:
            data_root = dataset_config["data_root"]
            if not Path(data_root).exists():
                raise FileNotFoundError(
                    f"RadioML 数据目录不存在: {data_root}。"
                    f"RadioML 2018.01A 下载完成后启用 RF 实验。"
                )
            from senseframe.scenes.radio.datasets import load_radioml_dataset
            bundle = load_radioml_dataset(config.target_dataset, root=data_root)
            train_ds = bundle["train"]
            val_ds = bundle["val"]
            pretrain_ds = None
        else:
            raise ValueError(f"Unsupported dataset: {config.target_dataset}")

        # ---- 3. 构造 DataLoader ----
        target_shape = (dataset_config["reshape_to"] if "reshape_to" in dataset_config
                        else dataset_config["input_shape"])
        collate_fn = _make_collate_fn(target_shape, dataset_name=config.target_dataset)

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # ---- 4. 构建 backbone ----
        d_model = 128
        backbone = _build_backbone(dataset_config, d_model=d_model)

        # ---- 5. MAE 预训练（若 pretrain_source != "none"）----
        if config.pretrain_source != "none" and pretrain_ds is not None:
            logger.info("experiment %s: MAE pretraining %d epochs",
                        config.experiment_id, config.pretrain_epochs)
            from senseframe.core.foundation_model import PretrainConfig
            pretrain_cfg = PretrainConfig(
                epochs=config.pretrain_epochs,
                batch_size=config.batch_size,
                learning_rate=config.pretrain_lr,
                mask_ratio=config.pretrain_mask_ratio,
            )
            pretrain_loader = DataLoader(
                pretrain_ds, batch_size=config.batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=0,
            )
            backbone.pretrain(pretrain_loader, pretrain_cfg)

        # ---- 6. 构建 PEFT 模型 ----
        peft_model = _build_peft_model(backbone, config.finetune_method, config.finetune_params)

        # ---- 7. 加分类头 ----
        num_classes = dataset_config["num_classes"]
        classifier = CSIClassifier(peft_model, d_model=d_model, num_classes=num_classes)

        total_params = sum(p.numel() for p in classifier.parameters())

        # ---- 8. 训练 ----
        logger.info("experiment %s: training %d epochs, method=%s",
                    config.experiment_id, config.epochs, config.finetune_method)
        t0 = time.time()
        val_acc, macro_f1, trainable_params = _train_classifier(
            model=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device,
            d_model=d_model,
        )
        training_time = time.time() - t0

        # ---- 9. 收集结果 ----
        result.status = "success"
        result.val_accuracy = val_acc
        result.macro_f1 = macro_f1
        result.trainable_params = trainable_params
        result.total_params = total_params
        result.total_epochs = config.epochs
        result.training_time_seconds = training_time
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(
            "experiment %s DONE: val_acc=%.4f, macro_f1=%.4f, "
            "trainable_params=%d, time=%.1fs",
            config.experiment_id, val_acc, macro_f1,
            trainable_params, training_time,
        )

    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {str(e)[:500]}"
        result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        logger.exception("experiment %s FAILED: %s", config.experiment_id, e)

    return result


# ============================================================
# 结果汇总与分析
# ============================================================
def aggregate_results(
    results: List[ExperimentResult],
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """汇总多组实验结果，可选写入 JSON。

    Args:
        results: 实验结果列表
        output_path: 输出 JSON 路径（None 仅返回 list）

    Returns:
        list of dict（每个实验一个 dict）
    """
    aggregated = [r.to_dict() for r in results]
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(aggregated, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("aggregated %d results to %s", len(aggregated), output_path)
    return aggregated


def compute_transfer_gain(
    baseline: ExperimentResult,
    transfer: ExperimentResult,
) -> float:
    """计算跨场景迁移增益（10.4 用）。

    transfer_gain = (transfer.val_accuracy - baseline.val_accuracy)
                  / baseline.val_accuracy × 100%

    Args:
        baseline: 从头训练 baseline（B1 / B4）
        transfer: 预训练+微调（B2 / B5 / ...）

    Returns:
        增益百分比（正=正向迁移，负=负向迁移）
    """
    if baseline.val_accuracy <= 0:
        return 0.0
    return (transfer.val_accuracy - baseline.val_accuracy) / baseline.val_accuracy * 100.0


def compute_search_effectiveness(
    fixed_baselines: List[ExperimentResult],
    sp_search_result: ExperimentResult,
) -> float:
    """计算 SP 搜索相对固定配置的提升（10.5 用）。

    improvement = sp_search.best_val_accuracy - max(fixed.val_accuracy for fixed in fixed_baselines)

    Args:
        fixed_baselines: 固定配置实验组（C1 / C2 / C3）
        sp_search_result: SP 搜索实验组（C4 / C5）

    Returns:
        提升百分比（正=搜索有效）
    """
    if not fixed_baselines:
        return 0.0
    best_fixed = max(r.val_accuracy for r in fixed_baselines)
    return sp_search_result.val_accuracy - best_fixed


# ============================================================
# 工具：实验组定义
# ============================================================
# 10.2 单场景性能验证（A1-A5）
SINGLE_SCENE_EXPERIMENTS: List[Dict[str, Any]] = [
    {"id": "A1", "pretrain": "none",        "finetune": "scratch",       "datasets": ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"]},
    {"id": "A2", "pretrain": "csi_4datasets", "finetune": "full",        "datasets": ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"]},
    {"id": "A3", "pretrain": "csi_4datasets", "finetune": "lora",        "datasets": ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"], "params": {"peft_rank": 8, "peft_alpha": 16}},
    {"id": "A4", "pretrain": "csi_4datasets", "finetune": "adapter",     "datasets": ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"], "params": {"adapter_bottleneck": 128}},
    {"id": "A5", "pretrain": "csi_4datasets", "finetune": "prompt_tuning", "datasets": ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"], "params": {"prompt_length": 10}},
]

# 10.4 跨场景迁移评估（B1-B8）
CROSS_DOMAIN_EXPERIMENTS: List[Dict[str, Any]] = [
    {"id": "B1", "pretrain": "none",          "target": "RadioML2018",     "finetune": "scratch"},
    {"id": "B2", "pretrain": "csi_4datasets", "target": "RadioML2018",     "finetune": "lora"},
    {"id": "B3", "pretrain": "csi_4datasets", "target": "RadioML2018",     "finetune": "full"},
    {"id": "B4", "pretrain": "none",          "target": "PhysioNet_MI",    "finetune": "scratch"},
    {"id": "B5", "pretrain": "csi_4datasets", "target": "PhysioNet_MI",    "finetune": "lora"},
    {"id": "B6", "pretrain": "csi_4datasets", "target": "PhysioNet_MI",    "finetune": "full"},
    {"id": "B7", "pretrain": "radioml",       "target": "PhysioNet_MI",    "finetune": "lora"},
    {"id": "B8", "pretrain": "eegmmidb",      "target": "RadioML2018",     "finetune": "lora"},
]

# 10.5 SP 搜索有效性（C1-C5）
SP_SEARCH_EXPERIMENTS: List[Dict[str, Any]] = [
    {"id": "C1", "method": "fixed",       "config": {"peft_method": "lora",          "peft_rank": 8, "peft_alpha": 16}},
    {"id": "C2", "method": "fixed",       "config": {"peft_method": "adapter",       "adapter_bottleneck": 128}},
    {"id": "C3", "method": "fixed",       "config": {"peft_method": "prompt_tuning", "prompt_length": 10}},
    {"id": "C4", "method": "sp_search",   "config": {"sampler": "random", "n_trials": 20}},
    {"id": "C5", "method": "sp_search",   "config": {"sampler": "grid",   "n_trials": None}},  # None=全网格
]

# 10.5 评估数据集
SP_SEARCH_DATASETS: List[str] = ["NTU-Fi_HAR", "RadioML2018", "PhysioNet_MI"]


# ============================================================
# 命令行共享参数
# ============================================================
def add_common_args(parser):
    """添加所有实验脚本的共享参数。"""
    parser.add_argument("--dry-run", action="store_true",
                        help="仅验证配置，不执行真实训练")
    parser.add_argument("--output-dir", default="results/p3_validation",
                        help="结果输出目录（默认 results/p3_validation）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    # 训练超参覆盖（默认 None 表示用 ExperimentConfig 默认值 50/64）
    parser.add_argument("--epochs", type=int, default=None,
                        help="微调训练 epoch 数（默认用 ExperimentConfig.epochs=50，"
                             "快速验证可传 10）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="batch size（默认用 ExperimentConfig.batch_size=64）")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="MAE 预训练 epoch 数（默认用 ExperimentConfig.pretrain_epochs=20，"
                             "快速验证可传 5）")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="微调学习率（默认用 ExperimentConfig.learning_rate=1e-3）")
    return parser


def apply_arg_overrides(args, config: "ExperimentConfig") -> "ExperimentConfig":
    """把 CLI 传入的 --epochs/--batch-size 等覆盖到 ExperimentConfig。

    args 中对应字段为 None 时不覆盖（保留 ExperimentConfig 默认值）。
    返回新的 ExperimentConfig 实例（不修改原对象）。
    """
    overrides = {}
    if getattr(args, "epochs", None) is not None:
        overrides["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        overrides["batch_size"] = args.batch_size
    if getattr(args, "pretrain_epochs", None) is not None:
        overrides["pretrain_epochs"] = args.pretrain_epochs
    if getattr(args, "learning_rate", None) is not None:
        overrides["learning_rate"] = args.learning_rate
    if not overrides:
        return config
    from dataclasses import replace
    return replace(config, **overrides)


def setup_logging(level: str = "INFO") -> None:
    """配置 logging。"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    )


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "run_single_experiment",
    "aggregate_results",
    "compute_transfer_gain",
    "compute_search_effectiveness",
    "SINGLE_SCENE_EXPERIMENTS",
    "CROSS_DOMAIN_EXPERIMENTS",
    "SP_SEARCH_EXPERIMENTS",
    "SP_SEARCH_DATASETS",
    "add_common_args",
    "setup_logging",
]
