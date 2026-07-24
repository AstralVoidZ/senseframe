"""P0-1: MAE 预训练 100 epoch + PSNR early stopping + checkpoint 保存。

质量优先策略：
- early stopping 用 PSNR（重建质量）而非 MAE loss（标量损失）
- PSNR 是业界图像/信号重建的标准质量评估指标
- 连续 patience 次无提升（< min_delta dB）则停止

设计要点：
- compute_psnr：CSI 归一化后范围约 [-5, 5]，用 MAX=5.0（5σ 边界）
- PSNREarlyStopping：状态机，跟踪 best_psnr + counter + should_stop
- main()：完整训练入口（Task 2 追加）
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# 项目根加入 sys.path（从 scripts/p0_pretrain_with_psnr.py 向上两级），
# 让 `from senseframe...` / `from scripts...` 在直接 python scripts/xxx.py 运行时也能工作
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


def compute_psnr(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 5.0,
) -> float:
    """计算 PSNR（dB），值越高重建质量越好。

    Args:
        reconstructed: 重建的 tensor
        target: 原始 tensor
        max_value: 信号最大值（CSI 归一化后约 5σ 边界，默认 5.0）

    Returns:
        PSNR 值（dB），完美重建返回 100.0
    """
    mse = F.mse_loss(reconstructed, target)
    if mse.item() < 1e-10:
        return 100.0
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr.item()


@dataclass
class PSNREarlyStopping:
    """PSNR-based early stopping。

    Args:
        patience: 连续无提升的最大 epoch 数
        min_delta: 视为提升的最小 PSNR 增量（dB）
    """
    patience: int = 10
    min_delta: float = 0.1

    best_psnr: float = -float("inf")
    counter: int = 0
    should_stop: bool = False

    def __call__(self, val_psnr: float) -> None:
        """更新 early stopping 状态。

        Args:
            val_psnr: 当前 epoch 的验证集 PSNR
        """
        if val_psnr > self.best_psnr + self.min_delta:
            self.best_psnr = val_psnr
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def evaluate_psnr(backbone, val_loader, mask_ratio: float, device: str) -> float:
    """在验证集上计算 MAE 重建 PSNR（仅 masked patches）。

    复用 backbone 内部方法：
    - patch_embedder.to_patches：获取重建 target
    - random_masking：随机 mask
    - _forward_encoder / _forward_decoder：编码 + 重建

    Args:
        backbone: CSIFoundationModel（已 train() 或 eval()）
        val_loader: 验证集 DataLoader
        mask_ratio: mask 比例（与训练一致）
        device: cuda / cpu

    Returns:
        平均 PSNR（dB）
    """
    backbone.eval()
    psnr_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            torch.manual_seed(123)  # 验证用固定 seed，保证 PSNR 可比
            # 复用 _mae_forward_loss 的前向流程，但获取重建结果而非 loss
            target = backbone.patch_embedder.to_patches(x)
            patches = backbone.patch_embedder.proj(target) + backbone.pos_embed
            x_visible, mask, ids_restore = backbone.random_masking(patches, mask_ratio)
            enc_out = backbone._forward_encoder(x_visible)
            recon = backbone._forward_decoder(enc_out, ids_restore)
            # PSNR on masked patches only（与 loss 计算口径一致）
            # mask: (B, n_patches)，recon/target: (B, n_patches, D)
            # 用 (B, n_patches) bool mask 索引前两维，返回 (n_masked_total, D)
            mask_bool = mask.bool()
            masked_recon = recon[mask_bool]
            masked_target = target[mask_bool]
            if masked_recon.numel() > 0:
                psnr_sum += compute_psnr(masked_recon, masked_target, max_value=5.0)
                n_batches += 1
    return psnr_sum / max(n_batches, 1)


def main():
    """P0-1 主入口：MAE 预训练 100 epoch + PSNR early stopping + checkpoint 保存。"""
    import argparse
    import json
    import time
    from pathlib import Path
    from torch.utils.data import DataLoader, random_split

    from senseframe.scenes.wifi_csi.foundation_model import CSIFoundationModel
    from scripts.p3_eval_common import (
        _load_csi_dataset,
        _make_collate_fn,
        _CSI_DATA_ROOT,
        CSI_DATASET_CONFIG,
    )

    parser = argparse.ArgumentParser(
        description="P0-1: MAE pretrain with PSNR early stopping"
    )
    parser.add_argument("--dataset", default="NTU-Fi_HAR", help="CSI dataset name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset_config = CSI_DATASET_CONFIG[args.dataset]
    bundle = _load_csi_dataset(
        args.dataset, _CSI_DATA_ROOT, learning_mode="supervised"
    )
    full_ds = bundle.train

    # 划分 train/val（80/10，test 10% 不用）
    n = len(full_ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_ds, val_ds, _ = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    target_shape = dataset_config["reshape_to"]
    collate_fn = _make_collate_fn(target_shape, dataset_name=args.dataset)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 构建模型
    backbone = CSIFoundationModel(
        input_shape=target_shape,
        d_model=128, n_heads=4,
        n_encoder_layers=4, n_decoder_layers=2,
        patch_len=dataset_config["patch_len"],
        decoder_dim=64,
    ).to(args.device)

    # 训练循环
    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    early_stopping = PSNREarlyStopping(
        patience=args.patience, min_delta=args.min_delta
    )
    history = []
    best_state = None
    # 独立跟踪绝对最高 PSNR（不依赖 early_stopping.best_psnr 的 min_delta 语义）
    # early_stopping.best_psnr 仅在 val_psnr > best + min_delta 时更新，
    # 导致 [best, best + min_delta) 区间内的提升不会更新 best_psnr，
    # 若用 >= best_psnr 保存权重，会把这个区间的 epoch 误当作 best 覆盖。
    # abs_best_psnr 任何严格提升都更新，保证 best_state 总是历史最高 PSNR 的权重。
    abs_best_psnr = -float("inf")

    logger.info(
        "P0-1: dataset=%s, epochs=%d, patience=%d, min_delta=%.2f, device=%s",
        args.dataset, args.epochs, args.patience, args.min_delta, args.device,
    )
    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练
        backbone.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(args.device)
            optimizer.zero_grad()
            torch.manual_seed(42 + epoch)  # 可复现的 mask
            loss = backbone._mae_forward_loss(x, mask_ratio=args.mask_ratio)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # 验证 PSNR
        val_psnr = evaluate_psnr(backbone, val_loader, args.mask_ratio, args.device)

        elapsed = time.time() - start_time
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "elapsed_seconds": elapsed,
        })
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_psnr=%.2f dB | best=%.2f dB | counter=%d/%d | elapsed=%.1fs",
            epoch + 1, args.epochs, train_loss, val_psnr,
            early_stopping.best_psnr, early_stopping.counter, early_stopping.patience,
            elapsed,
        )

        # 保存历史最高 PSNR 的权重（独立于 early_stopping 的 min_delta 语义）
        if val_psnr > abs_best_psnr:
            abs_best_psnr = val_psnr
            best_state = {
                k: v.cpu().clone() for k, v in backbone.state_dict().items()
            }

        # early stopping 决策（按 min_delta 判断是否重置 counter）
        early_stopping(val_psnr)

        if early_stopping.should_stop:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, min_delta=%.2f)",
                epoch + 1, args.patience, args.min_delta,
            )
            break

    total_elapsed = time.time() - start_time

    # 保存最佳 checkpoint（用 abs_best_psnr 而非 early_stopping.best_psnr，
    # 因为前者是真实历史最高，后者受 min_delta 语义影响可能偏低）
    safe_psnr = f"{abs_best_psnr:.1f}".replace(".", "p")
    ckpt_path = output_dir / f"ntu_pretrain_{args.dataset}_psnr{safe_psnr}.pt"
    if best_state is not None:
        torch.save(
            {
                "backbone_state_dict": best_state,
                "best_psnr": abs_best_psnr,
                "early_stop_best_psnr": early_stopping.best_psnr,
                "config": vars(args),
                "total_elapsed_seconds": total_elapsed,
                "final_epoch": epoch + 1,
            },
            ckpt_path,
        )
        logger.info(
            "Saved best checkpoint to %s (abs_best_psnr=%.2f dB, early_stop_best=%.2f dB, epochs=%d, %.1fs)",
            ckpt_path, abs_best_psnr, early_stopping.best_psnr, epoch + 1, total_elapsed,
        )

    # 保存训练历史
    history_path = output_dir / f"ntu_pretrain_{args.dataset}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
