"""
NTU-Fi_HAR val_loss 爆炸问题 — 系统性消融测试

在简化版 Lightning Trainer（已知能正常学习）基础上，逐个添加 pipeline.py 的配置项，
找出导致 train_loss 停滞 + val_loss 指数爆炸的具体配置。

测试矩阵：
  A: 基线（已知能工作）— 无 scheduler, wd=0, nw=0, 无 callbacks, max_epochs=5
  B: + cosine scheduler (T_max=214) + weight_decay=0.0001
  C: + num_workers=4 + persistent_workers=true + pin_memory=true
  D: + ModelCheckpoint + EarlyStopping(patience=5) callbacks
  E: + EpochLogCallback + IntermediateMetricLogger
  F: + max_epochs=214（cosine T_max=214）

每个测试运行 3 epoch（F 除外，运行 5 epoch），打印：
  - train_loss, val_loss, train_acc, val_acc, lr
  - BN running_mean/running_var 统计（第一个 BN 层）
  - 梯度范数
  - train/eval 模式输出差异

用法：
  cd <DEPLOY_ROOT>
  python tests/debug_ablation_full.py [A|B|C|D|E|F]
  python tests/debug_ablation_full.py  # 运行所有测试
"""

import sys
import os
import copy
from pathlib import Path

# 添加项目根到 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 激活 SenseFrame 场景（必须在使用前调用）
os.environ.setdefault("SENSEFRAME_SENSEFI_PATH",
                      "<DEPLOY_ROOT>/resource/WiFi-CSI-Sensing-Benchmark-main")

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from senseframe.scenes import activate_lazy_scenes, get_scene
activate_lazy_scenes()

from senseframe.engine.datamodule import GenericDataModule
from senseframe.engine.module import GenericLightningModule


DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS"
DATASET = "NTU-Fi_HAR"
MODEL_ID = "ResNet18"
NUM_CLASSES = 6
BATCH_SIZE = 64


def load_data(num_workers=0, pin_memory=False, persistent_workers=False):
    """加载 NTU-Fi_HAR 数据集，返回 GenericDataModule。"""
    scene = get_scene("wifi_csi")
    bundle = scene.load_dataset(DATASET, DATA_ROOT, learning_mode="supervised")
    transform_cfg = scene.get_transforms(DATASET)

    dm = GenericDataModule(
        train_dataset=bundle.train,
        test_dataset=bundle.test,
        val_dataset=bundle.val,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        train_transform=transform_cfg.train_transform,
        eval_transform=transform_cfg.eval_transform,
    )
    return dm


def build_model():
    """构建 NTU_Fi_ResNet18 模型。"""
    scene = get_scene("wifi_csi")
    model = scene.build_model_for_dataset(MODEL_ID, DATASET, NUM_CLASSES)
    return model


def build_module(model, scheduler=None, weight_decay=0.0, max_epochs=5):
    """构建 GenericLightningModule。"""
    module = GenericLightningModule(
        model=model,
        num_classes=NUM_CLASSES,
        learning_rate=1e-3,
        metrics=["accuracy", "macro_f1"],
        optimizer="adam",
        weight_decay=weight_decay,
        scheduler=scheduler,
        max_epochs=max_epochs,
    )
    return module


def collect_bn_stats(model):
    """收集模型中所有 BatchNorm 层的 running statistics。"""
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                rm = m.running_mean
                rv = m.running_var
                stats.append({
                    "name": name,
                    "rm_mean": float(rm.mean().item()),
                    "rm_std": float(rm.std().item()),
                    "rv_mean": float(rv.mean().item()),
                    "rv_min": float(rv.min().item()),
                    "rv_max": float(rv.max().item()),
                    "num_batches": m.num_batches_tracked.item(),
                })
    return stats


def measure_train_eval_diff(module, dm):
    """测量 train/eval 模式下模型输出的差异。"""
    module.eval()
    val_dl = dm.val_dataloader()
    batch = next(iter(val_dl))
    x, y = batch
    x = x.to(module.device)
    y = y.to(module.device)

    with torch.no_grad():
        # eval 模式
        module.eval()
        out_eval = module(x)
        loss_eval = torch.nn.functional.cross_entropy(out_eval, y)

        # train 模式（但不更新 BN running stats）
        module.train()
        with torch.no_grad():
            out_train = module(x)
            loss_train = torch.nn.functional.cross_entropy(out_train, y)

    # 恢复 eval 模式
    module.eval()

    diff = (out_eval - out_train).abs()
    return {
        "loss_eval": float(loss_eval.item()),
        "loss_train": float(loss_train.item()),
        "out_eval_abs_mean": float(out_eval.abs().mean().item()),
        "out_train_abs_mean": float(out_train.abs().mean().item()),
        "out_diff_abs_mean": float(diff.mean().item()),
        "out_diff_abs_max": float(diff.max().item()),
    }


class BNStatsCallback(pl.Callback):
    """每个 epoch 后打印 BN statistics 和 train/eval 输出差异。"""

    def __init__(self, dm, label=""):
        self.dm = dm
        self.label = label

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        # BN stats
        bn_stats = collect_bn_stats(pl_module.model)
        first_bn = bn_stats[0] if bn_stats else {}
        # train/eval diff
        try:
            diff = measure_train_eval_diff(pl_module, self.dm)
        except Exception as e:
            diff = {"error": str(e)}

        # 梯度范数
        grad_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        # callback_metrics
        cbm = trainer.callback_metrics
        train_loss = cbm.get("train_loss", "N/A")
        val_loss = cbm.get("val_loss", "N/A")
        train_acc = cbm.get("train_accuracy", "N/A")
        val_acc = cbm.get("val_accuracy", "N/A")
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else "N/A"

        def fmt(v):
            if isinstance(v, (int, float)):
                return f"{v:.6f}"
            return str(v)

        print(f"\n[{self.label}] Epoch {epoch}:", flush=True)
        print(f"  train_loss={fmt(train_loss)}, val_loss={fmt(val_loss)}, "
              f"train_acc={fmt(train_acc)}, val_acc={fmt(val_acc)}, lr={fmt(lr)}", flush=True)
        print(f"  grad_norm={grad_norm:.4f}", flush=True)
        if first_bn:
            print(f"  BN[0]({first_bn['name']}): "
                  f"rm_mean={first_bn['rm_mean']:.4f}, rm_std={first_bn['rm_std']:.4f}, "
                  f"rv_mean={first_bn['rv_mean']:.4f}, rv_min={first_bn['rv_min']:.6f}, "
                  f"rv_max={first_bn['rv_max']:.4f}, "
                  f"num_batches={first_bn['num_batches']}", flush=True)
        if "error" not in diff:
            print(f"  train/eval: loss_eval={diff['loss_eval']:.4f}, "
                  f"loss_train={diff['loss_train']:.4f}, "
                  f"out_eval_abs={diff['out_eval_abs_mean']:.4f}, "
                  f"out_train_abs={diff['out_train_abs_mean']:.4f}, "
                  f"out_diff={diff['out_diff_abs_mean']:.4f}, "
                  f"out_diff_max={diff['out_diff_abs_max']:.4f}", flush=True)
        else:
            print(f"  train/eval diff error: {diff['error']}", flush=True)
        # 打印所有 BN 层的 rv_min（检测是否有 BN running_var 趋零）
        rv_mins = [(s["name"], s["rv_min"]) for s in bn_stats]
        rv_mins_str = ", ".join(f"{n}:{v:.6f}" for n, v in rv_mins[:5])
        print(f"  rv_mins (first 5): {rv_mins_str}", flush=True)
        print(flush=True)


def run_test(label, scheduler=None, weight_decay=0.0, max_epochs=3,
             num_workers=0, pin_memory=False, persistent_workers=False,
             use_callbacks=False, use_extra_callbacks=False):
    """运行单个消融测试。"""
    print(f"\n{'='*60}", flush=True)
    print(f"Test {label}: scheduler={scheduler}, wd={weight_decay}, "
          f"nw={num_workers}, pin_mem={pin_memory}, persist={persistent_workers}, "
          f"callbacks={use_callbacks}, extra_cb={use_extra_callbacks}, "
          f"max_epochs={max_epochs}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    dm = load_data(num_workers=num_workers, pin_memory=pin_memory,
                   persistent_workers=persistent_workers)
    model = build_model()
    module = build_module(model, scheduler=scheduler, weight_decay=weight_decay,
                          max_epochs=max_epochs)

    callbacks = [BNStatsCallback(dm, label)]
    if use_callbacks:
        callbacks.append(ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1,
            save_on_train_epoch_end=False,
            dirpath=f"/tmp/ablation_{label}",
        ))
        callbacks.append(EarlyStopping(
            monitor="val_loss", mode="min", patience=5,
            check_on_train_epoch_end=False,
        ))
    if use_extra_callbacks:
        from senseframe.engine.runner.orchestrator import (
            EpochLogCallback, IntermediateMetricLogger,
        )
        callbacks.append(EpochLogCallback(log_every_n=1))
        callbacks.append(IntermediateMetricLogger(
            metric="val_accuracy",
            intermediate_values={},
        ))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=use_callbacks,
        deterministic=False,
    )

    trainer.fit(module, datamodule=dm)

    # 最终摘要
    cbm = trainer.callback_metrics
    print(f"\n[{label} FINAL] train_loss={cbm.get('train_loss', 'N/A')}, "
          f"val_loss={cbm.get('val_loss', 'N/A')}, "
          f"train_acc={cbm.get('train_accuracy', 'N/A')}, "
          f"val_acc={cbm.get('val_accuracy', 'N/A')}", flush=True)

    # 清理
    del trainer, module, model, dm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    if len(sys.argv) > 1:
        tests = sys.argv[1:]
    else:
        tests = ["A", "B", "C", "D", "E", "F"]

    for t in tests:
        if t == "A":
            # 基线（已知能工作）
            run_test("A", scheduler=None, weight_decay=0.0, max_epochs=3,
                     num_workers=0, pin_memory=False, persistent_workers=False,
                     use_callbacks=False, use_extra_callbacks=False)
        elif t == "B":
            # + cosine scheduler + weight_decay
            run_test("B", scheduler="cosine", weight_decay=0.0001, max_epochs=3,
                     num_workers=0, pin_memory=False, persistent_workers=False,
                     use_callbacks=False, use_extra_callbacks=False)
        elif t == "C":
            # + num_workers=4 + persistent_workers + pin_memory
            run_test("C", scheduler="cosine", weight_decay=0.0001, max_epochs=3,
                     num_workers=4, pin_memory=True, persistent_workers=True,
                     use_callbacks=False, use_extra_callbacks=False)
        elif t == "D":
            # + ModelCheckpoint + EarlyStopping
            run_test("D", scheduler="cosine", weight_decay=0.0001, max_epochs=3,
                     num_workers=4, pin_memory=True, persistent_workers=True,
                     use_callbacks=True, use_extra_callbacks=False)
        elif t == "E":
            # + EpochLogCallback + IntermediateMetricLogger
            run_test("E", scheduler="cosine", weight_decay=0.0001, max_epochs=3,
                     num_workers=4, pin_memory=True, persistent_workers=True,
                     use_callbacks=True, use_extra_callbacks=True)
        elif t == "F":
            # + max_epochs=214（cosine T_max=214）
            run_test("F", scheduler="cosine", weight_decay=0.0001, max_epochs=5,
                     num_workers=4, pin_memory=True, persistent_workers=True,
                     use_callbacks=True, use_extra_callbacks=True)
        else:
            print(f"Unknown test: {t}")


if __name__ == "__main__":
    main()
