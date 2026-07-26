"""
NTU-Fi_HAR val_loss 爆炸问题 — pipeline 完整复刻消融测试

在消融测试 E（已知正常学习）基础上，逐步添加完整 pipeline 的差异项，
通过二分法定位导致 train_loss 停滞 + val_loss 爆炸的根因。

测试矩阵：
  G: 消融测试 E + pipeline Trainer 配置（CSVLogger, max_time, gradient_clip_val, accumulate_grad_batches）
  H: 测试 G + pipeline 执行路径（ResourceProbe, DataProfiler）
  I: 测试 H 但跳过 DataProfiler（隔离 DataProfiler 副作用）
  J: 测试 H 但跳过 ResourceProbe（隔离 ResourceProbe 副作用）
  K: 直接调用 pipeline.run_pipeline（完整 pipeline 复现，作为对照）

每个测试运行 3 epoch，打印：
  - 训练前模型权重统计（均值/标准差/是否有 NaN/Inf）
  - train_loss, val_loss, train_acc, val_acc, lr
  - BN running_mean/running_var 统计
  - 梯度范数

用法：
  cd <DEPLOY_ROOT>
  .venv/bin/python tests/debug_pipeline_replica.py G
  .venv/bin/python tests/debug_pipeline_replica.py G H I J
"""

import sys
import os
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
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

from senseframe.scenes import activate_lazy_scenes, get_scene
activate_lazy_scenes()

from senseframe.engine.datamodule import GenericDataModule
from senseframe.engine.module import GenericLightningModule
from senseframe.engine.runner.preflight import set_seed
from senseframe.routing import ResourceProbe, ResourceRouter


DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS"
DATASET = "NTU-Fi_HAR"
MODEL_ID = "ResNet18"
NUM_CLASSES = 6
BATCH_SIZE = 64


def load_data(num_workers=4, pin_memory=True, persistent_workers=True):
    """加载 NTU-Fi_HAR 数据集，返回 (bundle, GenericDataModule)。"""
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
    return bundle, dm


def build_model():
    """构建 NTU_Fi_ResNet18 模型。"""
    scene = get_scene("wifi_csi")
    model = scene.build_model_for_dataset(MODEL_ID, DATASET, NUM_CLASSES)
    return model


def build_module(model, max_epochs=3):
    """构建 GenericLightningModule（与 pipeline stage_build 一致）。"""
    module = GenericLightningModule(
        model=model,
        num_classes=NUM_CLASSES,
        learning_rate=1e-3,
        metrics=["accuracy", "macro_f1"],
        optimizer="adam",
        weight_decay=0.0001,
        scheduler="cosine",
        max_epochs=max_epochs,
    )
    return module


def print_model_stats(model, label=""):
    """打印模型权重统计。"""
    total_params = 0
    nan_count = 0
    inf_count = 0
    all_means = []
    all_stds = []
    for name, p in model.named_parameters():
        total_params += p.numel()
        if torch.isnan(p).any():
            nan_count += 1
        if torch.isinf(p).any():
            inf_count += 1
        all_means.append(float(p.mean().item()))
        all_stds.append(float(p.std().item()))

    # BN running stats
    bn_stats = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                bn_stats.append({
                    "name": name,
                    "rm_mean": float(m.running_mean.mean().item()),
                    "rv_mean": float(m.running_var.mean().item()),
                    "rv_min": float(m.running_var.min().item()),
                    "num_batches": m.num_batches_tracked.item(),
                })

    print(f"\n[{label} Model Stats]", flush=True)
    print(f"  total_params={total_params}, nan_params={nan_count}, inf_params={inf_count}", flush=True)
    print(f"  weight_mean(avg)={sum(all_means)/len(all_means):.6f}, "
          f"weight_std(avg)={sum(all_stds)/len(all_stds):.6f}", flush=True)
    if bn_stats:
        first_bn = bn_stats[0]
        print(f"  BN[0]({first_bn['name']}): rm_mean={first_bn['rm_mean']:.4f}, "
              f"rv_mean={first_bn['rv_mean']:.4f}, rv_min={first_bn['rv_min']:.6f}, "
              f"num_batches={first_bn['num_batches']}", flush=True)
    print(flush=True)


def collect_bn_stats(model):
    """收集模型中所有 BatchNorm 层的 running statistics。"""
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                stats.append({
                    "name": name,
                    "rm_mean": float(m.running_mean.mean().item()),
                    "rm_std": float(m.running_mean.std().item()),
                    "rv_mean": float(m.running_var.mean().item()),
                    "rv_min": float(m.running_var.min().item()),
                    "rv_max": float(m.running_var.max().item()),
                    "num_batches": m.num_batches_tracked.item(),
                })
    return stats


class BNStatsCallback(pl.Callback):
    """每个 epoch 后打印 BN statistics 和梯度范数。"""

    def __init__(self, label=""):
        self.label = label

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        bn_stats = collect_bn_stats(pl_module.model)
        first_bn = bn_stats[0] if bn_stats else {}

        # 梯度范数
        grad_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

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
        rv_mins = [(s["name"], s["rv_min"]) for s in bn_stats]
        rv_mins_str = ", ".join(f"{n}:{v:.6f}" for n, v in rv_mins[:5])
        print(f"  rv_mins (first 5): {rv_mins_str}", flush=True)
        print(flush=True)


def run_subprocess_probe(label=""):
    """模拟 stage_probe_vram：在子进程中运行 probe_worker。"""
    import subprocess
    import sys as _sys

    cmd = [
        _sys.executable, "-m", "senseframe.engine.runner.probe_worker",
        "--model-id", MODEL_ID,
        "--dataset", DATASET,
        "--num-classes", str(NUM_CLASSES),
        "--learning-mode", "supervised",
        "--batch-size", str(BATCH_SIZE),
        "--precision", "32",
        "--optimizer", "adam",
        "--data-root", DATA_ROOT,
        "--scene-name", "wifi_csi",
    ]
    print(f"[{label}] Starting subprocess probe...", flush=True)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
        cwd=str(Path("<DEPLOY_ROOT>")),
    )
    import json
    if proc.returncode == 0:
        result = json.loads(proc.stdout.strip())
        print(f"[{label}] Probe result: ok={result.get('ok')}, "
              f"measured={result.get('measured_vram_mb')}MB, "
              f"batch_size={result.get('batch_size')}", flush=True)
    else:
        print(f"[{label}] Probe FAILED: returncode={proc.returncode}, "
              f"stderr={proc.stderr[:300]}", flush=True)
    print(f"[{label}] Subprocess probe done.", flush=True)


def run_test(label, use_pipeline_trainer_config=False, use_pipeline_path=False,
             use_dataprofiler=True, use_resourceprobe=True, max_epochs=3,
             use_probe_vram=False):
    """运行单个消融测试。

    Args:
        label: 测试标签
        use_pipeline_trainer_config: 是否使用 pipeline 的 Trainer 配置
            (CSVLogger, max_time, gradient_clip_val, accumulate_grad_batches)
        use_pipeline_path: 是否使用 pipeline 的执行路径
            (set_seed → ResourceProbe → DataProfiler → build_model)
        use_dataprofiler: 是否运行 DataProfiler（仅 use_pipeline_path=True 时生效）
        use_resourceprobe: 是否运行 ResourceProbe（仅 use_pipeline_path=True 时生效）
        use_probe_vram: 是否在 trainer.fit 前运行 subprocess probe（模拟 stage_probe_vram）
    """
    print(f"\n{'='*70}", flush=True)
    print(f"Test {label}:", flush=True)
    print(f"  pipeline_trainer_config={use_pipeline_trainer_config}", flush=True)
    print(f"  pipeline_path={use_pipeline_path}", flush=True)
    if use_pipeline_path:
        print(f"  use_dataprofiler={use_dataprofiler}", flush=True)
        print(f"  use_resourceprobe={use_resourceprobe}", flush=True)
    print(f"  max_epochs={max_epochs}", flush=True)
    print(f"{'='*70}", flush=True)

    # === RNG 设置 ===
    if use_pipeline_path:
        # pipeline 路径：set_seed → ResourceProbe → DataProfiler → build_model
        set_seed(42, deterministic=False)
        print("[path] After set_seed(42)", flush=True)

        if use_resourceprobe:
            report = ResourceProbe.probe()
            route_level = ResourceRouter.route(report)
            route_config = ResourceRouter.get_route_config(route_level)
            print(f"[path] After ResourceProbe: route_level={route_level}, "
                  f"gpu={report.gpu_name}, vram={report.gpu_total_vram_mb}MB", flush=True)

        if use_dataprofiler:
            # DataProfiler 会采样数据
            from senseframe.core.profiler import DataProfiler
            scene = get_scene("wifi_csi")
            bundle_tmp = scene.load_dataset(DATASET, DATA_ROOT, learning_mode="supervised")
            profiler = DataProfiler(max_samples=500)
            modality_hint = getattr(getattr(scene, "meta", None), "modality", None)
            data_profile = profiler.profile_bundle(
                bundle_tmp, dataset_name=DATASET, modality_hint=modality_hint,
                learning_mode="supervised",
            )
            print(f"[path] After DataProfiler: n_samples={data_profile.n_samples if data_profile else 'None'}, "
                  f"n_classes={data_profile.n_classes if data_profile else 'None'}", flush=True)
            del bundle_tmp

        # 加载数据（与 pipeline stage_load 一致）
        bundle, dm = load_data()
        # 构建模型
        model = build_model()
        print_model_stats(model, f"{label} pre-fit")
    else:
        # 消融测试路径：manual_seed → load_data → build_model
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        bundle, dm = load_data()
        model = build_model()
        print_model_stats(model, f"{label} pre-fit")

    module = build_module(model, max_epochs=max_epochs)

    # === Callbacks ===
    callbacks = [BNStatsCallback(label)]
    callbacks.append(ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        save_on_train_epoch_end=False,
        dirpath=f"/tmp/ablation_{label}",
    ))
    callbacks.append(EarlyStopping(
        monitor="val_loss", mode="min", patience=5,
        check_on_train_epoch_end=False,
    ))
    from senseframe.engine.runner.orchestrator import (
        EpochLogCallback, IntermediateMetricLogger,
    )
    callbacks.append(EpochLogCallback(log_every_n=1))
    callbacks.append(IntermediateMetricLogger(
        metric="val_accuracy",
        intermediate_values={},
    ))

    # === Trainer 构造 ===
    if use_pipeline_trainer_config:
        # 完全复制 pipeline 的 _build_trainer_kwargs + stage_train 配置
        output_dir = Path(f"/tmp/ablation_{label}_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_logger = CSVLogger(str(output_dir), name=f"ablation_{label}")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=csv_logger,  # CSVLogger（pipeline 用 CSVLogger）
            enable_checkpointing=True,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="32",
            enable_progress_bar=True,  # pipeline 默认 True
            enable_model_summary=False,
            deterministic=False,
            max_time="00:02:00:00",  # pipeline 有 max_time
            gradient_clip_val=None,  # pipeline 显式传 None
            gradient_clip_algorithm="norm",  # pipeline 显式传
            accumulate_grad_batches=1,  # pipeline 显式传
            # num_sanity_val_steps 用默认值 2（pipeline 不显式设置）
        )
    else:
        # 消融测试 E 的 Trainer 配置
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
            enable_checkpointing=True,
            deterministic=False,
        )

    # === Subprocess probe（模拟 stage_probe_vram）===
    if use_probe_vram:
        run_subprocess_probe(label)
        # probe 后再次打印模型权重统计（检查是否被修改）
        print_model_stats(model, f"{label} post-probe pre-fit")

    # DEBUG: 添加临时 callback 打印第一个训练 batch 的数据统计
    class _DebugBatchCallback2(pl.Callback):
        def __init__(self):
            self._printed = False
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if not self._printed:
                self._printed = True
                x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
                print(f"[{label} DEBUG batch[0]]: x.shape={tuple(x.shape)}, x.dtype={x.dtype}, "
                      f"x.mean={float(x.mean()):.6f}, x.std={float(x.std()):.6f}, "
                      f"x.min={float(x.min()):.6f}, x.max={float(x.max()):.6f}, "
                      f"y.shape={tuple(y.shape) if y is not None else 'N/A'}, "
                      f"y.unique={y.unique().tolist() if y is not None else 'N/A'}", flush=True)
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            if not self._printed:
                self._printed = True
                x, y = batch if isinstance(batch, (tuple, list)) else (batch, None)
                print(f"[{label} DEBUG val_batch[0]]: x.shape={tuple(x.shape)}, "
                      f"x.mean={float(x.mean()):.6f}, x.std={float(x.std()):.6f}, "
                      f"y.unique={y.unique().tolist() if y is not None else 'N/A'}", flush=True)
    trainer.callbacks.append(_DebugBatchCallback2())

    # DEBUG: 打印 module 和 dm 关键参数
    print(f"[{label} DEBUG module] lr={module.learning_rate}, opt={module.optimizer_type}, "
          f"sched={module.scheduler_type}, wd={module.weight_decay}, max_epochs={module.max_epochs}, "
          f"loss={module.task_spec.effective_loss}, criterion={type(module.criterion).__name__}, "
          f"has_log_writer={module._log_writer is not None}, "
          f"has_data_profile={getattr(module, 'data_profile', None) is not None}", flush=True)
    print(f"[{label} DEBUG dm] batch_size={getattr(dm, 'batch_size', 'N/A')}, "
          f"num_workers={getattr(dm, 'num_workers', 'N/A')}, "
          f"pin_memory={getattr(dm, 'pin_memory', 'N/A')}, "
          f"persistent_workers={getattr(dm, 'persistent_workers', 'N/A')}, "
          f"train_size={len(dm.train_dataset) if hasattr(dm, 'train_dataset') and dm.train_dataset else 'N/A'}, "
          f"val_size={len(dm.val_dataset) if hasattr(dm, 'val_dataset') and dm.val_dataset else 'N/A'}, "
          f"test_size={len(dm.test_dataset) if hasattr(dm, 'test_dataset') and dm.test_dataset else 'N/A'}", flush=True)

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
        tests = ["G", "H", "I", "J"]

    for t in tests:
        if t == "G":
            # 消融测试 E + pipeline Trainer 配置
            run_test("G", use_pipeline_trainer_config=True, use_pipeline_path=False,
                     max_epochs=3)
        elif t == "H":
            # 测试 G + pipeline 执行路径（ResourceProbe + DataProfiler）
            run_test("H", use_pipeline_trainer_config=True, use_pipeline_path=True,
                     use_dataprofiler=True, use_resourceprobe=True, max_epochs=3)
        elif t == "I":
            # 测试 H 但跳过 DataProfiler
            run_test("I", use_pipeline_trainer_config=True, use_pipeline_path=True,
                     use_dataprofiler=False, use_resourceprobe=True, max_epochs=3)
        elif t == "J":
            # 测试 H 但跳过 ResourceProbe
            run_test("J", use_pipeline_trainer_config=True, use_pipeline_path=True,
                     use_dataprofiler=True, use_resourceprobe=False, max_epochs=3)
        elif t == "K":
            # 测试 H + subprocess probe（模拟 stage_probe_vram）
            run_test("K", use_pipeline_trainer_config=True, use_pipeline_path=True,
                     use_dataprofiler=True, use_resourceprobe=True, max_epochs=3,
                     use_probe_vram=True)
        else:
            print(f"Unknown test: {t}")


if __name__ == "__main__":
    main()
