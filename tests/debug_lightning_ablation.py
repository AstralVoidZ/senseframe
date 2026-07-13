"""诊断脚本 4：逐步添加 pipeline.py 配置，找出导致模型不学习的设置。

测试顺序：
A. 基础配置 + num_sanity_val_steps=2（启用 sanity check）
B. 基础配置 + callbacks（EarlyStopping + ModelCheckpoint）
C. 基础配置 + IntermediateMetricLogger + EpochLogCallback
"""
import sys
import os

PROJECT_ROOT = "<DEPLOY_ROOT>"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "resource", "WiFi-CSI-Sensing-Benchmark-main"))
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

from senseframe.scenes import activate_lazy_scenes
activate_lazy_scenes()
from senseframe.scenes import get_scene
scene = get_scene("wifi_csi")

DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS"

# 加载数据
bundle = scene.load_dataset("NTU-Fi_HAR", DATA_ROOT, learning_mode="supervised")
train_ds = bundle.train
test_ds = bundle.test

transform_cfg = scene.get_transforms("NTU-Fi_HAR")
from senseframe.engine.datamodule import _TransformWrapper, GenericDataModule
train_wrapped = _TransformWrapper(train_ds, transform_cfg.train_transform)
test_wrapped = _TransformWrapper(test_ds, transform_cfg.eval_transform)

from NTU_Fi_model import NTU_Fi_ResNet18
from senseframe.engine.module import GenericLightningModule
from senseframe.core.task import TaskSpec

task_spec = TaskSpec(task_type="classification", num_classes=6, loss="cross_entropy", metrics=["accuracy", "macro_f1"])


def make_module():
    model = NTU_Fi_ResNet18(num_classes=6)
    return GenericLightningModule(
        model=model, num_classes=6, task_spec=task_spec,
        optimizer="adam", learning_rate=1e-3, weight_decay=0.0001,
        scheduler="cosine", max_epochs=3,
    )


def make_datamodule():
    return GenericDataModule(
        train_dataset=_TransformWrapper(train_ds, transform_cfg.train_transform),
        test_dataset=test_wrapped,
        val_dataset=test_wrapped,
        batch_size=64, num_workers=0, pin_memory=False,
        persistent_workers=False, learning_mode="supervised",
    )


def run_test(name, trainer_kwargs, callbacks=None):
    print(f"\n{'=' * 60}")
    print(f"测试 {name}")
    print(f"{'=' * 60}")
    module = make_module()
    dm = make_datamodule()
    kwargs = dict(
        max_epochs=3,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        deterministic=False,
    )
    kwargs.update(trainer_kwargs)
    if callbacks:
        kwargs["callbacks"] = callbacks
    trainer = pl.Trainer(**kwargs)
    trainer.fit(module, datamodule=dm)
    log = module.training_log
    print(f"  结果:")
    for entry in log:
        print(f"    {entry}")
    return log


# 测试 A：启用 sanity check
run_test("A: num_sanity_val_steps=2 (默认)", {"num_sanity_val_steps": 2})

# 测试 B：启用 sanity check + EarlyStopping + ModelCheckpoint
ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.3f}")
es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=5, check_on_train_epoch_end=False)
run_test("B: sanity + EarlyStopping + ModelCheckpoint",
         {"num_sanity_val_steps": 2},
         callbacks=[ckpt_cb, es_cb])

# 测试 C：启用 sanity check + IntermediateMetricLogger + EpochLogCallback
try:
    from senseframe.engine.runner.orchestrator import EpochLogCallback, IntermediateMetricLogger
    iml = IntermediateMetricLogger(monitor="val_loss", intermediate_values=None)
    elc = EpochLogCallback(log_every_n=10, monitor="val_loss")
    run_test("C: sanity + IntermediateMetricLogger + EpochLogCallback",
             {"num_sanity_val_steps": 2},
             callbacks=[iml, elc])
except Exception as e:
    print(f"测试 C 跳过: {e}")

# 测试 D：全部 callbacks
try:
    ckpt_cb2 = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.3f}")
    es_cb2 = EarlyStopping(monitor="val_loss", mode="min", patience=5, check_on_train_epoch_end=False)
    iml2 = IntermediateMetricLogger(monitor="val_loss", intermediate_values=None)
    elc2 = EpochLogCallback(log_every_n=10, monitor="val_loss")
    run_test("D: sanity + ALL callbacks",
             {"num_sanity_val_steps": 2},
             callbacks=[ckpt_cb2, es_cb2, iml2, elc2])
except Exception as e:
    print(f"测试 D 跳过: {e}")

print("\n诊断完成。")
