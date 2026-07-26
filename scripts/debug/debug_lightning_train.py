"""诊断脚本 3：用 Lightning Trainer + GenericLightningModule 训练，加详细诊断。

目的：精确找出 Lightning Trainer 做了什么导致模型不学习。
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
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

# 1. 激活场景
from senseframe.scenes import activate_lazy_scenes
activate_lazy_scenes()
from senseframe.scenes import get_scene
scene = get_scene("wifi_csi")

DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS"

# 2. 加载数据
print("加载 NTU-Fi_HAR 数据集...")
bundle = scene.load_dataset("NTU-Fi_HAR", DATA_ROOT, learning_mode="supervised")
train_ds = bundle.train
test_ds = bundle.test
print(f"  train: {len(train_ds)}, test: {len(test_ds)}")

# 3. 获取 transform
transform_cfg = scene.get_transforms("NTU-Fi_HAR")

# 4. 包装数据
from senseframe.engine.datamodule import _TransformWrapper, GenericDataModule
train_wrapped = _TransformWrapper(train_ds, transform_cfg.train_transform)
test_wrapped = _TransformWrapper(test_ds, transform_cfg.eval_transform)

# 5. 创建 DataModule
datamodule = GenericDataModule(
    train_dataset=train_wrapped,
    test_dataset=test_wrapped,
    val_dataset=test_wrapped,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    learning_mode="supervised",
)

# 6. 加载模型
from NTU_Fi_model import NTU_Fi_ResNet18
model = NTU_Fi_ResNet18(num_classes=6)

# 7. 创建 LightningModule
from senseframe.engine.module import GenericLightningModule
from senseframe.core.task import TaskSpec

task_spec = TaskSpec(
    task_type="classification",
    num_classes=6,
    loss="cross_entropy",
    metrics=["accuracy", "macro_f1"],
)

module = GenericLightningModule(
    model=model,
    num_classes=6,
    task_spec=task_spec,
    optimizer="adam",
    learning_rate=1e-3,
    weight_decay=0.0001,
    scheduler="cosine",
    max_epochs=3,
)

# 8. 注册梯度 hook 诊断
grad_norms = {}
def grad_hook(name):
    def hook(grad):
        grad_norms[name] = grad.norm().item()
    return hook

for name, param in module.named_parameters():
    if param.requires_grad:
        param.register_hook(grad_hook(name))

# 9. 前向 hook 诊断
forward_outputs = {}
def forward_hook(module_obj, input, output):
    if len(forward_outputs) < 3:
        forward_outputs["input_shape"] = input[0].shape if isinstance(input, tuple) else input.shape
        forward_outputs["output_shape"] = output.shape
        forward_outputs["output_sample"] = output[0].detach().cpu().tolist()

# 10. 创建最简单的 Lightning Trainer
print("\n" + "=" * 60)
print("诊断 3：Lightning Trainer 训练")
print("=" * 60)

trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="32",
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
    logger=False,
    num_sanity_val_steps=0,  # 关键：禁用 sanity check
    gradient_clip_val=None,
    accumulate_grad_batches=1,
    deterministic=False,
)

print("开始训练...")
trainer.fit(module, datamodule=datamodule)

print("\n训练完成。")
print(f"最终 training_log: {module.training_log}")
