"""诊断脚本 2：用 SenseFrame 的完整数据路径 + 纯 PyTorch 训练循环。

目的：隔离问题来源。
- 如果能学习 → 问题在 Lightning Trainer 配置
- 如果不能学习 → 问题在 SenseFrame 数据加载路径（CSIDataset + NTUFiTransform）
"""
import sys
import os

# 设置 SenseFrame 路径
PROJECT_ROOT = "<DEPLOY_ROOT>"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "resource", "WiFi-CSI-Sensing-Benchmark-main"))

os.environ.setdefault("SENSEFRAME_SENSEFI_PATH",
                      "<DEPLOY_ROOT>/resource/WiFi-CSI-Sensing-Benchmark-main")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. 激活 SenseFrame 场景
from senseframe.scenes import activate_lazy_scenes
activate_lazy_scenes()

from senseframe.scenes import get_scene
scene = get_scene("wifi_csi")

# 2. 用 SenseFrame 的 scene API 加载数据（与 pipeline.py stage_load 一致）
DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS"
print("=" * 60)
print("诊断 2：SenseFrame 数据路径 + 纯 PyTorch 训练循环")
print("=" * 60)

# 加载数据集
print("\n加载 NTU-Fi_HAR 数据集（通过 scene.load_dataset）...")
bundle = scene.load_dataset("NTU-Fi_HAR", DATA_ROOT, learning_mode="supervised")
train_ds = bundle.train
test_ds = bundle.test
print(f"  train samples: {len(train_ds)}")
print(f"  test samples: {len(test_ds)}")

# 3. 获取 transform（与 pipeline.py stage_build 一致）
print("\n获取 transform...")
transform_cfg = scene.get_transforms("NTU-Fi_HAR")
print(f"  train_transform: {transform_cfg.train_transform}")
print(f"  eval_transform: {transform_cfg.eval_transform}")

# 4. 用 _TransformWrapper 包装（与 GenericDataModule 一致）
from senseframe.engine.datamodule import _TransformWrapper
train_wrapped = _TransformWrapper(train_ds, transform_cfg.train_transform)
test_wrapped = _TransformWrapper(test_ds, transform_cfg.eval_transform)

# 5. 检查第一个样本
x0, y0 = train_wrapped[0]
print(f"\n包装后第一个样本:")
print(f"  x shape: {x0.shape}, dtype: {x0.dtype}")
print(f"  x min/max/mean: {x0.min():.4f}/{x0.max():.4f}/{x0.mean():.4f}")
print(f"  y: {y0}")

# 6. 创建 DataLoader
train_loader = DataLoader(train_wrapped, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 7. 加载模型
print("\n加载模型 NTU_Fi_ResNet18...")
from NTU_Fi_model import NTU_Fi_ResNet18
model = NTU_Fi_ResNet18(num_classes=6)
n_params = sum(p.numel() for p in model.parameters())
print(f"  参数量: {n_params:,}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device: {device}")

# 8. 纯 PyTorch 训练循环
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)

print("\n" + "=" * 60)
print("训练（纯 PyTorch，SenseFrame 数据路径）")
print("=" * 60)

model = model.to(device)

for epoch in range(5):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    n_batches = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        if epoch == 0 and batch_idx == 0:
            print(f"\n=== Epoch 0 Batch 0 诊断 ===")
            print(f"  inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
            print(f"  inputs min/max/mean: {inputs.min().item():.4f}/{inputs.max().item():.4f}/{inputs.mean().item():.4f}")
            print(f"  labels[:10]: {labels[:10].tolist()}")
            print(f"  labels unique: {torch.unique(labels).tolist()}")

        optimizer.zero_grad()
        outputs = model(inputs)

        if epoch == 0 and batch_idx == 0:
            print(f"  outputs shape: {outputs.shape}")
            print(f"  outputs[0] logits: {outputs[0].tolist()}")
            print(f"  softmax[0]: {torch.softmax(outputs[0], dim=0).tolist()}")

        loss = criterion(outputs, labels)

        if epoch == 0 and batch_idx == 0:
            print(f"  loss: {loss.item():.6f}")

        loss.backward()

        if epoch == 0 and batch_idx == 0:
            total_gn = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_gn += param.grad.norm().item() ** 2
            print(f"  总梯度范数: {total_gn ** 0.5:.6f}")

        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
        predict_y = torch.argmax(outputs, dim=1)
        epoch_acc += (predict_y == labels).sum().item() / labels.size(0)
        n_batches += 1

    epoch_loss = epoch_loss / len(train_loader.dataset)
    epoch_acc = epoch_acc / n_batches
    print(f"Epoch {epoch+1}/5: train_loss={epoch_loss:.6f}, train_acc={epoch_acc:.4f}")

print("\n验证完成。如果 loss 下降，问题在 Lightning Trainer 配置。")
