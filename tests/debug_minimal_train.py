"""最小化 PyTorch 训练脚本：绕过 Lightning，直接用原始 SenseFi 方式训练 NTU_Fi_ResNet18。

目的：验证模型是否能学习。如果 loss 下降 → 问题在 Lightning 配置；
如果 loss 不下降 → 问题在模型/数据。
"""
import sys
import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SENSEFI_PATH = "<DEPLOY_ROOT>/resource/WiFi-CSI-Sensing-Benchmark-main"
if SENSEFI_PATH not in sys.path:
    sys.path.insert(0, SENSEFI_PATH)
DATA_ROOT = "<DEPLOY_ROOT>/resource/CSI_DATASETS/NTU-Fi_HAR"


class CSI_Dataset(Dataset):
    """原始 SenseFi CSI_Dataset，完全复制。"""
    def __init__(self, root_dir, modal="CSIamp", transform=None):
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + "/*/*.mat")
        self.folder = glob.glob(root_dir + "/*/")
        self.category = {self.folder[i].split("/")[-2]: i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split("/")[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        x = (x - 42.3199) / 4.9802
        x = x[:, ::4]
        x = x.reshape(3, 114, 500)
        if self.transform:
            x = self.transform(x)
        x = torch.FloatTensor(x)
        return x, y


def train_minimal(model, train_loader, num_epochs, lr, criterion, device, wd=0.0, use_sched=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) if use_sched else None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if epoch == 0 and batch_idx == 0:
                print(f"\n=== Epoch 0 Batch 0 诊断 ===")
                print(f"  inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
                print(f"  inputs min/max/mean: {inputs.min().item():.4f}/{inputs.max().item():.4f}/{inputs.mean().item():.4f}")
                print(f"  labels[:10]: {labels[:10].tolist()}")
                print(f"  labels unique: {torch.unique(labels).tolist()}")
                print(f"  outputs shape: {outputs.shape}")
                print(f"  outputs[0] logits: {outputs[0].tolist()}")
                print(f"  softmax[0]: {torch.softmax(outputs[0], dim=0).tolist()}")
                print(f"  loss: {loss.item():.6f}")

            loss.backward()

            if epoch == 0 and batch_idx == 0:
                print(f"\n=== 梯度诊断 ===")
                total_gn = 0.0
                zero_g = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gn = param.grad.norm().item()
                        total_gn += gn ** 2
                        if gn < 1e-10:
                            zero_g += 1
                    else:
                        print(f"  {name}: grad is None!")
                print(f"  总梯度范数: {total_gn ** 0.5:.6f}")
                print(f"  零梯度参数数: {zero_g}/{len(list(model.parameters()))}")
                for i, (name, param) in enumerate(model.named_parameters()):
                    if param.grad is not None:
                        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
                    if i >= 5:
                        break

            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_acc += (predict_y == labels).sum().item() / labels.size(0)
            n_batches += 1

        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_acc = epoch_acc / n_batches
        if scheduler:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={epoch_loss:.6f}, train_acc={epoch_acc:.4f}, lr={cur_lr:.8f}")


def main():
    print("=" * 60)
    print("最小化 PyTorch 训练验证：NTU_Fi_ResNet18 + NTU-Fi_HAR")
    print("=" * 60)
    train_dir = os.path.join(DATA_ROOT, "train_amp")
    print(f"\ntrain_dir: {train_dir} (exists: {os.path.exists(train_dir)})")
    classes = sorted(glob.glob(train_dir + "/*/"))
    print(f"类别: {[c.split('/')[-2] for c in classes]}")
    print("\n加载数据集...")
    train_set = CSI_Dataset(train_dir)
    print(f"  train samples: {len(train_set)}")
    print(f"  categories: {train_set.category}")
    x0, y0 = train_set[0]
    print(f"  sample[0]: x shape={x0.shape}, dtype={x0.dtype}, y={y0}")
    print(f"  x min/max/mean: {x0.min():.4f}/{x0.max():.4f}/{x0.mean():.4f}")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    print("\n加载模型 NTU_Fi_ResNet18...")
    from NTU_Fi_model import NTU_Fi_ResNet18
    model = NTU_Fi_ResNet18(num_classes=6)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("实验 A：原始 SenseFi 风格（无 wd, 无 scheduler）")
    print("=" * 60)
    model_a = NTU_Fi_ResNet18(num_classes=6)
    train_minimal(model_a, train_loader, num_epochs=5, lr=1e-3, criterion=criterion, device=device, wd=0.0, use_sched=False)

    print("\n" + "=" * 60)
    print("实验 B：SenseFrame 风格（wd=0.0001, cosine scheduler）")
    print("=" * 60)
    model_b = NTU_Fi_ResNet18(num_classes=6)
    train_minimal(model_b, train_loader, num_epochs=5, lr=1e-3, criterion=criterion, device=device, wd=0.0001, use_sched=True)

    print("\n验证完成。")


if __name__ == "__main__":
    main()
