"""NTU-Fi_HAR val_loss 爆炸根因调查。

调查维度：
1. raw 数据统计特征（train vs val 分布是否一致）
2. 归一化后的数据统计（是否正确归一化）
3. transform 后的数据统计（shape/dtype/range）
4. 模型 forward 输出（logits range 是否正常）
5. mixed precision 行为（Half 精度是否溢出）
6. val set 的类别分布（是否严重不平衡）
"""
import os, sys, json, traceback
import numpy as np

os.environ["SENSEFRAME_DATA_ROOT"] = "<DEPLOY_ROOT>/resource/CSI_DATASETS"
os.environ["SENSEFRAME_SENSEFI_PATH"] = "<DEPLOY_ROOT>/resource/WiFi-CSI-Sensing-Benchmark-main"
sys.path.insert(0, "<DEPLOY_ROOT>")

import torch
import senseframe as sf
sf.activate_lazy_scenes()

from senseframe.registry import get_dataset_spec, get_model, get_normalization
from senseframe.scenes.wifi_csi.container import WiFiCSIContainer, NTUFiTransform

print("=" * 70)
print("1. 归一化策略检查")
print("=" * 70)
strategy = get_normalization("NTU-Fi_HAR")
print("strategy type:", type(strategy).__name__)
print("strategy dict:", strategy.to_dict())
print("is_noop:", strategy.is_noop())

print("\n" + "=" * 70)
print("2. 加载数据集 + _auto_split_val")
print("=" * 70)
container = WiFiCSIContainer()
bundle = container.load_dataset("NTU-Fi_HAR", os.environ["SENSEFRAME_DATA_ROOT"], learning_mode="supervised")
print("train len:", len(bundle.train))
print("val len:", len(bundle.val))
print("test len:", len(bundle.test))

print("\n" + "=" * 70)
print("3. raw 数据统计（train vs val 分布对比）")
print("=" * 70)

def collect_raw_stats(dataset, name, n=100):
    """收集 raw 数据统计"""
    xs, ys = [], []
    for i in range(min(n, len(dataset))):
        x, y = dataset[i]
        xs.append(x.numpy() if hasattr(x, "numpy") else np.array(x))
        ys.append(int(y))
    xs = np.array(xs)
    ys = np.array(ys)
    print(f"\n{name} raw stats (n={len(xs)}):")
    print(f"  x shape: {xs.shape}")
    print(f"  x dtype: {xs.dtype}")
    print(f"  x mean: {xs.mean():.4f}")
    print(f"  x std: {xs.std():.4f}")
    print(f"  x min: {xs.min():.4f}")
    print(f"  x max: {xs.max():.4f}")
    print(f"  y unique: {np.unique(ys, return_counts=True)}")
    return xs, ys

train_raw, train_y = collect_raw_stats(bundle.train, "TRAIN")
val_raw, val_y = collect_raw_stats(bundle.val, "VAL")
test_raw, test_y = collect_raw_stats(bundle.test, "TEST", n=100)

print("\n" + "=" * 70)
print("4. train/val 分布一致性检验")
print("=" * 70)
print(f"train mean={train_raw.mean():.4f} std={train_raw.std():.4f}")
print(f"val   mean={val_raw.mean():.4f} std={val_raw.std():.4f}")
print(f"test  mean={test_raw.mean():.4f} std={test_raw.std():.4f}")
print(f"train/val mean diff: {abs(train_raw.mean() - val_raw.mean()):.4f}")
print(f"train/val std diff: {abs(train_raw.std() - val_raw.std()):.4f}")

print("\n" + "=" * 70)
print("5. transform 后数据统计")
print("=" * 70)
transform = NTUFiTransform()
x_t, y_t = transform(torch.from_numpy(train_raw[0]), int(train_y[0]))
print(f"transformed x shape: {x_t.shape}")
print(f"transformed x dtype: {x_t.dtype}")
print(f"transformed x mean: {x_t.mean().item():.4f}")
print(f"transformed x std: {x_t.std().item():.4f}")
print(f"transformed x min: {x_t.min().item():.4f}")
print(f"transformed x max: {x_t.max().item():.4f}")

# 对 val 样本也做 transform
x_v, y_v = transform(torch.from_numpy(val_raw[0]), int(val_y[0]))
print(f"\nval transformed x mean: {x_v.mean().item():.4f}")
print(f"val transformed x std: {x_v.std().item():.4f}")
print(f"val transformed x min: {x_v.min().item():.4f}")
print(f"val transformed x max: {x_v.max().item():.4f}")

print("\n" + "=" * 70)
print("6. 模型 forward 输出检查（fp32 vs fp16）")
print("=" * 70)
spec = get_dataset_spec("NTU-Fi_HAR")
model = get_model("ResNet18", "NTU-Fi_HAR", spec.num_classes, "supervised", scene_name="wifi_csi")
print(f"model type: {type(model).__name__}")

# 构造 batch
batch_x = x_t.unsqueeze(0).repeat(4, 1, 1, 1)  # (4, 3, 114, 500)
print(f"batch_x shape: {batch_x.shape}, dtype: {batch_x.dtype}")

# fp32 forward
model.eval()
with torch.no_grad():
    out_fp32 = model(batch_x)
print(f"\nfp32 output shape: {out_fp32.shape}")
print(f"fp32 output dtype: {out_fp32.dtype}")
print(f"fp32 output min: {out_fp32.min().item():.4f}")
print(f"fp32 output max: {out_fp32.max().item():.4f}")
print(f"fp32 output mean: {out_fp32.mean().item():.4f}")
print(f"fp32 logits sample: {out_fp32[0]}")

# fp16 (autocast) forward
with torch.no_grad():
    with torch.autocast("cpu", enabled=True, dtype=torch.float16):
        out_fp16 = model(batch_x)
print(f"\nfp16 output shape: {out_fp16.shape}")
print(f"fp16 output dtype: {out_fp16.dtype}")
print(f"fp16 output min: {out_fp16.min().item():.4f}")
print(f"fp16 output max: {out_fp16.max().item():.4f}")
print(f"fp16 output mean: {out_fp16.mean().item():.4f}")
print(f"fp16 logits sample: {out_fp16[0]}")

# fp16 loss 计算（模拟 validation_step）
import torch.nn.functional as F
y_batch = torch.tensor([int(train_y[0])] * 4)
print(f"\ny_batch: {y_batch}")

# fp32 loss
with torch.no_grad():
    loss_fp32 = F.cross_entropy(out_fp32, y_batch)
print(f"fp32 loss: {loss_fp32.item():.6f}")

# fp16 loss（模拟 mixed precision val）
with torch.no_grad():
    with torch.autocast("cpu", enabled=True, dtype=torch.float16):
        out_fp16_2 = model(batch_x)
        loss_fp16 = F.cross_entropy(out_fp16_2, y_batch)
print(f"fp16 loss: {loss_fp16.item():.6f}")

print("\n" + "=" * 70)
print("7. 检查 val set 类别分布")
print("=" * 70)
val_classes, val_counts = np.unique(val_y, return_counts=True)
train_classes, train_counts = np.unique(train_y, return_counts=True)
print(f"train classes: {dict(zip(train_classes.tolist(), train_counts.tolist()))}")
print(f"val classes: {dict(zip(val_classes.tolist(), val_counts.tolist()))}")
print(f"val class balance: min={val_counts.min()}, max={val_counts.max()}, ratio={val_counts.max()/val_counts.min():.2f}")

print("\n" + "=" * 70)
print("8. 检查 raw 数据是否有异常值（NaN/Inf/极端值）")
print("=" * 70)
print(f"train raw has NaN: {np.isnan(train_raw).any()}")
print(f"train raw has Inf: {np.isinf(train_raw).any()}")
print(f"val raw has NaN: {np.isnan(val_raw).any()}")
print(f"val raw has Inf: {np.isinf(val_raw).any()}")
print(f"train raw abs max: {np.abs(train_raw).max():.4f}")
print(f"val raw abs max: {np.abs(val_raw).max():.4f}")

# 检查每个 val 样本的 raw 统计是否有异常
print("\n=== 逐个 val 样本 raw 统计 ===")
for i in range(min(10, len(bundle.val))):
    x, y = bundle.val[i]
    x_np = x.numpy() if hasattr(x, "numpy") else np.array(x)
    print(f"  val[{i}] y={y} shape={x_np.shape} mean={x_np.mean():.4f} std={x_np.std():.4f} absmax={np.abs(x_np).max():.4f}")
