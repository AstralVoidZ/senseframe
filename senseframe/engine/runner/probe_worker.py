"""Probe 子进程入口模块。

通过 `python -m senseframe.engine.runner.probe_worker` 调用，在独立进程中
执行显存探测，结果以 JSON 输出到 stdout。

设计目的：隔离 probe 的 CUDA 计算到子进程，子进程退出时 CUDA 上下文销毁，
主进程的 CUDA 状态不受影响。

通信协议：
- 输入：命令行参数（简单标量）+ JSON 文件（复杂参数：feature_spec, scene_kwargs）
- 输出：JSON 到 stdout（成功含 measured_vram_mb，失败含 error）

用法：
    python -m senseframe.engine.runner.probe_worker \
        --model-id ResNet18 --dataset UT_HAR_data --num-classes 7 \
        --batch-size 64 --precision 16-mixed --optimizer adam \
        --data-root /path/to/data --scene-name wifi_csi \
        --params-file /tmp/probe_params.json
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SenseFrame VRAM probe worker（子进程隔离）"
    )
    parser.add_argument("--model-id", required=True, help="模型 ID")
    parser.add_argument("--dataset", required=True, help="数据集名")
    parser.add_argument("--num-classes", type=int, required=True, help="类别数")
    parser.add_argument("--learning-mode", default="supervised",
                        help="学习模式（supervised/self_supervised）")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--precision", default="32", help="精度（32/16-mixed/...）")
    parser.add_argument("--optimizer", default="adam", help="优化器名")
    parser.add_argument("--data-root", required=True, help="数据根目录")
    parser.add_argument("--scene-name", required=True, help="场景名")
    parser.add_argument("--params-file", default=None,
                        help="JSON 文件路径，含复杂参数（feature_spec, scene_kwargs, scene_info）")

    args = parser.parse_args()

    # 加载复杂参数（可选）
    feature_spec = None
    scene_kwargs = {}
    scene_info = {}
    if args.params_file:
        try:
            with open(args.params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
            feature_spec = params.get("feature_spec")
            scene_kwargs = params.get("scene_kwargs", {})
            scene_info = params.get("scene_info", {})
        except Exception as e:
            print(json.dumps({
                "error": f"读取 params-file 失败: {e}",
                "error_type": type(e).__name__,
            }))
            sys.exit(1)

    # 执行 probe
    try:
        result = _do_probe(
            model_id=args.model_id,
            dataset=args.dataset,
            num_classes=args.num_classes,
            learning_mode=args.learning_mode,
            batch_size=args.batch_size,
            precision=args.precision,
            optimizer=args.optimizer,
            data_root=args.data_root,
            scene_name=args.scene_name,
            feature_spec=feature_spec,
            scene_kwargs=scene_kwargs,
            scene_info=scene_info,
        )
        print(json.dumps(result))
    except Exception as e:
        tb = traceback.format_exc()
        print(json.dumps({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb,
        }))
        sys.exit(1)


def _do_probe(
    model_id: str,
    dataset: str,
    num_classes: int,
    learning_mode: str,
    batch_size: int,
    precision: str,
    optimizer: str,
    data_root: str,
    scene_name: str,
    feature_spec: dict = None,
    scene_kwargs: dict = None,
    scene_info: dict = None,
) -> dict:
    """在子进程中执行显存探测。

    独立实现测量逻辑（不依赖 pipeline._run_vram_probe，该函数已于 2026-07-12 移除），
    通过 scene API 重建模型和加载数据集。
    """
    import torch

    # 1. 导入 scene container
    from ...scenes import get_scene
    scene = get_scene(scene_name)

    # 2. 重建模型
    # build_model_for_dataset 需要 input_dim/feature_spec，从 feature_spec 派生
    input_dim = None
    if feature_spec and isinstance(feature_spec, dict):
        input_dim = feature_spec.get("feature_dim")
    elif scene_info and isinstance(scene_info, dict):
        input_dim = scene_info.get("n_features")

    # 构造 FeatureSpec 对象（如果传入了）
    fs_obj = None
    if feature_spec and isinstance(feature_spec, dict):
        try:
            from ...core.features import FeatureSpec
            fs_obj = FeatureSpec(**feature_spec)
        except Exception:
            pass  # FeatureSpec 构造失败，回退到 input_dim

    model = scene.build_model_for_dataset(
        model_id, dataset, num_classes,
        learning_mode=learning_mode,
        data_root=data_root,
        input_dim=input_dim,
        feature_spec=fs_obj,
        **(scene_kwargs or {}),
    )

    # 3. 加载数据集，取样本
    bundle = scene.load_dataset(dataset, data_root, learning_mode=learning_mode)
    train_ds = bundle.train if hasattr(bundle, "train") else None
    if train_ds is None:
        # 自监督模式可能没有 train，用 unsupervised
        train_ds = getattr(bundle, "unsupervised", None)
    if train_ds is None:
        return {
            "skipped": "no_train_dataset",
            "measured_vram_mb": None,
        }

    # 4. 执行 probe（独立实现：eval + no_grad + 静态计算 optimizer state）
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # 取样本并扩展到 batch_size
    sample = train_ds[0]
    if isinstance(sample, (list, tuple)):
        x_sample = sample[0]
        y_sample = sample[1] if len(sample) >= 2 else None
    elif isinstance(sample, dict):
        x_sample = sample.get("x") or sample.get("input") or list(sample.values())[0]
        y_sample = sample.get("y") or sample.get("label")
    else:
        x_sample = sample
        y_sample = None

    # 修复：应用 scene 的 transform 到 sample
    # 旧逻辑直接用 raw sample 做前向传播，未应用 transform（归一化/reshape/stride），
    # 导致 NTU-Fi_HAR/Widar 等需要 transform 的数据集前向传播失败
    # （shape 不匹配 + dtype 不匹配：raw float64 vs 16-mixed 的 Half 权重）。
    # UT_HAR_data 不受影响（tensor_loader 直接返回正确 shape，无 transform）。
    # P5 P3-10：transform 失败时记录 warning 而非静默 pass。
    # 旧代码 except Exception: pass 吞掉所有错误（含归一化注册表丢失），
    # 导致 raw sample 进入 probe，VRAM 测量结果偏离实际训练场景。
    transform_applied = True
    transform_warning = None
    try:
        tc = scene.get_transforms(dataset, **(scene_kwargs or {}))
        if tc.train_transform is not None and y_sample is not None:
            x_sample, y_sample = tc.train_transform(x_sample, y_sample)
    except Exception as e:
        transform_applied = False
        transform_warning = f"probe_worker: transform failed ({type(e).__name__}: {e}), using raw sample"

    # 防御性 float()：确保 float32（raw sample 可能是 float64，16-mixed 权重是 Half）
    x = x_sample.float().unsqueeze(0).repeat(batch_size, *[1] * x_sample.dim()).to(device)

    # 静态计算参数显存
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params_mb = params_bytes / (1024 * 1024)

    # 重置峰值统计
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # 前向（no_grad）
    use_autocast = isinstance(precision, str) and precision.startswith("16")
    with torch.no_grad():
        with torch.autocast("cuda", enabled=use_autocast):
            output = model(x)

    # 测量峰值
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024 * 1024)
    activation_mb = max(0.0, peak_mb - params_mb)

    # 静态计算梯度 + optimizer state
    gradient_mb = params_mb

    opt_state_multiplier = 0
    opt_name_lower = optimizer.lower()
    if opt_name_lower in ("adam", "adamw"):
        opt_state_multiplier = 2
    elif opt_name_lower == "sgd":
        opt_state_multiplier = 1
    optimizer_state_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) * opt_state_multiplier
    optimizer_state_mb = optimizer_state_bytes / (1024 * 1024)

    # 总计
    measured_vram_mb = params_mb + activation_mb + gradient_mb + optimizer_state_mb

    # 清理
    try:
        del output, x, model
    except NameError:
        pass
    torch.cuda.empty_cache()

    # 判断
    needed_vram_mb = measured_vram_mb * 1.15
    free_vram_mb = torch.cuda.mem_get_info(device)[0] / (1024 * 1024)

    return {
        "measured_vram_mb": round(measured_vram_mb, 1),
        "needed_vram_mb": round(needed_vram_mb, 1),
        "free_vram_mb": round(free_vram_mb, 1),
        "ok": free_vram_mb >= needed_vram_mb,
        "batch_size": batch_size,
        "precision": precision,
        "optimizer": optimizer,
        "breakdown_mb": {
            "params": round(params_mb, 1),
            "activation": round(activation_mb, 1),
            "gradient": round(gradient_mb, 1),
            "optimizer_state": round(optimizer_state_mb, 1),
        },
        "transform_applied": transform_applied,
        "warnings": [transform_warning] if transform_warning else [],
    }


if __name__ == "__main__":
    main()
