#!/usr/bin/env python
"""生成 senseframe YAML 配置的封装接口。

用法：
    # 生成监督学习配置
    python generate_config.py --dataset UT_HAR_data --model ResNet18 --mode supervised

    # 生成自监督学习配置
    python generate_config.py --dataset NTU-Fi_HAR --model ResNet18 --mode self_supervised

    # 生成并保存到文件
    python generate_config.py --dataset UT_HAR_data --model MLP --output configs/exp.yaml

功能：
    1. 根据参数生成 ExperimentConfig YAML
    2. 自动填充数据集对应的 num_classes 和 input_shape
    3. 输出到 stdout 或指定文件
    4. 自监督模式自动设置 num_classes=14
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path（脚本可从任意位置调用）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

# 入口点契约：查询 registry 前必须显式激活场景（CQS 合规改造后，getter 无副作用）
from senseframe.scenes import activate_lazy_scenes  # noqa: E402
activate_lazy_scenes()

# 数据集元信息（num_classes, input_shape）
# Phase 0 优化：从 senseframe.registry.DATASET_INFO 单一来源导入，消除数据漂移
# 注意：input_shape 在 DATASET_INFO 中为 tuple，转为 list 供 YAML 序列化
from senseframe.registry import DATASET_INFO  # noqa: E402


def _get_dataset_meta(dataset: str) -> dict:
    """从 DATASET_INFO 获取数据集元信息，缺失时回退到通用默认值。"""
    info = DATASET_INFO.get(dataset)
    if info is None:
        return {"num_classes": 7, "input_shape": [270, 3]}
    return {
        "num_classes": info["num_classes"],
        "input_shape": list(info["input_shape"]),
    }


def build_config(dataset: str, model: str, mode: str, epochs: int,
                 batch_size: int, learning_rate: float, output_dir: str,
                 metrics: list, ss_epochs: int) -> dict:
    """构建 ExperimentConfig dict。"""
    meta = _get_dataset_meta(dataset)

    # 自监督模式硬编码 num_classes=14（NTU-Fi-HumanID）
    if mode == "self_supervised":
        num_classes = 14
    else:
        num_classes = meta["num_classes"]

    config = {
        "scene": {
            "name": "wifi_csi",
            "dataset": dataset,
            "model_id": model,
            "learning_mode": mode,
            "params": {
                "metrics": metrics,
                "average": "macro",
            },
        },
        "input_features": [
            {
                "name": "csi",
                "type": "csi",
                "shape": meta["input_shape"],
            }
        ],
        "output_features": [
            {
                "name": "action",
                "type": "category",
                "num_classes": num_classes,
            }
        ],
        "trainer": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": "adam",
            "seed": 42,
        },
        "output_dir": output_dir,
        "save_model": True,
    }

    # 自监督模式添加 Phase 1 轮数
    if mode == "self_supervised":
        config["scene"]["params"]["self_supervised_epochs"] = ss_epochs

    return config


def main():
    parser = argparse.ArgumentParser(
        prog="generate_config",
        description="生成 senseframe YAML 配置",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名（如 UT_HAR_data, NTU-Fi_HAR）")
    parser.add_argument("--model", type=str, required=True,
                        help="模型 ID（如 ResNet18, MLP）")
    parser.add_argument("--mode", type=str,
                        choices=["supervised", "self_supervised"],
                        default="supervised", help="学习模式")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（不指定则用数据集默认值）")
    parser.add_argument("--ss-epochs", type=int, default=100,
                        help="自监督预训练轮数（仅 self_supervised 模式）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="输出目录")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["accuracy", "macro_f1"],
                        help="评测指标列表")
    parser.add_argument("--output", type=str,
                        help="输出文件路径（不指定则输出到 stdout）")

    args = parser.parse_args()

    # 自监督模式校验
    if args.mode == "self_supervised" and args.dataset != "NTU-Fi_HAR":
        print(f"错误：自监督模式仅支持 dataset=NTU-Fi_HAR，当前为 {args.dataset}",
              file=sys.stderr)
        sys.exit(1)

    # 确定默认 epochs
    if args.epochs is None:
        if args.mode == "self_supervised":
            default_epochs = 30  # NTU-Fi_HAR 的监督微调默认 epochs
        else:
            # 从 EPOCHS_TABLE 获取默认值
            try:
                from senseframe.registry import get_default_epochs
                default_epochs = get_default_epochs(args.model, args.dataset)
            except Exception:
                default_epochs = 100
    else:
        default_epochs = args.epochs

    # 构建配置
    config = build_config(
        dataset=args.dataset,
        model=args.model,
        mode=args.mode,
        epochs=default_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        metrics=args.metrics,
        ss_epochs=args.ss_epochs,
    )

    # 输出 YAML
    yaml_content = yaml.dump(config, default_flow_style=False,
                             allow_unicode=True, sort_keys=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"配置已保存到: {args.output}", file=sys.stderr)
    else:
        print(yaml_content)


if __name__ == "__main__":
    main()
