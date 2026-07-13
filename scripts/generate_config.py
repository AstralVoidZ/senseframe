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
    4. 自监督模式从 supervised_source 数据集 spec 派生 num_classes
"""

import argparse
import logging
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

_logger = logging.getLogger(__name__)


def _get_dataset_meta(dataset: str) -> dict:
    """从 DATASET_INFO 获取数据集元信息，未注册则 raise（框架不猜测）。"""
    info = DATASET_INFO.get(dataset)
    if info is None:
        raise KeyError(
            f"数据集 '{dataset}' 未注册，无法派生 num_classes/input_shape。"
            f"请先通过场景注册（register_dataset）声明该数据集元数据。"
        )
    return {
        "num_classes": info["num_classes"],
        "input_shape": list(info["input_shape"]),
    }


def _derive_supervised_num_classes(dataset: str) -> int:
    """自监督模式：从 DatasetSpec.supervised_source 派生 num_classes。

    框架不猜测：supervised_source 为空或未注册则 raise，不回退 fallback，
    不硬编码 "NTU-Fi-HumanID"。
    """
    from senseframe.registry import get_dataset_spec, is_dataset_registered
    if not is_dataset_registered(dataset):
        raise KeyError(
            f"数据集 '{dataset}' 未注册，无法派生 supervised_source。"
            f"请先通过场景注册声明该数据集元数据。"
        )
    spec = get_dataset_spec(dataset)
    supervised_source = spec.supervised_source
    if not supervised_source:
        raise ValueError(
            f"数据集 '{dataset}' 的 DatasetSpec.supervised_source 为空，"
            f"无法派生 num_classes。请在注册时声明 supervised_source。"
        )
    if not is_dataset_registered(supervised_source):
        raise KeyError(
            f"数据集 '{dataset}' 的 supervised_source '{supervised_source}' 未注册，"
            f"无法派生 num_classes。"
        )
    return get_dataset_spec(supervised_source).num_classes


def build_config(dataset: str, model: str, mode: str, epochs: int,
                 batch_size: int, learning_rate: float, output_dir: str,
                 metrics: list, ss_epochs: int, data_root: str = "") -> dict:
    """构建 ExperimentConfig dict。"""
    meta = _get_dataset_meta(dataset)

    # 自监督模式：从 supervised_source 数据集 spec 派生 num_classes
    if mode == "self_supervised":
        num_classes = _derive_supervised_num_classes(dataset)
    else:
        num_classes = meta["num_classes"]

    config = {
        "scene": {
            "name": "wifi_csi",
            "dataset": dataset,
            "model_id": model,
            "learning_mode": mode,
            # P1-3 修复：data_root 由 main() 解析（CLI > env > 默认路径探测）
            "data_root": data_root,
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
            # RFC-004 方案 E：默认即最佳实践，显式写入便于用户感知/编辑
            "weight_decay": 1e-4,
            "early_stopping": 5,
            "early_stopping_min_delta": 0.001,
            "scheduler": "cosine",
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
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据集根目录（不指定则依次检查 SENSEFRAME_DATA_ROOT 环境变量、"
                             "resource/CSI_DATASETS 默认路径）")

    args = parser.parse_args()

    # 自监督模式校验：从注册表 DatasetSpec.supervised_source 派生，不硬编码数据集名
    if args.mode == "self_supervised":
        from senseframe.registry import get_dataset_spec, is_dataset_registered
        if not is_dataset_registered(args.dataset):
            print(f"错误：数据集 '{args.dataset}' 未注册，无法校验自监督模式支持",
                  file=sys.stderr)
            sys.exit(1)
        spec = get_dataset_spec(args.dataset)
        if not spec.supervised_source:
            print(f"错误：数据集 '{args.dataset}' 未声明 supervised_source，"
                  f"不支持自监督模式", file=sys.stderr)
            sys.exit(1)

    # 确定默认 epochs：方案 B 去静态化后，完全由 _compute_epochs_budget(n_samples) 动态计算
    if args.epochs is None:
        try:
            from senseframe.registry import get_default_epochs, get_dataset_spec
            # 方案 B：epochs 完全动态，必须传入 n_samples。
            # 数据集未注册时 spec 查询失败 → n_samples=None → get_default_epochs raise
            # （框架不猜测、不回退默认值）
            try:
                n_samples = get_dataset_spec(args.dataset).n_samples
            except Exception:
                n_samples = None
            default_epochs = get_default_epochs(
                args.model, args.dataset, scene_name="wifi_csi",
                n_samples=n_samples)
        except Exception as e:
            print(f"错误：get_default_epochs 失败 ({e})。"
                  f"请通过 --epochs 显式指定，或确保数据集已注册默认 epochs。",
                  file=sys.stderr)
            sys.exit(1)
    else:
        default_epochs = args.epochs

    # P1-3 修复：解析 data_root（优先级：CLI > env > 默认路径探测）
    # P4-6 修复：探测到的路径转为相对路径写入 YAML，提升配置可移植性。
    # 训练期的 resolve_data_root 会将相对路径 .resolve() 为绝对路径，功能不受影响。
    import os
    if args.data_root is not None:
        data_root = args.data_root
    elif os.environ.get("SENSEFRAME_DATA_ROOT"):
        data_root = os.environ["SENSEFRAME_DATA_ROOT"]
    else:
        # 探测默认路径 resource/CSI_DATASETS（相对于 cwd 或项目根）
        _candidate = Path.cwd() / "resource" / "CSI_DATASETS"
        if _candidate.exists():
            data_root = os.path.relpath(_candidate, Path.cwd())
        else:
            _candidate = _PROJECT_ROOT / "resource" / "CSI_DATASETS"
            if _candidate.exists():
                data_root = os.path.relpath(_candidate, Path.cwd())
            else:
                data_root = ""

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
        data_root=data_root,
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
