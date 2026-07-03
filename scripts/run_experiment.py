#!/usr/bin/env python
"""执行 senseframe 训练实验的封装接口。

用法：
    python run_experiment.py --config configs/exp.yaml
    python run_experiment.py --config configs/exp.yaml --model ResNet50 --epochs 100
    python run_experiment.py --config configs/exp.yaml --hpo --hpo-trials 30

功能：
    1. 加载 YAML 配置 → ExperimentConfig
    2. 应用 CLI 覆盖参数
    3. 调用 run_experiment 或 run_hpo 执行训练
    4. 输出结构化 JSON 结果
    5. 退出码反映训练状态（0=成功，1=错误）
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path（脚本可从任意位置调用）
# 脚本路径: <project_root>/scripts/<script>.py
# parents[1] = <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

from senseframe.engine import ExperimentConfig, run_experiment, run_hpo  # noqa: E402
from senseframe.observability import setup_logging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description="执行 senseframe 训练实验（YAML 配置 → ExperimentConfig → 训练）",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    # CLI 覆盖参数（可选）
    parser.add_argument("--scene", type=str, help="覆盖 scene.name")
    parser.add_argument("--dataset", type=str, help="覆盖 scene.dataset")
    parser.add_argument("--model", type=str, help="覆盖 scene.model_id")
    parser.add_argument("--epochs", type=int, help="覆盖 trainer.epochs")
    parser.add_argument("--batch-size", type=int, help="覆盖 trainer.batch_size")
    parser.add_argument("--learning-rate", type=float, help="覆盖 trainer.learning_rate")
    parser.add_argument("--output-dir", type=str, help="覆盖 output_dir")
    # HPO 选项
    parser.add_argument("--hpo", action="store_true",
                        help="启用超参搜索（覆盖 hpo.enabled=True）")
    parser.add_argument("--hpo-trials", type=int, help="HPO trial 数量")
    parser.add_argument("--hpo-metric", type=str, help="HPO 优化指标名")
    parser.add_argument("--hpo-direction", type=str,
                        choices=["minimize", "maximize"], help="HPO 优化方向")
    # 日志控制
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"], help="日志级别")
    parser.add_argument("--log-file", type=str, help="日志文件路径")

    args = parser.parse_args()

    # 配置日志
    setup_logging(level=args.log_level, log_file=args.log_file)

    # 加载 YAML 配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(json.dumps({
            "error": f"Config file not found: {args.config}",
            "code": "CONFIG_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        print(json.dumps({
            "error": "Config file must contain a YAML mapping at top level",
            "code": "INVALID_CONFIG_FORMAT",
        }, ensure_ascii=False))
        sys.exit(1)

    # 解析为 ExperimentConfig
    try:
        config = ExperimentConfig.from_dict(config_dict)
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "code": "CONFIG_PARSE_ERROR",
        }, ensure_ascii=False))
        sys.exit(1)

    # 应用 CLI 覆盖
    if args.scene:
        config.scene.name = args.scene
    if args.dataset:
        config.scene.dataset = args.dataset
    if args.model:
        config.scene.model_id = args.model
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.output_dir:
        config.output_dir = args.output_dir

    # HPO 模式
    if args.hpo:
        config.hpo.enabled = True
        if args.hpo_trials is not None:
            config.hpo.n_trials = args.hpo_trials
        if args.hpo_metric:
            config.hpo.metric = args.hpo_metric
        if args.hpo_direction:
            config.hpo.direction = args.hpo_direction

    # 校验配置（快速失败）
    try:
        config.validate()
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "code": "CONFIG_VALIDATION_ERROR",
        }, ensure_ascii=False))
        sys.exit(1)

    # 执行训练
    if config.hpo.enabled:
        result = run_hpo(config)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        output = run_experiment(config)
        print(json.dumps(output.to_dict(), indent=2, ensure_ascii=False, default=str))
        if output.status == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()
