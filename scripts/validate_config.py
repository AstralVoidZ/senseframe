#!/usr/bin/env python
"""校验 senseframe YAML 配置的封装接口。

用法：
    python validate_config.py --config configs/exp.yaml

功能：
    1. 加载 YAML 配置
    2. 解析为 ExperimentConfig
    3. 执行完整 schema 校验
    4. 报告所有校验错误（不执行训练）
    5. 退出码反映校验结果（0=通过，1=失败）

本脚本不产生任何副作用，仅做配置校验。
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
# 脚本路径: <project_root>/scripts/<script>.py
# parents[1] = <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

# 入口点契约：查询 registry 前必须显式激活场景（CQS 合规改造后，getter 无副作用）
from senseframe.scenes import activate_lazy_scenes  # noqa: E402
activate_lazy_scenes()

from senseframe.engine import ExperimentConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="validate_config",
        description="校验 senseframe YAML 配置（不执行训练）",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")

    args = parser.parse_args()

    # 加载 YAML 配置
    config_path = Path(args.config)
    if not config_path.exists():
        # P2 演进（2026-07-18）：错误输出到 stderr，与 cli.py 对齐
        print(json.dumps({
            "valid": False,
            "error": f"Config file not found: {args.config}",
            "code": "CONFIG_NOT_FOUND",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        print(json.dumps({
            "valid": False,
            "error": "Config file must contain a YAML mapping at top level",
            "code": "INVALID_CONFIG_FORMAT",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    # 解析为 ExperimentConfig
    try:
        config = ExperimentConfig.from_dict(config_dict)
    except ValueError as e:
        # P2 演进：from_dict 失败统一用 CONFIG_PARSE_ERROR（YAML 解析/结构阶段），
        # 与 config.validate() 阶段的 CONFIG_VALIDATION_ERROR 区分。
        print(json.dumps({
            "valid": False,
            "error": str(e),
            "code": "CONFIG_PARSE_ERROR",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    # 执行完整校验
    try:
        config.validate()
    except ValueError as e:
        print(json.dumps({
            "valid": False,
            "error": str(e),
            "code": "CONFIG_VALIDATION_ERROR",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    # 校验通过，输出配置摘要
    summary = {
        "valid": True,
        "scene": {
            "name": config.scene.name,
            "dataset": config.scene.dataset,
            "model_id": config.scene.model_id,
            "learning_mode": config.scene.learning_mode,
        },
        "trainer": {
            "epochs": config.trainer.epochs,
            "batch_size": config.trainer.batch_size,
            "learning_rate": config.trainer.learning_rate,
            "optimizer": config.trainer.optimizer,
        },
        "hpo_enabled": config.hpo.enabled,
        "output_dir": config.output_dir,
    }
    # Phase 12.3：task_spec 摘要
    if config.scene.task_spec is not None:
        summary["task_spec"] = {
            "task_type": config.scene.task_spec.task_type,
            "num_classes": config.scene.task_spec.num_classes,
            "loss": config.scene.task_spec.loss,
            "metrics": config.scene.task_spec.metrics,
            "output_activation": config.scene.task_spec.output_activation,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
