#!/usr/bin/env python
"""探测硬件资源并推荐可用模型的封装接口。

用法：
    python probe_resources.py
    python probe_resources.py --dataset UT_HAR_data
    python probe_resources.py --dataset UT_HAR_data --priority accuracy

功能：
    1. 探测硬件资源（CPU/GPU/内存）
    2. 确定路由级别
    3. 列出该路由下可用的模型
    4. 可选：按数据集过滤并按优先级排序推荐
    5. 输出结构化 JSON
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

from senseframe.registry import MODEL_TABLE, get_default_epochs, get_dataset_spec, is_dataset_registered  # noqa: E402
from senseframe.routing import ResourceProbe, ResourceRouter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="probe_resources",
        description="探测硬件资源 + 路由 + 可用模型推荐",
    )
    parser.add_argument("--dataset", type=str,
                        help="按数据集过滤并显示默认 epochs")
    parser.add_argument("--priority", type=str,
                        choices=["accuracy", "speed", "memory", "balanced"],
                        default="balanced", help="推荐优先级")

    args = parser.parse_args()

    # 探测资源
    report = ResourceProbe.probe()
    route_level = ResourceRouter.route(report)
    route_config = ResourceRouter.get_route_config(route_level)
    available_models = ResourceRouter.filter_models(route_level)

    result = {
        "resource": report.to_dict(),
        "route_level": route_level,
        "route_config": route_config,
        "available_models": available_models,
    }

    # 按数据集过滤并推荐
    if args.dataset:
        from senseframe.registry import DATASET_INFO
        if args.dataset not in DATASET_INFO:
            result["warning"] = (
                f"Dataset '{args.dataset}' not found in registry. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        else:
            recommendations = []
            for model_id in available_models:
                info = MODEL_TABLE[model_id].copy()
                info["model_id"] = model_id
                # 方案 B：epochs 完全动态，需 n_samples
                _n_samples = get_dataset_spec(args.dataset).n_samples if is_dataset_registered(args.dataset) else None
                info["default_epochs"] = get_default_epochs(model_id, args.dataset, n_samples=_n_samples)
                recommendations.append(info)

            # 按优先级排序
            if args.priority == "accuracy":
                recommendations.sort(
                    key=lambda x: x.get("estimated_params_m", 0), reverse=True)
            elif args.priority == "speed":
                recommendations.sort(
                    key=lambda x: x.get("estimated_params_m", 0))
            elif args.priority == "memory":
                recommendations.sort(
                    key=lambda x: x.get("estimated_vram_mb", 0))

            result["priority"] = args.priority
            result["dataset"] = args.dataset
            result["recommendations"] = recommendations

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
