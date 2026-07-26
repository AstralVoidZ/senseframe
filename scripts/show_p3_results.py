"""显示 P3 验证结果汇总表（临时工具脚本）。"""
import json
import sys
from pathlib import Path


def main(results_path: str) -> None:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    print(f"Total: {len(results)} experiments")
    print()
    header = f"{'exp_id':<25} {'status':<12} {'val_acc':<10} {'macro_f1':<10} {'trainable':<10} {'time(s)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['experiment_id']:<25} {r['status']:<12} "
            f"{r['val_accuracy']:<10.4f} {r['macro_f1']:<10.4f} "
            f"{r['trainable_params']:<10} {r['training_time_seconds']:<10.1f}"
        )
        if r.get("error"):
            print(f"  ERROR: {r['error'][:150]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/p3_validation/10_2_single_scene_results.json")
