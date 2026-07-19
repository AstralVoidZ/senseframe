#!/usr/bin/env python3
"""P3 验证评估 10.5：搜索空间有效性验证（C1-C5）。

验证 SP 协议驱动的 PEFT 搜索优于固定 PEFT 配置。

实验组（5 组 × 3 数据集 = 15 次实验）：
- C1（baseline）：固定 LoRA (rank=8, alpha=16)
- C2（baseline）：固定 Adapter (bottleneck=128)
- C3（baseline）：固定 Prompt Tuning (length=10)
- C4：RandomSampler × n_trials=20 SP 搜索
- C5：GridSampler × 全网格 SP 搜索

评估数据集：
- CSI: NTU-Fi_HAR（中规模）
- 雷达: RadioML 2018.01A 子集（10 万样本）
- EEG: PhysioNet eegmmidb 子集（10 受试者）

通过标准：
- C4 best_val_accuracy > max(C1, C2, C3) + 1%
- C4 n_completed / n_trials ≥ 90%
- C5 best_val_accuracy ≥ C4（Grid 是上界）

用法：
    python p3_eval_10_5_sp_search.py --dry-run
    python p3_eval_10_5_sp_search.py --datasets NTU-Fi_HAR --experiments C1 C4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from p3_eval_common import (
    ExperimentConfig,
    ExperimentResult,
    SP_SEARCH_EXPERIMENTS,
    SP_SEARCH_DATASETS,
    add_common_args,
    aggregate_results,
    compute_search_effectiveness,
    run_single_experiment,
    setup_logging,
)


def build_experiment_configs(
    experiments: list[str],
    datasets: list[str],
    output_dir: str,
    seed: int,
) -> list[ExperimentConfig]:
    """构造 C1-C5 × datasets 的实验配置列表。"""
    configs = []
    for exp_def in SP_SEARCH_EXPERIMENTS:
        if exp_def["id"] not in experiments:
            continue
        for ds in datasets:
            if ds not in SP_SEARCH_DATASETS:
                continue
            sp_search = None
            finetune_method = "lora"  # 默认
            finetune_params = {}
            if exp_def["method"] == "fixed":
                finetune_method = exp_def["config"]["peft_method"]
                finetune_params = {k: v for k, v in exp_def["config"].items()
                                   if k != "peft_method"}
            else:  # sp_search
                sp_search = exp_def["config"]
                # C4/C5 用 SP 搜索，finetune_method 由搜索决定（占位 "sp_search"）
                finetune_method = "sp_search"

            configs.append(ExperimentConfig(
                experiment_id=f"{exp_def['id']}_{ds}",
                experiment_group="sp_search",
                pretrain_source="csi_4datasets",  # C 组全部预训练
                finetune_method=finetune_method,
                target_dataset=ds,
                finetune_params=finetune_params,
                sp_search=sp_search,
                output_dir=output_dir,
                seed=seed,
            ))
    return configs


def main():
    parser = argparse.ArgumentParser(description="P3 验证 10.5：SP 搜索有效性 C1-C5")
    parser.add_argument("--datasets", nargs="+",
                        default=SP_SEARCH_DATASETS,
                        help=f"评估数据集（默认 {SP_SEARCH_DATASETS}）")
    parser.add_argument("--experiments", nargs="+",
                        default=["C1", "C2", "C3", "C4", "C5"],
                        help="实验组（默认全部 5 个）")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)

    configs = build_experiment_configs(
        experiments=args.experiments,
        datasets=args.datasets,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"=== 10.5 SP 搜索有效性验证 ===")
    print(f"实验组: {args.experiments}")
    print(f"数据集: {args.datasets}")
    print(f"总实验数: {len(configs)}")
    print()

    results: list[ExperimentResult] = []
    for cfg in configs:
        print(f"[{cfg.experiment_id}] method={cfg.finetune_method}, "
              f"dataset={cfg.target_dataset}, "
              f"sp_search={cfg.sp_search}")
        try:
            result = run_single_experiment(cfg, dry_run=args.dry_run)
            results.append(result)
        except NotImplementedError as e:
            print(f"  [SKIP] {e}")
            results.append(ExperimentResult(
                experiment_id=cfg.experiment_id,
                status="not_implemented",
                error=str(e),
            ))
        except Exception as e:
            print(f"  [FAIL] {e}")
            results.append(ExperimentResult(
                experiment_id=cfg.experiment_id,
                status="failed",
                error=str(e),
            ))

    # 汇总
    output_path = Path(args.output_dir) / "10_5_sp_search_results.json"
    aggregated = aggregate_results(results, output_path)
    print(f"\n=== 汇总 ===")
    print(f"结果数: {len(aggregated)}")
    print(f"输出: {output_path}")

    # 搜索有效性计算
    if not args.dry_run and len(results) >= 4:
        print(f"\n=== SP 搜索有效性 ===")
        # 按数据集分组
        for ds in args.datasets:
            fixed = [r for r in results
                     if r.experiment_id.startswith(("C1_", "C2_", "C3_"))
                     and r.experiment_id.endswith(f"_{ds}")]
            sp_random = next((r for r in results
                              if r.experiment_id == f"C4_{ds}"), None)
            sp_grid = next((r for r in results
                            if r.experiment_id == f"C5_{ds}"), None)
            if fixed and sp_random:
                improvement = compute_search_effectiveness(fixed, sp_random)
                print(f"  [{ds}] C4 vs max(C1,C2,C3): improvement = {improvement:+.2f}%")
            if sp_random and sp_grid:
                grid_vs_random = sp_grid.val_accuracy - sp_random.val_accuracy
                print(f"  [{ds}] C5 vs C4: grid_upper_bound_diff = {grid_vs_random:+.2f}%")


if __name__ == "__main__":
    main()
