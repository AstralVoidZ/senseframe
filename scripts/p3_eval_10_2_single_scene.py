#!/usr/bin/env python3
"""P3 验证评估 10.2：单场景性能验证（A1-A5）。

在 SenseFrame 已集成的 4 个 CSI 数据集上验证：
    CSIFoundationModel + PEFT 微调 > 从头训练

实验组（每组 × 4 数据集 = 20 次实验）：
- A1（baseline）：无预训练 + 从头训练
- A2：MAE on CSI 4 数据集聚合 + 全量微调
- A3：MAE on CSI 4 数据集聚合 + LoRA (rank=8, alpha=16)
- A4：MAE on CSI 4 数据集聚合 + Adapter (bottleneck=128)
- A5：MAE on CSI 4 数据集聚合 + Prompt Tuning (length=10)

通过标准：
- A3/A4/A5 至少一组在 4 数据集上 val_accuracy > A1 + 2%
- A3/A4/A5 trainable_params < A1 × 30%
- A2 ≥ A1（预训练有效）

用法：
    python p3_eval_10_2_single_scene.py --dry-run
    python p3_eval_10_2_single_scene.py --datasets UT_HAR NTU-Fi_HAR
    python p3_eval_10_2_single_scene.py --experiments A1 A3 --datasets UT_HAR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from p3_eval_common import (
    ExperimentConfig,
    ExperimentResult,
    SINGLE_SCENE_EXPERIMENTS,
    add_common_args,
    aggregate_results,
    apply_arg_overrides,
    run_single_experiment,
    setup_logging,
)


def build_experiment_configs(
    experiments: list[str],
    datasets: list[str],
    output_dir: str,
    seed: int,
    args=None,
) -> list[ExperimentConfig]:
    """构造 A1-A5 × datasets 的实验配置列表。

    若传入 args（argparse Namespace），用 apply_arg_overrides 把
    --epochs/--batch-size 等覆盖到每个 ExperimentConfig。
    """
    configs = []
    for exp_def in SINGLE_SCENE_EXPERIMENTS:
        if exp_def["id"] not in experiments:
            continue
        for ds in datasets:
            if ds not in exp_def["datasets"]:
                continue
            cfg = ExperimentConfig(
                experiment_id=f"{exp_def['id']}_{ds}",
                experiment_group="single_scene",
                pretrain_source=exp_def["pretrain"],
                finetune_method=exp_def["finetune"],
                target_dataset=ds,
                finetune_params=exp_def.get("params", {}),
                output_dir=output_dir,
                seed=seed,
            )
            if args is not None:
                cfg = apply_arg_overrides(args, cfg)
            configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="P3 验证 10.2：单场景性能验证 A1-A5")
    parser.add_argument("--datasets", nargs="+",
                        default=["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"],
                        help="评估数据集（默认全部 4 个）")
    parser.add_argument("--experiments", nargs="+",
                        default=["A1", "A2", "A3", "A4", "A5"],
                        help="实验组（默认全部 5 个）")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)

    configs = build_experiment_configs(
        experiments=args.experiments,
        datasets=args.datasets,
        output_dir=args.output_dir,
        seed=args.seed,
        args=args,
    )

    print(f"=== 10.2 单场景性能验证 ===")
    print(f"实验组: {args.experiments}")
    print(f"数据集: {args.datasets}")
    print(f"总实验数: {len(configs)}")
    print(f"输出目录: {args.output_dir}")
    print(f"dry_run: {args.dry_run}")
    # 显示生效的训练超参（CLI 覆盖优先，否则 ExperimentConfig 默认值）
    sample_cfg = configs[0] if configs else None
    if sample_cfg is not None:
        print(f"epochs: {sample_cfg.epochs}, batch_size: {sample_cfg.batch_size}, "
              f"pretrain_epochs: {sample_cfg.pretrain_epochs}, lr: {sample_cfg.learning_rate}")
    print()

    results: list[ExperimentResult] = []
    for cfg in configs:
        print(f"[{cfg.experiment_id}] pretrain={cfg.pretrain_source}, "
              f"finetune={cfg.finetune_method}, dataset={cfg.target_dataset}")
        try:
            result = run_single_experiment(cfg, dry_run=args.dry_run)
            results.append(result)
        except NotImplementedError as e:
            print(f"  [SKIP] {e}")
            results.append(ExperimentResult(
                experiment_id=cfg.experiment_id,
                status="not_implemented",
                error=str(e),
                config_snapshot={"pretrain": cfg.pretrain_source,
                                 "finetune": cfg.finetune_method,
                                 "dataset": cfg.target_dataset},
            ))
        except Exception as e:
            print(f"  [FAIL] {e}")
            results.append(ExperimentResult(
                experiment_id=cfg.experiment_id,
                status="failed",
                error=str(e),
            ))

    # 汇总（dry-run 不覆盖真实结果文件）
    output_path = Path(args.output_dir) / "10_2_single_scene_results.json"
    if args.dry_run:
        print(f"\n=== 汇总（dry-run，跳过写入 {output_path}）===")
        print(f"结果数: {len(results)}")
        aggregated = [r.to_dict() for r in results]
    else:
        aggregated = aggregate_results(results, output_path)
        print(f"\n=== 汇总 ===")
        print(f"结果数: {len(aggregated)}")
        print(f"输出: {output_path}")

    # 通过标准检查（仅 dry_run=False 时有意义）
    if not args.dry_run:
        print(f"\n=== 通过标准检查 ===")
        # TODO: 实现 A3/A4/A5 vs A1 的统计显著性检验
        print("（待真实训练结果后启用）")


if __name__ == "__main__":
    main()
