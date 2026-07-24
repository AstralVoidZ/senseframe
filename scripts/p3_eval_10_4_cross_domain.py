#!/usr/bin/env python3
"""P3 验证评估 10.4：跨场景迁移评估（B1-B8）。

验证 CSIFoundationModel 在雷达/EEG 数据集上的跨场景迁移有效性。

实验组（8 次）：
- B1（baseline）：无预训练 + RadioML 2018 从头训练
- B2：CSI 4 数据集 MAE + RadioML LoRA
- B3：CSI 4 数据集 MAE + RadioML 全量微调
- B4（baseline）：无预训练 + PhysioNet eegmmidb 从头训练
- B5：CSI 4 数据集 MAE + PhysioNet eegmmidb LoRA
- B6：CSI 4 数据集 MAE + PhysioNet eegmmidb 全量微调
- B7（对照）：RadioML MAE + PhysioNet eegmmidb LoRA
- B8（对照）：PhysioNet eegmmidb MAE + RadioML LoRA

通过标准：
- B2 val_accuracy > B1 + 3%（CSI→雷达正向迁移）
- B5 val_accuracy > B4 + 3%（CSI→EEG 正向迁移）
- transfer_gain > 0 在至少 6/8 个实验组中成立

用法：
    python p3_eval_10_4_cross_domain.py --dry-run
    python p3_eval_10_4_cross_domain.py --experiments B1 B2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from p3_eval_common import (
    ExperimentConfig,
    ExperimentResult,
    CROSS_DOMAIN_EXPERIMENTS,
    add_common_args,
    aggregate_results,
    apply_arg_overrides,
    compute_transfer_gain,
    run_single_experiment,
    setup_logging,
)


def build_experiment_configs(
    experiments: list[str],
    output_dir: str,
    seed: int,
    args=None,
) -> list[ExperimentConfig]:
    """构造 B1-B8 实验配置列表。

    若传入 args（argparse Namespace），用 apply_arg_overrides 把
    --epochs/--batch-size 等覆盖到每个 ExperimentConfig。
    """
    configs = []
    for exp_def in CROSS_DOMAIN_EXPERIMENTS:
        if exp_def["id"] not in experiments:
            continue
        # 构造 kwargs，仅传递 exp_def 中存在的字段（B5_DANN/B6_DANN 带 DANN 字段）
        cfg_kwargs = dict(
            experiment_id=exp_def["id"],
            experiment_group="cross_domain",
            pretrain_source=exp_def["pretrain"],
            finetune_method=exp_def["finetune"],
            target_dataset=exp_def["target"],
            output_dir=output_dir,
            seed=seed,
        )
        # 可选字段（B5_DANN/B6_DANN 用）
        if "params" in exp_def:
            cfg_kwargs["finetune_params"] = exp_def["params"]
        if "use_dann" in exp_def:
            cfg_kwargs["use_dann"] = exp_def["use_dann"]
        if "pretrain_checkpoint" in exp_def:
            cfg_kwargs["pretrain_checkpoint"] = exp_def["pretrain_checkpoint"]
        if "pretrain_epochs" in exp_def:
            cfg_kwargs["pretrain_epochs"] = exp_def["pretrain_epochs"]
        if "epochs" in exp_def:
            cfg_kwargs["epochs"] = exp_def["epochs"]
        if "batch_size" in exp_def:
            cfg_kwargs["batch_size"] = exp_def["batch_size"]
        cfg = ExperimentConfig(**cfg_kwargs)
        if args is not None:
            cfg = apply_arg_overrides(args, cfg)
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(description="P3 验证 10.4：跨场景迁移评估 B1-B8")
    parser.add_argument("--experiments", nargs="+",
                        default=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
                        help="实验组（默认 B1-B8；DANN 组需显式指定，如 --experiments B5_DANN B6_DANN）")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)

    configs = build_experiment_configs(
        experiments=args.experiments,
        output_dir=args.output_dir,
        seed=args.seed,
        args=args,
    )

    print(f"=== 10.4 跨场景迁移评估 ===")
    print(f"实验组: {args.experiments}")
    print(f"总实验数: {len(configs)}")
    print()

    results: list[ExperimentResult] = []
    for cfg in configs:
        print(f"[{cfg.experiment_id}] pretrain={cfg.pretrain_source}, "
              f"target={cfg.target_dataset}, finetune={cfg.finetune_method}")
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

    # 汇总（dry-run 不覆盖真实结果文件）
    output_path = Path(args.output_dir) / "10_4_cross_domain_results.json"
    if args.dry_run:
        print(f"\n=== 汇总（dry-run，跳过写入 {output_path}）===")
        print(f"结果数: {len(results)}")
        aggregated = [r.to_dict() for r in results]
    else:
        aggregated = aggregate_results(results, output_path)
        print(f"\n=== 汇总 ===")
        print(f"结果数: {len(aggregated)}")
        print(f"输出: {output_path}")

    # 迁移增益计算
    if not args.dry_run and len(results) >= 2:
        print(f"\n=== 迁移增益 ===")
        result_map = {r.experiment_id: r for r in results}
        # B2/B3 vs B1（CSI→雷达），B5/B6/B5_DANN/B6_DANN vs B4（CSI→EEG）
        pairs = [
            ("B1", "B2"), ("B1", "B3"),
            ("B4", "B5"), ("B4", "B6"),
            ("B4", "B5_DANN"), ("B4", "B6_DANN"),
            # DANN vs 非 DANN 对照（同 pretrain/finetune，仅多了对抗对齐）
            ("B5", "B5_DANN"), ("B6", "B6_DANN"),
        ]
        for baseline_id, transfer_id in pairs:
            if baseline_id in result_map and transfer_id in result_map:
                gain = compute_transfer_gain(
                    result_map[baseline_id], result_map[transfer_id]
                )
                print(f"  {transfer_id} vs {baseline_id}: transfer_gain = {gain:+.2f}%")


if __name__ == "__main__":
    main()
