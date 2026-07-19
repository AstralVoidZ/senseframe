"""Stage 8: 导出 + manifest 生成。

包含：
- stage_export：导出 stage
- _compute_config_hash：config 关键字段哈希（resume 时检测 config 变更）
- _generate_manifest：从 artifact_registry 生成 manifest.json
- _PIPELINE_VERSION：pipeline checkpoint 版本号
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import yaml

from ..context import (
    PipelineContext,
    _logger,
    _finalize_lightning_logger,
    _TRAINING_LOG_ENTRY_SCHEMA,
)
from ..stage_spec import stage
from ...artifacts import ArtifactManifest, sha256_str
from ...preflight import build_env_snapshot
from ...resolver import experiment_config_to_dict, load_manifest_for_metadata
from ....metadata import make_metadata_skeleton
from .....schemas import validate_training_log_entry
from .eval import _merge_metrics_csv

if TYPE_CHECKING:
    from ....config import ExperimentConfig


# P2：pipeline checkpoint 版本号，结构变更时递增
# v2.1：_serialize_stage_outputs 新增 preflight/resolve 产出持久化（report/route_config/
#       task_spec/feature_spec/scene_info/resolved/lightning_params/distributed_kwargs）；
#       _restore_stage_outputs 从 checkpoint 恢复这些字段，根治可序列化 stage 契约矛盾
_PIPELINE_VERSION = "2.1"


def _compute_config_hash(config: "ExperimentConfig") -> str:
    """计算 config 关键字段的哈希，用于 resume 时检测 config 变更。"""
    import hashlib
    key_fields = {
        "scene": config.scene.name,
        "dataset": config.scene.dataset,
        "model_id": config.scene.model_id,
        "learning_mode": config.scene.learning_mode,
        "epochs": config.trainer.epochs,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.trainer.learning_rate,
        "optimizer": config.trainer.optimizer,
    }
    raw = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================
# RFC-004 方案 G：manifest.json 生成
# ============================================================
def _generate_manifest(ctx: PipelineContext) -> Optional[Path]:
    """从 ctx.artifact_registry 生成 manifest.json（RFC-004 方案 G）。

    在 Pipeline.run() 的 finally 块中调用，记录所有产物的路径/hash/大小/生产者/内容契约。
    成功/失败/异常路径都会生成 manifest（失败时仅含已产出的部分产物）。

    Returns:
        manifest.json 路径，失败时返回 None
    """
    import uuid
    try:
        from ..... import __version__ as sf_version
    except Exception:
        sf_version = "unknown"

    # 任务1：config_hash 覆盖声明式配置 + 路由解析后实际生效值。
    # 旧逻辑仅 hash ExperimentConfig，不含 ctx.resolved（路由后的
    # device/batch_size/precision/...），导致同名配置不同路由产生相同
    # config_hash，溯源无法区分。合并后 config_hash 唯一标识运行时配置。
    try:
        config_hash = sha256_str(
            json.dumps(
                {**experiment_config_to_dict(ctx.config), **ctx.resolved},
                sort_keys=True, default=str,
            )
        )
    except Exception:
        config_hash = ""

    # 任务2：data_hash 从 ctx.data_hash 读取（stage_load 计算的数据集元数据哈希）。
    # 旧逻辑恒为空字符串，manifest.data_hash 无溯源价值。
    data_hash = ctx.data_hash or ""

    manifest = ArtifactManifest(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        senseframe_version=sf_version,
        pipeline_version=_PIPELINE_VERSION,
        config_hash=config_hash,
        data_hash=data_hash,
        artifacts=list(ctx.artifact_registry),  # copy
    )
    return manifest.save(ctx.output_dir)


@stage(
    name="export",
    reads=["config", "model", "module", "output", "output_dir",
           "scene", "scene_info", "scene_kwargs", "meta", "report",
           "route_level", "task_spec", "feature_spec", "resolved",
           "log_writer", "exploration_history", "num_classes",
           "model_id", "dataset", "learning_mode",
           # 任务3：补报 stage_export 读取字段（函数体实际访问的 ctx.xxx）
           "bundle", "data_profile",
           "training_duration_s", "best_model_path", "best_model_score",
           # 方案 B：显存探测结果写入 metadata.resource.vram_probe
           "vram_probe_result",
           # P1.1: 实时早停状态（写入 TrainOutput.training.pruned/pruned_epoch）
           "pruned", "pruned_epoch",
           "intermediate_values"],
    writes=["output", "artifact_registry"],  # 任务3：补报 artifact_registry（register_artifact 写入）
    description="Stage 8: 导出",
)
def stage_export(ctx: PipelineContext) -> PipelineContext:
    """Stage 8: 导出。"""
    # P5 P1-N：dry_run 模式下跳过导出
    if ctx.dry_run:
        _logger.info("Skipping stage_export in dry_run mode")
        return ctx

    final_eval = ctx.final_eval
    training_log = ctx.training_log
    early_stopped = ctx.early_stopped

    # P1-1: training_log schema 校验（拦截 LR 污染等类型错误）
    # strict_schema=True 时校验失败直接抛错；False 时降级保留原始 entry
    validated_log: List[Any] = []
    for entry in training_log:
        try:
            validated_entry = validate_training_log_entry(entry)
            validated_log.append(validated_entry.to_dict())
        except (ValueError, TypeError) as e:
            _logger.error(
                f"training_log entry failed schema validation: {e}",
                exc_info=True,
            )
            if getattr(ctx.config, "strict_schema", False):
                raise
            validated_log.append(entry)  # 保留原始（可能含污染）
    # 后续产物写入（ctx.output.training["log"] / metrics.csv）使用校验后的版本
    training_log = validated_log

    # 保存模型 + metadata
    model_path = None
    if ctx.config.save_model:
        model_path = ctx.output_dir / "model.pth"
        torch.save(ctx.model.state_dict(), model_path)
        model_path = str(model_path)

        normalization_info = ctx.scene.get_normalization_info(ctx.dataset, **ctx.scene_kwargs)
        label_map = {}
        manifest_info = None
        if ctx.meta.is_dynamic_dataset:
            manifest_info = ctx.scene.get_manifest_info(ctx.dataset, **ctx.scene_kwargs)
            if manifest_info is not None:
                try:
                    manifest = load_manifest_for_metadata(ctx.config.scene.params)
                    label_map = manifest.label_map
                except Exception:
                    pass

        # P3 演进（2026-07-18）：metadata.json schema_version 版本管理。
        # 写入端通过 make_metadata_skeleton() 统一注入 schema_version（当前版本），
        # 读取端通过 load_metadata() 协商迁移。pipeline 不直接引用版本常量，
        # 版本管理职责完全内聚到 metadata 模块。
        # 遗留问题 3 修复（2026-07-19）：从 dict 字面量改为 make_metadata_skeleton(**kwargs)，
        # 消除 make_metadata_skeleton 死代码状态，schema_version 注入由骨架函数统一负责。
        metadata = make_metadata_skeleton(
            model_id=ctx.model_id,
            dataset=ctx.dataset,
            learning_mode=ctx.learning_mode,
            num_classes=ctx.num_classes,
            input_shape=list(ctx.scene_info.get("input_shape", [])),
            normalization=normalization_info,
            label_map={str(k): v for k, v in label_map.items()},
            manifest=manifest_info,
            # metadata.config 是完整配置快照，供实验复现与下游消费者（generate_inference 等）使用。
            # 根因修复：ctx.resolved 仅含路由运行时字段（device/batch_size/precision/...），
            # 缺失 14 个训练级字段（epochs/seed/deterministic/max_time/...）和场景级字段（data_root/...）。
            # 方案 D：合并 experiment_config_to_dict(ctx.config)（声明式配置完整快照）
            # 与 ctx.resolved（路由解析后实际生效值），重叠字段以 ctx.resolved 为准。
            # 这样复现所需字段（epochs/seed/data_root/learning_mode/...）全部进入 metadata.config，
            # 且未来 ExperimentConfig 新增字段自动进入，无需逐字段补录。
            config={
                **experiment_config_to_dict(ctx.config),
                **ctx.resolved,
            },
            metrics=list(final_eval.keys()),
            final_eval=final_eval,
            # 对称性修复：显式提取 test_eval 字段，便于下游消费者直接访问 test 指标
            test_eval={
                k: v for k, v in final_eval.items()
                if k.startswith("test_")
            } if any(k.startswith("test_") for k in final_eval) else None,
            env=build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed}),
            # 方案 B：动态显存探测结果（stage_probe_vram 写入）
            # None/跳过时记 skipped 原因；探测成功时含 measured_vram_mb/needed_vram_mb/free_vram_mb/ok
            resource={
                **ctx.report.to_dict(),
                "vram_probe": ctx.vram_probe_result,
            },
            route_level=ctx.route_level,
            task_spec=ctx.task_spec.to_dict(),
            feature_spec=ctx.feature_spec.to_dict(),
            # Part 2：best checkpoint 溯源 + epoch 利用率（风险推演 R1/R4）
            # best_epoch/best_model_path/best_model_score 从 ctx 读取（stage_train 写入）
            # epoch_utilization = best_epoch / epochs，供 Agent 判断预算是否合理
            # （<0.3 预算过大，>0.9 预算不足）
            best_epoch=ctx.best_epoch,
            best_model_path=ctx.best_model_path,
            best_model_score=ctx.best_model_score,
            epoch_utilization=round(ctx.best_epoch / ctx.config.trainer.epochs, 3) if ctx.best_epoch and ctx.config.trainer.epochs else None,
            created_at=datetime.now().isoformat(),
        )
        # P5 P2-6：strict_schema=True 时对 metadata 关键字段做类型校验
        # 旧代码 strict_schema 仅控制 training_log，metadata 无类型校验，
        # 允许 num_classes=str / best_epoch=float 等类型污染传播到下游
        if getattr(ctx.config, "strict_schema", False):
            _type_checks = [
                ("schema_version", metadata["schema_version"], str),
                ("model_id", ctx.model_id, str),
                ("dataset", ctx.dataset, str),
                ("num_classes", ctx.num_classes, int),
                ("best_epoch", ctx.best_epoch, (type(None), int)),
                ("best_model_score", ctx.best_model_score, (type(None), float, int)),
                ("created_at", metadata["created_at"], str),
            ]
            for field_name, value, expected in _type_checks:
                if not isinstance(value, expected):
                    raise TypeError(
                        f"metadata.{field_name} type error: expected {expected}, "
                        f"got {type(value).__name__}={value!r}"
                    )
        (ctx.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (ctx.output_dir / "config.yaml").write_text(
            yaml.dump(ctx.resolved, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    # 构建 TrainOutput
    if ctx.output:
        ctx.output.status = "success"
        ctx.output.error_code = "SUCCESS"
        # P5 P2-7 阶段2：构造 TrainingSummary dataclass 实例（不再还原为 dict）。
        # 下游消费方已迁移为属性访问 + to_dict() 序列化兼容。
        # TrainOutput.to_dict() 已有多态序列化 helper，会自动调用 .to_dict()。
        from .....schemas import validate_training_summary, validate_env_snapshot
        ctx.output.training = validate_training_summary({
            "epochs_trained": len(training_log),
            "early_stopped": early_stopped,
            "log": training_log,
            "duration_s": ctx.training_duration_s,
            "best_val_loss": ctx.best_model_score,
            "best_checkpoint": ctx.best_model_path,
            "intermediate_values": ctx.intermediate_values,  # P2.3: ε5 Multi-fidelity
            # P1.1: 实时早停状态投影（ctx.pruned/pruned_epoch → TrainOutput.training）
            # MethodRunner 读取 training.pruned 区分实时剪枝 trial 与正常完成 trial
            "pruned": ctx.pruned,
            "pruned_epoch": ctx.pruned_epoch,
        })
        ctx.output.final_eval = final_eval
        ctx.output.model_path = model_path
        env_snapshot_dict = build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed})
        ctx.output.env_snapshot = validate_env_snapshot(env_snapshot_dict)

    # 可选多格式导出
    export_formats = getattr(ctx.config, "export_formats", None)
    if export_formats and model_path:
        try:
            from .....export import export_model
            export_dir = ctx.output_dir / "exports"
            export_result = export_model(
                model=ctx.model,
                output_dir=export_dir,
                formats=export_formats,
                input_shape=list(ctx.scene_info.get("input_shape", [])),
                metadata={
                    "model_id": ctx.model_id,
                    "dataset": ctx.dataset,
                    "learning_mode": ctx.learning_mode,
                    "num_classes": ctx.num_classes,
                    "final_eval": final_eval,
                },
            )
            if ctx.output:
                ctx.output.export = export_result.to_dict()
            # P4-3：导出有 errors（如 onnx 包缺失）时记录 warning，不再静默。
            if export_result.errors:
                _logger.warning(
                    "Export completed with errors: %s", export_result.errors
                )
        except Exception as e:
            # P4-3：导出异常不再静默吞没，记录 warning 供排查。
            _logger.warning("Export failed: %s", e)
            if ctx.output:
                ctx.output.export = {"error": str(e)}

    # 关闭日志写入器（release_resources 也会关闭，此处保留双保险）
    if ctx.log_writer is not None:
        try:
            ctx.log_writer.close()
        except Exception:
            pass

    # 清理显存
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

    # RFC-002 阶段 L + P1.5：持久化探索历史 + 结构化反馈 + 自动推荐，闭合探索-反馈回路
    feedback = ctx.feedback
    if feedback:
        # 对称性修复：在 feedback 中附加 test 指标摘要
        # P2-3 修复后 val/test 分离，feedback 应同时包含 val 和 test 指标
        if ctx.final_eval:
            _test_metrics_summary = {
                k: v for k, v in ctx.final_eval.items()
                if k.startswith("test_") and not k.startswith("test_confusion")
            }
            if _test_metrics_summary:
                # P5 P2-7 阶段2：feedback 现在是 FeedbackResult dataclass，
                # test_metrics 是可选字段，直接属性赋值
                feedback.test_metrics = _test_metrics_summary
        # P5 P2-7 阶段2：feedback.json 写入时调用 to_dict() 序列化
        (ctx.output_dir / "feedback.json").write_text(
            json.dumps(feedback.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if ctx.exploration_history:
        from .....exploration import ExplorationTracker
        tracker = ExplorationTracker(ctx.exploration_history)
        tracker.save(ctx.output_dir / "exploration.json")

        # P1.5：自动推荐下一步策略（闭合探索-反馈回路）
        task_type = ctx.task_spec.task_type if ctx.task_spec else None
        try:
            recommendations = tracker.recommend_next(task_type=task_type, top_k=5)
            if recommendations:
                (ctx.output_dir / "recommendations.json").write_text(
                    json.dumps(recommendations, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception as e:
            _logger.warning(f"recommend_next failed: {e}")

        # P1.8：success status 自动沉淀技能（闭合 Voyager 检索复用回路）
        # P5 P2-7 阶段2：feedback 是 FeedbackResult dataclass，用属性访问
        if feedback and feedback.status == "success":
            try:
                from .....skills import save_skill as _save_skill
                skill_name = f"{ctx.model_id}_{ctx.dataset}"
                _save_skill(
                    name=skill_name,
                    code=(
                        f"# Auto-saved from trial {ctx.trial_id}\n"
                        f"# model={ctx.model_id}, dataset={ctx.dataset}, "
                        f"learning_mode={ctx.learning_mode}\n"
                    ),
                    description=f"Auto-saved: {ctx.model_id} on {ctx.dataset}",
                    tags=[ctx.model_id, ctx.dataset, "auto"],
                )
                _logger.info(f"Auto-saved skill: {skill_name}")
            except Exception as e:
                _logger.warning(f"Auto save_skill failed: {e}")

    # ============================================================
    # RFC-004 方案 G：产物溯源注册 + 缺失产物补齐
    # ============================================================
    # 补齐 env_snapshot.json（独立文件，不再仅嵌在 metadata.json）
    try:
        env_snap = build_env_snapshot(ctx.resolved, {"seed": ctx.config.trainer.seed})
        env_snap_path = ctx.output_dir / "env_snapshot.json"
        env_snap_path.write_text(
            json.dumps(env_snap, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        ctx.register_artifact(
            "env_snapshot", env_snap_path,
            kind="log", producer_stage="stage_export",
            content_schema={"python": str, "torch": str, "cuda": bool, "device": str},
        )
    except Exception as e:
        _logger.warning(f"Failed to save env_snapshot.json: {e}")

    # 修复（任务4 / P1）：metrics.csv 双重写入修复。
    # 旧逻辑：此处手动写顶层 metrics.csv，与 CSVLogger（build_logger 中
    # CSVLogger(save_dir=output_dir, name="metrics", version="")）写入的
    # metrics/metrics.csv 内容重复，导致运行目录同时存在两份相同 metrics.csv。
    # 方案：删除手动顶层写入，仅注册 CSVLogger 产出的 metrics/metrics.csv 为产物。
    # CSVLogger 在训练过程中按 epoch 增量写入，内容与 training_log 一致。
    if training_log:
        # 修复（任务3 / P2）：CSVLogger 的 finalize 在 Pipeline.run finally 块中调用，
        # 晚于 stage_export，导致 metrics.csv 未 flush 时注册 artifact 失败
        # （warning "CSVLogger metrics.csv not found"）。
        # 方案：在注册 metrics artifact 前先 finalize csv_logger，确保文件已落盘。
        if ctx.csv_logger is not None:
            _finalize_lightning_logger(ctx.csv_logger)

        csv_logger_metrics_path = ctx.output_dir / "metrics" / "metrics.csv"
        if csv_logger_metrics_path.exists():
            # 修复（任务4 / P2）：合并 metrics.csv 的 train+val 分行为 1 行/epoch，
            # 与 training_log.jsonl 格式对齐。Lightning CSVLogger 在
            # on_train_epoch_end 和 on_validation_epoch_end 分别写入一行，
            # 导致每 epoch 有 2 行（train 行 + val 行），合并后每 epoch 1 行。
            try:
                _merge_metrics_csv(csv_logger_metrics_path)
            except Exception as e:
                _logger.warning("Failed to merge metrics.csv rows: %s", e)
            ctx.register_artifact(
                "metrics", csv_logger_metrics_path,
                kind="metrics", producer_stage="stage_build",
                content_schema=_TRAINING_LOG_ENTRY_SCHEMA,
            )
        else:
            _logger.warning(
                "CSVLogger metrics.csv not found at %s; metrics artifact not registered",
                csv_logger_metrics_path,
            )

    # 注册核心产物（model/metadata/config/training_log/feedback/exploration）
    if model_path is not None:
        ctx.register_artifact(
            "model_weights", Path(model_path),
            kind="model", producer_stage="stage_export",
            content_schema={"format": "state_dict", "num_classes": int},
        )
    metadata_path = ctx.output_dir / "metadata.json"
    if metadata_path.exists():
        ctx.register_artifact(
            "model_metadata", metadata_path,
            kind="metadata", producer_stage="stage_export",
            # 对称性修复：final_eval 现在含 val_* 和 test_* 指标
            content_schema={"model_id": str, "dataset": str, "final_eval": dict,
                            "test_eval": dict},
        )
    config_yaml_path = ctx.output_dir / "config.yaml"
    if config_yaml_path.exists():
        ctx.register_artifact(
            "config", config_yaml_path,
            kind="config", producer_stage="stage_export",
            content_schema={"scene": str, "dataset": str, "model_id": str, "trainer": dict},
        )
    training_log_path = ctx.output_dir / "training_log.jsonl"
    if training_log_path.exists():
        ctx.register_artifact(
            "training_log", training_log_path,
            kind="log", producer_stage="stage_train",
            content_schema=_TRAINING_LOG_ENTRY_SCHEMA,
        )
    feedback_path = ctx.output_dir / "feedback.json"
    if feedback_path.exists():
        ctx.register_artifact(
            "feedback", feedback_path,
            kind="feedback", producer_stage="stage_eval",
            # 对称性修复：content_schema 新增 test_metrics 字段
            content_schema={"status": str, "diagnosis": str, "suggestions": list,
                            "test_metrics": dict},
        )
    exploration_path = ctx.output_dir / "exploration.json"
    if exploration_path.exists():
        ctx.register_artifact(
            "exploration", exploration_path,
            kind="log", producer_stage="stage_eval",
            content_schema={"trial_id": str, "strategy": dict, "result": dict},
        )
    recommendations_path = ctx.output_dir / "recommendations.json"
    if recommendations_path.exists():
        ctx.register_artifact(
            "recommendations", recommendations_path,
            kind="log", producer_stage="stage_eval",
            content_schema={"strategy": dict, "priority": str},
        )

    return ctx
