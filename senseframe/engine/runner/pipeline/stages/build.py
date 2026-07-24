"""Stage 5: 构建模型 / DataModule / LightningModule。"""
from __future__ import annotations

from ..context import PipelineContext, _logger
from ..stage_spec import stage

try:
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ...preflight import build_logger
from ...callbacks import FrozenDict  # noqa: F401 (re-exported for stage_train)


@stage(
    name="build",
    reads=["config", "scene", "model_id", "dataset", "num_classes", "feature_spec",
           "bundle", "task_spec", "resolved", "output_dir", "scene_info",
           "route_config", "log_writer", "learning_mode", "pretrain_checkpoint"],
    writes=["model", "datamodule", "module", "callbacks", "pl_logger", "csv_logger", "monitor"],
    description="Stage 5: 构建模型 / DataModule / LightningModule",
)
def stage_build(ctx: PipelineContext) -> PipelineContext:
    """Stage 5: 构建模型 / DataModule / LightningModule。"""
    from ....datamodule import GenericDataModule
    from ....module import GenericLightningModule
    from ....self_supervised import SelfSupervisedModule

    is_self_supervised = (ctx.learning_mode == "self_supervised")
    # data_root 已由 SceneConfig.validate() 校验非空（YAML/CLI/env 三选一）
    data_root = ctx.config.scene.data_root

    # P1-4: stage 入口摘要日志
    _input_shape = ctx.scene_info.get("input_shape", []) if ctx.scene_info else []
    _feature_dim = getattr(ctx.feature_spec, "feature_dim", None) if ctx.feature_spec else None
    _logger.info(
        "stage_build input: model_id=%s, dataset=%s, learning_mode=%s, "
        "input_shape=%s, feature_dim=%s, num_classes=%s",
        ctx.model_id, ctx.dataset, ctx.learning_mode,
        _input_shape, _feature_dim, ctx.num_classes,
    )

    # 构建模型
    ctx.model = ctx.scene.build_model_for_dataset(
        ctx.model_id, ctx.dataset, ctx.num_classes,
        learning_mode=ctx.learning_mode,
        data_root=data_root,
        input_dim=ctx.feature_spec.feature_dim or ctx.scene_info.get("n_features"),
        feature_spec=ctx.feature_spec,
        **ctx.scene_kwargs,
    )
    # 修复（5.11）：stage_build 模型构建无日志
    # 旧逻辑：模型构造后无任何日志，参数量/输入输出形状/是否 DeviceMap 全无
    try:
        n_params = sum(p.numel() for p in ctx.model.parameters())
        n_trainable_params = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)
        # 探测输入形状（从 scene_info 或 feature_spec）
        input_shape = ctx.scene_info.get("input_shape", [])
        is_device_map = hasattr(ctx.model, "hf_device_map") or hasattr(ctx.model, "device_map")
        _logger.info(
            f"stage_build: model constructed, model_id={ctx.model_id}, "
            f"dataset={ctx.dataset}, model_class={type(ctx.model).__name__}, "
            f"total_params={n_params:,}, trainable_params={n_trainable_params:,}, "
            f"input_shape={input_shape}, is_device_map={is_device_map}"
        )
    except Exception as e:
        _logger.debug(f"stage_build: failed to log model info: {e}")

    # HIGH 1 修复：消费 ctx.pretrain_checkpoint（stage_load 产出）
    # 加载预训练权重到 ctx.model（跨模态迁移用 strict=False）
    if ctx.pretrain_checkpoint:
        from .....common.checkpoint import load_checkpoint_flexible
        try:
            ckpt_info = load_checkpoint_flexible(
                ctx.pretrain_checkpoint, ctx.model, strict=False,
            )
            _logger.info(
                "stage_build: pretrain checkpoint loaded from %s "
                "(format=%s, keys=%d)",
                ctx.pretrain_checkpoint,
                ckpt_info.get("source_format", "unknown"),
                ckpt_info.get("num_keys_loaded", 0),
            )
        except FileNotFoundError:
            _logger.warning(
                "stage_build: pretrain checkpoint not found: %s, "
                "training from scratch", ctx.pretrain_checkpoint,
            )
        except Exception as e:
            _logger.warning(
                "stage_build: failed to load pretrain checkpoint: %s, "
                "training from scratch. Error: %s", ctx.pretrain_checkpoint, e,
            )

    # metrics
    if ctx.config.scene.task_spec is not None:
        metrics = ctx.task_spec.effective_metrics
    else:
        metrics = ctx.resolved.get("metrics", ["accuracy", "macro_f1"])

    # Logger
    logger_type = ctx.resolved.get("logger", "csv")
    ctx.pl_logger = build_logger(logger_type, ctx.output_dir, ctx.model_id, ctx.dataset)
    ctx.csv_logger = ctx.pl_logger

    # Callbacks
    ctx.callbacks = []
    # P2-3 修复：monitor 可配置化（默认 val_loss，支持自定义指标）
    monitor_metric = getattr(ctx.config.trainer, "early_stopping_monitor", "val_loss")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ctx.output_dir / "checkpoints"),
        filename=f"best-{{epoch}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
        # PL 2.6.5: save_on_train_epoch_end=None 默认推断为 True，
        # 在 on_train_epoch_end 检查 val_loss（此时 validation 尚未执行）
        # 触发 "could not find the monitored key" 警告。
        # 显式设 False，只在 on_validation_epoch_end 检查。
        # 与 EarlyStopping(check_on_train_epoch_end=False) 对称修复。
        save_on_train_epoch_end=False,
    )
    ctx.callbacks.append(ckpt_cb)

    early_stopping_patience = ctx.config.trainer.early_stopping
    if early_stopping_patience is not None:
        # RFC-004 方案 E：使用 min_delta 避免微小波动误触发早停
        early_stopping_min_delta = getattr(
            ctx.config.trainer, "early_stopping_min_delta", 0.0
        )
        ctx.callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
            # pytorch_lightning 2.6.5: check_on_train_epoch_end=None 默认推断为 True，
            # 导致 on_train_epoch_end 时 val_loss 不可用而抛 RuntimeError。
            # 显式设为 False，只在 on_validation_epoch_end 检查（val_loss 在 validation 后才可用）。
            check_on_train_epoch_end=False,
        ))

    # P2: 创建 TrainingMonitor，供 EpochLogCallback 写入实时指标
    from .....observability import TrainingMonitor
    ctx.monitor = TrainingMonitor()

    from ...orchestrator import EpochLogCallback, IntermediateMetricLogger
    ctx.callbacks.append(EpochLogCallback(log_every_n=10, monitor=ctx.monitor))
    # P2.3 + P1.1: ε5 Multi-fidelity — 捕获 epoch 级中间值供 Pruner should_prune 使用
    # 回调写入 ctx.intermediate_values（dict 引用）；
    # P1.1 实时早停修复：若 ctx.pruner 注入，回调每个 epoch end 调 should_prune，
    # True 则设 trainer.should_stop=True，Lightning 提前终止训练。
    # on_pruned 回调直接写 ctx.pruned/pruned_epoch（闭包捕获 ctx），
    # stage_train 后续读取 ctx.pruned 即可感知剪枝状态。
    def _on_pruned(epoch_1indexed: int) -> None:
        ctx.pruned = True
        ctx.pruned_epoch = epoch_1indexed

    ctx.callbacks.append(IntermediateMetricLogger(
        metric="val_accuracy",
        intermediate_values=ctx.intermediate_values,
        pruner=ctx.pruner,
        trial_id=ctx.trial_id,
        on_pruned=_on_pruned,
    ))

    if ctx.config.extra_callbacks:
        ctx.callbacks.extend(ctx.config.extra_callbacks)

    # 修复（2.8）：若 ctx 含 Optuna trial 对象（通过 extra 传入），注册
    # OptunaReportingCallback 桥接 Lightning 中间指标到 trial.report()，
    # 让 Pruner 基于 epoch 级指标剪枝。Optuna 未安装时降级 warning。
    _optuna_trial = ctx.extra.get("optuna_trial") if ctx.extra else None
    if _optuna_trial is not None:
        try:
            from ...orchestrator import OptunaReportingCallback
            ctx.callbacks.append(
                OptunaReportingCallback(
                    trial=_optuna_trial,
                    metric=ctx.resolved.get("hpo_metric", "val_accuracy"),
                )
            )
        except ImportError:
            _logger.warning(
                "ctx.extra['optuna_trial'] set but OptunaReportingCallback "
                "unavailable (optuna not installed); HPO pruner will not "
                "receive intermediate values."
            )

    if is_self_supervised:
        # 自监督模式
        unsup_ds = ctx.bundle.unsupervised
        sup_ds = ctx.bundle.supervised_finetune
        val_ds = ctx.bundle.val  # P2-3 修复：传递独立 val_dataset
        test_ds = ctx.bundle.test
        # P0 修复：自监督分支漏传 scene_kwargs，导致 get_transforms 无法读取
        # params（如 transform.pipeline/augment 配置），与监督分支行为不一致。
        # 监督分支已透传 **ctx.scene_kwargs，此处对齐。
        transform_cfg = ctx.scene.get_transforms(ctx.dataset, **ctx.scene_kwargs)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=sup_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
                supervised_transform=transform_cfg.supervised_transform,
            )
        else:
            ctx.datamodule = GenericDataModule(
                train_dataset=sup_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="self_supervised",
                unsupervised_dataset=unsup_ds,
                supervised_dataset=sup_ds,
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
                supervised_transform=transform_cfg.supervised_transform,
            )

        ctx.module = SelfSupervisedModule(
            model=ctx.model,
            learning_rate=ctx.resolved["learning_rate"],
            weight_decay=ctx.resolved["weight_decay"],
            metrics=metrics,
            num_classes=ctx.num_classes,
            incremental_log_writer=ctx.log_writer,
        )
    else:
        # 监督模式
        train_ds = ctx.bundle.train
        val_ds = ctx.bundle.val  # P2-3 修复：传递独立 val_dataset
        test_ds = ctx.bundle.test
        transform_cfg = ctx.scene.get_transforms(ctx.dataset, **ctx.scene_kwargs)

        if ctx.config.datamodule_factory is not None:
            ctx.datamodule = ctx.config.datamodule_factory(
                train_dataset=train_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="supervised",
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )
        else:
            ctx.datamodule = GenericDataModule(
                train_dataset=train_ds, test_dataset=test_ds,
                val_dataset=val_ds,
                batch_size=ctx.resolved["batch_size"],
                num_workers=ctx.resolved["num_workers"],
                pin_memory=ctx.resolved.get("pin_memory", False),
                persistent_workers=ctx.resolved.get("persistent_workers", False),
                learning_mode="supervised",
                train_transform=transform_cfg.train_transform,
                eval_transform=transform_cfg.eval_transform,
            )

        epochs = ctx.config.trainer.epochs
        max_epochs = ctx.route_config.get("max_epochs", float("inf"))
        if epochs > max_epochs:
            epochs = max_epochs

        if ctx.config.module_factory is not None:
            ctx.module = ctx.config.module_factory(
                model=ctx.model,
                learning_rate=ctx.resolved["learning_rate"],
                metrics=metrics,
                num_classes=ctx.num_classes,
                optimizer=ctx.resolved["optimizer"],
                weight_decay=ctx.resolved["weight_decay"],
                scheduler=ctx.resolved["scheduler"],
                max_epochs=epochs,
                incremental_log_writer=ctx.log_writer,
                task_spec=ctx.task_spec,
            )
        else:
            ctx.module = GenericLightningModule(
                model=ctx.model,
                learning_rate=ctx.resolved["learning_rate"],
                metrics=metrics,
                num_classes=ctx.num_classes,
                optimizer=ctx.resolved["optimizer"],
                weight_decay=ctx.resolved["weight_decay"],
                scheduler=ctx.resolved["scheduler"],
                max_epochs=epochs,
                incremental_log_writer=ctx.log_writer,
                task_spec=ctx.task_spec,
            )

    # P2-4: 注入 DataProfile 到 module，供 on_train_start 一致性校验使用
    # （如 num_classes 与 data_profile.n_classes 不匹配时提前告警）
    # 自监督和监督模式统一注入；data_profile 为 None 时跳过。
    if ctx.data_profile is not None and ctx.module is not None:
        try:
            ctx.module.data_profile = ctx.data_profile
            _logger.debug("DataProfile injected into module for on_train_start validation")
        except Exception as e:
            # module 可能是 frozen dataclass 或不允许 setattr，留痕不中断
            _logger.debug(f"Failed to inject DataProfile into module: {e}")
    else:
        _logger.debug("DataProfile is None or module is None, skip injection into module")

    # P1-4: stage 出口摘要日志
    try:
        _model_class = type(ctx.model).__name__ if ctx.model is not None else "None"
        _total_params = sum(p.numel() for p in ctx.model.parameters()) if ctx.model is not None else 0
        _dm_batch_size = getattr(ctx.datamodule, "batch_size", None) if ctx.datamodule is not None else None
        _module_class = type(ctx.module).__name__ if ctx.module is not None else "None"
    except Exception:
        _model_class = "unknown"
        _total_params = 0
        _dm_batch_size = None
        _module_class = "unknown"
    _logger.info(
        "stage_build output: model_class=%s, total_params=%d, "
        "datamodule_batch_size=%s, module_class=%s, callbacks_count=%d",
        _model_class, _total_params, _dm_batch_size, _module_class,
        len(ctx.callbacks) if ctx.callbacks else 0,
    )

    return ctx
