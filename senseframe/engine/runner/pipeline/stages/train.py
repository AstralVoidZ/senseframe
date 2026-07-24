"""Stage 6: 训练执行 + 训练结果分析。

包含：
- _is_oom_error / _fit_with_oom_fallback：OOM 回退辅助
- stage_train：训练执行 stage
- analyze_training_result：训练结果分析（闭合探索-反馈回路）
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint

from ..context import PipelineContext, _logger
from ..stage_spec import stage
from .....common import load_checkpoint_flexible
from .....observability import Timer
from ...callbacks import FrozenDict


# ============================================================
# P3: OOM 回退辅助
# ============================================================
def _is_oom_error(exc: Exception) -> bool:
    """判断异常是否为 CUDA/内存 OOM。"""
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", type(None))):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            return True
    return False


def _fit_with_oom_fallback(
    ctx: PipelineContext,
    build_trainer: Callable[[], "pl.Trainer"],
    fit_fn: Callable[["pl.Trainer"], None],
    *,
    min_batch_size: int = 4,
) -> "pl.Trainer":
    """执行 trainer.fit() 并在 OOM 时自动减半 batch_size 重试一次。

    P3: 闭环 OOM 恢复——Agent 选的 batch_size 可能超出显存，
    框架自动降级而非直接失败，减少 Agent 重试往返。

    Args:
        ctx: PipelineContext（读取/写入 resolved["batch_size"] 和 datamodule.batch_size）
        build_trainer: 无参 callable，返回新的 Trainer 实例（重试时重新调用）
        fit_fn: 接收 trainer 的 callable，内部调用 trainer.fit(...)；
                重试时重新调用，应每次重新获取 dataloader 以反映新 batch_size
        min_batch_size: 最小 batch_size，低于此值不再重试

    Returns:
        成功完成 fit 的 Trainer 实例
    """
    trainer = build_trainer()
    try:
        fit_fn(trainer)
        return trainer
    except Exception as e:
        if not _is_oom_error(e):
            # 修复（2.10）：非 OOM 异常时 teardown trainer，避免资源泄露
            # （Trainer 内部持有 CUDA/dataloader worker 等资源，不 teardown 会泄露）
            if hasattr(trainer, "_teardown"):
                try:
                    trainer._teardown()
                except Exception:
                    pass
            raise
        current_bs = ctx.resolved.get("batch_size", 64)
        if current_bs <= min_batch_size:
            _logger.warning(
                f"OOM at batch_size={current_bs} (<= min {min_batch_size}), not retrying"
            )
            raise
        new_bs = max(min_batch_size, current_bs // 2)
        _logger.warning(
            f"OOM at batch_size={current_bs}, retrying with batch_size={new_bs}"
        )
        ctx.resolved["batch_size"] = new_bs
        # 更新 datamodule batch_size（Lightning DataModule 在 fit 时重新调用 dataloader 方法）
        if hasattr(ctx.datamodule, "batch_size"):
            ctx.datamodule.batch_size = new_bs
        # RFC-005：清理旧 Trainer（_teardown + del + CUDA 同步），避免残留 worker/显存泄露
        # 注意：用 _teardown()（Lightning 私有），teardown() 不存在会静默失败
        if hasattr(trainer, "_teardown"):
            try:
                trainer._teardown()
            except Exception:
                pass
        del trainer
        # 修复（2.9）：del 后需 gc.collect() 打断 Trainer/LightningModule 内部循环引用，
        # 否则 empty_cache() 时引用计数未归零，显存未真正释放。
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        # 重建 trainer（旧实例已 teardown）
        trainer = build_trainer()
        fit_fn(trainer)
        return trainer


# ============================================================
# v2 差距 2+3：DANN 训练分支（沉淀自 scripts/p3_eval_common.py:858 _train_dann）
# ============================================================
def _should_use_dann(scene_params) -> bool:
    """判断是否启用 DANN 训练分支。

    Args:
        scene_params: SceneParams 实例或 None

    Returns:
        True 当 scene.params.use_dann=True 时；False 否则
    """
    if scene_params is None:
        return False
    return bool(scene_params.get("use_dann", False))


def _train_dann_loop(
    ctx: "PipelineContext",
    epochs: int,
    learning_rate: float,
) -> None:
    """DANN 训练循环：任务分类 + 模态对抗对齐。

    沉淀自 scripts/p3_eval_common.py:858 _train_dann，适配主 Pipeline context。

    设计：双 loss 训练（task_loss + disc_loss），λ 调度按 Ganin & Lempitsky 2015。
    与 Lightning Trainer 单 loss fit 不兼容，故走独立循环。

    MEDIUM 5 修复（2026-07-22）：从 ctx.resolved 读取 optimizer/scheduler/
    gradient_clip_val/early_stopping，与 Lightning 路径对齐，让 HPO 搜索生效。

    Args:
        ctx: PipelineContext（含 model/datamodule/resolved/scene_kwargs）
        epochs: 训练 epoch 数
        learning_rate: 学习率
    """
    import itertools
    import torch.nn.functional as F
    from sklearn.metrics import f1_score

    from .....scenes.wifi_csi.dann import dann_lambda_schedule

    accelerator = (ctx.lightning_params or {}).get("accelerator")
    device = torch.device("cuda" if accelerator in ("gpu", "cuda") else "cpu")
    model = ctx.model.to(device)

    # MEDIUM 5：从 ctx.resolved 读取 optimizer/scheduler/grad_clip/early_stopping
    resolved = ctx.resolved or {}
    optimizer_type = resolved.get("optimizer", "adamw")
    weight_decay = resolved.get("weight_decay", 0.0)
    scheduler_type = resolved.get("scheduler")
    gradient_clip_val = resolved.get("gradient_clip_val")
    early_stopping_patience = resolved.get("early_stopping")

    # 构造 optimizer（dispatch 同 GenericLightningModule.configure_optimizers）
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type!r}. "
            f"Supported: adam, sgd, rmsprop, adamw"
        )

    # 构造 scheduler（同 GenericLightningModule.configure_optimizers）
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1))

    # 数据加载：EEG 任务集 + CSI 对抗集
    train_loader = ctx.datamodule.train_dataloader()
    val_loader = ctx.datamodule.val_dataloader()
    # CSI 对抗信号：从 scene_kwargs.csi_loader 获取（stage_load 注入），无则跳过对抗
    csi_loader = ctx.scene_kwargs.get("csi_loader") if ctx.scene_kwargs else None
    csi_iter = itertools.cycle(csi_loader) if csi_loader else None

    best_val_acc = 0.0
    no_improve_count = 0  # early stopping 计数
    # LOW 7：追踪 best epoch 的 val_loss/val_macro_f1（供 wrapper 写 final_eval）
    best_val_loss = None
    best_val_macro_f1 = None
    best_epoch = None

    for epoch in range(epochs):
        # λ 调度（Ganin & Lempitsky 2015：λ = 2/(1+exp(-10*p))-1）
        lambda_ = dann_lambda_schedule(epoch, epochs)

        # MEDIUM 6：累积 train loss（与 GenericLightningModule 对齐）
        epoch_train_loss_sum = 0.0
        epoch_train_steps = 0

        model.train()
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x_eeg, y_eeg = batch[0], batch[1]
            else:
                continue
            x_eeg = x_eeg.to(device).float()
            y_eeg = y_eeg.to(device).long()

            # CSI 对抗信号
            x_csi = None
            if csi_iter is not None:
                csi_batch = next(csi_iter)
                if isinstance(csi_batch, (list, tuple)):
                    x_csi = csi_batch[0].to(device).float()

            # DANN forward：返回 (logits, disc_loss)
            # disc_loss 在 model 内部已计算（CSI=0, EEG=1 的模态分类 CE）
            logits, disc_loss = model(x_eeg, x_csi, lambda_)
            task_loss = F.cross_entropy(logits, y_eeg)

            total_loss = task_loss
            if disc_loss is not None:
                total_loss = total_loss + disc_loss

            optimizer.zero_grad()
            total_loss.backward()
            # MEDIUM 5：梯度裁剪（与 Lightning Trainer gradient_clip_val 对齐）
            if gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            # MEDIUM 6：累积 train loss
            epoch_train_loss_sum += float(total_loss.item())
            epoch_train_steps += 1

        # 验证（仅 task 分类，无对抗）
        model.eval()
        all_preds, all_labels = [], []
        # MEDIUM 6：累积 val loss（与 GenericLightningModule 对齐）
        epoch_val_loss_sum = 0.0
        epoch_val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    continue
                x = x.to(device).float()
                y = y.to(device).long()
                # 验证时 x_csi=None, lambda_=0（不触发对抗路径）
                logits, _ = model(x, None, 0.0)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())
                # MEDIUM 6：累积 val loss
                val_loss = F.cross_entropy(logits, y)
                epoch_val_loss_sum += float(val_loss.item())
                epoch_val_steps += 1

        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
        # 空 val_loader 兜底：sklearn f1_score 不接受空数组，置 0.0
        if all_labels:
            macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        else:
            macro_f1 = 0.0

        # MEDIUM 5：scheduler step（每 epoch 后）
        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # LOW 7：追踪 best epoch 的 val_loss/val_macro_f1（供 wrapper 写 final_eval）
            best_val_loss = epoch_val_loss_sum / max(epoch_val_steps, 1)
            best_val_macro_f1 = macro_f1
            best_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1

        _logger.info(
            "DANN epoch %d/%d: λ=%.4f, val_acc=%.4f, macro_f1=%.4f",
            epoch + 1, epochs, lambda_, val_acc, macro_f1,
        )

        # MEDIUM 6 修复：写入 training_log（与 GenericLightningModule.on_validation_epoch_end 对齐）
        if not hasattr(ctx, "training_log") or ctx.training_log is None:
            ctx.training_log = []
        ctx.training_log.append({
            "epoch": epoch + 1,
            "lr": learning_rate,
            "train_loss": round(epoch_train_loss_sum / max(epoch_train_steps, 1), 6),
            "train_accuracy": None,  # DANN 路径暂不计算 train_accuracy
            "val_loss": round(epoch_val_loss_sum / max(epoch_val_steps, 1), 6),
            "val_accuracy": round(float(val_acc), 6),
            "val_macro_f1": round(float(macro_f1), 6),
            "phase": "train_val",
        })

        # MEDIUM 5：early stopping（手动实现，监控 val_acc）
        if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
            _logger.info(
                "DANN early stopping at epoch %d (patience=%d, no improve=%d)",
                epoch + 1, early_stopping_patience, no_improve_count,
            )
            break

    # 写回 ctx（与 Lightning 路径对齐）
    ctx.best_model_score = best_val_acc
    ctx.best_model_path = None  # DANN 无 Lightning checkpoint
    ctx.best_epoch = best_epoch  # DANN 路径：best_val_acc 对应的 epoch（1-based，与 training_log 对齐）
    ctx.pruned = False
    ctx.pruned_epoch = None
    # LOW 7 修复：保存 best epoch 的完整 metrics（供 wrapper 写 final_eval）
    ctx._dann_best_val_loss = best_val_loss
    ctx._dann_best_val_macro_f1 = best_val_macro_f1


@stage(
    name="train",
    reads=["config", "model", "datamodule", "module", "callbacks",
           "lightning_params", "pl_logger", "csv_logger", "resolved",
           "route_config", "distributed_kwargs", "learning_mode"],
    writes=["trainer", "training_duration_s", "best_model_path", "best_model_score",
            "intermediate_values",  # 任务3：补报 intermediate_values（FrozenDict 冻结写入）
            "pruned", "pruned_epoch"],  # P1.1: 实时早停状态（on_pruned 回调写入）
    description="Stage 6: 训练执行",
)
def stage_train(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: 训练执行。

    RFC-002 阶段 K：支持 trainer_factory 注入，Agent 可自定义 Trainer 构造。
    """
    is_self_supervised = (ctx.learning_mode == "self_supervised")
    deterministic = ctx.config.trainer.deterministic

    # 子进程隔离方案（2026-07-11）：probe 在独立子进程中运行，主进程不消耗 RNG，
    # 不需要在 stage_train 入口重新 set_seed。set_seed 仅在 stage_preflight 调用
    # 一次，之后 RNG 自然流经 stage_load/resolve/build，与 N0 基线（无 probe）
    # 路径一致。在 stage_train 入口额外 set_seed 会重置 RNG，导致 DataLoader
    # shuffle 顺序与 N0 基线不同（实测 ep0 从 1.210943 变为 1.258861）。

    enable_progress_bar = ctx.config.trainer.enable_progress_bar
    max_time = ctx.config.trainer.max_time or "00:02:00:00"

    # checkpoint 恢复
    resume_ckpt = ctx.config.trainer.resume
    if resume_ckpt is None and ctx.config.scene.params:
        resume_ckpt = ctx.config.scene.params.get("resume")

    # P1-4: stage 入口摘要日志
    _epochs = ctx.config.trainer.epochs
    _batch_size = ctx.resolved.get("batch_size") if ctx.resolved else None
    _lr = ctx.resolved.get("learning_rate") if ctx.resolved else None
    _optimizer = ctx.resolved.get("optimizer") if ctx.resolved else None
    _scheduler = ctx.resolved.get("scheduler") if ctx.resolved else None
    _logger.info(
        "stage_train input: epochs=%s, batch_size=%s, learning_rate=%s, "
        "optimizer=%s, scheduler=%s, learning_mode=%s, resume_ckpt=%s",
        _epochs, _batch_size, _lr, _optimizer, _scheduler,
        ctx.learning_mode, resume_ckpt,
    )

    # 修复（任务3 / P0）：dry-run 模式跳过 trainer.fit()，仅输出训练 plan。
    # 旧逻辑用 limit_train_batches=1 近似 dry-run，但仍执行完整 fit/validation/
    # checkpoint，产生副作用（写 checkpoint、占显存、跑验证）。改为入口直接短路。
    if ctx.dry_run:
        plan = {
            "epochs": _epochs,
            "batch_size": _batch_size,
            "learning_rate": _lr,
            "optimizer": _optimizer,
            "scheduler": _scheduler,
            "device": ctx.lightning_params.get("accelerator") if ctx.lightning_params else None,
            "devices": ctx.lightning_params.get("devices") if ctx.lightning_params else None,
            "precision": ctx.lightning_params.get("precision") if ctx.lightning_params else None,
            "max_time": max_time,
            "learning_mode": ctx.learning_mode,
            "resume_ckpt": resume_ckpt,
        }
        _logger.info("stage_train dry-run plan: %s", json.dumps(plan, default=str))

        # 修复（任务2 / P1）：dry-run 短路改为前向传播验证。
        # 旧逻辑 dry-run 完全跳过 fit()，不执行任何前向传播，无法验证模型可前向。
        # CLI 的 _cmd_dry_run 动态校验需要"1 epoch + 1 batch 前向"验证模型可前向。
        # 方案：从 datamodule 取 1 个 batch，验证模型可前向。失败不阻断 dry-run
        # （只 warning），因为前向验证的目的是验证模型可前向，非阻断性校验。
        try:
            if hasattr(ctx.datamodule, 'setup'):
                try:
                    ctx.datamodule.setup()
                except Exception:
                    pass
            train_dl = ctx.datamodule.train_dataloader() if hasattr(ctx.datamodule, 'train_dataloader') else None
            if train_dl is not None and ctx.model is not None:
                batch = next(iter(train_dl))
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch.get('x') or batch.get('input') or list(batch.values())[0]
                else:
                    x = batch
                with torch.no_grad():
                    output = ctx.model(x)
                _logger.info(
                    "stage_train dry-run: forward pass OK, output shape=%s",
                    output.shape if hasattr(output, 'shape') else type(output).__name__,
                )
        except Exception as e:
            _logger.warning("stage_train dry-run: forward pass failed: %s", e)
            # 前向失败不阻断 dry-run，只在报告中标记

        ctx.training_duration_s = 0.0
        ctx.best_model_path = None
        ctx.best_model_score = None
        ctx.best_epoch = None  # 任务1：dry-run 无训练，best_epoch 置 None
        _logger.info(
            "stage_train dry-run: skipped trainer.fit(), forward validation done"
        )
        return ctx

    # v2 差距 2+3：DANN 训练分支（use_dann=True 时走独立循环，不走 Lightning Trainer）
    if _should_use_dann(ctx.config.scene.params):
        _logger.info(
            "stage_train: DANN branch activated (use_dann=True), "
            "bypassing Lightning Trainer"
        )
        timer = Timer("training_dann")
        timer.__enter__()
        try:
            _train_dann_loop(
                ctx=ctx,
                epochs=_epochs,
                learning_rate=_lr or ctx.config.trainer.learning_rate or 1e-3,
            )
        finally:
            timer.__exit__(None, None, None)
        ctx.training_duration_s = round(timer.elapsed, 2)
        # Critical #1：DANN 路径无 Lightning Trainer，ctx.trainer 保持 None。
        # 写入 final_eval 供 stage_eval 使用（stage_eval 检测 ctx.trainer is None
        # 时跳过 trainer.validate/test，使用此处的 final_eval 计算 feedback）。
        # LOW 7 修复：final_eval 完整化（含 val_loss + val_macro_f1）
        final_eval = {"val_accuracy": round(float(ctx.best_model_score), 6)} if ctx.best_model_score is not None else {}
        best_val_loss = getattr(ctx, "_dann_best_val_loss", None)
        best_val_macro_f1 = getattr(ctx, "_dann_best_val_macro_f1", None)
        if best_val_loss is not None:
            final_eval["val_loss"] = round(float(best_val_loss), 6)
        if best_val_macro_f1 is not None:
            final_eval["val_macro_f1"] = round(float(best_val_macro_f1), 6)
        ctx.final_eval = final_eval
        # Important #2：与 Lightning 路径对齐，冻结 intermediate_values 防止
        # stage_eval 的 trainer.validate() 触发 IntermediateMetricLogger 写入。
        ctx.intermediate_values = FrozenDict(ctx.intermediate_values)
        _logger.info(
            "intermediate_values frozen with %d entries after DANN stage_train",
            len(ctx.intermediate_values),
        )
        return ctx

    timer = Timer("training")
    timer.__enter__()

    # P1-5.8: 训练入口 log 显存占用 + 梯度裁剪配置（可观测性补全）
    try:
        import torch as _torch
        _grad_clip_val = ctx.resolved.get("gradient_clip_val")
        _grad_clip_algo = ctx.resolved.get("gradient_clip_algorithm", "norm")
        if _grad_clip_val is not None:
            _logger.info(
                "stage_train: gradient_clip configured (val=%s, algorithm=%s)",
                _grad_clip_val, _grad_clip_algo,
            )
        else:
            _logger.info("stage_train: gradient_clip disabled (val=None)")
        if _torch.cuda.is_available():
            _allocated = _torch.cuda.memory_allocated() / (1024 ** 3)
            _reserved = _torch.cuda.memory_reserved() / (1024 ** 3)
            _logger.info(
                "stage_train: GPU memory before fit (allocated=%.3f GB, reserved=%.3f GB)",
                _allocated, _reserved,
            )
    except Exception as _e:
        _logger.debug("stage_train: failed to log GPU memory / gradient config: %s", _e)

    # RFC-002 阶段 K：Trainer 构造参数
    def _build_trainer_kwargs(**overrides):
        kwargs = {
            "accelerator": ctx.lightning_params["accelerator"],
            "devices": ctx.lightning_params["devices"],
            "precision": ctx.lightning_params["precision"],
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": False,
            "deterministic": deterministic,
            "max_time": max_time,
            "gradient_clip_val": ctx.resolved.get("gradient_clip_val"),
            "gradient_clip_algorithm": ctx.resolved.get("gradient_clip_algorithm", "norm"),
            "accumulate_grad_batches": ctx.resolved.get("accumulate_grad_batches", 1),
            **ctx.distributed_kwargs,
        }
        # P2-3: 从 config 读取 limit_train_batches / limit_val_batches
        # 仅在非 None 时添加（默认 None 保持向后兼容，dry-run 动态校验时设为 1）
        # 调用方可通过 overrides 覆盖（如自监督阶段 limit_val_batches=0）
        _limit_train = getattr(ctx.config.trainer, "limit_train_batches", None)
        _limit_val = getattr(ctx.config.trainer, "limit_val_batches", None)
        if _limit_train is not None:
            kwargs["limit_train_batches"] = _limit_train
        if _limit_val is not None:
            kwargs["limit_val_batches"] = _limit_val
        # Part 4：自动 LR 标定注入 Trainer 构造参数
        if ctx.config.trainer.auto_lr_find:
            kwargs["auto_lr_find"] = True
        kwargs.update(overrides)
        return kwargs

    if is_self_supervised:
        ss_epochs = ctx.resolved.get("self_supervised_epochs", 100)
        sup_epochs = ctx.config.trainer.epochs

        # Phase 1: 自监督预训练（P3: OOM 回退）
        ctx.module.phase = "self_supervised"
        def _build_ss_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=ss_epochs,
                    logger=ctx.csv_logger,
                    enable_checkpointing=False,
                    **_build_trainer_kwargs(limit_val_batches=0),
                )
            return pl.Trainer(
                max_epochs=ss_epochs,
                logger=ctx.csv_logger,
                enable_checkpointing=False,
                **_build_trainer_kwargs(limit_val_batches=0),
            )
        def _fit_ss(trainer):
            # 每次重新获取 dataloader，OOM 重试时反映新 batch_size
            trainer.fit(ctx.module, train_dataloaders=ctx.datamodule.train_dataloader())
        # RFC-005：存 SS Phase 1 Trainer 返回值，fit 后显式 _teardown 释放
        # 修复（2.10）：非 OOM 异常时 ss_trainer 资源泄露——用 try/finally 确保
        # 异常路径也 teardown。同时修复（2.9）：del 后加 gc.collect() 打断循环引用。
        ss_trainer = _fit_with_oom_fallback(ctx, _build_ss_trainer, _fit_ss)
        try:
            if hasattr(ss_trainer, "_teardown"):
                try:
                    ss_trainer._teardown()
                except Exception:
                    pass
        finally:
            del ss_trainer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()

        # Phase 2: 监督微调（P3: OOM 回退）
        ctx.module.phase = "supervised"
        ctx.module._current_epoch_loss = 0.0
        ctx.module._current_epoch_steps = 0
        def _build_sup_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=sup_epochs,
                    callbacks=ctx.callbacks,
                    logger=ctx.csv_logger,
                    enable_checkpointing=True,
                    **_build_trainer_kwargs(),
                )
            return pl.Trainer(
                max_epochs=sup_epochs,
                callbacks=ctx.callbacks,
                logger=ctx.csv_logger,
                enable_checkpointing=True,
                **_build_trainer_kwargs(),
            )
        def _fit_sup(trainer):
            trainer.fit(
                ctx.module,
                train_dataloaders=ctx.datamodule.supervised_dataloader(),
                val_dataloaders=ctx.datamodule.val_dataloader(),
                ckpt_path=resume_ckpt,
            )
        ctx.trainer = _fit_with_oom_fallback(ctx, _build_sup_trainer, _fit_sup)
    else:
        epochs = ctx.config.trainer.epochs
        max_epochs = ctx.route_config.get("max_epochs", float("inf"))
        if epochs > max_epochs:
            epochs = max_epochs

        # P3: OOM 回退——trainer 构造与 fit 分离，便于 OOM 时重建重试
        def _build_supervised_trainer():
            if ctx.config.trainer_factory is not None:
                return ctx.config.trainer_factory(
                    max_epochs=epochs,
                    callbacks=ctx.callbacks,
                    logger=ctx.csv_logger,
                    enable_checkpointing=True,
                    **_build_trainer_kwargs(),
                )
            return pl.Trainer(
                max_epochs=epochs,
                callbacks=ctx.callbacks,
                logger=ctx.csv_logger,
                enable_checkpointing=True,
                **_build_trainer_kwargs(),
            )

        def _fit_supervised(trainer):
            trainer.fit(ctx.module, datamodule=ctx.datamodule, ckpt_path=resume_ckpt)

        # Part 4（风险推演 R3）：自动 LR 标定。
        # 用独立 tune_trainer 隔离副作用——trainer.tune() 内部跑 1 epoch 训练，
        # 会触发回调写入 training_log、更新 metric 状态、可能触发 checkpoint。
        # 用独立 Trainer（关闭 checkpoint/validation/logger）隔离，tune 后清理状态。
        if ctx.config.trainer.auto_lr_find:
            _logger.info("stage_train: auto_lr_find enabled, running LR Range Test...")
            try:
                tune_trainer = pl.Trainer(
                    **_build_trainer_kwargs(
                        max_epochs=1,
                        enable_checkpointing=False,
                        limit_val_batches=0,
                        logger=False,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                    ),
                    auto_lr_find=True,
                )
                tune_result = tune_trainer.tune(ctx.module, datamodule=ctx.datamodule)
                suggestion = tune_result.get("lr_find", {}).get("suggestion")
                if suggestion is not None:
                    ctx.module.learning_rate = suggestion
                    ctx.resolved["learning_rate"] = suggestion
                    _logger.info(
                        "stage_train: auto_lr_find suggested lr=%.6f", suggestion
                    )
                else:
                    _logger.warning(
                        "stage_train: auto_lr_find failed to suggest lr, using default"
                    )
                # 清理 tune_trainer（释放显存）
                del tune_trainer
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 清理 tune 期间的副作用：清空 training_log + 重置累加器
                # tune_trainer.tune() 会触发 on_train_epoch_end 写入 epoch 0 entry
                ctx.module.training_log.clear()
                ctx.module._current_epoch_loss = 0.0
                ctx.module._current_epoch_steps = 0
                ctx.module._current_val_epoch_loss = 0.0
                ctx.module._current_val_epoch_steps = 0
                ctx.module._has_validation_run = False
                # 重置 metric 状态（tune 期间更新了 torchmetrics）
                for metric_dict in [ctx.module.train_metrics, ctx.module.val_metrics]:
                    for name in metric_dict:
                        try:
                            metric_dict[name].reset()
                        except Exception:
                            pass
                _logger.info("stage_train: auto_lr_find done, training_log cleared, starting fit")
            except Exception as e:
                _logger.warning(
                    "stage_train: auto_lr_find failed: %s, using default lr", e
                )

        ctx.trainer = _fit_with_oom_fallback(ctx, _build_supervised_trainer, _fit_supervised)

    # 训练结束：停止计时器 + 提取 checkpoint 信息到 first-class 字段
    timer.__exit__()
    ctx.training_duration_s = round(timer.elapsed, 2)
    for cb in ctx.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ctx.best_model_path = cb.best_model_path or None
            ctx.best_model_score = float(cb.best_model_score) if cb.best_model_score is not None else None
            # 任务1（P0）：从 best_model_path 文件名解析 best_epoch。
            # ModelCheckpoint filename 格式 "best-{epoch}-{val_loss:.3f}" 实际生成
            # "best-epoch=14-val_loss=0.066.ckpt"，用正则 r"epoch=(\d+)" 解析 epoch 号。
            # 解析失败时回退到 len(module.training_log)（最后完成的 epoch 号，1-based）。
            # best_epoch 用于 stage_eval 的 analyze_training_result 取 best epoch 那轮
            # train 指标，与 final_eval 的 val 指标配对算 gap，避免数据源不一致的过拟合误报。
            if ctx.best_model_path:
                import re as _re
                _m = _re.search(r"epoch=(\d+)", ctx.best_model_path)
                if _m:
                    ctx.best_epoch = int(_m.group(1))
                else:
                    _tl = getattr(ctx.module, "training_log", None) if ctx.module else None
                    ctx.best_epoch = len(_tl) if _tl else None
            else:
                ctx.best_epoch = None
            break

    # 修复：best model 加载回 ctx.model。
    # 旧逻辑训练结束后 ctx.model 仍是最后一代权重，导出的 model.pth 是最后一代
    # 而非最优，final_eval 反映最后一代性能（可能因 early stopping 远差于 best）。
    # 改为：若有 best checkpoint，加载回 ctx.model，确保后续 export/eval 用最优权重。
    # 复用 load_checkpoint_flexible（senseframe/common/checkpoint.py）统一三处
    # checkpoint 加载逻辑（stage_train / export / inference），消除反模式重复。
    if ctx.best_model_path:
        try:
            load_info = load_checkpoint_flexible(
                ctx.best_model_path, ctx.model,
                map_location="cpu", weights_only=False,
            )
            _logger.info(
                "stage_train: loaded best model weights from %s into ctx.model "
                "(best_score=%s, format=%s, keys=%d, prefix=%r)",
                ctx.best_model_path,
                ctx.best_model_score,
                load_info["source_format"],
                load_info["num_keys_loaded"],
                load_info["stripped_prefix"],
            )
        except FileNotFoundError:
            _logger.warning(
                "stage_train: best_model_path does not exist: %s",
                ctx.best_model_path,
            )
        except Exception as e:
            _logger.warning(
                "stage_train: failed to load best checkpoint %s: %s",
                ctx.best_model_path,
                e,
                exc_info=True,
            )

    # P0-1 防御性兜底：stage_train 后冻结 intermediate_values
    # 防止 stage_eval 的 trainer.validate() 触发 IntermediateMetricLogger 写入
    ctx.intermediate_values = FrozenDict(ctx.intermediate_values)
    _logger.info(
        "intermediate_values frozen with %d entries after stage_train",
        len(ctx.intermediate_values),
    )

    # P1.1 Multi-fidelity 实时早停 — stage_train 出口感知 pruned 状态
    # on_pruned 回调（stage_build 注入）在 IntermediateMetricLogger 触发 should_prune=True 时
    # 已写入 ctx.pruned=True / ctx.pruned_epoch=epoch。此处仅做日志留痕，不阻断后续流程。
    # best_model 加载仍执行（剪枝前 ModelCheckpoint 可能已保存 best），但下游 stage_export
    # 应感知 pruned 状态，将 pruned/pruned_epoch 写入 TrainOutput 供 MethodRunner 区分。
    if ctx.pruned:
        _logger.info(
            "stage_train: trial pruned at epoch=%d (real-time early stopping via pruner)",
            ctx.pruned_epoch,
        )

    # P1-4: stage 出口摘要日志
    _logger.info(
        "stage_train output: best_model_score=%s, best_model_path=%s, "
        "training_duration_s=%s, intermediate_values_count=%d, "
        "pruned=%s, pruned_epoch=%s",
        ctx.best_model_score, ctx.best_model_path,
        ctx.training_duration_s,
        len(ctx.intermediate_values),
        ctx.pruned, ctx.pruned_epoch,
    )

    return ctx


def analyze_training_result(
    final_eval: Dict[str, Any],
    training_log: List[Any],
    early_stopped: bool,
    task_type: str = "classification",
    best_epoch: Optional[int] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """分析训练结果，输出结构化反馈（RFC-002 阶段 L）。

    闭合探索-反馈回路：eval 结果 → 失败分类 + 改进建议 → Agent 调整策略。

    任务1（P0）修复：新增 best_epoch 参数。旧逻辑反向遍历 training_log 找
    **最后一轮**的 train/val 指标算 gap，但 final_eval 来自 best checkpoint
    （stage_train 已加载 best 权重到 ctx.model），两个数据源不一致导致过拟合误报
    （实测 best epoch val=0.982 但末轮 val=0.823，gap=0.161 误报 overfitting）。
    修复后：若 best_epoch 提供，从 training_log 找 entry["epoch"] == best_epoch
    的条目，用该条目的 train/val 指标算 gap，数据源与 final_eval 一致。

    对称性修复：新增 val-test gap 泛化分析。P2-3 修复后 val/test 分离，
    final_eval 同时含 val_* 和 test_* 指标。val-test gap 过大表示
    模型在 val 上调参（early_stopping）后在 test 上泛化能力下降。

    Args:
        final_eval: 最终评估指标（含 val_* 和 test_* 前缀）
        training_log: 训练日志（每 epoch 1 条）
        early_stopped: 是否早停
        task_type: 任务类型（classification/regression）
        best_epoch: best checkpoint 的 epoch 号（None 时回退到末轮逻辑）

    Returns:
        {"status", "diagnosis", "suggestions"}
    """
    import math

    # 1. 数值不稳定：指标含 NaN/Inf
    for k, v in final_eval.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return {
                "status": "numerical_instability",
                "diagnosis": f"指标 {k} 包含 NaN/Inf，训练数值不稳定",
                "suggestions": [
                    "降低 learning_rate",
                    "检查 loss 函数数值稳定性（可用 numerical_stability_validator）",
                    "启用梯度裁剪 (gradient_clip_val)",
                    "检查输入数据是否已归一化",
                ],
            }

    # 2. 从 training_log 提取 train/val metric
    # 修复（任务1 / P0）：feedback 基于 best epoch 而非 final epoch。
    # 旧逻辑反向遍历找最后一轮的 train/val 指标算 gap，但 final_eval 来自 best
    # checkpoint（stage_train 已加载 best 权重），两数据源不一致导致过拟合误报。
    # 修复：若 best_epoch 提供，从 training_log 找 entry["epoch"] == best_epoch
    # 的条目，用该条目的 train/val 指标算 gap，数据源与 final_eval 一致。
    # best_epoch 为 None 或找不到对应条目时回退原逻辑（反向遍历找最后一轮）。
    # Part 3（风险推演 R1）：过滤掉 final_eval 行，避免回退反向遍历取到
    # final validation 的 val_accuracy（来自 best checkpoint）与 epoch N 的
    # train_accuracy 错配，重新引入数据源不一致问题。
    # phase 字段可选，默认 "train_val"（向后兼容无 phase 字段的旧 entry）。
    trainable_log = [
        e for e in (training_log if isinstance(training_log, list) else [])
        if isinstance(e, dict) and e.get("phase", "train_val") != "final_eval"
    ]
    last_train_acc = None
    last_val_acc = None
    best_entry = None
    if best_epoch is not None:
        for entry in trainable_log:
            if isinstance(entry, dict) and entry.get("epoch") == best_epoch:
                best_entry = entry
                break
    if best_entry is not None:
        last_train_acc = best_entry.get("train_accuracy") or best_entry.get("train_acc")
        last_val_acc = best_entry.get("val_accuracy") or best_entry.get("val_acc")
    else:
        # 回退：反向遍历找最后一轮
        for entry in reversed(trainable_log):
            if not isinstance(entry, dict):
                continue
            if last_train_acc is None:
                last_train_acc = entry.get("train_accuracy") or entry.get("train_acc")
            if last_val_acc is None:
                last_val_acc = entry.get("val_accuracy") or entry.get("val_acc")
            if last_train_acc is not None and last_val_acc is not None:
                break

    # P4-5：val_acc 提取改为显式 is not None 检查链。
    # 旧代码用 or 链，当 val_accuracy=0.0（falsy）时会错误跳到 accuracy 或 last_val_acc，
    # 导致 underfitting 检查被静默跳过。
    val_acc = None
    for _key in ("val_accuracy", "accuracy"):
        _v = final_eval.get(_key)
        if _v is not None:
            val_acc = _v
            break
    if val_acc is None:
        val_acc = last_val_acc

    # 3. 欠拟合：验证准确率过低
    # P4-5：阈值动态化，基于 n_classes 计算随机猜测基线。
    # 旧代码硬编码 0.5，对多分类（如 7 类，随机基线≈0.143）过宽松。
    # 新阈值 = max(2/n_classes, 0.3)，即随机基线的 2 倍与 0.3 取较大值。
    if val_acc is not None and task_type == "classification":
        if n_classes is not None and n_classes > 1:
            underfit_threshold = max(2.0 / n_classes, 0.3)
        else:
            underfit_threshold = 0.5
        if val_acc < underfit_threshold:
            return {
                "status": "underfitting",
                "diagnosis": (
                    f"验证准确率 {val_acc:.3f} 低于阈值 {underfit_threshold:.3f}"
                    f"（n_classes={n_classes}, 随机基线≈{1.0/(n_classes or 2):.3f}），模型欠拟合"
                ),
                "suggestions": [
                    "增大模型容量（更多层/更宽）",
                    "增加训练轮数 (epochs)",
                    "降低正则化强度 (weight_decay)",
                    "尝试更丰富的特征工程 pipeline",
                ],
            }

    # 4. 过拟合：train-val gap 过大
    if last_train_acc is not None and last_val_acc is not None:
        gap = last_train_acc - last_val_acc
        if gap > 0.15:
            return {
                "status": "overfitting",
                "diagnosis": f"train-val gap {gap:.3f}（train={last_train_acc:.3f}, val={last_val_acc:.3f}），模型过拟合",
                "suggestions": [
                    "增加数据增强 (params.transform.augment)",
                    "增大 weight_decay",
                    "启用/增加 dropout",
                    "减小模型容量",
                    "启用 early_stopping",
                ],
            }

    # 对称性修复：val-test gap 泛化分析
    # P2-3 修复后 val/test 分离，final_eval 同时含 val_* 和 test_* 指标
    # val-test gap 过大表示模型在 val 上调参（early_stopping）后在 test 上泛化能力下降
    _test_acc = final_eval.get("test_accuracy") or final_eval.get("test_acc")
    if (val_acc is not None and _test_acc is not None
            and task_type == "classification"):
        val_test_gap = val_acc - _test_acc
        if val_test_gap > 0.10:
            return {
                "status": "generalization_gap",
                "diagnosis": (f"val-test gap {val_test_gap:.3f}"
                              f"（val={val_acc:.3f}, test={_test_acc:.3f}），"
                              f"模型在 val 上调参后 test 泛化能力下降"),
                "suggestions": [
                    "增大 val_split_ratio（如 0.1 → 0.2）以获得更稳健的 val 估计",
                    "检查 val/test 分布是否一致（domain shift）",
                    "使用 k-fold 交叉验证替代单次 split",
                    "增大 early_stopping patience 容忍 val 波动",
                ],
            }

    # 5. 已收敛：早停
    if early_stopped:
        return {
            "status": "converged",
            "diagnosis": "训练早停，模型已收敛",
            "suggestions": [
                "尝试更激进的策略（更大 lr、不同 loss）",
                "尝试不同的特征工程 pipeline（见 catalog.suggest_pipeline）",
                "探索兼容性矩阵中的其他组合",
            ],
        }

    # 6. 正常完成
    return {
        "status": "success",
        "diagnosis": "训练正常完成",
        "suggestions": [
            "记录当前策略到技能库供复用 (save_skill)",
            "探索 ExplorationTracker.recommend_next 推荐的下一步",
        ],
    }
