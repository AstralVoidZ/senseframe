"""Stage 7: 评估 + 指标合并。

包含：
- stage_eval：评估 stage（含 trainer.validate/test + 结构化反馈 + OTel 埋点）
- _merge_metrics_csv：合并 Lightning CSVLogger 的 train+val 分行为 1 行/epoch
"""
from __future__ import annotations

from pathlib import Path

from ..context import PipelineContext, _logger
from ..stage_spec import stage
from .....observability_otel import (
    record_training_metric, record_trial_metric,
    ML_VAL_LOSS, ML_VAL_ACCURACY,
    ML_TEST_LOSS, ML_TEST_ACCURACY,
    ML_TRIAL_COUNT,
)
from .train import analyze_training_result

try:
    from pytorch_lightning.callbacks import EarlyStopping
except ImportError:
    from lightning.pytorch.callbacks import EarlyStopping


@stage(
    name="eval",
    reads=["config", "trainer", "module", "datamodule",
           "task_spec", "exploration_history", "learning_mode"],
    writes=["output", "exploration_history",  # P2.1: 对齐函数体（写 exploration_history.feedback）
            "final_eval", "training_log", "early_stopped", "feedback"],  # 任务3：补报 stage_eval 写入字段
    description="Stage 7: 评估",
)
def stage_eval(ctx: PipelineContext) -> PipelineContext:
    """Stage 7: 评估。

    RFC-002 阶段 L：输出结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路。

    P4-2 文档澄清：本 stage 内部调用 ctx.trainer.validate()，Lightning Trainer
    会对每个 validation batch 调用 LightningModule.validation_step（约 N 次，
    N = validation batch 数）。这是 Lightning 的固有行为，与 replace_stage 无关。

    replace_stage("eval", fn) 的语义是完全取代 stage 函数，原 stage_eval 不执行。
    若需"eval 后钩子"，应使用 after("eval", hook) 且 hook 内部不调用 trainer.validate()。
    """
    # P5 P1-N：dry_run 模式下跳过评估（ctx.trainer 为 None，trainer.validate() 会崩溃）
    if ctx.dry_run:
        _logger.info("Skipping stage_eval in dry_run mode")
        return ctx

    is_self_supervised = (ctx.learning_mode == "self_supervised")

    # Critical #1：DANN 路径下 ctx.trainer 为 None（_train_dann_loop 不使用
    # Lightning Trainer）。_train_dann_loop 已计算 val 指标并写入 ctx.final_eval
    # 与 ctx.best_model_score。此处跳过 trainer.validate/test（避免 NoneType 崩溃），
    # 使用 stage_train 已写入的 final_eval 继续后续 feedback 计算。
    if ctx.trainer is None:
        _logger.info(
            "stage_eval: skipping trainer.validate/test (DANN path, "
            "ctx.trainer is None, using final_eval from _train_dann_loop)"
        )
        final_eval = ctx.final_eval
        training_log = ctx.training_log if ctx.training_log else []
        early_stopped = False
    else:
        # 修复（2.7）：_is_final_validation 标志在 trainer.validate() 完成后必须 reset，
        # 否则模块复用时（如 HPO 多 trial 复用同一 module）状态污染，后续训练中验证
        # 误走 final_validation 路径。用 try/finally 确保异常时也 reset。
        ctx.module._is_final_validation = True
        try:
            if is_self_supervised:
                ctx.trainer.validate(ctx.module, dataloaders=ctx.datamodule.val_dataloader())
            else:
                ctx.trainer.validate(ctx.module, datamodule=ctx.datamodule)
        finally:
            ctx.module._is_final_validation = False

        # 对称性修复：在 trainer.validate() 后新增 trainer.test() 调用
        # P2-3 修复后 val/test 分离，test 集需要独立评估以报告泛化能力
        # trainer.test() 触发 test_step → on_test_epoch_end，存储 _last_test_metrics
        # get_final_metrics 会合并 val_* 和 test_* 指标
        ctx.module._is_final_test = True
        try:
            if is_self_supervised:
                ctx.trainer.test(ctx.module, dataloaders=ctx.datamodule.test_dataloader())
            else:
                ctx.trainer.test(ctx.module, dataloaders=ctx.datamodule.test_dataloader())
        finally:
            ctx.module._is_final_test = False

        # 收集结果（get_final_metrics 现在合并 val_* 和 test_* 指标）
        final_eval = ctx.module.get_final_metrics()
        training_log = ctx.module.training_log
        early_stopped = any(
            isinstance(cb, EarlyStopping) and cb.stopped_epoch >= 0
            for cb in ctx.trainer.callbacks
        )
        # 修复（5.8）：early stopping 触发时无日志，加 INFO 留痕
        if early_stopped:
            stopped_epoch = -1
            for cb in ctx.trainer.callbacks:
                if isinstance(cb, EarlyStopping) and cb.stopped_epoch >= 0:
                    stopped_epoch = cb.stopped_epoch
                    break
            _logger.info(
                f"early stopping triggered at epoch {stopped_epoch} "
                f"(monitor={getattr(cb, 'monitor', 'val_loss')})"
            )

    # 保存结果到 first-class 字段
    ctx.final_eval = final_eval
    ctx.training_log = training_log
    ctx.early_stopped = early_stopped

    # RFC-002 阶段 L：结构化反馈（失败分类 + 改进建议），闭合探索-反馈回路
    task_type = ctx.task_spec.task_type if ctx.task_spec else "classification"
    # 任务1（P0）：传入 best_epoch，让 analyze_training_result 从 training_log
    # 取 best epoch 那轮的 train 指标，与 final_eval 的 val 指标配对算 gap，
    # 避免数据源不一致（final epoch train vs best checkpoint val）导致过拟合误报。
    ctx.feedback = analyze_training_result(
        final_eval, training_log, early_stopped, task_type=task_type,
        best_epoch=ctx.best_epoch,
        n_classes=ctx.num_classes,
    )

    # P5 P2-7 阶段2：在 analyze_training_result 出口做类型校验并切换为 FeedbackResult 实例。
    # 下游消费方已迁移为属性访问 + to_dict() 序列化兼容。
    # hpo.py 在传给 tracker 时会调用 .to_dict() 转为 dict。
    from .....schemas import validate_feedback
    ctx.feedback = validate_feedback(ctx.feedback)

    # RFC-002 阶段 R：feedback 回写到最近一次探索试验，闭合"训练→反馈→推荐"回路
    # recommend_next 将基于此 feedback 调整优先级
    if ctx.exploration_history:
        feedback = ctx.feedback
        last_trial = ctx.exploration_history[-1]
        # P0 修复：exploration_history 会被 ExplorationTracker.save 序列化为 JSON，
        # FeedbackResult dataclass 不可直接 json.dumps。调 to_dict() 转为原生 dict。
        # hpo.py 路径已在传给 tracker 前转换；直接走 Pipeline.run() 路径需在此处转换。
        last_trial["feedback"] = feedback.to_dict() if hasattr(feedback, "to_dict") else feedback
        last_trial["result"] = {
            k: v for k, v in final_eval.items()
            if isinstance(v, (int, float, str)) or v is None
        }
        last_trial["status"] = "completed"

    # P0.2: OBP 评估指标埋点（OTel 未初始化时 no-op）
    _val_acc = final_eval.get("val_accuracy") or final_eval.get("val_acc")
    _val_loss = final_eval.get("val_loss")
    if _val_acc is not None:
        record_training_metric(ML_VAL_ACCURACY, value=float(_val_acc),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    if _val_loss is not None:
        record_training_metric(ML_VAL_LOSS, value=float(_val_loss),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    # 对称性修复：test 指标 OTel 埋点（与 val 对称）
    _test_acc = final_eval.get("test_accuracy") or final_eval.get("test_acc")
    _test_loss = final_eval.get("test_loss")
    if _test_acc is not None:
        record_training_metric(ML_TEST_ACCURACY, value=float(_test_acc),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    if _test_loss is not None:
        record_training_metric(ML_TEST_LOSS, value=float(_test_loss),
                               stage="eval", model_id=ctx.config.scene.model_id,
                               dataset=ctx.config.scene.dataset)
    # 记录 trial count
    record_trial_metric(
        ML_TRIAL_COUNT, value=len(ctx.exploration_history),
        trial_id=ctx.trial_id,
    )

    return ctx


def _merge_metrics_csv(csv_path: Path) -> None:
    """合并 Lightning CSVLogger 的 train+val 分行为 1 行/epoch（任务4 / P2）。

    根因：Lightning CSVLogger 在 on_train_epoch_end 和 on_validation_epoch_end
    分别写入一行，导致每 epoch 有 2 行（train 行含 train_* 指标但 val_* 为空，
    val 行含 val_* 指标但 train_* 为空）。这与 training_log.jsonl 的 1 行/epoch
    格式不一致，下游消费者（如 Agent 分析、manifest 校验）难以对齐。

    合并后：每 epoch 1 行，train_* 和 val_* 在同一行，与 training_log.jsonl 对齐。
    同一 epoch 的多行合并时，非空值覆盖空值（train 行的 train_* + val 行的 val_*）。
    """
    import csv
    lines = csv_path.read_text(encoding='utf-8').splitlines()
    if not lines:
        return
    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames
    if not fieldnames:
        return

    # 按 epoch 聚合：同一 epoch 的多行合并，非空值覆盖空值
    merged = {}  # epoch -> row dict
    epoch_order = []
    for row in reader:
        try:
            ep = int(float(row.get('epoch', 0)))
        except (ValueError, TypeError):
            continue
        if ep not in merged:
            merged[ep] = {'epoch': ep}
            epoch_order.append(ep)
        for k, v in row.items():
            if k == 'epoch':
                continue
            if v is not None and v != '':
                merged[ep][k] = v

    # 覆写 csv 文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epoch_order:
            writer.writerow(merged[ep])
