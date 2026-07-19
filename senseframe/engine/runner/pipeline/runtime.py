"""Pipeline class + run_pipeline 入口函数。

包含：
- _NON_SERIALIZABLE_STAGES：不可序列化 stage 集合（resume 时强制重跑）
- Pipeline：可重组的 stage pipeline
- run_pipeline：Pipeline 入口函数
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from ...config import ExperimentConfig
from ....observability import Timer
from ....observability_otel import record_training_metric
from ....schemas import TrainOutput
from ....scenes import get_scene, has_scene
from ..callbacks import StageAwareCallback
from ..errors import classify_error
from ..preflight import set_seed
from .context import (
    PipelineContext,
    StageResult,
    ReadinessReport,
    DanglingRef,
    _FIELD_FILL_STAGE,
    _logger,
)
from .stage_spec import StageSpec, StageFn
from .stages import (
    stage_validate,
    stage_preflight,
    stage_resolve,
    stage_load,
    stage_build,
    stage_probe_vram,
    stage_train,
    stage_eval,
    stage_export,
    _compute_config_hash,
    _generate_manifest,
    _PIPELINE_VERSION,
)
from .errors import _classify_runtime_error

if TYPE_CHECKING:
    pass


# P0.2：不可序列化 stage — 产出对象引用（bundle/model/trainer）无法从 JSON checkpoint 恢复。
# resume 时这些 stage 必须强制重跑，仅跳过纯计算 stage（validate/preflight/resolve）。
# probe_vram 依赖 ctx.model/ctx.datamodule 对象引用，同样不可序列化恢复。
_NON_SERIALIZABLE_STAGES = frozenset({"load", "build", "probe_vram", "train", "eval"})


@dataclass
class Pipeline:
    """可重组的 stage pipeline。

    Agent 可：
    - 使用默认 pipeline：Pipeline.default()
    - 自定义 pipeline：Pipeline(stages=[...])
    - 替换单个 stage：pipeline.replace_stage("train", my_train)
    - 插入 hook：pipeline.before("train", my_hook)
    - 跳过 stage：pipeline.skip("export")
    """

    stages: List[tuple] = field(default_factory=list)  # [(name, fn), ...]

    @classmethod
    def default(cls) -> "Pipeline":
        """默认 pipeline（9 个 stage）。"""
        return cls(stages=[
            ("validate", stage_validate),
            ("preflight", stage_preflight),
            ("load", stage_load),
            ("resolve", stage_resolve),
            ("build", stage_build),
            ("probe_vram", stage_probe_vram),
            ("train", stage_train),
            ("eval", stage_eval),
            ("export", stage_export),
        ])

    def replace_stage(self, name: str, fn: StageFn) -> "Pipeline":
        """替换指定 stage。"""
        # 修复（5.5）：replace_stage 完全静默，加 INFO 日志记录替换操作
        found = any(n == name for n, _ in self.stages)
        self.stages = [(n, fn if n == name else f) for n, f in self.stages]
        _logger.info(
            f"Pipeline.replace_stage: stage='{name}', found={found}, "
            f"new_fn={getattr(fn, '__name__', repr(fn))}"
        )
        return self

    def before(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 前插入 hook。"""
        # 修复（5.5）：before 完全静默，加 INFO 日志记录插入操作
        new_stages = []
        inserted = False
        for n, f in self.stages:
            if n == name:
                new_stages.append((f"before_{name}", hook))
                inserted = True
            new_stages.append((n, f))
        self.stages = new_stages
        _logger.info(
            f"Pipeline.before: inserted hook before stage='{name}', "
            f"inserted={inserted}, hook_fn={getattr(hook, '__name__', repr(hook))}"
        )
        return self

    def after(self, name: str, hook: StageFn) -> "Pipeline":
        """在指定 stage 后插入 hook。"""
        # 修复（5.5）：after 完全静默，加 INFO 日志记录插入操作
        new_stages = []
        inserted = False
        for n, f in self.stages:
            new_stages.append((n, f))
            if n == name:
                new_stages.append((f"after_{name}", hook))
                inserted = True
        self.stages = new_stages
        _logger.info(
            f"Pipeline.after: inserted hook after stage='{name}', "
            f"inserted={inserted}, hook_fn={getattr(hook, '__name__', repr(hook))}"
        )
        return self

    def skip(self, name: str) -> "Pipeline":
        """跳过指定 stage。"""
        # 修复（5.5）：skip 完全静默，加 INFO 日志记录跳过操作
        before_count = len(self.stages)
        self.stages = [(n, f) for n, f in self.stages if n != name]
        removed = before_count - len(self.stages)
        _logger.info(
            f"Pipeline.skip: removed stage='{name}', removed_entries={removed}, "
            f"stages_before={before_count}, stages_after={len(self.stages)}"
        )
        return self

    def stages_with_spec(self) -> List[StageSpec]:
        """返回全部 stage 的 Spec（RFC-003 DSP-3）。

        遍历当前 pipeline 的所有 stage 函数，读取 @stage 装饰器附加的
        `_stage_spec` 属性。未声明的 stage 返回空 reads/writes 的 StageSpec。
        """
        specs: List[StageSpec] = []
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if spec is None:
                spec = StageSpec(name=name)
            specs.append(spec)
        return specs

    def check_readiness(self, ctx: PipelineContext, stage_name: str) -> "ReadinessReport":
        """检查指定 stage 的 reads 字段是否已在 ctx 中就绪（RFC-004 原则 9）。

        Advisory 查询：available=False 不阻断执行，仅记录信息。
        Agent 可据此决定是否跳过 stage 或手动填充缺失字段。

        Args:
            ctx: 当前 PipelineContext
            stage_name: 要检查的 stage 名

        Returns:
            ReadinessReport
        """
        spec = None
        for name, fn in self.stages:
            if name == stage_name:
                spec = getattr(fn, "_stage_spec", None)
                break
        if spec is None:
            return ReadinessReport(stage_name=stage_name, available=True, missing_reads=[])

        missing = []
        for field_spec in spec.reads:
            if field_spec.required:
                val = getattr(ctx, field_spec.name, None)
                if val is None or (hasattr(val, "__len__") and len(val) == 0 and not isinstance(val, (str, bytes))):
                    missing.append(field_spec.name)
        return ReadinessReport(
            stage_name=stage_name,
            available=len(missing) == 0,
            missing_reads=missing,
        )

    def validate_graph(self) -> List["DanglingRef"]:
        """编译期检查：reads 声明的字段是否有对应 stage 声明产出（RFC-004 原则 9）。

        遍历所有 stage 的 writes，构建"可产出字段集"，
        然后检查每个 stage 的 reads 是否有字段不在该集合中（dangling reference）。

        Advisory：返回非空列表不阻断执行，仅提示 Agent 数据通路可能断裂。
        config / extra / completed_stages 等 agent/init 填充字段视为已就绪。

        Returns:
            DanglingRef 列表（空列表表示无 dangling reference）
        """
        # 收集所有 stage 声明产出的字段
        produced: set = set()
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if spec:
                for w in spec.writes:
                    produced.add(w.name)
        # init/agent 填充的字段视为已产出（不由 stage 产出，由构造函数或 Agent 注入）
        for k, v in _FIELD_FILL_STAGE.items():
            if v in ("init", "agent"):
                produced.add(k)

        dangling: List["DanglingRef"] = []
        for name, fn in self.stages:
            spec = getattr(fn, "_stage_spec", None)
            if not spec:
                continue
            for r in spec.reads:
                if r.name not in produced:
                    dangling.append(DanglingRef(
                        stage_name=name,
                        field_name=r.name,
                        reason="field declared as read but no stage produces it",
                    ))
        return dangling

    def run(self, ctx: PipelineContext, *, dry_run: bool = False) -> StageResult:
        """执行 pipeline（P1：支持断点续跑）。

        依次执行所有 stage，返回最终结果。
        任一 stage 抛异常则停止并返回错误。
        每个 stage 完成后写 checkpoint；失败时也写 checkpoint（标记 failed_stage）。
        若 ctx.stage_checkpoint_path 存在，加载后跳过已完成的 stage。

        Args:
            ctx: Pipeline 上下文
            dry_run: dry-run 标志（任务3），True 时 stage_train 跳过 trainer.fit()，
                     仅输出训练 plan。也可直接在调用 run 前设置 ctx.dry_run=True。
        """
        # 修复（任务3 / P0）：从 kwargs 设置 dry-run 标志到 ctx，
        # 供 stage_train 检查后跳过 trainer.fit()，避免 dry-run 仍执行
        # 完整 fit/validation/checkpoint 产生副作用。
        if dry_run:
            ctx.dry_run = True
        # 修复（OTel 全链路失效）：Pipeline.run 入口调用 init_otel，
        # 否则 record_training_metric 全部 no-op，所有 OTel 埋点失效。
        # 旧逻辑 init_otel 从未在训练流程被调用，用户以为指标在采集实际全丢。
        try:
            from ....observability_otel import init_otel
            init_otel(
                pipeline_run_id=str(ctx.output_dir) if ctx.output_dir else "",
                trial_id=getattr(ctx, "trial_id", "") or "",
                model_id=ctx.config.scene.model_id if hasattr(ctx, "config") else "",
                dataset=ctx.config.scene.dataset if hasattr(ctx, "config") else "",
            )
        except Exception as e:
            _logger.warning(f"OTel init failed (training metrics will be no-op): {e}")

        # P1：加载 checkpoint（若存在）
        if ctx.stage_checkpoint_path and ctx.stage_checkpoint_path.exists():
            ckpt = json.loads(ctx.stage_checkpoint_path.read_text(encoding="utf-8"))
            ctx.completed_stages = ckpt.get("completed_stages", [])

            # P2：config_hash 校验 — 若 config 变更，全部重跑
            saved_hash = ckpt.get("config_hash", "")
            current_hash = _compute_config_hash(ctx.config)
            if saved_hash and saved_hash != current_hash:
                _logger.warning(
                    f"Config changed since last run (hash {saved_hash} → {current_hash}), "
                    f"re-running all stages"
                )
                ctx.completed_stages = []
            else:
                # P0.2：不可序列化 stage 强制重跑（bundle/model/trainer 无法从 checkpoint 恢复）
                replay = [s for s in ctx.completed_stages if s in _NON_SERIALIZABLE_STAGES]
                if replay:
                    ctx.completed_stages = [s for s in ctx.completed_stages if s not in _NON_SERIALIZABLE_STAGES]
                    _logger.info(
                        f"Resumed pipeline: re-running non-serializable stages {replay} "
                        f"(object refs lost on restart), skipping {len(ctx.completed_stages)} pure stages: "
                        f"{ctx.completed_stages}"
                    )
                else:
                    _logger.info(
                        f"Resumed pipeline, skipping {len(ctx.completed_stages)} completed stages: "
                        f"{ctx.completed_stages}"
                    )

            # 方案 A：从 checkpoint 恢复可序列化 stage 的产出（根治契约矛盾）。
            # _NON_SERIALIZABLE_STAGES 声明 validate/preflight/resolve 可跨进程跳过，
            # 其产出（report/route_config/task_spec/feature_spec 等）必须从 checkpoint 恢复，
            # 否则下游 stage 会因字段为 None 而失败。
            self._restore_stage_outputs(ctx)

            # Fallback：若可序列化 stage 的产出未从 checkpoint 恢复（无 checkpoint 或旧格式），
            # 从 completed_stages 移除以重跑，确保产出可用。对象引用 stage（load/build 等）
            # 已由 _NON_SERIALIZABLE_STAGES 强制重跑，无需检查。
            _serializable_output_checks = {
                "preflight": lambda c: c.report is not None and bool(c.route_config),
                "resolve": lambda c: c.task_spec is not None and c.feature_spec is not None,
            }
            for _s_name, _has_outputs in _serializable_output_checks.items():
                if _s_name in ctx.completed_stages and not _has_outputs(ctx):
                    ctx.completed_stages = [s for s in ctx.completed_stages if s != _s_name]
                    _logger.warning(
                        f"Stage '{_s_name}' in completed_stages but outputs not restored, "
                        f"re-running to regenerate outputs"
                    )

        # RFC-004 方案 F：try/finally 确保所有出口（成功/失败/异常）都释放资源
        try:
            for name, fn in self.stages:
                # P1：跳过已完成 stage
                if name in ctx.completed_stages:
                    # 补偿：validate 产出的 ctx.scene 是对象引用，跨进程不可恢复。
                    # 跳过 validate 时从注册表重建 scene 和 meta，避免下游 stage AttributeError。
                    if name == "validate" and ctx.scene is None:
                        if has_scene(ctx.config.scene.name):
                            ctx.scene = get_scene(ctx.config.scene.name)
                            ctx.meta = ctx.scene.meta()
                            _logger.info("Compensated ctx.scene after skipping validate")
                    # 补偿：validate 产出的标量字段（可从 config 重派生）。
                    # _restore_stage_outputs 优先从 checkpoint 恢复；此处为 fallback。
                    if name == "validate":
                        if not ctx.model_id:
                            ctx.model_id = ctx.config.scene.model_id
                        if not ctx.dataset:
                            ctx.dataset = ctx.config.scene.dataset
                        if not ctx.learning_mode:
                            ctx.learning_mode = ctx.config.scene.learning_mode
                    # 补偿：preflight 产出 set_seed 调用，跳过时需重新 set_seed 恢复 RNG 状态。
                    # 否则 resume 后 RNG 继承自上一次 run 的残留状态，导致 DataLoader shuffle
                    # 顺序与模型初始化非确定，val_acc 严重漂移（实测 0.982 → 0.129）。
                    if name == "preflight":
                        set_seed(ctx.config.trainer.seed,
                                 deterministic=ctx.config.trainer.deterministic)
                        _logger.info("Compensated set_seed after skipping preflight")
                    _logger.info(f"Skipping completed stage: {name}")
                    continue

                # P0-1: 在 stage 边界设置 callback active 状态
                # P5 P3-15：dry_run 下 ctx.trainer 为 None，需同时检查 ctx.callbacks
                callback_lists = []
                if ctx.trainer is not None and hasattr(ctx.trainer, "callbacks"):
                    callback_lists.append(ctx.trainer.callbacks)
                if getattr(ctx, "callbacks", None):
                    callback_lists.append(ctx.callbacks)
                for cb_list in callback_lists:
                    for cb in cb_list:
                        if isinstance(cb, StageAwareCallback):
                            cb.set_active(name)
                            _logger.debug(
                                "callback %s active=%s in stage=%s",
                                type(cb).__name__, cb.is_active(), name,
                            )

                # 修复（stage 边界日志 + stage duration Timer）：
                # 旧逻辑无 stage starting/completed 边界日志，Agent 无法追踪执行进度；
                # 旧逻辑 record_training_metric value=0.0 硬编码，stage duration 恒为 0。
                # 改为：用 Timer 包裹 fn(ctx)，回填实际耗时到 OTel 指标 + 加边界日志。
                _logger.info(f"[Stage {name}] starting")
                stage_timer = Timer()
                stage_timer.__enter__()
                try:
                    ctx = fn(ctx)
                    stage_timer.__exit__()
                    stage_duration = round(stage_timer.elapsed, 3)
                    # P1：记录完成 + 写 checkpoint
                    ctx.completed_stages.append(name)
                    self._write_checkpoint(ctx)
                    # P0.2: OBP 训练指标埋点（stage 完成时记录实际耗时）
                    record_training_metric(
                        f"senseframe.stage.{name}.duration_s",
                        value=stage_duration,
                        stage=name,
                        model_id=ctx.config.scene.model_id if hasattr(ctx, "config") else "",
                        dataset=ctx.config.scene.dataset if hasattr(ctx, "config") else "",
                    )
                    _logger.info(f"[Stage {name}] completed (duration={stage_duration}s)")
                except Exception as e:
                    try:
                        stage_timer.__exit__()
                    except Exception:
                        pass
                    _logger.error(f"[Stage {name}] failed: {e}", exc_info=True)
                    # 任务4：根据异常类型和 stage 上下文重新分类为具体异常类
                    # （OOMError/ModelBuildError/TrainingError/DataCorruptedError/
                    # CheckpointError/SaveError），使 Agent 可基于异常类型精确恢复。
                    actual_error = _classify_runtime_error(e, name)
                    # P1：记录失败 stage + 写 checkpoint
                    ctx.failed_stage = name
                    ctx.failed_error = repr(actual_error)
                    self._write_checkpoint(ctx, failed_stage=name)

                    # 异常时 traceback 落盘
                    import traceback as _tb
                    tb = _tb.format_exc()
                    if ctx.output:
                        ctx.output.status = "error"
                        ctx.output.error = str(e)
                        ctx.output.error_traceback = tb
                        ctx.output.error_code = classify_error(e, stage=name)
                    if ctx.output_dir and ctx.output_dir.exists():
                        try:
                            (ctx.output_dir / "FAILED").write_text(tb, encoding="utf-8")
                            for p in ctx.output_dir.glob("*.pth"):
                                p.unlink()
                            # P3-5：重命名为 FAILED_ 前缀，隔离失败目录，
                            # 避免失败目录的 manifest/checkpoint 干扰新 run 的扫描。
                            # 保留全部失败信息（checkpoint、metrics、logs、traceback）
                            # 供 resume 和诊断；更新 ctx.output_dir 指向新位置，
                            # 让 finally 分支的 _generate_manifest 写入隔离目录。
                            failed_dir = ctx.output_dir.parent / f"FAILED_{ctx.output_dir.name}"
                            if not failed_dir.exists():
                                ctx.output_dir.rename(failed_dir)
                                _logger.info(
                                    "P3-5: failed output_dir moved to %s", failed_dir
                                )
                                ctx.output_dir = failed_dir
                                if ctx.output is not None:
                                    ctx.output.output_dir = str(failed_dir)
                            else:
                                _logger.warning(
                                    "P3-5: FAILED_ dir already exists: %s, keeping original",
                                    failed_dir,
                                )
                        except Exception:
                            pass
                    # 关闭日志写入器（release_resources 也会关闭，此处保留双保险）
                    if ctx.log_writer is not None:
                        try:
                            ctx.log_writer.close()
                        except Exception:
                            pass
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                    return StageResult(context=ctx, skipped=False, error=e)

            return StageResult(context=ctx, skipped=False)
        finally:
            # RFC-004 方案 G：生成 manifest.json（产物溯源清单）
            # 在 release_resources 前，artifact_registry 已被各 stage 填充
            try:
                if ctx.output_dir is not None and ctx.output_dir.exists():
                    _generate_manifest(ctx)
            except Exception as e:
                _logger.warning(f"Failed to generate manifest.json: {e}")
            # RFC-004 方案 F：确定性资源释放（成功/失败/异常路径均执行）
            # 幂等：release_resources 内部对已 None 字段安全
            ctx.release_resources()
            # P4-4：资源释放后刷新 checkpoint，持久化 resources_released=True。
            # 旧代码 finally 块内无 checkpoint 刷新，导致 checkpoint 永远记录释放前的状态
            #（resources_released=False），跨进程无法验证资源是否已释放。
            try:
                self._write_checkpoint(ctx)
            except Exception as e:
                _logger.warning(f"Failed to write post-release checkpoint: {e}")
            # P5 P2-9：dry_run 模式清理临时目录
            if ctx.dry_run and ctx.output_dir is not None:
                import shutil
                try:
                    shutil.rmtree(ctx.output_dir)
                    _logger.info(f"dry_run cleanup: removed temp dir {ctx.output_dir}")
                except Exception as e:
                    _logger.warning(f"dry_run cleanup failed: {e}")

    def _write_checkpoint(self, ctx: PipelineContext, failed_stage: Optional[str] = None) -> None:
        """P1：写 stage checkpoint 到 output_dir/pipeline_checkpoint.json。

        P0.8 扩展：增加 stage_outputs 字段（仅可序列化字段），统一为 OP-4 真源。

        Args:
            ctx: PipelineContext（含 completed_stages、trial_id、output_dir）
            failed_stage: 若不为 None，表示在指定 stage 失败，记录到 checkpoint
        """
        if ctx.output_dir is None:
            return
        ckpt_path = ctx.output_dir / "pipeline_checkpoint.json"
        ctx.stage_checkpoint_path = ckpt_path
        data = {
            "pipeline_version": _PIPELINE_VERSION,
            "config_hash": _compute_config_hash(ctx.config),
            "completed_stages": ctx.completed_stages,
            "trial_id": ctx.trial_id,
            "timestamp": datetime.now().isoformat(),
            # P0.8：stage 输出快照（仅可序列化字段），作为 OP-4 唯一真源
            "stage_outputs": self._serialize_stage_outputs(ctx),
            # P4-4：资源释放状态（方案 F 持久化）。
            # finally 块的 release_resources() 后会追加一次 checkpoint 刷新，
            # 此时 trainer/module 已置 None，resources_released=True。
            # 若 checkpoint 在 try 块内写入（stage 执行后），此值为 False（尚未释放）。
            "resources_released": ctx.trainer is None and ctx.module is None,
        }
        if failed_stage:
            data["failed_stage"] = failed_stage
        try:
            ckpt_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 修复（5.6）：checkpoint 写入成功路径无日志，加 INFO 留痕
            # 旧逻辑只在失败时 warning，成功路径完全静默，无法追踪 checkpoint 落盘时机
            _logger.info(
                f"checkpoint written: {ckpt_path} "
                f"(completed_stages={len(ctx.completed_stages)}, "
                f"failed_stage={failed_stage})"
            )
        except Exception as e:
            _logger.warning(f"Failed to write pipeline checkpoint: {e}")

    def _serialize_stage_outputs(self, ctx: PipelineContext) -> Dict[str, Any]:
        """序列化 ctx 中可 JSON 化的轻量字段（P0.8，OP-4 真源扩展）。

        仅提取跨 stage 传递的"结果类"字段（str/int/float/bool/list/dict），
        跳过 torch/lightning 等不可序列化对象。
        final_eval/training_log 可能含 tensor，逐项 try/except。

        Returns:
            Dict[str, Any]: 可 JSON 序列化的 stage 输出快照
        """
        snapshot: Dict[str, Any] = {}
        # 简单可序列化字段（str/int/float/bool）
        simple_fields = [
            "model_id", "dataset", "learning_mode", "num_classes",
            "trial_id", "parent_trial_id",
            "training_duration_s", "best_model_path", "best_model_score",
            "best_epoch",  # Part 2：持久化 best_epoch
            "early_stopped", "failed_stage", "failed_error",
            "route_level",
        ]
        for name in simple_fields:
            val = getattr(ctx, name, None)
            if val is None:
                continue
            try:
                json.dumps(val)
                snapshot[name] = val
            except (TypeError, ValueError):
                # 不可序列化字段跳过（如 Path 对象转 str）
                if isinstance(val, Path):
                    snapshot[name] = str(val)
                else:
                    snapshot[name] = repr(val)

        # final_eval: dict，逐项 try/except
        if ctx.final_eval:
            serializable_eval: Dict[str, Any] = {}
            for k, v in ctx.final_eval.items():
                try:
                    json.dumps(v)
                    serializable_eval[k] = v
                except (TypeError, ValueError):
                    serializable_eval[k] = repr(v)
            snapshot["final_eval"] = serializable_eval

        # P5 P2-8：feedback 序列化（FeedbackResult dataclass，调用 to_dict() 后逐项处理）
        if ctx.feedback is not None:
            # P5 P2-7 阶段2：ctx.feedback 现在是 FeedbackResult dataclass
            _feedback_dict = ctx.feedback.to_dict()
            serializable_feedback: Dict[str, Any] = {}
            for k, v in _feedback_dict.items():
                try:
                    json.dumps(v)
                    serializable_feedback[k] = v
                except (TypeError, ValueError):
                    serializable_feedback[k] = repr(v)
            snapshot["feedback"] = serializable_feedback

        # P5 P2-8：training_log 序列化（list，逐项 try/except，可能含 tensor）
        if ctx.training_log:
            serializable_log: List[Any] = []
            for entry in ctx.training_log:
                try:
                    json.dumps(entry)
                    serializable_log.append(entry)
                except (TypeError, ValueError):
                    if hasattr(entry, "to_dict"):
                        try:
                            serializable_log.append(entry.to_dict())
                        except Exception:
                            serializable_log.append(repr(entry))
                    else:
                        serializable_log.append(repr(entry))
            snapshot["training_log"] = serializable_log

        # completed_stages: list[str]，必可序列化
        if ctx.completed_stages:
            snapshot["completed_stages"] = list(ctx.completed_stages)

        # preflight 产出持久化（可序列化 stage，跨进程恢复所需）
        if ctx.report is not None:
            snapshot["report"] = ctx.report.to_dict()
        if ctx.route_config:
            snapshot["route_config"] = ctx.route_config

        # resolve 产出持久化（可序列化 stage，跨进程恢复所需）
        # TaskSpec/FeatureSpec 有 to_dict/from_dict；其余为原生 dict
        if ctx.task_spec is not None and hasattr(ctx.task_spec, "to_dict"):
            snapshot["task_spec"] = ctx.task_spec.to_dict()
        if ctx.feature_spec is not None and hasattr(ctx.feature_spec, "to_dict"):
            snapshot["feature_spec"] = ctx.feature_spec.to_dict()
        if ctx.scene_info:
            snapshot["scene_info"] = ctx.scene_info
        if ctx.resolved:
            snapshot["resolved"] = ctx.resolved
        if ctx.lightning_params:
            snapshot["lightning_params"] = ctx.lightning_params
        if ctx.distributed_kwargs:
            snapshot["distributed_kwargs"] = ctx.distributed_kwargs

        return snapshot

    @staticmethod
    def _restore_stage_outputs(ctx: PipelineContext) -> None:
        """从 checkpoint 恢复可序列化 stage 的产出（根治契约矛盾）。

        _NON_SERIALIZABLE_STAGES 声明 validate/preflight/resolve 可跨进程跳过，
        但这些 stage 的产出（report/route_config/task_spec/feature_spec 等）
        若不从 checkpoint 恢复，下游 stage 会因字段为 None 而失败。

        本方法在 Pipeline.run 的 stage 循环前调用，从 pipeline_checkpoint.json
        的 stage_outputs 中恢复所有可序列化字段。仅恢复 ctx 中为 None/空的字段，
        不覆盖已由调用方或上游 stage 设置的值。

        对象引用（scene/meta/bundle/model 等）不可序列化，由补偿逻辑处理。
        """
        ckpt_path = ctx.stage_checkpoint_path
        if ckpt_path is None or not Path(ckpt_path).exists():
            return

        try:
            ckpt = json.loads(Path(ckpt_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        stage_outputs = ckpt.get("stage_outputs", {})
        if not stage_outputs:
            return

        # validate 标量产出
        if not ctx.model_id and stage_outputs.get("model_id"):
            ctx.model_id = stage_outputs["model_id"]
        if not ctx.dataset and stage_outputs.get("dataset"):
            ctx.dataset = stage_outputs["dataset"]
        if not ctx.learning_mode and stage_outputs.get("learning_mode"):
            ctx.learning_mode = stage_outputs["learning_mode"]

        # preflight 产出
        if ctx.report is None and stage_outputs.get("report"):
            from ...schemas import ResourceReport
            ctx.report = ResourceReport.from_dict(stage_outputs["report"])
        if not ctx.route_level and stage_outputs.get("route_level"):
            ctx.route_level = stage_outputs["route_level"]
        if not ctx.route_config and stage_outputs.get("route_config"):
            ctx.route_config = stage_outputs["route_config"]

        # resolve 产出
        if ctx.num_classes is None and stage_outputs.get("num_classes") is not None:
            ctx.num_classes = stage_outputs["num_classes"]
        if ctx.task_spec is None and stage_outputs.get("task_spec"):
            from ....core.task import TaskSpec
            ctx.task_spec = TaskSpec.from_dict(stage_outputs["task_spec"])
        if ctx.feature_spec is None and stage_outputs.get("feature_spec"):
            from ....core.features import FeatureSpec
            ctx.feature_spec = FeatureSpec.from_dict(stage_outputs["feature_spec"])
        if not ctx.scene_info and stage_outputs.get("scene_info"):
            ctx.scene_info = stage_outputs["scene_info"]
        if not ctx.resolved and stage_outputs.get("resolved"):
            ctx.resolved = stage_outputs["resolved"]
        if not ctx.lightning_params and stage_outputs.get("lightning_params"):
            ctx.lightning_params = stage_outputs["lightning_params"]
        if not ctx.distributed_kwargs and stage_outputs.get("distributed_kwargs"):
            ctx.distributed_kwargs = stage_outputs["distributed_kwargs"]

        _logger.info("Restored stage outputs from checkpoint")

    @classmethod
    def resume(cls, output_dir, pipeline_run=None) -> Tuple["Pipeline", List[str]]:
        """P1：从 output_dir 恢复 pipeline。

        读取 pipeline_checkpoint.json，返回默认 pipeline 与已完成的 stage 名列表。
        调用方可据此构造 PipelineContext 并设置 completed_stages +
        stage_checkpoint_path 以跳过已完成 stage。

        P1.2：支持传入 PipelineRun 实例，由 PipelineRun.phase 和
        PipelineRun.stages 状态驱动 completed_stages 恢复，实现 OP-3
        状态机集成。PipelineRun 优先于 checkpoint JSON。

        Args:
            output_dir: 之前的 pipeline 输出目录（含 pipeline_checkpoint.json）
            pipeline_run: PipelineRun 实例（OP 编排器提供），None 时走 JSON checkpoint

        Returns:
            (pipeline, completed_stages): 默认 pipeline 和已完成 stage 名列表

        Raises:
            FileNotFoundError: 若 checkpoint 不存在且 pipeline_run 为 None
        """
        pipeline = cls.default()

        if pipeline_run is not None:
            # P1.2: 从 PipelineRun 状态机恢复 completed_stages
            completed = [
                s.name for s in pipeline_run.stages
                if s.phase == "succeeded"
            ]
            return pipeline, completed

        # 向后兼容：从 JSON checkpoint 恢复
        output_dir = Path(output_dir)
        # P3-5：自动检测 FAILED_ 前缀。Pipeline.run 失败时将 output_dir 重命名
        # 为 FAILED_{原名}；resume 时若原路径不存在但 FAILED_ 候选存在，则从
        # 失败目录恢复（保留 checkpoint/metrics/logs 供续跑与诊断）。
        if not output_dir.exists():
            failed_candidate = output_dir.parent / f"FAILED_{output_dir.name}"
            if failed_candidate.exists():
                _logger.info(
                    "P3-5: detected FAILED_ prefix, resuming from %s", failed_candidate
                )
                output_dir = failed_candidate
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pipeline checkpoint found at {ckpt_path}")

        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        completed = ckpt.get("completed_stages", [])

        # 任务5：读取 failed_error 做诊断，输出恢复建议。
        # 不改变续跑行为（仍从 completed_stages 推断），仅增加诊断日志，
        # 帮助 Agent 理解上次失败原因并采取针对性措施。
        # failed_error 可能存于顶层（旧格式）或 stage_outputs 内（_serialize_stage_outputs）
        stage_outputs = ckpt.get("stage_outputs", {})
        failed_error = ckpt.get("failed_error") or stage_outputs.get("failed_error") or ""
        if failed_error:
            failed_error_lower = failed_error.lower()
            if "oom" in failed_error_lower or "outofmemory" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因 OOM 失败，建议降低 batch_size "
                    "(ctx.resolved['batch_size']) 或减少 num_workers 后重试"
                )
            if "datacorrupted" in failed_error_lower or "corrupt" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因数据损坏失败，建议检查数据集完整性 "
                    "(文件是否完整、未损坏) 后重试"
                )
            if "checkpoint" in failed_error_lower:
                _logger.warning(
                    "Resume: 上次运行因 checkpoint 问题失败，建议检查 checkpoint "
                    "文件是否损坏（可能需要删除旧 checkpoint 重新训练）"
                )

        return pipeline, completed


def run_pipeline(
    config: ExperimentConfig,
    pipeline: Optional[Pipeline] = None,
    *,
    pruner: Any = None,
    trial_id: str = "",
) -> TrainOutput:
    """Pipeline 入口（P1：失败时输出可恢复提示）。

    Agent 可传入自定义 pipeline，或使用默认 pipeline。
    默认 pipeline 执行完整的 9 stage 流程。
    P1 简化方案：失败时不自动 retry（重建 datamodule 过于复杂），
    而是输出 resume 提示到 stderr，引导用户从失败 stage 续跑。

    P1.1 Multi-fidelity 实时早停修复：新增 pruner/trial_id 参数。
    - pruner: Pruner Protocol 实例（None 时退化为旧路径，无实时剪枝）
    - trial_id: 当前 SP trial ID（传给 pruner.should_prune 用于跨 trial 比对）
    注入到 PipelineContext.pruner/trial_id，stage_build 读取并传给
    IntermediateMetricLogger，让每个 epoch end 调 should_prune 决定是否剪枝。

    Args:
        config: ExperimentConfig 实例
        pipeline: 自定义 pipeline（None 时使用默认）
        pruner: Pruner Protocol 实例（可选，启用实时早停）
        trial_id: SP trial ID（可选，配合 pruner 使用）

    Returns:
        TrainOutput
    """
    if pipeline is None:
        pipeline = Pipeline.default()

    ctx = PipelineContext(config=config)
    # P1.1: 实时早停注入 — agent/MethodRunner 通过参数传递 pruner/trial_id
    # stage_build 读取 ctx.pruner 注入到 IntermediateMetricLogger
    if pruner is not None:
        ctx.pruner = pruner
    if trial_id:
        ctx.trial_id = trial_id
    result = pipeline.run(ctx)

    # 确保 output 存在
    if ctx.output is None:
        ctx.output = TrainOutput(
            status="error" if result.error else "success",
            model_id=config.scene.model_id,
            dataset=config.scene.dataset,
            learning_mode=config.scene.learning_mode,
        )
        # P4-5：从 ctx 复制 feedback 到 TrainOutput，供 HPO tracker 消费。
        # 旧代码未复制，导致 hpo.py 硬编码 feedback={"status":"success"}，
        # 探索-反馈回路断裂（underfitting trial 被误标为 success）。
        if ctx.feedback is not None:
            ctx.output.feedback = ctx.feedback
        if result.error:
            ctx.output.error = str(result.error)

    # P1：失败时输出可恢复提示
    if result.error is not None and ctx.output_dir is not None:
        import sys
        failed_stage = ctx.failed_stage or "unknown"
        print(
            f"Pipeline failed at stage '{failed_stage}'. To resume: "
            f"Pipeline.resume('{ctx.output_dir}')",
            file=sys.stderr,
        )

    return ctx.output
