"""Stage 5.5: 动态显存探测（子进程隔离）。

方案 B 完整实现：在 stage_build 构造模型后、stage_train 正式训练前，
在子进程中跑 1 个 batch 的前向，测量峰值显存（含参数+梯度+optimizer
state+激活），与 gpu_free_vram_mb 比较。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..context import PipelineContext, _logger
from ..stage_spec import stage
from ...errors import PreflightError


def _probe_json_default(obj):
    """JSON 序列化 dataclass / pydantic BaseModel 的 default handler。

    SceneParams / FeatureSpec 等 dataclass 无法被 json.dump 直接序列化，
    需通过 dataclasses.asdict 转换为 dict。
    P1 演进（2026-07-18）：兼容 pydantic v2 BaseModel（用 model_dump()）。
    """
    from dataclasses import asdict, is_dataclass
    # pydantic v2 BaseModel
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _run_probe_in_subprocess(params: Dict[str, Any]) -> Dict[str, Any]:
    """在子进程中执行显存探测，隔离 CUDA 计算不影响主进程。

    设计目的：probe 的 CUDA 计算会初始化 cuBLAS/cuDNN handle，这些全局状态
    无法被 set_seed 重置，会改变后续 trainer.fit() 首步的 CUDA 状态。
    子进程隔离让 probe 的 CUDA 上下文在子进程退出时销毁，主进程不受影响。

    通信协议：
    - 主进程 → 子进程：命令行参数（标量）+ JSON 文件（复杂参数）
    - 子进程 → 主进程：JSON stdout（成功含 measured_vram_mb，失败含 error）

    Args:
        params: 探测参数 dict，含 model_id/dataset/num_classes/batch_size 等

    Returns:
        探测结果 dict（含 measured_vram_mb/needed_vram_mb/free_vram_mb/ok/breakdown_mb）

    Raises:
        PreflightError: 子进程启动失败、超时、退出码非 0 或输出非 JSON
    """
    # 1. 构造命令行参数
    cmd = [
        sys.executable, "-m", "senseframe.engine.runner.probe_worker",
        "--model-id", str(params["model_id"]),
        "--dataset", str(params["dataset"]),
        "--num-classes", str(params["num_classes"]),
        "--learning-mode", str(params.get("learning_mode", "supervised")),
        "--batch-size", str(params["batch_size"]),
        "--precision", str(params.get("precision", "32")),
        "--optimizer", str(params.get("optimizer", "adam")),
        "--data-root", str(params["data_root"]),
        "--scene-name", str(params["scene_name"]),
    ]

    # 2. 复杂参数写入临时 JSON 文件（feature_spec, scene_kwargs, scene_info）
    params_file = None
    complex_params = {}
    if params.get("feature_spec"):
        complex_params["feature_spec"] = params["feature_spec"]
    if params.get("scene_kwargs"):
        complex_params["scene_kwargs"] = params["scene_kwargs"]
    if params.get("scene_info"):
        complex_params["scene_info"] = params["scene_info"]
    if complex_params:
        fd, params_file = tempfile.mkstemp(suffix=".json", prefix="probe_params_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(complex_params, f, ensure_ascii=False, default=_probe_json_default)
        cmd.extend(["--params-file", params_file])

    # 3. 启动子进程
    try:
        _logger.info(
            "probe subprocess: model_id=%s, dataset=%s, batch_size=%s",
            params["model_id"], params["dataset"], params["batch_size"],
        )
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 分钟超时
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        raise PreflightError(
            f"VRAM probe subprocess timed out (120s). "
            f"model_id={params['model_id']}, dataset={params['dataset']}"
        )
    except FileNotFoundError as e:
        raise PreflightError(
            f"VRAM probe subprocess 启动失败（Python 不可用？）: {e}"
        )
    finally:
        # 清理临时文件
        if params_file and os.path.exists(params_file):
            os.unlink(params_file)

    # 4. 解析子进程输出
    if proc.returncode != 0:
        # 子进程异常退出，尝试解析 stderr
        stderr_snippet = (proc.stderr or "")[:500]
        # 也尝试从 stdout 解析 error JSON
        try:
            error_result = json.loads(proc.stdout.strip())
            if "error" in error_result:
                raise PreflightError(
                    f"VRAM probe subprocess 失败: {error_result['error']} "
                    f"(type={error_result.get('error_type', 'unknown')})"
                )
        except (json.JSONDecodeError, ValueError):
            pass
        raise PreflightError(
            f"VRAM probe subprocess 退出码 {proc.returncode}: {stderr_snippet}"
        )

    # 5. 解析结果 JSON
    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        stdout_snippet = (proc.stdout or "")[:500]
        raise PreflightError(
            f"VRAM probe subprocess 输出非 JSON: {e}. "
            f"stdout 前 500 字符: {stdout_snippet}"
        )

    # 6. 检查 error 字段
    if "error" in result:
        raise PreflightError(
            f"VRAM probe subprocess 内部错误: {result['error']} "
            f"(type={result.get('error_type', 'unknown')})"
        )

    return result


@stage(
    name="probe_vram",
    reads=["model", "datamodule", "module", "resolved", "report",
           "dry_run", "route_level", "model_id"],
    writes=["vram_probe_result"],
    description="Stage 5.5: 动态显存探测（方案 B：前向+反向+optimizer step 测峰值显存）",
)
def stage_probe_vram(ctx: PipelineContext) -> PipelineContext:
    """Stage 5.5: 动态显存探测（子进程隔离）。

    方案 B 完整实现：在 stage_build 构造模型后、stage_train 正式训练前，
    在子进程中跑 1 个 batch 的前向，测量峰值显存（含参数+梯度+optimizer
    state+激活），与 gpu_free_vram_mb 比较。

    子进程隔离（2026-07-11）：probe 在独立 Python 进程中运行，子进程退出时
    CUDA 上下文销毁，主进程的 CUDA 状态不受影响。主进程 trainer.fit() 首步
    就是进程中首次 CUDA 计算，等同无 probe 路径（N0 基线），不需要 GPU warmup。

    与现有三层防御的关系：
    - 第一层（stage_resolve.preflight_check）：静态粗筛，快速失败
    - 第二层（本 stage）：动态精确探测，给 batch_size 建议
    - 第三层（stage_train._fit_with_oom_fallback）：运行时兜底

    跳过条件（写 vram_probe_result=None）：
    - dry_run 模式（无实际训练，无需探测）
    - 非 CUDA 路由（CPU/MPS 无 CUDA 显存测量 API）
    - ctx.model 或 ctx.datamodule 为 None（无法探测）
    """
    # P1-4: stage 入口摘要日志
    _logger.info(
        "stage_probe_vram input: dry_run=%s, has_cuda=%s, route_level=%s, "
        "batch_size=%s, precision=%s",
        ctx.dry_run,
        ctx.report.has_cuda if ctx.report else False,
        ctx.route_level,
        ctx.resolved.get("batch_size") if ctx.resolved else None,
        ctx.resolved.get("precision") if ctx.resolved else None,
    )

    # 跳过条件 1：dry_run 模式无实际训练，探测无意义
    if ctx.dry_run:
        ctx.vram_probe_result = {"skipped": "dry_run", "measured_vram_mb": None}
        _logger.info("stage_probe_vram: skipped (dry_run mode)")
        return ctx

    # 跳过条件 2：非 CUDA 路由（CPU/MPS 无 CUDA 显存测量 API）
    has_cuda = ctx.report.has_cuda if ctx.report else False
    if not has_cuda:
        ctx.vram_probe_result = {"skipped": "no_cuda", "measured_vram_mb": None}
        _logger.info("stage_probe_vram: skipped (no CUDA, route_level=%s)", ctx.route_level)
        return ctx

    # 跳过条件 3：模型或 datamodule 缺失（无法构造探测输入）
    if ctx.model is None or ctx.datamodule is None:
        ctx.vram_probe_result = {"skipped": "missing_model_or_data", "measured_vram_mb": None}
        _logger.warning(
            "stage_probe_vram: skipped (model=%s, datamodule=%s)",
            ctx.model is not None, ctx.datamodule is not None,
        )
        return ctx

    # 子进程隔离：probe 在独立进程中运行，主进程不执行任何 CUDA 计算，
    # 因此不需要 set_seed / RNG 保存恢复 / 模式保存恢复 / empty_cache。
    # 主进程的 CUDA 状态完全干净，trainer.fit() 首步就是首次 CUDA 计算。

    # 构造子进程探测参数
    from .....common.paths import resolve_data_root
    data_root = ctx.config.scene.data_root
    try:
        data_root = str(resolve_data_root(data_root))
    except FileNotFoundError:
        pass  # 子进程会报告具体错误

    # 序列化 feature_spec（如果是 dataclass，转 dict）
    feature_spec_dict = None
    if ctx.feature_spec is not None:
        try:
            from dataclasses import asdict
            feature_spec_dict = asdict(ctx.feature_spec)
        except Exception:
            feature_spec_dict = None

    probe_params = {
        "model_id": ctx.model_id,
        "dataset": ctx.dataset,
        "num_classes": ctx.num_classes,
        "learning_mode": ctx.learning_mode,
        "batch_size": ctx.resolved.get("batch_size", 64),
        "precision": ctx.resolved.get("precision", "32"),
        "optimizer": ctx.resolved.get("optimizer", "adam"),
        "data_root": data_root,
        "scene_name": ctx.config.scene.name,
        "feature_spec": feature_spec_dict,
        "scene_kwargs": ctx.scene_kwargs,
        "scene_info": ctx.scene_info,
    }

    # 方案 A：batch_size 自动适配（二分搜索）
    # 首次探测用当前 batch_size；若超限，按比例降低并重测，直到通过或达到迭代上限。
    # 每次迭代启动新子进程（子进程 CUDA 上下文已污染，不可复用）。
    result = _run_probe_in_subprocess(probe_params)
    original_batch_size = result.get("batch_size")
    max_iterations = 5
    iteration = 0

    while not result.get("ok") and result.get("measured_vram_mb") is not None:
        iteration += 1
        if iteration > max_iterations:
            # 超过迭代上限仍未通过，raise 给出最终建议
            batch_size = result["batch_size"]
            free_vram_mb = result["free_vram_mb"]
            needed_vram_mb = result["needed_vram_mb"]
            suggested_bs = max(4, int(batch_size * free_vram_mb / max(needed_vram_mb, 1)))
            ctx.vram_probe_result = result
            raise PreflightError(
                f"VRAM probe failed after {max_iterations} iterations: "
                f"measured {result['measured_vram_mb']}MB "
                f"(needed {needed_vram_mb:.1f}MB with 15% margin) > free {free_vram_mb}MB. "
                f"建议：batch_size {original_batch_size} → {suggested_bs}，"
                f"或改 CPU route（device=cpu），或减小模型（当前 {ctx.model_id}）。"
            )

        # 计算建议 batch_size（激活显存约与 batch_size 成正比）
        current_bs = result["batch_size"]
        free_vram_mb = result["free_vram_mb"]
        needed_vram_mb = result["needed_vram_mb"]
        suggested_bs = max(4, int(current_bs * free_vram_mb / max(needed_vram_mb, 1)))

        if suggested_bs >= current_bs:
            # 建议值未降低，无法通过降 batch_size 解决（固定显存部分已超限）
            ctx.vram_probe_result = result
            raise PreflightError(
                f"VRAM probe failed: measured {result['measured_vram_mb']}MB "
                f"(needed {needed_vram_mb:.1f}MB) > free {free_vram_mb}MB. "
                f"batch_size 已降至 {current_bs}，无法进一步降低。"
                f"建议：改 CPU route（device=cpu），或减小模型（当前 {ctx.model_id}）。"
            )

        # 应用新 batch_size，重新探测（新子进程）
        _logger.info(
            "stage_probe_vram: batch_size %d → %d (iteration %d/%d, "
            "measured=%.1fMB > free=%.1fMB)",
            current_bs, suggested_bs, iteration, max_iterations,
            result["measured_vram_mb"], free_vram_mb,
        )
        ctx.resolved["batch_size"] = suggested_bs
        if hasattr(ctx.datamodule, "batch_size"):
            ctx.datamodule.batch_size = suggested_bs
        probe_params["batch_size"] = suggested_bs
        result = _run_probe_in_subprocess(probe_params)

    # 探测通过（或本就通过），记录最终结果
    if result.get("batch_size") != original_batch_size:
        _logger.info(
            "stage_probe_vram: batch_size auto-fitted %d → %d "
            "(measured=%.1fMB, needed=%.1fMB, free=%.1fMB)",
            original_batch_size, result["batch_size"],
            result["measured_vram_mb"], result["needed_vram_mb"],
            result["free_vram_mb"],
        )

    ctx.vram_probe_result = result
    _logger.info(
        "stage_probe_vram: measured=%.1fMB, needed=%.1fMB (15%% margin), "
        "free=%.1fMB, ok=%s, batch_size=%s, precision=%s",
        result.get("measured_vram_mb", 0),
        result.get("needed_vram_mb", 0),
        result.get("free_vram_mb", 0),
        result.get("ok"),
        result.get("batch_size"),
        result.get("precision"),
    )

    return ctx
