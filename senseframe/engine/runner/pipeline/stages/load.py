"""Stage 3: 加载数据 + 数据画像。"""
from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path

from ..context import PipelineContext, _logger
from ..stage_spec import stage
from .....observability import IncrementalLogWriter
from .....observability_otel import (
    record_training_metric,
    ML_DATA_LOAD_DURATION_S, ML_DATA_N_SAMPLES,
    ML_DATA_N_CLASSES, ML_DATA_IMBALANCE_RATIO,
)
from ...artifacts import sha256_str


def _compute_data_hash(data_root: str) -> str:
    """计算数据集目录的元数据哈希（任务2）。

    性能策略：不读取文件内容做全量 hash，只 hash 元数据
    （排序后的文件相对路径 + 文件大小 + 文件 mtime 的拼接）。
    大数据集（10k+ 文件）仍可在秒级完成。

    Args:
        data_root: 数据集根目录路径

    Returns:
        SHA256 十六进制字符串；目录不存在或为空时返回空字符串
    """
    root = Path(data_root)
    if not root.exists():
        return ""

    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            entries.append(f"{rel}|{stat.st_size}|{stat.st_mtime}")

    if not entries:
        return ""

    return sha256_str("\n".join(entries))


# ============================================================
# v2 差距 3：pretrain_checkpoint 加载（沉淀自 scripts/p3_eval_common.py:454）
# ============================================================
# CSI/EEG/Radio 数据集配置（用于跨模态预训练数据集解析）
_CSI_DATASETS = {"UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"}
_EEG_DATASETS = {"PhysioNet_MI", "BCI_Competition_IV_2a"}
_RADIO_DATASETS = {"RadioML2018"}


def _resolve_pretrain_source(pretrain_source: str, target_dataset: str) -> str | None:
    """解析 pretrain_source 到具体预训练数据集名。

    沉淀自 scripts/p3_eval_common.py:454 _resolve_pretrain_dataset。

    映射规则：
    - "none" → None（无预训练）
    - "csi_4datasets" + target 是 EEG/Radio → "NTU-Fi_HAR"（跨模态默认）
    - "csi_4datasets" + target 是 CSI → target_dataset（同模态）
    - "radioml" → "RadioML2018"
    - "eegmmidb" → "PhysioNet_MI"
    - 显式数据集名 → 直接返回
    - 未知 → None（fallback）
    """
    if pretrain_source == "none" or not pretrain_source:
        return None

    if pretrain_source == "csi_4datasets":
        if target_dataset in _EEG_DATASETS or target_dataset in _RADIO_DATASETS:
            return "NTU-Fi_HAR"
        return target_dataset

    if pretrain_source == "radioml":
        return "RadioML2018"

    if pretrain_source == "eegmmidb":
        return "PhysioNet_MI"

    # 显式数据集名
    if (pretrain_source in _CSI_DATASETS
            or pretrain_source in _EEG_DATASETS
            or pretrain_source in _RADIO_DATASETS):
        return pretrain_source

    _logger.warning(
        "Unknown pretrain_source=%s, treating as 'none'. "
        "Valid: none/csi_4datasets/radioml/eegmmidb/<dataset_name>",
        pretrain_source,
    )
    return None


def _load_pretrain_checkpoint(
    pretrain_dataset_name: str,
    output_dir,
) -> str | None:
    """加载预训练数据集的 checkpoint 路径。

    沉淀自 scripts/p3_eval_common.py:507 _load_pretrain_dataset（简化版）。

    Args:
        pretrain_dataset_name: 预训练数据集名
        output_dir: 输出目录（用于查找已有 checkpoint）

    Returns:
        checkpoint 路径，或 None（无可用 checkpoint）
    """
    # 在 output_dir/runs/ 下查找已训练的 pretrain checkpoint
    checkpoint_dir = Path(output_dir) / "runs" if output_dir else Path("runs")
    if not checkpoint_dir.exists():
        _logger.info(
            "pretrain checkpoint dir not found: %s, skipping pretrain load",
            checkpoint_dir,
        )
        return None

    # 查找匹配数据集名的 checkpoint（glob.escape 防御数据集名中的特殊字符）
    candidates = list(checkpoint_dir.glob(f"*{glob.escape(pretrain_dataset_name)}*.ckpt"))
    if not candidates:
        _logger.info(
            "no pretrain checkpoint found for dataset=%s in %s",
            pretrain_dataset_name, checkpoint_dir,
        )
        return None

    # 取最新的
    checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)
    _logger.info(
        "pretrain checkpoint loaded: %s (dataset=%s)",
        checkpoint_path, pretrain_dataset_name,
    )
    return str(checkpoint_path)


@stage(
    name="load",
    reads=["config", "scene", "dataset", "learning_mode", "output"],
    writes=["scene_kwargs", "bundle", "data_profile", "output_dir", "log_writer",
            "data_hash", "pretrain_checkpoint"],  # v2 差距 3：新增 pretrain_checkpoint
    description="Stage 3: 加载数据 + 数据画像 + 预训练 checkpoint",
)
def stage_load(ctx: PipelineContext) -> PipelineContext:
    """Stage 3: 加载数据 + 数据画像。"""
    import time as _time
    from .....core.profiler import DataProfiler

    # P1-4: stage 入口摘要日志
    _logger.info(
        "stage_load input: data_root=%s, dataset=%s, learning_mode=%s",
        ctx.config.scene.data_root or "(default)", ctx.dataset, ctx.learning_mode,
    )

    # P2-2: 数据加载耗时计时（包含 load_dataset + DataProfiler 全流程）
    load_timer = _time.time()

    # scene_kwargs 前置计算（供 load_dataset 使用，也供后续 resolve 读取）
    ctx.scene_kwargs = {"params": ctx.config.scene.params} if ctx.config.scene.params else {}

    # data_root 已由 SceneConfig.validate() 校验非空（YAML/CLI/env 三选一）
    data_root = ctx.config.scene.data_root

    ctx.bundle = ctx.scene.load_dataset(
        ctx.dataset, data_root, learning_mode=ctx.learning_mode,
        **ctx.scene_kwargs,
    )

    # v2 差距 3：pretrain_checkpoint 加载（scene.params.pretrain_source 触发）
    pretrain_source = None
    if ctx.config.scene.params:
        pretrain_source = ctx.config.scene.params.get("pretrain_source")
    if pretrain_source and pretrain_source != "none":
        pretrain_dataset = _resolve_pretrain_source(pretrain_source, ctx.dataset)
        if pretrain_dataset:
            ctx.pretrain_checkpoint = _load_pretrain_checkpoint(
                pretrain_dataset, ctx.config.output_dir,
            )
            _logger.info(
                "stage_load: pretrain_source=%s → dataset=%s, checkpoint=%s",
                pretrain_source, pretrain_dataset, ctx.pretrain_checkpoint,
            )
        else:
            ctx.pretrain_checkpoint = None
    else:
        ctx.pretrain_checkpoint = None

    # 任务2：计算 data_hash（数据集元数据哈希）。
    # 不读取文件内容做全量 hash，只 hash 元数据（路径+大小+mtime），
    # 性能远优于全量 hash，且能检测数据集变更/损坏/缺失。
    # 存入 ctx.data_hash，供 _generate_manifest 写入 manifest.data_hash。
    try:
        ctx.data_hash = _compute_data_hash(data_root)
    except Exception as e:
        _logger.warning(f"Failed to compute data_hash: {e}")
        ctx.data_hash = ""

    # 创建输出目录（在数据画像前，便于落盘）
    # M2 修复：model_id / dataset 来自配置（不可信），清洗后再拼接，避免路径逃逸
    # P5 P2-9：dry_run 使用 tempfile.mkdtemp 隔离临时产物，run() finally 中 rmtree
    import tempfile
    if ctx.dry_run:
        ctx.output_dir = Path(tempfile.mkdtemp(prefix="senseframe_dryrun_"))
        ctx.config.save_model = False
    else:
        from .....common.path_safe import sanitize_path_component
        safe_model_id = sanitize_path_component(ctx.model_id)
        safe_dataset = sanitize_path_component(ctx.dataset)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        ctx.output_dir = Path(ctx.config.output_dir).resolve() / f"{safe_model_id}_{safe_dataset}_{timestamp}_{pid}"
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
    if ctx.output:
        ctx.output.output_dir = str(ctx.output_dir)

    # 数据画像（落盘到 output_dir）
    # P2-4: 异常不再静默——DataProfiler 失败时记录 error 日志（含 traceback），
    # 仍降级为 None 不中断 stage_load，但留痕供 Agent 排查。
    try:
        profiler = DataProfiler(max_samples=500)
        # P0 修复：从 SceneMeta.modality 读取场景显式声明的数据模态，
        # 覆盖 profiler 的 shape 启发式（CSI (1,250,90) 与 image (1,H,W) 不可区分）
        modality_hint = getattr(ctx.meta, "modality", None)
        # P1 修复：透传 learning_mode，让 profile_bundle 按学习模式选择采样源
        # （自监督用 unsupervised 集，监督用 train 集），避免用 test 集做画像造成数据泄露。
        ctx.data_profile = profiler.profile_bundle(
            ctx.bundle, dataset_name=ctx.dataset, modality_hint=modality_hint,
            learning_mode=ctx.learning_mode,
        )
        if ctx.data_profile is not None:
            profile_path = ctx.output_dir / "data_profile.json"
            ctx.data_profile.save(profile_path)
            # RFC-004 方案 G：注册 data_profile 产物
            # P2-4: content_schema 补全 DataProfile 全部字段（与 profiler.py dataclass 对齐）
            ctx.register_artifact(
                "data_profile", profile_path,
                kind="profile", producer_stage="stage_load",
                content_schema={
                    "n_samples": "int",
                    "input_shape": "list",
                    "n_features": "int",
                    "n_classes": "int",
                    "class_distribution": "dict",
                    "imbalance_ratio": "float",
                    "missing_rate": "float",
                    "value_range": "list",
                    "mean": "float",
                    "std": "float",
                    "is_spatial": "bool",
                    "is_temporal": "bool",
                    "modality": "str",
                    "recommended_task_type": "str",
                    "recommended_loss": "str",
                    "recommended_metrics": "list",
                    "recommended_normalization": "str",
                    "dataset_name": "str",
                    "dtypes": "dict",
                    "feature_names": "list",
                    "nullable": "dict",
                    "shapes": "dict",
                    "profile_source": "str",
                },
            )
    except Exception as e:
        # P2-4: 留痕而非静默吞掉（旧逻辑 `except Exception: pass` 完全静默）
        _logger.error(f"DataProfiler failed: {e}", exc_info=True)
        ctx.data_profile = None

    # P2-2: 数据侧 OTel 指标埋点
    # 埋点失败不能中断 stage_load（用 try/except 兜底），OTel 未初始化时 no-op。
    load_duration = _time.time() - load_timer
    try:
        record_training_metric(
            ML_DATA_LOAD_DURATION_S,
            value=load_duration,
            stage="load",
            model_id=ctx.config.scene.model_id,
            dataset=ctx.config.scene.dataset,
        )
        if ctx.data_profile is not None:
            record_training_metric(
                ML_DATA_N_SAMPLES,
                value=ctx.data_profile.n_samples or 0,
                stage="load",
                model_id=ctx.config.scene.model_id,
                dataset=ctx.config.scene.dataset,
            )
            # n_classes 可能为 None（回归任务），OTel gauge 需数值，None 时记 0
            record_training_metric(
                ML_DATA_N_CLASSES,
                value=ctx.data_profile.n_classes or 0,
                stage="load",
                model_id=ctx.config.scene.model_id,
                dataset=ctx.config.scene.dataset,
            )
            # 类别不平衡比率：根因修复（P2）— 改用 DataProfile.imbalance_ratio
            # （自监督模式下 class_distribution 为空 → imbalance_ratio 为 None）。
            # 消费前做 None 守卫，避免对 None 求值。
            if ctx.data_profile.imbalance_ratio is not None:
                record_training_metric(
                    ML_DATA_IMBALANCE_RATIO,
                    value=float(ctx.data_profile.imbalance_ratio),
                    stage="load",
                    model_id=ctx.config.scene.model_id,
                    dataset=ctx.config.scene.dataset,
                )
    except Exception as e:
        _logger.debug("OTel data metrics recording failed: %s", e)

    # 增量日志写入器
    ctx.log_writer = IncrementalLogWriter(ctx.output_dir / "training_log.jsonl")

    # P1-4: stage 出口摘要日志
    train_samples = len(ctx.bundle.train) if ctx.bundle and getattr(ctx.bundle, "train", None) is not None else 0
    test_samples = len(ctx.bundle.test) if ctx.bundle and getattr(ctx.bundle, "test", None) is not None else 0
    _logger.info(
        "stage_load output: bundle.train_samples=%d, bundle.test_samples=%d, "
        "data_profile=%s, output_dir=%s, load_duration_s=%.3f",
        train_samples, test_samples,
        "present" if ctx.data_profile else "missing",
        str(ctx.output_dir) if ctx.output_dir else "None",
        load_duration,
    )

    return ctx
