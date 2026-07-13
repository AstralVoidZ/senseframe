"""
Phase 14.1.2：启动前预检模块。

从 runner.py 拆出，包含：
- 随机种子设置
- 启动前资源预检（数据存在性、显存、磁盘空间）
- 环境快照构建
- Lightning logger 构建
"""

import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.loggers import CSVLogger

from ...observability import setup_logging
from .errors import DataNotFoundError, PreflightError

logger = setup_logging()


def _get_dataset_dir_names(dataset: str) -> list:
    """从注册表获取数据集的可能目录名。"""
    from ...registry import get_dataset_spec, is_dataset_registered
    if is_dataset_registered(dataset):
        spec = get_dataset_spec(dataset)
        if spec.dir_names:
            return list(spec.dir_names)
    return [dataset]  # 回退：数据集名即目录名


def set_seed(seed: int, deterministic: bool = False):
    """
    设置随机种子，保证可复现性。

    Args:
        seed: 随机种子
        deterministic: 是否启用确定性算法（可能降低性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def preflight_check(
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    report,
    dataset: str,
    scene_name: Optional[str] = None,
    scene_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    启动前预检（C1）：数据存在性、GPU 显存、磁盘空间。

    在进入训练前尽早失败，避免训练到中途才报错浪费算力。

    Phase 8.1：CustomContainer 场景通过 manifest 预检样本文件存在性。
    """
    # data_root 从 config 或 SENSEFRAME_DATA_ROOT env 读（框架不猜测路径）
    from ...common.paths import resolve_data_root

    # 1. 数据集目录存在性
    # R-fix：custom 场景由 manifest 决定数据位置，跳过固定目录检查，
    # 改由下方 manifest 样本文件存在性检查覆盖。
    is_manifest_driven = (
        scene_name == "custom"
        and scene_params is not None
        and "manifest_path" in scene_params
    )
    if not is_manifest_driven:
        # P1-2: 用 resolve_data_root 候选列表探测，失败时错误信息含所有探测路径
        try:
            data_root = resolve_data_root(config.get("data_root"))
        except FileNotFoundError as e:
            # 候选都不存在：包装成 DataNotFoundError 抛出（保持 preflight 抛异常风格）
            raise DataNotFoundError(
                f"DATA_ROOT_NOT_FOUND: {e}"
            )
        expected_dirs = _get_dataset_dir_names(dataset)
        for d in expected_dirs:
            if not (data_root / d).exists():
                raise DataNotFoundError(
                    f"Dataset directory not found: {data_root / d} "
                    f"(dataset={dataset}, data_root={data_root})"
                )

        # 声明一致性检查：目录存在，但是否有声明扩展名的文件？
        # 不深入 loader 私有子目录结构（data/label、train_amp/test_amp 等），
        # 只做顶层递归检查——目录下找得到声明扩展名的文件即可。
        # 这样 dry-run 阶段就能暴露"扩展名不匹配"问题（如 .csv 实为 .npy），
        # 而非推迟到 load 阶段才报笼统的 FileNotFoundError。
        from ...registry import get_dataset_spec, is_dataset_registered
        spec = get_dataset_spec(dataset) if is_dataset_registered(dataset) else None
        if spec and spec.file_format and spec.file_format != "auto":
            _EXT_MAP = {"npy": ".npy", "csv": ".csv", "mat": ".mat", "image": ".jpg"}
            expected_ext = _EXT_MAP.get(spec.file_format)
            if expected_ext:
                import glob as _glob
                for d in expected_dirs:
                    search_dir = data_root / d
                    # 递归 glob：覆盖 nested（*/*.ext）和 flat（*.ext）两种 layout
                    if not _glob.glob(
                        str(search_dir / "**" / f"*{expected_ext}"),
                        recursive=True,
                    ):
                        raise DataNotFoundError(
                            f"Dataset directory exists but no '*{expected_ext}' files "
                            f"under {search_dir} (dataset={dataset}, "
                            f"file_format={spec.file_format}, layout={spec.layout}). "
                            f"请检查文件扩展名是否与 DatasetSpec.file_format 声明一致。"
                        )

    # Phase 8.1：CustomContainer 场景预检 manifest 样本文件
    if is_manifest_driven:
        from ...data.manifest import load_manifest
        manifest = load_manifest(scene_params["manifest_path"])
        # 抽样检查前 5 个样本文件存在性（全量检查太慢）
        for sample in manifest.samples[:5]:
            from ...data.manifest import _resolve_path
            p = _resolve_path(sample.path, manifest.data_root)
            if not p.exists():
                raise DataNotFoundError(
                    f"Manifest sample file not found: {p} "
                    f"(dataset={dataset}, manifest={scene_params['manifest_path']})"
                )

    # 2. GPU 显存预检（方案 C：防呆粗筛，精确预估由 stage_probe_vram 负责）
    # estimated_vram_mb 静态表实测高估 7.9 倍（2048 vs 260），
    # 不再作为精确预估，仅用于挡住"肯定跑不了"的情况。
    # 防呆阈值：min(estimated_vram_mb, 1024) * 1.2 — 只要空闲显存 > 1.2GB 就放行，
    # 让 stage_probe_vram 做精确判断。
    # 若空闲显存 < 1.2GB，即使最小模型也大概率 OOM，preflight 直接拦截。
    if report.has_cuda:
        free = report.gpu_free_vram_mb or 0
        # 防呆阈值：1GB（任何模型训练至少需要这么多）
        BLIND_THRESHOLD_MB = 1024
        if free < BLIND_THRESHOLD_MB:
            raise PreflightError(
                f"GPU free VRAM {free}MB < {BLIND_THRESHOLD_MB}MB blind threshold. "
                f"即使最小模型也大概率 OOM。建议：关闭其他 GPU 进程，或使用 CPU route。"
                f"（精确显存预估由 stage_probe_vram 在模型构造后执行）"
            )

    # 3. 磁盘剩余空间（至少 1GB）
    out_root = Path(config.get("output_dir") or "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(str(out_root)).free
    if free_disk < 1024 * 1024 * 1024:
        raise PreflightError(
            f"Disk free space {free_disk // 1024 // 1024}MB < 1024MB minimum "
            f"(output_dir={out_root})"
        )


def build_env_snapshot(resolved: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """构建环境快照（E3），用于可复现性追溯。"""
    import pytorch_lightning as pl
    return {
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "deterministic": resolved.get("deterministic", False),
        "seed": config.get("seed", 42),
    }


# ============================================================
# P1：结构化预检（CheckResult + 语义/模型/训练契约校验）
# ============================================================

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CheckResult:
    """单项预检结果。

    Agent 可基于 error_code 做程序化决策（如降级 batch_size、切换 route）。
    """
    name: str                          # 检查项名（如 "epochs_positive"）
    ok: bool                           # 是否通过
    severity: str = "info"             # info / warning / error
    detail: Any = None                 # 详细信息（字符串或 dict）
    error_code: Optional[str] = None   # 失败时的结构化错误码
    remediation: Optional[str] = None  # 修复建议（Agent 可读）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "detail": self.detail,
            "error_code": self.error_code,
            "remediation": self.remediation,
        }


def validate_config_semantics(
    trainer_config, n_samples: int = 0,
) -> List[CheckResult]:
    """P1-1：配置语义校验（跨字段逻辑约束）。

    TrainerConfig.validate() 已覆盖单字段校验（epochs>0/optimizer 合法等），
    本函数补充跨字段约束和数据集规模约束。

    Args:
        trainer_config: TrainerConfig 实例
        n_samples: 训练集样本数（0 表示未知，跳过 batch_size <= n_samples 检查）
    """
    checks = []
    t = trainer_config

    # 1. early_stopping < epochs（否则永远不触发）
    if t.early_stopping is not None and t.early_stopping >= t.epochs:
        checks.append(CheckResult(
            name="early_stopping_within_epochs",
            ok=False,
            severity="warning",
            detail=f"early_stopping={t.early_stopping} >= epochs={t.epochs}，早停永不触发",
            error_code="CONFIG_EARLY_STOPPING_USELESS",
            remediation=f"设置 early_stopping < {t.epochs} 或增大 epochs",
        ))
    else:
        checks.append(CheckResult(
            name="early_stopping_within_epochs",
            ok=True,
            severity="info",
            detail=f"early_stopping={t.early_stopping}, epochs={t.epochs}",
        ))

    # 2. batch_size <= n_samples（数据集规模约束）
    if n_samples > 0 and t.batch_size > n_samples:
        checks.append(CheckResult(
            name="batch_size_within_dataset",
            ok=False,
            severity="error",
            detail=f"batch_size={t.batch_size} > n_samples={n_samples}，每个 epoch 仅 1 step",
            error_code="CONFIG_BATCH_SIZE_TOO_LARGE",
            remediation=f"设置 batch_size <= {n_samples}",
        ))
    else:
        checks.append(CheckResult(
            name="batch_size_within_dataset",
            ok=True,
            severity="info",
            detail=f"batch_size={t.batch_size}, n_samples={n_samples if n_samples > 0 else 'unknown'}",
        ))

    # 3. scheduler 与 epochs 兼容（cosine/step 需 >= 1 epoch）
    if t.scheduler in ("cosine", "step") and t.epochs < 1:
        checks.append(CheckResult(
            name="scheduler_epochs_compatible",
            ok=False,
            severity="error",
            detail=f"scheduler='{t.scheduler}' 需要 epochs >= 1，实际 epochs={t.epochs}",
            error_code="CONFIG_SCHEDULER_INCOMPATIBLE",
            remediation="设置 epochs >= 1 或 scheduler=None",
        ))
    else:
        checks.append(CheckResult(
            name="scheduler_epochs_compatible",
            ok=True,
            severity="info",
            detail=f"scheduler={t.scheduler}, epochs={t.epochs}",
        ))

    # 4. deterministic 模式需要 cuDNN 支持（仅 warning，运行期才真正失败）
    if t.deterministic and not torch.cuda.is_available():
        checks.append(CheckResult(
            name="deterministic_cuda_available",
            ok=False,
            severity="warning",
            detail="deterministic=True 但 CUDA 不可用，确定性算法仅对 CUDA 有效",
            error_code="CONFIG_DETERMINISTIC_NO_CUDA",
            remediation="设置 deterministic=False 或启用 CUDA",
        ))
    else:
        checks.append(CheckResult(
            name="deterministic_cuda_available",
            ok=True,
            severity="info",
            detail=f"deterministic={t.deterministic}, cuda_available={torch.cuda.is_available()}",
        ))

    return checks


def validate_training_contract(
    task_spec, loss_name: str, metrics: List[str],
    optimizer: str, scheduler: Optional[str],
    epochs: int, early_stopping: Optional[int],
) -> List[CheckResult]:
    """P1-3：训练契约校验（task_spec 与 loss/metric/optimizer/scheduler 一致性）。

    Args:
        task_spec: TaskSpec 实例（含 task_type/num_classes 等字段）
        loss_name: loss 名称（如 "cross_entropy"）
        metrics: 评估指标列表
        optimizer: 优化器名
        scheduler: 调度器名（None 表示不用）
        epochs: 训练轮数
        early_stopping: 早停 patience（None 表示不启用）
    """
    checks = []
    # TaskSpec 字段名是 task_type（非 type），getattr 错误属性名会静默返回默认值
    task_type = getattr(task_spec, "task_type", "classification")

    # 1. loss 与 task_type 匹配
    _LOSS_TASK_MAP = {
        "classification": {"cross_entropy", "ce", "focal"},
        "regression": {"mse", "mae", "huber"},
        "anomaly_detection": {"mse", "bce"},
    }
    expected_losses = _LOSS_TASK_MAP.get(task_type, set())
    if expected_losses and loss_name.lower() not in expected_losses:
        checks.append(CheckResult(
            name="loss_task_match",
            ok=False,
            severity="warning",
            detail=f"loss='{loss_name}' 与 task_type='{task_type}' 可能不匹配，建议: {expected_losses}",
            error_code="TRAINING_LOSS_TASK_MISMATCH",
            remediation=f"使用 {expected_losses} 之一的 loss",
        ))
    else:
        checks.append(CheckResult(
            name="loss_task_match",
            ok=True,
            severity="info",
            detail=f"loss={loss_name}, task_type={task_type}",
        ))

    # 2. metrics 与 task_type 匹配
    _METRIC_TASK_MAP = {
        "classification": {"accuracy", "macro_f1", "micro_f1", "precision", "recall"},
        "regression": {"mse", "mae", "rmse", "r2"},
        "anomaly_detection": {"precision", "recall", "f1", "auc"},
    }
    expected_metrics = _METRIC_TASK_MAP.get(task_type, set())
    if expected_metrics:
        unsupported = [m for m in metrics if m.lower() not in expected_metrics]
        if unsupported:
            checks.append(CheckResult(
                name="metrics_task_match",
                ok=False,
                severity="warning",
                detail=f"metrics={unsupported} 与 task_type='{task_type}' 可能不匹配，建议: {expected_metrics}",
                error_code="TRAINING_METRIC_TASK_MISMATCH",
                remediation=f"使用 {expected_metrics} 之一的 metrics",
            ))
        else:
            checks.append(CheckResult(
                name="metrics_task_match",
                ok=True,
                severity="info",
                detail=f"metrics={metrics}, task_type={task_type}",
            ))
    else:
        checks.append(CheckResult(
            name="metrics_task_match",
            ok=True,
            severity="info",
            detail=f"metrics={metrics}, task_type={task_type}（无映射约束）",
        ))

    # 3. early_stopping < epochs
    if early_stopping is not None and early_stopping >= epochs:
        checks.append(CheckResult(
            name="early_stopping_within_epochs",
            ok=False,
            severity="warning",
            detail=f"early_stopping={early_stopping} >= epochs={epochs}，早停永不触发",
            error_code="TRAINING_EARLY_STOPPING_USELESS",
            remediation=f"设置 early_stopping < {epochs}",
        ))
    else:
        checks.append(CheckResult(
            name="early_stopping_within_epochs",
            ok=True,
            severity="info",
            detail=f"early_stopping={early_stopping}, epochs={epochs}",
        ))

    return checks


def validate_model_contract(
    model, num_classes: int, sample_batch,
) -> List[CheckResult]:
    """P1-2：模型契约校验（参数量 + 前向 + 输出形状 + backward）。

    在 CPU 上执行轻量前向校验，不启动 Lightning Trainer，不初始化 CUDA。

    Args:
        model: nn.Module 实例
        num_classes: 期望的输出类别数
        sample_batch: 样本 batch（torch.Tensor，已扩展到 batch_size）
    """
    import torch as _torch
    checks = []

    # 1. 参数量在合理范围
    n_params = sum(p.numel() for p in model.parameters())
    # WiFi CSI 场景典型范围：100K - 50M
    ok_params = 1e5 <= n_params <= 5e7
    checks.append(CheckResult(
        name="param_count_reasonable",
        ok=ok_params,
        severity="warning" if not ok_params else "info",
        detail=f"n_params={n_params:,}（合理范围 100K-50M）",
        error_code=None if ok_params else "MODEL_PARAM_COUNT_UNUSUAL",
        remediation=None if ok_params else "检查模型架构是否合理（过小/过大）",
    ))

    # 2. 前向传播测试
    model.eval()
    batch_size = sample_batch.shape[0]
    try:
        with _torch.no_grad():
            output = model(sample_batch)
        ok_fwd = True
        fwd_detail = f"output_shape={tuple(output.shape)}"
    except Exception as e:
        ok_fwd = False
        fwd_detail = f"forward failed: {e}"
        checks.append(CheckResult(
            name="forward_pass",
            ok=False,
            severity="error",
            detail=fwd_detail,
            error_code="MODEL_FORWARD_FAILED",
            remediation="检查模型架构与输入 shape 是否匹配",
        ))
        # 前向失败则后续校验无意义
        return checks

    checks.append(CheckResult(
        name="forward_pass",
        ok=True,
        severity="info",
        detail=fwd_detail,
    ))

    # 3. 输出形状匹配 num_classes
    expected_shape = (batch_size, num_classes)
    actual_shape = tuple(output.shape)
    ok_shape = actual_shape == expected_shape
    checks.append(CheckResult(
        name="output_shape_match",
        ok=ok_shape,
        severity="error" if not ok_shape else "info",
        detail=f"output.shape={actual_shape}, expected={expected_shape}",
        error_code="MODEL_OUTPUT_SHAPE_MISMATCH" if not ok_shape else None,
        remediation=(
            f"检查模型最后一层输出维度是否为 num_classes={num_classes}"
            if not ok_shape else None
        ),
    ))

    # 4. backward 1 step（验证梯度可回传）
    try:
        model.train()
        output = model(sample_batch)
        target = _torch.randint(0, num_classes, (batch_size,))
        loss = _torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        ok_bw = True
        bw_detail = f"loss={loss.item():.4f}"
    except Exception as e:
        ok_bw = False
        bw_detail = f"backward failed: {e}"
    finally:
        model.eval()

    checks.append(CheckResult(
        name="backward_pass",
        ok=ok_bw,
        severity="error" if not ok_bw else "info",
        detail=bw_detail,
        error_code="MODEL_BACKWARD_FAILED" if not ok_bw else None,
        remediation="检查模型是否有不可导操作" if not ok_bw else None,
    ))

    return checks


# ============================================================
# P2：增强预检（数据契约 / 依赖契约 / 资源契约 / 可复现性）
# ============================================================


@dataclass
class PreflightReport:
    """P3：统一预检报告，聚合所有 CheckResult。

    将 _cmd_dry_run 中分散的 dict 字段（config_semantics / dependency_contract /
    reproducibility / resource_contract / dynamic_validation / training_contract /
    data_contract / env_snapshot / plan）统一为结构化 dataclass。

    Agent 可基于 status 和 errors() 做程序化决策。
    """
    status: str = "ok"  # "ok" / "blocked" / "warning"
    checks: List[CheckResult] = field(default_factory=list)
    env_snapshot: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    # 分类存储（按检查类别）
    categories: Dict[str, List[CheckResult]] = field(default_factory=dict)

    def add_category(self, name: str, results: List[CheckResult]) -> None:
        """添加一类检查结果。"""
        self.categories[name] = results
        self.checks.extend(results)
        # 更新整体状态：有 error 级失败则 blocked
        for c in results:
            if not c.ok and c.severity == "error":
                self.status = "blocked"
            elif not c.ok and c.severity == "warning" and self.status == "ok":
                self.status = "warning"

    def errors(self) -> List[CheckResult]:
        """返回所有 error 级失败项。"""
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    def warnings(self) -> List[CheckResult]:
        """返回所有 warning 级失败项。"""
        return [c for c in self.checks if not c.ok and c.severity == "warning"]

    def has_blocking_errors(self) -> bool:
        """是否有阻断性错误。"""
        return len(self.errors()) > 0

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的 dict。

        保留向后兼容：同时输出分类字段（config_semantics 等）和统一 checks 数组。
        """
        result = {
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
            "env_snapshot": self.env_snapshot,
            "plan": self.plan,
            "summary": {
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c.ok),
                "warnings": len(self.warnings()),
                "errors": len(self.errors()),
            },
        }
        # 分类输出（向后兼容 _cmd_dry_run 的字段名）
        _CATEGORY_FIELD_MAP = {
            "config_semantics": "config_semantics",
            "dependency_contract": "dependency_contract",
            "reproducibility": "reproducibility",
            "resource_contract": "resource_contract",
            "model_contract": "dynamic_validation",  # 模型契约归入 dynamic_validation
            "training_contract": "training_contract",
            "data_contract": "data_contract",
        }
        for cat_name, results in self.categories.items():
            field_name = _CATEGORY_FIELD_MAP.get(cat_name, cat_name)
            if field_name == "dynamic_validation":
                # dynamic_validation 保持原结构（含 status/checks/detail）
                # 修复 all([]) 空真值 bug：results 为空时（动态校验异常失败、无检查项）
                # all([]) 返回 True 导致 status 错误设为 "passed"。
                # 修复后：results 非空且全部 ok 才为 "passed"，否则 "failed"
                result[field_name] = {
                    "status": "passed" if results and all(c.ok for c in results) else "failed",
                    "checks": [c.to_dict() for c in results],
                }
            else:
                result[field_name] = [c.to_dict() for c in results]
        return result


def validate_data_contract(
    profile, task_spec, learning_mode: str,
) -> List[CheckResult]:
    """P2-1：数据契约校验（基于 DataProfile）。

    在 stage_load 生成 DataProfile 后调用，校验数据与任务契约一致性。

    Args:
        profile: DataProfile 实例
        task_spec: TaskSpec 实例
        learning_mode: supervised / self_supervised
    """
    checks = []

    # 1. 训练集非空
    ok_nonempty = profile.n_samples > 0
    checks.append(CheckResult(
        name="train_set_nonempty",
        ok=ok_nonempty,
        severity="error" if not ok_nonempty else "info",
        detail=f"n_samples={profile.n_samples}",
        error_code="DATA_EMPTY" if not ok_nonempty else None,
        remediation="检查 data_root 路径和数据集加载逻辑" if not ok_nonempty else None,
    ))

    # 2. 类别分布覆盖 [0, num_classes)（仅 supervised 模式）
    if learning_mode == "supervised" and profile.class_distribution:
        num_classes = getattr(task_spec, "num_classes", 0)
        # class_distribution 的 key 可能是 str 或 int
        covered = set()
        for k in profile.class_distribution.keys():
            try:
                covered.add(int(k))
            except (ValueError, TypeError):
                covered.add(k)
        expected = set(range(num_classes))
        missing = expected - covered
        ok_coverage = len(missing) == 0
        checks.append(CheckResult(
            name="class_coverage",
            ok=ok_coverage,
            severity="warning" if not ok_coverage else "info",
            detail=f"missing classes: {sorted(missing)}" if missing else f"all {num_classes} classes covered",
            error_code="DATA_CLASS_MISSING" if not ok_coverage else None,
            remediation=f"补充缺失类别数据或调整 num_classes" if not ok_coverage else None,
        ))
    else:
        checks.append(CheckResult(
            name="class_coverage",
            ok=True,
            severity="info",
            detail=f"learning_mode={learning_mode}，跳过类别覆盖检查",
        ))

    # 3. 缺失率 < 30%
    ok_missing = profile.missing_rate < 0.3
    checks.append(CheckResult(
        name="missing_rate_acceptable",
        ok=ok_missing,
        severity="warning" if not ok_missing else "info",
        detail=f"missing_rate={profile.missing_rate:.2%}（阈值 30%）",
        error_code="DATA_MISSING_RATE_HIGH" if not ok_missing else None,
        remediation="数据清洗或填充缺失值" if not ok_missing else None,
    ))

    # 4. 类别不平衡（imbalance_ratio > 10 时 warning）
    if profile.imbalance_ratio is not None:
        ok_balanced = profile.imbalance_ratio <= 10
        checks.append(CheckResult(
            name="class_balance",
            ok=ok_balanced,
            severity="warning" if not ok_balanced else "info",
            detail=f"imbalance_ratio={profile.imbalance_ratio:.2f}（阈值 10）",
            error_code="DATA_CLASS_IMBALANCED" if not ok_balanced else None,
            remediation="使用 class_weight 或过采样/欠采样" if not ok_balanced else None,
        ))
    else:
        checks.append(CheckResult(
            name="class_balance",
            ok=True,
            severity="info",
            detail="imbalance_ratio=None（自监督或无标签数据）",
        ))

    # 5. modality 识别（CSI 场景应为 csi 或 temporal）
    ok_modality = profile.modality != "unknown"
    checks.append(CheckResult(
        name="modality_identified",
        ok=ok_modality,
        severity="warning" if not ok_modality else "info",
        detail=f"modality={profile.modality}",
        error_code="DATA_MODALITY_UNKNOWN" if not ok_modality else None,
        remediation="检查数据加载逻辑是否正确识别数据类型" if not ok_modality else None,
    ))

    return checks


def validate_dependency_contract(
    config, export_formats: List[str] = None,
) -> List[CheckResult]:
    """P2-2：依赖契约校验（import / 包可用性）。

    在 stage_preflight 中调用，提前发现缺失依赖。

    Args:
        config: ExperimentConfig 实例
        export_formats: 导出格式列表（如 ["onnx", "torchscript"]）
    """
    checks = []
    import importlib

    # 1. logger 依赖
    logger_type = getattr(config.trainer, "logger", "csv")
    if logger_type == "tensorboard":
        try:
            importlib.import_module("tensorboard")
            ok_tb = True
            tb_detail = "tensorboard 可用"
        except ImportError:
            ok_tb = False
            tb_detail = "tensorboard 未安装"
        checks.append(CheckResult(
            name="logger_dependency",
            ok=ok_tb,
            severity="error" if not ok_tb else "info",
            detail=tb_detail,
            error_code="DEP_LOGGER_MISSING" if not ok_tb else None,
            remediation="pip install tensorboard 或改用 logger=csv" if not ok_tb else None,
        ))
    elif logger_type == "wandb":
        try:
            importlib.import_module("wandb")
            ok_wb = True
            wb_detail = "wandb 可用"
        except ImportError:
            ok_wb = False
            wb_detail = "wandb 未安装"
        checks.append(CheckResult(
            name="logger_dependency",
            ok=ok_wb,
            severity="error" if not ok_wb else "info",
            detail=wb_detail,
            error_code="DEP_LOGGER_MISSING" if not ok_wb else None,
            remediation="pip install wandb 或改用 logger=csv" if not ok_wb else None,
        ))
    else:
        checks.append(CheckResult(
            name="logger_dependency",
            ok=True,
            severity="info",
            detail=f"logger={logger_type}，无额外依赖",
        ))

    # 2. export 格式依赖
    if export_formats:
        _EXPORT_DEPS = {
            "onnx": "onnx",
            "torchscript": None,  # torch 自带
            "state_dict": None,   # torch 自带
        }
        for fmt in export_formats:
            dep = _EXPORT_DEPS.get(fmt)
            if dep is None:
                checks.append(CheckResult(
                    name=f"export_dep_{fmt}",
                    ok=True,
                    severity="info",
                    detail=f"export={fmt}，无额外依赖（torch 自带）",
                ))
            else:
                try:
                    importlib.import_module(dep)
                    ok_dep = True
                    dep_detail = f"{dep} 可用"
                except ImportError:
                    ok_dep = False
                    dep_detail = f"{dep} 未安装"
                checks.append(CheckResult(
                    name=f"export_dep_{fmt}",
                    ok=ok_dep,
                    severity="warning" if not ok_dep else "info",
                    detail=dep_detail,
                    error_code="DEP_EXPORT_MISSING" if not ok_dep else None,
                    remediation=f"pip install {dep}" if not ok_dep else None,
                ))

    # 3. deterministic 模式需要 cuDNN
    if getattr(config.trainer, "deterministic", False):
        try:
            importlib.import_module("torch.backends.cudnn")
            ok_det = True
            det_detail = "cudnn 可用"
        except ImportError:
            ok_det = False
            det_detail = "cudnn 不可用"
        checks.append(CheckResult(
            name="deterministic_dependency",
            ok=ok_det,
            severity="warning" if not ok_det else "info",
            detail=det_detail,
            error_code="DEP_DETERMINISTIC_MISSING" if not ok_det else None,
            remediation="安装 CUDA 版 torch 或关闭 deterministic" if not ok_det else None,
        ))

    return checks


def validate_resource_contract(
    report, route_config: Dict[str, Any],
    vram_probe_result: Optional[Dict[str, Any]] = None,
    n_samples: int = 0, batch_size: int = 0,
) -> List[CheckResult]:
    """P2-3：资源契约校验（显存 / CPU / 磁盘）。

    在 stage_probe_vram 后调用，使用 probe 精确值。

    Args:
        report: ResourceReport 实例（含 cpu_count/gpu 信息）
        route_config: 路由配置（含 num_workers/precision 等）
        vram_probe_result: stage_probe_vram 的结果（含 measured_vram_mb/free_vram_mb/ok）
        n_samples: 训练集样本数
        batch_size: 配置的 batch_size
    """
    checks = []

    # 1. GPU 显存（使用 probe 精确值，取代旧 1GB 防呆）
    has_cuda = getattr(report, "has_cuda", False)
    if has_cuda and vram_probe_result:
        ok_vram = vram_probe_result.get("ok", False)
        measured = vram_probe_result.get("measured_vram_mb", 0)
        needed = vram_probe_result.get("needed_vram_mb", 0)
        free = vram_probe_result.get("free_vram_mb", 0)
        checks.append(CheckResult(
            name="vram_sufficient",
            ok=ok_vram,
            severity="error" if not ok_vram else "info",
            detail=f"measured={measured}MB, needed={needed}MB, free={free}MB",
            error_code="OOM_ERROR" if not ok_vram else None,
            remediation=f"降低 batch_size（当前 {batch_size}）或 precision=16-mixed" if not ok_vram else None,
        ))
    elif has_cuda and not vram_probe_result:
        # GPU 可用但 probe 未运行（如 dry-run 跳过 probe stage）
        checks.append(CheckResult(
            name="vram_sufficient",
            ok=True,
            severity="info",
            detail="GPU 可用但 probe 未运行（dry-run 模式）",
        ))
    else:
        checks.append(CheckResult(
            name="vram_sufficient",
            ok=True,
            severity="info",
            detail="CPU 模式，无显存约束",
        ))

    # 2. num_workers <= CPU 核数
    num_workers = route_config.get("num_workers", 0) if route_config else 0
    cpu_count = getattr(report, "cpu_count", 1) or 1
    ok_workers = num_workers <= cpu_count
    checks.append(CheckResult(
        name="num_workers_reasonable",
        ok=ok_workers,
        severity="warning" if not ok_workers else "info",
        detail=f"num_workers={num_workers}, cpu_count={cpu_count}",
        error_code="RESOURCE_WORKERS_EXCESS" if not ok_workers else None,
        remediation=f"设置 num_workers <= {cpu_count}" if not ok_workers else None,
    ))

    # 3. 磁盘空间（估算 checkpoint 大小）
    # 粗略估算：checkpoint ~ model_params * 4 bytes * 3（模型/优化器/scheduler）
    # 这里仅做 warning 级别检查（实际磁盘检查已在 preflight_check 中）
    if batch_size > 0 and n_samples > 0:
        # 简单估算训练时长：epochs ~ 150000 / n_samples
        est_epochs = max(30, min(300, 150000 // n_samples))
        est_total_steps = (n_samples // batch_size) * est_epochs
        checks.append(CheckResult(
            name="training_scale_estimate",
            ok=True,
            severity="info",
            detail=f"est_epochs={est_epochs}, est_total_steps={est_total_steps}",
        ))

    return checks


def validate_reproducibility(
    config, report,
) -> List[CheckResult]:
    """P2-4：可复现性检查（seed / deterministic / 版本记录）。

    在 stage_preflight 中调用，确保训练可复现。

    Args:
        config: ExperimentConfig 实例
        report: ResourceReport 实例
    """
    checks = []

    # 1. seed 已设置
    seed = getattr(config.trainer, "seed", None)
    ok_seed = seed is not None
    checks.append(CheckResult(
        name="seed_set",
        ok=ok_seed,
        severity="warning" if not ok_seed else "info",
        detail=f"seed={seed}",
        error_code="REPRO_SEED_MISSING" if not ok_seed else None,
        remediation="在 trainer 配置中设置 seed（如 42）" if not ok_seed else None,
    ))

    # 2. deterministic 模式与 CUDA 一致性
    deterministic = getattr(config.trainer, "deterministic", False)
    has_cuda = getattr(report, "has_cuda", False)
    if deterministic and not has_cuda:
        checks.append(CheckResult(
            name="deterministic_consistent",
            ok=False,
            severity="warning",
            detail="deterministic=True 但 CUDA 不可用，确定性算法仅对 CUDA 有效",
            error_code="REPRO_DETERMINISTIC_NO_CUDA",
            remediation="设置 deterministic=False 或启用 CUDA",
        ))
    else:
        checks.append(CheckResult(
            name="deterministic_consistent",
            ok=True,
            severity="info",
            detail=f"deterministic={deterministic}, cuda={has_cuda}",
        ))

    # 3. 框架版本记录（用于复现追溯）
    import sys
    try:
        import pytorch_lightning as pl
        pl_version = pl.__version__
    except ImportError:
        pl_version = "not installed"
    checks.append(CheckResult(
        name="framework_version_recorded",
        ok=True,
        severity="info",
        detail=f"torch={torch.__version__}, pl={pl_version}, python={sys.version.split()[0]}",
    ))

    # 4. CUDA 版本兼容性（仅 GPU 模式）
    if has_cuda:
        cuda_version = torch.version.cuda or "unknown"
        # CUDA 11.8+ 为推荐版本
        ok_cuda = cuda_version != "unknown"
        checks.append(CheckResult(
            name="cuda_version_compatible",
            ok=ok_cuda,
            severity="warning" if not ok_cuda else "info",
            detail=f"CUDA={cuda_version}",
            error_code="REPRO_CUDA_UNKNOWN" if not ok_cuda else None,
            remediation="检查 torch 安装是否匹配 CUDA 版本" if not ok_cuda else None,
        ))

    return checks



def build_logger(logger_type: str, output_dir: Path, model_id: str, dataset: str):
    """
    Phase 2.2a：根据配置构建 Lightning logger。

    Args:
        logger_type: csv / tensorboard / wandb / none
        output_dir: 输出根目录
        model_id: 模型 ID（用于 wandb run name）
        dataset: 数据集名（用于 wandb run name）

    Returns:
        Lightning logger 实例，或 None（logger_type=none 时）
    """
    if logger_type == "none":
        return False  # Lightning 接受 logger=False 关闭日志

    if logger_type == "csv":
        return CSVLogger(save_dir=str(output_dir), name="metrics", version="")

    if logger_type == "tensorboard":
        try:
            try:
                from pytorch_lightning.loggers import TensorBoardLogger
            except ImportError:
                from lightning.pytorch.loggers import TensorBoardLogger
            return TensorBoardLogger(save_dir=str(output_dir), name="metrics", version="")
        except ImportError as e:
            raise RuntimeError(
                f"logger='tensorboard' 需要 tensorboard 包，但导入失败: {e}. "
                f"请 `pip install tensorboard` 或改用 logger='csv'."
            ) from e

    if logger_type == "wandb":
        try:
            try:
                from pytorch_lightning.loggers import WandbLogger
            except ImportError:
                from lightning.pytorch.loggers import WandbLogger
            return WandbLogger(
                save_dir=str(output_dir),
                name=f"{model_id}_{dataset}",
                project="senseframe",
            )
        except ImportError as e:
            raise RuntimeError(
                f"logger='wandb' 需要 wandb 包，但导入失败: {e}. "
                f"请 `pip install wandb` 或改用 logger='csv'."
            ) from e

    # 兜底：未知类型回退到 csv
    logger.warning(f"Unknown logger type '{logger_type}', fallback to csv")
    return CSVLogger(save_dir=str(output_dir), name="metrics", version="")
