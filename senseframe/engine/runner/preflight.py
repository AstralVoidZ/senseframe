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
    from ...engine.config import DEFAULT_DATA_ROOT

    # 1. 数据集目录存在性
    # R-fix：custom 场景由 manifest 决定数据位置，跳过固定目录检查，
    # 改由下方 manifest 样本文件存在性检查覆盖。
    is_manifest_driven = (
        scene_name == "custom"
        and scene_params is not None
        and "manifest_path" in scene_params
    )
    if not is_manifest_driven:
        data_root = Path(config.get("data_root") or DEFAULT_DATA_ROOT)
        expected_dirs = _get_dataset_dir_names(dataset)
        for d in expected_dirs:
            if not (data_root / d).exists():
                raise DataNotFoundError(
                    f"Dataset directory not found: {data_root / d} "
                    f"(dataset={dataset}, data_root={data_root})"
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

    # 2. GPU 显存预检（含 20% 安全余量）
    if report.has_cuda and model_info.get("estimated_vram_mb"):
        needed = model_info["estimated_vram_mb"] * 1.2
        free = report.gpu_free_vram_mb or 0
        if free < needed:
            raise PreflightError(
                f"GPU free VRAM {free}MB < required {needed:.0f}MB "
                f"(model {model_info['id']} needs ~{model_info['estimated_vram_mb']}MB + 20% margin). "
                f"Consider a smaller model or use CPU route."
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
