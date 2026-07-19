"""9 个 stage 函数的统一导出。

包含：
- stage_validate / stage_preflight / stage_resolve / stage_load
- stage_build / stage_probe_vram / stage_train / stage_eval / stage_export
- analyze_training_result（stage_eval 依赖）
- _merge_metrics_csv（stage_export 依赖）
- _compute_config_hash / _generate_manifest / _PIPELINE_VERSION（Pipeline.run 依赖）
"""
from .validate import stage_validate
from .preflight import stage_preflight
from .resolve import stage_resolve
from .load import stage_load
from .build import stage_build
from .probe_vram import stage_probe_vram, _run_probe_in_subprocess
from .train import stage_train, analyze_training_result
from .eval import stage_eval, _merge_metrics_csv
from .export import (
    stage_export,
    _compute_config_hash,
    _generate_manifest,
    _PIPELINE_VERSION,
)

__all__ = [
    "stage_validate",
    "stage_preflight",
    "stage_resolve",
    "stage_load",
    "stage_build",
    "stage_probe_vram",
    "stage_train",
    "stage_eval",
    "stage_export",
    "analyze_training_result",
    "_run_probe_in_subprocess",
    "_merge_metrics_csv",
    "_compute_config_hash",
    "_generate_manifest",
    "_PIPELINE_VERSION",
]
