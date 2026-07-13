"""
senseframe.engine.runner：训练运行器子包。

- orchestrator.py：EpochLogCallback + run_experiment 薄适配器（委托 run_pipeline）
- preflight.py：种子/预检/环境快照/logger 构建
- resolver.py：ExperimentConfig→dict / TaskSpec / FeatureSpec / 场景能力校验
- errors.py：异常→错误码映射
- pipeline.py：可编程 Stage Pipeline（执行逻辑单一真相源）
"""

from .orchestrator import run_experiment, EpochLogCallback
from .preflight import set_seed, preflight_check, build_env_snapshot, build_logger
from .resolver import (
    experiment_config_to_dict,
    load_manifest_for_metadata,
    resolve_task_spec,
    resolve_feature_spec,
    validate_scene_capabilities,
)
from .errors import (
    classify_error,
    SenseFrameError,
    SceneNotRegisteredError,
    DatasetNotSupportedError,
    ModelNotSupportedError,
    DataNotFoundError,
    DataCorruptedError,
    OOMError,
    CheckpointError,
    PreflightError,
    TrainingError,
    ModelBuildError,
    SaveError,
    ConfigValidationError,
)
# RFC Phase D：可编程 Stage Pipeline
from .pipeline import (
    Pipeline,
    PipelineContext,
    StageResult,
    ReadinessReport,
    DanglingRef,
    stage_validate,
    stage_preflight,
    stage_resolve,
    stage_load,
    stage_build,
    stage_train,
    stage_eval,
    stage_export,
    run_pipeline,
    # RFC-003 DSP-3：Stage IO 声明
    FieldSpec,
    StageSpec,
    stage as stage_decorator,
    # RFC-004 方案 G：产物溯源体系
    ArtifactDescriptor,
    ArtifactManifest,
    load_manifest,
    verify_artifacts,
    verify_artifacts_recursive,
)
# RFC-003 DSP-3：stage 装饰器对外暴露为同名 `stage`
stage = stage_decorator

__all__ = [
    "run_experiment",
    "EpochLogCallback",
    "set_seed",
    "preflight_check",
    "build_env_snapshot",
    "build_logger",
    "experiment_config_to_dict",
    "load_manifest_for_metadata",
    "resolve_task_spec",
    "resolve_feature_spec",
    "validate_scene_capabilities",
    "classify_error",
    # 异常层级体系
    "SenseFrameError",
    "SceneNotRegisteredError",
    "DatasetNotSupportedError",
    "ModelNotSupportedError",
    "DataNotFoundError",
    "DataCorruptedError",
    "OOMError",
    "CheckpointError",
    "PreflightError",
    "TrainingError",
    "ModelBuildError",
    "SaveError",
    "ConfigValidationError",
    # RFC Phase D：Stage Pipeline
    "Pipeline",
    "PipelineContext",
    "StageResult",
    "ReadinessReport",
    "DanglingRef",
    "stage_validate",
    "stage_preflight",
    "stage_resolve",
    "stage_load",
    "stage_build",
    "stage_train",
    "stage_eval",
    "stage_export",
    "run_pipeline",
    # RFC-003 DSP-3：Stage IO 声明
    "FieldSpec",
    "StageSpec",
    "stage",
    # RFC-004 方案 G：产物溯源体系
    "ArtifactDescriptor",
    "ArtifactManifest",
    "load_manifest",
    "verify_artifacts",
    "verify_artifacts_recursive",
]
