"""RFC Phase D：可编程训练流程 — Stage Pipeline。

将 run_experiment 拆解为可重组的 stage，Agent 可：
- 调用单个 stage（如只做 preflight 预检）
- 替换 stage（如自定义 eval）
- 插入 pre/post hook
- 跳过 stage
- 编排自定义 pipeline

9 个顶层 stage（RFC 决策：不进一步拆解）：
1. validate    — 校验配置 schema + 场景注册
2. preflight   — 资源探测 + 路由 + 预检
3. load        — 加载数据 + 数据画像
4. resolve     — 解析 TaskSpec / FeatureSpec / 配置
5. build       — 构建模型 / DataModule / LightningModule
6. probe_vram  — 动态显存探测（方案 B）
7. train       — 训练执行
8. eval        — 评估
9. export      — 导出

设计原则（RFC 原则 4）：
- 训练流程是可组合的 stage pipeline，不是黑盒函数
- 每个 stage 独立可调用，支持 pre/post hook、替换、跳过

向后兼容：run_experiment 保持不变，Pipeline 是并行入口供 Agent 使用。

包结构：
- protocols.py：8 个 Protocol 定义（结构子类型契约）
- context.py：PipelineContext + StageResult + ReadinessReport + DanglingRef + 字段填充映射
- stage_spec.py：FieldSpec + StageSpec + @stage 装饰器 + StageFn
- stages/：9 个 stage 文件（validate/preflight/resolve/load/build/probe_vram/train/eval/export）
- errors.py：_classify_runtime_error（运行时异常分类）
- runtime.py：Pipeline class + run_pipeline
- artifacts_api.py：公共溯源 API（load_manifest / verify_artifacts 等）
"""
from __future__ import annotations

# RFC-003 DSP-1：Protocol 类型（结构子类型契约）
from .protocols import (
    SceneProtocol,
    ModelProtocol,
    DataModuleProtocol,
    TrainerProtocol,
    LoggerProtocol,
    SceneMetaProtocol,
    TaskSpecProtocol,
    FeatureSpecProtocol,
)
# PipelineContext + Stage 结果类型 + 字段填充映射 + 资源清理辅助
from .context import (
    PipelineContext,
    StageResult,
    ReadinessReport,
    DanglingRef,
    _FIELD_FILL_STAGE,
)
# RFC-003 DSP-3：Stage IO 声明
from .stage_spec import (
    FieldSpec,
    StageSpec,
    stage,
    StageFn,
)
# 9 个默认 stage
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
    analyze_training_result,
    _run_probe_in_subprocess,
)
# Pipeline class + run_pipeline 入口
from .runtime import (
    Pipeline,
    run_pipeline,
    _NON_SERIALIZABLE_STAGES,
)
# RFC-004 方案 G：产物溯源体系
from ..artifacts import ArtifactDescriptor, ArtifactManifest
from .artifacts_api import (
    load_manifest,
    verify_artifacts,
    verify_artifacts_recursive,
    verify_manifest_schema,
    verify_artifacts_full,
)

__all__ = [
    "PipelineContext",
    "Pipeline",
    "StageResult",
    "ReadinessReport",
    "DanglingRef",
    "_FIELD_FILL_STAGE",
    "StageFn",
    # RFC-003 DSP-1：Protocol 类型
    "SceneProtocol",
    "ModelProtocol",
    "DataModuleProtocol",
    "TrainerProtocol",
    "SceneMetaProtocol",
    "TaskSpecProtocol",
    "FeatureSpecProtocol",
    # RFC-003 DSP-3：Stage IO 声明
    "FieldSpec",
    "StageSpec",
    "stage",
    # 9 个默认 stage
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
    "_NON_SERIALIZABLE_STAGES",
    "run_pipeline",
    # RFC-004 方案 G：产物溯源体系
    "ArtifactDescriptor",
    "ArtifactManifest",
    "load_manifest",
    "verify_artifacts",
    "verify_artifacts_recursive",
    "verify_manifest_schema",
    "verify_artifacts_full",
]
