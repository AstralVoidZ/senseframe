"""
SenseFrame: AI Agent 驱动的 AutoML 训练框架（RFC 控制权反转）。

框架提供可编程原语 + 执行底座 + 安全护栏，Agent 持有控制权。
- 开放策略空间：所有策略（task_type/loss/metric/model）可运行时注册
- 数据驱动：DataProfiler 探查数据特征，策略选择基于数据画像
- 可编程训练流程：run_experiment 拆解为可重组 stage pipeline
- 声明式+命令式融合：YAML 快车道 + 代码注入逃生舱
"""

__version__ = "0.2.0"

# Python 3.14+ Linux 默认 multiprocessing start method 从 fork 改为 forkserver。
# forkserver 在 python -c / REPL 模式下无法重新导入 <stdin> 主模块，导致
# DataLoader worker 崩溃（ConnectionResetError: forkserver can't find <stdin>）。
# 此处显式设置为 spawn，跨平台行为一致，避免 fork/forkserver/spawn 三种路径分歧。
# spawn 要求 worker 用对象（Dataset/Transform/collate_fn）可 pickle，框架已修复闭包问题。
# 若外部已设置 start method（如用户在脚本中显式 mp.set_start_method），则跳过不覆盖。
import multiprocessing as _mp
try:
    if _mp.get_start_method(allow_none=True) is None:
        _mp.set_start_method("spawn")
except RuntimeError:
    pass  # start method 已被外部设置，不覆盖

# Phase 10：动态注册中心对外暴露（Phase R2：统一从 registry 导入）
from .registry import (
    ModelSpec,
    DatasetSpec,
    ZScoreStrategy,
    IdentityStrategy,
    NormalizationStrategy,
    register_model,
    register_dataset,
    register_normalization,
    unregister_model,
    unregister_dataset,
    unregister_normalization,
    bind_model_factory,
    bind_scene_factory,
    set_scene_epochs,
    get_model_spec,
    get_dataset_spec,
    get_normalization,
    get_normalization_or_none,
    is_model_registered,
    is_dataset_registered,
    has_normalization,
    resolve_factory,
    # P2-1 修复：与 list_losses/list_metrics/list_task_types 对称暴露
    list_models,
    list_datasets,
    get_model_info,
)
from .data import normalize, Normalize

# Phase 11：任务类型 + 损失 + 特征 + 场景参数正交化
# RFC Phase A：开放策略空间，TaskType 从封闭枚举改为开放注册表
from .core import (
    TaskType,
    TaskSpec,
    FeatureSpec,
    SceneParams,
    register_loss,
    get_loss,
    has_loss,
    list_losses,
    loss_from_spec,
    LossConfig,
    DEFAULT_LOSS,
    DEFAULT_METRICS,
    register_metric,
    get_metric,
    has_metric,
    list_metrics,
    register_task_type,
    list_task_types,
    has_task_type,
    get_task_type_default_loss,
    get_task_type_default_metrics,
    DataProfiler,
    DataProfile,
    ValidationResult,
    Validator,
    shape_validator,
    numerical_stability_validator,
    signature_validator,
    performance_validator,
    transform_pipeline_validator,
    compose,
    run_validation,
)

# RFC Phase D：可编程训练流程 — Stage Pipeline + run_experiment 入口
from .engine.runner import (
    run_experiment,
    Pipeline,
    PipelineContext,
    StageResult,
    stage_validate,
    stage_preflight,
    stage_resolve,
    stage_load,
    stage_build,
    stage_train,
    stage_eval,
    stage_export,
    run_pipeline,
    # RFC-004 方案 G：产物溯源体系
    ArtifactDescriptor,
    ArtifactManifest,
    load_manifest,
    verify_artifacts,
    verify_artifacts_recursive,
    verify_manifest_schema,
    verify_artifacts_full,
)

# RFC Phase F：代码注入 — load_extension API
from .extensions import load_extension

# RFC-002 阶段 H：技能库
from .skills import (
    Skill,
    SkillLibrary,
    save_skill,
    load_skill,
    search_skills,
    list_skills,
    get_skill_library,
)

# RFC-002 阶段 L/P：探索状态管理 + 搜索空间地图
from .exploration import ExplorationTracker, SearchSpaceMap

# RFC-003 场景入口点（方案 B）：顶层暴露 activate_lazy_scenes / get_scene
# CQS 合规：getter 不再有自动注册副作用，调用方需显式激活延迟场景
# 命令文件（commands/*.md）与文档均按 sf.activate_lazy_scenes() / sf.get_scene() 心智模型编写
from .scenes import activate_lazy_scenes, get_scene, list_scenes, has_scene, register_scene

# RFC-003 SP：搜索协议 — Ask-Tell 标准化接口（P0.5 顶层导出）
from .search_protocol import (
    ParameterSpec,
    SearchSpace,
    StudySpec,
    TrialSpec,
    TrialResult,
    StudyManager,
    get_study_manager,
    Sampler,
    register_sampler,
    get_sampler,
    list_samplers,
    RandomSampler,
    GridSampler,
)

# RFC-003 ε1：损失函数搜索（P1，SP 应用层）
from .automl import (
    build_loss_search_space,
    run_loss_search,
    LossSearchResult,
)

# RFC-003 ε6：对比实验模块（P1，过渡形态，DSP 合规）
from .experiment import (
    TrialGroup,
    TrialStatus,
    TrialResult as ExperimentTrialResult,  # 避免与 SP TrialResult 冲突
    ExperimentBudget,
    MethodConfig,
    BaselineConfig,
    ExperimentDesign,
    MethodRunner,
    BaselineRunner,
    ExperimentRunner,
    ComparisonReport,
)

# P3：推理服务 + 模型版本管理
from .inference import (
    PredictionResult,
    InferenceModel,
    ONNXInferenceModel,
    load_model_for_inference,
    predict,
)
from .serving import InferenceServer

# RFC-003 DSP-5：自省模块
from .introspect import (
    StageIOSpec,
    context_schema,
    context_describe,
    stage_io,
    list_stages,
    pipeline_graph,
    data_bundle_schema,
    data_bundle_describe,
    data_profile_schema,
    data_profile_describe,
)

# RFC-005 资源泄露修复：注册 atexit 兜底清理
# 进程退出时关闭 OTel 后台线程 + Prometheus HTTP server 端口
# 避免 serving.py / HPO 进程退出后端口/线程残留
import atexit as _atexit


def _cleanup_at_exit():
    """进程退出兜底清理（atexit 注册）。"""
    try:
        from .observability_otel import shutdown_otel
        shutdown_otel()
    except Exception:
        pass


_atexit.register(_cleanup_at_exit)
