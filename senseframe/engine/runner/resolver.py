"""
Phase 14.1.2：配置与 Spec 解析模块。

从 runner.py 拆出，包含：
- ExperimentConfig → dict 转换
- TaskSpec 解析（YAML 优先 > 场景默认）
- FeatureSpec 解析（场景默认 > input_features 派生）
- 场景能力校验
- manifest 元数据加载
"""

import dataclasses
from typing import Any, Dict, Optional

from ...core.task import TaskSpec, TaskType
from ...engine.config import ExperimentConfig
from ...observability import setup_logging

logger = setup_logging()


def experiment_config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    将声明式 ExperimentConfig 转换为 resolve_config 所需的 dict 配置。

    转换规则：
    - scene.* → 顶层 model_id/dataset/learning_mode/data_root（向后兼容，供 resolve_config 路由读取）
    - scene → 完整递归序列化 dict（含 name/dataset/model_id/learning_mode/data_root/params/task_spec），
      供 metadata.config.scene 使用（P2 修复：原实现缺失此字段导致下游为 None）
    - scene.params → 合并到顶层（支持 self_supervised_epochs/metrics/gpu/resume 等透传）
    - trainer.* → 顶层 epochs/learning_rate/batch_size/optimizer/...
    - output_dir/save_model → 顶层
    - input_features/output_features → 暂存于 _features 元数据键

    缺省字段使用 None，由 run_experiment 内部走默认值/路由填充逻辑。
    """
    d: Dict[str, Any] = {
        # 场景
        "model_id": cfg.scene.model_id,
        "dataset": cfg.scene.dataset,
        "learning_mode": cfg.scene.learning_mode,
        "data_root": cfg.scene.data_root,  # 已由 SceneConfig.validate() 校验非空
        # P2 修复：递归序列化完整 SceneConfig（含嵌套 task_spec dataclass），
        # 供 metadata.config.scene 使用。根因：原实现仅展开 scene 标量字段到顶层，
        # 无 scene 整体 dict，导致下游 metadata.config.scene 为 None。
        # asdict() 递归转换嵌套 dataclass（TaskSpec）/dict/list/tuple，原始类型保持不变。
        "scene": dataclasses.asdict(cfg.scene) if cfg.scene is not None else None,
        # 训练器
        "epochs": cfg.trainer.epochs,
        "learning_rate": cfg.trainer.learning_rate,
        "batch_size": cfg.trainer.batch_size,
        "optimizer": cfg.trainer.optimizer,
        "weight_decay": cfg.trainer.weight_decay,
        "early_stopping": cfg.trainer.early_stopping,
        "deterministic": cfg.trainer.deterministic,
        "max_time": cfg.trainer.max_time,
        "seed": cfg.trainer.seed,
        # Phase 1.1a：scheduler 正式从 trainer 读取（向后兼容 scene.params 透传）
        "scheduler": cfg.trainer.scheduler,
        # Phase 1.2a：梯度裁剪与累积
        "gradient_clip_val": cfg.trainer.gradient_clip_val,
        "gradient_clip_algorithm": cfg.trainer.gradient_clip_algorithm,
        "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
        # Phase 2.2a：logger 后端
        "logger": cfg.trainer.logger,
        # 输出
        "output_dir": cfg.output_dir,
        "save_model": cfg.save_model,
        # HPO（Stage 4 使用，此处透传供后续读取）
        "hpo": {
            "enabled": cfg.hpo.enabled,
            "n_trials": cfg.hpo.n_trials,
            "sampler": cfg.hpo.sampler,
            "pruner": cfg.hpo.pruner,
            "metric": cfg.hpo.metric,
            "direction": cfg.hpo.direction,
        },
        # 特征声明（Stage 5+ 使用，当前仅作为元数据保存）
        "_features": {
            "input": [f.__dict__ if hasattr(f, "__dict__") else dict(f)
                      for f in cfg.input_features],
            "output": [f.__dict__ if hasattr(f, "__dict__") else dict(f)
                       for f in cfg.output_features],
        },
    }

    # Phase 11.4：scene.params 正交化 — 新字段从 trainer 读取
    # 向后兼容：scene.params 中的同名键仍可覆盖（escape hatch）
    d["self_supervised_epochs"] = cfg.trainer.self_supervised_epochs
    d["metrics"] = cfg.trainer.metrics
    d["gpu"] = cfg.trainer.gpu
    d["resume"] = cfg.trainer.resume
    # Phase 14.3.3：mixed_precision 正交化
    d["mixed_precision"] = cfg.trainer.mixed_precision

    # scene.params 透传到顶层（允许覆盖 trainer 字段，提供 escape hatch）
    # 已知透传键：self_supervised_epochs, metrics, average, scheduler,
    #             gpu, resume, mixed_precision
    for k, v in (cfg.scene.params or {}).items():
        d[k] = v

    return d


def load_manifest_for_metadata(scene_params):
    """R-fix：从 scene.params 加载 manifest（用于 metadata 记录）。

    通过 data.manifest 公共 API 加载，不越层访问场景私有函数。
    """
    from ...data.manifest import load_manifest
    manifest_path = scene_params.get("manifest_path") if scene_params else None
    if manifest_path is None:
        return None
    return load_manifest(manifest_path)


def resolve_task_spec(
    config: ExperimentConfig,
    scene,
    dataset: str,
    model_id: str,
    num_classes: int,
    scene_kwargs: Optional[Dict[str, Any]] = None,
    data_profile=None,
) -> TaskSpec:
    """
    Phase 13.1：解析最终 TaskSpec（RFC Phase B：数据驱动推断）。

    优先级：
    1. config.scene.task_spec（YAML 显式声明，已是 TaskSpec 实例）
    2. scene.get_task_spec()（场景容器默认）
    3. 数据画像推断（RFC Phase B：基于 DataProfiler 推荐，可被上面两层覆盖）

    Args:
        data_profile: 可选的 DataProfile 实例。若提供且上面两层都未显式指定
                      task_type/loss/metrics，则用画像推荐值填充。
    """
    ts_field = config.scene.task_spec
    if ts_field is not None:
        # R-fix：TaskSpecField 已合并为 TaskSpec，直接使用
        if ts_field.num_classes is None:
            ts_field.num_classes = num_classes
        return ts_field

    # 场景容器默认
    try:
        return scene.get_task_spec(dataset, model_id, **(scene_kwargs or {}))
    except Exception:
        # RFC Phase B：场景容器无法提供时，用数据画像推断
        if data_profile is not None:
            from ...core.task import TaskSpec
            return TaskSpec(
                task_type=data_profile.recommended_task_type,
                num_classes=data_profile.n_classes,
                loss=data_profile.recommended_loss,
                metrics=data_profile.recommended_metrics,
            )
        raise


def resolve_feature_spec(
    config: ExperimentConfig,
    scene,
    dataset: str,
    scene_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Phase 11.3：解析最终 FeatureSpec。

    优先级：
    1. scene.get_feature_spec()（场景容器默认，可覆写）
    2. 从 ExperimentConfig.input_features 派生（YAML 声明）
    """
    from ...core.features import FeatureSpec
    # 1. 场景容器默认
    # 缩小 except 范围：仅捕获数据集不存在/manifest 缺失等预期异常，
    # 其他异常（如代码 bug）应正常抛出
    try:
        spec = scene.get_feature_spec(dataset, **(scene_kwargs or {}))
        if spec is not None and spec.input_shape is not None:
            return spec
    except (KeyError, FileNotFoundError, ValueError) as e:
        logger.debug(f"Scene get_feature_spec fallback to input_features: {e}")
    # 2. 从 input_features 派生
    if config.input_features:
        f = config.input_features[0]
        return FeatureSpec(
            input_shape=tuple(f.shape) if f.shape else None,
            modality=f.type,
        )
    return FeatureSpec()


def validate_scene_capabilities(
    meta,
    task_spec: TaskSpec,
    learning_mode: str,
    scene_name: str,
) -> None:
    """
    Phase 13.2：校验场景能力与配置匹配。

    - task_spec.task_type 必须在 scene.meta().supported_tasks 中
    - learning_mode 必须在 scene.meta().supported_learning_modes 中

    快速失败，不进入训练编排层。
    """
    if meta.supported_tasks:
        if task_spec.task_type not in meta.supported_tasks:
            raise ValueError(
                f"Scene '{scene_name}' does not support task type "
                f"'{task_spec.task_type}'. "
                f"Supported: {meta.supported_tasks}"
            )
    if meta.supported_learning_modes:
        if learning_mode not in meta.supported_learning_modes:
            raise ValueError(
                f"Scene '{scene_name}' does not support learning_mode "
                f"'{learning_mode}'. "
                f"Supported: {meta.supported_learning_modes}"
            )
