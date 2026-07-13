"""
数据模型定义：ResourceReport 和 TrainOutput。

这些 dataclass 是框架内部各模块之间传递的结构化数据载体，
也是 CLI JSON 输出的序列化来源。

Phase 6.1：新增结构化错误码（error_code）和机器可读状态摘要（summary）。
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


# ============================================================
# Phase 6.1：结构化错误码枚举
# ============================================================
# Agent 可基于 error_code 做程序化分支，无需字符串匹配
ERROR_CODES = {
    "OK": "成功",
    "CONFIG_VALIDATION_ERROR": "配置校验失败",
    "SCENE_NOT_FOUND": "场景未注册",
    "DATASET_NOT_SUPPORTED": "数据集不被场景支持",
    "MODEL_NOT_SUPPORTED": "模型不被场景支持",
    "DATA_NOT_FOUND": "数据集文件未找到",
    "DATA_LOAD_ERROR": "数据加载失败",
    "MODEL_BUILD_ERROR": "模型构建失败",
    "TRAINING_ERROR": "训练过程异常",
    "OOM_ERROR": "显存/内存不足",
    "CHECKPOINT_ERROR": "Checkpoint 加载/保存失败",
    "SAVE_ERROR": "模型/元数据保存失败",
    "PREFLIGHT_ERROR": "预检失败（显存/磁盘不足）",
    "UNKNOWN_ERROR": "未知错误",
}


@dataclass
class ResourceReport:
    """硬件资源探测结果。"""

    has_cuda: bool
    gpu_name: Optional[str]
    gpu_total_vram_mb: Optional[int]
    gpu_free_vram_mb: Optional[int]
    cpu_count: int
    cpu_memory_total_mb: int
    cpu_memory_available_mb: int
    # P3: Apple Silicon MPS 支持
    has_mps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_cuda": self.has_cuda,
            "gpu_name": self.gpu_name,
            "gpu_total_vram_mb": self.gpu_total_vram_mb,
            "gpu_free_vram_mb": self.gpu_free_vram_mb,
            "cpu_count": self.cpu_count,
            "cpu_memory_total_mb": self.cpu_memory_total_mb,
            "cpu_memory_available_mb": self.cpu_memory_available_mb,
            "has_mps": self.has_mps,
        }


@dataclass
class TrainOutput:
    """
    单次训练的 ML 过程输出。

    框架只关注 ML 过程的结果，不定死 AutoML 层结果结构。
    上层编排器可以在此基础上封装自己的迭代结果格式。

    Phase 6.1：新增 error_code 字段（结构化错误码，Agent 友好）。
    P5 P2-7 阶段3：training/env_snapshot/feedback 字段类型收窄为对应 dataclass。
    """

    status: str  # "success" / "error"
    model_id: str
    dataset: str
    learning_mode: str  # "supervised" / "self_supervised"
    resource: Dict[str, Any] = field(default_factory=dict)
    route_config: Dict[str, Any] = field(default_factory=dict)
    # P5 P2-7 阶段3：training 从 Dict[str, Any] 收窄为 Optional[TrainingSummary]
    training: Optional["TrainingSummary"] = None
    final_eval: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None
    # 可观测性/可复现性扩展字段
    error_traceback: Optional[str] = None
    # P5 P2-7 阶段3：env_snapshot 从 Dict[str, Any] 收窄为 Optional[EnvSnapshot]
    env_snapshot: Optional["EnvSnapshot"] = None
    # Phase 6.1：结构化错误码（Agent 友好，无需字符串匹配）
    error_code: Optional[str] = None
    # Phase 7.1：多格式导出结果（None=未导出）
    export: Optional[Dict[str, Any]] = None
    # Phase 7.2：自愈重试记录（None=未启用重试）
    retries: Optional[Dict[str, Any]] = None
    # P5 P1-I：训练反馈（失败分类 + 改进建议），由 analyze_training_result 生成。
    # P5 P2-7 阶段3：feedback 从 Dict[str, Any] 收窄为 Optional[FeedbackResult]
    feedback: Optional["FeedbackResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        # P5 P2-7 Step 1 前置修复：多态序列化 helper。
        # 当字段为 dataclass 实例（有 to_dict 方法）时自动转换，为未来阶段2 切换铺路；
        # 当前字段仍是 dict 时直接返回，零破坏性。
        def _serialize(v):
            if hasattr(v, "to_dict") and callable(v.to_dict):
                return v.to_dict()
            return v
        return {
            "status": self.status,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "learning_mode": self.learning_mode,
            "resource": _serialize(self.resource),
            "route_config": _serialize(self.route_config),
            "training": _serialize(self.training),
            "final_eval": _serialize(self.final_eval),
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "error": self.error,
            "error_traceback": self.error_traceback,
            "env_snapshot": _serialize(self.env_snapshot),
            "error_code": self.error_code,
            "export": _serialize(self.export),
            "retries": _serialize(self.retries),
            "feedback": _serialize(self.feedback),
        }

    def summary(self) -> Dict[str, Any]:
        """
        Phase 6.1：生成机器可读的状态摘要。

        Agent 可快速判断训练结果，无需解析完整 to_dict：
        - status: success/error
        - error_code: 结构化错误码（error 时）
        - key_metrics: 核心指标摘要（success 时）
        - model_path: 模型路径（success 时）

        Returns:
            摘要字典
        """
        s = {
            "status": self.status,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "learning_mode": self.learning_mode,
        }
        if self.status == "success":
            # 提取核心指标（RFC-004 方案 C：final_eval 字段统一 val_ 前缀）
            key_metrics = {}
            final_eval = self.final_eval or {}
            for k in ["val_accuracy", "val_macro_f1", "val_micro_f1", "val_weighted_f1"]:
                if k in final_eval:
                    # 摘要中保留无前缀名，便于跨工具消费（如 CLI 表格输出）
                    key_metrics[k[len("val_"):]] = final_eval[k]
            s["key_metrics"] = key_metrics
            s["model_path"] = self.model_path
            s["output_dir"] = self.output_dir
            # P5 P2-7 Step 1 前置修复：training 字段未来可能为 TrainingSummary dataclass，
            # 此处用 getattr 兼容 dict 和 dataclass 两种形态。
            training = self.training or {}
            duration_s = training.get("duration_s") if hasattr(training, "get") \
                else getattr(training, "duration_s", None)
            if duration_s is not None:
                s["duration_s"] = duration_s
        else:
            s["error_code"] = self.error_code or "UNKNOWN_ERROR"
            s["error"] = self.error
        return s


# ============================================================
# P1-1：产物 Schema 层 — training_log 单条记录契约
# ============================================================
@dataclass
class TrainingLogEntry:
    """training_log.jsonl 单条记录的 schema（P1-1: 产物 Schema 层）。

    强制字段类型契约，拦截 LR 污染等类型错误。
    None 表示该 epoch 未产生该指标（不强制必填），但非 None 时必须是 float/int。
    """
    epoch: int                              # 1-based epoch 序号（与 trainer.current_epoch 对齐时 +1）
    lr: Optional[float] = None              # 学习率（float 或 None，不允许 str）
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    train_macro_f1: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None

    def __post_init__(self):
        """字段类型校验，拦截类型污染。"""
        # epoch 必须是 int >= 0
        if not isinstance(self.epoch, int) or self.epoch < 0:
            raise ValueError(
                f"TrainingLogEntry.epoch must be int >= 0, got {self.epoch!r}"
            )
        # 可选字段：None 或 (int, float)
        for f_name in ["lr", "train_loss", "train_accuracy", "train_macro_f1",
                       "val_loss", "val_accuracy", "val_macro_f1"]:
            v = getattr(self, f_name)
            if v is not None and not isinstance(v, (int, float)):
                raise TypeError(
                    f"TrainingLogEntry.{f_name} must be float/int or None, "
                    f"got {type(v).__name__}: {v!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（仅包含非 None 字段）。

        用于写入 training_log.jsonl，避免 None 字段污染产物。
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


def validate_training_log_entry(raw: Dict[str, Any]) -> TrainingLogEntry:
    """从 dict 构造 TrainingLogEntry，做类型转换 + 校验。

    LR 污染为 str 时尝试 float() 转换，转换失败抛 ValueError。

    Args:
        raw: 原始 dict（通常来自 module.py 的 training_log 列表）

    Returns:
        TrainingLogEntry 实例

    Raises:
        ValueError: 字段缺失或类型转换失败
        TypeError: 字段类型不匹配且无法转换
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict, got {type(raw).__name__}")

    def _to_float(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # 尝试转换（处理 str 数字、tensor 等）
        try:
            if hasattr(v, "item"):
                return float(v.item())
            return float(v)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field {key!r} cannot be converted to float: {v!r} ({type(v).__name__})"
            ) from e

    try:
        epoch = int(raw["epoch"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Field 'epoch' missing or invalid: {e}") from e

    return TrainingLogEntry(
        epoch=epoch,
        lr=_to_float("lr"),
        train_loss=_to_float("train_loss"),
        train_accuracy=_to_float("train_accuracy"),
        train_macro_f1=_to_float("train_macro_f1"),
        val_loss=_to_float("val_loss"),
        val_accuracy=_to_float("val_accuracy"),
        val_macro_f1=_to_float("val_macro_f1"),
    )


# ============================================================
# P5 P2-7 阶段1：TrainOutput Dict[str, Any] 字段的类型化 dataclass
# ============================================================
# 阶段1 仅新增 dataclass + validate 函数，不修改 TrainOutput 字段类型（零破坏性）。
# 阶段2 将构造点切换为 dataclass，阶段3 收窄 TrainOutput 字段类型注解。

@dataclass
class FeedbackResult:
    """训练反馈（stage_eval 的 analyze_training_result 产出）。

    P5 P2-7 阶段1：替代 TrainOutput.feedback 的 Dict[str, Any]，
    提供 6 值枚举 status + 固定结构 + 可选 test_metrics 扩展。
    """
    status: str  # numerical_instability / underfitting / overfitting /
                 # generalization_gap / converged / success
    diagnosis: str
    suggestions: List[str] = field(default_factory=list)
    test_metrics: Optional[Dict[str, float]] = None  # 可选，stage_export 追加

    def __post_init__(self):
        valid_statuses = {
            "numerical_instability", "underfitting", "overfitting",
            "generalization_gap", "converged", "success",
        }
        if self.status not in valid_statuses:
            raise ValueError(
                f"FeedbackResult.status 非法值 '{self.status}'，"
                f"合法值: {sorted(valid_statuses)}"
            )
        if not isinstance(self.suggestions, list):
            raise TypeError(
                f"FeedbackResult.suggestions 必须为 list，实际: {type(self.suggestions).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "diagnosis": self.diagnosis,
            "suggestions": list(self.suggestions),
        }
        if self.test_metrics is not None:
            d["test_metrics"] = dict(self.test_metrics)
        return d


@dataclass
class TrainingSummary:
    """训练过程摘要（stage_export 填充 TrainOutput.training）。

    P5 P2-7 阶段1：替代 TrainOutput.training 的 Dict[str, Any]，
    log 字段引用已有 TrainingLogEntry dataclass。
    """
    epochs_trained: int
    early_stopped: bool
    log: List[Any] = field(default_factory=list)  # List[TrainingLogEntry | dict]
    duration_s: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_checkpoint: Optional[str] = None
    intermediate_values: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.epochs_trained, int) or self.epochs_trained < 0:
            raise ValueError(
                f"TrainingSummary.epochs_trained 必须为非负 int，实际: {self.epochs_trained}"
            )
        if not isinstance(self.early_stopped, bool):
            raise TypeError(
                f"TrainingSummary.early_stopped 必须为 bool，实际: {type(self.early_stopped).__name__}"
            )
        if not isinstance(self.log, list):
            raise TypeError(
                f"TrainingSummary.log 必须为 list，实际: {type(self.log).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        serialized_log = []
        for entry in self.log:
            if hasattr(entry, "to_dict"):
                serialized_log.append(entry.to_dict())
            elif isinstance(entry, dict):
                serialized_log.append(entry)
            else:
                serialized_log.append(repr(entry))
        return {
            "epochs_trained": self.epochs_trained,
            "early_stopped": self.early_stopped,
            "log": serialized_log,
            "duration_s": self.duration_s,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": self.best_checkpoint,
            "intermediate_values": dict(self.intermediate_values),
        }


@dataclass
class EnvSnapshot:
    """环境快照（build_env_snapshot 产出）。

    P5 P2-7 阶段1：替代 TrainOutput.env_snapshot 的 Dict[str, Any]，
    6 个固定字段全部类型化。
    """
    torch: str
    pytorch_lightning: str
    cuda: Optional[str]
    python: str
    deterministic: bool = False
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "torch": self.torch,
            "pytorch_lightning": self.pytorch_lightning,
            "cuda": self.cuda,
            "python": self.python,
            "deterministic": self.deterministic,
            "seed": self.seed,
        }


def validate_feedback(d: Dict[str, Any]) -> FeedbackResult:
    """从 dict 构造 FeedbackResult，做类型校验。

    P5 P2-7 阶段1：在 analyze_training_result 出口调用，捕获类型污染。

    Args:
        d: analyze_training_result 返回的 dict

    Returns:
        FeedbackResult 实例

    Raises:
        ValueError: status 非法或 suggestions 为空
        TypeError: 字段类型不匹配
    """
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict, got {type(d).__name__}")
    status = d.get("status")
    diagnosis = d.get("diagnosis", "")
    suggestions = d.get("suggestions", [])
    test_metrics = d.get("test_metrics")
    if not status:
        raise ValueError("feedback dict 缺少 'status' 字段")
    if not isinstance(suggestions, list):
        raise TypeError(f"suggestions 必须为 list，实际: {type(suggestions).__name__}")
    if test_metrics is not None and not isinstance(test_metrics, dict):
        raise TypeError(f"test_metrics 必须为 dict 或 None，实际: {type(test_metrics).__name__}")
    return FeedbackResult(
        status=str(status),
        diagnosis=str(diagnosis),
        suggestions=[str(s) for s in suggestions],
        test_metrics=test_metrics,
    )


def validate_training_summary(d: Dict[str, Any]) -> TrainingSummary:
    """从 dict 构造 TrainingSummary，做类型校验。

    P5 P2-7 阶段1：在 stage_export 出口调用，捕获类型污染。

    Args:
        d: stage_export 填充的 training dict

    Returns:
        TrainingSummary 实例

    Raises:
        ValueError: epochs_trained 非法
        TypeError: 字段类型不匹配
    """
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict, got {type(d).__name__}")
    intermediate = d.get("intermediate_values", {})
    if not isinstance(intermediate, dict):
        raise TypeError(
            f"intermediate_values 必须为 dict，实际: {type(intermediate).__name__}"
        )
    return TrainingSummary(
        epochs_trained=int(d.get("epochs_trained", 0)),
        early_stopped=bool(d.get("early_stopped", False)),
        log=list(d.get("log", [])),
        duration_s=d.get("duration_s"),
        best_val_loss=d.get("best_val_loss"),
        best_checkpoint=d.get("best_checkpoint"),
        intermediate_values=intermediate,
    )


def validate_env_snapshot(d: Dict[str, Any]) -> EnvSnapshot:
    """从 dict 构造 EnvSnapshot，做类型校验。

    P5 P2-7 阶段1：在 build_env_snapshot 出口调用，捕获类型污染。

    Args:
        d: build_env_snapshot 返回的 dict

    Returns:
        EnvSnapshot 实例
    """
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict, got {type(d).__name__}")
    return EnvSnapshot(
        torch=str(d.get("torch", "")),
        pytorch_lightning=str(d.get("pytorch_lightning", "")),
        cuda=d.get("cuda"),
        python=str(d.get("python", "")),
        deterministic=bool(d.get("deterministic", False)),
        seed=int(d.get("seed", 42)),
    )


# ============================================================
# P5 P3-4 选项 B：scene.params 渐进式 schema 验证
# ============================================================
# 借鉴 P2-7 阶段1 范式：不修改 SceneConfig.params 类型（仍为 Dict[str, Any]），
# 在入口（cli.py）和透传前（resolver.py）校验已知键的类型，捕获类型污染。
# 完整正交化（Phase 11.4）暂缓，等待真实场景需求驱动。

# 已知 scene.params 键的类型约束（None 表示允许任意类型，仅检查存在性）
_SCENE_PARAMS_SCHEMA: Dict[str, type] = {
    # Phase 11.4 已提升到 TrainerConfig 顶层的字段（scene.params 作为 escape hatch 覆盖）
    "self_supervised_epochs": int,
    "metrics": list,
    "gpu": (int, type(None)),
    "resume": (str, type(None)),
    "mixed_precision": (str, type(None)),
    # 任务相关
    "loss": str,
    "loss_kwargs": dict,
    "average": str,
    "scheduler": dict,
    # 数据/场景元数据
    "manifest_path": str,
    "transform": dict,
}


def validate_scene_params(params) -> None:
    """校验 scene.params 的已知键类型（P5 P3-4 选项 B）。

    渐进式 schema 验证：不修改 SceneConfig.params 类型，仅在入口和透传前
    校验已知键的类型，捕获类型污染（如 metrics 误传字符串而非 list）。

    P5 P3-4 完整正交化后：params 可能为 SceneParams 实例或 dict，
    用 to_flat_dict() 统一转为 dict 后校验。

    未知键不报错（允许场景特定扩展），仅校验已知键。
    None 或空直接通过。

    Args:
        params: SceneConfig.params 字段（SceneParams 或 dict）

    Raises:
        TypeError: 已知键类型不匹配
    """
    if not params:
        return
    # P5 P3-4：兼容 SceneParams 和 dict 两种形态
    if hasattr(params, "to_flat_dict"):
        params = params.to_flat_dict()
    if not isinstance(params, dict):
        raise TypeError(
            f"scene.params 必须为 dict 或 SceneParams，实际: {type(params).__name__}"
        )
    for key, expected_type in _SCENE_PARAMS_SCHEMA.items():
        if key not in params:
            continue
        v = params[key]
        # 允许 None 值（未设置）跳过校验，除非 None 本身就是合法类型
        if v is None and not (isinstance(expected_type, tuple) and type(None) in expected_type):
            continue
        if not isinstance(v, expected_type):
            raise TypeError(
                f"scene.params['{key}'] 类型错误：期望 {expected_type}，"
                f"实际 {type(v).__name__}: {v!r}"
            )


# ============================================================
# P1-1：产物 Schema 层 — 类型安全的 JSON 序列化
# ============================================================
def safe_json_dumps(obj: Any, strict: bool = False, **kwargs) -> str:
    """类型安全的 JSON 序列化（P1-1: 产物 Schema 层）。

    Args:
        obj: 待序列化对象
        strict: True 时类型不匹配抛 TypeError（生产路径，schema 校验场景）；
                False 时 default=str 兜底（调试路径，保持向后兼容）
        **kwargs: 透传给 json.dumps（如 ensure_ascii, indent）

    Returns:
        JSON 字符串

    Raises:
        TypeError: strict=True 时遇到不可序列化对象
    """
    import json
    if strict:
        # strict 模式：不传 default，类型错误直接抛
        return json.dumps(obj, **kwargs)
    else:
        # 兜底模式：default=str 保持向后兼容
        return json.dumps(obj, default=str, **kwargs)
