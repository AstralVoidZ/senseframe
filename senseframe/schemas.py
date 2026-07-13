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
    """

    status: str  # "success" / "error"
    model_id: str
    dataset: str
    learning_mode: str  # "supervised" / "self_supervised"
    resource: Dict[str, Any] = field(default_factory=dict)
    route_config: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    final_eval: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None
    # 可观测性/可复现性扩展字段
    error_traceback: Optional[str] = None
    env_snapshot: Dict[str, Any] = field(default_factory=dict)
    # Phase 6.1：结构化错误码（Agent 友好，无需字符串匹配）
    error_code: Optional[str] = None
    # Phase 7.1：多格式导出结果（None=未导出）
    export: Optional[Dict[str, Any]] = None
    # Phase 7.2：自愈重试记录（None=未启用重试）
    retries: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "dataset": self.dataset,
            "learning_mode": self.learning_mode,
            "resource": self.resource,
            "route_config": self.route_config,
            "training": self.training,
            "final_eval": self.final_eval,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "error": self.error,
            "error_traceback": self.error_traceback,
            "env_snapshot": self.env_snapshot,
            "error_code": self.error_code,
            "export": self.export,
            "retries": self.retries,
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
            for k in ["val_accuracy", "val_macro_f1", "val_micro_f1", "val_weighted_f1"]:
                if k in self.final_eval:
                    # 摘要中保留无前缀名，便于跨工具消费（如 CLI 表格输出）
                    key_metrics[k[len("val_"):]] = self.final_eval[k]
            s["key_metrics"] = key_metrics
            s["model_path"] = self.model_path
            s["output_dir"] = self.output_dir
            # 训练时长（如有，字段名与 runner.py 中 training dict 一致）
            if "duration_s" in self.training:
                s["duration_s"] = self.training["duration_s"]
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
