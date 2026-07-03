"""
数据模型定义：ResourceReport 和 TrainOutput。

这些 dataclass 是框架内部各模块之间传递的结构化数据载体，
也是 CLI JSON 输出的序列化来源。

Phase 6.1：新增结构化错误码（error_code）和机器可读状态摘要（summary）。
"""

from dataclasses import dataclass, field
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
