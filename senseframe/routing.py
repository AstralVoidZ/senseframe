"""
资源路由模块：探测硬件 → 查资源配置表 → 输出路由配置 + 过滤可用模型。

路由级别：
- cpu_minimal:  纯CPU + 内存<2GB，只支持极轻量模型
- cpu_standard: 纯CPU + 内存≥2GB，支持大部分模型（除 GPU-only）
- gpu_entry:    GPU显存<4GB，入门级 GPU
- gpu_standard: GPU显存4-8GB，标准 GPU
- gpu_high:     GPU显存≥8GB，高端 GPU，支持全部模型
- mps_standard: Apple Silicon MPS，统一内存<16GB
- mps_high:     Apple Silicon MPS，统一内存≥16GB

P3: 支持运行时扩展路由级别（register_route_level）。
"""

from typing import Any, Dict, List, Optional

import psutil
import torch

from .registry import MODEL_TABLE
from .schemas import ResourceReport


# ============================================================
# 结构化资源配置表
# ============================================================
RESOURCE_ROUTES: Dict[str, Dict[str, Any]] = {
    "cpu_minimal": {
        "device": "cpu",
        "max_params_m": 1.0,
        "max_epochs": 50,
        "batch_size": 32,
        "num_workers": 0,
        "precision": "32",
    },
    "cpu_standard": {
        "device": "cpu",
        "max_params_m": 25.0,
        "max_epochs": 200,
        "batch_size": 64,
        "num_workers": 0,
        "precision": "32",
    },
    "gpu_entry": {
        "device": "cuda",
        "min_vram_mb": 2048,
        "max_params_m": 50.0,
        "batch_size": 64,
        "num_workers": 4,
        "precision": "16-mixed",
    },
    "gpu_standard": {
        "device": "cuda",
        "min_vram_mb": 4096,
        "max_params_m": 100.0,
        "batch_size": 128,
        "num_workers": 4,
        "precision": "16-mixed",
    },
    "gpu_high": {
        "device": "cuda",
        "min_vram_mb": 8192,
        "max_params_m": float("inf"),
        "batch_size": 256,
        "num_workers": 8,
        "precision": "16-mixed",
    },
    # P3: Apple Silicon MPS 路由（统一内存，与 CPU 共享）
    "mps_standard": {
        "device": "mps",
        "max_params_m": 50.0,
        "max_epochs": 200,
        "batch_size": 64,
        "num_workers": 0,
        "precision": "16-mixed",
    },
    "mps_high": {
        "device": "mps",
        "max_params_m": 100.0,
        "max_epochs": 200,
        "batch_size": 128,
        "num_workers": 2,
        "precision": "16-mixed",
    },
}

# P3: 运行时扩展路由级别（register_route_level 注册到此表）
_EXTENSION_ROUTES: Dict[str, Dict[str, Any]] = {}


# ============================================================
# 资源探测器
# ============================================================
class ResourceProbe:
    """探测硬件资源：CPU/GPU/内存。"""

    @staticmethod
    def _get_device_memory(props) -> Optional[int]:
        """健壮的显存探测，跨后端兼容（属性探测契约）。

        不假设后端 API 的属性名稳定，用 getattr + 多名探测 + 默认值。
        探测顺序：total_memory（PyTorch CUDA 标准名）→ total_mem（历史兼容）→ None。
        None 表示无法探测，调用方降级到 CPU 路由。

        See: RFC-004 方案 A — 显存探测健壮性
        """
        for attr in ("total_memory", "total_mem"):
            val = getattr(props, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    @staticmethod
    def probe() -> ResourceReport:
        has_cuda = torch.cuda.is_available()
        gpu_name = None
        gpu_total_vram_mb = None
        gpu_free_vram_mb = None

        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_bytes = ResourceProbe._get_device_memory(props)
            if total_bytes is not None:
                gpu_total_vram_mb = int(total_bytes / 1024 / 1024)
            gpu_free_vram_mb = int(torch.cuda.mem_get_info(0)[0] / 1024 / 1024)

        # P3: Apple Silicon MPS 探测
        has_mps = (
            not has_cuda
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        cpu_count = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        cpu_memory_total_mb = int(mem.total / 1024 / 1024)
        cpu_memory_available_mb = int(mem.available / 1024 / 1024)

        return ResourceReport(
            has_cuda=has_cuda,
            gpu_name=gpu_name,
            gpu_total_vram_mb=gpu_total_vram_mb,
            gpu_free_vram_mb=gpu_free_vram_mb,
            cpu_count=cpu_count,
            cpu_memory_total_mb=cpu_memory_total_mb,
            cpu_memory_available_mb=cpu_memory_available_mb,
            has_mps=has_mps,
        )


# ============================================================
# 资源路由器
# ============================================================
# RFC-002 阶段 K：路由策略注册表，Agent 可覆盖路由决策
_ROUTE_POLICIES: Dict[str, Any] = {}


class ResourceRouter:
    """根据资源探测结果确定路由级别和配置。

    RFC-002 阶段 K：新增 register_route_policy，Agent 可声明式覆盖路由。
    P3：新增 register_route_level，支持运行时扩展路由配置表。
    """

    @staticmethod
    def register_route_policy(name: str, policy_fn: Any) -> None:
        """注册路由策略（RFC-002 阶段 K）。

        Agent 可注册自定义路由策略，覆盖默认静态路由。

        Args:
            name: 策略名（如 "custom_gpu_route"）
            policy_fn: Callable(report: ResourceReport) -> str，返回路由级别
        """
        _ROUTE_POLICIES[name] = policy_fn

    @staticmethod
    def register_route_level(name: str, config: Dict[str, Any]) -> None:
        """注册自定义路由级别（P3: 动态路由表扩展）。

        Agent 可运行时注册新硬件的路由配置（如 TPU、ROCm），无需改源码。
        已存在的内置路由级别不可覆盖。

        Args:
            name: 路由级别名（如 "tpu_standard"）
            config: 路由配置 dict，需包含以下键：
                - device: 设备类型（如 "tpu" / "cuda" / "cpu" / "mps"）
                - max_params_m: 最大参数量（M）
                - batch_size: 默认 batch_size
                - num_workers: 默认 dataloader workers
                - precision: 默认精度（如 "16-mixed" / "32"）
                - max_epochs: 最大 epoch 数
        """
        if name in RESOURCE_ROUTES:
            import logging
            logging.getLogger(__name__).warning(
                f"Route level '{name}' is built-in, skipping registration"
            )
            return
        _EXTENSION_ROUTES[name] = config

    @staticmethod
    def route(report: ResourceReport) -> str:
        """确定路由级别。

        RFC-002 阶段 K：优先查询已注册路由策略，否则用默认静态路由。
        P3：MPS 设备路由到 mps_standard / mps_high。
        """
        # 优先查询已注册策略（最后注册的优先）
        for name, policy_fn in reversed(list(_ROUTE_POLICIES.items())):
            try:
                result = policy_fn(report)
                if result is not None:
                    return result
            except Exception:
                continue

        # 默认静态路由
        if report.has_cuda:
            vram = report.gpu_total_vram_mb or 0
            if vram < 4096:
                return "gpu_entry"
            elif vram < 8192:
                return "gpu_standard"
            else:
                return "gpu_high"

        # P3: Apple Silicon MPS 路由（按统一内存总量分级）
        if report.has_mps:
            mem = report.cpu_memory_total_mb or 0
            if mem < 16 * 1024:
                return "mps_standard"
            return "mps_high"

        # CPU 路由
        if report.cpu_memory_available_mb < 2048:
            return "cpu_minimal"
        return "cpu_standard"

    @staticmethod
    def get_route_config(route_level: str) -> Dict[str, Any]:
        """查路由配置表（内置 + 扩展）获取路由配置。"""
        if route_level in RESOURCE_ROUTES:
            return RESOURCE_ROUTES[route_level].copy()
        if route_level in _EXTENSION_ROUTES:
            return _EXTENSION_ROUTES[route_level].copy()
        raise ValueError(f"Unknown route level: {route_level}")

    @staticmethod
    def filter_models(route_level: str) -> List[str]:
        """
        根据路由级别过滤可用模型 ID。

        过滤规则：
        - requires_gpu=True 的模型在 CPU 路由下不可用
        - estimated_params_m 超过路由 max_params_m 的模型不可用

        注意：estimated_vram_mb 的运行时显存过滤在 runner._preflight_check 中执行
        （需要 ResourceReport 的空闲显存数据，此处仅按路由级别静态过滤）。
        """
        route_config = ResourceRouter.get_route_config(route_level)
        device = route_config["device"]
        max_params = route_config["max_params_m"]

        available = []
        for model_id, info in MODEL_TABLE.items():
            if not info["enabled"]:
                continue
            # GPU-only 模型在 CPU 路由下不可用（MPS 视为 GPU 类设备，可用）
            if device == "cpu" and info["requires_gpu"]:
                continue
            # 参数量超限
            if info["estimated_params_m"] > max_params:
                continue
            available.append(model_id)

        return available

    @staticmethod
    def resolve_config(
        yaml_config: Dict[str, Any],
        route_config: Dict[str, Any],
        model_info: Dict[str, Any],
        report: ResourceReport,
    ) -> Dict[str, Any]:
        """
        合并配置：YAML 非 null 优先 → 路由配置 → 模型默认值。

        Args:
            yaml_config: YAML 配置（可能含 null 值）
            route_config: 路由配置表
            model_info: 模型信息（含 default_lr 等）
            report: 资源探测结果

        Returns:
            合并后的最终配置
        """
        resolved = {}

        # device: YAML > 路由
        # P5 P2-4：用 is not None 判断，避免空字符串被 falsy 错误回退
        yaml_device = yaml_config.get("device")
        if yaml_device is not None and yaml_device != "auto":
            resolved["device"] = yaml_device
        else:
            resolved["device"] = route_config["device"]

        # batch_size: YAML > 路由
        # P2 隐藏 bug 修复：用 is not None 判断，避免 batch_size=0 被 or 当 falsy 错误回退
        # （0 虽不合法，但应由下游校验报错，而非静默回退到路由默认值）
        yaml_bs = yaml_config.get("batch_size")
        if yaml_bs is not None:
            resolved["batch_size"] = yaml_bs
        else:
            resolved["batch_size"] = route_config["batch_size"]

        # num_workers: 路由 + 平台感知 + 上限保护
        # 优化 5：Windows CPU 环境尝试 2 个 worker 加速数据加载
        # （原 CPU 路由强制 0，导致数据加载瓶颈；Windows 下 num_workers>0
        #   需要 if __name__ == "__main__" 保护，runner.py 已满足此条件）
        import os as _os
        import sys as _sys
        _base_workers = route_config["num_workers"]
        if _sys.platform == "win32" and _base_workers == 0 and route_config["device"] == "cpu":
            # Windows CPU 环境：尝试 2 个 worker（persistent_workers 加速）
            # 多进程启动失败时由 datamodule._safe_dataloader 自动降级到 num_workers=0
            _base_workers = 2
        resolved["num_workers"] = min(_base_workers, _os.cpu_count() or 1, 8)
        # 用户显式配置 num_workers 优先于 routing 派生
        _yaml_nw = yaml_config.get("num_workers")
        if _yaml_nw is not None:
            resolved["num_workers"] = int(_yaml_nw)
        # pin_memory 仅 GPU 路由启用，加速 CPU→GPU 数据搬运
        resolved["pin_memory"] = route_config["device"] == "cuda"
        # persistent_workers 避免每 epoch 重建 worker（num_workers>0 时）
        resolved["persistent_workers"] = resolved["num_workers"] > 0

        # precision: Phase 1.2b — 支持 bf16-mixed
        # YAML mixed_precision: true=16-mixed, false=32, 字符串直接透传（如 "bf16-mixed"）
        yaml_precision = yaml_config.get("mixed_precision")
        if yaml_precision is True:
            resolved["precision"] = "16-mixed"
        elif yaml_precision is False:
            resolved["precision"] = "32"
        elif isinstance(yaml_precision, str):
            # Phase 1.2b：支持直接指定精度字符串（如 "bf16-mixed" / "16-mixed" / "32"）
            resolved["precision"] = yaml_precision
        else:
            resolved["precision"] = route_config["precision"]

        # learning_rate: Phase 1.1b 三级回退（YAML > 模型 default_lr > 1e-3）
        # 修复：TrainerConfig.learning_rate 默认 None，不再短路 model default_lr
        yaml_lr = yaml_config.get("learning_rate")
        if yaml_lr is not None:
            resolved["learning_rate"] = yaml_lr
        else:
            resolved["learning_rate"] = model_info.get("default_lr", 1e-3)

        # optimizer: YAML > 默认 adam
        # P5 P2-4：用 is not None 判断，避免空字符串被 falsy 错误回退
        yaml_opt = yaml_config.get("optimizer")
        resolved["optimizer"] = yaml_opt if yaml_opt is not None else "adam"

        # weight_decay: YAML > 默认 0.0
        # P2 隐藏 bug 修复：用 is not None 判断，避免 weight_decay=0.0 被 or 当 falsy 错误回退
        # （0.0 是合法值，表示无权重衰减，不应回退到路由默认值）
        yaml_wd = yaml_config.get("weight_decay")
        if yaml_wd is not None:
            resolved["weight_decay"] = yaml_wd
        else:
            resolved["weight_decay"] = 0.0

        # scheduler: Phase 1.1a — YAML trainer.scheduler > scene.params.scheduler > null
        # 优先读 trainer 级 scheduler（新路径），回退到 scene.params 透传（向后兼容）
        resolved["scheduler"] = yaml_config.get("scheduler")

        # early_stopping: YAML > 默认 null
        resolved["early_stopping"] = yaml_config.get("early_stopping")

        # Phase 1.2a：梯度裁剪与累积
        resolved["gradient_clip_val"] = yaml_config.get("gradient_clip_val")
        resolved["gradient_clip_algorithm"] = (
            yaml_config.get("gradient_clip_algorithm")
            if yaml_config.get("gradient_clip_algorithm") is not None
            else "norm"
        )
        # P2 隐藏 bug 修复：用 is not None 判断，避免 accumulate_grad_batches=0 被 or 当 falsy 错误回退
        # （0 虽不合法，但应由下游校验报错，而非静默回退到 1）
        yaml_agb = yaml_config.get("accumulate_grad_batches")
        if yaml_agb is not None:
            resolved["accumulate_grad_batches"] = yaml_agb
        else:
            resolved["accumulate_grad_batches"] = 1

        # Phase 2.2a：logger 后端透传（默认 csv）
        # P5 P2-4：用 is not None 判断，避免空字符串被 falsy 错误回退
        yaml_logger = yaml_config.get("logger")
        resolved["logger"] = yaml_logger if yaml_logger is not None else "csv"

        # Phase 4.3：分布式训练配置透传
        # devices: GPU 数量（int 或 "auto"），默认 1（单卡，向后兼容）
        resolved["devices"] = yaml_config.get("devices", 1)
        # strategy: 分布式策略，None=单设备，"ddp"/"ddp_spawn"/"fsdp" 等
        resolved["strategy"] = yaml_config.get("strategy")
        # num_nodes: 多节点训练节点数，默认 1
        resolved["num_nodes"] = yaml_config.get("num_nodes", 1)
        # sync_batchnorm: 分布式同步 BN，默认 False
        resolved["sync_batchnorm"] = yaml_config.get("sync_batchnorm", False)
        # num_processes: CPU 并行进程数（仅 CPU 模式），默认 1
        resolved["num_processes"] = yaml_config.get("num_processes", 1)

        # route_level
        resolved["route_level"] = ResourceRouter.route(report)

        # P4-1：透传非路由字段（如 self_supervised_epochs/metrics 等 scene.params 透传值）。
        # 路由层只负责 device/batch_size/precision 等路由相关字段的覆盖，
        # 非路由字段应由上游 experiment_config_to_dict 透传到下游 stage 消费。
        # 旧实现是白名单模式（只复制路由字段），导致 self_supervised_epochs/metrics 被丢弃。
        _ROUTE_OWNED_FIELDS = {
            "device", "batch_size", "num_workers", "pin_memory",
            "persistent_workers", "precision", "learning_rate",
            "optimizer", "weight_decay", "scheduler", "early_stopping",
            "gradient_clip_val", "gradient_clip_algorithm",
            "accumulate_grad_batches", "logger", "devices", "strategy",
            "num_nodes", "sync_batchnorm", "num_processes", "route_level",
            "mixed_precision",  # 已被 precision 消费
        }
        for k, v in yaml_config.items():
            if k not in _ROUTE_OWNED_FIELDS and k not in resolved:
                resolved[k] = v

        return resolved

    @staticmethod
    def to_lightning_params(resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将路由配置映射为 Lightning Trainer 参数。

        Phase 4.3：支持多卡分布式训练：
        - devices: 从 resolved_config 读取，支持 int（如 2=2卡）或 "auto"
        - strategy: 从 resolved_config 读取，支持 "ddp" / "ddp_spawn" / "fsdp" 等
        - num_nodes: 多节点训练
        - sync_batchnorm: 分布式同步 BN
        """
        device = resolved_config["device"]
        if device == "cpu":
            accelerator = "cpu"
            # Phase 4.3：CPU 支持 num_processes 并行
            devices = resolved_config.get("num_processes", 1)
        elif device == "mps":
            # P3: Apple Silicon MPS
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "gpu"
            # Phase 4.3：GPU 支持多卡，默认 1，可由配置覆盖
            devices = resolved_config.get("devices", 1)

        params = {
            "accelerator": accelerator,
            "devices": devices,
            "precision": resolved_config["precision"],
        }

        # Phase 4.3：分布式策略（仅 devices > 1 或显式指定时生效）
        strategy = resolved_config.get("strategy")
        if strategy is not None:
            params["strategy"] = strategy

        # Phase 4.3：多节点训练
        num_nodes = resolved_config.get("num_nodes")
        if num_nodes is not None and num_nodes > 1:
            params["num_nodes"] = num_nodes

        # Phase 4.3：同步 BatchNorm（分布式训练常用）
        sync_bn = resolved_config.get("sync_batchnorm")
        if sync_bn:
            params["sync_batchnorm"] = True

        return params
