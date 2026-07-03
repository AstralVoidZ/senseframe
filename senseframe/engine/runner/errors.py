"""
异常层级体系 + 错误分类模块。

SenseFrameError 层级使 stage 能抛出结构化异常（携带 error_code），
classify_error 优先走 isinstance 检查，消除字符串匹配的脆弱性。
非 SenseFrame 异常（torch/pl/系统异常）走 heuristic 兜底。
"""

import json
import torch
from typing import Optional


# ============================================================
# SenseFrame 异常层级体系
# ============================================================

class SenseFrameError(Exception):
    """所有 SenseFrame 异常的基类。

    子类通过 error_code 类属性声明结构化错误码，
    classify_error 优先读取此属性，消除字符串匹配。
    """
    error_code: str = "UNKNOWN_ERROR"


class SceneNotRegisteredError(SenseFrameError):
    error_code = "SCENE_NOT_FOUND"


class DatasetNotSupportedError(SenseFrameError):
    error_code = "DATASET_NOT_SUPPORTED"


class ModelNotSupportedError(SenseFrameError):
    error_code = "MODEL_NOT_SUPPORTED"


class DataNotFoundError(SenseFrameError, FileNotFoundError):
    error_code = "DATA_NOT_FOUND"


class DataCorruptedError(SenseFrameError):
    error_code = "DATA_LOAD_ERROR"


class OOMError(SenseFrameError):
    error_code = "OOM_ERROR"


class CheckpointError(SenseFrameError):
    error_code = "CHECKPOINT_ERROR"


class PreflightError(SenseFrameError):
    error_code = "PREFLIGHT_ERROR"


class TrainingError(SenseFrameError):
    error_code = "TRAINING_ERROR"


class ModelBuildError(SenseFrameError):
    error_code = "MODEL_BUILD_ERROR"


class SaveError(SenseFrameError, OSError):
    error_code = "SAVE_ERROR"


class ConfigValidationError(SenseFrameError, ValueError):
    error_code = "CONFIG_VALIDATION_ERROR"


def classify_error(exc: Exception, stage: Optional[str] = None) -> str:
    """
    根据异常类型映射到结构化错误码。

    优先级：
    1. SenseFrameError 子类 → 直接返回 exc.error_code（isinstance 检查，无字符串匹配）
    2. torch.cuda.OutOfMemoryError → OOM_ERROR
    3. 第三方异常 → heuristic 兜底（stage 感知 + 字符串匹配）
    """
    # 1. SenseFrame 异常：直接返回 error_code
    if isinstance(exc, SenseFrameError):
        return exc.error_code

    msg = str(exc).lower()

    # 2. 显存/内存不足（最高优先级，任何 stage 都需精确识别）
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", type(None))):
        return "OOM_ERROR"
    if isinstance(exc, RuntimeError) and (
        "out of memory" in msg
        or "cuda out of memory" in msg
    ):
        return "OOM_ERROR"

    # 3. 数据集文件缺失（任何 stage）
    if isinstance(exc, FileNotFoundError):
        return "DATA_NOT_FOUND"

    # 4. 数据损坏：JSONDecodeError 是 ValueError 子类，但语义是数据损坏而非配置错误
    if isinstance(exc, (json.JSONDecodeError,)):
        return "DATA_LOAD_ERROR"
    # pickle 损坏
    try:
        import pickle
        if isinstance(exc, pickle.UnpicklingError):
            return "DATA_LOAD_ERROR"
    except Exception:
        pass

    # 5. 权限问题
    if isinstance(exc, PermissionError):
        return "DATA_LOAD_ERROR"

    # 6. stage 感知分类（非 SenseFrame 异常的 heuristic 兜底）
    if stage == "validate":
        if isinstance(exc, ValueError):
            if "scene" in msg and "not registered" in msg:
                return "SCENE_NOT_FOUND"
            if "dataset" in msg and "not supported" in msg:
                return "DATASET_NOT_SUPPORTED"
            if "model" in msg and "not supported" in msg:
                return "MODEL_NOT_SUPPORTED"
            return "CONFIG_VALIDATION_ERROR"

    if stage == "load":
        if not isinstance(exc, (ValueError,)):
            return "DATA_LOAD_ERROR"
        if "shape" in msg or "format" in msg or "parse" in msg or "decode" in msg:
            return "DATA_LOAD_ERROR"
        return "DATA_LOAD_ERROR"

    if stage == "train":
        if isinstance(exc, RuntimeError):
            if "vram" in msg or "disk free space" in msg:
                return "PREFLIGHT_ERROR"
            if "checkpoint" in msg or "ckpt" in msg or ".ckpt" in msg:
                return "CHECKPOINT_ERROR"
            return "TRAINING_ERROR"

    if stage == "build":
        if isinstance(exc, (KeyError, AttributeError)):
            return "MODEL_BUILD_ERROR"

    if stage == "export":
        if isinstance(exc, (OSError, IOError)):
            return "SAVE_ERROR"

    # 7. 预检失败
    if isinstance(exc, RuntimeError) and (
        "vram" in msg or "disk free space" in msg
    ):
        return "PREFLIGHT_ERROR"

    # 8. Checkpoint 加载/保存失败
    if isinstance(exc, RuntimeError) and (
        "checkpoint" in msg or "ckpt" in msg or ".ckpt" in msg
    ):
        return "CHECKPOINT_ERROR"

    # 9. 模型保存失败
    if isinstance(exc, (OSError, IOError)) and (
        "model.pth" in msg or "metadata.json" in msg
    ):
        return "SAVE_ERROR"

    # 10. 配置校验失败
    if isinstance(exc, ValueError):
        return "CONFIG_VALIDATION_ERROR"

    # 11. 模型构建/属性访问失败
    if isinstance(exc, (KeyError, AttributeError)):
        return "MODEL_BUILD_ERROR"

    # 12. 兜底
    return "UNKNOWN_ERROR"
