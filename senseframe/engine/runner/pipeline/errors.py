"""Pipeline 运行时异常分类。

包含：
- _classify_runtime_error：根据异常类型和 stage 上下文重新分类为具体 SenseFrame 异常类
"""
from __future__ import annotations

import torch

from ..errors import (
    OOMError,
    ModelBuildError,
    TrainingError,
    DataCorruptedError,
    CheckpointError,
    SaveError,
)


def _classify_runtime_error(e: Exception, stage_name: str) -> Exception:
    """根据异常类型和 stage 上下文重新分类为具体 SenseFrame 异常类（任务4）。

    Pipeline.run 的 except 块捕获 Exception 后，调用此函数将通用异常
    包装为具体异常类（OOMError/ModelBuildError/TrainingError/
    DataCorruptedError/CheckpointError/SaveError），使 Agent 可基于
    异常类型做精确恢复决策（如 OOM → 降 batch_size，Checkpoint → 删旧 ckpt）。

    分类优先级（从高到低）：
    1. torch.cuda.OutOfMemoryError → OOMError
    2. stage="build" + RuntimeError → ModelBuildError
    3. stage="train" + RuntimeError → TrainingError
    4. stage="load" + "corrupt" in msg → DataCorruptedError
    5. "checkpoint" / ".ckpt" in msg → CheckpointError
    6. "save" / "permission" in msg → SaveError
    7. 其他 → 保持原异常

    Args:
        e: 原始异常
        stage_name: 当前执行的 stage 名（如 "build" / "train" / "load"）

    Returns:
        重新分类后的异常（具体异常类实例或原异常）
    """
    msg = str(e).lower()

    # 1. torch.cuda.OutOfMemoryError → OOMError（最高优先级，任何 stage 都需精确识别）
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", type(None))):
        return OOMError(str(e))

    # 2. stage="build" 且 RuntimeError → ModelBuildError
    if stage_name == "build" and isinstance(e, RuntimeError):
        return ModelBuildError(str(e))

    # 3. stage="train" 且 RuntimeError → TrainingError
    if stage_name == "train" and isinstance(e, RuntimeError):
        return TrainingError(str(e))

    # 4. stage="load" 且 "corrupt" in msg → DataCorruptedError
    if stage_name == "load" and "corrupt" in msg:
        return DataCorruptedError(str(e))

    # 5. "checkpoint" / ".ckpt" in msg → CheckpointError
    if "checkpoint" in msg or ".ckpt" in msg:
        return CheckpointError(str(e))

    # 6. "save" / "permission" in msg → SaveError
    if "save" in msg or "permission" in msg:
        return SaveError(str(e))

    # 7. 其他 → 保持原异常
    return e
