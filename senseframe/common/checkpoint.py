"""Checkpoint 加载公共工具。

处理 Lightning checkpoint、backbone_state_dict 与裸 state_dict 三种格式的兼容加载，
消除 export/inference/pipeline 三处的反模式重复。

背景：
- GenericLightningModule（engine/module.py）将裸模型存为 self.model，
  导致 state_dict 内 key 带 "model." 前缀。
- Lightning checkpoint 顶层含 epoch/global_step/optimizer_states/callbacks
  等非权重字段，直接传给 model.load_state_dict 会触发
  `Unexpected key(s) in state_dict: "epoch", "global_step", ...`。
- 旧代码三处各自实现 "提取 state_dict + 剥离 model. 前缀" 逻辑，
  export.py / inference.py 漏写该逻辑导致 F 阶段 ONNX 导出失败，
  抽取公共函数根治。

Lightning checkpoint 格式：
    {
        "epoch": int,
        "global_step": int,
        "pytorch-lightning_version": str,
        "state_dict": {
            "model.<param_name>": Tensor,  # 带 "model." 前缀
            ...
        },
        "loops": ...,
        "callbacks": ...,
        "optimizer_states": ...,
        "lr_schedulers": ...,
    }

裸 state_dict 格式（torch.save(model.state_dict(), path)）：
    {"<param_name>": Tensor, ...}
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

# LightningModule 将裸模型存为 self.model，state_dict key 前缀由此产生。
_LIGHTNING_MODEL_PREFIX = "model."


def load_checkpoint_flexible(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    map_location: Any = "cpu",
    weights_only: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    从 checkpoint 加载权重到 model，兼容 Lightning checkpoint 与裸 state_dict。

    四层兼容策略：
    1. Lightning checkpoint（含 "state_dict" 顶层 key）
       - 若 state_dict 内 key 带 "model." 前缀：剥离后加载
       - 否则：直接加载 state_dict（兼容非 GenericLightningModule 的 Lightning ckpt）
    1.5 backbone_state_dict 格式（含 "backbone_state_dict" 顶层 key）
       - 自定义 MAE checkpoint（scripts/p0_pretrain_with_psnr.py 产出）
       - 直接加载 backbone_state_dict
    2. 裸 state_dict（torch.save(model.state_dict(), path)）：直接加载
    3. 其他格式：抛 RuntimeError

    Args:
        checkpoint_path: checkpoint 文件路径
        model: 待加载权重的模型实例
        map_location: torch.load 的 map_location 参数（默认 "cpu"）
        weights_only: torch.load 的 weights_only 参数。
            False（默认）：加载完整对象，兼容 Lightning ckpt 的 callbacks 字段。
            True：仅加载 tensor，更安全但无法读取含 Python 对象的 ckpt。
        strict: model.load_state_dict 的 strict 参数。
            True（默认）：要求 key 完全匹配，不匹配则抛 RuntimeError。
            False：允许 missing/unexpected keys。

    Returns:
        Dict 含加载统计信息，便于调用方记录日志：
        - "source_format": "lightning" / "backbone_state_dict" / "bare_state_dict"
        - "num_keys_loaded": 实际加载到 model 的 key 数量
        - "stripped_prefix": 被剥离的前缀（"model." / ""）
        - "raw_keys": 原始 state_dict 前 5 个 key，用于诊断

    Raises:
        FileNotFoundError: checkpoint 文件不存在
        RuntimeError: model.load_state_dict 失败（strict=True 时 key 不匹配），
                      或 checkpoint 格式无法识别
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
    )

    # 安全加固：非 Lightning 分支（backbone_state_dict / bare_state_dict）强制 weights_only=True，
    # 防止反序列化任意 Python 对象（这两种格式应只含 tensor + 基础类型）。
    # Lightning ckpt 因含 callbacks/optimizer_states 等 Python 对象，仍按调用方 weights_only 处理。
    if (
        not weights_only
        and isinstance(ckpt, dict)
        and "state_dict" not in ckpt
    ):
        ckpt = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

    # 情况 1：Lightning checkpoint，含 "state_dict" 顶层 key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
        # 剥离 "model." 前缀（GenericLightningModule.self.model 产生）
        stripped = {
            k[len(_LIGHTNING_MODEL_PREFIX):]: v
            for k, v in raw_state_dict.items()
            if k.startswith(_LIGHTNING_MODEL_PREFIX)
        }
        if stripped:
            # 标准 GenericLightningModule checkpoint
            model.load_state_dict(stripped, strict=strict)
            return {
                "source_format": "lightning",
                "num_keys_loaded": len(stripped),
                "stripped_prefix": _LIGHTNING_MODEL_PREFIX,
                "raw_keys": list(raw_state_dict.keys())[:5],
            }
        # 无 "model." 前缀：非 GenericLightningModule 的 Lightning ckpt
        # （如用户自定义 LightningModule 未将模型存为 self.model）
        # 直接加载原始 state_dict，strict 模式下若 key 不匹配由调用方处理
        _logger.info(
            "load_checkpoint_flexible: Lightning checkpoint has no '%s' prefixed keys, "
            "loading raw state_dict (path=%s)",
            _LIGHTNING_MODEL_PREFIX,
            checkpoint_path,
        )
        model.load_state_dict(raw_state_dict, strict=strict)
        return {
            "source_format": "lightning",
            "num_keys_loaded": len(raw_state_dict),
            "stripped_prefix": "",
            "raw_keys": list(raw_state_dict.keys())[:5],
        }

    # 情况 1.5：自定义 MAE checkpoint，含 "backbone_state_dict" 顶层 key
    # （scripts/p0_pretrain_with_psnr.py 产出的格式：{"backbone_state_dict": ..., "best_psnr": ...}）
    if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        backbone_state = ckpt["backbone_state_dict"]
        if not isinstance(backbone_state, dict):
            raise RuntimeError(
                f"backbone_state_dict must be a dict, got {type(backbone_state).__name__} "
                f"at {checkpoint_path}"
            )
        if not strict:
            # strict=False：跨模态迁移场景，过滤 shape 不匹配的 key
            # （PyTorch load_state_dict(strict=False) 仅容忍 missing/unexpected keys，
            # 不容忍 shape mismatch，需手动过滤）
            model_state = model.state_dict()
            loadable_state = {
                k: v for k, v in backbone_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model.load_state_dict(loadable_state, strict=False)
            return {
                "source_format": "backbone_state_dict",
                "num_keys_loaded": len(loadable_state),
                "stripped_prefix": "",
                "raw_keys": list(backbone_state.keys())[:5],
            }
        model.load_state_dict(backbone_state, strict=strict)
        return {
            "source_format": "backbone_state_dict",
            "num_keys_loaded": len(backbone_state),
            "stripped_prefix": "",
            "raw_keys": list(backbone_state.keys())[:5],
        }

    # 情况 2：裸 state_dict（torch.save(model.state_dict(), path)）
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=strict)
        return {
            "source_format": "bare_state_dict",
            "num_keys_loaded": len(ckpt),
            "stripped_prefix": "",
            "raw_keys": list(ckpt.keys())[:5],
        }

    # 情况 3：未知格式（如直接 torch.save(model, path) 保存了模型对象）
    raise RuntimeError(
        f"Unexpected checkpoint format at {checkpoint_path}: "
        f"expected dict with 'state_dict' key or bare state_dict, "
        f"got {type(ckpt).__name__}"
    )


__all__ = ["load_checkpoint_flexible"]
