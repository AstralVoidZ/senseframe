"""senseframe.engine.callbacks：框架级 Lightning Callback 包。

v2 差距 3：PSNREarlyStoppingCallback 从脚本级沉淀为框架级 Callback。
"""
from .psnr_early_stopping import PSNREarlyStoppingCallback, compute_psnr

__all__ = [
    "PSNREarlyStoppingCallback",
    "compute_psnr",
]
