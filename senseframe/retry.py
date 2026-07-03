"""
自愈重试机制：基于结构化 error_code 的策略化重试。

设计目标：
- 不侵入 runner.py：retry 作为上层编排，runner 保持单次执行语义
- 基于 error_code 决策：OOM 降 batch_size 重试，瞬时 IO 错误重试，配置错误快速失败
- config 修改可追溯：每次重试的 config 变更记入 retries 字段
- Agent 友好：retries 字段让 Agent 知道"已尝试过什么"，避免重复降级

重试策略：
- OOM_ERROR：降 batch_size（每次减半，最低 4），最多重试 2 次
- DATA_LOAD_ERROR：简单重试（可能是瞬时 IO），最多 1 次，延迟 5s
- 其他错误码：不重试（快速失败）
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .schemas import TrainOutput

logger = logging.getLogger(__name__)


# ============================================================
# 重试策略定义
# ============================================================
@dataclass
class RetryPolicy:
    """单个错误码的重试策略。"""
    max_retries: int
    strategy: str  # "reduce_batch_size" | "retry"
    min_batch_size: int = 4
    delay_seconds: float = 0.0
    batch_size_divisor: int = 2  # 每次降 batch_size 的除数


# 默认重试策略表
DEFAULT_RETRY_POLICIES: Dict[str, Optional[RetryPolicy]] = {
    "OOM_ERROR": RetryPolicy(
        max_retries=2,
        strategy="reduce_batch_size",
        min_batch_size=4,
        batch_size_divisor=2,
    ),
    "DATA_LOAD_ERROR": RetryPolicy(
        max_retries=1,
        strategy="retry",
        delay_seconds=5.0,
    ),
    # 以下错误码不重试（快速失败）
    "CONFIG_VALIDATION_ERROR": None,
    "SCENE_NOT_FOUND": None,
    "DATASET_NOT_SUPPORTED": None,
    "MODEL_NOT_SUPPORTED": None,
    "DATA_NOT_FOUND": None,
    "MODEL_BUILD_ERROR": None,
    "TRAINING_ERROR": None,
    "CHECKPOINT_ERROR": None,
    "SAVE_ERROR": None,
    "PREFLIGHT_ERROR": None,
    "UNKNOWN_ERROR": None,
}


@dataclass
class RetryAttempt:
    """单次重试记录。"""
    attempt: int                          # 1=首次，2=第一次重试，...
    error_code: Optional[str] = None
    error: Optional[str] = None
    config_diff: Dict[str, Any] = field(default_factory=dict)  # 本次相对上次的 config 变更
    status: str = "error"                 # error | success


@dataclass
class RetryResult:
    """重试整体结果。"""
    final_output: TrainOutput
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_retries: int = 0
    succeeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_retries": self.total_retries,
            "succeeded": self.succeeded,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "error_code": a.error_code,
                    "error": a.error,
                    "config_diff": a.config_diff,
                    "status": a.status,
                }
                for a in self.attempts
            ],
        }


# ============================================================
# Config 修改器
# ============================================================
def _apply_policy_to_config(
    config,
    policy: RetryPolicy,
    attempt: int,
) -> Dict[str, Any]:
    """
    根据策略修改 config，返回变更记录。

    Args:
        config: ExperimentConfig（会被原地修改）
        policy: 重试策略
        attempt: 当前重试序号（1=第一次重试）

    Returns:
        config_diff: 变更记录，如 {"trainer.batch_size": [32, 16]}
    """
    config_diff: Dict[str, Any] = {}

    if policy.strategy == "reduce_batch_size":
        old_bs = config.trainer.batch_size
        new_bs = max(old_bs // policy.batch_size_divisor, policy.min_batch_size)
        config.trainer.batch_size = new_bs
        config_diff["trainer.batch_size"] = [old_bs, new_bs]
        logger.info(
            f"Retry {attempt}: reduce batch_size {old_bs} → {new_bs}"
        )

    return config_diff


def _make_output_dir_unique(output_dir: str, attempt: int) -> str:
    """为重试生成独立 output_dir，避免覆盖。"""
    if attempt <= 1:
        return output_dir
    return f"{output_dir}_retry{attempt}"


# ============================================================
# 主接口
# ============================================================
def run_experiment_with_retry(
    config,
    run_fn: Callable,
    policies: Optional[Dict[str, Optional[RetryPolicy]]] = None,
) -> RetryResult:
    """
    带自愈重试的实验执行。

    Args:
        config: ExperimentConfig
        run_fn: 执行函数，签名 run_fn(config) -> TrainOutput（通常就是 run_experiment）
        policies: 自定义重试策略表，None 用默认

    Returns:
        RetryResult: 含最终输出与所有重试记录
    """
    policies = policies or DEFAULT_RETRY_POLICIES
    attempts: List[RetryAttempt] = []
    total_retries = 0
    original_output_dir = config.output_dir
    original_batch_size = config.trainer.batch_size

    attempt_num = 0
    while True:
        attempt_num += 1
        attempt = RetryAttempt(attempt=attempt_num)

        # 为重试生成独立 output_dir
        if attempt_num > 1:
            config.output_dir = _make_output_dir_unique(
                original_output_dir, attempt_num
            )

        # 执行
        output = run_fn(config)
        attempts.append(attempt)

        if output.status == "success":
            attempt.status = "success"
            result = RetryResult(
                final_output=output,
                attempts=attempts,
                total_retries=total_retries,
                succeeded=True,
            )
            # 把重试记录附到最终输出
            output.retries = result.to_dict()
            return result

        # 失败：检查 error_code 是否有重试策略
        attempt.error_code = output.error_code
        attempt.error = output.error

        error_code = output.error_code or "UNKNOWN_ERROR"
        policy = policies.get(error_code)

        if policy is None:
            # 无重试策略，快速失败
            logger.info(
                f"Attempt {attempt_num} failed with {error_code}, no retry policy"
            )
            result = RetryResult(
                final_output=output,
                attempts=attempts,
                total_retries=total_retries,
                succeeded=False,
            )
            output.retries = result.to_dict()
            return result

        # 检查是否超过最大重试次数
        if total_retries >= policy.max_retries:
            logger.info(
                f"Attempt {attempt_num} failed with {error_code}, "
                f"max_retries={policy.max_retries} exhausted"
            )
            result = RetryResult(
                final_output=output,
                attempts=attempts,
                total_retries=total_retries,
                succeeded=False,
            )
            output.retries = result.to_dict()
            return result

        # 应用策略修改 config
        total_retries += 1
        next_attempt = attempt_num + 1
        config_diff = _apply_policy_to_config(config, policy, next_attempt)
        attempt.config_diff = config_diff

        # 延迟
        if policy.delay_seconds > 0:
            logger.info(f"Retry in {policy.delay_seconds}s...")
            time.sleep(policy.delay_seconds)

        logger.info(
            f"Attempt {attempt_num} failed with {error_code}, "
            f"retrying (attempt {next_attempt}) with config_diff={config_diff}"
        )


__all__ = [
    "RetryPolicy",
    "RetryAttempt",
    "RetryResult",
    "DEFAULT_RETRY_POLICIES",
    "run_experiment_with_retry",
]
