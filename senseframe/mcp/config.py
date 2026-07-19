"""MCP 服务器环境配置 + 日志初始化（stderr only）。

借鉴 pipeflow config.py 的「一次性收集所有错误」设计，
但去掉 DB 相关逻辑（SenseFrame 暂不引入 SQLite）。

可配置环境变量：
- SENSEFRAME_LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL，默认 INFO）
- SENSEFRAME_LOG_FORMAT: 日志格式（text/json，默认 text）
- SENSEFRAME_RATE_LIMIT: per-tool 限流（calls/min，0 禁用，默认 60）
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "senseframe_mcp"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RATE_LIMIT = "60"

VALID_LOG_FORMATS = ("text", "json")


def log_level() -> str:
    """返回日志级别字符串（大写）。"""
    return os.environ.get("SENSEFRAME_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


def log_format() -> str:
    """返回日志格式（text 或 json）。"""
    return os.environ.get("SENSEFRAME_LOG_FORMAT", "text").lower()


def rate_limit() -> int:
    """返回 per-tool 限流（calls/min），0 表示禁用。

    解析失败时回退到默认值 60，保证服务器可启动。
    """
    raw = os.environ.get("SENSEFRAME_RATE_LIMIT", DEFAULT_RATE_LIMIT)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(DEFAULT_RATE_LIMIT)


def validate_config() -> None:
    """启动时一次性校验所有环境变量，收集全部错误后再 raise。

    Raises:
        RuntimeError: 任何一个环境变量校验失败时抛出，错误信息含全部失败项。
    """
    errors: list[str] = []

    # --- SENSEFRAME_LOG_LEVEL ---
    raw_level = log_level()
    if raw_level not in logging._nameToLevel:
        errors.append(
            f"SENSEFRAME_LOG_LEVEL: {raw_level!r} is not a valid Python log level "
            f"(DEBUG, INFO, WARNING, ERROR, CRITICAL)"
        )

    # --- SENSEFRAME_LOG_FORMAT ---
    raw_fmt = log_format()
    if raw_fmt not in VALID_LOG_FORMATS:
        errors.append(
            f"SENSEFRAME_LOG_FORMAT: {raw_fmt!r} is not valid "
            f"(must be one of {VALID_LOG_FORMATS})"
        )

    # --- SENSEFRAME_RATE_LIMIT ---
    raw_rate = os.environ.get("SENSEFRAME_RATE_LIMIT", DEFAULT_RATE_LIMIT)
    try:
        rate_val = int(raw_rate)
        if rate_val < 0:
            errors.append(
                f"SENSEFRAME_RATE_LIMIT: {rate_val} is negative (must be >= 0)"
            )
    except (TypeError, ValueError):
        errors.append(
            f"SENSEFRAME_RATE_LIMIT: {raw_rate!r} is not a valid integer"
        )

    if errors:
        raise RuntimeError(
            "configuration validation failed:\n  " + "\n  ".join(errors)
        )


def configure_logging() -> None:
    """配置 root logger 写 stderr only（stdout 保留给 JSON-RPC）。

    SENSEFRAME_LOG_FORMAT=json 时输出 JSON 行；默认纯文本。
    """
    level = getattr(logging, log_level(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger(LOGGER_NAME)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
