"""不透明 cursor 编码/解码（设计文档 0.6 节 + pipeflow §6.18 对齐）。

Cursor wire format::

    base64-urlsafe(last_id || "|" || filter_fingerprint)  # padding stripped

`last_id` 是 str（SenseFrame 的 run_id 是 uuid4 hex 字符串）。
`filter_fingerprint` 是 `sha256(canonical_json(filter_dict))[:8]`（8 字符 hex）。
当 filter_dict 为空 / None 时 fingerprint 是字面量 `"00000000"`。

客户端必须将 cursor 视为不透明 — 不能解析、修改或跨会话持久化。
服务端使用嵌入的 `filter_fingerprint` 检测 cursor 与 filter_dict 不匹配。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any

from senseframe.mcp.errors import CursorFilterMismatch, InvalidCursor

__all__ = [
    "EMPTY_FINGERPRINT",
    "decode_cursor",
    "encode_cursor",
    "filter_fingerprint",
    "assert_fingerprint_matches",
]

EMPTY_FINGERPRINT = "00000000"
_FINGERPRINT_LEN = 8


def _canonical_json(d: dict[str, Any] | None) -> str:
    """稳定 JSON 序列化：sort_keys + 最小分隔符。

    保证同样的 dict 内容总是产生同样的字符串（无视插入顺序）。
    """
    if not d:
        return ""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def filter_fingerprint(filter_dict: dict[str, Any] | None) -> str:
    """返回 filter_dict 的 8 字符 hex fingerprint。

    None 和 {} 都归一化为 `EMPTY_FINGERPRINT`。
    """
    canonical = _canonical_json(filter_dict)
    if not canonical:
        return EMPTY_FINGERPRINT
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_LEN]


def encode_cursor(last_id: str, filter_dict: dict[str, Any] | None) -> str:
    """将 `(last_id, filter_dict)` 编码为不透明 base64-urlsafe cursor。

    Args:
        last_id: str（SenseFrame run_id 是 uuid4 hex 字符串）。
        filter_dict: 当前请求的 filter，用于嵌入 fingerprint。

    Returns:
        base64-urlsafe 编码字符串（无 padding）。
    """
    fingerprint = filter_fingerprint(filter_dict)
    raw = f"{last_id!r}|{fingerprint}".encode()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, str]:
    """解码不透明 cursor，返回 `(last_id, fingerprint)`。

    Args:
        cursor: base64-urlsafe 编码字符串。

    Returns:
        `(last_id, fingerprint)` 元组。

    Raises:
        InvalidCursor: base64 解码失败 / 缺少分隔符 / 空输入。
    """
    if not isinstance(cursor, str) or not cursor:
        raise InvalidCursor("cursor must be a non-empty string")
    # 拒绝空白 / 控制字符（防御性）
    if any(ch.isspace() or ord(ch) < 0x20 for ch in cursor):
        raise InvalidCursor("cursor contains whitespace or control characters")
    padding = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidCursor("cursor base64 decode failed") from exc
    if "|" not in raw:
        raise InvalidCursor("cursor missing separator")
    last_id_part, _, fingerprint = raw.partition("|")
    if not fingerprint:
        raise InvalidCursor("cursor missing fingerprint")
    last_id = _parse_last_id(last_id_part)
    return last_id, fingerprint


def _parse_last_id(part: str) -> str:
    """解析 last_id wire form（Python `repr()` 输出）。

    `repr(str)` 输出 `'...'`（含引号）；本函数剥离引号返回原始 str。
    数值形式（全数字 / `-` 开头）按 str 保留（SenseFrame run_id 一律 str）。
    """
    # 匹配单/双引号包裹的字符串字面量
    if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"'):
        return part[1:-1]
    return part


def assert_fingerprint_matches(
    cursor: str | None, filter_dict: dict[str, Any] | None
) -> str | None:
    """校验 cursor 的 fingerprint 与当前请求的 filter 一致。

    Args:
        cursor: 客户端传入的 cursor，None 表示首次请求。
        filter_dict: 当前请求的 filter。

    Returns:
        从 cursor 解码出的 `last_id`，cursor 为 None 时返回 None。

    Raises:
        CursorFilterMismatch: cursor 的 fingerprint 与当前 filter 不一致。
        InvalidCursor: cursor 解码失败。
    """
    if cursor is None:
        return None
    last_id, fingerprint = decode_cursor(cursor)
    expected = filter_fingerprint(filter_dict)
    if fingerprint != expected:
        raise CursorFilterMismatch(
            "cursor filter fingerprint does not match current filter; restart from cursor=None"
        )
    return last_id
