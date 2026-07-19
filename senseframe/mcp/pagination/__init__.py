"""不透明 cursor 分页（设计文档 0.6 节 + pipeflow §6.18 对齐）。

公开 API：
- cursor.encode_cursor / decode_cursor / filter_fingerprint / assert_fingerprint_matches
- page.Page / build_page / clamp_limit
"""

from senseframe.mcp.pagination import cursor, page

__all__ = ["cursor", "page"]
