"""FastMCP server entry — lifespan + signal handlers + ToolAnnotations 矩阵。

本阶段（3.1-4.3）扩展 L4 SP 搜索协议 + AutoMLOrchestrator + Artifact + Skill 工具组：
- FastMCP 实例（name="senseframe-mcp"）
- _lifespan async context manager（startup: 日志；shutdown: drain flag）
- _install_signal_handlers（SIGTERM/SIGINT，10s drain，二次信号强制 exit）
- main() 入口（validate_config + configure_logging + signal handlers + mcp.run）
- 程序化注册 29 个 tool（28 个真实 handler + 1 个 stub）+ 12 个 Resource 端点

阶段 2 已实现 7 个 pipeline tool + 12 个 Resource 端点 + HATEOAS + cursor 分页。
阶段 3 新增 11 个 tool 到 ToolAnnotations 矩阵，3 个 stub 升级为真实 handler：
- 7 个 study_* 工具组（create/ask/tell 升级 + get/list/compare/stop 新增）
- 1 个 hpo_setup tool（新增）
- 1 个 exploration_recommend tool（新增）
- 4 个 automl_* 工具组（create/advance/get/list，新增）
- 1 个 apply_params_extended tool（新增）
阶段 4.2 新增 2 个 artifact tool（list/export）+ 升级 1 个 stub（artifact_verify）：
- 3 个 artifact_* 工具组（verify/list/export）
阶段 4.3 新增 2 个 skill tool（get/search）+ 升级 2 个 stub（skill_save/skill_remove）：
- 4 个 skill_* 工具组（save/get/search/remove）
- 剩余 1 个 stub（config_parse）在 config_parse 阶段实现

ToolAnnotations 矩阵（设计文档 0.4 节）覆盖 29 个 tool，已注册到 mcp。

注意：SenseFrame 不依赖 SQLite，故 lifespan 不做 DB 连通性检查，
使用 ``_shutdown_event = threading.Event()`` 替代 pipeflow 的
``_db._shutting_down`` 做停机协调。
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from senseframe.mcp.config import configure_logging, validate_config
from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS, _TOOL_REGISTRY

log = logging.getLogger("senseframe_mcp")

# Shutdown 协调：替代 pipeflow 的 _db._shutting_down（SenseFrame 无 DB）。
# 第一信号 set 此 Event，in-flight 任务可检查此 flag 提前退出。
_shutdown_event: threading.Event = threading.Event()


@asynccontextmanager
async def _lifespan(_app: FastMCP) -> AsyncIterator[None]:
    """Startup: 初始化日志 + 验证依赖。Shutdown: 设置 drain flag。

    SenseFrame 不依赖 SQLite，故不进行 DB 连通性检查（区别于 pipeflow）。
    """
    # --- startup ---
    log.info("lifespan startup: senseframe-mcp ready")
    yield
    # --- shutdown ---
    log.info("lifespan shutdown: setting drain flag")
    _shutdown_event.set()


mcp = FastMCP(
    name="senseframe-mcp",
    instructions=(
        "SenseFrame MCP server: perception-domain AutoML primitives + "
        "validation guardrails. Agent holds control loop. "
        "BEFORE: senseframe_config_parse -> senseframe_pipeline_create. "
        "DURING: senseframe_pipeline_get -> read _transitions -> senseframe_pipeline_advance. "
        "AFTER: senseframe_artifact_verify. "
        "Long tasks (HPO/NAS) use Ask-Tell: senseframe_study_create/ask/tell."
    ),
    lifespan=_lifespan,
)


# ToolAnnotations 矩阵（设计文档 0.4 节，29 个 tool）。
# 阶段 3 扩展：新增 11 个 tool（study_get/list/compare/stop + hpo_setup +
# exploration_recommend + automl_create/advance/get/list + apply_params_extended）。
# 阶段 4.2 扩展：新增 2 个 tool（artifact_list/export）+ artifact_verify 升级为真实 handler。
# 阶段 4.3 扩展：新增 2 个 tool（skill_get/search）+ skill_save/skill_remove 升级为真实 handler。
_ANNOTATIONS: dict[str, ToolAnnotations] = {
    # --- 阶段 2：pipeline_* 工具组（7 个真实）---
    "senseframe_config_parse": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_pipeline_create": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_pipeline_advance": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "senseframe_pipeline_run": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_pipeline_get": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_pipeline_list": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_pipeline_pause": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "senseframe_pipeline_resume": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    # --- 阶段 3.1：study_* 工具组（7 个真实）---
    "senseframe_study_create": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_study_ask": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_study_tell": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "senseframe_study_get": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_study_list": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_study_compare": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_study_stop": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    # --- 阶段 3.2：hpo_setup（1 个真实）---
    "senseframe_hpo_setup": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    # --- 阶段 3.3：exploration_recommend（1 个真实）---
    "senseframe_exploration_recommend": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    # --- 阶段 3.4：automl_* 工具组（4 个真实）---
    "senseframe_automl_create": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_automl_advance": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "senseframe_automl_get": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_automl_list": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    # --- 阶段 3.5：apply_params_extended（1 个真实）---
    "senseframe_apply_params_extended": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    # --- 阶段 4.2：artifact_* 工具组（verify/list/export，3 个真实）---
    "senseframe_artifact_verify": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    # --- 阶段 4.3：skill_* 工具组（save/get/search/remove，4 个真实）---
    "senseframe_skill_save": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
    "senseframe_skill_get": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_skill_search": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_skill_remove": ToolAnnotations(
        readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
    ),
    # --- 阶段 4.2 新增：artifact_list / artifact_export（真实 handler）---
    "senseframe_artifact_list": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
    "senseframe_artifact_export": ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
    ),
}


# ── 程序化注册（参考 pipeflow server.py）─────────────────────────────
# 注册 29 个 tool（28 个真实 handler + 1 个 stub）+ 12 个 Resource 端点。
# 延迟导入 resources.__init__ 避免 server.py 启动时执行 Resource handler。
def _register_tools_and_resources() -> None:
    """注册 _TOOL_REGISTRY 中的 29 个 tool + 12 个 Resource 到 FastMCP。"""
    for _name, _desc, _fn in _TOOL_REGISTRY:
        mcp.add_tool(
            _fn, name=_name, description=_desc, annotations=_ANNOTATIONS[_name]
        )
    # 延迟导入避免循环依赖
    from senseframe.mcp.resources import _RESOURCE_REGISTRY

    for _uri, _rname, _rdesc, _rfn in _RESOURCE_REGISTRY:
        mcp.resource(_uri, name=_rname, description=_rdesc)(_rfn)


_register_tools_and_resources()


def _install_signal_handlers() -> None:
    """注册 SIGTERM/SIGINT 处理器：第一信号 drain 10s 后 exit 0，第二信号强制 exit 1。

    使用 ``_shutdown_event``（threading.Event）协调 in-flight 任务，
    替代 pipeflow 的 ``_db._shutting_down``（SenseFrame 无 DB）。
    """
    def _on_signal(signum: int, _frame: Any) -> None:
        if _shutdown_event.is_set():
            log.warning("second signal %s received; forcing exit", signum)
            sys.exit(1)
        log.warning("signal %s received; starting graceful drain (10s)", signum)
        _shutdown_event.set()
        # 等待 in-flight 任务完成（最多 10s）
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            time.sleep(0.25)
        log.info("graceful drain complete")
        sys.exit(0)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def main() -> None:
    """服务器入口：校验配置 + 初始化日志 + 注册信号处理器 + 启动 stdio 传输。"""
    validate_config()
    configure_logging()
    _install_signal_handlers()
    log.info("senseframe-mcp starting with %d tools declared", len(EXPECTED_TOOLS))
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
