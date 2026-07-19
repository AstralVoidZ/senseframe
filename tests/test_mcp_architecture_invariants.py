"""MCP 架构不变量测试：钉死 senseframe/mcp/ 5 层单向依赖。

分层不变量（设计文档 0.2 节）：
- tools/ 不 import storage.dao / storage.db
- views/ 不 import orchestration
- views/ 不 import tools / storage / spec
- models/ 不 import 任何上层（仅 stdlib + errors）
- resources/ 不 import tools
- core/ 不 import orchestration / modules / storage / tools / spec / views
  （当前 mcp/ 结构无 core/ 子目录，测试在 core/ 不存在时 trivially 通过）

此外还覆盖：
- EXPECTED_TOOLS 与 _TOOL_REGISTRY 名称集合一致
- ToolErrorResponse category 严格 7 类
- FrozenModel extra='forbid' + frozen=True
- ToolErrorResponse.envelope_from 异常路由
- to_tool_error 返回 ToolError（payload 为信封 JSON）
"""

from __future__ import annotations

import ast
import json
import pathlib
from typing import get_args

import pytest

SENSEFRAME_ROOT = pathlib.Path(__file__).resolve().parent.parent
MCP_ROOT = SENSEFRAME_ROOT / "senseframe" / "mcp"


def _iter_py_files(dir_name: str):
    """遍历 mcp/<dir_name>/ 下所有 .py 文件，yield (path, ast_tree)。"""
    dir_path = MCP_ROOT / dir_name
    if not dir_path.exists():
        return
    for py in sorted(dir_path.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        yield py, tree


def _find_forbidden_import(tree, forbidden_prefixes):
    """检查 AST 是否含任何匹配 forbidden_prefixes 的 import。

    同时检查 ast.ImportFrom（from X import Y）和 ast.Import（import X）。
    匹配规则：mod == prefix 或 mod.startswith(prefix + ".")，精确前缀匹配。

    Returns:
        第一个匹配的 module 名，无匹配返回 None。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for prefix in forbidden_prefixes:
                if mod == prefix or mod.startswith(prefix + "."):
                    return mod
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for prefix in forbidden_prefixes:
                    if mod == prefix or mod.startswith(prefix + "."):
                        return mod
    return None


# ============================================================
# 分层不变量（AST 守卫）
# ============================================================


def test_tools_have_no_storage_imports() -> None:
    """Invariant: tools/ MUST NOT import storage.dao / storage.db。

    tools 层通过 orchestration 层访问数据，直接 DAO import 绕过业务逻辑，
    违反 5 层单向依赖。
    """
    forbidden = ["senseframe.mcp.storage.dao", "senseframe.mcp.storage.db"]
    for py, tree in _iter_py_files("tools"):
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


def test_views_have_no_orchestration_imports() -> None:
    """Invariant: views/ MUST NOT import orchestration。

    views 层在 orchestration 之下，反向 import 创建循环依赖风险。
    """
    forbidden = ["senseframe.mcp.orchestration"]
    for py, tree in _iter_py_files("views"):
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


def test_views_have_no_tools_or_storage_imports() -> None:
    """Invariant: views/ MUST NOT import tools / storage。

    views 是公共 JSON 契约层，不得依赖 tool 实现或持久化细节。
    """
    forbidden = ["senseframe.mcp.tools", "senseframe.mcp.storage"]
    for py, tree in _iter_py_files("views"):
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


def test_models_have_no_upward_imports() -> None:
    """Invariant: models/ MUST NOT import 上层（orchestration/modules/storage/tools/views/spec）。

    models 是域对象层（dataclass only），仅依赖 stdlib + errors。
    """
    forbidden = [
        "senseframe.mcp.orchestration",
        "senseframe.mcp.modules",
        "senseframe.mcp.storage",
        "senseframe.mcp.tools",
        "senseframe.mcp.views",
        "senseframe.mcp.spec",
    ]
    for py, tree in _iter_py_files("models"):
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


def test_resources_have_no_tools_imports() -> None:
    """Invariant: resources/ MUST NOT import tools。

    resources（L4 ISP 自省）在 tools（L5 OPP 操作）之上提供只读视图，
    不得依赖 tool 实现。
    """
    forbidden = ["senseframe.mcp.tools"]
    for py, tree in _iter_py_files("resources"):
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


def test_core_has_no_heavy_imports() -> None:
    """Invariant: core/ MUST NOT import orchestration / modules / storage / tools / spec / views。

    注意：senseframe/mcp/ 当前结构无 core/ 子目录（区别于 pipeflow）。
    core/ 不存在时此测试 trivially 通过；未来引入 core/ 时自动启用守卫。
    """
    core_path = MCP_ROOT / "core"
    if not core_path.exists():
        return  # 当前 mcp/ 结构无 core/ 子目录
    forbidden = [
        "senseframe.mcp.orchestration",
        "senseframe.mcp.modules",
        "senseframe.mcp.storage",
        "senseframe.mcp.tools",
        "senseframe.mcp.spec",
        "senseframe.mcp.views",
    ]
    for py in sorted(core_path.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        bad = _find_forbidden_import(tree, forbidden)
        assert not bad, f"{py}: imports {bad}"


# ============================================================
# Tool 注册契约
# ============================================================


def test_expected_tools_matches_registry() -> None:
    """EXPECTED_TOOLS 必须与 _TOOL_REGISTRY 的工具名集合完全一致。"""
    from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS, _TOOL_REGISTRY

    expected = set(EXPECTED_TOOLS)
    registered = {name for name, _, _ in _TOOL_REGISTRY}
    assert expected == registered, (
        f"EXPECTED_TOOLS != _TOOL_REGISTRY names: "
        f"missing={sorted(expected - registered)} "
        f"extra={sorted(registered - expected)}"
    )


def test_expected_tools_has_twenty_five_entries() -> None:
    """EXPECTED_TOOLS 必须含 29 个 tool（设计文档 0.4 节 ToolAnnotations 矩阵）。

    阶段 3 扩展：从 14 个 tool 增至 25 个 tool：
    - 原 14 个（pipeline 7 + study_create/ask/tell 3 + artifact/skill stub 3 + config_parse 1）
    - 新增 11 个：study_get/list/compare/stop（4）+ hpo_setup（1）+
      exploration_recommend（1）+ automl_create/advance/get/list（4）+
      apply_params_extended（1）
    阶段 4.2 扩展：从 25 个 tool 增至 27 个 tool：
    - 新增 2 个：senseframe_artifact_list + senseframe_artifact_export
    - senseframe_artifact_verify stub 升级为真实 handler（数量不变）
    阶段 4.3 扩展：从 27 个 tool 增至 29 个 tool：
    - 新增 2 个：senseframe_skill_get + senseframe_skill_search
    - senseframe_skill_save / senseframe_skill_remove stub 升级为真实 handler
    """
    from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS

    assert len(EXPECTED_TOOLS) == 29, (
        f"EXPECTED_TOOLS 应有 29 项，实际 {len(EXPECTED_TOOLS)}: {EXPECTED_TOOLS}"
    )


def test_error_categories_are_exactly_seven() -> None:
    """ToolErrorResponse 的 category 必须严格是 7 类。

    设计文档 0.5 节定义：pipeline/study/scene/artifact/config/search/internal。
    """
    from senseframe.mcp.views.tool_error import CategoryT

    categories = set(get_args(CategoryT))
    expected = {
        "pipeline",
        "study",
        "scene",
        "artifact",
        "config",
        "search",
        "internal",
    }
    assert categories == expected, (
        f"category 集合不匹配: expected={sorted(expected)} got={sorted(categories)}"
    )
    assert len(categories) == 7


# ============================================================
# FrozenModel 契约
# ============================================================


def test_frozen_model_rejects_extra_fields() -> None:
    """FrozenModel 必须拒绝额外字段（extra='forbid'）。"""
    from pydantic import ValidationError

    from senseframe.mcp.views._base import FrozenModel

    class TestModel(FrozenModel):
        x: int

    with pytest.raises(ValidationError):
        TestModel(x=1, unknown_field=2)


def test_frozen_model_is_immutable() -> None:
    """FrozenModel 必须不可变（frozen=True）。"""
    from pydantic import ValidationError

    from senseframe.mcp.views._base import FrozenModel

    class TestModel(FrozenModel):
        x: int

    m = TestModel(x=1)
    with pytest.raises(ValidationError):
        m.x = 2


# ============================================================
# 错误信封路由
# ============================================================


def test_tool_error_response_envelope_from_routes_categories() -> None:
    """ToolErrorResponse.envelope_from 正确路由异常到 category。

    本阶段（1.2-1.6）_CATEGORY_BY_EXC 覆盖 4 类 category：
    - pipeline: 协议层 PipelineRun 状态机错误
    - scene: ML 业务场景错误
    - config: 协议层 + ML 业务配置校验错误
    - internal: 限流 + ML 业务内部错误（OOM/Checkpoint/Training 等）

    study / artifact / search 类别将在后续阶段（study/artifact tool 实现）补充映射。
    """
    from senseframe.mcp.errors import (
        IllegalTransition,
        InvalidPathError,
        OOMError,
        PipelineNotFound,
        RateLimitExceeded,
        SceneNotRegisteredError,
        SchemaValidationError,
    )
    from senseframe.mcp.views.tool_error import ToolErrorResponse

    # --- pipeline 类别 ---
    env = ToolErrorResponse.envelope_from(PipelineNotFound("run-123 not found"))
    assert env.category == "pipeline"
    assert env.code == "PipelineNotFound"
    assert "run-123" in env.message

    env = ToolErrorResponse.envelope_from(IllegalTransition("bad transition"))
    assert env.category == "pipeline"
    assert env.code == "IllegalTransition"

    # --- scene 类别 ---
    env = ToolErrorResponse.envelope_from(SceneNotRegisteredError("wifi_csi missing"))
    assert env.category == "scene"
    assert env.code == "SceneNotRegisteredError"

    # --- config 类别 ---
    env = ToolErrorResponse.envelope_from(SchemaValidationError("schema fail"))
    assert env.category == "config"
    assert env.code == "SchemaValidationError"

    env = ToolErrorResponse.envelope_from(InvalidPathError("bad path"))
    assert env.category == "config"
    assert env.code == "InvalidPathError"

    # --- internal 类别（限流 + OOM）---
    env = ToolErrorResponse.envelope_from(RateLimitExceeded("60/min exceeded"))
    assert env.category == "internal"
    assert env.code == "RateLimitExceeded"

    env = ToolErrorResponse.envelope_from(OOMError("cuda OOM"))
    assert env.category == "internal"
    assert env.code == "OOMError"

    # --- 未知异常 → internal 兜底 ---
    env = ToolErrorResponse.envelope_from(RuntimeError("unknown failure"))
    assert env.category == "internal"
    assert env.code == "RuntimeError"


def test_to_tool_error_returns_tool_error_with_envelope_json() -> None:
    """to_tool_error 返回 ToolError，payload 是 ToolErrorResponse 的 JSON。"""
    from mcp.server.fastmcp.exceptions import ToolError

    from senseframe.mcp.errors import PipelineNotFound
    from senseframe.mcp.tools._errors import to_tool_error

    exc = PipelineNotFound("run-xyz not found")
    tool_err = to_tool_error(exc)

    assert isinstance(tool_err, ToolError)
    # ToolError 的 message 是 ToolErrorResponse 的 JSON 字符串
    payload = json.loads(str(tool_err))
    assert payload["code"] == "PipelineNotFound"
    assert payload["category"] == "pipeline"
    assert "run-xyz" in payload["message"]


# ============================================================
# 阶段 2.1-2.5 新增 AST 守卫
# ============================================================


def _has_call_pattern(tree, func_name: str, method_name: str) -> bool:
    """检查 AST 是否含 ``<func_name>.<method_name>(...)`` 调用。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == method_name:
                value = node.value
                if isinstance(value, ast.Name) and value.id == func_name:
                    return True
    return False


def _has_import_alias(tree, module: str, alias: str) -> bool:
    """检查 AST 是否 ``from <module> import <alias>`` 或 ``import <module> as <alias>``。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == module:
                for alias_node in node.names:
                    name = alias_node.asname or alias_node.name
                    if name == alias:
                        return True
        elif isinstance(node, ast.Import):
            for alias_node in node.names:
                if alias_node.name == module:
                    name = alias_node.asname or alias_node.name
                    if name == alias:
                        return True
    return False


def test_pipeline_tools_use_middleware_stack() -> None:
    """Invariant: tools/pipeline.py 必须通过 MiddlewareStack.instrument 调用。

    设计文档 0.4 节：每个 tool 必须通过 ``async with _stack.instrument(...)``
    调用核心逻辑，确保 RequestId + RateLimit 中间件生效。
    """
    py = MCP_ROOT / "tools" / "pipeline.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    # 检查 _stack.instrument 调用（async with _stack.instrument(...)）
    assert _has_call_pattern(tree, "_stack", "instrument"), (
        f"{py}: must call _stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_pipeline_views_inherit_frozen_model() -> None:
    """Invariant: views/pipeline.py 所有顶层 class 必须继承 FrozenModel。

    设计文档 0.5 节：所有 view 类必须继承 FrozenModel（extra='forbid' + frozen=True）。
    """
    py = MCP_ROOT / "views" / "pipeline.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))

    # 收集所有顶层 ClassDef
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"

    # FrozenModel 可以通过直接继承或子类继承
    # 检查每个 class 至少有一个 base 是 FrozenModel（或其已知子类）
    # 简化策略：第一个 base 名为 FrozenModel，或者继承 PipelineRunView/TransitionView 等
    # 已知 view 类的子类
    KNOWN_VIEW_BASES = {"FrozenModel", "PipelineRunView", "_PaginatedResponse"}
    for cls in classes:
        # 跳过内部辅助类（以 _ 开头）
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                # 处理 module.FrozenModel 形式
                base_names.append(base.attr)
        # 至少一个 base 是 FrozenModel 或已知 view base
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )


def test_pipeline_run_store_uses_threading_lock() -> None:
    """Invariant: orchestration/pipeline_run.py 必须用 threading.RLock 保护内存存储。

    设计文档 0.6 节：PipelineRunStore 用 ``dict[str, PipelineRun] + threading.RLock``，
    不引入 SQLite。
    """
    py = MCP_ROOT / "orchestration" / "pipeline_run.py"
    assert py.exists(), f"{py} must exist"
    src = py.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 检查 import threading
    has_threading_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "threading":
                    has_threading_import = True
                    break
        elif isinstance(node, ast.ImportFrom):
            if node.module == "threading":
                has_threading_import = True
    assert has_threading_import, (
        f"{py}: must import threading for thread-safety (RLock)"
    )
    # 检查 threading.RLock() 调用
    assert "RLock" in src, (
        f"{py}: must use threading.RLock() to protect in-memory dict"
    )


# ============================================================
# 阶段 3.1-3.5 新增 AST 守卫
# ============================================================


def test_study_tools_use_middleware_stack() -> None:
    """Invariant: tools/study.py 必须通过 MiddlewareStack.instrument 调用。

    设计文档 0.4 节：每个 tool 必须通过 ``async with _stack.instrument(...)``
    调用核心逻辑，确保 RequestId + RateLimit 中间件生效。
    """
    py = MCP_ROOT / "tools" / "study.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    # 检查 _study_stack.instrument 调用
    assert _has_call_pattern(tree, "_study_stack", "instrument"), (
        f"{py}: must call _study_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_hpo_tools_use_middleware_stack() -> None:
    """Invariant: tools/hpo.py 必须通过 MiddlewareStack.instrument 调用。"""
    py = MCP_ROOT / "tools" / "hpo.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    assert _has_call_pattern(tree, "_hpo_stack", "instrument"), (
        f"{py}: must call _hpo_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_exploration_tools_use_middleware_stack() -> None:
    """Invariant: tools/exploration.py 必须通过 MiddlewareStack.instrument 调用。"""
    py = MCP_ROOT / "tools" / "exploration.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    assert _has_call_pattern(tree, "_exploration_stack", "instrument"), (
        f"{py}: must call _exploration_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_automl_tools_use_middleware_stack() -> None:
    """Invariant: tools/automl.py 必须通过 MiddlewareStack.instrument 调用。"""
    py = MCP_ROOT / "tools" / "automl.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    assert _has_call_pattern(tree, "_automl_stack", "instrument"), (
        f"{py}: must call _automl_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_param_bridge_tools_use_middleware_stack() -> None:
    """Invariant: tools/param_bridge.py 必须通过 MiddlewareStack.instrument 调用。"""
    py = MCP_ROOT / "tools" / "param_bridge.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    assert _has_call_pattern(tree, "_param_bridge_stack", "instrument"), (
        f"{py}: must call _param_bridge_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_study_views_inherit_frozen_model() -> None:
    """Invariant: views/study.py 所有顶层 class 必须继承 FrozenModel。"""
    py = MCP_ROOT / "views" / "study.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"
    KNOWN_VIEW_BASES = {"FrozenModel", "StudyView", "TrialView"}
    for cls in classes:
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )


def test_automl_views_inherit_frozen_model() -> None:
    """Invariant: views/automl.py 所有顶层 class 必须继承 FrozenModel。"""
    py = MCP_ROOT / "views" / "automl.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"
    KNOWN_VIEW_BASES = {"FrozenModel", "AutoMLPipelineView"}
    for cls in classes:
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )


def test_exploration_views_inherit_frozen_model() -> None:
    """Invariant: views/exploration.py 所有顶层 class 必须继承 FrozenModel。"""
    py = MCP_ROOT / "views" / "exploration.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"
    KNOWN_VIEW_BASES = {"FrozenModel"}
    for cls in classes:
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )


def test_automl_orchestrator_uses_threading_lock() -> None:
    """Invariant: orchestration/automl_orchestrator.py 必须用 threading.RLock。

    设计文档 0.7.3 节：AutoMLOrchestrator 用 dict + threading.RLock，
    不引入 SQLite。
    """
    py = MCP_ROOT / "orchestration" / "automl_orchestrator.py"
    assert py.exists(), f"{py} must exist"
    src = py.read_text(encoding="utf-8")
    assert "threading" in src, f"{py}: must import threading"
    assert "RLock" in src, f"{py}: must use threading.RLock()"


def test_study_manager_uses_threading_lock() -> None:
    """Invariant: orchestration/study_manager.py 必须用 threading.RLock。"""
    py = MCP_ROOT / "orchestration" / "study_manager.py"
    assert py.exists(), f"{py} must exist"
    src = py.read_text(encoding="utf-8")
    assert "threading" in src, f"{py}: must import threading"
    assert "RLock" in src, f"{py}: must use threading.RLock()"


def test_param_bridge_does_not_modify_engine_hpo() -> None:
    """Invariant: orchestration/param_bridge.py 不得修改 engine/hpo.py 的 apply_params。

    通过检查 param_bridge.py 是 import apply_params（复用），而非重新定义。
    """
    py = MCP_ROOT / "orchestration" / "param_bridge.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    # 检查 from senseframe.engine.hpo import apply_params
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "senseframe.engine.hpo":
                for alias in node.names:
                    if alias.name == "apply_params":
                        found = True
                        break
    assert found, (
        f"{py}: must import apply_params from senseframe.engine.hpo "
        f"(reuse existing impl, not redefine)"
    )


def test_tool_dispatch_has_twenty_five_tools() -> None:
    """Invariant: tool_dispatch.py 的 _TOOL_REGISTRY 必须含 29 个 tool。

    设计文档 0.4 节 ToolAnnotations 矩阵扩展到 29 个 tool：
    - 8 个 pipeline/config（含 1 个 stub）
    - 7 个 study_*（real）
    - 1 个 hpo_setup（real）
    - 1 个 exploration_recommend（real）
    - 4 个 automl_*（real）
    - 1 个 apply_params_extended（real）
    - 3 个 artifact_*（real，阶段 4.2 升级 + 新增）
    - 4 个 skill_*（real，阶段 4.3 升级 + 新增）
    """
    from senseframe.mcp.tool_dispatch import _TOOL_REGISTRY

    assert len(_TOOL_REGISTRY) == 29, (
        f"_TOOL_REGISTRY 应有 29 项，实际 {len(_TOOL_REGISTRY)}"
    )


def test_server_annotations_matrix_has_twenty_five_entries() -> None:
    """Invariant: server.py 的 _ANNOTATIONS 必须含 29 个 entry。"""
    from senseframe.mcp.server import _ANNOTATIONS

    assert len(_ANNOTATIONS) == 29, (
        f"_ANNOTATIONS 应有 29 项，实际 {len(_ANNOTATIONS)}"
    )


def test_annotations_keys_match_expected_tools() -> None:
    """Invariant: _ANNOTATIONS 的 key 集合必须与 EXPECTED_TOOLS 完全一致。"""
    from senseframe.mcp.server import _ANNOTATIONS
    from senseframe.mcp.tool_dispatch import EXPECTED_TOOLS

    annotations_keys = set(_ANNOTATIONS.keys())
    expected = set(EXPECTED_TOOLS)
    assert annotations_keys == expected, (
        f"_ANNOTATIONS keys != EXPECTED_TOOLS: "
        f"missing={sorted(expected - annotations_keys)} "
        f"extra={sorted(annotations_keys - expected)}"
    )


# ============================================================
# 阶段 4.2 新增 AST 守卫
# ============================================================


def test_artifact_tools_use_middleware_stack() -> None:
    """Invariant: tools/artifact.py 必须通过 MiddlewareStack.instrument 调用。

    设计文档 0.4 节：每个 tool 必须通过 ``async with _artifact_stack.instrument(...)``
    调用核心逻辑，确保 RequestId + RateLimit 中间件生效。
    """
    py = MCP_ROOT / "tools" / "artifact.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    # 检查 _artifact_stack.instrument 调用
    assert _has_call_pattern(tree, "_artifact_stack", "instrument"), (
        f"{py}: must call _artifact_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_artifact_views_inherit_frozen_model() -> None:
    """Invariant: views/artifact.py 所有顶层 class 必须继承 FrozenModel。"""
    py = MCP_ROOT / "views" / "artifact.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"
    KNOWN_VIEW_BASES = {"FrozenModel", "ArtifactDescriptorView"}
    for cls in classes:
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )


# ============================================================
# 阶段 4.3 新增 AST 守卫
# ============================================================


def test_skill_tools_use_middleware_stack() -> None:
    """Invariant: tools/skill.py 必须通过 MiddlewareStack.instrument 调用。

    设计文档 0.4 节：每个 tool 必须通过 ``async with _skill_stack.instrument(...)``
    调用核心逻辑，确保 RequestId + RateLimit 中间件生效。
    """
    py = MCP_ROOT / "tools" / "skill.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    # 检查 _skill_stack.instrument 调用
    assert _has_call_pattern(tree, "_skill_stack", "instrument"), (
        f"{py}: must call _skill_stack.instrument(...) to wrap tool logic "
        f"with MiddlewareStack"
    )


def test_skill_views_inherit_frozen_model() -> None:
    """Invariant: views/skill.py 所有顶层 class 必须继承 FrozenModel。"""
    py = MCP_ROOT / "views" / "skill.py"
    assert py.exists(), f"{py} must exist"
    tree = ast.parse(py.read_text(encoding="utf-8"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert classes, f"{py}: must contain at least one class definition"
    KNOWN_VIEW_BASES = {"FrozenModel", "SkillView"}
    for cls in classes:
        if cls.name.startswith("_"):
            continue
        base_names = []
        for base in cls.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert any(b in KNOWN_VIEW_BASES for b in base_names), (
            f"{py}: class {cls.name} must inherit FrozenModel "
            f"(or a known view base); got bases={base_names}"
        )
