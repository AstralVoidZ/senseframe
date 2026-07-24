"""L1 Protocol：_TOOL_REGISTRY + EXPECTED_TOOLS + dispatch 骨架。

集中式 tool 注册（非装饰器），便于 AST 守卫测试钉死 EXPECTED_TOOLS 与
_TOOL_REGISTRY 名称集合一致。

阶段 2.1-2.5：实现 7 个 pipeline tool 的真实 handler。
阶段 3.1-3.5：实现 11 个新 tool 的真实 handler（study_* 工具组 4 个新增 +
hpo_setup + exploration_recommend + automl_* 4 个 + apply_params_extended），
并将 3 个 study_create/ask/tell stub 升级为真实 handler。
阶段 4.2：senseframe_artifact_verify 升级为真实 handler，新增
senseframe_artifact_list / senseframe_artifact_export。
阶段 4.3：senseframe_skill_save / senseframe_skill_remove 升级为真实 handler，
新增 senseframe_skill_get / senseframe_skill_search。
阶段 6（v2 次要差距修复）：senseframe_config_parse stub 升级为真实 handler，
全部 29 个 tool 已实装。

设计文档 0.3 节定义 29 个 tool（按 ToolAnnotations 矩阵 0.4 节）：
- 声明类（2）：senseframe_config_parse, senseframe_pipeline_create
- 状态转移类（1）：senseframe_pipeline_advance
- 执行类（1）：senseframe_pipeline_run
- 查询类（4）：senseframe_pipeline_get, senseframe_pipeline_list,
              senseframe_pipeline_pause, senseframe_pipeline_resume
- Study 类（7）：senseframe_study_create/ask/tell/get/list/compare/stop
- HPO 类（1）：senseframe_hpo_setup
- Exploration 类（1）：senseframe_exploration_recommend
- AutoML 类（4）：senseframe_automl_create/advance/get/list
- Param Bridge 类（1）：senseframe_apply_params_extended
- Artifact 类（3）：senseframe_artifact_verify/list/export（阶段 4.2）
- 技能类（4）：senseframe_skill_save/get/search/remove（阶段 4.3）

ToolAnnotations 矩阵覆盖 29 个 tool，见 server.py 的 _ANNOTATIONS。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from senseframe.mcp.tools.artifact import (
    senseframe_artifact_export,
    senseframe_artifact_list,
    senseframe_artifact_verify,
)
from senseframe.mcp.tools.automl import (
    senseframe_automl_advance,
    senseframe_automl_create,
    senseframe_automl_get,
    senseframe_automl_list,
)
from senseframe.mcp.tools.config import senseframe_config_parse
from senseframe.mcp.tools.exploration import senseframe_exploration_recommend
from senseframe.mcp.tools.hpo import senseframe_hpo_setup
from senseframe.mcp.tools.param_bridge import senseframe_apply_params_extended
from senseframe.mcp.tools.pipeline import (
    senseframe_pipeline_advance,
    senseframe_pipeline_create,
    senseframe_pipeline_get,
    senseframe_pipeline_list,
    senseframe_pipeline_pause,
    senseframe_pipeline_resume,
    senseframe_pipeline_run,
)
from senseframe.mcp.tools.skill import (
    senseframe_skill_get,
    senseframe_skill_remove,
    senseframe_skill_save,
    senseframe_skill_search,
)
from senseframe.mcp.tools.study import (
    senseframe_study_ask,
    senseframe_study_compare,
    senseframe_study_create,
    senseframe_study_get,
    senseframe_study_list,
    senseframe_study_stop,
    senseframe_study_tell,
)

__all__ = ["EXPECTED_TOOLS", "_TOOL_REGISTRY"]


# 29 个 tool 的规范名称列表（设计文档 0.4 节 ToolAnnotations 矩阵）。
EXPECTED_TOOLS: tuple[str, ...] = (
    # 声明类
    "senseframe_config_parse",
    "senseframe_pipeline_create",
    # 状态转移类
    "senseframe_pipeline_advance",
    # 执行类
    "senseframe_pipeline_run",
    # 查询类
    "senseframe_pipeline_get",
    "senseframe_pipeline_list",
    "senseframe_pipeline_pause",
    "senseframe_pipeline_resume",
    # Study 类（L4 SP，阶段 3.1）
    "senseframe_study_create",
    "senseframe_study_ask",
    "senseframe_study_tell",
    "senseframe_study_get",
    "senseframe_study_list",
    "senseframe_study_compare",
    "senseframe_study_stop",
    # HPO 类（阶段 3.2）
    "senseframe_hpo_setup",
    # Exploration 类（阶段 3.3）
    "senseframe_exploration_recommend",
    # AutoML 类（阶段 3.4）
    "senseframe_automl_create",
    "senseframe_automl_advance",
    "senseframe_automl_get",
    "senseframe_automl_list",
    # Param Bridge 类（阶段 3.5）
    "senseframe_apply_params_extended",
    # Artifact 类（阶段 4.2，verify 升级 + list/export 新增）
    "senseframe_artifact_verify",
    "senseframe_artifact_list",
    "senseframe_artifact_export",
    # 技能类（阶段 4.3，save/remove 升级 + get/search 新增）
    "senseframe_skill_save",
    "senseframe_skill_get",
    "senseframe_skill_search",
    "senseframe_skill_remove",
)


# Tool 注册表：(name, description, handler_function)。
# 全部 29 个 tool 已实装（v2 次要差距修复：config_parse stub 升级为真实 handler）。
_TOOL_REGISTRY: list[tuple[str, str, Callable[..., Any]]] = [
    (
        "senseframe_config_parse",
        "Parse YAML config string into ExperimentConfig (with extra='forbid' validation).",
        senseframe_config_parse,  # v2 次要差距修复：实装，替换 _not_implemented
    ),
    (
        "senseframe_pipeline_create",
        "Create a new PipelineRun (declarative: accepts config + stages).",
        senseframe_pipeline_create,
    ),
    (
        "senseframe_pipeline_advance",
        "Advance a PipelineRun state machine (action=start/complete/fail/retry/skip/pause/resume). Idempotent.",
        senseframe_pipeline_advance,
    ),
    (
        "senseframe_pipeline_run",
        "Execute a complete pipeline (black-box, blocking). Advisory execution tool.",
        senseframe_pipeline_run,
    ),
    (
        "senseframe_pipeline_get",
        "Query PipelineRun state (includes _transitions HATEOAS hints).",
        senseframe_pipeline_get,
    ),
    (
        "senseframe_pipeline_list",
        "List all PipelineRuns with cursor pagination.",
        senseframe_pipeline_list,
    ),
    (
        "senseframe_pipeline_pause",
        "Pause a running PipelineRun (idempotent).",
        senseframe_pipeline_pause,
    ),
    (
        "senseframe_pipeline_resume",
        "Resume a paused PipelineRun (idempotent).",
        senseframe_pipeline_resume,
    ),
    (
        "senseframe_study_create",
        "Create a search study (name + direction + search_space + sampler).",
        senseframe_study_create,
    ),
    (
        "senseframe_study_ask",
        "Sample the next trial from a study (returns TrialSpec).",
        senseframe_study_ask,
    ),
    (
        "senseframe_study_tell",
        "Report a trial result (value + intermediate_values + state + feedback).",
        senseframe_study_tell,
    ),
    (
        "senseframe_study_get",
        "Query study state and best trial (includes _transitions HATEOAS).",
        senseframe_study_get,
    ),
    (
        "senseframe_study_list",
        "List all studies with cursor pagination.",
        senseframe_study_list,
    ),
    (
        "senseframe_study_compare",
        "Compare multiple studies (structured comparison table).",
        senseframe_study_compare,
    ),
    (
        "senseframe_study_stop",
        "Stop a study (terminal state, idempotent).",
        senseframe_study_stop,
    ),
    (
        "senseframe_hpo_setup",
        "Convert ExperimentConfig HPOConfig to a Study search space (Ask-Tell 3-step interface).",
        senseframe_hpo_setup,
    ),
    (
        "senseframe_exploration_recommend",
        "Recommend next exploration strategy based on study feedback (ExplorationTracker.recommend_next).",
        senseframe_exploration_recommend,
    ),
    (
        "senseframe_automl_create",
        "Create an AutoML pipeline (config + stages like [nas, hpo, autoaugment]).",
        senseframe_automl_create,
    ),
    (
        "senseframe_automl_advance",
        "Advance an AutoML pipeline state machine (action=start/complete/fail/pause/resume/retry). Idempotent.",
        senseframe_automl_advance,
    ),
    (
        "senseframe_automl_get",
        "Query AutoML pipeline state (includes _transitions HATEOAS).",
        senseframe_automl_get,
    ),
    (
        "senseframe_automl_list",
        "List all AutoML pipelines with cursor pagination.",
        senseframe_automl_list,
    ),
    (
        "senseframe_apply_params_extended",
        "Apply sampled params + inject module_factory/datamodule_factory to ExperimentConfig (joint search).",
        senseframe_apply_params_extended,
    ),
    (
        "senseframe_artifact_verify",
        "Verify artifact integrity (hash + manifest schema + required-artifact triple check).",
        senseframe_artifact_verify,
    ),
    (
        "senseframe_artifact_list",
        "List artifacts in a manifest with cursor pagination and filter support.",
        senseframe_artifact_list,
    ),
    (
        "senseframe_artifact_export",
        "Export artifacts to zip / tar / manifest format (with SHA256 hash of the export file).",
        senseframe_artifact_export,
    ),
    (
        "senseframe_skill_save",
        "Save a skill (with validation + version management).",
        senseframe_skill_save,
    ),
    (
        "senseframe_skill_get",
        "Get a skill by name (with optional version).",
        senseframe_skill_get,
    ),
    (
        "senseframe_skill_search",
        "Semantic search over skills (returns scored results).",
        senseframe_skill_search,
    ),
    (
        "senseframe_skill_remove",
        "Remove a skill (with dependency check).",
        senseframe_skill_remove,
    ),
]
