"""View 层包导出。

公开 API（本阶段已实现）：
- FrozenModel / ViewError：所有 view 模型的基类
- ToolErrorResponse：错误信封
- pipeline.TransitionView / StageView / PipelineRunView / PipelineRunListView
  / PipelineCreateResponse / PipelineAdvanceResponse：PipelineRun 工具组视图
- study.StudyView / TrialView / StudyCreateResponse / StudyAskResponse /
  StudyTellResponse / StudyListView / StudyCompareView：Study 工具组视图
- exploration.ExplorationRecommendationItem / ExplorationRecommendationView：
  Exploration 推荐视图
- automl.AutoMLRunView / AutoMLCreateResponse / AutoMLAdvanceResponse /
  AutoMLListView：AutoMLOrchestrator 工具组视图
- artifact.ArtifactDescriptorView / ArtifactVerifyResponse / ArtifactListView /
  ArtifactExportResponse：Artifact 工具组视图（阶段 4.2）
- skill.SkillView / SkillSaveResponse / SkillRemoveResponse /
  SkillSearchResultView / SkillSearchResponse：Skill 工具组视图（阶段 4.3）
"""

from senseframe.mcp.views._base import FrozenModel, ViewError
from senseframe.mcp.views.artifact import (
    ArtifactDescriptorView,
    ArtifactExportResponse,
    ArtifactListView,
    ArtifactVerifyResponse,
)
from senseframe.mcp.views.automl import (
    AutoMLAdvanceResponse,
    AutoMLCreateResponse,
    AutoMLPipelineListView,
    AutoMLPipelineView,
    AutoMLStageView,
)
from senseframe.mcp.views.exploration import (
    ExplorationRecommendationItem,
    ExplorationRecommendationView,
)
from senseframe.mcp.views.pipeline import (
    PipelineAdvanceResponse,
    PipelineCreateResponse,
    PipelineRunListView,
    PipelineRunView,
    StageView,
    TransitionView,
)
from senseframe.mcp.views.skill import (
    SkillRemoveResponse,
    SkillSaveResponse,
    SkillSearchResponse,
    SkillSearchResultView,
    SkillView,
)
from senseframe.mcp.views.study import (
    StudyAskResponse,
    StudyCompareView,
    StudyCreateResponse,
    StudyListView,
    StudyTellResponse,
    StudyView,
    TrialView,
)
from senseframe.mcp.views.tool_error import ToolErrorResponse

__all__ = [
    "FrozenModel",
    "ViewError",
    "ToolErrorResponse",
    "TransitionView",
    "StageView",
    "PipelineRunView",
    "PipelineRunListView",
    "PipelineCreateResponse",
    "PipelineAdvanceResponse",
    "StudyView",
    "TrialView",
    "StudyCreateResponse",
    "StudyAskResponse",
    "StudyTellResponse",
    "StudyListView",
    "StudyCompareView",
    "ExplorationRecommendationItem",
    "ExplorationRecommendationView",
    "AutoMLStageView",
    "AutoMLPipelineView",
    "AutoMLPipelineListView",
    "AutoMLCreateResponse",
    "AutoMLAdvanceResponse",
    "ArtifactDescriptorView",
    "ArtifactVerifyResponse",
    "ArtifactListView",
    "ArtifactExportResponse",
    "SkillView",
    "SkillSaveResponse",
    "SkillRemoveResponse",
    "SkillSearchResultView",
    "SkillSearchResponse",
]
