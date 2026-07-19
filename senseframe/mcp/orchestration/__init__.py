"""L3 Orchestration：业务逻辑层。

公开 API：
- pipeline_run.PipelineRunStore + trigger + VALID_TRANSITIONS + IDEMPOTENT_ACTIONS
- transitions.get_transitions（HATEOAS 转换建议，advisory 模式）

后续阶段将实现：
- study_manager.py：Study 管理（ask/tell/best_trial）
- artifact_verify.py：产物校验
- automl_orchestrator.py：AutoMLOrchestrator（HPO → NAS → AutoAugment 串联）

分层不变量：orchestration 可 import views / models / storage / errors，
但 views 不得反向 import orchestration（AST 守卫测试钉死）。
"""

from senseframe.mcp.orchestration import pipeline_run, transitions

__all__ = ["pipeline_run", "transitions"]
