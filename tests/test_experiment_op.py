"""ε6 OP 迁移测试（P2.11-P2.13）。

反假绿测试策略：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- 真实 Orchestrator 实例（不 mock，验证真实状态转换 + 事件发射）
- mock run_pipeline 避免训练开销（OP 测试聚焦编排逻辑，不测真实训练）
- 真实 PipelineRun 状态验证（phase 转换 + CloudEvent 发射）

覆盖：
- P2.11: Orchestrator 异步执行（start_and_execute + wait_for_completion + shutdown）
- P2.12: MethodRunner OP 路径（use_op=True 时走 _run_pipeline_via_op）
- P2.13: ExperimentRunner 事件驱动聚合（use_op=True 时订阅 OP 事件）
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from senseframe.orchestration import (
    EVENT_PIPELINE_FAILED,
    EVENT_PIPELINE_SUCCEEDED,
    PHASE_FAILED,
    PHASE_PENDING,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
    CloudEvent,
    Orchestrator,
    PipelineDef,
    PipelineRun,
    get_orchestrator,
)
from senseframe.search_protocol import (
    ParameterSpec,
    SearchSpace,
    StudyManager,
)
from senseframe.experiment.design import MethodConfig
from senseframe.experiment.method import MethodRunner
from senseframe.experiment.runner import ExperimentRunner
from senseframe.experiment.types import TrialGroup, TrialStatus
from senseframe.experiment.design import (
    BaselineConfig,
    ExperimentBudget,
    ExperimentDesign,
)
from senseframe.engine.config import (
    ExperimentConfig,
    InputFeature,
    OutputFeature,
    SceneConfig,
    TrainerConfig,
)


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


def _make_method_config() -> MethodConfig:
    """构造测试用 MethodConfig。"""
    base_config = ExperimentConfig(
        scene=SceneConfig(name="test", dataset="synthetic", model_id="MLP"),
        input_features=[InputFeature(name="features", type="tabular", shape=[10])],
        output_features=[OutputFeature(name="label", type="category", num_classes=3)],
        trainer=TrainerConfig(epochs=1, batch_size=4, enable_progress_bar=False, logger="csv"),
        output_dir="/tmp/test_op",
    )
    return MethodConfig(
        name="test_method_op",
        base_config=base_config,
        search_space=SearchSpace(parameters=[
            ParameterSpec(name="lr", type="float", low=0.001, high=0.1),
        ]),
        metric="val_accuracy",
        direction="maximize",
    )


def _make_mock_train_output(status: str = "success"):
    """构造 mock TrainOutput。"""
    mock = MagicMock()
    mock.status = status
    mock.model_path = "/tmp/model.pt"
    mock.output_dir = "/tmp/output"
    mock.error = None
    mock.error_code = None
    mock.final_eval = {"val_accuracy": 0.8, "val_loss": 0.5}
    mock.training = {
        "duration_s": 10.0,
        "epochs_trained": 2,
        "intermediate_values": {0: 0.5, 1: 0.8},
    }
    return mock


def _make_experiment_design(method_config: MethodConfig) -> ExperimentDesign:
    """构造测试用 ExperimentDesign。"""
    return ExperimentDesign(
        name="test_op_exp",
        datasets=["synthetic"],
        models=["MLP"],
        method=method_config,
        baselines=[],
        budget=ExperimentBudget(max_trials_per_group=1, n_repeats=1),
    )


# ============================================================
# P2.11: Orchestrator 异步执行
# ============================================================
class TestOrchestratorAsyncExecution:
    """Orchestrator 异步执行验证（P2.11）。"""

    def test_start_and_execute_returns_future(self):
        """start_and_execute 应返回 Future。"""
        from concurrent.futures import Future
        orch = Orchestrator()
        try:
            pdef = PipelineDef.default()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            # 绑定 mock ctx（reconcile 需要）
            ctx = MagicMock()
            ctx.completed_stages = []
            ctx.failed_stage = None
            ctx.output_dir = None
            orch.bind_context(run_id, ctx)
            future = orch.start_and_execute(run_id)
            assert isinstance(future, Future)
            # 等待完成
            run = orch.wait_for_completion(run_id, timeout=5.0)
            assert run.phase in (PHASE_SUCCEEDED, PHASE_FAILED)
        finally:
            orch.shutdown()

    def test_wait_for_completion_returns_terminal_run(self):
        """wait_for_completion 应返回终态 PipelineRun。"""
        orch = Orchestrator()
        try:
            pdef = PipelineDef.default()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            ctx = MagicMock()
            ctx.completed_stages = []
            ctx.failed_stage = None
            ctx.output_dir = None
            orch.bind_context(run_id, ctx)
            orch.start_and_execute(run_id)
            run = orch.wait_for_completion(run_id, timeout=5.0)
            assert run.phase in (PHASE_SUCCEEDED, PHASE_FAILED)
        finally:
            orch.shutdown()

    def test_start_and_execute_emits_events(self):
        """start_and_execute 应发射 pipeline.started + succeeded/failed 事件。"""
        orch = Orchestrator()
        try:
            events = []
            orch.subscribe("*", lambda e: events.append(e))
            pdef = PipelineDef.default()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            ctx = MagicMock()
            ctx.completed_stages = []
            ctx.failed_stage = None
            ctx.output_dir = None
            orch.bind_context(run_id, ctx)
            orch.start_and_execute(run_id)
            orch.wait_for_completion(run_id, timeout=5.0)
            # 应至少有 started 事件 + succeeded/failed 事件
            event_types = [e.type for e in events]
            assert "senseframe.pipeline.started" in event_types
            assert (
                "senseframe.pipeline.succeeded" in event_types
                or "senseframe.pipeline.failed" in event_types
            )
        finally:
            orch.shutdown()

    def test_duplicate_start_and_execute_raises(self):
        """重复 start_and_execute 同一 run_id 应抛 RuntimeError。"""
        orch = Orchestrator()
        try:
            pdef = PipelineDef.default()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            ctx = MagicMock()
            ctx.completed_stages = []
            ctx.failed_stage = None
            ctx.output_dir = None
            orch.bind_context(run_id, ctx)
            future = orch.start_and_execute(run_id)
            # 不等完成，立即尝试重复提交（future 可能已完成，需用新 run_id 验证）
            # 用新 run_id 测试重复提交保护
            run_id2 = orch.create_run(pdef.name)
            orch.bind_context(run_id2, ctx)
            orch.start_and_execute(run_id2)
            with pytest.raises(RuntimeError, match="already executing"):
                orch.start_and_execute(run_id2)
            orch.wait_for_completion(run_id, timeout=5.0)
            orch.wait_for_completion(run_id2, timeout=5.0)
        finally:
            orch.shutdown()

    def test_wait_for_completion_timeout(self):
        """wait_for_completion 超时应抛 TimeoutError。"""
        import time
        orch = Orchestrator()
        try:
            pdef = PipelineDef.default()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            # 不 start，phase 仍是 PENDING，wait 应超时
            with pytest.raises(TimeoutError):
                orch.wait_for_completion(run_id, timeout=0.2)
        finally:
            orch.shutdown()

    def test_shutdown_releases_executor(self):
        """shutdown 应清理线程池。"""
        orch = Orchestrator()
        pdef = PipelineDef.default()
        orch.create_pipeline(pdef)
        run_id = orch.create_run(pdef.name)
        ctx = MagicMock()
        ctx.completed_stages = []
        ctx.failed_stage = None
        ctx.output_dir = None
        orch.bind_context(run_id, ctx)
        orch.start_and_execute(run_id)
        orch.wait_for_completion(run_id, timeout=5.0)
        orch.shutdown()
        assert orch._executor is None
        assert orch._run_futures == {}


# ============================================================
# P2.12: MethodRunner OP 路径
# ============================================================
class TestMethodRunnerOpPath:
    """MethodRunner OP 迁移路径验证（P2.12）。"""

    def test_use_op_default_false(self):
        """use_op 默认应为 False（向后兼容）。"""
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        runner = MethodRunner(config=config, study_id=study_id, study_manager=sm)
        assert runner.use_op is False

    def test_use_op_true_uses_op_path(self):
        """use_op=True 时应走 _run_pipeline_via_op 路径。"""
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("success")
                result = runner.run("synthetic", "MLP", 0)

            # 应通过 OP 路径执行（OP 应有 run 记录）
            assert result.status == TrialStatus.SUCCESS
            runs = orch.list_runs()
            assert len(runs) >= 1
            # 至少有一个 run 是 succeeded
            succeeded_runs = [r for r in runs if r.phase == PHASE_SUCCEEDED]
            assert len(succeeded_runs) >= 1
        finally:
            orch.shutdown()

    def test_use_op_true_failure_marks_run_failed(self):
        """use_op=True 时训练失败应标记 OP run 为 failed。"""
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("failed")
                result = runner.run("synthetic", "MLP", 0)

            # 应标记为 FAILED
            assert result.status == TrialStatus.FAILED
            failed_runs = [r for r in orch.list_runs() if r.phase == PHASE_FAILED]
            assert len(failed_runs) >= 1
        finally:
            orch.shutdown()

    def test_use_op_true_exception_marks_run_failed(self):
        """use_op=True 时 run_pipeline 抛异常应标记 OP run 为 failed。"""
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.side_effect = RuntimeError("training crashed")
                result = runner.run("synthetic", "MLP", 0)

            assert result.status == TrialStatus.FAILED
            failed_runs = [r for r in orch.list_runs() if r.phase == PHASE_FAILED]
            assert len(failed_runs) >= 1
        finally:
            orch.shutdown()

    def test_use_op_true_params_persisted_in_run(self):
        """use_op=True 时 trial.params 应持久化到 PipelineRun.params。"""
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("success")
                runner.run("synthetic", "MLP", 0)

            # PipelineRun.params 应含 SP trial 的 params
            runs = orch.list_runs()
            assert len(runs) >= 1
            # 至少一个 run 有非空 params
            runs_with_params = [r for r in runs if r.params]
            assert len(runs_with_params) >= 1
            # params 应含 lr（搜索空间参数）
            assert "lr" in runs_with_params[0].params
        finally:
            orch.shutdown()

    def test_use_op_true_complete_failure_does_not_propagate(self):
        """资源泄露修复：orch.complete 抛异常不应传播给调用方。

        场景：run_pipeline 成功，但 orch.complete 因状态机/I/O 异常失败。
        不保护时：异常会传播，调用方看到非预期错误，OP run 卡 PHASE_RUNNING。
        修复后：异常仅记录日志，run() 返回 SUCCESS TrialResult。
        """
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline, \
                 patch.object(orch, "complete", side_effect=RuntimeError("OP I/O failed")):
                mock_pipeline.return_value = _make_mock_train_output("success")
                # 不应抛异常（OP complete 失败被隔离）
                result = runner.run("synthetic", "MLP", 0)

            # 训练本身成功 → TrialResult 仍应标记 SUCCESS
            assert result.status == TrialStatus.SUCCESS
        finally:
            orch.shutdown()

    def test_use_op_true_fail_failure_does_not_mask_original_error(self):
        """资源泄露修复：orch.fail 抛异常不应掩盖原 run_pipeline 异常。

        场景：run_pipeline 抛 RuntimeError("training crashed")，随后
        orch.fail 又因 I/O 异常失败。不保护时：调用方看到 OP I/O 异常
        而非原始 training crashed 异常，调试困难。修复后：仍抛原始异常。
        """
        config = _make_method_config()
        sm = StudyManager()
        study_id = sm.create_study(
            "test", direction="maximize",
            search_space=config.search_space, sampler="random",
        )
        orch = Orchestrator()
        try:
            runner = MethodRunner(
                config=config, study_id=study_id, study_manager=sm,
                use_op=True, orchestrator=orch,
            )

            original_error = RuntimeError("training crashed")
            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline, \
                 patch.object(orch, "fail", side_effect=RuntimeError("OP I/O failed")):
                mock_pipeline.side_effect = original_error
                # MethodRunner.run() 内部捕获异常并返回 FAILED TrialResult
                # （见 method.py run() 方法的 try/except 块）
                result = runner.run("synthetic", "MLP", 0)

            # 应标记 FAILED，且 error_msg 含原始异常信息
            assert result.status == TrialStatus.FAILED
            assert "training crashed" in (result.error_msg or "")
            # OP I/O 异常信息不应出现在 error_msg 中（被隔离）
            assert "OP I/O failed" not in (result.error_msg or "")
        finally:
            orch.shutdown()


# ============================================================
# P2.13: ExperimentRunner 事件驱动聚合
# ============================================================
class TestExperimentRunnerEventDriven:
    """ExperimentRunner 事件驱动聚合验证（P2.13）。"""

    def test_use_op_default_false(self):
        """use_op 默认应为 False（向后兼容）。"""
        config = _make_method_config()
        design = _make_experiment_design(config)
        runner = ExperimentRunner(design=design)
        assert runner.use_op is False

    def test_use_op_true_subscribes_events(self):
        """use_op=True 时应订阅 OP 事件。"""
        config = _make_method_config()
        design = _make_experiment_design(config)
        orch = Orchestrator()
        try:
            runner = ExperimentRunner(
                design=design, use_op=True, orchestrator=orch,
            )
            # 初始 event_log 应为空
            assert runner.get_event_log() == []

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("success")
                report = runner.run()

            # 应收集到事件（至少 1 个 succeeded）
            events = runner.get_event_log()
            assert len(events) >= 1
            # 应有 succeeded 事件
            succeeded_events = [
                e for e in events
                if e["event_type"] == EVENT_PIPELINE_SUCCEEDED
            ]
            assert len(succeeded_events) >= 1
        finally:
            orch.shutdown()

    def test_use_op_true_collects_failure_events(self):
        """use_op=True + 训练失败时应收集 failed 事件。"""
        config = _make_method_config()
        design = _make_experiment_design(config)
        orch = Orchestrator()
        try:
            runner = ExperimentRunner(
                design=design, use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("failed")
                report = runner.run()

            events = runner.get_event_log()
            # 应有 failed 事件
            failed_events = [
                e for e in events
                if e["event_type"] == EVENT_PIPELINE_FAILED
            ]
            assert len(failed_events) >= 1
        finally:
            orch.shutdown()

    def test_use_op_true_unsubscribes_after_run(self):
        """use_op=True 时 run() 结束后应取消订阅。"""
        config = _make_method_config()
        design = _make_experiment_design(config)
        orch = Orchestrator()
        try:
            runner = ExperimentRunner(
                design=design, use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("success")
                runner.run()

            # 取消订阅后，手动发射事件不应被收集
            initial_count = len(runner.get_event_log())
            # 手动通过 orch 发射一个事件
            from senseframe.orchestration import make_event
            orch._emit_event(
                EVENT_PIPELINE_SUCCEEDED, "fake_run",
                {"phase": PHASE_SUCCEEDED},
            )
            # event_log 不应增长
            assert len(runner.get_event_log()) == initial_count
        finally:
            orch.shutdown()

    def test_use_op_true_event_log_contains_run_id(self):
        """use_op=True 时事件应包含 run_id。"""
        config = _make_method_config()
        design = _make_experiment_design(config)
        orch = Orchestrator()
        try:
            runner = ExperimentRunner(
                design=design, use_op=True, orchestrator=orch,
            )

            with patch("senseframe.experiment.method.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = _make_mock_train_output("success")
                runner.run()

            events = runner.get_event_log()
            assert len(events) >= 1
            # 每个事件应有 run_id 字段
            for e in events:
                assert "run_id" in e
                assert e["run_id"]  # 非空
        finally:
            orch.shutdown()


# ============================================================
# 反假绿：grep 实证检查（源码不可绕过）
# ============================================================
class TestGrepEvidence:
    """源码 grep 实证：mock 可绕过运行时，但绕不过源码 grep。"""

    def test_orchestration_has_start_and_execute(self):
        """orchestration.py 应定义 start_and_execute 方法。"""
        path = _source_path("orchestration.py")
        assert _grep_source(path, "def start_and_execute("), \
            "Orchestrator 应有 start_and_execute 方法（P2.11）"

    def test_orchestration_has_wait_for_completion(self):
        """orchestration.py 应定义 wait_for_completion 方法。"""
        path = _source_path("orchestration.py")
        assert _grep_source(path, "def wait_for_completion("), \
            "Orchestrator 应有 wait_for_completion 方法（P2.11）"

    def test_orchestration_has_execute_pipeline(self):
        """orchestration.py 应定义 _execute_pipeline 内部方法。"""
        path = _source_path("orchestration.py")
        assert _grep_source(path, "def _execute_pipeline("), \
            "Orchestrator 应有 _execute_pipeline 方法（P2.11）"

    def test_orchestration_has_thread_pool_executor(self):
        """orchestration.py 应使用 ThreadPoolExecutor。"""
        path = _source_path("orchestration.py")
        assert _grep_source(path, "ThreadPoolExecutor"), \
            "应使用 ThreadPoolExecutor（P2.11 异步执行）"
        assert _grep_source(path, "from concurrent.futures"), \
            "应 import concurrent.futures（P2.11）"

    def test_orchestration_has_shutdown(self):
        """orchestration.py 应有 shutdown 方法（资源清理）。"""
        path = _source_path("orchestration.py")
        assert _grep_source(path, "def shutdown("), \
            "Orchestrator 应有 shutdown 方法（P2.11 资源清理）"

    def test_method_py_has_use_op_parameter(self):
        """method.py 应有 use_op 参数。"""
        path = _source_path("experiment/method.py")
        assert _grep_source(path, "use_op: bool = False"), \
            "MethodRunner 应有 use_op 参数（P2.12）"

    def test_method_py_has_run_pipeline_via_op(self):
        """method.py 应有 _run_pipeline_via_op 方法。"""
        path = _source_path("experiment/method.py")
        assert _grep_source(path, "def _run_pipeline_via_op("), \
            "MethodRunner 应有 _run_pipeline_via_op 方法（P2.12）"

    def test_method_py_uses_create_run_and_start(self):
        """method.py 应调用 OP create_run + start + complete/fail。"""
        path = _source_path("experiment/method.py")
        assert _grep_source(path, "orch.create_run("), \
            "应调用 orch.create_run（P2.12）"
        assert _grep_source(path, "orch.start("), \
            "应调用 orch.start（P2.12）"
        assert _grep_source(path, "orch.complete("), \
            "应调用 orch.complete（P2.12）"
        assert _grep_source(path, "orch.fail("), \
            "应调用 orch.fail（P2.12）"

    def test_runner_py_has_use_op_parameter(self):
        """runner.py 应有 use_op 参数。"""
        path = _source_path("experiment/runner.py")
        assert _grep_source(path, "use_op: bool = False"), \
            "ExperimentRunner 应有 use_op 参数（P2.13）"

    def test_runner_py_has_subscribe_op_events(self):
        """runner.py 应有 _subscribe_op_events 方法。"""
        path = _source_path("experiment/runner.py")
        assert _grep_source(path, "def _subscribe_op_events("), \
            "ExperimentRunner 应有 _subscribe_op_events 方法（P2.13）"

    def test_runner_py_subscribes_pipeline_events(self):
        """runner.py 应订阅 EVENT_PIPELINE_SUCCEEDED / FAILED。"""
        path = _source_path("experiment/runner.py")
        assert _grep_source(path, "EVENT_PIPELINE_SUCCEEDED"), \
            "应订阅 EVENT_PIPELINE_SUCCEEDED（P2.13）"
        assert _grep_source(path, "EVENT_PIPELINE_FAILED"), \
            "应订阅 EVENT_PIPELINE_FAILED（P2.13）"

    def test_runner_py_has_get_event_log(self):
        """runner.py 应有 get_event_log 方法。"""
        path = _source_path("experiment/runner.py")
        assert _grep_source(path, "def get_event_log("), \
            "ExperimentRunner 应有 get_event_log 方法（P2.13）"

    def test_runner_py_thread_safe_event_collection(self):
        """runner.py 应使用线程锁保护事件收集。"""
        path = _source_path("experiment/runner.py")
        assert _grep_source(path, "threading.Lock"), \
            "应使用 threading.Lock 保护事件收集（P2.13 线程安全）"
        assert _grep_source(path, "_event_lock"), \
            "应有 _event_lock 字段（P2.13 线程安全）"
