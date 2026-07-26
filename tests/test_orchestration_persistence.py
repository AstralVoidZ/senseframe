"""P3.4 OP 持久化 + K8s Operator 适配测试。

反假绿测试策略（同 test_meta_learning.py / test_nas.py 风格）：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- 真实 FileOrchestrationStore 实例（不 mock，验证真实文件 I/O）
- 真实 Orchestrator + 真实 PipelineRun 生命周期（验证真实状态持久化）
- 真实 FileEventSink + 真实 CloudEvent 序列化（验证 JSONL 日志真实写入）
- 真实 K8sOperatorAdapter 双向序列化（to_cr_manifest / from_cr_manifest round-trip）

覆盖（P3.4 五个工作项）：
- P3.4.1: StageStatus.from_dict / PipelineRun.from_dict + FileOrchestrationStore
- P3.4.2: Orchestrator 集成持久化 + recover
- P3.4.3: CloudEvent 外部 sink（FileEventSink）
- P3.4.4: K8sOperatorAdapter 接口准备
- P3.4.5: 反假绿 grep 实证 + 端到端集成
"""
from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from senseframe.orchestration import (
    EVENT_PIPELINE_FAILED,
    EVENT_PIPELINE_STARTED,
    EVENT_PIPELINE_SUCCEEDED,
    EVENT_STAGE_SUCCEEDED,
    EVENT_TRIAL_COMPLETED,
    PHASE_FAILED,
    PHASE_PAUSED,
    PHASE_PENDING,
    PHASE_RUNNING,
    PHASE_SUCCEEDED,
    CloudEvent,
    EventSink,
    FileEventSink,
    K8sOperatorAdapter,
    Orchestrator,
    PipelineDef,
    PipelineRun,
    StageStatus,
    make_event,
)
from senseframe.orchestration_store import (
    FileOrchestrationStore,
    OrchestrationStore,
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


def _make_pipeline_def(name: str = "test_pipeline") -> PipelineDef:
    """构造测试用 PipelineDef（含 2 个 stage，最小化）。"""
    return PipelineDef(
        name=name,
        stages=[
            _make_stage_template("validate"),
            _make_stage_template("train"),
        ],
    )


def _make_stage_template(name: str):
    """构造 StageTemplate（避免直接导入内部类，用 PipelineDef.stages 默认工厂）。"""
    from senseframe.orchestration import StageTemplate
    return StageTemplate(name=name)


def _make_stage_status(
    name: str = "validate",
    phase: str = "succeeded",
    started_at: str = "2026-07-05T00:00:00",
    finished_at: str = "2026-07-05T00:01:00",
    checkpoint_uri: str = "file:///tmp/ckpt",
    error: str = "",
) -> StageStatus:
    """构造测试用 StageStatus（完整字段）。"""
    return StageStatus(
        name=name,
        phase=phase,
        started_at=started_at,
        finished_at=finished_at,
        checkpoint_uri=checkpoint_uri,
        error=error,
    )


def _make_pipeline_run(
    run_id: str = "run_test1234",
    pipeline_ref: str = "test_pipeline",
    owner_reference: str = "test_pipeline",
    phase: str = PHASE_RUNNING,
    with_stages: bool = True,
) -> PipelineRun:
    """构造测试用 PipelineRun（含完整字段）。"""
    stages = [_make_stage_status(), _make_stage_status(name="train", phase="running")] if with_stages else []
    return PipelineRun(
        run_id=run_id,
        pipeline_ref=pipeline_ref,
        owner_reference=owner_reference,
        params={"lr": 0.01, "batch_size": 32},
        checkpoint_uri="file:///tmp/run_ckpt",
        phase=phase,
        stages=stages,
        started_at="2026-07-05T00:00:00",
        finished_at="",
        output_uri="",
        error="",
        retry_count=0,
    )


# ============================================================
# P3.4.1 (前置): StageStatus.from_dict 测试
# ============================================================
class TestStageStatusFromDict:
    """StageStatus.from_dict 反序列化测试（P3.4.1 前置依赖）。"""

    def test_stage_status_from_dict_roundtrip(self):
        """to_dict → from_dict 等价（round-trip）。"""
        original = _make_stage_status()
        d = original.to_dict()
        restored = StageStatus.from_dict(d)
        assert restored.name == original.name
        assert restored.phase == original.phase
        assert restored.started_at == original.started_at
        assert restored.finished_at == original.finished_at
        assert restored.checkpoint_uri == original.checkpoint_uri
        assert restored.error == original.error

    def test_stage_status_from_dict_with_defaults(self):
        """缺失字段用默认值（向后兼容旧版序列化数据）。"""
        # 仅 name 字段
        minimal = {"name": "validate"}
        restored = StageStatus.from_dict(minimal)
        assert restored.name == "validate"
        assert restored.phase == "pending"  # 默认
        assert restored.started_at == ""
        assert restored.finished_at == ""
        assert restored.checkpoint_uri == ""
        assert restored.error == ""

    def test_stage_status_from_dict_complete(self):
        """完整字段构造（含 error）。"""
        d = {
            "name": "train",
            "phase": "failed",
            "started_at": "2026-07-05T00:00:00",
            "finished_at": "2026-07-05T00:02:00",
            "checkpoint_uri": "file:///tmp/train_ckpt",
            "error": "OOM",
        }
        restored = StageStatus.from_dict(d)
        assert restored.name == "train"
        assert restored.phase == "failed"
        assert restored.started_at == "2026-07-05T00:00:00"
        assert restored.finished_at == "2026-07-05T00:02:00"
        assert restored.checkpoint_uri == "file:///tmp/train_ckpt"
        assert restored.error == "OOM"


# ============================================================
# P3.4.1 (前置): PipelineRun.from_dict 测试
# ============================================================
class TestPipelineRunFromDict:
    """PipelineRun.from_dict 反序列化测试（P3.4.1 前置依赖）。"""

    def test_pipeline_run_from_dict_roundtrip(self):
        """to_dict → from_dict 等价（round-trip）。"""
        original = _make_pipeline_run()
        d = original.to_dict()
        restored = PipelineRun.from_dict(d)
        assert restored.run_id == original.run_id
        assert restored.pipeline_ref == original.pipeline_ref
        assert restored.owner_reference == original.owner_reference
        assert restored.params == original.params
        assert restored.checkpoint_uri == original.checkpoint_uri
        assert restored.phase == original.phase
        assert restored.started_at == original.started_at
        assert restored.finished_at == original.finished_at
        assert restored.output_uri == original.output_uri
        assert restored.error == original.error
        assert restored.retry_count == original.retry_count

    def test_pipeline_run_from_dict_with_stages(self):
        """含 stages 列表递归构造（PipelineRun.from_dict 调用 StageStatus.from_dict）。"""
        original = _make_pipeline_run(with_stages=True)
        assert len(original.stages) == 2
        d = original.to_dict()
        restored = PipelineRun.from_dict(d)
        assert len(restored.stages) == 2
        # stages 递归构造
        assert all(isinstance(s, StageStatus) for s in restored.stages)
        assert restored.stages[0].name == "validate"
        assert restored.stages[0].phase == "succeeded"
        assert restored.stages[1].name == "train"
        assert restored.stages[1].phase == "running"

    def test_pipeline_run_from_dict_empty_stages(self):
        """空 stages 列表。"""
        original = _make_pipeline_run(with_stages=False)
        assert original.stages == []
        d = original.to_dict()
        restored = PipelineRun.from_dict(d)
        assert restored.stages == []

    def test_pipeline_run_from_dict_preserves_all_fields(self):
        """所有字段（run_id/pipeline_ref/owner_reference/params/checkpoint_uri/
        phase/started_at/finished_at/output_uri/error/retry_count）完整保留。"""
        original = PipelineRun(
            run_id="run_abc12345",
            pipeline_ref="my_pipeline",
            owner_reference="my_pipeline",
            params={"lr": 0.001, "epochs": 10, "loss": "focal"},
            checkpoint_uri="s3://bucket/run_abc12345/ckpt",
            phase=PHASE_FAILED,
            stages=[_make_stage_status(name="train", phase="failed", error="OOM")],
            started_at="2026-07-05T10:00:00",
            finished_at="2026-07-05T10:30:00",
            output_uri="s3://bucket/run_abc12345/output",
            error="Training failed: OOM",
            retry_count=2,
        )
        d = original.to_dict()
        restored = PipelineRun.from_dict(d)
        # 所有字段完整保留
        assert restored.run_id == "run_abc12345"
        assert restored.pipeline_ref == "my_pipeline"
        assert restored.owner_reference == "my_pipeline"
        assert restored.params == {"lr": 0.001, "epochs": 10, "loss": "focal"}
        assert restored.checkpoint_uri == "s3://bucket/run_abc12345/ckpt"
        assert restored.phase == PHASE_FAILED
        assert restored.started_at == "2026-07-05T10:00:00"
        assert restored.finished_at == "2026-07-05T10:30:00"
        assert restored.output_uri == "s3://bucket/run_abc12345/output"
        assert restored.error == "Training failed: OOM"
        assert restored.retry_count == 2
        # stages 也完整保留
        assert len(restored.stages) == 1
        assert restored.stages[0].name == "train"
        assert restored.stages[0].phase == "failed"
        assert restored.stages[0].error == "OOM"


# ============================================================
# P3.4.1: FileOrchestrationStore 测试
# ============================================================
class TestFileOrchestrationStore:
    """FileOrchestrationStore 文件系统持久化测试（P3.4.1）。"""

    def test_save_and_load_run(self, tmp_path):
        """保存后加载内容一致。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run = _make_pipeline_run()
        store.save_run(run)
        loaded = store.load_run(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.pipeline_ref == run.pipeline_ref
        assert loaded.phase == run.phase
        assert loaded.params == run.params
        # stages 也一致
        assert len(loaded.stages) == len(run.stages)
        assert loaded.stages[0].name == run.stages[0].name

    def test_load_nonexistent_run_returns_none(self, tmp_path):
        """加载不存在的 run_id 返回 None（不抛异常）。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        result = store.load_run("nonexistent_run")
        assert result is None

    def test_list_runs_empty(self, tmp_path):
        """空目录返回空 list。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        result = store.list_runs()
        assert result == []
        assert isinstance(result, list)

    def test_list_runs_all(self, tmp_path):
        """列出所有 run。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run1 = _make_pipeline_run(run_id="run_aaaa1111")
        run2 = _make_pipeline_run(run_id="run_bbbb2222")
        run3 = _make_pipeline_run(run_id="run_cccc3333")
        store.save_run(run1)
        store.save_run(run2)
        store.save_run(run3)

        runs = store.list_runs()
        assert len(runs) == 3
        run_ids = {r.run_id for r in runs}
        assert run_ids == {"run_aaaa1111", "run_bbbb2222", "run_cccc3333"}

    def test_list_runs_filtered_by_pipeline_ref(self, tmp_path):
        """按 pipeline_ref 过滤。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run1 = _make_pipeline_run(run_id="run_aaaa1111", pipeline_ref="pipeline_a")
        run2 = _make_pipeline_run(run_id="run_bbbb2222", pipeline_ref="pipeline_b")
        run3 = _make_pipeline_run(run_id="run_cccc3333", pipeline_ref="pipeline_a")
        store.save_run(run1)
        store.save_run(run2)
        store.save_run(run3)

        # 过滤 pipeline_a
        runs_a = store.list_runs(pipeline_ref="pipeline_a")
        assert len(runs_a) == 2
        assert all(r.pipeline_ref == "pipeline_a" for r in runs_a)

        # 过滤 pipeline_b
        runs_b = store.list_runs(pipeline_ref="pipeline_b")
        assert len(runs_b) == 1
        assert runs_b[0].run_id == "run_bbbb2222"

        # 过滤不存在的 pipeline_ref
        runs_none = store.list_runs(pipeline_ref="nonexistent")
        assert runs_none == []

    def test_delete_run(self, tmp_path):
        """删除后 load 返回 None。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run = _make_pipeline_run()
        store.save_run(run)
        assert store.load_run(run.run_id) is not None

        store.delete_run(run.run_id)
        assert store.load_run(run.run_id) is None

    def test_delete_nonexistent_run_no_error(self, tmp_path):
        """删除不存在的 run 不报错（幂等）。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        # 不抛异常
        store.delete_run("nonexistent_run")

    def test_save_creates_directory(self, tmp_path):
        """保存时自动创建目录（含父目录）。"""
        store = FileOrchestrationStore(base_dir=tmp_path / "nested" / "deep")
        # 目录尚未存在
        assert not (tmp_path / "nested" / "deep").exists()

        run = _make_pipeline_run()
        store.save_run(run)

        # 目录已自动创建
        assert (tmp_path / "nested" / "deep").exists()
        assert (tmp_path / "nested" / "deep" / f"{run.run_id}.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        """重复保存覆盖旧文件。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run = _make_pipeline_run(phase=PHASE_PENDING)
        store.save_run(run)

        # 修改 phase 后再保存
        run.phase = PHASE_SUCCEEDED
        run.output_uri = "file:///tmp/output"
        store.save_run(run)

        loaded = store.load_run(run.run_id)
        assert loaded is not None
        assert loaded.phase == PHASE_SUCCEEDED
        assert loaded.output_uri == "file:///tmp/output"

    def test_save_preserves_run_structure(self, tmp_path):
        """保存的 run 结构完整（含 stages）。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        run = _make_pipeline_run(with_stages=True)
        store.save_run(run)

        loaded = store.load_run(run.run_id)
        assert loaded is not None
        assert len(loaded.stages) == 2
        assert loaded.stages[0].name == "validate"
        assert loaded.stages[0].phase == "succeeded"
        assert loaded.stages[0].checkpoint_uri == "file:///tmp/ckpt"
        assert loaded.stages[1].name == "train"
        assert loaded.stages[1].phase == "running"
        # params 也完整保留
        assert loaded.params == {"lr": 0.01, "batch_size": 32}


# ============================================================
# P3.4.2: Orchestrator 集成持久化测试
# ============================================================
class TestOrchestratorPersistence:
    """Orchestrator 持久化集成测试（P3.4.2）。"""

    def test_orchestrator_default_no_store(self):
        """默认构造无 store（向后兼容）。"""
        orch = Orchestrator()
        assert orch._store is None
        try:
            pass
        finally:
            orch.shutdown()

    def test_orchestrator_with_store_persists_create_run(self, tmp_path):
        """create_run 后 store 有对应文件。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)

            # store 应有对应文件
            loaded = store.load_run(run_id)
            assert loaded is not None
            assert loaded.run_id == run_id
            assert loaded.pipeline_ref == pdef.name
            assert loaded.phase == PHASE_PENDING
        finally:
            orch.shutdown()

    def test_orchestrator_with_store_persists_complete(self, tmp_path):
        """complete 后 store 文件状态更新为 succeeded。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            orch.start(run_id)
            orch.complete(run_id, output_uri="file:///tmp/output")

            loaded = store.load_run(run_id)
            assert loaded is not None
            assert loaded.phase == PHASE_SUCCEEDED
            assert loaded.output_uri == "file:///tmp/output"
        finally:
            orch.shutdown()

    def test_orchestrator_with_store_persists_fail(self, tmp_path):
        """fail 后 store 文件状态更新为 failed。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            orch.start(run_id)
            orch.fail(run_id, error="OOM", stage_name="train")

            loaded = store.load_run(run_id)
            assert loaded is not None
            assert loaded.phase == PHASE_FAILED
            assert loaded.error == "OOM"
            # stage 也更新
            stage = next(s for s in loaded.stages if s.name == "train")
            assert stage.phase == "failed"
            assert stage.error == "OOM"
        finally:
            orch.shutdown()

    def test_orchestrator_recover_empty_store(self, tmp_path):
        """recover 空存储返回空 list。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch = Orchestrator(store=store)
        try:
            recovered = orch.recover()
            assert recovered == []
        finally:
            orch.shutdown()

    def test_orchestrator_recover_restores_runs(self, tmp_path):
        """recover 后 _runs 含原 run。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        # 第一个 Orchestrator 创建并持久化 run
        orch1 = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch1.create_pipeline(pdef)
            run_id = orch1.create_run(pdef.name)
            orch1.start(run_id)
            orch1.complete(run_id, output_uri="file:///tmp/output")
        finally:
            orch1.shutdown()

        # 第二个 Orchestrator 从 store 恢复
        orch2 = Orchestrator(store=store)
        try:
            recovered = orch2.recover()
            assert run_id in recovered
            # _runs 应含原 run
            restored_run = orch2.get_run(run_id)
            assert restored_run is not None
            assert restored_run.run_id == run_id
            assert restored_run.phase == PHASE_SUCCEEDED
            assert restored_run.output_uri == "file:///tmp/output"
        finally:
            orch2.shutdown()

    def test_orchestrator_recover_returns_run_ids(self, tmp_path):
        """recover 返回 run_id 列表。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch1 = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch1.create_pipeline(pdef)
            run_id1 = orch1.create_run(pdef.name)
            run_id2 = orch1.create_run(pdef.name)
            run_id3 = orch1.create_run(pdef.name)
        finally:
            orch1.shutdown()

        orch2 = Orchestrator(store=store)
        try:
            recovered = orch2.recover()
            assert len(recovered) == 3
            assert run_id1 in recovered
            assert run_id2 in recovered
            assert run_id3 in recovered
        finally:
            orch2.shutdown()

    def test_orchestrator_recover_without_store_returns_empty(self):
        """无 store 时 recover 返回 []（向后兼容）。"""
        orch = Orchestrator()
        try:
            recovered = orch.recover()
            assert recovered == []
        finally:
            orch.shutdown()

    def test_orchestrator_default_no_event_sink(self):
        """默认构造无 event_sink（向后兼容）。"""
        orch = Orchestrator()
        assert orch.event_sink is None
        try:
            pass
        finally:
            orch.shutdown()

    def test_orchestrator_state_transitions_persisted(self, tmp_path):
        """所有状态变更方法都持久化（start/pause/resume/retry/stop/complete/fail）。"""
        store = FileOrchestrationStore(base_dir=tmp_path)
        orch = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)

            # create_run
            run_id = orch.create_run(pdef.name)
            assert store.load_run(run_id).phase == PHASE_PENDING

            # start
            orch.start(run_id)
            assert store.load_run(run_id).phase == PHASE_RUNNING

            # pause
            orch.pause(run_id)
            assert store.load_run(run_id).phase == PHASE_PAUSED

            # resume
            orch.resume(run_id)
            assert store.load_run(run_id).phase == PHASE_RUNNING

            # stop（标记 failed）
            orch.stop(run_id)
            assert store.load_run(run_id).phase == PHASE_FAILED
            assert store.load_run(run_id).error == "Stopped by orchestrator"
        finally:
            orch.shutdown()


# ============================================================
# P3.4.3: FileEventSink 测试
# ============================================================
class TestFileEventSink:
    """FileEventSink CloudEvent 文件 sink 测试（P3.4.3）。"""

    def test_file_event_sink_emit_writes_jsonl(self, tmp_path):
        """emit 写入 JSONL 格式（每行一个 event JSON）。"""
        log_path = tmp_path / "events.jsonl"
        sink = FileEventSink(log_path)
        event = make_event(EVENT_PIPELINE_STARTED, "run_test", {"phase": "running"})
        sink.emit(event)

        content = log_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1
        # 每行是合法 JSON
        parsed = json.loads(lines[0])
        assert parsed["type"] == EVENT_PIPELINE_STARTED
        assert parsed["source"] == "/senseframe/pipeline/run_test"
        assert parsed["data"] == {"phase": "running"}

    def test_file_event_sink_creates_parent_dir(self, tmp_path):
        """自动创建父目录（含嵌套父目录）。"""
        log_path = tmp_path / "nested" / "deep" / "events.jsonl"
        # 父目录不存在
        assert not log_path.parent.exists()

        sink = FileEventSink(log_path)
        # 父目录已自动创建
        assert log_path.parent.exists()

        # emit 也能正常写入
        event = make_event(EVENT_PIPELINE_STARTED, "run_test", {"phase": "running"})
        sink.emit(event)
        assert log_path.exists()

    def test_file_event_sink_appends(self, tmp_path):
        """多次 emit 追加写入（不覆盖）。"""
        log_path = tmp_path / "events.jsonl"
        sink = FileEventSink(log_path)

        sink.emit(make_event(EVENT_PIPELINE_STARTED, "run_1", {"phase": "running"}))
        sink.emit(make_event(EVENT_PIPELINE_SUCCEEDED, "run_1", {"phase": "succeeded"}))
        sink.emit(make_event(EVENT_PIPELINE_STARTED, "run_2", {"phase": "running"}))

        content = log_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3
        # 每行都是合法 JSON
        events = [json.loads(line) for line in lines]
        assert events[0]["type"] == EVENT_PIPELINE_STARTED
        assert events[0]["source"] == "/senseframe/pipeline/run_1"
        assert events[1]["type"] == EVENT_PIPELINE_SUCCEEDED
        assert events[2]["type"] == EVENT_PIPELINE_STARTED
        assert events[2]["source"] == "/senseframe/pipeline/run_2"

    def test_file_event_sink_with_real_cloud_event(self, tmp_path):
        """与真实 CloudEvent 集成（验证 to_json 输出可被 sink 写入并解析）。"""
        log_path = tmp_path / "events.jsonl"
        sink = FileEventSink(log_path)

        # 直接用 CloudEvent 实例（不经 make_event）
        event = CloudEvent(
            source="/senseframe/pipeline/run_real",
            type=EVENT_TRIAL_COMPLETED,
            data={"trial_id": "t1", "value": 0.85},
        )
        sink.emit(event)

        content = log_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        # CloudEvent 字段完整
        assert parsed["specversion"] == "1.0"
        assert parsed["id"] == event.id  # 自动生成
        assert parsed["source"] == "/senseframe/pipeline/run_real"
        assert parsed["type"] == EVENT_TRIAL_COMPLETED
        assert parsed["datacontenttype"] == "application/json"
        assert parsed["data"] == {"trial_id": "t1", "value": 0.85}

    def test_file_event_sink_to_json_format(self, tmp_path):
        """每行是合法 JSON（与 to_json 等价）。"""
        log_path = tmp_path / "events.jsonl"
        sink = FileEventSink(log_path)
        event = make_event(EVENT_STAGE_SUCCEEDED, "run_x", {"stage_name": "train"})

        sink.emit(event)

        content = log_path.read_text(encoding="utf-8")
        # sink 写入的内容应与 event.to_json() + "\n" 完全一致
        assert content == event.to_json() + "\n"


# ============================================================
# P3.4.3: Orchestrator + EventSink 集成测试
# ============================================================
class TestOrchestratorEventSinkIntegration:
    """Orchestrator + EventSink 集成测试（P3.4.3）。"""

    def test_orchestrator_with_event_sink_emits_to_sink(self, tmp_path):
        """Orchestrator 事件触发 sink emit。"""
        log_path = tmp_path / "events.jsonl"
        sink = FileEventSink(log_path)
        orch = Orchestrator(event_sink=sink)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            orch.start(run_id)

            # sink 日志应含 started 事件
            content = log_path.read_text(encoding="utf-8")
            assert EVENT_PIPELINE_STARTED in content
            lines = content.strip().split("\n")
            assert len(lines) >= 1
            # 第一行应是 started 事件
            parsed = json.loads(lines[0])
            assert parsed["type"] == EVENT_PIPELINE_STARTED
            assert run_id in parsed["source"]
        finally:
            orch.shutdown()

    def test_orchestrator_without_event_sink_no_error(self):
        """无 sink 时不报错（向后兼容）。"""
        orch = Orchestrator()  # 无 event_sink
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            # 不抛异常（_emit_event 中 self._event_sink is None 分支）
            orch.start(run_id)
            orch.complete(run_id, output_uri="file:///tmp/out")
        finally:
            orch.shutdown()

    def test_orchestrator_event_sink_with_real_pipeline_lifecycle(self, tmp_path):
        """真实生命周期事件写入 sink（start → complete）。"""
        log_path = tmp_path / "lifecycle.jsonl"
        sink = FileEventSink(log_path)
        orch = Orchestrator(event_sink=sink)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            orch.start(run_id)
            orch.complete(run_id, output_uri="file:///tmp/output")

            content = log_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            events = [json.loads(line) for line in lines]
            event_types = [e["type"] for e in events]

            # 应含 started + succeeded 事件
            assert EVENT_PIPELINE_STARTED in event_types
            assert EVENT_PIPELINE_SUCCEEDED in event_types
            # succeeded 事件的 data 应含 output_uri
            succeeded_events = [e for e in events if e["type"] == EVENT_PIPELINE_SUCCEEDED]
            assert len(succeeded_events) >= 1
            assert succeeded_events[-1]["data"]["output_uri"] == "file:///tmp/output"
        finally:
            orch.shutdown()


# ============================================================
# P3.4.4: K8sOperatorAdapter 测试
# ============================================================
class TestK8sOperatorAdapter:
    """K8sOperatorAdapter K8s CR 适配层测试（P3.4.4）。"""

    def test_to_cr_manifest_returns_dict(self):
        """to_cr_manifest 返回 dict。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run()
        manifest = adapter.to_cr_manifest(run)
        assert isinstance(manifest, dict)

    def test_to_cr_manifest_has_apiversion_kind(self):
        """含 apiVersion + kind（K8s CRD 规范）。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run()
        manifest = adapter.to_cr_manifest(run)
        assert manifest["apiVersion"] == "senseframe.io/v1"
        assert manifest["kind"] == "PipelineRun"

    def test_to_cr_manifest_metadata_name_matches_run_id(self):
        """metadata.name == run.run_id。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run(run_id="run_k8s_test")
        manifest = adapter.to_cr_manifest(run)
        assert manifest["metadata"]["name"] == "run_k8s_test"

    def test_to_cr_manifest_spec_contains_pipeline_ref(self):
        """spec.pipelineRef == run.pipeline_ref。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run(pipeline_ref="my_pipeline")
        manifest = adapter.to_cr_manifest(run)
        assert manifest["spec"]["pipelineRef"] == "my_pipeline"
        assert manifest["spec"]["params"] == run.params
        assert manifest["spec"]["checkpointUri"] == run.checkpoint_uri

    def test_to_cr_manifest_status_contains_phase(self):
        """status.phase == run.phase。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run(phase=PHASE_SUCCEEDED)
        manifest = adapter.to_cr_manifest(run)
        assert manifest["status"]["phase"] == PHASE_SUCCEEDED
        assert manifest["status"]["startedAt"] == run.started_at
        assert manifest["status"]["finishedAt"] == run.finished_at
        assert manifest["status"]["outputUri"] == run.output_uri
        assert manifest["status"]["error"] == run.error
        assert manifest["status"]["retryCount"] == run.retry_count
        # stages 也应被序列化
        assert len(manifest["status"]["stages"]) == len(run.stages)

    def test_to_cr_manifest_with_owner_reference(self):
        """owner_reference 非 None 时含 ownerReferences。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run(owner_reference="parent_pipeline")
        manifest = adapter.to_cr_manifest(run)
        owner_refs = manifest["metadata"]["ownerReferences"]
        assert len(owner_refs) == 1
        assert owner_refs[0]["apiVersion"] == "senseframe.io/v1"
        assert owner_refs[0]["kind"] == "PipelineDef"
        assert owner_refs[0]["name"] == "parent_pipeline"

    def test_to_cr_manifest_without_owner_reference(self):
        """owner_reference 为 None 时不含 ownerReferences。"""
        adapter = K8sOperatorAdapter()
        run = _make_pipeline_run(owner_reference=None)
        manifest = adapter.to_cr_manifest(run)
        # ownerReferences 不应存在
        assert "ownerReferences" not in manifest["metadata"]

    def test_from_cr_manifest_roundtrip(self):
        """to_cr_manifest → from_cr_manifest 等价（关键字段）。"""
        adapter = K8sOperatorAdapter()
        original = _make_pipeline_run(
            run_id="run_roundtrip",
            pipeline_ref="rt_pipeline",
            owner_reference="rt_pipeline",
            phase=PHASE_SUCCEEDED,
        )
        original.output_uri = "file:///tmp/rt_output"
        original.error = ""
        manifest = adapter.to_cr_manifest(original)
        restored = adapter.from_cr_manifest(manifest)

        # 关键字段等价
        assert restored.run_id == original.run_id
        assert restored.pipeline_ref == original.pipeline_ref
        assert restored.owner_reference == original.owner_reference
        assert restored.params == original.params
        assert restored.checkpoint_uri == original.checkpoint_uri
        assert restored.phase == original.phase
        assert restored.started_at == original.started_at
        assert restored.finished_at == original.finished_at
        assert restored.output_uri == original.output_uri
        assert restored.error == original.error
        assert restored.retry_count == original.retry_count
        # stages 也应等价
        assert len(restored.stages) == len(original.stages)
        assert restored.stages[0].name == original.stages[0].name
        assert restored.stages[0].phase == original.stages[0].phase

    def test_from_cr_manifest_with_minimal_manifest(self):
        """最小 manifest（仅 metadata.name + spec.pipelineRef）可构造。"""
        adapter = K8sOperatorAdapter()
        minimal = {
            "apiVersion": "senseframe.io/v1",
            "kind": "PipelineRun",
            "metadata": {"name": "run_minimal"},
            "spec": {"pipelineRef": "minimal_pipeline"},
        }
        restored = adapter.from_cr_manifest(minimal)
        assert restored.run_id == "run_minimal"
        assert restored.pipeline_ref == "minimal_pipeline"
        # 缺失字段使用默认值
        assert restored.phase == PHASE_PENDING
        assert restored.params == {}
        assert restored.checkpoint_uri == ""
        assert restored.stages == []
        assert restored.owner_reference is None
        assert restored.retry_count == 0


# ============================================================
# P3.4.5: 反假绿 grep 实证测试
# ============================================================
class TestGrepEvidenceOpPersistence:
    """反射 + grep 实证：源码结构检查所有 P3.4 实现关键点（反假绿）。

    A 类（存在性）用反射 API；B 类（行为）保留 grep。
    """

    @pytest.mark.parametrize("module_path,attr_name", [
        ("senseframe.orchestration_store", "OrchestrationStore"),
        ("senseframe.orchestration_store", "OrchestrationStore.save_run"),
        ("senseframe.orchestration_store", "OrchestrationStore.load_run"),
        ("senseframe.orchestration_store", "OrchestrationStore.list_runs"),
        ("senseframe.orchestration_store", "OrchestrationStore.delete_run"),
        ("senseframe.orchestration_store", "FileOrchestrationStore"),
        ("senseframe.orchestration", "PipelineRun.from_dict"),
        ("senseframe.orchestration", "StageStatus.from_dict"),
        ("senseframe.orchestration", "Orchestrator.recover"),
        ("senseframe.orchestration", "FileEventSink"),
        ("senseframe.orchestration", "EventSink"),
        ("senseframe.orchestration", "EventSink.emit"),
        ("senseframe.orchestration", "K8sOperatorAdapter.to_cr_manifest"),
        ("senseframe.orchestration", "K8sOperatorAdapter.from_cr_manifest"),
    ])
    def test_attr_exists(self, module_path, attr_name):
        """反射实证：模块属性存在性（参数化）。"""
        mod = importlib.import_module(module_path)
        parts = attr_name.split(".")
        obj = mod
        for part in parts[:-1]:
            assert hasattr(obj, part), f"{module_path}.{part} 不存在"
            obj = getattr(obj, part)
        assert hasattr(obj, parts[-1]), f"{module_path}.{attr_name} 不存在"

    def test_orchestrator_init_store_param(self):
        """反射实证：Orchestrator.__init__ 含 store 和 event_sink 参数。"""
        sig = inspect.signature(Orchestrator.__init__)
        assert "store" in sig.parameters
        assert "event_sink" in sig.parameters

    def test_k8s_operator_adapter_class(self):
        """反射实证：K8sOperatorAdapter 类存在且含 K8s CRD 常量。"""
        assert inspect.isclass(K8sOperatorAdapter)
        assert hasattr(K8sOperatorAdapter, "API_VERSION")
        assert K8sOperatorAdapter.API_VERSION == "senseframe.io/v1"
        assert hasattr(K8sOperatorAdapter, "KIND")
        assert K8sOperatorAdapter.KIND == "PipelineRun"

    # test_grep_orchestrator_event_sink_emit 已删除：
    # 行为由 TestOrchestratorEventSinkIntegration 运行时测试覆盖
    # （test_orchestrator_with_event_sink_emits_to_sink / test_orchestrator_without_event_sink_no_error）


# ============================================================
# P3.4.5: 端到端集成测试
# ============================================================
class TestOpPersistenceIntegration:
    """P3.4 OP 持久化端到端集成测试。"""

    def test_full_persistence_flow(self, tmp_path):
        """完整持久化流程：Orchestrator(store) → create_run → start → complete
        → 新 Orchestrator → recover → 验证 run 状态恢复。"""
        store_dir = tmp_path / "store"
        store = FileOrchestrationStore(base_dir=store_dir)

        # 第一个 Orchestrator：创建并执行 run
        orch1 = Orchestrator(store=store)
        try:
            pdef = _make_pipeline_def()
            orch1.create_pipeline(pdef)
            run_id = orch1.create_run(pdef.name)
            orch1.start(run_id)
            orch1.complete(run_id, output_uri="file:///tmp/final_output")
        finally:
            orch1.shutdown()

        # 第二个 Orchestrator：从 store 恢复（模拟进程重启）
        orch2 = Orchestrator(store=store)
        try:
            recovered = orch2.recover()
            assert run_id in recovered

            # 验证 run 状态完整恢复
            restored_run = orch2.get_run(run_id)
            assert restored_run is not None
            assert restored_run.run_id == run_id
            assert restored_run.pipeline_ref == pdef.name
            assert restored_run.phase == PHASE_SUCCEEDED
            assert restored_run.output_uri == "file:///tmp/final_output"
            # stages 也恢复
            assert len(restored_run.stages) == 2
            assert restored_run.stages[0].name == "validate"
            assert restored_run.stages[1].name == "train"
        finally:
            orch2.shutdown()

    def test_full_event_sink_flow(self, tmp_path):
        """完整 event sink 流程：Orchestrator(event_sink) → create_run → start →
        complete → 验证日志文件含对应事件（JSONL 格式）。"""
        log_path = tmp_path / "lifecycle_events.jsonl"
        sink = FileEventSink(log_path)
        orch = Orchestrator(event_sink=sink)
        try:
            pdef = _make_pipeline_def()
            orch.create_pipeline(pdef)
            run_id = orch.create_run(pdef.name)
            orch.start(run_id)
            orch.complete(run_id, output_uri="file:///tmp/sink_output")

            # 验证日志文件含对应事件
            assert log_path.exists()
            content = log_path.read_text(encoding="utf-8")
            lines = content.strip().split("\n")
            assert len(lines) >= 2  # 至少 started + succeeded

            # 每行都是合法 JSON
            events = [json.loads(line) for line in lines]
            event_types = [e["type"] for e in events]

            # 应含 started + succeeded
            assert EVENT_PIPELINE_STARTED in event_types
            assert EVENT_PIPELINE_SUCCEEDED in event_types

            # 验证 run_id 在 source 中
            for e in events:
                assert run_id in e["source"]
        finally:
            orch.shutdown()
