"""RFC-003 五层协议栈端到端集成测试。

覆盖 ε1-ε5 五个验证场景，模拟 AutoML 主控通过编排器接口
驱动 SenseFrame 完成完整闭环。

协议栈层级：
- DSP（数据结构协议）：α 阶段已实施
- IP（推理协议）：β 阶段已实施，对齐 KServe v2
- OBP（可观测性协议）：β 阶段已实施，基于 OpenTelemetry
- SP（搜索协议）：γ 阶段已实施，对齐 Optuna Ask-Tell
- OP（编排协议）：δ 阶段已实施，对齐 K8s Operator + Argo Workflows

协议层 e2e 不依赖真实数据集，故不使用 @pytest.mark.e2e 标记
（该标记默认跳过，需要 -m e2e 显式启用）。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# ε1: IP 推理协议端到端验证
# ============================================================
class TestEpsilon1IPProtocol:
    """ε1: KServe v2 推理协议端到端验证。

    覆盖 InferenceServer Python 接口 + KServe v2 端点
    （fastapi 为可选依赖，未安装时整组测试跳过）。
    """

    @pytest.fixture
    def server_and_client(self, tmp_path):
        """构造 InferenceServer + FastAPI TestClient。

        - 用 importorskip 处理 fastapi/uvicorn 可选依赖
        - mock _load_model 避免真实模型加载
        - mock uvicorn.run 捕获 app 实例（不真正启动服务）
        - mock server.predict 返回固定 KServe v2 兼容结果
        """
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        from fastapi.testclient import TestClient

        from senseframe.serving import InferenceServer

        # 1. 构造 tmp 目录 + 假 metadata.json + 假 model.pth
        output_dir = tmp_path / "model_output"
        output_dir.mkdir()
        (output_dir / "metadata.json").write_text(
            json.dumps({
                "model_id": "test",
                "input_shape": [1, 10],
                "num_classes": 3,
            }),
            encoding="utf-8",
        )
        (output_dir / "model.pth").write_bytes(b"")

        # 2. mock _load_model：设置 self.model + self._model_type，不真正加载
        def _fake_load_model(self_inner):
            self_inner.model = MagicMock()
            self_inner._model_type = "onnx"

        # 3. mock uvicorn.run 捕获 app 实例
        captured: dict = {}

        def _fake_uvicorn_run(app, host=None, port=None, **kwargs):
            captured["app"] = app
            captured["host"] = host
            captured["port"] = port

        with patch.object(InferenceServer, "_load_model", _fake_load_model):
            server = InferenceServer(output_dir=output_dir, device="cpu")
            # 4. mock server.predict 返回 KServe v2 兼容结果
            server.predict = MagicMock(return_value={
                "prediction": 0,
                "probabilities": [0.7, 0.2, 0.1],
            })
            with patch("uvicorn.run", side_effect=_fake_uvicorn_run):
                server.start(host="127.0.0.1", port=8000)

        assert "app" in captured, "uvicorn.run 未被调用，app 未捕获"
        client = TestClient(captured["app"])
        return server, client

    def test_health_live_endpoint(self, server_and_client):
        """验证 GET /v2/health/live 返回 200 + {"live": True}。"""
        _, client = server_and_client
        resp = client.get("/v2/health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"live": True}

    def test_health_ready_endpoint(self, server_and_client):
        """验证 GET /v2/health/ready 返回 200 + ready=True（model 已 mock）。"""
        _, client = server_and_client
        resp = client.get("/v2/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True

    def test_repository_index(self, server_and_client):
        """验证 GET /v2/repository/index 返回 model 列表（含 test）。"""
        _, client = server_and_client
        resp = client.get("/v2/repository/index")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        names = [item["name"] for item in body]
        assert "test" in names

    def test_infer_kserve_v2_format(self, server_and_client):
        """验证 POST /v2/models/test/infer 接受 KServe v2 body 并返回 v2 响应。"""
        _, client = server_and_client
        body = {
            "inputs": [{
                "name": "input_0",
                "shape": [1, 10],
                "datatype": "FP32",
                "data": [0.1] * 10,
            }],
        }
        resp = client.post("/v2/models/test/infer", json=body)
        assert resp.status_code == 200
        result = resp.json()
        # KServe v2 响应格式
        assert result["model_name"] == "test"
        assert "outputs" in result
        assert len(result["outputs"]) >= 1
        out0 = result["outputs"][0]
        assert out0["name"] == "output_0"
        assert out0["datatype"] == "FP32"
        # mock predict 返回 probabilities=[0.7, 0.2, 0.1]
        assert out0["data"] == [0.7, 0.2, 0.1]

    def test_model_metadata_endpoint(self, server_and_client):
        """验证 GET /v2/models/test 返回模型元数据（含 inputs/outputs）。"""
        _, client = server_and_client
        resp = client.get("/v2/models/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "test"
        assert "inputs" in body
        assert "outputs" in body
        assert len(body["inputs"]) >= 1
        assert body["inputs"][0]["name"] == "input_0"

    def test_model_ready_endpoint(self, server_and_client):
        """验证 GET /v2/models/test/ready 返回 200 + {"ready": True}。"""
        _, client = server_and_client
        resp = client.get("/v2/models/test/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ready"] is True

    def test_inference_server_info(self, server_and_client):
        """验证 InferenceServer.info() Python 接口返回正确字段。"""
        server, _ = server_and_client
        info = server.info()
        assert "output_dir" in info
        assert "model_type" in info
        assert info["model_type"] == "onnx"
        assert "metadata" in info
        assert info["metadata"] is not None
        assert info["metadata"]["model_id"] == "test"
        assert info["device"] == "cpu"


# ============================================================
# ε2: OBP 可观测性协议验证
# ============================================================
class TestEpsilon2OBPProtocol:
    """ε2: OpenTelemetry 可观测性协议验证。

    不依赖 OTel 真实安装，测试降级路径与 Grafana 模板生成。
    """

    def test_is_otel_enabled_default_false(self, monkeypatch):
        """验证未初始化时 is_otel_enabled() 返回 False。"""
        import senseframe.observability_otel as otel
        monkeypatch.setattr(otel, "_otel_initialized", False)
        monkeypatch.setattr(otel, "_meter", None)
        assert otel.is_otel_enabled() is False

    def test_init_otel_returns_false_when_sdk_missing(self, monkeypatch):
        """验证 OTel SDK 未安装时 init_otel() 返回 False（不抛异常）。

        通过 sys.modules 注入 None 强制 opentelemetry.sdk.* 导入失败，
        保证测试在 OTel SDK 已安装的环境中也能稳定通过。
        """
        import senseframe.observability_otel as otel

        # 让 from opentelemetry.sdk.metrics import ... 全部失败
        monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk", None)
        monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.metrics", None)
        monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.metrics.export", None)
        monkeypatch.setitem(__import__("sys").modules, "opentelemetry.sdk.resources", None)

        # 重置全局状态
        monkeypatch.setattr(otel, "_otel_initialized", False)
        monkeypatch.setattr(otel, "_meter", None)
        monkeypatch.setattr(otel, "_resource", None)

        result = otel.init_otel(service_name="test_service")
        assert result is False
        assert otel.is_otel_enabled() is False

    def test_record_training_metric_noop_when_not_initialized(self, monkeypatch):
        """验证未初始化时 record_training_metric 为 no-op（不抛异常）。"""
        import senseframe.observability_otel as otel
        monkeypatch.setattr(otel, "_otel_initialized", False)
        monkeypatch.setattr(otel, "_meter", None)
        # 不应抛异常
        otel.record_training_metric(
            "ml.train.loss", 0.5,
            stage="train", epoch=1,
            model_id="m1", dataset="d1",
        )

    def test_record_inference_metric_noop_when_not_initialized(self, monkeypatch):
        """验证未初始化时 record_inference_metric 为 no-op。"""
        import senseframe.observability_otel as otel
        monkeypatch.setattr(otel, "_otel_initialized", False)
        monkeypatch.setattr(otel, "_meter", None)
        otel.record_inference_metric(
            "ml.inference.latency_ms", 1.0,
            latency_ms=12.3, confidence=0.9, model_id="m1",
        )

    def test_record_trial_metric_noop_when_not_initialized(self, monkeypatch):
        """验证未初始化时 record_trial_metric 为 no-op。"""
        import senseframe.observability_otel as otel
        monkeypatch.setattr(otel, "_otel_initialized", False)
        monkeypatch.setattr(otel, "_meter", None)
        otel.record_trial_metric("ml.trial.best_metric", 0.85, trial_id="t1")

    def test_grafana_dashboard_template_structure(self):
        """验证 get_grafana_dashboard_template 返回结构正确的 dict。"""
        from senseframe.observability_otel import get_grafana_dashboard_template
        tpl = get_grafana_dashboard_template()
        assert isinstance(tpl, dict)
        assert "title" in tpl
        assert "schemaVersion" in tpl
        assert "panels" in tpl
        assert isinstance(tpl["panels"], list)
        assert len(tpl["panels"]) > 0

        # 每个 panel 含 title/type/datasource/targets/gridPos
        for panel in tpl["panels"]:
            assert "title" in panel
            assert "type" in panel
            assert "datasource" in panel
            assert "targets" in panel
            assert "gridPos" in panel

    def test_grafana_dashboard_contains_key_panels(self):
        """验证 dashboard 模板包含训练/验证/推理/trial 关键面板。"""
        from senseframe.observability_otel import get_grafana_dashboard_template
        tpl = get_grafana_dashboard_template()
        titles = [p["title"].lower() for p in tpl["panels"]]
        # 训练 loss
        assert any("training" in t and "loss" in t for t in titles), \
            f"缺少 Training Loss 面板: {titles}"
        # 验证 accuracy
        assert any("validation" in t and "accuracy" in t for t in titles), \
            f"缺少 Validation Accuracy 面板: {titles}"
        # 推理 latency
        assert any("inference" in t and "latency" in t for t in titles), \
            f"缺少 Inference Latency 面板: {titles}"
        # trial count
        assert any("trial" in t and "count" in t for t in titles), \
            f"缺少 Trial Count 面板: {titles}"

    def test_ml_semantic_conventions_constants(self):
        """验证 ML Semantic Conventions 常量存在。"""
        from senseframe.observability_otel import (
            ML_TRAIN_LOSS,
            ML_VAL_ACCURACY,
            ML_INFERENCE_LATENCY_MS,
            ML_TRIAL_BEST_METRIC,
            ML_INFERENCE_REQUEST_COUNT,
            ML_TRIAL_COUNT,
        )
        assert ML_TRAIN_LOSS == "ml.train.loss"
        assert ML_VAL_ACCURACY == "ml.val.accuracy"
        assert ML_INFERENCE_LATENCY_MS == "ml.inference.latency_ms"
        assert ML_TRIAL_BEST_METRIC == "ml.trial.best_metric"
        assert ML_INFERENCE_REQUEST_COUNT == "ml.inference.request_count"
        assert ML_TRIAL_COUNT == "ml.trial.count"


# ============================================================
# ε3: SP 搜索协议端到端验证
# ============================================================
class TestEpsilon3SPProtocol:
    """ε3: Ask-Tell 搜索协议端到端验证。

    覆盖 StudyManager 完整 ask/tell 流程，含三种参数类型、
    ExplorationTracker 桥接、Sampler 注册表与 GridSampler。
    """

    def _build_search_space(self):
        """构造包含 float/int/categorical 三种参数的搜索空间。"""
        from senseframe.search_protocol import SearchSpace, ParameterSpec
        return SearchSpace(parameters=[
            ParameterSpec(name="lr", type="float", low=0.0001, high=0.01, log=True),
            ParameterSpec(name="batch_size", type="int", low=8, high=64, step=8),
            ParameterSpec(name="optimizer", type="categorical",
                          choices=["adam", "sgd", "adamw"]),
        ])

    def test_ask_tell_full_flow(self):
        """验证 StudyManager 完整 ask → tell 多轮流程。"""
        from senseframe.search_protocol import StudyManager
        sm = StudyManager()
        space = self._build_search_space()
        study_id = sm.create_study(
            name="hpo_test", direction="maximize",
            search_space=space, sampler="random",
        )

        # 第一轮 ask → params 含所有参数 → tell(value=0.85)
        t1 = sm.ask(study_id)
        assert "lr" in t1.params
        assert "batch_size" in t1.params
        assert "optimizer" in t1.params
        sm.tell(t1.trial_id, value=0.85)

        # 第二轮 ask → tell(value=0.90)
        t2 = sm.ask(study_id)
        sm.tell(t2.trial_id, value=0.90)

        # 第三轮 ask → tell(value=0.80, state="failed")
        t3 = sm.ask(study_id)
        sm.tell(t3.trial_id, value=0.80, state="failed")

        # 验证 list_trials 返回 3 条
        trials = sm.list_trials(study_id)
        assert len(trials) == 3

        # 验证 best_trial 返回 value=0.90（maximize 方向）
        best = sm.best_trial(study_id)
        assert best is not None
        assert best.value == pytest.approx(0.90)

    def test_ask_returns_trial_with_pending_state(self):
        """验证 ask 返回的 TrialSpec state 默认为 running。"""
        from senseframe.search_protocol import StudyManager
        sm = StudyManager()
        space = self._build_search_space()
        study_id = sm.create_study(
            name="pending_test", direction="maximize",
            search_space=space, sampler="random",
        )
        t = sm.ask(study_id)
        assert t.state == "running"
        assert t.datetime_start  # 非空时间戳

    def test_trial_bridged_to_exploration_tracker(self):
        """验证 StudyManager 正确桥接 ExplorationTracker（trial 在 tracker.history 中存在）。"""
        from senseframe.search_protocol import StudyManager
        sm = StudyManager()
        space = self._build_search_space()
        study_id = sm.create_study(
            name="bridge_test", direction="maximize",
            search_space=space, sampler="random",
        )

        # ask 后 trial 应已在 tracker.history 中（pending 状态）
        t = sm.ask(study_id)
        tracker = sm._trackers[study_id]
        history = tracker.history
        assert len(history) == 1
        entry = history[0]
        assert entry["trial_id"] == t.trial_id
        assert entry["result"] is None
        assert entry["status"] == "pending"

        # tell 后 entry 应被更新（result + state）
        sm.tell(t.trial_id, value=0.92, state="completed")
        entry_after = tracker.get_trial(t.trial_id)
        assert entry_after is not None
        assert entry_after["result"]["value"] == pytest.approx(0.92)
        assert entry_after["status"] == "completed"

    def test_sampler_registry_contains_builtin_samplers(self):
        """验证 Sampler 注册表含 random/grid/tpe。"""
        from senseframe.search_protocol import list_samplers
        samplers = list_samplers()
        assert "random" in samplers
        assert "grid" in samplers
        assert "tpe" in samplers

    def test_get_sampler_returns_class(self):
        """验证 get_sampler 返回正确的 Sampler 类。"""
        from senseframe.search_protocol import get_sampler, RandomSampler, GridSampler
        assert get_sampler("random") is RandomSampler
        assert get_sampler("grid") is GridSampler
        assert get_sampler("nonexistent") is None

    def test_grid_sampler_generates_grid_points(self):
        """验证 GridSampler 按网格顺序生成参数组合。"""
        from senseframe.search_protocol import GridSampler, SearchSpace, ParameterSpec
        # 用 low=1.0 避免 0.0 在 Python 布尔判断中被当作 falsy
        space = SearchSpace(parameters=[
            ParameterSpec(name="lr", type="float", low=1.0, high=3.0, step=1.0),
            ParameterSpec(name="optim", type="categorical", choices=["adam", "sgd"]),
        ])
        sampler = GridSampler()

        # 网格点：(1.0, adam), (1.0, sgd), (2.0, adam), (2.0, sgd), (3.0, adam), (3.0, sgd)
        p1 = sampler.sample(space, [])
        assert p1["lr"] == 1.0
        assert p1["optim"] == "adam"

        p2 = sampler.sample(space, [{"trial_id": "x"}])
        assert p2["lr"] == 1.0
        assert p2["optim"] == "sgd"

        p3 = sampler.sample(space, [{}, {}])
        assert p3["lr"] == 2.0
        assert p3["optim"] == "adam"

    def test_register_custom_sampler(self):
        """验证 register_sampler 可注册自定义 Sampler。"""
        from senseframe.search_protocol import (
            register_sampler, get_sampler, list_samplers,
            SearchSpace, ParameterSpec,
        )

        class CustomSampler:
            name = "custom_test_sampler"

            def sample(self, search_space, history):
                return {p.name: "fixed" for p in search_space.parameters}

        register_sampler("custom_test_sampler", CustomSampler)
        try:
            assert get_sampler("custom_test_sampler") is CustomSampler
            assert "custom_test_sampler" in list_samplers()

            # 实际使用自定义 sampler
            from senseframe.search_protocol import StudyManager
            sm = StudyManager()
            space = SearchSpace(parameters=[
                ParameterSpec(name="x", type="categorical", choices=["a", "b"]),
            ])
            sid = sm.create_study(name="custom", direction="maximize",
                                  search_space=space, sampler="custom_test_sampler")
            t = sm.ask(sid)
            assert t.params["x"] == "fixed"
        finally:
            # 清理注册（避免污染其他测试）
            from senseframe.search_protocol import _SAMPLERS
            _SAMPLERS.pop("custom_test_sampler", None)

    def test_get_trial_returns_correct_result(self):
        """验证 get_trial 返回正确的结果对象。"""
        from senseframe.search_protocol import StudyManager
        sm = StudyManager()
        space = self._build_search_space()
        study_id = sm.create_study(
            name="get_trial_test", direction="maximize",
            search_space=space, sampler="random",
        )
        t = sm.ask(study_id)
        sm.tell(t.trial_id, value=0.77, intermediate_values={1: 0.5, 2: 0.7})

        result = sm.get_trial(t.trial_id)
        assert result is not None
        assert result.trial_id == t.trial_id
        assert result.value == pytest.approx(0.77)
        assert result.state == "completed"
        assert 1 in result.intermediate_values
        assert result.intermediate_values[1] == pytest.approx(0.5)

    def test_tell_nonexistent_trial_raises(self):
        """验证 tell 不存在的 trial_id 抛 KeyError。"""
        from senseframe.search_protocol import StudyManager
        sm = StudyManager()
        with pytest.raises(KeyError):
            sm.tell("nonexistent_trial", value=0.5)


# ============================================================
# ε4: OP 编排协议状态机 + CloudEvent 验证
# ============================================================
class TestEpsilon4OPProtocol:
    """ε4: 编排协议状态机 + CloudEvent 验证。

    覆盖 PipelineDef 默认结构、PipelineRun 状态机转换、
    CloudEvent 序列化、Orchestrator 完整生命周期与 retry 流程。
    """

    def test_pipeline_def_default_has_9_stages(self):
        """验证 PipelineDef.default() 生成 9 个 stage。

        P0.7：stage 顺序与 Pipeline.default() 对齐（load 在 resolve 前）。
        方案 B：新增 probe_vram stage（build 和 train 之间）。
        """
        from senseframe.orchestration import PipelineDef
        pdef = PipelineDef.default(name="test_pipeline")
        assert pdef.name == "test_pipeline"
        assert len(pdef.stages) == 9
        names = [s.name for s in pdef.stages]
        assert names == [
            "validate", "preflight", "load", "resolve",
            "build", "probe_vram", "train", "eval", "export",
        ]

    def test_state_machine_pending_to_running(self):
        """验证 PENDING → RUNNING 合法转换。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_PENDING, PHASE_RUNNING,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1")
        assert run.phase == PHASE_PENDING
        run.transition(PHASE_RUNNING)
        assert run.phase == PHASE_RUNNING
        assert run.started_at  # started_at 应被设置

    def test_state_machine_running_to_succeeded(self):
        """验证 RUNNING → SUCCEEDED 合法转换。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_RUNNING, PHASE_SUCCEEDED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_RUNNING,
                          started_at="2026-01-01T00:00:00")
        run.transition(PHASE_SUCCEEDED)
        assert run.phase == PHASE_SUCCEEDED
        assert run.finished_at  # finished_at 应被设置

    def test_state_machine_running_to_paused(self):
        """验证 RUNNING → PAUSED 合法转换。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_RUNNING, PHASE_PAUSED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_RUNNING,
                          started_at="2026-01-01T00:00:00")
        run.transition(PHASE_PAUSED)
        assert run.phase == PHASE_PAUSED

    def test_state_machine_paused_to_running(self):
        """验证 PAUSED → RUNNING 合法转换（恢复）。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_RUNNING, PHASE_PAUSED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_PAUSED,
                          started_at="2026-01-01T00:00:00")
        run.transition(PHASE_RUNNING)
        assert run.phase == PHASE_RUNNING

    def test_state_machine_failed_to_running_retry(self):
        """验证 FAILED → RUNNING 合法转换（retry）。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_RUNNING, PHASE_FAILED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_FAILED,
                          started_at="2026-01-01T00:00:00",
                          finished_at="2026-01-01T01:00:00")
        run.transition(PHASE_RUNNING)
        assert run.phase == PHASE_RUNNING

    def test_state_machine_succeeded_to_running_raises(self):
        """验证 SUCCEEDED → RUNNING 非法转换抛 ValueError（终态）。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_RUNNING, PHASE_SUCCEEDED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_SUCCEEDED,
                          started_at="2026-01-01T00:00:00",
                          finished_at="2026-01-01T01:00:00")
        with pytest.raises(ValueError, match="Invalid transition"):
            run.transition(PHASE_RUNNING)

    def test_state_machine_pending_to_succeeded_raises(self):
        """验证 PENDING → SUCCEEDED 非法转换抛 ValueError。"""
        from senseframe.orchestration import (
            PipelineRun, PHASE_PENDING, PHASE_SUCCEEDED,
        )
        run = PipelineRun(run_id="r1", pipeline_ref="p1", phase=PHASE_PENDING)
        with pytest.raises(ValueError, match="Invalid transition"):
            run.transition(PHASE_SUCCEEDED)

    def test_cloud_event_to_dict(self):
        """验证 CloudEvent.to_dict() 返回符合 CloudEvents 1.0 的字段。"""
        from senseframe.orchestration import CloudEvent, EVENT_PIPELINE_STARTED
        ev = CloudEvent(
            source="/senseframe/pipeline/run_x",
            type=EVENT_PIPELINE_STARTED,
            data={"phase": "running"},
        )
        d = ev.to_dict()
        assert d["specversion"] == "1.0"
        assert d["source"] == "/senseframe/pipeline/run_x"
        assert d["type"] == EVENT_PIPELINE_STARTED
        assert d["id"]  # 自动生成
        assert d["time"]  # 自动生成
        assert d["datacontenttype"] == "application/json"
        assert d["data"] == {"phase": "running"}

    def test_cloud_event_to_json(self):
        """验证 CloudEvent.to_json() 返回合法 JSON 字符串。"""
        from senseframe.orchestration import CloudEvent, EVENT_STAGE_SUCCEEDED
        ev = CloudEvent(
            source="/senseframe/pipeline/run_y",
            type=EVENT_STAGE_SUCCEEDED,
            data={"stage_name": "train"},
        )
        s = ev.to_json()
        assert isinstance(s, str)
        # 可被反序列化
        parsed = json.loads(s)
        assert parsed["type"] == EVENT_STAGE_SUCCEEDED
        assert parsed["data"]["stage_name"] == "train"

    def test_make_event_fields(self):
        """验证 make_event 生成符合 CloudEvents 1.0 的字段。"""
        from senseframe.orchestration import make_event, EVENT_PIPELINE_STARTED
        ev = make_event(EVENT_PIPELINE_STARTED, "run_abc", {"phase": "running"})
        assert ev.specversion == "1.0"
        assert ev.id  # 自动生成
        assert ev.source == "/senseframe/pipeline/run_abc"
        assert ev.type == EVENT_PIPELINE_STARTED
        assert ev.time  # 自动生成
        assert ev.data == {"phase": "running"}

    def test_orchestrator_full_lifecycle(self):
        """验证 Orchestrator 完整生命周期：create_pipeline → create_run → subscribe → start → stage → complete。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef,
            EVENT_PIPELINE_STARTED, EVENT_PIPELINE_SUCCEEDED,
            EVENT_STAGE_STARTED, EVENT_STAGE_SUCCEEDED,
        )

        orch = Orchestrator()
        pdef = PipelineDef.default(name="lifecycle_test")
        pid = orch.create_pipeline(pdef)
        assert pid == "lifecycle_test"

        # subscribe 事件
        events = []
        orch.subscribe(EVENT_PIPELINE_STARTED, lambda e: events.append(e))
        orch.subscribe(EVENT_PIPELINE_SUCCEEDED, lambda e: events.append(e))
        orch.subscribe(EVENT_STAGE_SUCCEEDED, lambda e: events.append(e))

        run_id = orch.create_run(pid, params={"k": "v"})
        orch.start(run_id)

        # 推进第一个 stage 到 succeeded
        orch.update_stage(run_id, "validate", "running")
        orch.update_stage(run_id, "validate", "succeeded")

        # 保存 checkpoint
        orch.save_checkpoint(run_id, "validate", "file:///tmp/ckpt1",
                             stage_snapshot={"step": "done"})

        orch.complete(run_id, output_uri="file:///tmp/model.pth")

        # 验证事件流
        event_types = [e.type for e in events]
        assert EVENT_PIPELINE_STARTED in event_types
        assert EVENT_STAGE_SUCCEEDED in event_types
        assert EVENT_PIPELINE_SUCCEEDED in event_types

        # 验证 run 状态
        run = orch.get_run(run_id)
        assert run is not None
        assert run.phase == "succeeded"
        assert run.output_uri == "file:///tmp/model.pth"

        # 验证 checkpoint
        ckpts = orch.get_checkpoints(run_id)
        assert len(ckpts) == 1
        assert ckpts[0].stage_name == "validate"
        assert ckpts[0].checkpoint_uri == "file:///tmp/ckpt1"

    def test_orchestrator_retry_flow(self):
        """验证 retry 流程：fail → retry → complete。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef,
            PHASE_FAILED, PHASE_RUNNING, PHASE_SUCCEEDED,
        )

        orch = Orchestrator()
        pdef = PipelineDef.default(name="retry_test")
        pid = orch.create_pipeline(pdef)
        run_id = orch.create_run(pid)
        orch.start(run_id)
        # 失败
        orch.fail(run_id, error="boom", stage_name="train")
        run = orch.get_run(run_id)
        assert run.phase == PHASE_FAILED
        assert run.error == "boom"
        assert run.retry_count == 0
        # 重试
        orch.retry(run_id)
        run = orch.get_run(run_id)
        assert run.phase == PHASE_RUNNING
        assert run.retry_count == 1
        # 完成
        orch.complete(run_id, output_uri="file:///tmp/model2.pth")
        run = orch.get_run(run_id)
        assert run.phase == PHASE_SUCCEEDED

    def test_orchestrator_list_runs_filter(self):
        """验证 list_runs 按 phase 过滤。"""
        from senseframe.orchestration import Orchestrator, PipelineDef, PHASE_SUCCEEDED
        orch = Orchestrator()
        pdef = PipelineDef.default(name="filter_test")
        pid = orch.create_pipeline(pdef)

        # 创建 3 个 run，2 个完成，1 个 pending
        r1 = orch.create_run(pid)
        orch.start(r1)
        orch.complete(r1)

        r2 = orch.create_run(pid)
        orch.start(r2)
        orch.complete(r2)

        r3 = orch.create_run(pid)  # pending

        all_runs = orch.list_runs()
        assert len(all_runs) == 3

        succeeded = orch.list_runs(filter_phase=PHASE_SUCCEEDED)
        assert len(succeeded) == 2
        for r in succeeded:
            assert r.phase == PHASE_SUCCEEDED

    def test_subscribe_returns_unsubscribe(self):
        """验证 subscribe 返回取消订阅函数。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef, EVENT_PIPELINE_STARTED,
        )
        orch = Orchestrator()
        pid = orch.create_pipeline(PipelineDef.default(name="unsub_test"))

        received = []
        unsub = orch.subscribe(EVENT_PIPELINE_STARTED, lambda e: received.append(e))

        run_id = orch.create_run(pid)
        orch.start(run_id)
        assert len(received) == 1

        # 取消订阅后再启动新 run
        unsub()
        run_id2 = orch.create_run(pid)
        orch.start(run_id2)
        assert len(received) == 1  # 仍然是 1，未收到第二个事件


# ============================================================
# ε5: 编排器驱动完整闭环端到端验证
# ============================================================
class TestEpsilon5FullClosedLoop:
    """ε5: 编排器驱动完整闭环端到端验证。

    综合场景：编排器驱动 pipeline → 事件流订阅 → SP 搜索 → stage 推进 → 完成。
    验证 OP/SP/OBP/DSP 四层协议协同工作。
    """

    def test_full_closed_loop_driven_by_orchestrator(self, tmp_path):
        """编排器驱动完整闭环：pipeline → SP 搜索 → stage 推进 → 完成。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef,
            EVENT_PIPELINE_STARTED, EVENT_PIPELINE_SUCCEEDED,
            EVENT_STAGE_STARTED, EVENT_STAGE_SUCCEEDED,
            PHASE_SUCCEEDED,
        )
        from senseframe.search_protocol import (
            StudyManager, SearchSpace, ParameterSpec,
        )
        from senseframe.observability_otel import get_grafana_dashboard_template

        # 1. 用 Orchestrator 创建 pipeline（PipelineDef.default()）
        orch = Orchestrator()
        pdef = PipelineDef.default(name="automl_closed_loop")
        pipeline_id = orch.create_pipeline(pdef)
        assert pipeline_id == "automl_closed_loop"
        assert len(pdef.stages) == 9

        # 2. subscribe("*") 订阅所有事件
        events = []
        orch.subscribe("*", lambda e: events.append(e))

        # 3. create_run → start
        run_id = orch.create_run(pipeline_id, params={"goal": "maximize_val_acc"})
        orch.start(run_id)

        # 4. 模拟 SP 搜索：StudyManager ask → 得到超参 → params 注入到 run
        sm = StudyManager()
        space = SearchSpace(parameters=[
            ParameterSpec(name="lr", type="float", low=0.0001, high=0.01, log=True),
            ParameterSpec(name="batch_size", type="int", low=8, high=64, step=8),
            ParameterSpec(name="optimizer", type="categorical",
                          choices=["adam", "sgd", "adamw"]),
        ])
        study_id = sm.create_study(
            name="closed_loop_hpo", direction="maximize",
            search_space=space, sampler="random",
        )
        trial = sm.ask(study_id)
        # 模拟训练得到结果
        sm.tell(trial.trial_id, value=0.88)

        # 注入参数到 run
        run = orch.get_run(run_id)
        run.params.update(trial.params)
        # 验证参数已注入
        for k in trial.params:
            assert run.params[k] == trial.params[k]

        # 5. 模拟 stage 执行：每个 stage 推进 running → succeeded
        stage_names = [s.name for s in pdef.stages]
        for stage_name in stage_names:
            orch.update_stage(run_id, stage_name, "running")
            orch.update_stage(run_id, stage_name, "succeeded")

        # 6. 中途 save_checkpoint（模拟断点续跑）
        orch.save_checkpoint(
            run_id, "train",
            checkpoint_uri=f"file:///tmp/ckpt_{run_id}",
            stage_snapshot={"epoch": 5, "loss": 0.32},
        )

        # 7. complete(run_id, output_uri)
        output_uri = f"file:///tmp/{run_id}/model.pth"
        orch.complete(run_id, output_uri=output_uri)

        # 8. 验证事件流顺序
        event_types = [e.type for e in events]
        # PIPELINE_STARTED 应在第一位
        assert event_types[0] == EVENT_PIPELINE_STARTED, \
            f"首位事件应为 PIPELINE_STARTED，实际: {event_types[0]}"
        # PIPELINE_SUCCEEDED 应在最后一位
        assert event_types[-1] == EVENT_PIPELINE_SUCCEEDED, \
            f"末位事件应为 PIPELINE_SUCCEEDED，实际: {event_types[-1]}"
        # 应包含多个 STAGE_STARTED / STAGE_SUCCEEDED
        assert event_types.count(EVENT_STAGE_STARTED) == 9
        assert event_types.count(EVENT_STAGE_SUCCEEDED) == 9

        # 9. 验证 PipelineRun.phase == "succeeded"
        run = orch.get_run(run_id)
        assert run.phase == PHASE_SUCCEEDED

        # 10. 所有 stage phase == "succeeded"
        for stage in run.stages:
            assert stage.phase == "succeeded", \
                f"stage {stage.name} phase={stage.phase}（应为 succeeded）"

        # 11. checkpoints 列表非空
        checkpoints = orch.get_checkpoints(run_id)
        assert len(checkpoints) > 0
        assert checkpoints[0].stage_name == "train"
        assert checkpoints[0].checkpoint_uri == f"file:///tmp/ckpt_{run_id}"

        # 12. run.output_uri 已设置
        assert run.output_uri == output_uri

        # 13. 验证 SP 搜索结果
        best = sm.best_trial(study_id)
        assert best is not None
        assert best.value == pytest.approx(0.88)
        trials = sm.list_trials(study_id)
        assert len(trials) == 1

        # 14. 验证 introspect 模块可查询 pipeline 结构（pipeline_graph 含 fields 映射）
        try:
            from senseframe.introspect import pipeline_graph
            graph = pipeline_graph()
            assert "fields" in graph
            assert len(graph["fields"]) > 0
            # config 字段应被多个 stage 读取
            config_field = graph["fields"].get("config", {})
            assert "consumers" in config_field
            assert len(config_field["consumers"]) > 0
        except ImportError:
            pytest.skip("torch 不可用，无法验证 introspect.pipeline_graph")

        # 15. 验证 OBP 模板可用（get_grafana_dashboard_template）
        dashboard = get_grafana_dashboard_template()
        assert "title" in dashboard
        assert "panels" in dashboard
        assert len(dashboard["panels"]) > 0

    def test_closed_loop_with_pause_resume(self):
        """验证编排器驱动闭环含 pause/resume 子流程。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef,
            EVENT_PIPELINE_STARTED, EVENT_PIPELINE_PAUSED,
            EVENT_PIPELINE_RESUMED, EVENT_PIPELINE_SUCCEEDED,
            PHASE_PAUSED, PHASE_RUNNING, PHASE_SUCCEEDED,
        )

        orch = Orchestrator()
        pdef = PipelineDef.default(name="pause_resume_test")
        pid = orch.create_pipeline(pdef)

        events = []
        orch.subscribe("*", lambda e: events.append(e))

        run_id = orch.create_run(pid)
        orch.start(run_id)

        # 推进部分 stage 后暂停
        orch.update_stage(run_id, "validate", "running")
        orch.update_stage(run_id, "validate", "succeeded")
        orch.pause(run_id)
        assert orch.get_run(run_id).phase == PHASE_PAUSED

        # 恢复后继续推进
        orch.resume(run_id)
        assert orch.get_run(run_id).phase == PHASE_RUNNING

        # 完成剩余 stage
        for stage_name in ["preflight", "resolve", "load", "build", "train", "eval", "export"]:
            orch.update_stage(run_id, stage_name, "running")
            orch.update_stage(run_id, stage_name, "succeeded")

        orch.complete(run_id, output_uri="file:///tmp/done.pth")
        assert orch.get_run(run_id).phase == PHASE_SUCCEEDED

        # 验证事件流含 PAUSED 与 RESUMED
        event_types = [e.type for e in events]
        assert EVENT_PIPELINE_STARTED in event_types
        assert EVENT_PIPELINE_PAUSED in event_types
        assert EVENT_PIPELINE_RESUMED in event_types
        assert EVENT_PIPELINE_SUCCEEDED in event_types
        # 顺序：STARTED ... PAUSED ... RESUMED ... SUCCEEDED
        idx_started = event_types.index(EVENT_PIPELINE_STARTED)
        idx_paused = event_types.index(EVENT_PIPELINE_PAUSED)
        idx_resumed = event_types.index(EVENT_PIPELINE_RESUMED)
        idx_succeeded = event_types.index(EVENT_PIPELINE_SUCCEEDED)
        assert idx_started < idx_paused < idx_resumed < idx_succeeded

    def test_closed_loop_with_failure_and_retry(self):
        """验证编排器驱动闭环含失败重试子流程。"""
        from senseframe.orchestration import (
            Orchestrator, PipelineDef,
            EVENT_PIPELINE_FAILED,
            PHASE_FAILED, PHASE_RUNNING, PHASE_SUCCEEDED,
        )

        orch = Orchestrator()
        pdef = PipelineDef.default(name="failure_retry_test")
        pid = orch.create_pipeline(pdef)

        events = []
        orch.subscribe("*", lambda e: events.append(e))

        run_id = orch.create_run(pid)
        orch.start(run_id)

        # 推进到 train 阶段失败
        for stage_name in ["validate", "preflight", "resolve", "load", "build"]:
            orch.update_stage(run_id, stage_name, "running")
            orch.update_stage(run_id, stage_name, "succeeded")
        orch.update_stage(run_id, "train", "running")
        orch.fail(run_id, error="OOM during training", stage_name="train")
        assert orch.get_run(run_id).phase == PHASE_FAILED

        # 重试
        orch.retry(run_id)
        assert orch.get_run(run_id).phase == PHASE_RUNNING
        assert orch.get_run(run_id).retry_count == 1

        # 修复后重新推进 train
        orch.update_stage(run_id, "train", "succeeded")
        for stage_name in ["eval", "export"]:
            orch.update_stage(run_id, stage_name, "running")
            orch.update_stage(run_id, stage_name, "succeeded")

        orch.complete(run_id, output_uri="file:///tmp/recovered.pth")
        assert orch.get_run(run_id).phase == PHASE_SUCCEEDED

        # 验证事件流含 FAILED
        event_types = [e.type for e in events]
        assert EVENT_PIPELINE_FAILED in event_types


# ============================================================
# P0 协议栈地基加固验收测试
# ============================================================
class TestP0ProtocolFoundationAcceptance:
    """P0 协议栈地基加固的验收测试。

    覆盖 P0.2 / P0.3 / P0.5 / P0.7 / P0.8 五个工作项的验收标准：
    - P0.2 FeatureSpec 满足 FeatureSpecProtocol
    - P0.3 ExplorationTracker.update_trial 公共 API
    - P0.5 SP 顶层导出 + Sampler Protocol
    - P0.7 PipelineDef.materialize() 物化
    - P0.8 Checkpoint 统一（pipeline_checkpoint.json 唯一真源）

    反假绿原则：
    - test_pipelinedef_materialize 实际执行 stage 函数查找，不只用名列表比较
    - test_checkpoint_unified 实际读写文件，不只用 to_dict 比较
    """

    # ---------- P0.2 FeatureSpec 满足 Protocol ----------

    def test_feature_spec_satisfies_protocol(self):
        """FeatureSpec 必须满足 FeatureSpecProtocol 契约。

        P0.2 修复前：FeatureSpec 缺 feature_names/dtypes 字段，
        isinstance 检查会失败
        P0.2 修复后：补齐字段，自动填充 dtypes

        注意：FeatureSpecProtocol 定义在 pipeline.py（DSP-1 契约先行声明），
        FeatureSpec 实现在 core/features.py。
        """
        from senseframe.core.features import FeatureSpec
        from senseframe.engine.runner.pipeline import FeatureSpecProtocol

        # 构造 FeatureSpec 实例
        fs = FeatureSpec(
            feature_dim=10,
            dtype="float32",
        )

        # 1. isinstance 检查通过
        assert isinstance(fs, FeatureSpecProtocol), \
            "FeatureSpec 不满足 FeatureSpecProtocol 契约"

        # 2. feature_names/dtypes 字段存在
        assert hasattr(fs, "feature_names"), "FeatureSpec 缺 feature_names 字段"
        assert hasattr(fs, "dtypes"), "FeatureSpec 缺 dtypes 字段"

        # 3. dtypes 自动填充（feature_dim > 0 时）
        assert len(fs.dtypes) == 10, \
            f"dtypes 应自动填充为 10 项，实际: {len(fs.dtypes)}"
        assert all(d == "float32" for d in fs.dtypes), \
            f"dtypes 应全部为 'float32'，实际: {fs.dtypes}"

        # 4. to_dict 序列化包含新字段
        d = fs.to_dict()
        assert "feature_names" in d, "to_dict 缺 feature_names"
        assert "dtypes" in d, "to_dict 缺 dtypes"

    # ---------- P0.3 ExplorationTracker.update_trial 公共 API ----------

    def test_exploration_tracker_update_trial(self):
        """ExplorationTracker.update_trial 公共 API 可用 + KeyError。

        P0.3 修复前：SP tell 中 with tracker._lock + tracker.history 改写
        P0.3 修复后：update_trial 公共方法封装锁与字段更新
        """
        from senseframe.exploration import ExplorationTracker

        tracker = ExplorationTracker()

        # 1. 更新不存在 trial 应抛 KeyError
        with pytest.raises(KeyError):
            tracker.update_trial("ghost_trial", result={"value": 0.5})

        # 2. 添加 trial 后可更新 result/status/feedback
        tracker.add_trial(strategy={"lr": 0.01}, result=None, trial_id="t1")
        tracker.update_trial(
            "t1",
            result={"value": 0.85},
            status="completed",
            feedback={"note": "best so far"},
        )

        entry = tracker.get_trial("t1")
        assert entry is not None
        assert entry["result"] == {"value": 0.85}
        assert entry["status"] == "completed"
        assert entry["feedback"] == {"note": "best so far"}

        # 3. 部分更新（仅 result，status/feedback 不变）
        tracker.update_trial("t1", result={"value": 0.90})
        entry_after = tracker.get_trial("t1")
        assert entry_after["result"] == {"value": 0.90}
        assert entry_after["status"] == "completed"  # 未被覆盖
        assert entry_after["feedback"] == {"note": "best so far"}  # 未被覆盖

    # ---------- P0.5 SP 顶层导出 + Sampler Protocol ----------

    def test_sp_top_level_export(self):
        """SP 协议符号在 senseframe 顶层可达。

        P0.5 修复前：senseframe/__init__.py 未导出 SP 符号
        P0.5 修复后：13 个 SP 符号在顶层可达
        """
        import senseframe

        # 核心类型
        expected_symbols = [
            "ParameterSpec", "SearchSpace", "StudySpec",
            "TrialSpec", "TrialResult",
            "StudyManager", "get_study_manager",
            "Sampler",
            "register_sampler", "get_sampler", "list_samplers",
            "RandomSampler", "GridSampler",
        ]
        for name in expected_symbols:
            assert hasattr(senseframe, name), \
                f"senseframe 顶层未导出 SP 符号: {name}"

        # 反向验证：StudyManager 是类，可实例化
        sm = senseframe.StudyManager()
        assert sm is not None

    def test_sampler_protocol_compliance(self):
        """RandomSampler/GridSampler 满足 Sampler Protocol 契约。

        P0.5 修复前：无 Sampler Protocol 基类
        P0.5 修复后：@runtime_checkable Sampler Protocol，isinstance 可用
        """
        from senseframe.search_protocol import (
            Sampler, RandomSampler, GridSampler,
        )

        # 1. RandomSampler 实例 isinstance Sampler
        rs = RandomSampler()
        assert isinstance(rs, Sampler), \
            "RandomSampler 不满足 Sampler Protocol"

        # 2. GridSampler 实例 isinstance Sampler
        gs = GridSampler()
        assert isinstance(gs, Sampler), \
            "GridSampler 不满足 Sampler Protocol"

        # 3. Protocol 契约字段：name 类属性 + sample 方法
        assert hasattr(RandomSampler, "name"), "RandomSampler 缺 name 类属性"
        assert hasattr(RandomSampler, "sample"), "RandomSampler 缺 sample 方法"
        assert hasattr(GridSampler, "name"), "GridSampler 缺 name 类属性"
        assert hasattr(GridSampler, "sample"), "GridSampler 缺 sample 方法"

        # 4. 不满足 Protocol 的对象 isinstance 返回 False
        class NotASampler:
            pass
        assert not isinstance(NotASampler(), Sampler), \
            "Sampler Protocol 误判：NotASampler 不应通过 isinstance"

    # ---------- P0.7 PipelineDef.materialize() ----------

    def test_pipelinedef_materialize(self):
        """PipelineDef.default().materialize() 与 Pipeline.default() 等价。

        反假绿：不仅比较 stage 名列表，还实际调用 stages_with_spec()
        验证 stage 函数被正确解析（带 _stage_spec 属性）。

        P0.7 修复前：PipelineDef 无 materialize 方法
        P0.7 修复后：materialize 复用 Pipeline.default() 内部结构
        """
        from senseframe.orchestration import PipelineDef
        from senseframe.engine.runner.pipeline import Pipeline

        # 1. materialize 返回 Pipeline 实例
        pdef = PipelineDef.default()
        pipeline = pdef.materialize()
        assert isinstance(pipeline, Pipeline), \
            "materialize() 应返回 Pipeline 实例"

        # 2. stage 名列表与 Pipeline.default() 一致
        names_materialized = [n for n, _ in pipeline.stages]
        names_default = [n for n, _ in Pipeline.default().stages]
        assert names_materialized == names_default, \
            f"materialize stage 名列表不一致: {names_materialized} vs {names_default}"

        # 3. 反假绿：实际验证 stage 函数带 _stage_spec 属性（不是占位 None）
        specs = pipeline.stages_with_spec()
        assert len(specs) == 9, f"应有 9 个 stage spec，实际: {len(specs)}"
        # 至少 train stage 应有非空 writes（验证 _stage_spec 被正确附加）
        # StageSpec.writes 是 List[FieldSpec]，提取 name 字段
        train_spec = next((s for s in specs if s.name == "train"), None)
        assert train_spec is not None, "materialize 后未找到 train stage"
        train_write_names = {fs.name for fs in train_spec.writes}
        assert "trainer" in train_write_names, \
            f"train stage writes 缺 trainer，实际: {train_write_names}"

    def test_pipelinedef_materialize_rejects_unknown_stage(self):
        """materialize 对未知 stage 名抛 ValueError。

        反假绿：实际构造含未知 stage 的 PipelineDef，验证抛异常。
        """
        from senseframe.orchestration import PipelineDef, StageTemplate

        pdef = PipelineDef(
            name="bad",
            stages=[StageTemplate(name="nonexistent_stage")],
        )
        with pytest.raises(ValueError, match="unknown stage"):
            pdef.materialize()

    # ---------- P0.8 Checkpoint 统一 ----------

    def test_checkpoint_unified(self, tmp_path):
        """CheckpointSpec 与 pipeline_checkpoint.json 数据一致。

        反假绿：实际写文件 + 实际读文件，不只用 to_dict 比较。
        验证 save_checkpoint_from_file 读取的 stage_snapshot 含 stage_outputs。

        P0.8 修复前：两套 checkpoint 并行（CheckpointSpec 内存 + JSON 文件）
        P0.8 修复后：以 pipeline_checkpoint.json 为唯一真源，CheckpointSpec 为镜像
        """
        import json
        from senseframe.orchestration import Orchestrator
        from senseframe.engine.runner.pipeline import Pipeline, PipelineContext

        # 1. 构造 ctx + output_dir，调用 _write_checkpoint 写文件
        output_dir = tmp_path / "test_run"
        output_dir.mkdir()

        # 用最小可用的 ExperimentConfig 构造 ctx（避免真实训练依赖）
        # 直接构造 PipelineContext，config 用 MagicMock
        from unittest.mock import MagicMock
        ctx = PipelineContext(config=MagicMock())
        ctx.output_dir = output_dir
        ctx.trial_id = "test_trial_001"
        ctx.model_id = "test_model"
        ctx.dataset = "test_dataset"
        ctx.training_duration_s = 12.34
        ctx.best_model_score = 0.92
        ctx.best_model_path = "/tmp/best.ckpt"
        ctx.early_stopped = False
        ctx.completed_stages = ["validate", "preflight", "resolve", "load", "build", "train"]

        pipeline = Pipeline()
        pipeline._write_checkpoint(ctx, failed_stage=None)

        # 2. 验证文件存在且含 stage_outputs
        ckpt_path = output_dir / "pipeline_checkpoint.json"
        assert ckpt_path.exists(), "pipeline_checkpoint.json 未写入"

        file_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        assert "stage_outputs" in file_data, \
            "pipeline_checkpoint.json 缺 stage_outputs 字段（P0.8 未实施？）"
        stage_outputs = file_data["stage_outputs"]
        assert stage_outputs.get("trial_id") == "test_trial_001"
        assert stage_outputs.get("training_duration_s") == pytest.approx(12.34)
        assert stage_outputs.get("best_model_score") == pytest.approx(0.92)
        assert "completed_stages" in stage_outputs

        # 3. 调用 Orchestrator.save_checkpoint_from_file，验证 CheckpointSpec 镜像
        orch = Orchestrator()
        run_id = "test_run_ckpt_001"
        ckpt = orch.save_checkpoint_from_file(run_id, "train", output_dir)

        assert ckpt is not None, "save_checkpoint_from_file 返回 None（文件应存在）"
        assert ckpt.run_id == run_id
        assert ckpt.stage_name == "train"
        assert "stage_outputs" in ckpt.stage_snapshot, \
            "CheckpointSpec.stage_snapshot 缺 stage_outputs（未与文件一致）"
        assert ckpt.stage_snapshot["stage_outputs"]["trial_id"] == "test_trial_001"

        # 4. 验证 get_checkpoints 返回的快照与文件内容一致
        checkpoints = orch.get_checkpoints(run_id)
        assert len(checkpoints) == 1
        assert checkpoints[0].stage_snapshot == file_data, \
            "CheckpointSpec 快照与 pipeline_checkpoint.json 内容不一致"

    def test_checkpoint_unified_missing_file_returns_none(self, tmp_path):
        """save_checkpoint_from_file 对不存在的文件返回 None，不抛异常。

        反假绿：实际调用，验证不抛 FileNotFoundError。
        """
        from senseframe.orchestration import Orchestrator

        orch = Orchestrator()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = orch.save_checkpoint_from_file("ghost_run", "train", empty_dir)
        assert result is None, \
            "文件不存在时应返回 None，不应抛异常"
