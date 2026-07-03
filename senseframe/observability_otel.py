"""RFC-003 OBP：基于 OpenTelemetry 的可观测性协议。

用 OTel 作为可观测性地基，SenseFrame 指标可被 Prometheus/Grafana 消费。
OTel 为可选依赖，未安装时降级为 no-op。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# OBP-1: Resource 属性常量
RESOURCE_SERVICE_NAME = "senseframe"
RESOURCE_SERVICE_VERSION = ""  # 运行时填充

# OBP-2: ML Semantic Conventions 常量
ML_FRAMEWORK = "senseframe"
ML_STAGE = "ml.stage"          # train / eval / export
ML_EPOCH = "ml.epoch"
ML_PIPELINE_RUN_ID = "ml.pipeline_run_id"
ML_TRIAL_ID = "ml.trial_id"
ML_MODEL_ID = "ml.model_id"
ML_DATASET = "ml.dataset"

# Metric 名称常量
ML_TRAIN_LOSS = "ml.train.loss"
ML_VAL_LOSS = "ml.val.loss"
ML_VAL_ACCURACY = "ml.val.accuracy"
ML_LEARNING_RATE = "ml.learning_rate"

ML_INFERENCE_REQUEST_COUNT = "ml.inference.request_count"
ML_INFERENCE_LATENCY_MS = "ml.inference.latency_ms"
ML_INFERENCE_CONFIDENCE = "ml.inference.confidence"

ML_TRIAL_COUNT = "ml.trial.count"
ML_TRIAL_BEST_METRIC = "ml.trial.best_metric"
ML_SEARCH_COVERAGE_RATIO = "ml.search.coverage_ratio"

_otel_initialized = False
_meter = None
_resource = None
_memory_reader = None  # P1.1: InMemoryMetricReader 引用，供 /metrics 端点读取


def init_otel(
    service_name: str = "senseframe",
    service_version: str = "",
    pipeline_run_id: str = "",
    trial_id: str = "",
    model_id: str = "",
    dataset: str = "",
    enable_prometheus: bool = True,
    prometheus_port: int = 9464,
    enable_otlp: bool = False,
    otlp_endpoint: str = "http://localhost:4317",
) -> bool:
    """初始化 OpenTelemetry（OBP-1~3）。

    OTel 为可选依赖，未安装时返回 False，不抛异常。

    RFC-005：幂等保护——已初始化时直接返回 True，避免重复创建 MeterProvider
    导致后台导出线程 + Prometheus HTTP server 线程累积泄露。

    Returns:
        True 表示初始化成功，False 表示 OTel 未安装（降级为 no-op）
    """
    global _otel_initialized, _meter, _resource
    # RFC-005：幂等早退，避免重复初始化泄露后台线程
    if _otel_initialized:
        return True
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    except ImportError:
        return False

    # OBP-1: Resource
    attributes = {
        "service.name": service_name,
        "service.version": service_version,
        "service.namespace": "senseframe",  # P2.2: 补全 OBP-1 service.namespace
        "ml.framework": ML_FRAMEWORK,
    }
    if pipeline_run_id:
        attributes[ML_PIPELINE_RUN_ID] = pipeline_run_id
    if trial_id:
        attributes[ML_TRIAL_ID] = trial_id
    if model_id:
        attributes[ML_MODEL_ID] = model_id
    if dataset:
        attributes[ML_DATASET] = dataset

    _resource = Resource.create(attributes)

    # OBP-3: Exporter
    readers = []
    if enable_prometheus:
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            # P1.1: prometheus_port 传入 PrometheusMetricReader（修复死代码）
            readers.append(PrometheusMetricReader(port=prometheus_port))
        except ImportError:
            pass

    if enable_otlp:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            otlp_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otlp_endpoint),
                export_interval_millis=5000,
            )
            readers.append(otlp_reader)
        except ImportError:
            pass

    # Fallback: InMemoryMetricReader if no readers
    if not readers:
        _mem_reader = InMemoryMetricReader()
        readers.append(_mem_reader)
    else:
        # P1.1: 始终附加 InMemoryMetricReader 供 /metrics 端点读取
        _mem_reader = InMemoryMetricReader()
        readers.append(_mem_reader)

    provider = MeterProvider(resource=_resource, metric_readers=readers)
    otel_metrics.set_meter_provider(provider)
    _meter = otel_metrics.get_meter(service_name)
    _otel_initialized = True
    global _memory_reader
    _memory_reader = _mem_reader
    return True


def get_meter():
    """获取 OTel Meter（OBP-2）。

    未初始化时返回 None。
    """
    return _meter


def is_otel_enabled() -> bool:
    """检查 OTel 是否可用。"""
    return _otel_initialized


def shutdown_otel() -> None:
    """关闭 OTel MeterProvider（RFC-005 资源泄露修复）。

    终止后台导出线程 + Prometheus HTTP server 线程，释放端口。
    幂等：未初始化时安全调用。

    应在进程退出前（atexit）或 serving.py 的 finally 块中调用。
    """
    global _otel_initialized, _meter, _resource, _memory_reader
    if not _otel_initialized:
        return
    # 调用 MeterProvider.shutdown() 终止所有 MetricReader 的后台线程
    try:
        if _meter is not None:
            # _meter 是 opentelemetry.metrics.Meter，其 provider 可通过全局获取
            from opentelemetry import metrics as otel_metrics
            provider = otel_metrics.get_meter_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
    except Exception:
        pass
    _meter = None
    _resource = None
    _memory_reader = None
    _otel_initialized = False


def get_metrics_snapshot() -> Dict[str, Any]:
    """获取 OTel InMemoryMetricReader 指标快照（P1.1: /metrics 端点用）。

    Returns:
        {"status": "ok", "metrics": [...]} 或 {"status": "not_initialized"}
    """
    if not _otel_initialized or _memory_reader is None:
        return {"status": "not_initialized", "metrics": []}
    try:
        # OTel SDK get_metrics_data API
        metrics_data = _memory_reader.get_metrics_data()
        resource_metrics = metrics_data.resource_metrics
        result = []
        for rm in resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    result.append({
                        "name": m.name,
                        "description": m.description,
                        "unit": m.unit,
                        "data": {
                            "data_points": [
                                {
                                    "attributes": dict(dp.attributes) if hasattr(dp, "attributes") else {},
                                    "value": dp.value if hasattr(dp, "value") else None,
                                }
                                for dp in (m.data.data_points if hasattr(m.data, "data_points") else [])
                            ],
                        },
                    })
        return {"status": "ok", "metrics": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "metrics": []}


def record_training_metric(
    metric_name: str,
    value: float,
    stage: str = "train",
    epoch: Optional[int] = None,
    model_id: str = "",
    dataset: str = "",
):
    """记录训练指标（OBP-2）。OTel 未安装时 no-op。"""
    if not _otel_initialized or _meter is None:
        return
    try:
        attributes = {ML_STAGE: stage}
        if epoch is not None:
            attributes[ML_EPOCH] = epoch
        if model_id:
            attributes[ML_MODEL_ID] = model_id
        if dataset:
            attributes[ML_DATASET] = dataset

        # 使用 ObservableGauge（每次记录最新值）
        gauge = _meter.create_gauge(
            name=metric_name,
            description=f"Training metric: {metric_name}",
            unit="1",
        )
        gauge.set(value, attributes=attributes)
    except Exception:
        pass  # OTel 失败不影响主流程


def record_inference_metric(
    metric_name: str,
    value: float = 1.0,
    latency_ms: Optional[float] = None,
    confidence: Optional[float] = None,
    model_id: str = "",
):
    """记录推理指标（OBP-2）。OTel 未安装时 no-op。"""
    if not _otel_initialized or _meter is None:
        return
    try:
        attributes = {}
        if model_id:
            attributes[ML_MODEL_ID] = model_id

        if metric_name == ML_INFERENCE_REQUEST_COUNT:
            counter = _meter.create_counter(
                name=metric_name,
                description="Inference request count",
                unit="1",
            )
            counter.add(value, attributes=attributes)
        elif metric_name == ML_INFERENCE_LATENCY_MS and latency_ms is not None:
            histogram = _meter.create_histogram(
                name=metric_name,
                description="Inference latency",
                unit="ms",
            )
            histogram.record(latency_ms, attributes=attributes)
        elif metric_name == ML_INFERENCE_CONFIDENCE and confidence is not None:
            gauge = _meter.create_gauge(
                name=metric_name,
                description="Inference confidence",
                unit="1",
            )
            gauge.set(confidence, attributes=attributes)
    except Exception:
        pass


def record_trial_metric(
    metric_name: str,
    value: float,
    trial_id: str = "",
):
    """记录探索指标（OBP-2）。OTel 未安装时 no-op。"""
    if not _otel_initialized or _meter is None:
        return
    try:
        attributes = {}
        if trial_id:
            attributes[ML_TRIAL_ID] = trial_id

        gauge = _meter.create_gauge(
            name=metric_name,
            description=f"Trial metric: {metric_name}",
            unit="1",
        )
        gauge.set(value, attributes=attributes)
    except Exception:
        pass


def get_grafana_dashboard_template() -> dict:
    """返回 SenseFrame 标准 Grafana dashboard JSON 模板（OBP-4）。

    Agent 可基于此模板生成 dashboard，导入 Grafana 后即可可视化。
    """
    return {
        "title": "SenseFrame ML Dashboard",
        "schemaVersion": 38,
        "panels": [
            {
                "title": "Training Loss",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": "ml_train_loss", "legendFormat": "train loss"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            {
                "title": "Validation Accuracy",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": "ml_val_accuracy", "legendFormat": "val acc"}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            {
                "title": "Inference Latency",
                "type": "timeseries",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": "ml_inference_latency_ms_bucket", "legendFormat": "latency"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            },
            {
                "title": "Trial Count",
                "type": "stat",
                "datasource": {"type": "prometheus"},
                "targets": [{"expr": "ml_trial_count", "legendFormat": "trials"}],
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8},
            },
        ],
    }


def get_vegalite_templates() -> Dict[str, Any]:
    """返回 Vega-Lite 可视化模板（P2.3: OBP-4 渲染层）。

    返回 4 个 Vega-Lite spec dict，覆盖训练损失、验证精度、推理延迟、试验数。
    Grafana 8+ 原生支持 Vega-Lite 面板。

    Returns:
        {"training_loss": {...}, "val_accuracy": {...}, "inference_latency": {...}, "trial_count": {...}}
    """
    return {
        "training_loss": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Training Loss",
            "mark": "line",
            "encoding": {
                "x": {"field": "epoch", "type": "quantitative", "title": "Epoch"},
                "y": {"field": "ml.train.loss", "type": "quantitative", "title": "Loss"},
                "color": {"field": "model_id", "type": "nominal"},
            },
        },
        "val_accuracy": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Validation Accuracy",
            "mark": "line",
            "encoding": {
                "x": {"field": "epoch", "type": "quantitative", "title": "Epoch"},
                "y": {"field": "ml.val.accuracy", "type": "quantitative", "title": "Accuracy"},
                "color": {"field": "model_id", "type": "nominal"},
            },
        },
        "inference_latency": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Inference Latency",
            "mark": "bar",
            "encoding": {
                "x": {"field": "timestamp", "type": "temporal", "title": "Time"},
                "y": {"field": "ml.inference.latency_ms", "type": "quantitative", "title": "Latency (ms)"},
            },
        },
        "trial_count": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Trial Count",
            "mark": "text",
            "encoding": {
                "text": {"field": "ml.trial.count", "type": "quantitative"},
            },
        },
    }
