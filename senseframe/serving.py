"""P3：推理服务 — 将训练完的模型部署为 HTTP 推理 API。

支持加载导出的模型（state_dict / ONNX），提供 /predict /predict/batch /health /info 端点。
FastAPI 为可选依赖，未安装时 import 报错并提示安装。

Usage:
    from senseframe.serving import InferenceServer
    server = InferenceServer(output_dir="experiments/model_dataset_ts_pid/")
    server.start(host="0.0.0.0", port=8000)

    # 或通过 CLI:
    # senseframe serve experiments/model_dataset_ts_pid/ --port 8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class InferenceServer:
    """推理服务（P3）。

    加载导出的模型，提供 HTTP 推理 API。
    """

    def __init__(self, output_dir: Union[str, Path], device: str = "cpu"):
        """初始化推理服务。

        Args:
            output_dir: 训练输出目录（含 model.pth + metadata.json）
            device: 推理设备（cpu/cuda）
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.model = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.config = None
        self._model_type: str = "unknown"

        # 加载元数据
        meta_path = self.output_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        # 加载模型
        self._load_model()

    def _load_model(self) -> None:
        """加载模型（优先 ONNX，其次 state_dict）。"""
        # 优先 ONNX
        onnx_path = self.output_dir / "model.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(str(onnx_path))
                self._model_type = "onnx"
                return
            except ImportError:
                pass  # 回退到 state_dict

        # state_dict
        pth_path = self.output_dir / "model.pth"
        if pth_path.exists():
            try:
                import torch
                # 需要 metadata 中的模型架构信息来重建模型
                # 简化：只加载 state_dict，实际推理需要重建模型实例
                self.model = torch.load(pth_path, map_location=self.device, weights_only=False)
                self._model_type = "state_dict"
                return
            except ImportError:
                raise ImportError(
                    "Neither onnxruntime nor torch is available to load the model. "
                    "Install onnxruntime for ONNX inference or torch for state_dict inference."
                )

        raise FileNotFoundError(
            f"No model file found in {self.output_dir}. "
            f"Expected model.onnx or model.pth."
        )

    def predict(self, input_data: Any) -> Any:
        """单样本推理。"""
        if self._model_type == "onnx":
            return self._predict_onnx(input_data)
        elif self._model_type == "state_dict":
            return self._predict_torch(input_data)
        raise RuntimeError(f"Unknown model type: {self._model_type}")

    def predict_batch(self, input_batch: List[Any]) -> List[Any]:
        """批量推理。"""
        return [self.predict(x) for x in input_batch]

    def _predict_onnx(self, input_data: Any) -> Any:
        """ONNX 推理。"""
        import numpy as np
        # 简化：假设 input_data 是 numpy 数组或可转换
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        # 单样本 → 添加 batch 维
        if len(input_data.shape) == 1:
            input_data = input_data[np.newaxis, ...]

        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: input_data})
        # 返回预测结果（argmax 或原始 logits）
        result = outputs[0]
        if len(result.shape) > 1 and result.shape[-1] > 1:
            # 分类任务：返回 argmax + probabilities
            return {
                "prediction": int(np.argmax(result, axis=-1)[0]),
                "probabilities": result.tolist()[0],
            }
        return result.tolist()

    def _predict_torch(self, input_data: Any) -> Any:
        """state_dict 推理（简化：需要外部提供模型架构）。"""
        raise NotImplementedError(
            "state_dict inference requires model architecture. "
            "Use ONNX export for production inference, or provide a model_factory."
        )

    def info(self) -> Dict[str, Any]:
        """返回模型和服务信息。"""
        return {
            "output_dir": str(self.output_dir),
            "model_type": getattr(self, "_model_type", "unknown"),
            "metadata": self.metadata,
            "device": self.device,
        }

    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """启动 HTTP 推理服务（需要 FastAPI + uvicorn）。

        端点对齐 KServe v2 dataplane（RFC-003 IP）：
        - /v2/health/live, /v2/health/ready, /v2/health/model/{name}
        - /v2/repository/index, /v2/repository/models/{name}/load, /unload
        - /v2/models/{name}/infer
        - /v2/models/{name}, /v2/models/{name}/ready
        - /predict （向后兼容别名）

        Args:
            host: 监听地址
            port: 监听端口
        """
        try:
            from fastapi import FastAPI, HTTPException, Request
            from pydantic import BaseModel
            import uvicorn
        except ImportError as e:
            raise ImportError(
                f"FastAPI/uvicorn not installed: {e}. "
                f"Install with: pip install fastapi uvicorn"
            ) from e

        app = FastAPI(title="SenseFrame Inference Server")

        # P1.1: 初始化 OTel + 注册 /metrics 端点
        try:
            from .observability_otel import init_otel
            init_otel(
                service_name="senseframe-inference",
                model_id=self.metadata.get("model_id", "") if self.metadata else "",
                dataset=self.metadata.get("dataset", "") if self.metadata else "",
                enable_prometheus=True,
                prometheus_port=9464,
            )
        except ImportError:
            pass  # OTel 可选依赖未安装

        @app.get("/metrics")
        async def metrics():
            """Prometheus /metrics 端点（P1.1: OBP-3 Exporter）。

            当 opentelemetry-exporter-prometheus 可用时，指标通过
            OTel PrometheusMetricReader 在 9464 端口暴露。
            此端点作为 FastAPI 内的代理，读取 OTel InMemoryMetricReader 产出。
            """
            try:
                from .observability_otel import get_metrics_snapshot
                return get_metrics_snapshot()
            except (ImportError, AttributeError):
                return {"status": "otel_not_available", "metrics": []}

        class PredictRequest(BaseModel):
            input: Any

        # IP-1: Health 端点
        @app.get("/v2/health/live")
        async def health_live():
            return {"live": True}

        @app.get("/v2/health/ready")
        async def health_ready():
            return {"ready": self.model is not None}

        @app.get("/v2/health/model/{model_name}")
        async def model_ready(model_name: str):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return {"ready": True, "name": model_name}

        # IP-2: 模型仓库端点
        @app.get("/v2/repository/index")
        async def repo_index():
            return [{"name": self.metadata.get("model_id", "model") if self.metadata else "model",
                     "version": "1", "state": "READY", "reason": ""}]

        @app.post("/v2/repository/models/{model_name}/load")
        async def model_load(model_name: str):
            # 已加载，直接返回
            return {"name": model_name, "version": "1"}

        @app.post("/v2/repository/models/{model_name}/unload")
        async def model_unload(model_name: str):
            self.model = None
            return {"name": model_name, "version": "1"}

        # IP-3: 推理端点（KServe v2 格式）
        @app.post("/v2/models/{model_name}/infer")
        async def infer(model_name: str, request: Request):
            """KServe v2 推理端点。"""
            try:
                body = await request.json()
                inputs = body.get("inputs", [])
                if not inputs:
                    raise HTTPException(status_code=400, detail="No inputs provided")

                import numpy as np
                # KServe v2: inputs[0] 含 name/shape/datatype/data
                inp = inputs[0]
                data = inp.get("data", [])
                shape = inp.get("shape", [1])
                datatype = inp.get("datatype", "FP32")

                # 转换为 numpy
                dtype_map = {"FP32": np.float32, "FP64": np.float64, "INT8": np.int8,
                             "INT16": np.int16, "INT32": np.int32, "INT64": np.int64, "BOOL": np.bool_}
                np_dtype = dtype_map.get(datatype, np.float32)
                arr = np.array(data, dtype=np_dtype).reshape(shape)

                # P0.2: OBP 推理指标埋点（OTel 未初始化时 no-op）
                _t0 = time.perf_counter()
                # 推理
                result = self.predict(arr)
                _latency_ms = (time.perf_counter() - _t0) * 1000
                try:
                    from .observability_otel import record_inference_metric, ML_INFERENCE_LATENCY_MS, ML_INFERENCE_REQUEST_COUNT
                    record_inference_metric(ML_INFERENCE_LATENCY_MS, value=_latency_ms,
                                           model_id=model_name)
                    record_inference_metric(ML_INFERENCE_REQUEST_COUNT, value=1,
                                           model_id=model_name)
                except ImportError:
                    pass

                # KServe v2 响应格式
                if isinstance(result, dict):
                    # predict 返回 {"prediction": int, "probabilities": [...]}
                    output_data = result.get("probabilities", [result.get("prediction", 0)])
                elif isinstance(result, list):
                    output_data = result
                else:
                    output_data = [result]

                return {
                    "model_name": model_name,
                    "model_version": "1",
                    "outputs": [{
                        "name": "output_0",
                        "shape": [len(output_data)] if isinstance(output_data, list) else [1],
                        "datatype": "FP32",
                        "data": output_data,
                    }],
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # IP-4: 模型元数据端点
        @app.get("/v2/models/{model_name}")
        async def model_metadata(model_name: str):
            """KServe v2 模型元数据。"""
            meta = self.metadata or {}
            input_shape = meta.get("input_shape", [])
            return {
                "name": model_name,
                "versions": ["1"],
                "platform": "senseframe",
                "inputs": [{"name": "input_0", "shape": input_shape or [1], "datatype": "FP32"}],
                "outputs": [{"name": "output_0", "shape": [1, meta.get("num_classes", 1)], "datatype": "FP32"}],
            }

        @app.get("/v2/models/{model_name}/ready")
        async def model_ready_status(model_name: str):
            return {"ready": self.model is not None}

        # 向后兼容：保留 /predict 作为别名
        @app.post("/predict")
        async def predict_legacy(req: PredictRequest):
            """向后兼容端点（推荐使用 /v2/models/{name}/infer）。"""
            try:
                result = self.predict(req.input)
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        print(f"Starting inference server at http://{host}:{port}")
        print(f"  Model: {self._model_type} from {self.output_dir}")
        print(f"  Endpoints: /v2/health/live /v2/health/ready")
        print(f"             /v2/repository/index")
        print(f"             /v2/models/{{model_name}}/infer")
        print(f"             /v2/models/{{model_name}} (metadata)")
        print(f"             /predict (legacy)")
        # RFC-005：uvicorn.run 是阻塞调用，用 try/finally 确保 OTel 后台线程/端口被清理
        try:
            uvicorn.run(app, host=host, port=port)
        finally:
            try:
                from .observability_otel import shutdown_otel
                shutdown_otel()
            except Exception:
                pass


__all__ = ["InferenceServer"]
