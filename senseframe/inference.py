"""
推理 API 一等公民：从训练产物加载模型，执行单样本/批量推理。

设计目标：
- 训练与推理解耦：推理只需 model.pth + metadata.json，无需训练配置
- 支持任意场景（WiFi CSI / Generic / Custom），通过 metadata 重建模型
- 输出结构化 PredictionResult，便于序列化为 JSON
- 自动应用训练时的归一化（从 metadata 读取）
- 支持单样本 + 批量推理

核心接口：
    model = load_model_for_inference(model_path, metadata_path)
    result = model.predict_single(x: np.ndarray)
    results = model.predict_batch(samples: List[np.ndarray])

    # 便捷函数
    results = predict(model_path, metadata_path, samples, output_format="json")
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .common import load_checkpoint_flexible
from .engine.metadata import load_metadata

_logger = logging.getLogger(__name__)


# ============================================================
# 推理结果数据结构
# ============================================================
@dataclass
class PredictionResult:
    """单样本推理结果。"""
    label: int                              # 预测类别（整数）
    label_name: Optional[str] = None        # 类别名称（从 label_map 解析）
    confidence: float = 0.0                 # 置信度（softmax 最大值）
    logits: Optional[List[float]] = None    # 原始 logits（可选）
    path: Optional[str] = None              # 样本路径（批量推理时填充）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 推理模型包装器
# ============================================================
class InferenceModel:
    """
    推理模型包装器：加载训练产物，提供单样本/批量推理接口。

    通过 metadata.json 重建模型架构，加载权重，应用归一化。

    使用方式：
        model = load_model_for_inference(
            model_path="runs/exp/model.pth",
            metadata_path="runs/exp/metadata.json",
        )
        result = model.predict_single(np.load("sample.npy"))
        print(result.label_name, result.confidence)
    """

    def __init__(
        self,
        model: nn.Module,
        metadata: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Args:
            model: 已加载权重的模型实例
            metadata: 训练时保存的 metadata.json 内容
            device: 推理设备（cpu / cuda）
        """
        self.model = model
        self.metadata = metadata
        self.device = device

        # 从 metadata 提取推理所需信息
        self.num_classes = metadata.get("num_classes", 0)
        self.input_shape = metadata.get("input_shape", [])
        self.label_map = {int(k): v for k, v in metadata.get("label_map", {}).items()}
        self.normalization = metadata.get("normalization")
        self.model_id = metadata.get("model_id", "unknown")
        self.dataset = metadata.get("dataset", "unknown")
        self.learning_mode = metadata.get("learning_mode", "supervised")

        # 解析归一化常量
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._parse_normalization()

        # 模型设为评估模式
        self.model.to(device)
        self.model.eval()

    def _parse_normalization(self) -> None:
        """从 metadata 解析归一化常量。"""
        norm = self.normalization
        if norm is None:
            return

        # WiFi CSI 场景：{"mean": 42.3199, "std": 4.9802}
        if "mean" in norm and "std" in norm:
            mean = norm["mean"]
            std = norm["std"]
            self._mean = float(mean[0]) if isinstance(mean, list) else float(mean)
            self._std = float(std[0]) if isinstance(std, list) else float(std)

        # minmax 归一化
        if "min" in norm and "max" in norm:
            mn = norm["min"]
            mx = norm["max"]
            self._min = float(mn[0]) if isinstance(mn, list) else float(mn)
            self._max = float(mx[0]) if isinstance(mx, list) else float(mx)

    def _apply_normalization(self, x: np.ndarray) -> np.ndarray:
        """应用归一化到输入。"""
        x = x.astype(np.float32)
        if self._mean is not None and self._std is not None:
            x = (x - self._mean) / self._std
        elif self._min is not None and self._max is not None:
            x = (x - self._min) / max(self._max - self._min, 1e-12)
        return x

    def _preprocess(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """预处理：归一化 + flatten（多维输入）+ 转 tensor + 添加 batch 维度。"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self._apply_normalization(x)

        # 多维输入需 flatten 给 GenericMLP（与训练时 get_transforms 一致）
        # CustomContainer 场景：metadata 含 manifest 字段
        manifest_info = self.metadata.get("manifest")
        is_custom = manifest_info is not None and len(self.input_shape) > 1
        if is_custom:
            x = x.reshape(-1)

        t = torch.from_numpy(x).float()
        # 添加 batch 维度（若输入无 batch 维度）
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() == len(self.input_shape):
            t = t.unsqueeze(0)
        return t.to(self.device)

    def _postprocess(
        self,
        logits: torch.Tensor,
        path: Optional[str] = None,
        include_logits: bool = False,
    ) -> PredictionResult:
        """后处理：logits → PredictionResult。"""
        # softmax 计算置信度
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred_idx].item())

        label_name = self.label_map.get(pred_idx)

        result = PredictionResult(
            label=pred_idx,
            label_name=label_name,
            confidence=round(confidence, 6),
            path=path,
        )
        if include_logits:
            result.logits = [round(float(v), 6) for v in logits[0].tolist()]

        return result

    def predict_single(
        self,
        x: Union[np.ndarray, torch.Tensor],
        include_logits: bool = False,
    ) -> PredictionResult:
        """
        单样本推理。

        Args:
            x: 输入数据（np.ndarray 或 torch.Tensor）
            include_logits: 是否在结果中包含原始 logits

        Returns:
            PredictionResult
        """
        with torch.no_grad():
            x_tensor = self._preprocess(x)
            logits = self.model(x_tensor)
            return self._postprocess(logits, path=None, include_logits=include_logits)

    def predict_batch(
        self,
        samples: List[Union[np.ndarray, torch.Tensor, Dict[str, Any]]],
        include_logits: bool = False,
    ) -> List[PredictionResult]:
        """
        批量推理。

        Args:
            samples: 样本列表。元素可为：
                - np.ndarray / torch.Tensor：直接推理
                - dict：{"path": "...", "data": np.ndarray}，path 会记录到结果
            include_logits: 是否在结果中包含原始 logits

        Returns:
            PredictionResult 列表
        """
        results: List[PredictionResult] = []
        for sample in samples:
            path = None
            if isinstance(sample, dict):
                path = sample.get("path")
                data = sample.get("data")
                if data is None and path is not None:
                    # 从 path 加载数据
                    data = _load_sample_from_path(path, self.metadata)
            else:
                data = sample

            if data is None:
                raise ValueError(f"无法获取样本数据: {sample}")

            with torch.no_grad():
                x_tensor = self._preprocess(data)
                logits = self.model(x_tensor)
                result = self._postprocess(logits, path=path, include_logits=include_logits)
                results.append(result)

        return results


# ============================================================
# ONNX 推理模型（P3：封装 onnxruntime）
# ============================================================
class ONNXInferenceModel:
    """ONNX 推理模型（P3：封装 onnxruntime）。

    加载导出的 ONNX 模型，提供 predict 接口，无需 torch 依赖。

    使用方式：
        model = ONNXInferenceModel("exports/model.onnx")
        out = model.predict(np.array([...]))
    """

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                f"onnxruntime not installed: {e}. Install with: pip install onnxruntime"
            ) from e

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, input_data: Any) -> Any:
        """单样本推理。"""
        import numpy as np
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        if len(input_data.shape) == 1:
            input_data = input_data[np.newaxis, ...]
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return outputs[0].tolist()

    def predict_batch(self, input_batch: List[Any]) -> List[Any]:
        """批量推理。"""
        import numpy as np
        batch = np.array(input_batch, dtype=np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: batch})
        return outputs[0].tolist()


# ============================================================
# 模型加载
# ============================================================
def _load_sample_from_path(
    path: str,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """根据 metadata 中的 manifest 信息加载样本文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sample file not found: {p}")

    manifest_info = metadata.get("manifest") or {}
    file_format = manifest_info.get("file_format", "npy")
    mat_key = manifest_info.get("mat_key")

    from .data.manifest import _load_sample_file
    return _load_sample_file(p, file_format, mat_key)


def _build_model_from_metadata(metadata: Dict[str, Any]) -> nn.Module:
    """
    根据 metadata 重建模型架构。

    策略：
    1. 若 metadata 含 manifest 信息 → CustomContainer 场景，构建 GenericMLP
    2. 若 dataset 是 WiFi CSI 内置数据集 → 用 registry 构建
    3. 若以上都不匹配 → 构建 GenericMLP（generic 场景回退）
    """
    model_id = metadata.get("model_id")
    dataset = metadata.get("dataset")
    num_classes = metadata.get("num_classes", 2)
    input_shape = metadata.get("input_shape", [])
    manifest_info = metadata.get("manifest")

    # CustomContainer 场景
    if manifest_info is not None:
        from .scenes.generic.container import GenericMLP
        input_dim = int(np.prod(input_shape)) if input_shape else 1
        return GenericMLP(input_dim=input_dim, num_classes=num_classes)

    # WiFi CSI 内置数据集
    from .registry import is_dataset_registered, is_model_registered
    if (dataset is not None
        and is_dataset_registered(dataset)
        and is_model_registered(model_id)
        and _has_factory(model_id, dataset)
    ):
        from .registry import get_model
        return get_model(model_id, dataset, num_classes, learning_mode="supervised")

    # generic 场景回退
    from .scenes.generic.container import GenericMLP
    input_dim = int(np.prod(input_shape)) if input_shape else 1
    return GenericMLP(input_dim=input_dim, num_classes=num_classes)


def _has_factory(model_id: str, dataset: str) -> bool:
    """检查模型是否绑定了指定数据集的工厂。"""
    from .registry import resolve_factory
    try:
        resolve_factory(model_id, dataset)
        return True
    except Exception:
        return False


def load_model_for_inference(
    model_path: Union[str, Path],
    metadata_path: Union[str, Path],
    device: str = "cpu",
) -> InferenceModel:
    """
    从训练产物加载推理模型。

    Args:
        model_path: 模型权重路径（.pth）
        metadata_path: metadata.json 路径
        device: 推理设备（cpu / cuda）

    Returns:
        InferenceModel 实例

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 模型加载失败
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 加载 metadata（P3：通过 load_metadata 自动协商 schema_version 迁移）
    metadata = load_metadata(metadata_path)

    # 重建模型架构
    model = _build_model_from_metadata(metadata)

    # 加载权重
    # 修复：使用 load_checkpoint_flexible 兼容 Lightning checkpoint 与裸 state_dict。
    # 旧代码直接 torch.load + model.load_state_dict(state_dict) 与 export.py 同源 bug：
    # Lightning ckpt 顶层含 epoch/global_step/optimizer_states 等非权重字段触发
    # `Unexpected key(s) in state_dict`，且未剥离 "model." 前缀导致 key 不匹配。
    load_info = load_checkpoint_flexible(
        model_path, model, map_location=device, weights_only=False,
    )
    _logger.info(
        "load_model_for_inference: loaded checkpoint %s (format=%s, keys=%d, prefix=%r)",
        model_path,
        load_info["source_format"],
        load_info["num_keys_loaded"],
        load_info["stripped_prefix"],
    )

    return InferenceModel(model=model, metadata=metadata, device=device)


# ============================================================
# 便捷函数
# ============================================================
def predict(
    model_path: Union[str, Path],
    metadata_path: Union[str, Path],
    samples: List[Union[np.ndarray, Dict[str, Any]]],
    output_format: str = "dict",
    include_logits: bool = False,
    device: str = "cpu",
) -> Union[List[PredictionResult], List[Dict[str, Any]], str]:
    """
    便捷推理函数：加载模型 + 批量推理 + 可选 JSON 输出。

    Args:
        model_path: 模型权重路径
        metadata_path: metadata.json 路径
        samples: 样本列表（np.ndarray 或 {"path": "..."} dict）
        output_format: 输出格式
            - "dict": List[Dict]（默认，便于 JSON 序列化）
            - "result": List[PredictionResult]
            - "json": JSON 字符串
        include_logits: 是否包含 logits
        device: 推理设备

    Returns:
        按 output_format 返回推理结果

    Examples:
        # 从 .npy 文件批量推理
        results = predict(
            model_path="runs/exp/model.pth",
            metadata_path="runs/exp/metadata.json",
            samples=[{"path": "data/x1.npy"}, {"path": "data/x2.npy"}],
            output_format="dict",
        )
        # results: [{"path": "...", "label": 3, "label_name": "walk", "confidence": 0.92}, ...]
    """
    # 修复（5.18）：离线批量推理无 OTel 埋点。
    # 旧逻辑：predict 入口未调用 init_otel，record_inference_metric 全部 no-op。
    # 与 Pipeline.run 入口一致，predict 入口也调用 init_otel（OTel 未安装时 no-op）。
    try:
        from .observability_otel import init_otel
        init_otel()
    except Exception:
        # OTel 初始化失败不影响推理主流程（与 Pipeline.run 同语义）
        pass

    model = load_model_for_inference(model_path, metadata_path, device=device)
    results = model.predict_batch(samples, include_logits=include_logits)

    if output_format == "result":
        return results
    elif output_format == "dict":
        return [r.to_dict() for r in results]
    elif output_format == "json":
        return json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)
    else:
        raise ValueError(
            f"output_format '{output_format}' 不支持，可选: dict / result / json"
        )


__all__ = [
    "PredictionResult",
    "InferenceModel",
    "ONNXInferenceModel",
    "load_model_for_inference",
    "predict",
]
