"""
模型多格式导出：将训练好的 PyTorch 模型导出为 ONNX / TorchScript / state_dict 等格式。

设计目标：
- 统一接口 export_model()，屏蔽各格式导出差异
- 可选依赖懒加载：onnx/onnxruntime 缺失时给清晰提示
- 自动构造示例输入（从 input_shape 推断），无需用户手动指定
- 生成 export_manifest.json，记录各格式文件路径、SHA256、输入/输出规格
- 动态 batch 维度，导出后支持可变 batch 推理

支持格式：
- state_dict：torch.save(model.state_dict())，复现/继续训练用
- torchscript：torch.jit.trace，C++ 部署用，无 Python 依赖
- onnx：torch.onnx.export，跨框架部署（ONNX Runtime / TensorRT）
- quantized_onnx：onnxruntime 动态量化，边缘设备/低延迟场景
"""

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .engine.metadata import load_metadata

import torch
import torch.nn as nn

from .common import load_checkpoint_flexible

_logger = logging.getLogger(__name__)


# ============================================================
# 支持的导出格式
# ============================================================
SUPPORTED_FORMATS = ["state_dict", "torchscript", "onnx", "quantized_onnx"]

# 需要可选依赖的格式 → (import_name, pip_name)
_OPTIONAL_DEPS = {
    "onnx": ("onnx", "onnx"),
    "quantized_onnx": ("onnxruntime.quantization", "onnxruntime"),
}


@dataclass
class ExportResult:
    """导出结果：含各格式文件路径与清单。"""
    output_dir: str
    formats: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)   # format → file_path
    manifest_path: Optional[str] = None
    errors: Dict[str, str] = field(default_factory=dict)  # format → error msg
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "formats": self.formats,
            "files": self.files,
            "manifest_path": self.manifest_path,
            "errors": self.errors,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }


# ============================================================
# 工具函数
# ============================================================
def _file_sha256(path: Path) -> str:
    """计算文件 SHA256 校验和。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_optional_dep(format_name: str) -> Optional[str]:
    """检查可选依赖是否可用，返回缺失提示（None 表示可用）。"""
    if format_name not in _OPTIONAL_DEPS:
        return None
    import_name, pip_name = _OPTIONAL_DEPS[format_name]
    try:
        __import__(import_name)
        return None
    except ImportError:
        # P4-3：统一英文提示（与 inference.py 风格一致），便于自动化测试匹配
        return (
            f"Format '{format_name}' requires optional dependency '{pip_name}'. "
            f"Install with: pip install {pip_name}"
        )


def _build_sample_input(
    input_shape: List[int],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    根据模型 input_shape 构造示例输入张量。

    input_shape 不含 batch 维度时自动补 1，如 [1, 250, 90] → [1, 1, 250, 90]。
    """
    shape = list(input_shape)
    # 若首维不是 1 且看起来缺 batch 维，补 batch=1
    # 启发式：CSI 模型 input_shape 已含 channel 维（如 [1, 250, 90]），
    # 需补 batch；generic MLP input_shape=[4] 也需补 batch
    sample = torch.randn(1, *shape, device=device)
    return sample


# ============================================================
# Phase P0-2：自监督 _Parrallel 模型导出适配
# ============================================================
# _Parrallel 模型（来自 SenseFi）的 forward 签名为 forward(x1, x2, flag=...),
# 返回 (out1, out2) 元组。export_model 假设单输入 forward + 单输出，
# 直接调用会抛 TypeError: forward() missing 1 required positional argument: 'x2'.
# 解决：用 wrapper 将 _Parrallel 的监督路径 (x, x, flag='supervised') → (y1, y2)
# 包装为单输入 → y1 的标准 nn.Module，state_dict 共享原模型权重。
class _SelfSupervisedInferenceWrapper(nn.Module):
    """将 _Parrallel 双输入 forward 包装为单输入 forward，仅返回 y1（监督 logits）。

    - wrapper 与 base 共享权重（state_dict key 一致），导出 state_dict 仍为原始 _Parrallel 结构
    - 推理/导出只需单输出 y1（y2 在训练时作正则，推理时丢弃）
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        # 监督路径：x1=x2=x，返回 (y1, y2) logits，只取 y1
        y1, _y2 = self.base(x, x, flag="supervised")
        return y1

    # 透传 base 的属性（state_dict / eval / parameters 等由 nn.Module 自动处理）
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def _maybe_wrap_self_supervised(
    model: nn.Module,
    learning_mode: Optional[str] = None,
) -> nn.Module:
    """检测模型是否为 _Parrallel 风格（自监督双输入），如是则包装为单输入。

    检测条件（duck-typed）：
    - learning_mode == "self_supervised"
    - model.forward 签名参数数 >= 2（含 self）

    对监督模型（learning_mode=None / "supervised"）完全透明，直接返回原模型。
    """
    if learning_mode != "self_supervised":
        return model
    # duck-typed：检测 forward 签名
    import inspect
    sig = inspect.signature(model.forward)
    params = [p for p in sig.parameters.values()
              if p.name != "self" and p.default is inspect.Parameter.empty]
    if len(params) >= 2:
        return _SelfSupervisedInferenceWrapper(model)
    return model


def _infer_output_shape(model: nn.Module, sample_input: torch.Tensor) -> List[int]:
    """推断模型输出形状（不含 batch 维度）。"""
    model.eval()
    with torch.no_grad():
        out = model(sample_input)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return list(out.shape[1:])


# ============================================================
# Phase 12.2：输出激活包装
# ============================================================
_ACTIVATION_REGISTRY = {
    "none": None,
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
}


def _wrap_with_activation(model: nn.Module, activation_name: Optional[str]) -> nn.Module:
    """根据 activation_name 在模型输出后串接一个 nn.Module。

    返回的 _ActivationWrapper 与原模型共享权重，eval() 时行为一致。
    activation_name=None / "none" / 未注册 → 返回原模型。
    """
    if activation_name is None or activation_name == "none":
        return model
    if activation_name not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown output_activation '{activation_name}'. "
            f"Supported: {list(_ACTIVATION_REGISTRY.keys())}"
        )
    act = _ACTIVATION_REGISTRY[activation_name]
    if act is None:
        return model

    class _ActivationWrapper(nn.Module):
        def __init__(self, base, act_module):
            super().__init__()
            self.base = base
            self.act = act_module

        def forward(self, x):
            out = self.base(x)
            if isinstance(out, (tuple, list)):
                return (self.act(out[0]), *out[1:])
            return self.act(out)

        # 透传属性（state_dict / eval / parameters 等）
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.base, name)

    return _ActivationWrapper(model, act)


def list_supported_activations() -> List[str]:
    return list(_ACTIVATION_REGISTRY.keys())


# ============================================================
# 各格式导出实现
# ============================================================
def _export_state_dict(
    model: nn.Module,
    output_dir: Path,
    sample_input: torch.Tensor,
) -> Path:
    """导出 state_dict（.pth）。"""
    path = output_dir / "model_state_dict.pth"
    torch.save(model.state_dict(), path)
    return path


def _export_torchscript(
    model: nn.Module,
    output_dir: Path,
    sample_input: torch.Tensor,
) -> Path:
    """导出 TorchScript（.pt， traced）。"""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_input)
    path = output_dir / "model_torchscript.pt"
    traced.save(str(path))
    return path


def _export_onnx(
    model: nn.Module,
    output_dir: Path,
    sample_input: torch.Tensor,
    output_path: Optional[Path] = None,
) -> Path:
    """导出 ONNX（.onnx），支持动态 batch。

    torch >= 2.9 且 onnxscript 可用时使用 dynamo=True + dynamic_shapes（新路径）；
    否则使用 dynamo=False + dynamic_axes（经典 TorchScript 路径）。
    """
    model.eval()
    path = output_path or (output_dir / "model.onnx")

    _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    _use_dynamo = _torch_ver >= (2, 9)

    if _use_dynamo:
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            _use_dynamo = False

    with torch.no_grad():
        if _use_dynamo:
            from torch.export import Dim
            batch = Dim("batch", min=1)
            torch.onnx.export(
                model,
                (sample_input,),
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_shapes=({0: batch},),
                opset_version=18,
                dynamo=True,
            )
        else:
            torch.onnx.export(
                model,
                sample_input,
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "output": {0: "batch"},
                },
                opset_version=18,
                dynamo=False,
            )
    return path


def _export_quantized_onnx(
    model: nn.Module,
    output_dir: Path,
    sample_input: torch.Tensor,
) -> Path:
    """导出量化 ONNX（基于 onnxruntime 动态量化）。"""
    # 先导出普通 ONNX 到临时路径
    tmp_onnx = output_dir / "_tmp_for_quant.onnx"
    _export_onnx(model, output_dir, sample_input, output_path=tmp_onnx)

    # 用 onnxruntime 量化
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = output_dir / "model_quantized.onnx"
    quantize_dynamic(
        str(tmp_onnx),
        str(quant_path),
        weight_type=QuantType.QUInt8,
    )
    # 清理临时文件
    try:
        tmp_onnx.unlink()
    except OSError:
        pass
    return quant_path


# 格式 → 导出函数
_EXPORTERS = {
    "state_dict": _export_state_dict,
    "torchscript": _export_torchscript,
    "onnx": _export_onnx,
    "quantized_onnx": _export_quantized_onnx,
}


# ============================================================
# Phase 8.4：导出模型精度对齐验证
# ============================================================
def validate_export(
    original_model: nn.Module,
    exported_path: str,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    """
    验证导出模型与原模型输出一致性。

    用相同输入跑原 PyTorch 模型与导出模型，比较输出差异。
    支持验证 TorchScript 和 ONNX 格式。state_dict 无需验证（与原模型同源）。

    Args:
        original_model: 原始 PyTorch 模型（已加载权重）
        exported_path: 导出模型文件路径（.pt 或 .onnx）
        sample_input: 验证用输入张量
        atol: 绝对误差容限
        rtol: 相对误差容限

    Returns:
        {
            "format": "torchscript" / "onnx",
            "max_abs_diff": float,
            "max_rel_diff": float,
            "mean_abs_diff": float,
            "passed": bool,
            "atol": float,
            "rtol": float,
        }

    Raises:
        ValueError: 不支持的文件格式
        RuntimeError: 加载导出模型失败
    """
    original_model.eval()
    exported_path = Path(exported_path)

    # 原模型输出
    with torch.no_grad():
        orig_out = original_model(sample_input)
        if isinstance(orig_out, (tuple, list)):
            orig_out = orig_out[0]

    suffix = exported_path.suffix.lower()

    if suffix in (".pt",):
        # TorchScript
        loaded = torch.jit.load(str(exported_path))
        loaded.eval()
        with torch.no_grad():
            exp_out = loaded(sample_input)
            if isinstance(exp_out, (tuple, list)):
                exp_out = exp_out[0]
        fmt = "torchscript"

    elif suffix == ".onnx":
        # ONNX Runtime
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError(
                "ONNX 验证需要 onnxruntime，请 `pip install onnxruntime`"
            )
        sess = ort.InferenceSession(str(exported_path))
        input_name = sess.get_inputs()[0].name
        input_np = sample_input.detach().cpu().numpy()
        exp_out_np = sess.run(None, {input_name: input_np})[0]
        exp_out = torch.from_numpy(exp_out_np)
        fmt = "onnx"

    else:
        raise ValueError(
            f"不支持的导出格式 '{suffix}'，仅支持 .pt (TorchScript) 和 .onnx"
        )

    # 计算差异
    abs_diff = (orig_out - exp_out).abs()
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())

    # 相对差异（避免除零）
    denom = orig_out.abs().clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel_diff = float(rel_diff.max().item())

    passed = (max_abs_diff <= atol) and (max_rel_diff <= rtol)

    return {
        "format": fmt,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_abs_diff": mean_abs_diff,
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
    }


def _run_validation_for_export(
    model: nn.Module,
    result: ExportResult,
    sample_input: torch.Tensor,
) -> Dict[str, Any]:
    """
    为 ExportResult 中各格式运行精度对齐验证。

    Returns:
        {format: validation_result_dict} 仅含可验证的格式
    """
    validations: Dict[str, Any] = {}

    for fmt, path in result.files.items():
        if fmt == "state_dict":
            # state_dict 与原模型同源，无需验证
            continue
        try:
            val = validate_export(model, path, sample_input)
            validations[fmt] = val
        except Exception as e:
            validations[fmt] = {
                "format": fmt,
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
            }

    return validations


# ============================================================
# 主接口
# ============================================================
def export_model(
    model: nn.Module,
    output_dir: Path,
    formats: List[str],
    input_shape: List[int],
    metadata: Optional[Dict[str, Any]] = None,
    validate: bool = False,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    output_activation: Optional[str] = None,
    learning_mode: Optional[str] = None,
) -> ExportResult:
    """
    将模型导出为多种格式。

    Args:
        model: 已加载权重的 PyTorch 模型（会调用 model.eval()）
        output_dir: 导出目录
        formats: 导出格式列表，如 ["onnx", "torchscript", "state_dict"]
        input_shape: 模型输入形状（不含 batch 维），如 [1, 250, 90]
        metadata: 透传到 export_manifest.json 的额外元数据
        validate: Phase 8.4 是否运行精度对齐验证（默认 False）
        atol: 验证绝对误差容限
        rtol: 验证相对误差容限
        output_activation: Phase 12.2 输出激活名称
            （"none"/"softmax"/"sigmoid"/"tanh"/"relu"），None=不加激活
            通常来自 task_spec.output_activation
        learning_mode: P0-2 学习模式（"self_supervised" / "supervised" / None）。
            self_supervised 时若检测到 _Parrallel 双输入 forward，自动包装为单输入 wrapper。
            None / "supervised" 时透明（直接用原模型）。

    Returns:
        ExportResult: 含各格式文件路径与清单

    Raises:
        ValueError: formats 含不支持的格式或 output_activation 不合法
    """
    # 校验 formats
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的导出格式 '{fmt}'。支持: {SUPPORTED_FORMATS}"
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # P0-2：自监督 _Parrallel 模型双输入 forward 适配
    # 检测到双输入 forward 时包装为单输入 wrapper，state_dict 共享原模型权重
    model = _maybe_wrap_self_supervised(model, learning_mode)

    # Phase 12.2：先评估原模型，再用激活包装后的模型导出。
    # 包装后模型的 state_dict 仍兼容原模型的 key（权重共享），
    # state_dict 导出跳过 activation 层（直接保存原模型权重）。
    model.eval()
    sample_input = _build_sample_input(input_shape)
    output_shape = _infer_output_shape(model, sample_input)

    # 校验 output_activation（_wrap_with_activation 内部也会校验）
    if output_activation and output_activation not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown output_activation '{output_activation}'. "
            f"Supported: {list_supported_activations()}"
        )

    result = ExportResult(
        output_dir=str(output_dir),
        input_shape=list(input_shape),
        output_shape=output_shape,
    )

    # 逐格式导出
    for fmt in formats:
        # 检查可选依赖
        dep_err = _check_optional_dep(fmt)
        if dep_err is not None:
            result.errors[fmt] = dep_err
            continue

        try:
            # state_dict 保持原模型语义（不附加 activation）
            # 其它格式（torchscript/onnx/quantized_onnx）按需附加
            if fmt == "state_dict":
                target_model = model
            else:
                target_model = _wrap_with_activation(model, output_activation)
            exporter = _EXPORTERS[fmt]
            path = exporter(target_model, output_dir, sample_input)
            result.formats.append(fmt)
            result.files[fmt] = str(path)
        except Exception as e:
            result.errors[fmt] = f"{type(e).__name__}: {e}"

    # Phase 8.4：精度对齐验证（可选）
    validations: Dict[str, Any] = {}
    if validate and result.files:
        # 验证时使用带激活的模型（与导出模型一致）
        validate_model = _wrap_with_activation(model, output_activation)
        validations = _run_validation_for_export(validate_model, result, sample_input)

    # 生成导出清单
    manifest = _build_manifest(
        result, model, input_shape, output_shape, metadata or {},
        output_activation=output_activation,
    )
    if validations:
        manifest["validation"] = validations
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    result.manifest_path = str(manifest_path)

    return result


def _build_manifest(
    result: ExportResult,
    model: nn.Module,
    input_shape: List[int],
    output_shape: List[int],
    extra_metadata: Dict[str, Any],
    output_activation: Optional[str] = None,
) -> Dict[str, Any]:
    """构建 export_manifest.json 内容。"""
    files_info = []
    for fmt, path in result.files.items():
        p = Path(path)
        files_info.append({
            "format": fmt,
            "filename": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "sha256": _file_sha256(p) if p.exists() else None,
        })

    manifest = {
        "exported_at": datetime.now().isoformat(),
        "model_class": type(model).__name__,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "input_spec": {
            "name": "input",
            "dtype": "float32",
            "shape": ["batch"] + list(input_shape),
        },
        "output_spec": {
            "name": "output",
            "dtype": "float32",
            "shape": ["batch"] + list(output_shape),
        },
        "files": files_info,
        "errors": result.errors,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "metadata": extra_metadata,
    }
    if output_activation and output_activation != "none":
        manifest["output_activation"] = output_activation
    return manifest


# ============================================================
# 便捷函数：从 metadata.json 加载模型并导出
# ============================================================
def export_from_metadata(
    metadata_path: str,
    checkpoint_path: str,
    output_dir: str,
    formats: List[str],
    validate: bool = False,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    output_activation: Optional[str] = None,
) -> ExportResult:
    """
    从训练输出的 metadata.json 加载模型并导出。

    Args:
        metadata_path: metadata.json 路径
        checkpoint_path: model.pth 路径
        output_dir: 导出目录
        formats: 导出格式列表
        validate: Phase 8.4 是否运行精度对齐验证
        atol: 验证绝对误差容限
        rtol: 验证相对误差容限
        output_activation: Phase 12.2 输出激活名称
            默认从 metadata.task_spec.output_activation 读取
            CLI --output-activation 可显式覆盖

    Returns:
        ExportResult
    """
    from .scenes import get_scene

    metadata_path = Path(metadata_path)
    # P3：通过 load_metadata 自动协商 schema_version 迁移
    metadata = load_metadata(metadata_path)

    model_id = metadata["model_id"]
    dataset = metadata["dataset"]
    num_classes = metadata["num_classes"]
    learning_mode = metadata.get("learning_mode", "supervised")
    input_shape = metadata.get("input_shape", [])

    # Phase 12.2：未显式指定时，从 metadata.task_spec.output_activation 推断
    if output_activation is None:
        task_spec = metadata.get("task_spec") or {}
        output_activation = task_spec.get("output_activation")

    if not input_shape:
        raise ValueError(
            "metadata.json 缺少 input_shape 字段，无法构造示例输入。"
            "请手动调用 export_model() 并传入 input_shape。"
        )

    # Phase 8.4：支持 CustomContainer 场景（manifest 数据集）
    manifest_info = metadata.get("manifest")
    if manifest_info is not None:
        # CustomContainer 场景：用 custom scene + GenericMLP
        # GenericMLP 期望 2D 输入 (batch, features)，故将 input_shape 展平
        from .scenes.generic.container import GenericMLP
        import numpy as np
        input_dim = int(np.prod(input_shape)) if input_shape else 1
        model = GenericMLP(input_dim=input_dim, num_classes=num_classes)
        # 展平 input_shape 以匹配 GenericMLP 的输入
        input_shape = [input_dim]
    else:
        scene = get_scene("wifi_csi" if dataset in [
            "UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"
        ] else "generic")

        # 构建模型并加载权重
        model = scene.build_model_for_dataset(
            model_id, dataset, num_classes, learning_mode=learning_mode,
        )
    # 修复：使用 load_checkpoint_flexible 兼容 Lightning checkpoint 与裸 state_dict。
    # 旧代码直接 torch.load + model.load_state_dict(state_dict) 会因 Lightning ckpt
    # 顶层含 epoch/global_step/optimizer_states 等非权重字段触发
    # `Unexpected key(s) in state_dict: "epoch", "global_step", ...` 错误，
    # 且未剥离 "model." 前缀导致 key 不匹配，F 阶段 ONNX 导出全部失败。
    # weights_only=False 兼容 Lightning ckpt 的 callbacks 字段（含 Python 对象）。
    load_info = load_checkpoint_flexible(
        checkpoint_path, model, map_location="cpu", weights_only=False,
    )
    _logger.info(
        "export_from_metadata: loaded checkpoint %s (format=%s, keys=%d, prefix=%r)",
        checkpoint_path,
        load_info["source_format"],
        load_info["num_keys_loaded"],
        load_info["stripped_prefix"],
    )
    model.eval()

    return export_model(
        model=model,
        output_dir=Path(output_dir),
        formats=formats,
        input_shape=input_shape,
        metadata={
            "model_id": model_id,
            "dataset": dataset,
            "learning_mode": learning_mode,
            "num_classes": num_classes,
            "source_checkpoint": checkpoint_path,
        },
        validate=validate,
        atol=atol,
        rtol=rtol,
        output_activation=output_activation,
        # P0-2 修复：CLI export 路径同样需传入 learning_mode，否则 export_model 无法触发
        # _maybe_wrap_self_supervised，对 _Parrallel 双输入 forward 仍抛 TypeError。
        # 之前只修复了 stage_export.py（pipeline 路径），漏改了 export_from_metadata（CLI 路径）。
        learning_mode=learning_mode,
    )


__all__ = [
    "SUPPORTED_FORMATS",
    "ExportResult",
    "export_model",
    "export_from_metadata",
    "validate_export",
    "list_supported_activations",
]
