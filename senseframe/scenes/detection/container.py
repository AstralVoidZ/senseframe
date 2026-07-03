"""
Phase 12.5 — Detection 场景 stub。

目的：
- 验证 Phase 11 TaskType / Loss / FeatureSpec / SceneParams 在非分类任务下的
  端到端接入能力
- 提供一个最小可用的目标检测容器（仅 stub，不含真实检测算法）
- 演示用户如何从零接入新任务类型

使用方式：
    from senseframe.scenes import get_scene
    scene = get_scene("detection")
    info = scene.get_dataset_info("dummy_box")
    task_spec = scene.get_task_spec("dummy_box")  # 自动产出 detection task_type
    feature_spec = scene.get_feature_spec("dummy_box")
    model = scene.build_model_for_dataset("SimpleDetector", "dummy_box")

注意：本容器仅用于演示 TaskType 接入，不实现真实的目标检测算法
（不接 NMS / IoU / mAP / box regression 等）。真实检测任务应继承
本容器并补全实现。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..base import (
    DatasetBundle,
    DefaultConfig,
    SceneContainer,
    SceneMeta,
    SearchSpace,
    TransformConfig,
)
from ...core.task import TaskSpec
from ...core.features import FeatureSpec
from ...core.params import SceneParams


# ============================================================
# Phase 13.3：NMS 工具函数（纯 torch 实现，无 torchvision 依赖）
# ============================================================
def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Non-Maximum Suppression（纯 torch 实现）。

    Args:
        boxes: (N, 4) 边界框，xyxy 格式
        scores: (N,) 置信度分数
        iou_threshold: IoU 阈值，高于此值的重叠框被抑制

    Returns:
        keep: (K,) 保留的框索引
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        # 当前最高分框
        xi1, yi1, xi2, yi2 = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
        area_i = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)

        # 剩余框
        rest = order[1:]
        x1 = boxes[rest, 0].clamp(min=xi1)
        y1 = boxes[rest, 1].clamp(min=yi1)
        x2 = boxes[rest, 2].clamp(max=xi2)
        y2 = boxes[rest, 3].clamp(max=yi2)

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_j = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_j - inter
        iou = inter / union.clamp(min=1e-6)

        mask = iou <= iou_threshold
        order = rest[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """将 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """将 (x, y, w, h) 格式转换为 (x1, y1, x2, y2) 格式。"""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


# ============================================================
# Stub 模型：SimpleDetector
# ============================================================
class SimpleDetector(nn.Module):
    """
    最小可用的"检测器"stub。

    输入：(B, C, H, W) 图像张量
    输出：dict {bboxes: (B, 4), logits: (B, num_classes)}

    仅为演示，不实现真实目标检测逻辑（无 anchor / NMS / box regression）。
    用户可在此基础上扩展为 Faster R-CNN / YOLO / DETR 等。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.bbox_head = nn.Linear(in_channels, 4)
        self.cls_head = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            # (B, C) → reshape to (B, C, 1, 1) for AdaptiveAvgPool2d
            x = x.unsqueeze(-1).unsqueeze(-1)
        feat = self.backbone(x)
        return {
            "bboxes": self.bbox_head(feat),
            "logits": self.cls_head(feat),
        }


# ============================================================
# 内置数据集 stub
# ============================================================
_DETECTION_DATASETS = {
    "dummy_box": {
        "name": "dummy_box",
        "num_classes": 3,            # 3 个目标类别
        "input_shape": [3, 64, 64],  # 图像
        "modality": "image",
    },
    "tiny_coco": {
        "name": "tiny_coco",
        "num_classes": 80,
        "input_shape": [3, 224, 224],
        "modality": "image",
    },
}


_DETECTION_MODELS = {
    "SimpleDetector": SimpleDetector,
}


# ============================================================
# DetectionContainer
# ============================================================
class DetectionContainer(SceneContainer):
    """
    Phase 12.5：检测场景 stub 容器。

    支持的 task_type：DETECTION
    支持的 datasets：dummy_box / tiny_coco
    支持的 models：SimpleDetector
    """

    def meta(self) -> SceneMeta:
        return SceneMeta(
            name="detection",
            supported_tasks=["detection"],
            supported_models=list(_DETECTION_MODELS.keys()),
            supported_datasets=list(_DETECTION_DATASETS.keys()),
        )

    def load_dataset(
        self,
        dataset_name: str,
        root: Optional[str] = None,
        learning_mode: str = "supervised",
        **kwargs,
    ) -> DatasetBundle:
        if dataset_name not in _DETECTION_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}' for detection scene. "
                f"Available: {list(_DETECTION_DATASETS.keys())}"
            )
        info = _DETECTION_DATASETS[dataset_name]
        num_classes = info["num_classes"]

        # stub dataset：随机样本，labels 为 (bbox, cls_label)
        from torch.utils.data import TensorDataset
        torch.manual_seed(0)
        x = torch.randn(32, *info["input_shape"])
        bboxes = torch.rand(32, 4)
        cls_labels = torch.randint(0, num_classes, (32,))
        train_ds = TensorDataset(x, bboxes, cls_labels)
        test_ds = TensorDataset(x[:8], bboxes[:8], cls_labels[:8])

        from ..base import DatasetBundle
        bundle = DatasetBundle(
            train=train_ds, test=test_ds, val=None,
            unsupervised=None, supervised_finetune=None,
        )
        # Phase 12.5：detection 场景元信息（场景特定字段）
        # Phase 9.1 DatasetBundle 未预留 extra，故这里通过属性动态挂载
        bundle.num_classes = num_classes
        bundle.task_type = "detection"
        return bundle

    def build_model_for_dataset(
        self,
        model_id: str,
        dataset_name: str,
        num_classes: int = None,
        learning_mode: str = "supervised",
        **kwargs,
    ) -> nn.Module:
        if model_id not in _DETECTION_MODELS:
            raise ValueError(
                f"Unknown model '{model_id}' for detection scene. "
                f"Available: {list(_DETECTION_MODELS.keys())}"
            )
        info = _DETECTION_DATASETS[dataset_name]
        in_channels = info["input_shape"][0]
        n_cls = num_classes if num_classes is not None else info["num_classes"]
        cls = _DETECTION_MODELS[model_id]
        return cls(in_channels=in_channels, num_classes=n_cls)

    def get_dataset_info(
        self,
        dataset_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if dataset_name not in _DETECTION_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return dict(_DETECTION_DATASETS[dataset_name])

    # --------------------------------------------------------
    # Phase 11.1 默认覆写：返回 DETECTION TaskSpec
    # --------------------------------------------------------
    def get_task_spec(
        self,
        dataset_name: str,
        model_id: str = "",
        **kwargs,
    ) -> TaskSpec:
        info = self.get_dataset_info(dataset_name)
        return TaskSpec(
            task_type="detection",
            num_classes=info["num_classes"],
            loss="bce_with_logits",
            metrics=["map"],
            output_activation="sigmoid",
        )

    # --------------------------------------------------------
    # Phase 11.3 默认覆写：返回 image FeatureSpec
    # --------------------------------------------------------
    def get_feature_spec(
        self,
        dataset_name: str,
        **kwargs,
    ) -> FeatureSpec:
        info = self.get_dataset_info(dataset_name)
        return FeatureSpec(
            input_shape=tuple(info["input_shape"]),
            modality="image",
        )

    # --------------------------------------------------------
    # Phase 11.4 默认覆写：返回标准 SceneParams（带 detection 特定字段）
    # --------------------------------------------------------
    def get_scene_params(self, dataset_name: str, **kwargs) -> SceneParams:
        return SceneParams(
            target_frames=1,            # 检测任务单帧输入
            window_size=None,
            overlap_ratio=None,
            task_type="detection",
            loss="bce_with_logits",
            metrics=["map"],
            extra={
                "bbox_format": "cxcywh",   # detection 场景特定字段
                "nms_threshold": 0.5,
                "score_threshold": 0.05,
            },
        )

    def get_default_config(
        self,
        model_id: str,
        dataset_name: str,
        **kwargs,
    ) -> DefaultConfig:
        return DefaultConfig(
            epochs=50,
            batch_size=8,
            learning_rate=1e-3,
        )

    def get_search_space(
        self,
        model_id: str,
        dataset_name: str,
        **kwargs,
    ) -> SearchSpace:
        space = SearchSpace()
        space.params["bbox_format"] = {
            "type": "categorical",
            "values": ["cxcywh", "xyxy", "xywh"],
        }
        space.params["nms_threshold"] = {
            "type": "float",
            "low": 0.1, "high": 0.9, "log": False,
        }
        # Phase 12.1：注入 task / loss 搜索空间
        from ...engine.hpo import get_task_search_space_extension
        space.params.update(get_task_search_space_extension())
        return space

    def get_transforms(self, dataset_name: str, **kwargs) -> TransformConfig:
        """返回数据集的变换配置（RFC-002 阶段 U：接入 detection transforms 原语）。

        Agent 可通过 params.transform.pipeline 配置原语序列，
        通过 params.transform.augment 注入数据增强（仅 train）。

        支持的 params.transform 字段：
        - pipeline: 原语名列表，如 ["bbox_clip"]
        - augment: 增强原语名列表，如 ["hsv_jitter", "cutout"]（仅 train）
        - pipeline_params: 原语参数
        """
        params = kwargs.get("params") or kwargs.get("scene_params") or {}
        transform_cfg_params = params.get("transform", {}) if isinstance(params, dict) else {}
        pipeline = transform_cfg_params.get("pipeline")
        augment = transform_cfg_params.get("augment")
        pipeline_params = transform_cfg_params.get("pipeline_params", {})

        train_transform = None
        eval_transform = None

        if pipeline:
            from .transforms import compose_transforms
            pipeline_fn = compose_transforms(pipeline, **pipeline_params)
            train_transform = pipeline_fn
            eval_transform = pipeline_fn

        if augment:
            from .transforms import compose_transforms
            augment_fn = compose_transforms(augment, **pipeline_params)
            if train_transform is not None:
                def _train_with_aug(x, y, _base=train_transform, _aug=augment_fn):
                    x, y = _base(x, y)
                    return _aug(x, y)
                train_transform = _train_with_aug
            else:
                train_transform = augment_fn

        return TransformConfig(train_transform=train_transform, eval_transform=eval_transform)

    # --------------------------------------------------------
    # Phase 13.3：NMS 后处理
    # --------------------------------------------------------
    def postprocess(
        self,
        model_output: Dict[str, torch.Tensor],
        scene_params: Optional[SceneParams] = None,
        dataset_name: str = "dummy_box",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """对检测模型输出应用 NMS 后处理。

        流程：
        1. bbox 格式转换 → xyxy
        2. sigmoid 激活 logits → 置信度分数
        3. 按 score_threshold 过滤低置信度框
        4. NMS 去除重叠框

        Args:
            model_output: {"bboxes": (B, 4), "logits": (B, num_classes)}
            scene_params: 含 nms_threshold / score_threshold / bbox_format
            dataset_name: 数据集名（用于获取默认 scene_params）

        Returns:
            {"bboxes": (K, 4), "scores": (K,), "labels": (K,)}
        """
        params = scene_params or self.get_scene_params(dataset_name)
        nms_threshold = params.extra.get("nms_threshold", 0.5)
        score_threshold = params.extra.get("score_threshold", 0.05)
        bbox_format = params.extra.get("bbox_format", "cxcywh")

        bboxes = model_output["bboxes"]
        logits = model_output["logits"]

        # 1. bbox 格式转换
        if bbox_format == "cxcywh":
            boxes_xyxy = _cxcywh_to_xyxy(bboxes)
        elif bbox_format == "xywh":
            boxes_xyxy = _xywh_to_xyxy(bboxes)
        else:
            boxes_xyxy = bboxes  # 已是 xyxy

        # 2. sigmoid 激活 → 置信度
        scores = torch.sigmoid(logits)
        max_scores, labels = scores.max(dim=-1)

        # 3. score 过滤
        mask = max_scores >= score_threshold
        boxes_f = boxes_xyxy[mask]
        scores_f = max_scores[mask]
        labels_f = labels[mask]

        # 4. NMS
        keep = _nms(boxes_f, scores_f, nms_threshold)

        return {
            "bboxes": boxes_f[keep],
            "scores": scores_f[keep],
            "labels": labels_f[keep],
        }

    def get_catalog(self):
        """返回 detection 场景的技术目录（RFC-002 阶段 U）。"""
        from .catalog import CATALOG
        return CATALOG
