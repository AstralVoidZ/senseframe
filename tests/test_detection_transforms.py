"""senseframe.scenes.detection.transforms 与 catalog 模块测试。

覆盖 RFC-002 阶段 U 新增的 detection 场景 6 个 transform 原语
（4 图像增强 + 2 bbox 处理）+ 注册表 + compose_transforms + 技术目录。
bbox 统一使用 xyxy 格式。
"""

import numpy as np
import pytest

from senseframe.scenes.detection.transforms import (
    hsv_jitter,
    cutout,
    mixup,
    random_erasing,
    bbox_clip,
    bbox_flip,
    TRANSFORM_REGISTRY,
    get_transform,
    list_transforms,
    compose_transforms,
)
from senseframe.scenes.detection.catalog import (
    list_techniques,
    list_categories,
    get_applicable_techniques,
    is_augment,
    suggest_pipeline,
    suggest_augment,
)


# ============================================================
# 图像增强原语
# ============================================================
class TestHSVJitter:
    """hsv_jitter：HSV 空间抖动。"""

    def test_shape_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.rand(3, 32, 32) * 255.0
        result = hsv_jitter(x, hue=0.1, saturation=0.1, brightness=0.1)
        assert result.shape == (3, 32, 32)


class TestCutout:
    """cutout：随机遮挡。"""

    def test_has_zero_region(self):
        x = np.ones((3, 32, 32)) * 128.0
        result = cutout(x, n_holes=1, length=8)
        assert result.shape == (3, 32, 32)
        # 遮挡区域被置零
        assert (result == 0.0).any()


class TestMixup:
    """mixup：样本与标签混合。"""

    def test_shapes_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.rand(4, 3, 32, 32)
        y = np.eye(3)[np.array([0, 1, 2, 0])]  # one-hot (4, 3)
        x_out, y_out = mixup(x, y, alpha=0.2)
        assert x_out.shape == x.shape
        assert y_out.shape == y.shape


class TestRandomErasing:
    """random_erasing：随机擦除。"""

    def test_shape_unchanged(self):
        rng = np.random.RandomState(0)
        x = rng.rand(3, 32, 32) * 255.0
        result = random_erasing(x, area_ratio=(0.02, 0.2), min_aspect=0.3)
        assert result.shape == (3, 32, 32)


# ============================================================
# bbox 处理原语
# ============================================================
class TestBboxClip:
    """bbox_clip：裁剪到图像边界。"""

    def test_clip_out_of_bounds(self):
        # xyxy 格式，超出 (32, 32) 边界
        boxes = np.array([[-5.0, -5.0, 40.0, 40.0]])
        result = bbox_clip(boxes, img_size=(32, 32))
        assert result.shape == (1, 4)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 32.0
        assert result[0, 3] == 32.0


class TestBboxFlip:
    """bbox_flip：水平翻转坐标。"""

    def test_horizontal_flip(self):
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
        result = bbox_flip(boxes, img_width=32, flip_type="horizontal")
        # flipped x1 = 32 - 20 = 12, x2 = 32 - 10 = 22
        assert result[0, 0] == 12.0
        assert result[0, 2] == 22.0
        # y 坐标不变
        assert result[0, 1] == 10.0
        assert result[0, 3] == 20.0


# ============================================================
# 组合与注册表
# ============================================================
class TestComposeTransforms:
    """compose_transforms：组合原语。"""

    def test_compose_no_nan(self):
        pytest.importorskip("torch")
        rng = np.random.RandomState(0)
        x = rng.rand(3, 32, 32).astype(np.float32) * 255.0
        composed = compose_transforms(["hsv_jitter", "cutout"])
        out, y = composed(x, None)
        assert out.shape == x.shape
        assert not np.any(np.isnan(out))

    def test_unknown_primitive_raises(self):
        with pytest.raises(ValueError):
            compose_transforms(["unknown_name"])


class TestTransformRegistry:
    """TRANSFORM_REGISTRY：6 个原语。"""

    def test_registry_has_6_primitives(self):
        assert len(TRANSFORM_REGISTRY) == 6

    def test_get_transform(self):
        assert get_transform("hsv_jitter") is hsv_jitter
        assert get_transform("nonexistent") is None

    def test_list_transforms_sorted(self):
        names = list_transforms()
        assert len(names) == 6
        assert names == sorted(names)


# ============================================================
# 技术目录
# ============================================================
class TestDetectionCatalog:
    """detection 场景技术目录。"""

    def test_list_techniques_count(self):
        assert len(list_techniques()) == 6

    def test_list_categories(self):
        assert list_categories() == ["augmentation", "bbox_processing"]

    def test_is_augment(self):
        assert is_augment("hsv_jitter") is True
        assert is_augment("bbox_clip") is False

    def test_suggest_pipeline_excludes_augment(self):
        pipeline = suggest_pipeline("dummy_box")
        assert len(pipeline) > 0
        for name in pipeline:
            assert is_augment(name) is False

    def test_suggest_augment_only_augment(self):
        augment = suggest_augment("dummy_box")
        assert len(augment) > 0
        for name in augment:
            assert is_augment(name) is True
