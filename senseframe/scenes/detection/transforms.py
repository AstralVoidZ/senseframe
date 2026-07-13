"""
目标检测图像增强 + bbox 处理原语库。

设计理念（RFC-002 阶段 U：场景包深度扩展）：
- 为 detection 场景提供图像增强 + bbox 处理原语
- 每个原语是独立可组合的 transform 函数
- Agent 可通过 get_transforms 的 pipeline 配置组合多个原语
- 也可通过 load_extension 生成自定义原语

原语分类：
- 图像增强：hsv_jitter, cutout, mixup, random_erasing
- bbox 处理：bbox_clip, bbox_flip

图像原语输入是 numpy 数组，shape (C,H,W) 或 (H,W,C)，统一按 (C,H,W) 处理。
bbox 原语输入是 numpy 数组，shape (N,4)，xyxy 格式。
mixup 签名：fn(x, y, alpha=0.2) -> (x_out, y_out)，与 compose_transforms 的
fn(x, y) -> (x, y) 签名兼容。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...common.transforms import ComposedTransform


# ============================================================
# 图像增强原语
# ============================================================
def _to_chw(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """将输入统一为 (C, H, W) 格式，返回 (数组, 是否转置过)。"""
    was_hwc = False
    if x.ndim == 3:
        # 启发式判断：最后一维是 1/3/4 时认为是 (H,W,C)
        if x.shape[-1] in (1, 3, 4) and x.shape[0] not in (1, 3, 4):
            x = np.transpose(x, (2, 0, 1))
            was_hwc = True
    return x, was_hwc


def _from_chw(x: np.ndarray, was_hwc: bool) -> np.ndarray:
    """如果原始输入是 (H,W,C)，转回。"""
    if was_hwc and x.ndim == 3:
        return np.transpose(x, (1, 2, 0))
    return x


def hsv_jitter(
    x: np.ndarray,
    hue: float = 0.1,
    saturation: float = 0.1,
    brightness: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """HSV 空间抖动增强。

    将 RGB 图像转到 HSV 空间，对 H/S/V 通道分别加随机噪声，再转回 RGB。

    Args:
        x: 输入图像，shape (C,H,W) 或 (H,W,C)，C=3
        hue: H 通道抖动幅度（0~1）
        saturation: S 通道抖动幅度（0~1）
        brightness: V 通道抖动幅度（0~1）
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        增强后的图像，shape 与输入一致
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    x_chw, was_hwc = _to_chw(x)
    if x_chw.ndim != 3 or x_chw.shape[0] != 3:
        # 非 RGB 三通道图像，直接返回（不做 HSV 变换）
        return x.copy()

    r = rng if rng is not None else np.random.default_rng()

    # 归一化到 [0, 1] 用于 HSV 转换
    h, w = x_chw.shape[1], x_chw.shape[2]
    rgb = np.clip(x_chw, 0, 255) / 255.0 if x_chw.max() > 1.0 else x_chw.copy()

    # RGB -> HSV（向量化）
    r_chan, g_chan, b_chan = rgb[0], rgb[1], rgb[2]
    mx = np.max(rgb, axis=0)
    mn = np.min(rgb, axis=0)
    diff = mx - mn

    v = mx
    s = np.where(mx > 0, diff / np.maximum(mx, 1e-10), 0.0)

    # 计算 H
    h_chan = np.zeros((h, w), dtype=np.float64)
    # R 最大
    mask_r = (mx == r_chan) & (diff > 0)
    h_chan[mask_r] = (60.0 * ((g_chan - b_chan) / np.maximum(diff, 1e-10)))[mask_r] % 360.0
    # G 最大
    mask_g = (mx == g_chan) & (diff > 0)
    h_chan[mask_g] = (60.0 * (2.0 + (b_chan - r_chan) / np.maximum(diff, 1e-10)))[mask_g] % 360.0
    # B 最大
    mask_b = (mx == b_chan) & (diff > 0)
    h_chan[mask_b] = (60.0 * (4.0 + (r_chan - g_chan) / np.maximum(diff, 1e-10)))[mask_b] % 360.0

    # 加噪声
    h_chan = (h_chan + r.uniform(-hue, hue, (h, w)) * 360.0) % 360.0
    s = np.clip(s + r.uniform(-saturation, saturation, (h, w)), 0.0, 1.0)
    v = np.clip(v + r.uniform(-brightness, brightness, (h, w)), 0.0, 1.0)

    # HSV -> RGB
    c = v * s
    h_sec = h_chan / 60.0
    x_val = c * (1 - np.abs(h_sec % 2 - 1))
    m = v - c

    rgb_out = np.zeros((3, h, w), dtype=np.float64)
    for i in range(6):
        mask = ((h_sec >= i) & (h_sec < i + 1))
        if i == 0:
            rgb_out[0][mask] = c[mask]; rgb_out[1][mask] = x_val[mask]
        elif i == 1:
            rgb_out[0][mask] = x_val[mask]; rgb_out[1][mask] = c[mask]
        elif i == 2:
            rgb_out[1][mask] = c[mask]; rgb_out[2][mask] = x_val[mask]
        elif i == 3:
            rgb_out[1][mask] = x_val[mask]; rgb_out[2][mask] = c[mask]
        elif i == 4:
            rgb_out[0][mask] = x_val[mask]; rgb_out[2][mask] = c[mask]
        else:
            rgb_out[0][mask] = c[mask]; rgb_out[2][mask] = x_val[mask]

    rgb_out += m
    rgb_out = np.clip(rgb_out, 0.0, 1.0)

    # 还原到原始值域
    if x_chw.max() > 1.0:
        rgb_out = rgb_out * 255.0

    return _from_chw(rgb_out, was_hwc)


def cutout(x: np.ndarray, n_holes: int = 1, length: int = 16, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """随机遮挡增强：在图像上随机选 n_holes 个矩形区域置零。

    Args:
        x: 输入图像，shape (C,H,W) 或 (H,W,C)
        n_holes: 遮挡区域数量（必须 >= 1）
        length: 遮挡区域边长（像素，必须 >= 1）
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        遮挡后的图像

    Raises:
        ValueError: n_holes < 1 或 length < 1
    """
    if n_holes < 1:
        raise ValueError(f"n_holes must be >= 1, got {n_holes}")
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    x_chw, was_hwc = _to_chw(x)
    if x_chw.ndim != 3:
        return x.copy()

    r = rng if rng is not None else np.random.default_rng()
    c, h, w = x_chw.shape
    result = x_chw.copy()
    for _ in range(n_holes):
        cy = r.integers(0, h)
        cx = r.integers(0, w)
        y1 = np.clip(cy - length // 2, 0, h)
        y2 = np.clip(cy + length // 2, 0, h)
        x1 = np.clip(cx - length // 2, 0, w)
        x2 = np.clip(cx + length // 2, 0, w)
        result[:, y1:y2, x1:x2] = 0.0

    return _from_chw(result, was_hwc)


def mixup(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """MixUp 增强：对样本及其标签做线性插值混合。

    输入 x 和 one-hot 标签 y，返回 mixup 后的 x 和 y。
    与 compose_transforms 的 fn(x, y) -> (x, y) 签名兼容。

    Args:
        x: 输入样本，shape (N, ...) 或单样本 (...)
        y: one-hot 标签，shape (N, num_classes) 或 (num_classes,)
        alpha: Beta 分布参数（0 时无混合）
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        (x_out, y_out)：混合后的样本与标签
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return x.copy(), y.copy()

    # 单样本：随机打乱自身维度不可行，退化为加噪
    if x.ndim == 1 or (x.ndim == 3 and x.shape[0] not in (x.shape[-1],)):
        # 视为单样本，直接返回（mixup 需要 batch）
        return x.copy(), y.copy()

    n = x.shape[0]
    if n < 2:
        return x.copy(), y.copy()

    if alpha <= 0:
        return x.copy(), y.copy()

    r = rng if rng is not None else np.random.default_rng()
    # 从 Beta(alpha, alpha) 采样混合系数
    lam = r.beta(alpha, alpha)
    # 随机打乱索引
    perm = r.permutation(n)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    # y 需要是 one-hot 才能线性插值
    if y.ndim == 1:
        # 整数标签转 one-hot
        num_classes = int(y.max()) + 1
        y_onehot = np.eye(num_classes)[y.astype(int)]
    else:
        y_onehot = y
    y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return x_mix, y_mix


def random_erasing(
    x: np.ndarray,
    area_ratio: Tuple[float, float] = (0.02, 0.2),
    min_aspect: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """随机擦除增强：随机选择一块区域置零（区域面积和长宽比随机）。

    与 cutout 类似，但擦除区域的面积比例和长宽比是随机的。

    Args:
        x: 输入图像，shape (C,H,W) 或 (H,W,C)
        area_ratio: 擦除区域面积占图像面积的比例范围 (min, max)
        min_aspect: 擦除区域最小长宽比
        rng: 可选的独立随机数生成器（P3 上策）

    Returns:
        擦除后的图像

    Raises:
        ValueError: area_ratio 边界不合法
    """
    lo, hi = area_ratio
    if not (0.0 < lo < hi <= 1.0):
        raise ValueError(
            f"area_ratio must satisfy 0 < lo < hi <= 1, got {area_ratio}"
        )
    if min_aspect <= 0:
        raise ValueError(f"min_aspect must be > 0, got {min_aspect}")

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    x_chw, was_hwc = _to_chw(x)
    if x_chw.ndim != 3:
        return x.copy()

    r = rng if rng is not None else np.random.default_rng()
    c, h, w = x_chw.shape
    img_area = float(h * w)
    # 最多尝试 10 次找到合法的擦除区域
    for _ in range(10):
        erase_area = img_area * r.uniform(lo, hi)
        aspect = r.uniform(min_aspect, 1.0 / min_aspect)
        eh = int(round(np.sqrt(erase_area * aspect)))
        ew = int(round(np.sqrt(erase_area / aspect)))
        if eh < 1 or ew < 1 or eh >= h or ew >= w:
            continue
        cy = r.integers(0, h - eh)
        cx = r.integers(0, w - ew)
        result = x_chw.copy()
        result[:, cy:cy + eh, cx:cx + ew] = 0.0
        return _from_chw(result, was_hwc)

    # 未找到合法区域，返回原图
    return _from_chw(x_chw, was_hwc)


# ============================================================
# bbox 处理原语
# ============================================================
def bbox_clip(boxes: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """裁剪 bbox 到图像边界内。

    Args:
        boxes: 边界框，shape (N, 4)，xyxy 格式
        img_size: 图像尺寸 (height, width)

    Returns:
        裁剪后的边界框，shape (N, 4)
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.copy()
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    h, w = img_size
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, w)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, h)
    return clipped


def bbox_flip(
    boxes: np.ndarray,
    img_width: int,
    flip_type: str = "horizontal",
) -> np.ndarray:
    """水平/垂直翻转 bbox 坐标。

    Args:
        boxes: 边界框，shape (N, 4)，xyxy 格式
        img_width: 图像宽度（水平翻转用）或高度（垂直翻转用）
        flip_type: "horizontal" 或 "vertical"

    Returns:
        翻转后的边界框，shape (N, 4)

    Raises:
        ValueError: flip_type 不合法
    """
    if flip_type not in ("horizontal", "vertical"):
        raise ValueError(
            f"flip_type must be 'horizontal' or 'vertical', got '{flip_type}'"
        )

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.copy()
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    flipped = boxes.copy()
    if flip_type == "horizontal":
        # 水平翻转：x 坐标关于 img_width 对称
        flipped[:, 0] = img_width - boxes[:, 2]
        flipped[:, 2] = img_width - boxes[:, 0]
    else:
        # 垂直翻转：y 坐标关于 img_width（此处为高度）对称
        flipped[:, 1] = img_width - boxes[:, 3]
        flipped[:, 3] = img_width - boxes[:, 1]
    return flipped


# ============================================================
# 原语注册表（供 catalog.py 和 get_transforms 使用）
# ============================================================
TRANSFORM_REGISTRY = {
    "hsv_jitter": hsv_jitter,
    "cutout": cutout,
    "mixup": mixup,
    "random_erasing": random_erasing,
    "bbox_clip": bbox_clip,
    "bbox_flip": bbox_flip,
}


def get_transform(name: str):
    """按名获取 transform 原语。"""
    return TRANSFORM_REGISTRY.get(name)


def list_transforms() -> list:
    """列出所有已注册的 transform 原语名。"""
    return sorted(TRANSFORM_REGISTRY.keys())


def compose_transforms(names: list, seed: Optional[int] = None, **kwargs) -> callable:
    """组合多个 transform 原语为单一函数。

    Args:
        names: 原语名列表，如 ["hsv_jitter", "cutout"]
        seed: 可选的随机种子（P3 上策，详见 wifi_csi.transforms.compose_transforms）
        **kwargs: 传递给每个原语的参数（按原语名分组）

    Returns:
        ComposedTransform 实例（callable，可 pickle 供 DataLoader multi-worker 使用）
    """
    transforms = []
    for name in names:
        fn = TRANSFORM_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown transform: {name}. Available: {list_transforms()}")
        transforms.append((name, fn))

    return ComposedTransform(transforms, kwargs, seed=seed)


__all__ = [
    "hsv_jitter",
    "cutout",
    "mixup",
    "random_erasing",
    "bbox_clip",
    "bbox_flip",
    "TRANSFORM_REGISTRY",
    "get_transform",
    "list_transforms",
    "compose_transforms",
]
