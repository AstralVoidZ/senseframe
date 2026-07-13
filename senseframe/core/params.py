"""
Phase 11.4 — SceneParams 正交化。

将场景参数由原始 dict 升级为 SceneParams 数据类，提供：
- 标准字段（target_frames / window_size / overlap_ratio / sampling_rate）
- 自定义字段（extra 透传）
- 向后兼容：支持从 dict 构造，提供 to_dict() 序列化
- dict-like 接口：__getitem__/__setitem__/__contains__/items，减少下游迁移成本

P5 P3-4 完整正交化（2026-07-13）：
- SceneConfig.params 类型从 Dict[str, Any] 收窄为 Optional[SceneParams]
- SceneParams 提供 dict-like 兼容层，下游 []/= /in/.get()/.items() 零改动
- 10 个标准字段为接口契约（即使无消费者也定义清楚），extra 承载场景特定扩展
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class SceneParams:
    """场景参数：标准化 + 可扩展。

    标准字段覆盖最常见的时序/窗口参数；未在此处定义的场景特定参数可放 extra。
    """

    # 时序/窗口相关
    target_frames: Optional[int] = None       # 目标帧数（信号下采样/上采样目标）
    window_size: Optional[int] = None         # 单样本窗口长度
    overlap_ratio: Optional[float] = None     # 窗口重叠比例（0.0~1.0）
    sampling_rate: Optional[int] = None        # 采样率（Hz）

    # 数据划分
    train_ratio: Optional[float] = None       # 训练集占比
    val_ratio: Optional[float] = None         # 验证集占比
    test_ratio: Optional[float] = None        # 测试集占比

    # 任务配置（Phase 11.1 透传）
    task_type: Optional[str] = None           # classification/regression/...
    loss: Optional[str] = None                # loss 名称
    metrics: Optional[list] = None            # metrics 列表

    # 自定义扩展
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metrics is not None and not isinstance(self.metrics, list):
            self.metrics = list(self.metrics)

    def get(self, key: str, default: Any = None) -> Any:
        """统一字段或 extra 字段取值。"""
        if key in self.__dataclass_fields__:
            val = getattr(self, key)
            if val is not None:
                return val
        return self.extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置标准字段（如有）或追加到 extra。"""
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SceneParams":
        """从 dict 构造；未知字段自动归入 extra。"""
        if d is None:
            return cls()
        if isinstance(d, SceneParams):
            return d
        standard = {f for f in cls.__dataclass_fields__}
        kwargs = {}
        extra = {}
        # 1) 先处理 "extra" 字段（如有），将其中所有键合并到 extra
        if "extra" in d and isinstance(d["extra"], dict):
            extra.update(d["extra"])
        # 2) 遍历所有键
        for k, v in d.items():
            if k == "extra":
                continue  # 已处理
            if k in standard:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in self.__dataclass_fields__
            if f != "extra"
        ) and len(self.extra) == 0

    def __bool__(self) -> bool:
        return not self.is_empty()

    # ============================================================
    # P5 P3-4：dict-like 兼容层
    # ============================================================
    # 让 SceneParams 兼容 [] / []= / in / .get() / .items() 等 dict 操作，
    # 下游消费方零改动。标准字段和 extra 统一访问。

    def __getitem__(self, key: str) -> Any:
        """支持下标取值：params["key"]。"""
        if key in self.__dataclass_fields__ and key != "extra":
            val = getattr(self, key)
            if val is not None:
                return val
            raise KeyError(key)
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """支持下标赋值：params["key"] = value。委托给 .set()。"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """支持 in 操作：key in params。"""
        if key in self.__dataclass_fields__ and key != "extra":
            return getattr(self, key) is not None
        return key in self.extra

    def items(self):
        """返回扁平 dict 的 items（标准字段非 None 值 + extra）。"""
        result = {}
        for f in self.__dataclass_fields__:
            if f == "extra":
                continue
            val = getattr(self, f)
            if val is not None:
                result[f] = val
        result.update(self.extra)
        return result.items()

    def keys(self):
        """返回扁平 dict 的 keys。"""
        return [k for k, _ in self.items()]

    def to_flat_dict(self) -> Dict[str, Any]:
        """返回扁平 dict（标准字段非 None 值 + extra 合并）。

        用于 JSON 序列化或需要 dict 形态的下游消费方。
        to_dict() 返回嵌套结构（含 extra 子 dict），to_flat_dict() 返回扁平结构。
        """
        return dict(self.items())
