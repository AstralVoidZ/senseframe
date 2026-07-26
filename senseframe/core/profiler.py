"""
RFC Phase B：数据驱动决策 — DataProfiler 原语。

DataProfiler 探查数据特征，输出数据画像（DataProfile），
作为 Agent 策略决策的输入（task_type/loss/metric/model 推荐）。

设计原则（RFC 原则 2）：
- 策略选择必须基于数据特征，而非预设常量
- 框架"看数据"而非"查注册表挑候选"

DataProfile 包含：
- 基础统计：样本量、特征维度、类别数、类别分布
- 分布特征：缺失率、数值范围、均值/方差
- 结构特征：空间性（图像）、时序性（序列）、模态
- 推荐策略：基于画像推荐 task_type/loss/metric/model（非强制，可被 Agent 覆盖）

缓存：结果可缓存到 {output_dir}/data_profile.json，避免重复计算。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


def _count_dtypes(dtypes: Dict[str, str]) -> Dict[str, int]:
    """统计 dtype 分布。"""
    from collections import Counter
    return dict(Counter(dtypes.values()))


# P1-3：高维特征展开阈值。超过此值时 per-feature dict 折叠为 "_all" 共享原型，
# 避免 684K 维 CSI 数据生成 ~93MB 冗余 JSON。
# 阈值选择 10000：覆盖典型表格数据集（UCI 成人 14K 行 × 100+ 列 ≈ 1.4M cells，
# 但 feature 数通常 < 1000），同时对高维信号（CSI 22.5K / 图像 3072 / 音频 64000）
# 触发折叠。下游消费者通过 "_all" 键识别折叠模式。
_FEATURE_EXPANSION_THRESHOLD = 10000


@dataclass
class DataProfile:
    """数据画像：DataProfiler 的输出。

    Agent 基于此画像决策 task_type/loss/metric/model。
    """
    # 基础统计
    n_samples: int = 0
    input_shape: Tuple[int, ...] = ()
    n_features: int = 0
    n_classes: Optional[int] = None
    class_distribution: Dict[str, int] = field(default_factory=dict)
    # 根因修复（P2）：类别不平衡比率 = max(counts)/max(min(counts),1)。
    # 自监督模式下无标签 → class_distribution 为空 → 此字段为 None。
    # 消费者（stage_load OTel 埋点）必须做 None 守卫，避免对 None 求值。
    imbalance_ratio: Optional[float] = None

    # 分布特征
    missing_rate: float = 0.0
    value_range: Tuple[float, float] = (0.0, 1.0)
    mean: float = 0.0
    std: float = 1.0

    # 结构特征
    is_spatial: bool = False  # 图像/空间数据（input_shape len >= 3，含 H/W）
    is_temporal: bool = False  # 时序数据（input_shape len >= 2，含 sequence_length）
    modality: str = "unknown"  # csi/image/text/tabular/sequence/audio

    # 推荐策略（非强制，Agent 可覆盖）
    recommended_task_type: str = "classification"
    recommended_loss: str = "cross_entropy"
    recommended_metrics: List[str] = field(default_factory=lambda: ["accuracy", "macro_f1"])
    recommended_normalization: str = "zscore"  # none/zscore/minmax
    # P3-6：类别权重推荐（imbalance_ratio > 5 时自动计算 inverse frequency 权重）。
    # resolver 自动注入到 cross_entropy_weighted 的 loss_kwargs["weights"]。
    # None 表示数据平衡或非分类任务，不注入权重。
    recommended_class_weights: Optional[List[float]] = None

    # 元信息
    dataset_name: str = ""
    # RFC-003 DSP-4：结构深化
    dtypes: Dict[str, str] = field(default_factory=dict)           # 特征名 → dtype 字符串
    feature_names: List[str] = field(default_factory=list)          # 特征名列表（有序）
    nullable: Dict[str, bool] = field(default_factory=dict)         # 特征名 → 是否允许缺失
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict) # 特征名 → 形状
    profile_source: str = ""  # "train" / "test" / "full"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["value_range"] = list(self.value_range)
        # RFC-003 DSP-4：shapes 的 tuple 值转 list
        d["shapes"] = {k: list(v) for k, v in self.shapes.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataProfile":
        d = {**d}
        d["input_shape"] = tuple(d.get("input_shape", ()))
        d["value_range"] = tuple(d.get("value_range", (0.0, 1.0)))
        # RFC-003 DSP-4：兼容旧数据（无新字段时用默认值）
        d["shapes"] = {k: tuple(v) for k, v in d.get("shapes", {}).items()}
        return cls(**d)

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """返回字段结构（RFC-003 DSP-4）。"""
        return {
            "schema_version": "1.0.0",
            "fields": [
                {"name": "dtypes", "type": "Dict[str, str]"},
                {"name": "feature_names", "type": "List[str]"},
                {"name": "nullable", "type": "Dict[str, bool]"},
                {"name": "shapes", "type": "Dict[str, Tuple[int, ...]]"},
            ],
        }

    def describe(self) -> Dict[str, Any]:
        """返回当前画像摘要（RFC-003 DSP-4）。"""
        return {
            # P1-3 修复：用 self.n_features 而非 len(self.feature_names)。
            # 高维折叠时 feature_names 为空列表（用 "_all" 键存共享原型），
            # len() 返回 0，但 n_features 仍为原始数量（如 684000）。
            "n_features": self.n_features,
            "dtype_distribution": _count_dtypes(self.dtypes),
            "nullable_ratio": sum(1 for v in self.nullable.values() if v) / max(len(self.nullable), 1),
        }

    def save(self, path: str | Path) -> None:
        """缓存数据画像到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "DataProfile":
        """从 JSON 文件加载缓存的数据画像。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class DataProfiler:
    """数据探查器：探查 Dataset/DatasetBundle，输出 DataProfile。

    用法：
        profiler = DataProfiler()
        profile = profiler.profile_dataset(train_dataset, dataset_name="my_data")
        profile.save("output/data_profile.json")

        # Agent 基于 profile 决策
        task_type = profile.recommended_task_type
    """

    def __init__(self, max_samples: int = 1000):
        """
        Args:
            max_samples: 探查时最多采样多少样本（避免全量遍历大数据集）
        """
        self.max_samples = max_samples

    def profile_dataset(
        self,
        dataset,
        dataset_name: str = "",
        profile_source: str = "train",
        modality_hint: Optional[str] = None,
    ) -> DataProfile:
        """探查 torch Dataset，输出数据画像。

        Args:
            dataset: torch Dataset（支持 __getitem__ 返回 (x, y) 或 (x,)）
            dataset_name: 数据集名称（用于元信息）
            profile_source: 数据来源标记
            modality_hint: 场景显式声明的数据模态（如 "csi"），非 None 时覆盖 shape 启发式。
                           P0 修复：CSI (1,250,90) 与 image (1,H,W) 在 shape 上不可区分，
                           需场景通过 SceneMeta.modality 显式声明。

        Returns:
            DataProfile 数据画像
        """
        n_total = len(dataset) if hasattr(dataset, "__len__") else 0
        n_sample = min(n_total, self.max_samples) if n_total > 0 else 0

        if n_sample == 0:
            return DataProfile(dataset_name=dataset_name, profile_source=profile_source)

        # 采样
        samples_x: List[np.ndarray] = []
        samples_y: List[Any] = []
        indices = np.linspace(0, max(n_total - 1, 0), n_sample, dtype=int)
        for idx in indices:
            try:
                item = dataset[int(idx)]
                if isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        samples_x.append(self._to_numpy(item[0]))
                        samples_y.append(item[1])
                    elif len(item) == 1:
                        samples_x.append(self._to_numpy(item[0]))
                else:
                    samples_x.append(self._to_numpy(item))
            except Exception:
                continue

        if not samples_x:
            return DataProfile(dataset_name=dataset_name, profile_source=profile_source)

        # 基础统计
        first_x = samples_x[0]
        input_shape = first_x.shape
        n_features = int(np.prod(input_shape)) if input_shape else 0

        # 类别统计
        n_classes = None
        class_dist: Dict[str, int] = {}
        # 根因修复（P2）：imbalance_ratio 从 class_distribution 派生；
        # 自监督模式无标签 → class_dist 为空 → imbalance_ratio 保持 None。
        imbalance_ratio: Optional[float] = None
        if samples_y:
            try:
                y_arr = np.array(samples_y)
                unique, counts = np.unique(y_arr, return_counts=True)
                n_classes = len(unique)
                class_dist = {str(int(u)): int(c) for u, c in zip(unique, counts)}
                if counts is not None and len(counts) > 0:
                    imbalance_ratio = float(max(counts)) / max(float(min(counts)), 1.0)
            except Exception:
                pass

        # 分布特征
        all_x = np.stack([x.flatten() for x in samples_x]) if samples_x else np.array([])
        missing_rate = float(np.isnan(all_x).mean()) if all_x.size > 0 else 0.0
        finite_mask = np.isfinite(all_x) if all_x.size > 0 else np.array([])
        finite_x = all_x[finite_mask] if finite_mask.any() else all_x
        value_min = float(finite_x.min()) if finite_x.size > 0 else 0.0
        value_max = float(finite_x.max()) if finite_x.size > 0 else 1.0
        mean = float(finite_x.mean()) if finite_x.size > 0 else 0.0
        std = float(finite_x.std()) if finite_x.size > 0 else 1.0

        # 结构特征推断
        is_spatial = len(input_shape) >= 3  # (C, H, W) 或更高
        is_temporal = len(input_shape) >= 2 and not is_spatial  # (C, T) 或 (T, F)
        # P0 修复：modality_hint 非 None 时覆盖 shape 启发式
        # 场景通过 SceneMeta.modality 显式声明，优先级高于 shape 启发式
        modality = self._infer_modality(
            input_shape, is_spatial, is_temporal, modality_hint=modality_hint,
        )
        # CSI 模态修正：CSI 数据是时序的（time × subcarrier），非空间
        if modality == "csi":
            is_spatial = False
            is_temporal = True

        # 推荐策略
        rec_task, rec_loss, rec_metrics, rec_norm, rec_weights = self._recommend(
            n_classes=n_classes,
            input_shape=input_shape,
            is_spatial=is_spatial,
            is_temporal=is_temporal,
            missing_rate=missing_rate,
            class_dist=class_dist,
            value_range=(value_min, value_max),
        )

        # RFC-003 DSP-4：结构深化 — 计算 dtypes/feature_names/nullable/shapes
        # P1-3 修复：高维数据（如 CSI 684000 维）按 feature 展开 4 个 dict 会产生
        # ~93MB JSON（每条目内容相同却重复 684K 次）。引入阈值折叠：超过阈值时
        # 用 "_all" 键存共享原型 + n_features 记录原始数量，避免冗余展开。
        # 同质数据（典型场景：所有 feature 共享 dtype/shape/nullable）折叠为单条目，
        # 异质数据（罕见）仍按 feature 展开。
        if input_shape:
            n_feat = int(np.prod(input_shape)) if input_shape else 0
            first_dtype = str(first_x.dtype) if hasattr(first_x, 'dtype') else "unknown"
            first_nullable = bool(missing_rate > 0)
            if n_feat > _FEATURE_EXPANSION_THRESHOLD:
                # 高维折叠：用 "_all" 键存共享原型，feature_names 仅存数量
                feature_names_list = []
                dtypes_dict = {"_all": first_dtype}
                nullable_dict = {"_all": first_nullable}
                shapes_dict = {"_all": tuple(input_shape)}
            else:
                feature_names_list = [f"feature_{i}" for i in range(n_feat)]
                dtypes_dict = {name: first_dtype for name in feature_names_list}
                nullable_dict = {name: first_nullable for name in feature_names_list}
                shapes_dict = {name: tuple(input_shape) for name in feature_names_list}
        else:
            feature_names_list = []
            dtypes_dict = {}
            nullable_dict = {}
            shapes_dict = {}

        return DataProfile(
            n_samples=n_total,
            input_shape=input_shape,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
            missing_rate=missing_rate,
            value_range=(value_min, value_max),
            mean=mean,
            std=std,
            is_spatial=is_spatial,
            is_temporal=is_temporal,
            modality=modality,
            recommended_task_type=rec_task,
            recommended_loss=rec_loss,
            recommended_metrics=rec_metrics,
            recommended_normalization=rec_norm,
            recommended_class_weights=rec_weights,
            dataset_name=dataset_name,
            dtypes=dtypes_dict,
            feature_names=feature_names_list,
            nullable=nullable_dict,
            shapes=shapes_dict,
            profile_source=profile_source,
        )

    def profile_bundle(
        self,
        bundle,
        dataset_name: str = "",
        modality_hint: Optional[str] = None,
        learning_mode: str = "supervised",
        scene: Optional[Any] = None,
    ) -> DataProfile:
        """探查 DatasetBundle，按学习模式选择采样源。

        P1 修复：旧实现恒用 bundle.train（fallback test），但自监督模式下
        bundle.train 为 None（按 filling_rule 为 forbidden），回退到 test 集
        做画像采样——normalization 统计量来自测试集造成数据泄露，且分布
        与实际训练集（unsupervised）不一致。改为按 learning_mode 选择：
        - supervised → train 集（无 train 时回退 test）
        - self_supervised → unsupervised 集（无 unsupervised 时回退 test）
        确保画像统计量来自训练集，避免数据泄露。

        P0-A 修复（2026-07-18）：自监督模式下分两路采样：
        - distribution 统计（mean/std/value_range/input_shape/modality/n_samples/missing_rate）
          ← bundle.unsupervised（encoder 预训练数据分布，避免 test 泄露）
        - task 统计（n_classes/class_distribution/imbalance_ratio/recommended_class_weights
          /recommended_task_type/recommended_loss/recommended_metrics）
          ← bundle.supervised_finetune（Stage 2 微调任务的真实分类空间）
        旧实现把两类统计合并到 unsupervised 集采样，导致 n_classes=6（NTU-Fi_HAR 6 类
        动作标签）而非 14（NTU-Fi-HumanID 14 类身份标签），class_weights 维度不匹配
        14 类 CrossEntropy，且 Agent 基于 data_profile.n_classes=6 错误决策。

        P1 改进（2026-07-26）：新增 scene 参数，支持从 scene.meta.modality 自动继承
        modality_hint。Pipeline 内部 stage_load 已从 ctx.meta.modality 传入，此参数
        主要服务直接调用 profile_bundle 的场景（如 notebook 探索），减少必须显式
        传入 modality_hint 的认知负担。modality_hint 显式传入时优先级最高。

        Args:
            bundle: DatasetBundle（含 train/test/val/unsupervised/supervised_finetune）
            dataset_name: 数据集名称
            modality_hint: 场景显式声明的数据模态（如 "csi"），覆盖 shape 启发式。
                           None 时尝试从 scene.meta.modality 继承。
            learning_mode: "supervised" 或 "self_supervised"，决定采样源
            scene: 可选的场景容器实例，modality_hint 为 None 时从 scene.meta.modality 读取

        Returns:
            DataProfile 数据画像
        """
        # P1 改进：modality_hint 未显式传入时，从 scene.meta.modality 自动继承。
        # Pipeline 内部 stage_load 已显式传入 modality_hint，此分支主要服务直接调用场景。
        # 注意：LazyWiFiCSIContainer 的 meta 是方法（需调用），而非属性。
        if modality_hint is None and scene is not None:
            _scene_meta = getattr(scene, "meta", None)
            if _scene_meta is not None:
                if callable(_scene_meta):
                    _scene_meta = _scene_meta()
                modality_hint = getattr(_scene_meta, "modality", None)
        if learning_mode == "self_supervised":
            # P5 P2-3：自监督模式必须从 unsupervised 集采样画像。
            # 旧代码回退到 test 集，导致 normalization 常量来自测试集（数据泄露）。
            ds = getattr(bundle, "unsupervised", None)
            if ds is None:
                raise ValueError(
                    f"profile_bundle: learning_mode='self_supervised' but "
                    f"bundle.unsupervised is None. 禁止回退到 test 集以避免数据泄露。"
                    f"dataset={dataset_name}"
                )
            profile_source = "unsupervised"
            # 先用 unsupervised 集做 distribution 统计
            profile = self.profile_dataset(
                ds, dataset_name=dataset_name, profile_source=profile_source,
                modality_hint=modality_hint,
            )
            # P0-A：再用 supervised_finetune 集覆盖 task 统计
            # n_classes/class_distribution/imbalance_ratio/recommended_class_weights
            # 必须来自 Stage 2 微调任务数据，而非 unsupervised 预训练数据。
            # unsupervised 集的 label 是预训练数据集的原生标签（如 NTU-Fi_HAR 6 类动作），
            # 与下游 finetune 任务的真实分类空间（如 NTU-Fi-HumanID 14 类身份）无关。
            task_ds = getattr(bundle, "supervised_finetune", None)
            if task_ds is not None:
                task_profile = self.profile_dataset(
                    task_ds, dataset_name=dataset_name,
                    profile_source="supervised_finetune",
                    modality_hint=modality_hint,
                )
                # 仅覆盖 task 级字段，保留 distribution 级字段来自 unsupervised
                profile.n_classes = task_profile.n_classes
                profile.class_distribution = task_profile.class_distribution
                profile.imbalance_ratio = task_profile.imbalance_ratio
                profile.recommended_class_weights = task_profile.recommended_class_weights
                profile.recommended_task_type = task_profile.recommended_task_type
                profile.recommended_loss = task_profile.recommended_loss
                profile.recommended_metrics = task_profile.recommended_metrics
                # P1-C 修复：input_shape 应来自 supervised_finetune（单样本形状），
                # 而非 unsupervised ConcatDataset（拼接后矩阵形状）。
                # 旧逻辑用 unsupervised 的 (342, 2000) 而非 spec 声明的 (3, 114, 500)，
                # Agent 基于错误 input_shape 做模型选择。
                # 注：P1-a 修复后，_apply_spec_shape 会用注册表 DatasetSpec.input_shape
                # 进一步覆盖（优先级最高），此处作为未注册数据集的兜底。
                if task_profile.input_shape:
                    profile.input_shape = task_profile.input_shape
                    profile.n_features = task_profile.n_features
                    profile.is_spatial = task_profile.is_spatial
                    profile.is_temporal = task_profile.is_temporal
            return self._apply_spec_shape(profile, dataset_name)
        else:
            ds = getattr(bundle, "train", None)
            if ds is None:
                raise ValueError(
                    f"profile_bundle: bundle.train is None. "
                    f"dataset={dataset_name}"
                )
            profile_source = "train"
        return self._apply_spec_shape(
            self.profile_dataset(
                ds, dataset_name=dataset_name, profile_source=profile_source,
                modality_hint=modality_hint,
            ),
            dataset_name,
        )

    def _apply_spec_shape(self, profile: "DataProfile", dataset_name: str) -> "DataProfile":
        """P1-a 修复：用 DatasetSpec 声明的变换后 input_shape 覆盖 profiler 采样值。

        根因：profiler 直接采样 CSIDataset.__getitem__ 返回原始 .mat 形状（如 (342, 2000)），
        而 transform（stride + reshape）在 stage_build 才注入 GenericDataModule。注册表
        DatasetSpec.input_shape 声明的是变换后形状（如 (3, 114, 500)），是模型实际接收的
        权威形状，应覆盖采样值。仅对已注册数据集覆盖；未注册（generic/CustomContainer）
        保留采样值（前序 P1-C 的 supervised_finetune 覆盖作为兜底）。

        Args:
            profile: 待修正的数据画像
            dataset_name: 数据集名称（用于查注册表）

        Returns:
            修正后的 DataProfile（input_shape/n_features/is_spatial/is_temporal 已覆盖）
        """
        if not dataset_name:
            return profile
        try:
            from ..registry import is_dataset_registered, get_dataset_spec
            if is_dataset_registered(dataset_name):
                spec = get_dataset_spec(dataset_name)
                spec_shape = getattr(spec, "input_shape", None)
                if spec_shape:
                    profile.input_shape = tuple(spec_shape)
                    profile.n_features = int(np.prod(spec_shape))
                    profile.is_spatial = len(spec_shape) >= 3
                    profile.is_temporal = len(spec_shape) >= 2
        except Exception as e:
            _logger.debug("spec input_shape override skipped for %s: %s", dataset_name, e)
        return profile

    def _to_numpy(self, x) -> np.ndarray:
        """转换为 numpy array。"""
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _infer_modality(
        self,
        input_shape: Tuple[int, ...],
        is_spatial: bool,
        is_temporal: bool,
        modality_hint: Optional[str] = None,
    ) -> str:
        """推断数据模态。

        P5 P1-D：消除 shape-based fallback。旧代码在 modality_hint 为 None/unknown 时
        回退到 shape 启发式（3D→image, 2D→sequence, 1D→tabular），但 CSI (1,250,90)
        与 image (1,H,W) 在 shape 上不可区分，导致 CSI 被误判为 image。

        现在强制要求场景通过 SceneMeta.modality 显式声明模态，未声明时 raise。
        """
        if modality_hint is not None and modality_hint != "unknown":
            return modality_hint
        raise ValueError(
            f"modality_hint is required (got {modality_hint!r}). "
            f"Scene must declare modality via SceneMeta.modality "
            f"(e.g., 'csi', 'image', 'sequence', 'tabular'). "
            f"Shape-based inference is unreliable: CSI (1,250,90) and "
            f"image (1,H,W) are indistinguishable by shape alone. "
            f"If calling profile_bundle directly, pass modality_hint explicitly; "
            f"if using Pipeline, ensure SceneMeta.modality is set in the scene container."
        )

    def _recommend(
        self,
        n_classes: Optional[int],
        input_shape: Tuple[int, ...],
        is_spatial: bool,
        is_temporal: bool,
        missing_rate: float,
        class_dist: Dict[str, int],
        value_range: Tuple[float, float],
    ) -> Tuple[str, str, List[str], str, Optional[List[float]]]:
        """基于数据画像推荐策略（非强制，Agent 可覆盖）。

        推荐优先从注册表查询可用策略，回退到内置默认。

        Returns:
            (task_type, loss, metrics, normalization, class_weights)
        """
        from .task import has_task_type, get_task_type_default_loss, get_task_type_default_metrics
        from .losses import has_loss

        # 任务类型：有类别数 → 分类；否则 → 回归
        if n_classes is not None and n_classes > 1:
            task_type = "classification"
        else:
            task_type = "regression"

        # 从注册表获取默认 loss/metrics
        if has_task_type(task_type):
            loss = get_task_type_default_loss(task_type)
            metrics = get_task_type_default_metrics(task_type)
        else:
            loss = "cross_entropy"
            metrics = ["accuracy", "macro_f1"]

        # 归一化推荐：值域不在 [0,1] 且 std > 0 → zscore
        vmin, vmax = value_range
        if std := (vmax - vmin):
            if vmin < -0.1 or vmax > 1.1 or abs(vmax - vmin) > 10:
                norm = "zscore"
            else:
                norm = "minmax"
        else:
            norm = "none"

        # 类别不平衡 → 推荐 focal loss（Agent 可覆盖）
        # P3-6：同时自动计算 class_weights（inverse frequency weighting）
        class_weights = None
        if n_classes is not None and n_classes > 1 and class_dist:
            counts = list(class_dist.values())
            if counts:
                ratio = max(counts) / max(min(counts), 1)
                if ratio > 5 and has_loss("focal"):
                    loss = "focal"
                # P3-6：计算 inverse frequency 权重 w_i = N / (n_classes * count_i)
                # 与 recommended_loss 协同：ratio>5 时同时推荐 focal_loss + class_weights。
                # resolver 优先注入 class_weights 到 cross_entropy_weighted（当 loss 仍为
                # cross_entropy 时），focal_loss 自身不需 weights（内置 alpha 参数）。
                if ratio > 5:
                    total = sum(counts)
                    class_weights = [total / (len(counts) * c) for c in counts]

        return task_type, loss, metrics, norm, class_weights
