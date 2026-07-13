"""集成测试 fixtures。

提供共享 fixtures：
- _register_test_scene：注册测试专用 generic_test 场景（session 级，autouse）
- synthetic_csv：生成合成 CSV（3 类，60 样本，10 特征）
- experiment_config：构造最小 ExperimentConfig（epochs=1, batch_size=4）

背景：GenericContainer.get_dataset_info 要求显式 root 参数，
但 pipeline 的 stage_resolve 通过 scene_kwargs 只传 params。
_TestableGenericContainer 从 params.data_root 提取 root，使 generic 场景能跑通完整 pipeline。
"""
import sys
from pathlib import Path

# bootstrap：senseframe 可导入前的必要本地推导
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

# 单一数据源：bootstrap 后从 senseframe.common.paths 导入 PROJECT_ROOT
from senseframe.common.paths import PROJECT_ROOT  # noqa: E402

import csv as _csv
import random as _random

import pytest

from senseframe.scenes import has_scene, register_scene
from senseframe.scenes.generic.container import GenericContainer

# Framework compat shim：senseframe/runner.py 向后兼容 re-export 缺失。
# hpo._default_objective 用 `from ..runner import run_experiment`（即 senseframe.runner），
# 但 Phase R5 重构后顶层 runner.py 未保留。在测试侧 alias engine.runner → senseframe.runner，
# 不修改框架源码。try/except 保护：torch 未装时跳过（HPO 测试本就 importorskip torch）。
try:
    import senseframe.engine.runner as _engine_runner
    sys.modules.setdefault("senseframe.runner", _engine_runner)
except Exception:
    pass


class _TestableGenericContainer(GenericContainer):
    """测试用 GenericContainer 子类。

    原始 GenericContainer.get_dataset_info 要求显式 root 参数（位置/关键字），
    但 pipeline 的 stage_resolve 调用 scene.get_dataset_info(dataset, **scene_kwargs)，
    其中 scene_kwargs = {"params": {...}}，不包含 root，导致 generic 场景无法跑通完整 pipeline。

    本子类从 params.data_root 提取 root，桥接此差异。
    注册为独立场景名 "generic_test"，不污染全局 "generic" 注册。
    """

    def meta(self):
        # 复用父类 meta，仅修改 name 以匹配注册名
        from senseframe.scenes.base import SceneMeta
        base = super().meta()
        return SceneMeta(
            name="generic_test",
            supported_tasks=base.supported_tasks,
            supported_models=base.supported_models,
            supported_datasets=base.supported_datasets,
            input_shape_hint=base.input_shape_hint,
            requires_custom_dataloader=base.requires_custom_dataloader,
            is_dynamic_dataset=base.is_dynamic_dataset,
        )

    def get_dataset_info(self, dataset_name, **kwargs):
        # 优先用显式 root；否则从 params.data_root 提取（pipeline 路径）
        root = kwargs.get("root")
        if root is None:
            params = kwargs.get("params") or {}
            root = params.get("data_root")
        if root is None:
            raise ValueError(
                "_TestableGenericContainer.get_dataset_info 需要 root 或 params.data_root"
            )
        return super().get_dataset_info(dataset_name, root=root)


@pytest.fixture(scope="session", autouse=True)
def _register_test_scene():
    """注册测试用 generic_test 场景（session 级，仅注册一次）。"""
    if not has_scene("generic_test"):
        register_scene("generic_test", _TestableGenericContainer())
    yield


def _write_synthetic_csv(csv_path, n_samples=60, n_features=10, n_classes=3, seed=42):
    """生成合成 CSV：特征与标签有相关性（features[0] += label*2），便于模型学到东西。"""
    rng = _random.Random(seed)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([f"f{i}" for i in range(n_features)] + ["label"])
        for _ in range(n_samples):
            features = [rng.gauss(0, 1) for _ in range(n_features)]
            label = rng.randint(0, n_classes - 1)
            # 让特征与标签相关，避免纯随机导致 val_acc 恒为 0（numerical_instability）
            features[0] += label * 2
            writer.writerow([f"{x:.6f}" for x in features] + [label])


@pytest.fixture
def synthetic_csv(tmp_path):
    """生成合成 CSV 文件并返回路径。

    数据集名固定为 "synthetic"（与 experiment_config 的 scene.dataset 一致），
    3 类、60 样本、10 特征。

    同时创建空目录 tmp_path/synthetic/ 以满足 preflight_check 的数据集目录存在性检查
    （preflight 假设数据集是目录，但 GenericContainer.load_dataset 优先匹配 CSV 文件）。
    """
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)
    # preflight_check 会检查 data_root/{dataset_name} 目录是否存在
    # GenericContainer.load_dataset 优先用 CSV 文件模式，空目录不影响数据加载
    (tmp_path / "synthetic").mkdir(exist_ok=True)
    return csv_path


@pytest.fixture
def experiment_config(tmp_path, synthetic_csv):
    """构造最小 ExperimentConfig（epochs=1, batch_size=4，快速跑通）。

    使用 generic_test 场景 + 合成 CSV。
    scene.data_root 与 scene.params.data_root 均指向 tmp_path
    （前者供 stage_load.load_dataset 使用，后者供 _TestableGenericContainer.get_dataset_info 使用）。
    """
    from senseframe.engine.config import (
        ExperimentConfig,
        InputFeature,
        OutputFeature,
        SceneConfig,
        TrainerConfig,
    )

    data_root = str(synthetic_csv.parent)
    config = ExperimentConfig(
        scene=SceneConfig(
            name="generic_test",
            dataset="synthetic",
            model_id="GenericMLP",
            learning_mode="supervised",
            data_root=data_root,
            params={"data_root": data_root},
        ),
        input_features=[InputFeature(name="features", type="tabular", shape=[10])],
        output_features=[OutputFeature(name="label", type="category", num_classes=3)],
        trainer=TrainerConfig(
            epochs=1,
            batch_size=4,
            enable_progress_bar=False,
            logger="csv",
        ),
        output_dir=str(tmp_path / "output"),
    )
    return config
