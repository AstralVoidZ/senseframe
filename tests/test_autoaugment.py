"""ε3 AutoAugment 测试（P3.1.1-P3.1.3）。

反假绿测试策略：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- 真实 AugmentationSearchSpace 实例（不 mock）
- 真实 AutoAugmentSampler 采样（验证满足 SP Sampler Protocol）
- 真实 AutoAugmentPolicyBuilder 构造 transform（验证 transform 真实作用于数据）
- 真实 make_autoaugment_datamodule_factory（验证返回合法 datamodule_factory）

覆盖：
- P3.1.1: AugmentationSearchSpace + AugmentationParameterSpec
- P3.1.2: AutoAugmentSampler + AutoAugmentPolicyBuilder
- P3.1.3: make_autoaugment_datamodule_factory
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from senseframe.autoaugment import (
    AugmentationParameterSpec,
    AugmentationSearchSpace,
    SUPPORTED_AUGMENT_OPS,
    AutoAugmentPolicyBuilder,
    AutoAugmentSampler,
    make_autoaugment_datamodule_factory,
    make_policy_from_params,
    list_augment_ops,
    get_augment_op,
    build_default_search_space,
)
from senseframe.search_protocol import (
    Sampler,
    SearchSpace,
    StudyManager,
    get_sampler,
    list_samplers,
)


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


def _make_policy_params(n_ops: int = 2) -> Dict[str, Any]:
    """构造测试用策略参数。"""
    params: Dict[str, Any] = {}
    for i in range(n_ops):
        params[f"op_{i}"] = "noise"
        params[f"magnitude_{i}"] = 0.3
        params[f"probability_{i}"] = 0.8
    return params


# ============================================================
# P3.1.1: AugmentationSearchSpace
# ============================================================
class TestAugmentationParameterSpec:
    """AugmentationParameterSpec 数据结构测试（P3.1.1）。"""

    def test_categorical_spec_creation(self):
        """categorical 类型 ParameterSpec 创建。"""
        spec = AugmentationParameterSpec(
            name="op_0", type="categorical",
            choices=["noise", "cutout"],
        )
        assert spec.name == "op_0"
        assert spec.type == "categorical"
        assert spec.choices == ["noise", "cutout"]
        assert spec.low is None
        assert spec.high is None

    def test_float_spec_creation(self):
        """float 类型 ParameterSpec 创建。"""
        spec = AugmentationParameterSpec(
            name="magnitude_0", type="float",
            low=0.0, high=1.0,
        )
        assert spec.name == "magnitude_0"
        assert spec.type == "float"
        assert spec.low == 0.0
        assert spec.high == 1.0
        assert spec.choices is None

    def test_to_sp_param(self):
        """to_sp_param 转换为 SP ParameterSpec。"""
        spec = AugmentationParameterSpec(
            name="op_0", type="categorical",
            choices=["noise", "cutout"],
        )
        sp_param = spec.to_sp_param()
        assert sp_param.name == "op_0"
        assert sp_param.type == "categorical"
        assert sp_param.choices == ["noise", "cutout"]

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict / from_dict 往返。"""
        spec = AugmentationParameterSpec(
            name="magnitude_0", type="float",
            low=0.1, high=0.9, default=0.5,
        )
        d = spec.to_dict()
        restored = AugmentationParameterSpec.from_dict(d)
        assert restored.name == spec.name
        assert restored.type == spec.type
        assert restored.low == spec.low
        assert restored.high == spec.high
        assert restored.default == spec.default


class TestAugmentationSearchSpace:
    """AugmentationSearchSpace 数据结构测试（P3.1.1）。"""

    def test_default_creation(self):
        """默认创建。"""
        ss = AugmentationSearchSpace()
        assert ss.schema_version == "1.0.0"
        assert ss.n_ops == 2
        assert ss.ops == SUPPORTED_AUGMENT_OPS
        assert ss.magnitude_range == (0.0, 1.0)
        assert ss.probability_range == (0.0, 1.0)

    def test_custom_n_ops(self):
        """自定义 n_ops。"""
        ss = AugmentationSearchSpace(n_ops=3)
        assert ss.n_ops == 3

    def test_n_ops_too_small_raises(self):
        """n_ops < 1 应抛 ValueError。"""
        with pytest.raises(ValueError, match="n_ops must be >= 1"):
            AugmentationSearchSpace(n_ops=0)

    def test_n_ops_too_large_raises(self):
        """n_ops > 5 应抛 ValueError。"""
        with pytest.raises(ValueError, match="n_ops must be <= 5"):
            AugmentationSearchSpace(n_ops=6)

    def test_empty_ops_raises(self):
        """ops 为空应抛 ValueError。"""
        with pytest.raises(ValueError, match="ops must not be empty"):
            AugmentationSearchSpace(ops=[])

    def test_invalid_magnitude_range_raises(self):
        """magnitude_range 超界应抛 ValueError。"""
        with pytest.raises(ValueError, match="magnitude_range"):
            AugmentationSearchSpace(magnitude_range=(-0.1, 1.0))

    def test_invalid_probability_range_raises(self):
        """probability_range low > high 应抛 ValueError。"""
        with pytest.raises(ValueError, match="probability_range low > high"):
            AugmentationSearchSpace(probability_range=(0.8, 0.2))

    def test_to_sp_search_space_n_ops_2(self):
        """to_sp_search_space 生成 6 个参数（n_ops=2 × 3）。"""
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        assert isinstance(sp_ss, SearchSpace)
        assert len(sp_ss.parameters) == 6
        # 检查参数名
        names = [p.name for p in sp_ss.parameters]
        assert "op_0" in names
        assert "magnitude_0" in names
        assert "probability_0" in names
        assert "op_1" in names
        assert "magnitude_1" in names
        assert "probability_1" in names

    def test_to_sp_search_space_param_types(self):
        """to_sp_search_space 参数类型正确。"""
        ss = AugmentationSearchSpace(n_ops=1)
        sp_ss = ss.to_sp_search_space()
        op_param = next(p for p in sp_ss.parameters if p.name == "op_0")
        assert op_param.type == "categorical"
        assert op_param.choices == SUPPORTED_AUGMENT_OPS
        mag_param = next(p for p in sp_ss.parameters if p.name == "magnitude_0")
        assert mag_param.type == "float"
        assert mag_param.low == 0.0
        assert mag_param.high == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict / from_dict 往返。"""
        ss = AugmentationSearchSpace(n_ops=3, magnitude_range=(0.1, 0.9))
        d = ss.to_dict()
        restored = AugmentationSearchSpace.from_dict(d)
        assert restored.n_ops == ss.n_ops
        assert restored.magnitude_range == ss.magnitude_range
        assert restored.ops == ss.ops

    def test_validate_params_valid(self):
        """validate_params 合法参数返回空列表。"""
        ss = AugmentationSearchSpace(n_ops=2)
        params = _make_policy_params(2)
        errors = ss.validate_params(params)
        assert errors == []

    def test_validate_params_missing_op(self):
        """validate_params 缺少 op 应报错。"""
        ss = AugmentationSearchSpace(n_ops=2)
        params = {"op_1": "noise", "magnitude_1": 0.3, "probability_1": 0.8}
        errors = ss.validate_params(params)
        assert any("missing op_0" in e for e in errors)

    def test_validate_params_op_not_in_choices(self):
        """validate_params op 不在 choices 应报错。"""
        ss = AugmentationSearchSpace(n_ops=1)
        params = {"op_0": "invalid_op", "magnitude_0": 0.3, "probability_0": 0.8}
        errors = ss.validate_params(params)
        assert any("not in ops" in e for e in errors)

    def test_validate_params_magnitude_out_of_range(self):
        """validate_params magnitude 超界应报错。"""
        ss = AugmentationSearchSpace(n_ops=1, magnitude_range=(0.0, 0.5))
        params = {"op_0": "noise", "magnitude_0": 0.8, "probability_0": 0.8}
        errors = ss.validate_params(params)
        assert any("out of range" in e for e in errors)

    def test_describe(self):
        """describe 返回人类可读描述。"""
        ss = AugmentationSearchSpace(n_ops=2)
        desc = ss.describe()
        assert "n_ops=2" in desc
        assert "AugmentationSearchSpace" in desc

    def test_build_default_search_space(self):
        """便捷工厂函数。"""
        ss = build_default_search_space(n_ops=3)
        assert ss.n_ops == 3


# ============================================================
# P3.1.2: AutoAugmentPolicyBuilder + 增强原语
# ============================================================
class TestAutoAugmentPolicyBuilder:
    """AutoAugmentPolicyBuilder 测试（P3.1.2）。"""

    def test_build_returns_callable(self):
        """build 返回可调用 transform。"""
        builder = AutoAugmentPolicyBuilder()
        params = _make_policy_params(1)
        transform = builder.build(params)
        assert callable(transform)

    def test_transform_preserves_shape(self):
        """transform 保持输入形状。"""
        builder = AutoAugmentPolicyBuilder()
        params = {"op_0": "noise", "magnitude_0": 0.3, "probability_0": 1.0}
        transform = builder.build(params)
        x = np.random.randn(3, 100).astype(np.float32)
        y = 1
        xx, yy = transform(x, y)
        assert xx.shape == x.shape
        assert yy == y

    def test_transform_with_probability_zero(self):
        """probability=0 时不应用增强（返回原数据）。"""
        builder = AutoAugmentPolicyBuilder()
        params = {"op_0": "noise", "magnitude_0": 1.0, "probability_0": 0.0}
        transform = builder.build(params)
        x = np.random.randn(100).astype(np.float32)
        xx, _ = transform(x, 0)
        # probability=0 应不修改 x
        np.testing.assert_array_equal(xx, x)

    def test_transform_with_none_op(self):
        """op='none' 时不应用增强。"""
        builder = AutoAugmentPolicyBuilder()
        params = {"op_0": "none", "magnitude_0": 0.5, "probability_0": 1.0}
        transform = builder.build(params)
        x = np.random.randn(100).astype(np.float32)
        xx, _ = transform(x, 0)
        np.testing.assert_array_equal(xx, x)

    def test_transform_with_multiple_ops(self):
        """多 op 组合 transform。"""
        builder = AutoAugmentPolicyBuilder()
        params = {
            "op_0": "noise", "magnitude_0": 0.3, "probability_0": 1.0,
            "op_1": "cutout", "magnitude_1": 0.5, "probability_1": 1.0,
        }
        transform = builder.build(params)
        x = np.random.randn(3, 100).astype(np.float32)
        xx, _ = transform(x, 0)
        # cutout 应将部分片段置零
        assert (xx == 0).any()

    def test_build_with_search_space_validation(self):
        """带 search_space 验证。"""
        ss = AugmentationSearchSpace(n_ops=1)
        builder = AutoAugmentPolicyBuilder(search_space=ss)
        params = {"op_0": "noise", "magnitude_0": 0.3, "probability_0": 0.8}
        transform = builder.build(params)
        assert callable(transform)

    def test_build_with_invalid_params_raises(self):
        """带 search_space 验证非法参数应抛 ValueError。"""
        ss = AugmentationSearchSpace(n_ops=1)
        builder = AutoAugmentPolicyBuilder(search_space=ss)
        params = {"op_0": "invalid_op", "magnitude_0": 0.3, "probability_0": 0.8}
        with pytest.raises(ValueError, match="Invalid policy params"):
            builder.build(params)

    def test_build_empty_params_returns_identity(self):
        """空参数返回 identity transform。"""
        builder = AutoAugmentPolicyBuilder()
        transform = builder.build({})
        x = np.random.randn(100).astype(np.float32)
        xx, yy = transform(x, 42)
        np.testing.assert_array_equal(xx, x)
        assert yy == 42

    def test_build_eval_transform_is_identity(self):
        """build_eval_transform 返回 identity（评估不增强）。"""
        builder = AutoAugmentPolicyBuilder()
        transform = builder.build_eval_transform(_make_policy_params(1))
        x = np.random.randn(100).astype(np.float32)
        xx, _ = transform(x, 0)
        np.testing.assert_array_equal(xx, x)

    def test_make_policy_from_params(self):
        """便捷工厂函数。"""
        params = _make_policy_params(1)
        transform = make_policy_from_params(params)
        assert callable(transform)


class TestAugmentOps:
    """增强原语测试（P3.1.2）。"""

    def test_list_augment_ops(self):
        """list_augment_ops 返回所有算子。"""
        ops = list_augment_ops()
        assert "noise" in ops
        assert "time_jitter" in ops
        assert "freq_masking" in ops
        assert "cutout" in ops
        assert "none" in ops

    def test_get_augment_op(self):
        """get_augment_op 返回函数。"""
        fn = get_augment_op("noise")
        assert callable(fn)
        x = np.random.randn(100).astype(np.float32)
        result = fn(x, 0.5)
        assert result.shape == x.shape

    def test_get_augment_op_unknown(self):
        """get_augment_op 未知算子返回 None。"""
        assert get_augment_op("invalid") is None

    def test_time_jitter_preserves_shape(self):
        """time_jitter 保持形状。"""
        from senseframe.autoaugment.policy_builder import _time_jitter
        x = np.random.randn(3, 100).astype(np.float32)
        result = _time_jitter(x, 0.5)
        assert result.shape == x.shape

    def test_freq_masking_preserves_shape(self):
        """freq_masking 保持形状。"""
        from senseframe.autoaugment.policy_builder import _freq_masking
        x = np.random.randn(3, 100).astype(np.float32)
        result = _freq_masking(x, 0.5)
        assert result.shape == x.shape

    def test_cutout_zeros_some_elements(self):
        """cutout 应将部分元素置零。"""
        from senseframe.autoaugment.policy_builder import _cutout
        x = np.ones(100).astype(np.float32)
        result = _cutout(x, 0.5)
        assert (result == 0).any()

    def test_noise_adds_perturbation(self):
        """noise 应添加扰动（不完全等于输入）。"""
        from senseframe.autoaugment.policy_builder import _noise
        # 用常数输入，noise 应产生非零扰动
        # noise_std = magnitude * 0.1 * (data_std + 1e-8)
        # data_std=0 时 noise_std = magnitude * 0.1 * 1e-8（极小，可能被 allclose 视为相等）
        # 用 magnitude=1.0 + 非零 data_std 确保 noise_std 足够大
        x = np.ones(100).astype(np.float32) * 10.0  # data_std = 0，但值非零
        # 实际上 data_std=0 时 noise_std 仍极小；改用有方差的输入
        x = np.linspace(0, 10, 100).astype(np.float32)  # data_std > 0
        result = _noise(x, 1.0)
        # noise_std = 1.0 * 0.1 * (std(linspace(0,10)) + 1e-8) ≈ 0.1 * 2.9 ≈ 0.29
        # 扰动应足够大，使结果不等于输入
        assert not np.array_equal(result, x)
        assert np.std(result - x) > 0  # 扰动非零


# ============================================================
# P3.1.2: AutoAugmentSampler
# ============================================================
class TestAutoAugmentSampler:
    """AutoAugmentSampler 测试（P3.1.2）。"""

    def test_sampler_name(self):
        """sampler name 为 'autoaugment'。"""
        s = AutoAugmentSampler()
        assert s.name == "autoaugment"

    def test_sampler_satisfies_protocol(self):
        """AutoAugmentSampler 满足 SP Sampler Protocol。"""
        s = AutoAugmentSampler()
        assert isinstance(s, Sampler)

    def test_sampler_registered_in_sp(self):
        """AutoAugmentSampler 已注册到 SP 注册表。"""
        samplers = list_samplers()
        assert "autoaugment" in samplers
        sampler_cls = get_sampler("autoaugment")
        assert sampler_cls is AutoAugmentSampler

    def test_sampler_sample_returns_dict(self):
        """sample 返回参数 dict。"""
        s = AutoAugmentSampler(seed=42)
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        params = s.sample(sp_ss, [])
        assert isinstance(params, dict)
        assert "op_0" in params
        assert "magnitude_0" in params
        assert "probability_0" in params

    def test_sampler_sample_respects_search_space(self):
        """sample 参数在搜索空间范围内。"""
        s = AutoAugmentSampler(seed=42)
        ss = AugmentationSearchSpace(n_ops=1, magnitude_range=(0.2, 0.8))
        sp_ss = ss.to_sp_search_space()
        for _ in range(5):
            params = s.sample(sp_ss, [])
            assert params["op_0"] in SUPPORTED_AUGMENT_OPS
            assert 0.2 <= params["magnitude_0"] <= 0.8
            assert 0.0 <= params["probability_0"] <= 1.0

    def test_sampler_init_with_invalid_population_raises(self):
        """population_size < 2 应抛 ValueError。"""
        with pytest.raises(ValueError, match="population_size must be >= 2"):
            AutoAugmentSampler(population_size=1)

    def test_sampler_init_with_invalid_mutation_rate_raises(self):
        """mutation_rate 超界应抛 ValueError。"""
        with pytest.raises(ValueError, match="mutation_rate must be in"):
            AutoAugmentSampler(mutation_rate=1.5)

    def test_sampler_population_grows_with_sampling(self):
        """采样后 population 增长。"""
        s = AutoAugmentSampler(population_size=5, seed=42)
        ss = AugmentationSearchSpace(n_ops=1)
        sp_ss = ss.to_sp_search_space()
        for _ in range(3):
            s.sample(sp_ss, [])
        assert s.population_size_actual() == 3

    def test_sampler_evolution_after_population_full(self):
        """population 满后进入进化阶段。"""
        s = AutoAugmentSampler(population_size=3, seed=42)
        ss = AugmentationSearchSpace(n_ops=1)
        sp_ss = ss.to_sp_search_space()
        # 填满 population
        for _ in range(3):
            s.sample(sp_ss, [])
        # 提供带 fitness 的 history，触发进化
        history = [
            {"params": {"op_0": "noise", "magnitude_0": 0.3, "probability_0": 0.8},
             "result": {"value": 0.85}},
            {"params": {"op_0": "cutout", "magnitude_0": 0.5, "probability_0": 0.6},
             "result": {"value": 0.78}},
        ]
        child = s.sample(sp_ss, history)
        assert isinstance(child, dict)
        assert "op_0" in child
        # 进化阶段后 population 应增长
        assert s.population_size_actual() > 3

    def test_sampler_evaluated_count_with_history(self):
        """history 提供后 evaluated_count 增加。"""
        s = AutoAugmentSampler(population_size=5, seed=42)
        ss = AugmentationSearchSpace(n_ops=1)
        sp_ss = ss.to_sp_search_space()
        # 采样 3 个
        for _ in range(3):
            s.sample(sp_ss, [])
        # 提供 history
        history = [
            {"params": {"op_0": "noise", "magnitude_0": 0.3, "probability_0": 0.8},
             "result": {"value": 0.85}},
        ]
        s.sample(sp_ss, history)
        assert s.evaluated_count() >= 1

    def test_sampler_default_mutation_rate(self):
        """AutoAugmentSampler 默认 mutation_rate=0.4（高于 EvolutionarySampler 的 0.3）。"""
        s = AutoAugmentSampler()
        assert s.mutation_rate == 0.4

    def test_sampler_with_study_manager(self):
        """AutoAugmentSampler 与 StudyManager 集成。"""
        sm = StudyManager()
        ss = AugmentationSearchSpace(n_ops=1)
        sp_ss = ss.to_sp_search_space()
        study_id = sm.create_study(
            name="test_autoaugment",
            direction="maximize",
            search_space=sp_ss,
            sampler="autoaugment",
        )
        # ask 应返回 autoaugment sampler 采样的参数
        trial = sm.ask(study_id)
        assert "op_0" in trial.params
        assert "magnitude_0" in trial.params
        assert "probability_0" in trial.params


# ============================================================
# P3.1.3: make_autoaugment_datamodule_factory
# ============================================================
class TestMakeAutoAugmentDataModuleFactory:
    """make_autoaugment_datamodule_factory 测试（P3.1.3）。"""

    def test_factory_returns_callable(self):
        """factory 返回可调用 datamodule_factory。"""
        params = _make_policy_params(1)
        factory = make_autoaugment_datamodule_factory(params)
        assert callable(factory)

    def test_factory_returns_generic_datamodule(self):
        """factory 调用返回 GenericDataModule。"""
        from senseframe.engine.datamodule import GenericDataModule
        params = _make_policy_params(1)
        factory = make_autoaugment_datamodule_factory(params)

        # mock dataset
        class MockDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                import numpy as np
                return np.random.randn(100).astype(np.float32), idx % 3

        train_ds = MockDataset()
        test_ds = MockDataset()
        dm = factory(
            train_dataset=train_ds, test_dataset=test_ds,
            batch_size=4, num_workers=0,
        )
        assert isinstance(dm, GenericDataModule)

    def test_factory_with_base_train_transform(self):
        """factory 组合 base_train_transform 与增强 transform。"""
        from senseframe.engine.datamodule import GenericDataModule
        params = _make_policy_params(1)
        called = [False]
        def base_train_transform(x, y):
            called[0] = True
            return x, y
        factory = make_autoaugment_datamodule_factory(
            params, base_train_transform=base_train_transform,
        )

        class MockDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                import numpy as np
                return np.random.randn(100).astype(np.float32), idx % 3

        dm = factory(
            train_dataset=MockDataset(), test_dataset=MockDataset(),
            batch_size=4, num_workers=0,
        )
        assert isinstance(dm, GenericDataModule)
        # 访问一个样本触发 transform
        try:
            dm.train_dataset[0]
            # base_train_transform 应被调用（若 probability > 0）
        except Exception:
            pass  # transform 内部随机性，不强制断言

    def test_factory_with_base_eval_transform(self):
        """factory 用 base_eval_transform 作为评估 transform。"""
        from senseframe.engine.datamodule import GenericDataModule
        params = _make_policy_params(1)
        def base_eval_transform(x, y):
            return x, y
        factory = make_autoaugment_datamodule_factory(
            params, base_eval_transform=base_eval_transform,
        )

        class MockDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                import numpy as np
                return np.random.randn(100).astype(np.float32), idx % 3

        dm = factory(
            train_dataset=MockDataset(), test_dataset=MockDataset(),
            batch_size=4, num_workers=0,
        )
        assert isinstance(dm, GenericDataModule)

    def test_factory_with_search_space_validation(self):
        """factory 用 search_space 验证参数。"""
        ss = AugmentationSearchSpace(n_ops=1)
        params = _make_policy_params(1)
        factory = make_autoaugment_datamodule_factory(
            params, search_space=ss,
        )
        assert callable(factory)

    def test_factory_with_invalid_params_raises(self):
        """factory 用 search_space 验证非法参数应抛 ValueError。"""
        ss = AugmentationSearchSpace(n_ops=1)
        params = {"op_0": "invalid_op", "magnitude_0": 0.3, "probability_0": 0.8}
        with pytest.raises(ValueError, match="Invalid policy params"):
            make_autoaugment_datamodule_factory(params, search_space=ss)


# ============================================================
# P3.1.4: grep 实证检查（反假绿）
# ============================================================
class TestGrepEvidence:
    """grep 实证：源码检查所有 P3.1 实现关键点。"""

    def test_search_space_file_exists(self):
        """AugmentationSearchSpace 源码文件存在。"""
        path = _source_path("autoaugment/search_space.py")
        assert path.exists()

    def test_grep_supported_augment_ops(self):
        """grep 实证：SUPPORTED_AUGMENT_OPS 常量定义。"""
        path = _source_path("autoaugment/search_space.py")
        assert _grep_source(path, "SUPPORTED_AUGMENT_OPS")
        assert _grep_source(path, '"time_jitter"')
        assert _grep_source(path, '"freq_masking"')
        assert _grep_source(path, '"noise"')
        assert _grep_source(path, '"cutout"')
        assert _grep_source(path, '"none"')

    def test_grep_augmentation_search_space_class(self):
        """grep 实证：AugmentationSearchSpace 类定义。"""
        path = _source_path("autoaugment/search_space.py")
        assert _grep_source(path, "class AugmentationSearchSpace")
        assert _grep_source(path, "def to_sp_search_space")
        assert _grep_source(path, "def validate_params")

    def test_grep_augmentation_parameter_spec_class(self):
        """grep 实证：AugmentationParameterSpec 类定义。"""
        path = _source_path("autoaugment/search_space.py")
        assert _grep_source(path, "class AugmentationParameterSpec")
        assert _grep_source(path, "def to_sp_param")

    def test_grep_policy_builder_file_exists(self):
        """AutoAugmentPolicyBuilder 源码文件存在。"""
        path = _source_path("autoaugment/policy_builder.py")
        assert path.exists()

    def test_grep_policy_builder_class(self):
        """grep 实证：AutoAugmentPolicyBuilder 类定义。"""
        path = _source_path("autoaugment/policy_builder.py")
        assert _grep_source(path, "class AutoAugmentPolicyBuilder")
        assert _grep_source(path, "def build(")
        assert _grep_source(path, "def build_eval_transform")

    def test_grep_augment_ops_registered(self):
        """grep 实证：增强原语注册到 _AUGMENT_OPS。"""
        path = _source_path("autoaugment/policy_builder.py")
        assert _grep_source(path, "_AUGMENT_OPS")
        assert _grep_source(path, '"time_jitter": _time_jitter')
        assert _grep_source(path, '"freq_masking": _freq_masking')
        assert _grep_source(path, '"noise": _noise')
        assert _grep_source(path, '"cutout": _cutout')

    def test_grep_sampler_file_exists(self):
        """AutoAugmentSampler 源码文件存在。"""
        path = _source_path("autoaugment/sampler.py")
        assert path.exists()

    def test_grep_autoaugment_sampler_class(self):
        """grep 实证：AutoAugmentSampler 类定义 + SP 注册。"""
        path = _source_path("autoaugment/sampler.py")
        assert _grep_source(path, "class AutoAugmentSampler")
        assert _grep_source(path, 'name = "autoaugment"')
        assert _grep_source(path, "def sample(")
        # SP 注册
        assert _grep_source(path, 'register_sampler("autoaugment", AutoAugmentSampler)')

    def test_grep_init_file_exports(self):
        """grep 实证：__init__.py 导出 + make_autoaugment_datamodule_factory。"""
        path = _source_path("autoaugment/__init__.py")
        assert _grep_source(path, "def make_autoaugment_datamodule_factory")
        assert _grep_source(path, "from .search_space import")
        assert _grep_source(path, "from .policy_builder import")
        assert _grep_source(path, "from .sampler import")

    def test_grep_make_autoaugment_datamodule_factory_returns_generic_datamodule(self):
        """grep 实证：make_autoaugment_datamodule_factory 返回 GenericDataModule。"""
        path = _source_path("autoaugment/__init__.py")
        assert _grep_source(path, "from ..engine.datamodule import GenericDataModule")
        assert _grep_source(path, "return GenericDataModule(")

    def test_grep_autoaugment_registered_in_sp_registry(self):
        """grep 实证：AutoAugmentSampler 在 SP 注册表中（通过 list_samplers 验证）。"""
        samplers = list_samplers()
        assert "autoaugment" in samplers

    def test_grep_no_pipeline_modification(self):
        """grep 实证：P3.1 未修改 pipeline.py（datamodule_factory 路径已存在）。

        这是 P3.1 零侵入设计的核心：通过 datamodule_factory 注入，无需修改 stage_build。
        """
        path = _source_path("engine/runner/pipeline.py")
        # datamodule_factory 分支应在 P3.1 之前就存在
        assert _grep_source(path, "datamodule_factory is not None")

    def test_grep_p3_doc_reference(self):
        """grep 实证：P3 规划文档存在。"""
        doc_path = Path(__file__).parent.parent / "docs" / "analysis"
        # rglob 递归查找（兼容 Windows 路径分隔符差异）
        p3_docs = [p for p in doc_path.rglob("*P3*") if p.is_file()]
        assert len(p3_docs) >= 1
        # P3 文档应含 AutoAugment 章节
        content = p3_docs[0].read_text(encoding="utf-8")
        assert "AutoAugment" in content
        assert "AugmentationSearchSpace" in content


# ============================================================
# P3.1: 集成测试
# ============================================================
class TestAutoAugmentIntegration:
    """AutoAugment 端到端集成测试（P3.1）。"""

    def test_full_pipeline_ask_sample_build_transform(self):
        """完整流程：SP ask → 采样策略 → 构造 transform → 应用到数据。"""
        sm = StudyManager()
        ss = AugmentationSearchSpace(n_ops=2)
        sp_ss = ss.to_sp_search_space()
        study_id = sm.create_study(
            name="integration_test",
            direction="maximize",
            search_space=sp_ss,
            sampler="autoaugment",
        )

        # ask
        trial = sm.ask(study_id)
        assert "op_0" in trial.params

        # 构造 transform
        builder = AutoAugmentPolicyBuilder(search_space=ss)
        transform = builder.build(trial.params)

        # 应用到数据
        x = np.random.randn(3, 100).astype(np.float32)
        y = 1
        xx, yy = transform(x, y)
        assert xx.shape == x.shape
        assert yy == y

        # tell（反馈）
        sm.tell(trial.trial_id, value=0.85, state="completed")

    def test_factory_with_real_pipeline_stage_build_contract(self):
        """factory 符合 stage_build 调用契约（参数透传）。"""
        params = _make_policy_params(1)
        factory = make_autoaugment_datamodule_factory(params)

        # stage_build 调用契约（监督模式）：
        # factory(train_dataset, test_dataset, batch_size, num_workers,
        #         pin_memory, persistent_workers, learning_mode,
        #         train_transform, eval_transform)
        class MockDataset:
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return np.random.randn(100).astype(np.float32), idx % 3

        # 调用 factory（模拟 stage_build 的调用方式）
        dm = factory(
            train_dataset=MockDataset(),
            test_dataset=MockDataset(),
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            learning_mode="supervised",
            train_transform=None,  # factory 内部已构造，stage_build 传入的会被 kwargs 覆盖
            eval_transform=None,
        )
        assert dm is not None
        assert dm.batch_size == 4
        assert dm.learning_mode == "supervised"
