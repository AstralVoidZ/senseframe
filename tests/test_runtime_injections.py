"""RuntimeInjections 工厂字段拆分契约测试。

P2 演进（2026-07-18）：测试工厂字段从 ExperimentConfig 拆分到 RuntimeInjections dataclass。

测试维度：
1. RuntimeInjections dataclass 结构与默认值
2. ExperimentConfig 字段拆分（model_fields 不含工厂字段）
3. 属性代理：cfg.X 等价于 cfg.runtime.X（读取 + 赋值 + list 操作）
4. 构造兼容性：ExperimentConfig(scene=..., module_factory=...) 转发到 runtime
5. RuntimeInjections 实例直接构造
6. JSON Schema 纯净化（不含 runtime / 工厂字段）
7. 序列化纯净（model_dump / to_dict 不含 runtime）
8. YAML 安全：拒绝 runtime dict / 工厂字段注入
9. 访问点源码契约（pipeline.py / hpo.py / autoaugment / nas 零改动）
"""

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, List, Optional

import pytest

from senseframe.engine.config import (
    ExperimentConfig,
    RuntimeInjections,
    SceneConfig,
    InputFeature,
    OutputFeature,
    _RUNTIME_FACTORY_FIELDS,
)


# ============================================================
# 测试 fixtures
# ============================================================
@pytest.fixture
def scene_kwargs():
    return {"name": "x", "dataset": "y", "model_id": "z"}


@pytest.fixture
def input_features():
    return [InputFeature(name="a", type="csi")]


@pytest.fixture
def output_features():
    return [OutputFeature(name="b", type="category", num_classes=2)]


@pytest.fixture
def cfg(scene_kwargs, input_features, output_features):
    """构造一个默认 ExperimentConfig（无工厂注入）。"""
    return ExperimentConfig(
        scene=SceneConfig(**scene_kwargs),
        input_features=input_features,
        output_features=output_features,
    )


def _make_callable():
    """返回一个可调用对象，用于工厂字段测试。"""
    return lambda *args, **kwargs: None


# ============================================================
# 1. RuntimeInjections dataclass 结构
# ============================================================
class TestRuntimeInjectionsStructure:
    """RuntimeInjections dataclass 结构契约。"""

    def test_is_dataclass(self):
        """RuntimeInjections 是 dataclass（非 pydantic BaseModel）。"""
        assert is_dataclass(RuntimeInjections)

    def test_default_values(self):
        """默认构造：4 个字段均为 None 或空 list。"""
        ri = RuntimeInjections()
        assert ri.module_factory is None
        assert ri.datamodule_factory is None
        assert ri.extra_callbacks == []
        assert ri.trainer_factory is None

    def test_field_count(self):
        """RuntimeInjections 含 4 个字段（与 _RUNTIME_FACTORY_FIELDS 对齐）。"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(RuntimeInjections)}
        assert field_names == set(_RUNTIME_FACTORY_FIELDS)

    def test_construct_with_values(self):
        """构造时传入字段值。"""
        mf = _make_callable()
        dmf = _make_callable()
        tf = _make_callable()
        ri = RuntimeInjections(
            module_factory=mf,
            datamodule_factory=dmf,
            extra_callbacks=["cb1", "cb2"],
            trainer_factory=tf,
        )
        assert ri.module_factory is mf
        assert ri.datamodule_factory is dmf
        assert ri.extra_callbacks == ["cb1", "cb2"]
        assert ri.trainer_factory is tf

    def test_extra_callbacks_independent_default(self):
        """extra_callbacks 默认值用 field(default_factory=list)，每个实例独立。"""
        ri1 = RuntimeInjections()
        ri2 = RuntimeInjections()
        ri1.extra_callbacks.append("cb1")
        assert ri1.extra_callbacks == ["cb1"]
        assert ri2.extra_callbacks == []  # 不受影响


# ============================================================
# 2. ExperimentConfig 字段拆分
# ============================================================
class TestExperimentConfigFieldSplit:
    """ExperimentConfig 字段拆分契约。"""

    def test_factory_fields_not_in_model_fields(self):
        """ExperimentConfig.model_fields 不含 4 个工厂字段。"""
        model_field_names = set(ExperimentConfig.model_fields.keys())
        for f in _RUNTIME_FACTORY_FIELDS:
            assert f not in model_field_names, (
                f"工厂字段 {f} 不应在 ExperimentConfig.model_fields 中（已拆到 RuntimeInjections）"
            )

    def test_runtime_field_in_model_fields(self):
        """ExperimentConfig.model_fields 含 runtime 字段。"""
        assert "runtime" in ExperimentConfig.model_fields

    def test_runtime_field_excluded_from_model_dump(self, cfg):
        """runtime 字段用 Field(exclude=True)，model_dump 不含。"""
        dump = cfg.model_dump()
        assert "runtime" not in dump

    def test_runtime_default_factory(self, cfg):
        """ExperimentConfig 默认构造含 runtime=RuntimeInjections()。"""
        assert isinstance(cfg.runtime, RuntimeInjections)
        assert cfg.runtime.module_factory is None
        assert cfg.runtime.extra_callbacks == []


# ============================================================
# 3. 属性代理：读取 + 赋值 + list 操作
# ============================================================
class TestPropertyProxy:
    """@property 代理契约：cfg.X 等价于 cfg.runtime.X。"""

    def test_read_default(self, cfg):
        """默认值：cfg.X == cfg.runtime.X == None/[]。"""
        assert cfg.module_factory is cfg.runtime.module_factory is None
        assert cfg.datamodule_factory is cfg.runtime.datamodule_factory is None
        assert cfg.extra_callbacks is cfg.runtime.extra_callbacks
        assert cfg.extra_callbacks == []
        assert cfg.trainer_factory is cfg.runtime.trainer_factory is None

    def test_assign_module_factory(self, cfg):
        """赋值 cfg.module_factory 转发到 cfg.runtime.module_factory。"""
        mf = _make_callable()
        cfg.module_factory = mf
        assert cfg.module_factory is mf
        assert cfg.runtime.module_factory is mf

    def test_assign_datamodule_factory(self, cfg):
        """赋值 cfg.datamodule_factory 转发到 cfg.runtime.datamodule_factory。"""
        dmf = _make_callable()
        cfg.datamodule_factory = dmf
        assert cfg.datamodule_factory is dmf
        assert cfg.runtime.datamodule_factory is dmf

    def test_assign_trainer_factory(self, cfg):
        """赋值 cfg.trainer_factory 转发到 cfg.runtime.trainer_factory。"""
        tf = _make_callable()
        cfg.trainer_factory = tf
        assert cfg.trainer_factory is tf
        assert cfg.runtime.trainer_factory is tf

    def test_assign_extra_callbacks(self, cfg):
        """赋值 cfg.extra_callbacks 转发到 cfg.runtime.extra_callbacks。"""
        cbs = ["cb1", "cb2"]
        cfg.extra_callbacks = cbs
        assert cfg.extra_callbacks is cbs
        assert cfg.runtime.extra_callbacks is cbs

    def test_extra_callbacks_append(self, cfg):
        """list 操作：cfg.extra_callbacks.append 直接修改 runtime.extra_callbacks。"""
        cfg.extra_callbacks.append("cb1")
        assert cfg.extra_callbacks == ["cb1"]
        assert cfg.runtime.extra_callbacks == ["cb1"]
        # 再次 append 验证引用一致性
        cfg.extra_callbacks.append("cb2")
        assert cfg.runtime.extra_callbacks == ["cb1", "cb2"]

    def test_extra_callbacks_extend(self, cfg):
        """list 操作：cfg.extra_callbacks.extend 直接修改 runtime.extra_callbacks。"""
        cfg.extra_callbacks.extend(["cb1", "cb2"])
        assert cfg.runtime.extra_callbacks == ["cb1", "cb2"]


# ============================================================
# 4. 构造兼容性：工厂字段转发到 runtime
# ============================================================
class TestConstructorForwarding:
    """构造时传入工厂字段，转发到 runtime。"""

    def test_construct_with_module_factory(
        self, scene_kwargs, input_features, output_features
    ):
        """ExperimentConfig(..., module_factory=...) 转发到 runtime。"""
        mf = _make_callable()
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            module_factory=mf,
        )
        assert cfg.module_factory is mf
        assert cfg.runtime.module_factory is mf

    def test_construct_with_extra_callbacks(
        self, scene_kwargs, input_features, output_features
    ):
        """ExperimentConfig(..., extra_callbacks=...) 转发到 runtime。"""
        cbs = ["cb1"]
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            extra_callbacks=cbs,
        )
        assert cfg.extra_callbacks is cbs
        assert cfg.runtime.extra_callbacks is cbs

    def test_construct_with_all_factory_fields(
        self, scene_kwargs, input_features, output_features
    ):
        """ExperimentConfig(..., 4 个工厂字段=...) 全部转发到 runtime。"""
        mf, dmf, tf = _make_callable(), _make_callable(), _make_callable()
        cbs = ["cb1"]
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            module_factory=mf,
            datamodule_factory=dmf,
            extra_callbacks=cbs,
            trainer_factory=tf,
        )
        assert cfg.runtime.module_factory is mf
        assert cfg.runtime.datamodule_factory is dmf
        assert cfg.runtime.extra_callbacks is cbs
        assert cfg.runtime.trainer_factory is tf

    def test_construct_runtime_and_factory_together(
        self, scene_kwargs, input_features, output_features
    ):
        """同时传入 runtime 和工厂字段：工厂字段合并到新 runtime 实例。

        Review 修复（2026-07-18）：合并不修改原 RuntimeInjections 实例，
        而是创建新实例（避免共享 ri 被构造污染的副作用）。
        """
        original_mf = _make_callable()
        existing_runtime = RuntimeInjections(module_factory=original_mf)
        new_dmf = _make_callable()
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            runtime=existing_runtime,
            datamodule_factory=new_dmf,  # 合并到新 runtime 实例
        )
        # 新 runtime 实例（非原 existing_runtime）
        assert cfg.runtime is not existing_runtime
        # 原 runtime 的 module_factory 保留（复制到新实例）
        assert cfg.runtime.module_factory is original_mf
        # 原 existing_runtime 未被修改（无副作用）
        assert existing_runtime.datamodule_factory is None
        # 新增的 datamodule_factory 合并到新实例
        assert cfg.runtime.datamodule_factory is new_dmf


# ============================================================
# 5. RuntimeInjections 实例直接构造
# ============================================================
class TestRuntimeInjectionsInstance:
    """通过 runtime=RuntimeInjections(...) 构造。"""

    def test_construct_with_runtime_instance(
        self, scene_kwargs, input_features, output_features
    ):
        """ExperimentConfig(..., runtime=RuntimeInjections(...)) 构造。"""
        mf = _make_callable()
        ri = RuntimeInjections(module_factory=mf, extra_callbacks=["cb1"])
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            runtime=ri,
        )
        assert cfg.runtime is ri
        assert cfg.module_factory is mf
        assert cfg.extra_callbacks == ["cb1"]


# ============================================================
# 6. JSON Schema 纯净化
# ============================================================
class TestJsonSchemaPurity:
    """JSON Schema 不含 runtime / 工厂字段。"""

    def test_schema_excludes_runtime(self):
        """model_json_schema() 不含 runtime 字段。"""
        schema = ExperimentConfig.model_json_schema()
        props = schema.get("properties", {})
        assert "runtime" not in props

    def test_schema_excludes_factory_fields(self):
        """model_json_schema() 不含 4 个工厂字段。"""
        schema = ExperimentConfig.model_json_schema()
        props = schema.get("properties", {})
        for f in _RUNTIME_FACTORY_FIELDS:
            assert f not in props, f"schema 不应含工厂字段 {f}"

    def test_schema_contains_declarative_fields(self):
        """model_json_schema() 含声明式字段（scene/trainer/hpo 等）。"""
        schema = ExperimentConfig.model_json_schema()
        props = schema.get("properties", {})
        for f in ("scene", "input_features", "output_features", "trainer", "hpo",
                  "output_dir", "save_model", "export_formats", "strict_schema",
                  "devices", "strategy", "num_nodes", "sync_batchnorm", "num_processes"):
            assert f in props, f"schema 应含声明式字段 {f}"

    def test_schema_required_excludes_runtime(self):
        """model_json_schema() 的 required 列表不含 runtime。"""
        schema = ExperimentConfig.model_json_schema()
        required = schema.get("required", [])
        assert "runtime" not in required


# ============================================================
# 7. 序列化纯净
# ============================================================
class TestSerializationPurity:
    """model_dump / to_dict 不含 runtime / 工厂字段。"""

    def test_model_dump_excludes_runtime(self, cfg):
        """model_dump() 不含 runtime 字段。"""
        dump = cfg.model_dump()
        assert "runtime" not in dump

    def test_model_dump_excludes_factory_fields(self, cfg):
        """model_dump() 不含 4 个工厂字段。"""
        dump = cfg.model_dump()
        for f in _RUNTIME_FACTORY_FIELDS:
            assert f not in dump

    def test_to_dict_excludes_runtime(self, cfg):
        """to_dict() 不含 runtime 字段。"""
        d = cfg.to_dict()
        assert "runtime" not in d

    def test_to_dict_excludes_factory_fields(self, cfg):
        """to_dict() 不含 4 个工厂字段。"""
        d = cfg.to_dict()
        for f in _RUNTIME_FACTORY_FIELDS:
            assert f not in d

    def test_to_dict_round_trip_no_runtime(self, cfg):
        """to_dict() → from_dict() 往返不含 runtime。"""
        d = cfg.to_dict()
        cfg2 = ExperimentConfig.from_dict(d)
        assert cfg2.runtime.module_factory is None
        assert cfg2.runtime.extra_callbacks == []


# ============================================================
# 8. YAML 安全：拒绝 runtime dict / 工厂字段注入
# ============================================================
class TestYamlSafety:
    """YAML 安全契约：拒绝运行时对象注入。"""

    def test_from_dict_rejects_module_factory(self, scene_kwargs):
        """from_dict 拒绝 YAML 中的 module_factory。"""
        with pytest.raises(ValueError, match="运行时工厂字段"):
            ExperimentConfig.from_dict({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "module_factory": _make_callable(),
            })

    def test_from_dict_rejects_datamodule_factory(self, scene_kwargs):
        """from_dict 拒绝 YAML 中的 datamodule_factory。"""
        with pytest.raises(ValueError, match="运行时工厂字段"):
            ExperimentConfig.from_dict({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "datamodule_factory": _make_callable(),
            })

    def test_from_dict_rejects_extra_callbacks(self, scene_kwargs):
        """from_dict 拒绝 YAML 中的 extra_callbacks。"""
        with pytest.raises(ValueError, match="运行时工厂字段"):
            ExperimentConfig.from_dict({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "extra_callbacks": ["cb1"],
            })

    def test_from_dict_rejects_trainer_factory(self, scene_kwargs):
        """from_dict 拒绝 YAML 中的 trainer_factory。"""
        with pytest.raises(ValueError, match="运行时工厂字段"):
            ExperimentConfig.from_dict({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "trainer_factory": _make_callable(),
            })

    def test_from_dict_rejects_runtime_dict(self, scene_kwargs):
        """from_dict 拒绝 YAML 中的 runtime dict（防止 dict 构造 RuntimeInjections）。"""
        with pytest.raises(ValueError, match="runtime"):
            ExperimentConfig.from_dict({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "runtime": {"module_factory": "evil"},
            })

    def test_model_validate_rejects_runtime_dict(self, scene_kwargs):
        """model_validate 拒绝 runtime dict（model_validator before 拦截）。"""
        with pytest.raises(ValueError, match="runtime"):
            ExperimentConfig.model_validate({
                "scene": scene_kwargs,
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "runtime": {},
            })


# ============================================================
# 9. 访问点源码契约（pipeline.py / hpo.py / autoaugment / nas 零改动）
# ============================================================
class TestAccessPointsUnchanged:
    """访问点源码契约：所有 ctx.config.X / cfg.X 访问无需修改。

    通过源码检查验证访问点仍通过 cfg.X（而非 cfg.runtime.X）访问工厂字段，
    @property 代理保证向后兼容。
    """

    @pytest.fixture
    def pipeline_source(self):
        root = Path(__file__).resolve().parents[1]
        return (root / "senseframe" / "engine" / "runner" / "pipeline.py").read_text(encoding="utf-8")

    @pytest.fixture
    def hpo_source(self):
        root = Path(__file__).resolve().parents[1]
        return (root / "senseframe" / "engine" / "hpo.py").read_text(encoding="utf-8")

    def test_pipeline_accesses_config_module_factory(self, pipeline_source):
        """pipeline.py 仍通过 ctx.config.module_factory 访问（非 ctx.config.runtime.module_factory）。"""
        assert "ctx.config.module_factory" in pipeline_source
        # 确保没有改用 runtime.module_factory（访问点零改动契约）
        assert "ctx.config.runtime.module_factory" not in pipeline_source

    def test_pipeline_accesses_config_datamodule_factory(self, pipeline_source):
        """pipeline.py 仍通过 ctx.config.datamodule_factory 访问。"""
        assert "ctx.config.datamodule_factory" in pipeline_source
        assert "ctx.config.runtime.datamodule_factory" not in pipeline_source

    def test_pipeline_accesses_config_extra_callbacks(self, pipeline_source):
        """pipeline.py 仍通过 ctx.config.extra_callbacks 访问。"""
        assert "ctx.config.extra_callbacks" in pipeline_source
        assert "ctx.config.runtime.extra_callbacks" not in pipeline_source

    def test_pipeline_accesses_config_trainer_factory(self, pipeline_source):
        """pipeline.py 仍通过 ctx.config.trainer_factory 访问。"""
        assert "ctx.config.trainer_factory" in pipeline_source
        assert "ctx.config.runtime.trainer_factory" not in pipeline_source

    def test_hpo_accesses_config_extra_callbacks(self, hpo_source):
        """hpo.py 仍通过 modified_config.extra_callbacks 访问。"""
        assert "modified_config.extra_callbacks" in hpo_source
        assert "modified_config.runtime.extra_callbacks" not in hpo_source

    def test_pipeline_extra_callbacks_extend_works(self, pipeline_source):
        """pipeline.py 用 ctx.callbacks.extend(ctx.config.extra_callbacks)。

        @property 代理返回 runtime.extra_callbacks 引用，extend 操作正常。
        """
        assert "ctx.callbacks.extend(ctx.config.extra_callbacks)" in pipeline_source

    def test_pipeline_extra_callbacks_append_works(self, hpo_source):
        """hpo.py 用 modified_config.extra_callbacks.append(...)。

        @property 代理返回 runtime.extra_callbacks 引用，append 操作正常。
        """
        assert "modified_config.extra_callbacks.append(" in hpo_source


# ============================================================
# 10. 端到端集成：工厂注入 + 访问点
# ============================================================
class TestEndToEndFactoryInjection:
    """端到端集成：工厂注入后访问点能正常读取。"""

    def test_module_factory_inject_and_read(self, cfg):
        """注入 module_factory 后，通过 cfg.module_factory 读取。"""
        mf = _make_callable()
        cfg.module_factory = mf
        # 模拟 pipeline.py 访问
        assert cfg.module_factory is not None
        assert cfg.module_factory is mf

    def test_extra_callbacks_inject_and_append(self, cfg):
        """注入 extra_callbacks 后，append 操作生效。"""
        # 模拟 hpo.py 注入
        cfg.extra_callbacks.append("OptunaReportingCallback")
        # 模拟 pipeline.py 读取
        callbacks = cfg.extra_callbacks
        assert "OptunaReportingCallback" in callbacks

    def test_datamodule_factory_inject_and_read(self, cfg):
        """注入 datamodule_factory 后，通过 cfg.datamodule_factory 读取。"""
        dmf = _make_callable()
        cfg.datamodule_factory = dmf
        # 模拟 pipeline.py 访问
        assert cfg.datamodule_factory is not None
        assert cfg.datamodule_factory is dmf

    def test_trainer_factory_inject_and_read(self, cfg):
        """注入 trainer_factory 后，通过 cfg.trainer_factory 读取。"""
        tf = _make_callable()
        cfg.trainer_factory = tf
        # 模拟 pipeline.py 访问
        assert cfg.trainer_factory is not None
        assert cfg.trainer_factory is tf

    def test_construct_with_factory_and_read_via_property(
        self, scene_kwargs, input_features, output_features
    ):
        """构造时传入工厂字段，通过 @property 读取（向后兼容）。"""
        mf = _make_callable()
        cfg = ExperimentConfig(
            scene=SceneConfig(**scene_kwargs),
            input_features=input_features,
            output_features=output_features,
            module_factory=mf,
            extra_callbacks=["cb1"],
        )
        # 模拟 pipeline.py / hpo.py 访问
        assert cfg.module_factory is mf
        assert cfg.extra_callbacks == ["cb1"]
        # 追加后访问点能看到
        cfg.extra_callbacks.append("cb2")
        assert cfg.extra_callbacks == ["cb1", "cb2"]


# ============================================================
# 11. deepcopy 路径（Review 补充 2026-07-18）
# ============================================================
class TestDeepCopyCompatibility:
    """deepcopy + @property 代理兼容性契约。

    Review 补充（2026-07-18）：HPO/重试/baseline/automl 4 处 deepcopy(config) 后
    修改 extra_callbacks / module_factory，需验证 @property 代理在 deepcopy 后正常工作。
    """

    def test_deepcopy_extra_callbacks_reference_decoupled(self, cfg):
        """deepcopy 后 extra_callbacks 引用断开（修改副本不影响原 cfg）。"""
        import copy
        cfg.extra_callbacks.append("cb1")
        cfg2 = copy.deepcopy(cfg)
        # 引用断开
        assert cfg2.extra_callbacks is not cfg.extra_callbacks
        assert cfg2.runtime.extra_callbacks is not cfg.runtime.extra_callbacks
        # 修改副本不影响原
        cfg2.extra_callbacks.append("cb2")
        assert cfg.extra_callbacks == ["cb1"]
        assert cfg2.extra_callbacks == ["cb1", "cb2"]

    def test_deepcopy_module_factory_independent(self, cfg):
        """deepcopy 后 module_factory 独立赋值（修改副本不影响原 cfg）。"""
        import copy
        cfg.module_factory = _make_callable()
        cfg2 = copy.deepcopy(cfg)
        # 赋值新工厂不影响原
        cfg2.module_factory = _make_callable()
        assert cfg.module_factory is not cfg2.module_factory

    def test_deepcopy_runtime_instance_decoupled(self, cfg):
        """deepcopy 后 runtime 实例是新对象（非共享引用）。"""
        import copy
        cfg2 = copy.deepcopy(cfg)
        assert cfg2.runtime is not cfg.runtime

    def test_deepcopy_simulates_hpo_path(self, cfg):
        """模拟 hpo.py 的 deepcopy + extra_callbacks.append 路径。"""
        import copy
        # 模拟 hpo.py:268 new_config = copy.deepcopy(config)
        new_config = copy.deepcopy(cfg)
        # 模拟 hpo.py:389 modified_config.extra_callbacks.append(...)
        new_config.extra_callbacks.append("OptunaReportingCallback")
        # 原 config 不受影响
        assert cfg.extra_callbacks == []
        assert new_config.extra_callbacks == ["OptunaReportingCallback"]

    def test_deepcopy_simulates_retry_path(self, cfg):
        """模拟 retry.py:167 config = copy.deepcopy(config) 后修改字段。"""
        import copy
        cfg.extra_callbacks.append("original_cb")
        # 模拟 retry.py deepcopy
        new_cfg = copy.deepcopy(cfg)
        new_cfg.extra_callbacks.append("retry_cb")
        new_cfg.output_dir = "retry_run"
        # 原 cfg 不受影响
        assert cfg.extra_callbacks == ["original_cb"]
        assert "retry_cb" not in cfg.extra_callbacks
        assert cfg.output_dir != "retry_run"


# ============================================================
# 12. model_copy 工厂字段代理（遗留问题 4 修复 2026-07-19）
# ============================================================
class TestModelCopyFactoryFields:
    """ExperimentConfig.model_copy 工厂字段代理契约。

    遗留问题 4 修复（2026-07-19）：pydantic v2 BaseModel.model_copy 绕过 __init__
    和 model_validator，直接写 __dict__。但工厂字段是 @property（data descriptor），
    描述符协议优先于 __dict__ 访问，导致 update={"module_factory": X} 不生效——
    __dict__["module_factory"] 被遮蔽，访问返回 self.runtime.module_factory（原值）。

    修复：覆写 ExperimentConfig.model_copy，从 update 中提取工厂字段，
    调用 super().model_copy 处理剩余字段，然后通过 setattr 应用工厂字段——
    setattr 触发 @module_factory.setter，写入 self.runtime.module_factory。
    """

    def test_model_copy_update_module_factory(self, cfg):
        """model_copy(update={"module_factory": X}) 后 new_cfg.module_factory is X。"""
        factory = _make_callable()
        new_cfg = cfg.model_copy(update={"module_factory": factory})
        assert new_cfg.module_factory is factory
        assert new_cfg.runtime.module_factory is factory

    def test_model_copy_update_datamodule_factory(self, cfg):
        """model_copy(update={"datamodule_factory": X}) 后新值生效。"""
        factory = _make_callable()
        new_cfg = cfg.model_copy(update={"datamodule_factory": factory})
        assert new_cfg.datamodule_factory is factory
        assert new_cfg.runtime.datamodule_factory is factory

    def test_model_copy_update_extra_callbacks(self, cfg):
        """model_copy(update={"extra_callbacks": [...]}) 替换 extra_callbacks 列表。"""
        new_callbacks = ["cb1", "cb2"]
        new_cfg = cfg.model_copy(update={"extra_callbacks": new_callbacks})
        assert new_cfg.extra_callbacks == ["cb1", "cb2"]
        assert new_cfg.runtime.extra_callbacks == ["cb1", "cb2"]

    def test_model_copy_update_trainer_factory(self, cfg):
        """model_copy(update={"trainer_factory": X}) 后新值生效。"""
        factory = _make_callable()
        new_cfg = cfg.model_copy(update={"trainer_factory": factory})
        assert new_cfg.trainer_factory is factory
        assert new_cfg.runtime.trainer_factory is factory

    def test_model_copy_update_all_factory_fields(self, cfg):
        """同时更新 4 个工厂字段。"""
        mf, dm, ecb, tf = (
            _make_callable(), _make_callable(), ["cb"], _make_callable()
        )
        new_cfg = cfg.model_copy(update={
            "module_factory": mf,
            "datamodule_factory": dm,
            "extra_callbacks": ecb,
            "trainer_factory": tf,
        })
        assert new_cfg.module_factory is mf
        assert new_cfg.datamodule_factory is dm
        assert new_cfg.extra_callbacks == ["cb"]
        assert new_cfg.trainer_factory is tf

    def test_model_copy_update_declaration_field(self, cfg):
        """model_copy(update={"output_dir": "..."}) 声明字段正常更新（无回归）。"""
        new_cfg = cfg.model_copy(update={"output_dir": "new_run"})
        assert new_cfg.output_dir == "new_run"
        # 原 cfg 不受影响
        assert cfg.output_dir != "new_run"

    def test_model_copy_update_mixed_fields(self, cfg):
        """同时更新声明字段和工厂字段。"""
        factory = _make_callable()
        new_cfg = cfg.model_copy(update={
            "output_dir": "mixed_run",
            "module_factory": factory,
        })
        assert new_cfg.output_dir == "mixed_run"
        assert new_cfg.module_factory is factory
        assert new_cfg.runtime.module_factory is factory

    def test_model_copy_default_shallow_runtime_shared(self, cfg):
        """model_copy() 默认浅拷贝：runtime 是共享引用。"""
        new_cfg = cfg.model_copy()
        assert new_cfg.runtime is cfg.runtime

    def test_model_copy_deep_runtime_decoupled(self, cfg):
        """model_copy(deep=True) 后 runtime 是新实例（非共享引用）。"""
        new_cfg = cfg.model_copy(deep=True)
        assert new_cfg.runtime is not cfg.runtime

    def test_model_copy_deep_with_factory_update(self, cfg):
        """model_copy(deep=True, update={"module_factory": X}) 不影响原 cfg。"""
        cfg.module_factory = _make_callable()  # 原工厂
        original_factory = cfg.module_factory
        new_factory = _make_callable()
        new_cfg = cfg.model_copy(deep=True, update={"module_factory": new_factory})
        # 副本用新工厂
        assert new_cfg.module_factory is new_factory
        # 原 cfg 工厂不变
        assert cfg.module_factory is original_factory

    def test_model_copy_deep_extra_callbacks_decoupled(self, cfg):
        """model_copy(deep=True) 后 extra_callbacks 列表独立（修改副本不影响原）。"""
        cfg.extra_callbacks.append("original_cb")
        new_cfg = cfg.model_copy(deep=True)
        new_cfg.extra_callbacks.append("new_cb")
        assert cfg.extra_callbacks == ["original_cb"]
        assert new_cfg.extra_callbacks == ["original_cb", "new_cb"]

    def test_model_copy_update_runtime_instance(self, cfg):
        """model_copy(update={"runtime": new_ri}) 替换 runtime 实例。"""
        new_ri = RuntimeInjections(module_factory=_make_callable())
        new_cfg = cfg.model_copy(update={"runtime": new_ri})
        assert new_cfg.runtime is new_ri
        assert new_cfg.module_factory is new_ri.module_factory

    def test_model_copy_no_factory_in_update_no_regression(self, cfg):
        """无工厂字段的 model_copy 行为与 pydantic 默认一致（无回归）。"""
        new_cfg = cfg.model_copy(update={"output_dir": "x"})
        assert new_cfg.output_dir == "x"
        # runtime 保持默认行为（浅拷贝共享）
        assert new_cfg.runtime is cfg.runtime

    def test_model_copy_original_unchaged_after_factory_update(self, cfg):
        """model_copy(update={"module_factory": X}) 不修改原 cfg 的工厂字段。"""
        original_factory = _make_callable()
        cfg.module_factory = original_factory
        new_factory = _make_callable()
        new_cfg = cfg.model_copy(update={"module_factory": new_factory})
        # 原 cfg 工厂不变
        assert cfg.module_factory is original_factory
        assert cfg.runtime.module_factory is original_factory
        # 副本用新工厂
        assert new_cfg.module_factory is new_factory
