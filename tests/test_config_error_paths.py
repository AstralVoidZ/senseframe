"""配置错误路径契约测试。

P2 演进（2026-07-18）：验证 pydantic v2 迁移后的错误处理契约：
- ExperimentConfig.from_dict 失败时抛 ConfigValidationError（同时是 ValueError 子类）
- classify_error 能精确返回 CONFIG_VALIDATION_ERROR 错误码
- 错误消息包含字段路径上下文（ExperimentConfig.<field>）
- pydantic ValidationError 也被 classify_error 正确分类
- 错误码体系闭环（ERROR_CODES 字典与代码使用一致）

设计原则：
- 只测试"用户可见的契约"（异常类型、错误码、消息关键字），不测试 pydantic 内部格式
- 不依赖具体错误消息措辞（措辞可变，契约不变）
"""
from __future__ import annotations

import pytest

from senseframe.engine.config import (
    ExperimentConfig,
    SceneConfig,
    TrainerConfig,
    HPOConfig,
    InputFeature,
    OutputFeature,
)
from senseframe.engine.runner.errors import (
    ConfigValidationError,
    SenseFrameError,
    classify_error,
)
from senseframe.schemas import ERROR_CODES


# ============================================================
# 1. ExperimentConfig.from_dict 错误路径
# ============================================================
class TestFromDictErrorPath:
    """from_dict 失败时应抛 ConfigValidationError，错误码精确。"""

    def test_non_dict_input_raises_config_validation_error(self):
        """非 dict 输入抛 ConfigValidationError（同时是 ValueError 子类）。"""
        with pytest.raises(ConfigValidationError) as exc_info:
            ExperimentConfig.from_dict("not a dict")

        # 契约 1：是 ConfigValidationError
        assert isinstance(exc_info.value, ConfigValidationError)
        # 契约 2：同时是 ValueError（向后兼容现有 except ValueError）
        assert isinstance(exc_info.value, ValueError)
        # 契约 3：是 SenseFrameError（让 classify_error 命中精确分支）
        assert isinstance(exc_info.value, SenseFrameError)
        # 契约 4：错误码为 CONFIG_VALIDATION_ERROR
        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"
        # 契约 5：消息含上下文（ExperimentConfig）
        assert "ExperimentConfig" in str(exc_info.value)

    def test_missing_required_field_raises_config_validation_error(self):
        """缺必需字段抛 ConfigValidationError，消息含字段名。"""
        with pytest.raises(ConfigValidationError) as exc_info:
            ExperimentConfig.from_dict({})  # 缺 scene/input_features/output_features

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"
        # 消息应含缺失字段名
        msg = str(exc_info.value)
        assert "scene" in msg or "input_features" in msg or "output_features" in msg

    def test_missing_scene_field_message_contains_field_path(self):
        """缺 scene 字段时消息含 'scene' 字段路径。"""
        with pytest.raises(ConfigValidationError, match="scene"):
            ExperimentConfig.from_dict({"input_features": [], "output_features": []})

    def test_missing_input_features_field_message_contains_field_path(self):
        """缺 input_features 字段时消息含字段路径。"""
        with pytest.raises(ConfigValidationError, match="input_features"):
            ExperimentConfig.from_dict({
                "scene": {"name": "x", "dataset": "y", "model_id": "z"},
                "output_features": [],
            })

    def test_empty_input_features_raises_config_validation_error(self):
        """input_features 为空 list 抛 ConfigValidationError。"""
        with pytest.raises(ConfigValidationError) as exc_info:
            ExperimentConfig.from_dict({
                "scene": {"name": "x", "dataset": "y", "model_id": "z"},
                "input_features": [],
                "output_features": [],
            })

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"
        assert "input_features" in str(exc_info.value)


# ============================================================
# 2. pydantic ValidationError 分类
# ============================================================
class TestPydanticValidationErrorClassification:
    """pydantic ValidationError（子配置校验失败）也应被正确分类。"""

    def test_trainer_config_invalid_epochs_classified_as_config_validation_error(self):
        """TrainerConfig.epochs <= 0 触发 pydantic ValidationError，分类为 CONFIG_VALIDATION_ERROR。"""
        with pytest.raises(ValueError) as exc_info:
            TrainerConfig(epochs=0)

        # pydantic ValidationError 是 ValueError 子类
        error_code = classify_error(exc_info.value)
        assert error_code == "CONFIG_VALIDATION_ERROR"

    def test_scene_config_empty_name_classified_as_config_validation_error(self):
        """SceneConfig.name 为空触发 pydantic ValidationError，分类正确。"""
        with pytest.raises(ValueError) as exc_info:
            SceneConfig(name="", dataset="x", model_id="y")

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"

    def test_hpo_config_invalid_sampler_classified_as_config_validation_error(self):
        """HPOConfig.sampler 非法值触发 ValidationError，分类正确。"""
        with pytest.raises(ValueError) as exc_info:
            HPOConfig(sampler="garbage")

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"

    def test_input_feature_invalid_type_classified_as_config_validation_error(self):
        """InputFeature.type 非法值触发 ValidationError，分类正确。"""
        with pytest.raises(ValueError) as exc_info:
            InputFeature(name="x", type="garbage")

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"

    def test_output_feature_category_without_num_classes_raises(self):
        """OutputFeature.type=category 但缺 num_classes 触发 model_validator 错误。"""
        with pytest.raises(ValueError) as exc_info:
            OutputFeature(name="y", type="category")

        assert classify_error(exc_info.value) == "CONFIG_VALIDATION_ERROR"


# ============================================================
# 3. 错误码体系闭环
# ============================================================
class TestErrorCodeClosure:
    """ERROR_CODES 字典应包含所有代码中使用的错误码。"""

    def test_ghost_error_codes_now_in_dictionary(self):
        """5 个原幽灵错误码应已在 ERROR_CODES 字典中定义。"""
        ghost_codes = [
            "MISSING_CONFIG",
            "CONFIG_NOT_FOUND",
            "INVALID_CONFIG_FORMAT",
            "UNSUPPORTED_FORMAT",
            "CONFIG_PARSE_ERROR",
        ]
        for code in ghost_codes:
            assert code in ERROR_CODES, (
                f"错误码 '{code}' 应在 ERROR_CODES 字典中定义，"
                f"消除幽灵错误码契约缺口"
            )

    def test_all_senseframe_error_subclasses_have_codes_in_dictionary(self):
        """所有 SenseFrameError 子类的 error_code 应在 ERROR_CODES 字典中。"""
        from senseframe.engine.runner import errors as err_mod
        import inspect

        for name, cls in inspect.getmembers(err_mod, inspect.isclass):
            if not issubclass(cls, SenseFrameError) or cls is SenseFrameError:
                continue
            if not hasattr(cls, "error_code"):
                continue
            code = cls.error_code
            assert code in ERROR_CODES, (
                f"{cls.__name__}.error_code='{code}' 不在 ERROR_CODES 字典中"
            )


# ============================================================
# 4. CLI 错误流向契约（stdout vs stderr）
# ============================================================
class TestCliErrorStream:
    """CLI 错误输出应到 stderr，不污染 stdout（供程序化解析）。"""

    def test_cli_experiment_missing_config_uses_stderr(self, capsys):
        """--config 缺失时错误 JSON 应输出到 stderr，stdout 为空。"""
        from senseframe.cli import _cmd_experiment

        # 构造 args namespace
        class Args:
            config = None
            log_level = "INFO"
            log_file = None

        # sys.exit 会抛 SystemExit，捕获它以检查 stdout/stderr
        with pytest.raises(SystemExit) as exc_info:
            _cmd_experiment(Args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        # stdout 应为空（错误不走 stdout）
        assert captured.out == "", (
            f"错误输出应到 stderr，但 stdout 有内容: {captured.out!r}"
        )
        # stderr 应含 JSON 错误
        assert "MISSING_CONFIG" in captured.err
        assert "error" in captured.err


# ============================================================
# 5. ConfigValidationError 类型层级
# ============================================================
class TestConfigValidationErrorHierarchy:
    """ConfigValidationError 类型层级保证向后兼容。"""

    def test_config_validation_error_is_value_error(self):
        """ConfigValidationError 必须是 ValueError 子类（向后兼容 except ValueError）。"""
        e = ConfigValidationError("test")
        assert isinstance(e, ValueError)

    def test_config_validation_error_is_senseframe_error(self):
        """ConfigValidationError 必须是 SenseFrameError 子类（让 classify_error 精确命中）。"""
        e = ConfigValidationError("test")
        assert isinstance(e, SenseFrameError)

    def test_config_validation_error_has_error_code(self):
        """ConfigValidationError 必须有 error_code 类属性。"""
        assert ConfigValidationError.error_code == "CONFIG_VALIDATION_ERROR"


# ============================================================
# 6. extra="forbid" 契约（P2 演进）
# ============================================================
class TestExtraForbidContract:
    """所有 6 个配置类应使用 extra="forbid"，捕获 YAML 字段拼写错误。

    P2 演进（2026-07-18）：从 extra="ignore" 切换到 extra="forbid"，
    在 schema 层强制字段名正确性，而非依赖文档检查。
    """

    def test_input_feature_rejects_extra_keys(self):
        """InputFeature 拒绝未声明字段（如 shapes 误拼）。"""
        with pytest.raises(ValueError, match="extra"):
            InputFeature(name="x", type="csi", shapes=[1, 2])  # shapes 不是声明字段

    def test_output_feature_rejects_extra_keys(self):
        """OutputFeature 拒绝未声明字段（如 num_class 单数误拼）。"""
        with pytest.raises(ValueError, match="extra"):
            OutputFeature(name="y", type="category", num_class=7, num_classes=7)

    def test_trainer_config_rejects_max_epochs(self):
        """TrainerConfig 拒绝 max_epochs（Lightning 字段名，非我们的 epochs）。"""
        with pytest.raises(ValueError, match="extra"):
            TrainerConfig(max_epochs=100)

    def test_trainer_config_rejects_lr_abbreviation(self):
        """TrainerConfig 拒绝 lr（learning_rate 的常见缩写）。"""
        with pytest.raises(ValueError, match="extra"):
            TrainerConfig(lr=0.001)

    def test_hpo_config_rejects_extra_keys(self):
        """HPOConfig 拒绝未声明字段（如 trials 误拼）。"""
        with pytest.raises(ValueError, match="extra"):
            HPOConfig(trials=10)

    def test_scene_config_rejects_model_abbreviation(self):
        """SceneConfig 拒绝 model（应为 model_id）。"""
        with pytest.raises(ValueError, match="extra"):
            SceneConfig(name="x", dataset="y", model="MLP")

    def test_experiment_config_rejects_train_typo(self):
        """ExperimentConfig.from_dict 拒绝 train（应为 trainer）。"""
        with pytest.raises(ValueError, match="未知字段"):
            ExperimentConfig.from_dict({
                "scene": {"name": "x", "dataset": "y", "model_id": "z"},
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "train": {"epochs": 10},  # typo: should be "trainer"
            })

    def test_experiment_config_rejects_factory_field_in_yaml(self):
        """ExperimentConfig.from_dict 拒绝 YAML 中的运行时工厂字段。"""
        with pytest.raises(ValueError, match="运行时工厂字段"):
            ExperimentConfig.from_dict({
                "scene": {"name": "x", "dataset": "y", "model_id": "z"},
                "input_features": [{"name": "a", "type": "csi"}],
                "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
                "module_factory": "some_callable",  # 运行时字段，不应在 YAML 中
            })

    def test_scene_params_escape_hatch_still_works(self):
        """scene.params 内的任意键不算 SceneConfig extra（escape hatch 保留）。"""
        # params 是声明字段，其中的键不算 extra
        cfg = SceneConfig(
            name="x", dataset="y", model_id="z",
            params={"custom_key": "value", "another": 123},
        )
        assert cfg.params["custom_key"] == "value"


# ============================================================
# 7. 分布式训练字段契约（P2 修复）
# ============================================================
class TestDistributedTrainingFields:
    """ExperimentConfig 应包含文档化的分布式训练字段。

    P2 修复（2026-07-18）：devices/strategy/num_nodes/sync_batchnorm/num_processes
    原在 config_schema.md 文档化为顶层 YAML 字段，但未在 ExperimentConfig 中声明，
    被 extra="ignore" 静默丢弃。现在提升为声明字段。
    """

    def test_distributed_fields_have_defaults(self):
        """5 个分布式训练字段应有默认值（与 routing.py fallback 对齐）。"""
        from senseframe.engine.config import ExperimentConfig, SceneConfig, InputFeature, OutputFeature
        cfg = ExperimentConfig(
            scene=SceneConfig(name="x", dataset="y", model_id="z"),
            input_features=[InputFeature(name="a", type="csi")],
            output_features=[OutputFeature(name="b", type="category", num_classes=2)],
        )
        assert cfg.devices == 1
        assert cfg.strategy is None
        assert cfg.num_nodes == 1
        assert cfg.sync_batchnorm is False
        assert cfg.num_processes == 1

    def test_distributed_fields_from_dict(self):
        """from_dict 能从 YAML 顶层读取分布式训练字段。"""
        from senseframe.engine.config import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "scene": {"name": "x", "dataset": "y", "model_id": "z"},
            "input_features": [{"name": "a", "type": "csi"}],
            "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
            "devices": 2,
            "strategy": "ddp",
            "num_nodes": 4,
            "sync_batchnorm": True,
        })
        assert cfg.devices == 2
        assert cfg.strategy == "ddp"
        assert cfg.num_nodes == 4
        assert cfg.sync_batchnorm is True

    def test_distributed_fields_in_to_dict(self):
        """to_dict 输出应包含分布式训练字段（供 routing.py 消费）。"""
        from senseframe.engine.config import ExperimentConfig
        cfg = ExperimentConfig.from_dict({
            "scene": {"name": "x", "dataset": "y", "model_id": "z"},
            "input_features": [{"name": "a", "type": "csi"}],
            "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
            "devices": 4,
            "strategy": "ddp",
        })
        d = cfg.to_dict()
        assert d["devices"] == 4
        assert d["strategy"] == "ddp"
        assert "num_nodes" in d
        assert "sync_batchnorm" in d
        assert "num_processes" in d

    def test_experiment_config_to_dict_passes_distributed_fields_to_routing(self):
        """experiment_config_to_dict 输出应含分布式字段（供 routing.py 消费）。"""
        from senseframe.engine.config import ExperimentConfig
        from senseframe.engine.runner.resolver import experiment_config_to_dict
        cfg = ExperimentConfig.from_dict({
            "scene": {"name": "x", "dataset": "y", "model_id": "z"},
            "input_features": [{"name": "a", "type": "csi"}],
            "output_features": [{"name": "b", "type": "category", "num_classes": 2}],
            "devices": 2,
            "strategy": "ddp",
        })
        d = experiment_config_to_dict(cfg)
        assert d["devices"] == 2
        assert d["strategy"] == "ddp"
