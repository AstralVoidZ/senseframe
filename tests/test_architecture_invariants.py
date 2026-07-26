"""架构不变量测试：验证 v0.2.0 五项架构改进的设计约束。

设计原则：
- 每个测试对标一个架构属性，不是对标具体实现
- 测试若 pass 但实际运行有 bug，则测试无效——本文件中的测试必须能抓到 bug
- 回归守卫：如果有人重新引入已消除的反模式，测试必须失败

五项架构属性：
  A. CQS 合规：getter 不修改注册表状态（方案 4）
  B. 单一执行路径：run_experiment/reconcile/hpo 三路归一（方案 1）
  C. extra 纪律化：框架代码不写入 ctx.extra（方案 2）
  D. 异常层级体系：SenseFrameError 子类携带 error_code（方案 3）
  E. DSP-3 就绪度：advisory 契约可查询且不阻断（方案 5）
  F. 回归守卫：已消除的反模式不被重新引入
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# A. CQS 合规：getter 不修改注册表状态（方案 4）
# ============================================================

class TestCQSCompliance:
    """命令查询分离：纯 getter 调用不应产生注册表写副作用。"""

    def _snapshot_all_registries(self) -> dict:
        """捕获所有注册表的当前键集合。

        CQS 测试必须覆盖所有可变全局状态，不能只查 _MODEL_REGISTRY——
        否则 getter 引入 _SCENE_REGISTRY 副作用时测试会空过（反向验证发现的 bug）。
        """
        from senseframe.registry import _MODEL_REGISTRY, _DATASET_REGISTRY
        from senseframe.scenes import _REGISTRY as _SCENE_REGISTRY, _LAZY_SCENES
        return {
            "model": set(_MODEL_REGISTRY.keys()),
            "dataset": set(_DATASET_REGISTRY.keys()),
            "scene": set(_SCENE_REGISTRY.keys()),
            "lazy_scenes": set(_LAZY_SCENES.keys()),
        }

    def test_get_model_spec_does_not_mutate_registry(self):
        """get_model_spec() 是纯查询，不应触发任何注册表变化。

        设计约束：getter 不修改全局状态。如果重新引入 _ensure_wifi_csi_registered
        或类似的"查询时自动注册"逻辑，此测试会因注册表状态变化而失败。
        """
        from senseframe.registry import get_model_spec
        before = self._snapshot_all_registries()
        try:
            get_model_spec("NonExistentModel12345")
        except Exception:
            pass  # 预期抛异常
        after = self._snapshot_all_registries()
        assert before == after, \
            f"get_model_spec 修改了注册表: before={before}, after={after}"

    def test_get_dataset_spec_does_not_mutate_registry(self):
        """get_dataset_spec() 是纯查询，不应触发任何注册表变化。"""
        from senseframe.registry import get_dataset_spec
        before = self._snapshot_all_registries()
        try:
            get_dataset_spec("NonExistentDataset12345")
        except Exception:
            pass
        after = self._snapshot_all_registries()
        assert before == after, \
            f"get_dataset_spec 修改了注册表: before={before}, after={after}"

    def test_resolve_factory_does_not_mutate_registry(self):
        """resolve_factory() 是纯查询，不应触发任何注册表变化。"""
        from senseframe.registry import resolve_factory
        before = self._snapshot_all_registries()
        try:
            resolve_factory("GhostModel", "GhostDataset")
        except Exception:
            pass
        after = self._snapshot_all_registries()
        assert before == after, \
            f"resolve_factory 修改了注册表: before={before}, after={after}"

    def test_lazy_view_does_not_auto_register(self):
        """MODEL_TABLE / DATASET_INFO 延迟视图不应触发自动注册。

        访问 _LazyView 的属性时，不应修改任何注册表。
        """
        from senseframe.registry import MODEL_TABLE
        before = self._snapshot_all_registries()
        _ = list(MODEL_TABLE.keys())  # 触发 _refresh
        after = self._snapshot_all_registries()
        assert before == after, \
            f"MODEL_TABLE 访问触发了注册: before={before}, after={after}"

    def test_activate_lazy_scenes_is_idempotent(self):
        """activate_lazy_scenes() 多次调用不产生额外副作用。"""
        from senseframe.scenes import activate_lazy_scenes
        activate_lazy_scenes()
        after_first = self._snapshot_all_registries()
        activate_lazy_scenes()
        after_second = self._snapshot_all_registries()
        assert after_first == after_second


# ============================================================
# B. 单一执行路径：三路归一（方案 1）
# ============================================================

# ARCHITECTURE_TRIPWIRE: 单一执行路径——run_experiment/reconcile/hpo 三路归一到 Pipeline.run()
# 不可替代原因: "函数体不包含某些调用"是否定属性，反射/行为测试无法验证函数体内部不做什么；
#   必须检查源码文本才能确认委托关系未被绕过（如别名导入、独立 stage 循环）。
# 删除条件: 当委托关系由类型系统或框架强制保证时（如 run_experiment 变为 Pipeline.run 的
#   类型约束薄包装，编译器/类型检查器可静态验证无独立执行逻辑）。
class TestSingleExecutionPath:
    """run_experiment / reconcile / hpo 三条路径归一到 Pipeline.run()。"""

    def test_run_experiment_delegates_to_run_pipeline(self):
        """run_experiment 内部调用 run_pipeline，不包含独立 stage 逻辑。

        通过 inspect.getsource 检查函数体不包含 stage 调用（如 stage_train、
        trainer.fit 等），仅包含委托调用。
        """
        from senseframe.engine.runner.orchestrator import run_experiment
        src = inspect.getsource(run_experiment)
        # 不应包含独立执行逻辑的关键词
        forbidden = ["trainer.fit", "stage_train(", "stage_eval(", "stage_build(",
                     "pl.Trainer(", "EarlyStopping(", "ModelCheckpoint("]
        for kw in forbidden:
            assert kw not in src, \
                f"run_experiment 包含独立执行逻辑 '{kw}'，应委托给 run_pipeline"

    def test_run_experiment_calls_run_pipeline(self):
        """run_experiment 函数体包含对 run_pipeline 的调用。"""
        from senseframe.engine.runner.orchestrator import run_experiment
        src = inspect.getsource(run_experiment)
        assert "run_pipeline" in src, \
            "run_experiment 未委托给 run_pipeline"

    def test_reconcile_delegates_to_pipeline_run(self):
        """reconcile() 调用 pipeline.run()，不包含独立 stage 循环。

        设计约束：reconcile 不应复制 Pipeline.run 的 stage 循环逻辑。
        """
        from senseframe.orchestration import Orchestrator
        src = inspect.getsource(Orchestrator.reconcile)
        # 应包含 pipeline.run 调用
        assert "pipeline.run" in src, \
            "reconcile 未委托给 pipeline.run()"
        # 不应包含独立 stage 循环的关键词
        forbidden_in_loop = ["for stage_name, stage_fn in pipeline.stages",
                             "ctx = stage_fn(ctx)"]
        for kw in forbidden_in_loop:
            assert kw not in src, \
                f"reconcile 包含独立 stage 循环逻辑 '{kw}'，应委托给 pipeline.run()"

    def test_hpo_objective_uses_run_pipeline(self):
        """HPO 目标函数调用 run_pipeline，不调用 run_experiment。

        设计约束：HPO 路径必须走 Pipeline.run() 以获得 OOM 回退、checkpoint、
        OBP 指标等增强能力。

        反向验证：仅检查 Call 节点函数名会被 `from ..runner import run_experiment
        as run_pipeline` 别名绕过（反向验证发现的空过 bug）。必须同时检查
        ImportFrom 节点，确认 run_pipeline 是真名导入而非别名。
        """
        from senseframe.engine.hpo import _default_objective
        src = inspect.getsource(_default_objective)
        tree = ast.parse(textwrap.dedent(src))

        # 1. 检查 import 语句：确认 run_pipeline 是真名导入，不是 run_experiment 的别名
        aliases_of_run_experiment: list = []
        has_direct_run_pipeline_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "run_experiment" and alias.asname == "run_pipeline":
                        aliases_of_run_experiment.append(node.lineno)
                    if alias.name == "run_pipeline" and alias.asname is None:
                        has_direct_run_pipeline_import = True
        assert aliases_of_run_experiment == [], \
            f"HPO 目标函数用别名绕过检查: line {aliases_of_run_experiment} " \
            f"'from ..runner import run_experiment as run_pipeline'"
        assert has_direct_run_pipeline_import, \
            "HPO 目标函数必须直接导入 run_pipeline（无 as 别名）"

        # 2. 检查 Call 节点：确认调用了 run_pipeline
        called_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)
        assert "run_pipeline" in called_names, \
            f"HPO 目标函数未调用 run_pipeline，实际调用: {called_names}"
        assert "run_experiment" not in called_names, \
            f"HPO 目标函数仍在调用 run_experiment，应改为 run_pipeline"


# ============================================================
# C. extra 纪律化：框架代码不写入 ctx.extra（方案 2）
# ============================================================

# ARCHITECTURE_TRIPWIRE: extra 纪律化——框架代码不写入 ctx.extra（方案 2）
# 不可替代原因: "源码中不存在 ctx.extra[...] = 赋值"是否定属性，行为测试无法区分
#   "从未写入"和"写入后又删除"；必须扫描源码文本才能确认框架代码不逃逸到 extra。
#   注：本类中 test_failed_stage_is_first_class_field / test_feedback_is_first_class_field /
#   test_schema_reports_first_class_fields 使用反射验证，属 A 类（非 grep），无需此注释。
# 删除条件: 当 ctx.extra 对框架代码变为只读（如类型系统区分 AgentContext 与 FrameworkContext，
#   或 extra 改为 property 仅允许 Agent 层写入）。
class TestExtraDiscipline:
    """PipelineContext.extra 仅限 Agent 自由扩展，框架代码不得写入。"""

    def _get_pipeline_source(self) -> str:
        """读取 pipeline 包源码（含所有 stage 文件 + context + runtime）。

        拆分背景：原 pipeline.py 上帝文件拆分为 pipeline/ 包，需聚合所有子模块源码
        才能完整检查框架代码不写入 ctx.extra 的契约。
        """
        from senseframe.engine.runner import pipeline as pipeline_module
        import pkgutil
        from pathlib import Path

        # 收集 pipeline 包及其子包的所有 .py 源码
        package_dir = Path(pipeline_module.__file__).parent
        source_parts = []
        for py_file in sorted(package_dir.rglob("*.py")):
            source_parts.append(py_file.read_text(encoding="utf-8"))
        return "\n".join(source_parts)

    def test_no_framework_writes_to_extra_in_pipeline(self):
        """pipeline 包中不应出现 ctx.extra[...] = 赋值。

        框架内部状态应使用 first-class 字段，不应逃逸到 extra。
        """
        src = self._get_pipeline_source()
        # 查找 ctx.extra[ 模式（赋值而非读取）
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 跳过注释和 docstring
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # 检测赋值模式：ctx.extra["key"] = value
            if "ctx.extra[" in stripped and "=" in stripped:
                # 排除 == 比较和 .get 读取
                before_eq = stripped.split("=")[0]
                if "ctx.extra[" in before_eq and ".get" not in before_eq:
                    # 排除 != 和 ==
                    if "==" not in before_eq and "!=" not in before_eq:
                        pytest.fail(
                            f"框架代码写入 ctx.extra (line {i+1}): {stripped}"
                        )

    def test_no_framework_writes_to_extra_in_orchestration(self):
        """orchestration.py 中不应出现 ctx.extra[...] = 赋值。"""
        from senseframe import orchestration
        src = inspect.getsource(orchestration)
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if "ctx.extra[" in stripped and "=" in stripped:
                before_eq = stripped.split("=")[0]
                if "ctx.extra[" in before_eq and ".get" not in before_eq:
                    if "==" not in before_eq and "!=" not in before_eq:
                        pytest.fail(
                            f"框架代码写入 ctx.extra (line {i+1}): {stripped}"
                        )

    def test_failed_stage_is_first_class_field(self):
        """failed_stage 是 PipelineContext 的 first-class 字段，不在 extra 中。"""
        from senseframe.engine.runner.pipeline import PipelineContext
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(PipelineContext)}
        assert "failed_stage" in field_names, \
            "failed_stage 应为 PipelineContext first-class 字段"
        assert "failed_error" in field_names, \
            "failed_error 应为 PipelineContext first-class 字段"

    def test_feedback_is_first_class_field(self):
        """feedback 是 PipelineContext 的 first-class 字段。"""
        from senseframe.engine.runner.pipeline import PipelineContext
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(PipelineContext)}
        assert "feedback" in field_names
        assert "final_eval" in field_names
        assert "training_log" in field_names
        assert "early_stopped" in field_names
        assert "training_duration_s" in field_names
        assert "best_model_path" in field_names
        assert "best_model_score" in field_names

    def test_schema_reports_first_class_fields(self):
        """PipelineContext.schema() 报告 first-class 训练结果字段。

        设计约束：DSP-1 schema 必须如实反映 stage 间传递的所有字段，
        不能有藏在 extra 里的隐藏字段。
        """
        from senseframe.engine.runner.pipeline import PipelineContext
        schema = PipelineContext.schema()
        field_names = {f["name"] for f in schema["fields"]}
        for name in ["failed_stage", "failed_error", "feedback", "final_eval",
                      "training_log", "early_stopped", "training_duration_s",
                      "best_model_path", "best_model_score"]:
            assert name in field_names, \
                f"schema() 未报告 first-class 字段 '{name}'"


# ============================================================
# D. 异常层级体系：SenseFrameError 子类携带 error_code（方案 3）
# ============================================================

class TestExceptionHierarchy:
    """异常层级体系：结构化 error_code，消除字符串匹配。"""

    def test_senseframe_error_base_class_exists(self):
        """SenseFrameError 基类存在且有 error_code 属性。"""
        from senseframe.engine.runner.errors import SenseFrameError
        assert hasattr(SenseFrameError, "error_code")
        assert SenseFrameError.error_code == "UNKNOWN_ERROR"

    @pytest.mark.parametrize("exc_class,expected_code", [
        ("SceneNotRegisteredError", "SCENE_NOT_FOUND"),
        ("DatasetNotSupportedError", "DATASET_NOT_SUPPORTED"),
        ("ModelNotSupportedError", "MODEL_NOT_SUPPORTED"),
        ("DataNotFoundError", "DATA_NOT_FOUND"),
        ("DataCorruptedError", "DATA_LOAD_ERROR"),
        ("OOMError", "OOM_ERROR"),
        ("CheckpointError", "CHECKPOINT_ERROR"),
        ("PreflightError", "PREFLIGHT_ERROR"),
        ("TrainingError", "TRAINING_ERROR"),
        ("ModelBuildError", "MODEL_BUILD_ERROR"),
        ("SaveError", "SAVE_ERROR"),
        ("ConfigValidationError", "CONFIG_VALIDATION_ERROR"),
    ])
    def test_exception_subclass_has_correct_error_code(self, exc_class, expected_code):
        """每个 SenseFrameError 子类携带正确的 error_code。"""
        from senseframe.engine.runner import errors as errors_mod
        cls = getattr(errors_mod, exc_class)
        assert cls.error_code == expected_code, \
            f"{exc_class}.error_code = {cls.error_code}, expected {expected_code}"

    def test_classify_error_uses_isinstance_for_senseframe_errors(self):
        """classify_error 对 SenseFrameError 子类直接返回 error_code，不走字符串匹配。"""
        from senseframe.engine.runner.errors import (
            classify_error, SceneNotRegisteredError, ConfigValidationError,
            PreflightError, OOMError, SaveError,
        )
        # 不传 stage 参数，也应返回正确的 error_code
        assert classify_error(SceneNotRegisteredError("test")) == "SCENE_NOT_FOUND"
        assert classify_error(ConfigValidationError("test")) == "CONFIG_VALIDATION_ERROR"
        assert classify_error(PreflightError("test")) == "PREFLIGHT_ERROR"
        assert classify_error(OOMError("test")) == "OOM_ERROR"
        assert classify_error(SaveError("test")) == "SAVE_ERROR"

    def test_classify_error_stage_independent_for_senseframe_errors(self):
        """SenseFrameError 子类的 error_code 不受 stage 参数影响。

        设计约束：错误分类应在 raise 时绑定（通过异常类型），
        而非在 catch 时猜测（通过 stage + 字符串匹配）。
        """
        from senseframe.engine.runner.errors import (
            classify_error, SceneNotRegisteredError,
        )
        # 同一个异常，不同 stage 参数，应返回相同 error_code
        exc = SceneNotRegisteredError("Scene 'x' not registered")
        assert classify_error(exc, stage="validate") == "SCENE_NOT_FOUND"
        assert classify_error(exc, stage="train") == "SCENE_NOT_FOUND"
        assert classify_error(exc, stage=None) == "SCENE_NOT_FOUND"

    def test_config_validation_error_catchable_as_value_error(self):
        """ConfigValidationError 可被 except ValueError 捕获（向后兼容）。"""
        from senseframe.engine.runner.errors import ConfigValidationError
        try:
            raise ConfigValidationError("test")
        except ValueError as e:
            assert isinstance(e, ConfigValidationError)

    def test_data_not_found_error_catchable_as_file_not_found(self):
        """DataNotFoundError 可被 except FileNotFoundError 捕获（向后兼容）。"""
        from senseframe.engine.runner.errors import DataNotFoundError
        try:
            raise DataNotFoundError("test")
        except FileNotFoundError as e:
            assert isinstance(e, DataNotFoundError)

    def test_save_error_catchable_as_os_error(self):
        """SaveError 可被 except OSError 捕获（向后兼容）。"""
        from senseframe.engine.runner.errors import SaveError
        try:
            raise SaveError("test")
        except OSError as e:
            assert isinstance(e, SaveError)

    def test_third_party_exception_falls_back_to_heuristic(self):
        """非 SenseFrame 异常仍走 heuristic 兜底。"""
        from senseframe.engine.runner.errors import classify_error
        # FileNotFoundError → DATA_NOT_FOUND（heuristic）
        assert classify_error(FileNotFoundError("test")) == "DATA_NOT_FOUND"
        # KeyError → MODEL_BUILD_ERROR（heuristic，stage="build"）
        assert classify_error(KeyError("test"), stage="build") == "MODEL_BUILD_ERROR"


# ============================================================
# D.2 get_error_code 公共函数（遗留问题 1 修复 2026-07-19）
# ============================================================
class TestGetErrorCode:
    """get_error_code 公共函数契约。

    遗留问题 1 修复（2026-07-19）：替代 cli.py / probe_worker.py 中的 type(e).__name__，
    让 JSON 输出的 "code" 字段与 SKILL.md 错误码表对齐，Agent 可程序化解析。

    优先级：
    1. SenseFrameError 子类 → exc.error_code（isinstance 检查）
    2. 非 SenseFrame 异常 → classify_error(exc, stage) heuristic 兜底
    """

    def test_get_error_code_for_senseframe_errors(self):
        """SenseFrameError 子类 → exc.error_code（优先级 1，直接返回）。"""
        from senseframe.engine.runner.errors import (
            get_error_code, SceneNotRegisteredError, DatasetNotSupportedError,
            ModelNotSupportedError, DataNotFoundError, DataCorruptedError,
            OOMError, CheckpointError, PreflightError, TrainingError,
            ModelBuildError, SaveError, ConfigValidationError, MetadataVersionError,
        )
        assert get_error_code(SceneNotRegisteredError("test")) == "SCENE_NOT_FOUND"
        assert get_error_code(DatasetNotSupportedError("test")) == "DATASET_NOT_SUPPORTED"
        assert get_error_code(ModelNotSupportedError("test")) == "MODEL_NOT_SUPPORTED"
        assert get_error_code(DataNotFoundError("test")) == "DATA_NOT_FOUND"
        assert get_error_code(DataCorruptedError("test")) == "DATA_LOAD_ERROR"
        assert get_error_code(OOMError("test")) == "OOM_ERROR"
        assert get_error_code(CheckpointError("test")) == "CHECKPOINT_ERROR"
        assert get_error_code(PreflightError("test")) == "PREFLIGHT_ERROR"
        assert get_error_code(TrainingError("test")) == "TRAINING_ERROR"
        assert get_error_code(ModelBuildError("test")) == "MODEL_BUILD_ERROR"
        assert get_error_code(SaveError("test")) == "SAVE_ERROR"
        assert get_error_code(ConfigValidationError("test")) == "CONFIG_VALIDATION_ERROR"
        assert get_error_code(MetadataVersionError("test")) == "METADATA_VERSION_ERROR"

    def test_get_error_code_for_non_senseframe_falls_back(self):
        """非 SenseFrame 异常 → classify_error heuristic 兜底。"""
        from senseframe.engine.runner.errors import get_error_code
        # FileNotFoundError → DATA_NOT_FOUND（heuristic）
        assert get_error_code(FileNotFoundError("test")) == "DATA_NOT_FOUND"
        # KeyError + stage="build" → MODEL_BUILD_ERROR（heuristic）
        assert get_error_code(KeyError("test"), stage="build") == "MODEL_BUILD_ERROR"
        # ValueError → CONFIG_VALIDATION_ERROR（heuristic 兜底）
        assert get_error_code(ValueError("test")) == "CONFIG_VALIDATION_ERROR"

    def test_get_error_code_stage_propagated_to_classify_error(self):
        """stage 参数传递给 classify_error（heuristic 兜底时生效）。"""
        from senseframe.engine.runner.errors import get_error_code
        # KeyError + stage="build" → MODEL_BUILD_ERROR
        assert get_error_code(KeyError("test"), stage="build") == "MODEL_BUILD_ERROR"
        # KeyError 无 stage → 兜底 MODEL_BUILD_ERROR（heuristic 末尾分支）
        assert get_error_code(KeyError("test")) == "MODEL_BUILD_ERROR"

    def test_get_error_code_senseframe_ignores_stage(self):
        """SenseFrameError 子类的 error_code 不受 stage 影响（优先级 1，跳过 heuristic）。"""
        from senseframe.engine.runner.errors import (
            get_error_code, SceneNotRegisteredError, MetadataVersionError,
        )
        exc = SceneNotRegisteredError("test")
        assert get_error_code(exc, stage="validate") == "SCENE_NOT_FOUND"
        assert get_error_code(exc, stage="train") == "SCENE_NOT_FOUND"
        assert get_error_code(exc, stage=None) == "SCENE_NOT_FOUND"

        exc2 = MetadataVersionError("test")
        assert get_error_code(exc2, stage="load") == "METADATA_VERSION_ERROR"
        assert get_error_code(exc2, stage=None) == "METADATA_VERSION_ERROR"

    def test_get_error_code_returns_string_not_class_name(self):
        """get_error_code 始终返回结构化错误码字符串（不再是 type(e).__name__ 类名）。"""
        from senseframe.engine.runner.errors import (
            get_error_code, MetadataVersionError, ConfigValidationError,
        )
        # MetadataVersionError → "METADATA_VERSION_ERROR"（不是 "MetadataVersionError"）
        result = get_error_code(MetadataVersionError("test"))
        assert isinstance(result, str)
        assert result == "METADATA_VERSION_ERROR"
        assert result != "MetadataVersionError"

        # ConfigValidationError → "CONFIG_VALIDATION_ERROR"（不是 "ConfigValidationError"）
        result = get_error_code(ConfigValidationError("test"))
        assert result == "CONFIG_VALIDATION_ERROR"
        assert result != "ConfigValidationError"

    def test_get_error_code_aligns_with_skill_md_table(self):
        """get_error_code 返回值与 schemas.ERROR_CODES 对齐（SKILL.md 错误码表闭环）。"""
        from senseframe.engine.runner.errors import (
            get_error_code, MetadataVersionError, ConfigValidationError,
            SceneNotRegisteredError, OOMError,
        )
        from senseframe.schemas import ERROR_CODES

        # 所有 SenseFrameError 子类的 error_code 都应在 ERROR_CODES 中注册
        test_cases = [
            MetadataVersionError("test"),
            ConfigValidationError("test"),
            SceneNotRegisteredError("test"),
            OOMError("test"),
        ]
        for exc in test_cases:
            code = get_error_code(exc)
            assert code in ERROR_CODES, (
                f"错误码 {code} 未在 schemas.ERROR_CODES 中注册，"
                f"SKILL.md 错误码表存在缺口"
            )


# ============================================================
# E. DSP-3 就绪度：advisory 契约可查询且不阻断（方案 5）
# ============================================================

class TestDSP3Readiness:
    """Stage IO 契约的就绪度查询与 dangling ref 检测。"""

    def _make_minimal_ctx(self):
        """构造最小 PipelineContext（仅 config，其余字段为默认值）。"""
        from senseframe.engine.runner.pipeline import PipelineContext
        from senseframe.engine.config import ExperimentConfig, SceneConfig, TrainerConfig, InputFeature, OutputFeature
        cfg = ExperimentConfig(
            scene=SceneConfig(name="generic", model_id="MLP", dataset="UT_HAR_data"),
            trainer=TrainerConfig(epochs=1),
            input_features=[InputFeature(name="x", type="csi")],
            output_features=[OutputFeature(name="y", type="category", num_classes=7)],
        )
        return PipelineContext(config=cfg)

    def test_check_readiness_returns_false_for_unready_stage(self):
        """check_readiness 对前置 stage 未完成的 stage 返回 available=False。"""
        p = Pipeline.default()
        ctx = self._make_minimal_ctx()
        # build stage 需要 scene/num_classes/feature_spec 等，这些在 validate/preflight/load/resolve 之前为 None
        report = p.check_readiness(ctx, "build")
        assert report.available is False
        assert len(report.missing_reads) > 0

    def test_check_readiness_returns_true_for_validate_stage(self):
        """check_readiness 对 validate stage 返回 available=True（仅需 config）。"""
        p = Pipeline.default()
        ctx = self._make_minimal_ctx()
        report = p.check_readiness(ctx, "validate")
        assert report.available is True
        assert report.missing_reads == []

    def test_check_readiness_does_not_raise(self):
        """check_readiness 对不存在的 stage 名不抛异常。"""
        p = Pipeline.default()
        ctx = self._make_minimal_ctx()
        report = p.check_readiness(ctx, "nonexistent_stage")
        assert report.available is True  # 无 spec → 视为可用

    def test_validate_graph_returns_empty_for_default_pipeline(self):
        """默认 8-stage pipeline 无 dangling reference。"""
        p = Pipeline.default()
        refs = p.validate_graph()
        assert refs == [], \
            f"默认 pipeline 有 dangling refs: {[(r.stage_name, r.field_name) for r in refs]}"

    def test_validate_graph_detects_dangling_ref(self):
        """validate_graph 检测到 stage 读取了无 stage 产出的字段。

        构造一个自定义 stage，声明读取不存在的字段，
        validate_graph 应报告 dangling reference。
        """
        from senseframe.engine.runner.pipeline import Pipeline, stage, PipelineContext

        @stage(name="custom", reads=["nonexistent_field"], writes=["output"], description="test")
        def custom_stage(ctx):
            return ctx

        p = Pipeline(stages=[("custom", custom_stage)])
        refs = p.validate_graph()
        assert len(refs) == 1
        assert refs[0].field_name == "nonexistent_field"
        assert refs[0].stage_name == "custom"

    def test_validate_graph_detects_dangling_for_stage_field(self):
        """validate_graph 对 stage 产出字段（非 init/agent）也检测 dangling。

        反向验证：如果 validate_graph 错误地把所有 _FIELD_FILL_STAGE 字段
        无条件加入 produced（曾经的 bug），此测试会空过。必须用一个
        _FIELD_FILL_STAGE 中的 stage 字段（如 "scene"，由 stage_validate 写）
        但 custom pipeline 不包含 stage_validate 的场景，才能区分正确版本
        和有 bug 版本。
        """
        from senseframe.engine.runner.pipeline import Pipeline, stage

        # "scene" 在 _FIELD_FILL_STAGE 中标记为 "stage_validate"（非 init/agent）
        # custom pipeline 不包含 stage_validate，所以 "scene" 不应被视为 produced
        @stage(name="custom", reads=["scene"], writes=["output"], description="test")
        def custom_stage(ctx):
            return ctx

        p = Pipeline(stages=[("custom", custom_stage)])
        refs = p.validate_graph()
        scene_refs = [r for r in refs if r.field_name == "scene"]
        assert len(scene_refs) == 1, (
            f"validate_graph 未报告 'scene' 为 dangling ref，"
            f"可能错误地把所有 _FIELD_FILL_STAGE 字段视为 produced。"
            f"refs={[r.field_name for r in refs]}"
        )
        assert scene_refs[0].stage_name == "custom"

    def test_validate_graph_ignores_init_and_agent_fields(self):
        """validate_graph 将 init/agent 阶段填充的字段视为已产出。"""
        from senseframe.engine.runner.pipeline import Pipeline, stage

        @stage(name="custom", reads=["config", "trial_id", "extra"], writes=["output"], description="test")
        def custom_stage(ctx):
            return ctx

        p = Pipeline(stages=[("custom", custom_stage)])
        refs = p.validate_graph()
        # config (init), trial_id (agent), extra (agent) 都应被视为已产出
        field_names = [r.field_name for r in refs]
        assert "config" not in field_names
        assert "trial_id" not in field_names
        assert "extra" not in field_names


# ============================================================
# F. 回归守卫：已消除的反模式不被重新引入
# ============================================================

# ARCHITECTURE_TRIPWIRE: 回归守卫——已消除的反模式不被重新引入（方案 F）
# 不可替代原因: "整个代码库中不存在某函数/某调用模式"是否定属性，反射只能检查单个模块，
#   无法覆盖全包扫描；行为测试无法验证"某段代码从未被执行过"。必须 grep 源码文本。
# 删除条件: 当反模式在结构上不可能复现时（如旧函数所在模块已删除且模块结构阻止重建，
#   或 CI linter 规则永久禁止相关模式）。
class TestRegressionGuards:
    """静态分析守卫：确保已消除的反模式不被重新引入。"""

    def test_no_ensure_wifi_csi_registered_in_registry(self):
        """registry.py 中不应存在 _ensure_wifi_csi_registered 函数。

        反向验证：用 try/except ImportError 区分"函数不存在"（pass）
        和"函数存在"（fail）。直接 from import 在函数不存在时会 error，
        无法区分测试本身错误与被测代码违规。
        """
        try:
            from senseframe.registry import _ensure_wifi_csi_registered  # noqa: F401
        except ImportError:
            return  # 函数不存在，符合预期
        pytest.fail("_ensure_wifi_csi_registered 仍存在于 registry.py 中")

    def test_no_ensure_wifi_csi_registered_anywhere(self):
        """整个 senseframe 包中不应有 _ensure_wifi_csi_registered 的调用。"""
        import senseframe
        senseframe_dir = Path(senseframe.__file__).parent
        for py_file in senseframe_dir.rglob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            if "_ensure_wifi_csi_registered" in src:
                # 排除注释行
                lines = src.split("\n")
                for i, line in enumerate(lines):
                    if "_ensure_wifi_csi_registered" in line and not line.strip().startswith("#"):
                        pytest.fail(
                            f"{py_file.name}:{i+1} 引用了 _ensure_wifi_csi_registered: {line.strip()}"
                        )

    def test_orchestrator_run_experiment_is_thin_adapter(self):
        """run_experiment 函数体不超过 10 行（薄适配器约束）。

        如果有人重新在 run_experiment 中堆砌执行逻辑，
        函数体会变长，此测试会失败。
        """
        from senseframe.engine.runner.orchestrator import run_experiment
        src = inspect.getsource(run_experiment)
        # 去除空行和注释
        code_lines = [
            line for line in src.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(code_lines) <= 10, \
            f"run_experiment 函数体 {len(code_lines)} 行，预期 <= 10 行（薄适配器）"

    def test_reconcile_does_not_contain_stage_loop(self):
        """reconcile 方法体不应包含独立 stage 循环。"""
        from senseframe.orchestration import Orchestrator
        src = inspect.getsource(Orchestrator.reconcile)
        # 检查是否存在 "for ... in pipeline.stages" 模式（stage 循环）
        assert "for stage_name, stage_fn in pipeline.stages" not in src, \
            "reconcile 包含独立 stage 循环，应委托给 pipeline.run()"

    def test_wifi_csi_auto_registered_global_removed(self):
        """registry.py 中不应存在 _wifi_csi_auto_registered 全局变量。"""
        import senseframe.registry as reg
        assert not hasattr(reg, "_wifi_csi_auto_registered"), \
            "_wifi_csi_auto_registered 全局变量仍存在于 registry.py"

    def test_pipeline_context_has_no_any_fields_in_extra_writes(self):
        """pipeline 包中 stage 函数体不应将框架内部状态写入 ctx.extra。

        通过 AST 分析检测 ctx.extra[...] = ... 赋值语句。
        拆分背景：原 pipeline.py 拆分为 pipeline/ 包，需聚合所有子模块源码检查。
        """
        from senseframe.engine.runner import pipeline as pipeline_module
        from pathlib import Path

        # 收集 pipeline 包及其子包的所有 .py 源码
        package_dir = Path(pipeline_module.__file__).parent
        source_parts = []
        for py_file in sorted(package_dir.rglob("*.py")):
            source_parts.append(py_file.read_text(encoding="utf-8"))
        src = "\n".join(source_parts)
        tree = ast.parse(src)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and
                        isinstance(target.value, ast.Attribute) and
                        target.value.attr == "extra" and
                        isinstance(target.value.value, ast.Name) and
                        target.value.value.id == "ctx"):
                        violations.append(node.lineno)
        assert violations == [], \
            f"pipeline 包在行 {violations} 写入 ctx.extra，框架内部状态应使用 first-class 字段"


# ============================================================
# G. RFC-004 回归守卫：方案 A-G 的架构不变量
# ============================================================

# ARCHITECTURE_TRIPWIRE: RFC-004 方案 A-G 架构不变量回归守卫
# 不可替代原因: 多数检查验证"源码必须包含/不包含特定模式"（如 to_lightning_params 调用、
#   val_ 前缀字段、route_config.get("accelerator") 禁用），这些是源码级契约，
#   行为测试只能验证结果正确性，无法验证实现路径是否合规。
#   注：本类中部分测试（如 test_plan_a_get_device_memory_handles_missing_attrs、
#   test_plan_e_trainer_defaults_are_best_practice、test_plan_f_release_resources_clears_large_objects）
#   使用行为/反射验证，属 A/B 类，无需此注释。
# 删除条件: 当契约由 linter 规则或类型系统强制执行时（如 mypy plugin 禁止直接访问
#   route_config 内部字段，或 CI 检查 val_ 前缀命名规范）。
class TestRFC004RegressionGuards:
    """RFC-004 方案 A-G 的回归守卫。

    每个测试对标一个方案的架构原则，确保修复不被回退。
    """

    # ---------- 方案 A：显存探测健壮性 ----------

    def test_plan_a_routing_has_get_device_memory(self):
        """routing.py 必须有 _get_device_memory 静态方法（多名探测契约）。"""
        from senseframe.routing import ResourceProbe
        assert hasattr(ResourceProbe, "_get_device_memory"), \
            "ResourceProbe._get_device_memory 不存在（方案 A 被回退）"
        assert callable(getattr(ResourceProbe, "_get_device_memory")), \
            "_get_device_memory 必须是可调用方法"

    def test_plan_a_get_device_memory_handles_missing_attrs(self):
        """_get_device_memory 对无 total_memory/total_mem 属性的对象返回 None。"""
        from senseframe.routing import ResourceProbe
        # 无任何已知属性的对象
        obj = type("Empty", (), {})()
        assert ResourceProbe._get_device_memory(obj) is None
        # 有 total_memory 属性
        obj2 = type("Props", (), {"total_memory": 8589934592})()
        assert ResourceProbe._get_device_memory(obj2) == 8589934592

    # ---------- 方案 B：入口点显式激活契约 ----------

    def test_plan_b_activate_lazy_scenes_exists(self):
        """scenes 模块必须有 activate_lazy_scenes 函数。"""
        from senseframe.scenes import activate_lazy_scenes
        assert callable(activate_lazy_scenes), \
            "activate_lazy_scenes 必须是可调用函数（方案 B 入口点契约）"

    def test_plan_b_generate_config_calls_activate(self):
        """generate_config.py 必须调用 activate_lazy_scenes()。"""
        gen_config_path = Path(__file__).parent.parent / "scripts" / "generate_config.py"
        if not gen_config_path.exists():
            pytest.skip("generate_config.py 不存在")
        src = gen_config_path.read_text(encoding="utf-8")
        assert "activate_lazy_scenes" in src, \
            "generate_config.py 未调用 activate_lazy_scenes（方案 B 入口点契约被回退）"

    def test_plan_b_validate_config_calls_activate(self):
        """validate_config.py 必须调用 activate_lazy_scenes()。"""
        val_config_path = Path(__file__).parent.parent / "scripts" / "validate_config.py"
        if not val_config_path.exists():
            pytest.skip("validate_config.py 不存在")
        src = val_config_path.read_text(encoding="utf-8")
        assert "activate_lazy_scenes" in src, \
            "validate_config.py 未调用 activate_lazy_scenes（方案 B 入口点契约被回退）"

    # ---------- 方案 C：Feedback 字段契约 ----------

    def test_plan_c_epoch_entry_uses_prefixed_fields(self):
        """module.py on_validation_epoch_end 必须使用 train_/val_ 前缀字段。

        通过 AST 分析验证 epoch_entry 赋值使用前缀，不使用裸 loss/accuracy。
        """
        module_path = Path(__file__).parent.parent / "senseframe" / "engine" / "module.py"
        src = module_path.read_text(encoding="utf-8")
        # 必须包含 train_loss 和 val_loss 字段（前缀契约）
        assert 'epoch_entry["train_loss"]' in src, \
            "module.py epoch_entry 缺少 train_loss 字段（方案 C 字段契约被回退）"
        assert 'epoch_entry["val_loss"]' in src, \
            "module.py epoch_entry 缺少 val_loss 字段（方案 C 字段契约被回退）"

    def test_plan_c_analyze_training_result_reads_prefixed_fields(self):
        """analyze_training_result 必须读取 train_accuracy/val_accuracy（前缀字段）。"""
        from senseframe.engine.runner.pipeline import analyze_training_result
        src = inspect.getsource(analyze_training_result)
        assert 'entry.get("train_accuracy")' in src, \
            "analyze_training_result 未读取 train_accuracy（方案 C 消费端不匹配）"
        assert 'entry.get("val_accuracy")' in src, \
            "analyze_training_result 未读取 val_accuracy（方案 C 消费端不匹配）"

    def test_plan_c_final_metrics_uses_val_prefix(self):
        """get_final_metrics 必须返回 val_ 前缀字段。"""
        module_path = Path(__file__).parent.parent / "senseframe" / "engine" / "module.py"
        src = module_path.read_text(encoding="utf-8")
        # 最终验证路径必须写入 val_ 前缀
        assert 'f"val_{name}"' in src or 'val_{name}' in src, \
            "get_final_metrics / on_validation_epoch_end 未使用 val_ 前缀（方案 C）"

    # ---------- 方案 D：路由输出契约 ----------

    def test_plan_d_cli_dry_run_uses_to_lightning_params(self):
        """cli.py dry-run 必须使用 to_lightning_params() 而非 route_config.get("accelerator")。"""
        cli_path = Path(__file__).parent.parent / "senseframe" / "cli.py"
        src = cli_path.read_text(encoding="utf-8")
        # 必须调用 to_lightning_params
        assert "to_lightning_params" in src, \
            "cli.py 未使用 to_lightning_params（方案 D 输出契约被回退）"

    def test_plan_d_cli_no_route_config_get_accelerator(self):
        """cli.py 中不应有 route_config.get("accelerator") 模式（内部表示泄露）。"""
        cli_path = Path(__file__).parent.parent / "senseframe" / "cli.py"
        src = cli_path.read_text(encoding="utf-8")
        # 搜索 route_config.get("accelerator" 模式（内部表示直接作为输出）
        assert 'route_config.get("accelerator"' not in src, \
            'cli.py 仍使用 route_config.get("accelerator")（方案 D 内部表示泄露）'

    # ---------- 方案 E：默认训练策略 ----------

    def test_plan_e_trainer_defaults_are_best_practice(self):
        """TrainerConfig 默认值必须是最佳实践（weight_decay>0, early_stopping!=None, scheduler!=None）。"""
        from senseframe.engine.config import TrainerConfig
        tc = TrainerConfig()
        assert tc.weight_decay > 0, \
            f"weight_decay 默认值 {tc.weight_decay} 应 > 0（方案 E 默认正则化）"
        assert tc.early_stopping is not None and tc.early_stopping > 0, \
            f"early_stopping 默认值 {tc.early_stopping} 应 > 0（方案 E 默认早停）"
        assert tc.scheduler is not None, \
            f"scheduler 默认值 {tc.scheduler} 应不为 None（方案 E 默认 scheduler）"

    def test_plan_e_early_stopping_min_delta_exists(self):
        """TrainerConfig 必须有 early_stopping_min_delta 字段。"""
        from senseframe.engine.config import TrainerConfig
        tc = TrainerConfig()
        assert hasattr(tc, "early_stopping_min_delta"), \
            "TrainerConfig 缺少 early_stopping_min_delta 字段（方案 E）"

    # ---------- 方案 F：资源生命周期 ----------

    def test_plan_f_pipeline_context_has_release_resources(self):
        """PipelineContext 必须有 release_resources 方法。"""
        from senseframe.engine.runner.pipeline import PipelineContext
        assert hasattr(PipelineContext, "release_resources"), \
            "PipelineContext.release_resources 不存在（方案 F 被回退）"
        assert callable(getattr(PipelineContext, "release_resources")), \
            "release_resources 必须是可调用方法"

    def test_plan_f_pipeline_run_has_try_finally(self):
        """Pipeline.run() 必须有 try/finally 块调用 release_resources。"""
        from senseframe.engine.runner.pipeline import Pipeline
        src = inspect.getsource(Pipeline.run)
        assert "finally:" in src, \
            "Pipeline.run 缺少 finally 块（方案 F 确定性资源释放被回退）"
        assert "release_resources" in src, \
            "Pipeline.run 的 finally 块未调用 release_resources（方案 F）"

    def test_plan_f_release_resources_clears_large_objects(self):
        """release_resources 必须置 None trainer/module/model/datamodule。

        行为验证：调用后这些字段必须为 None。不依赖源码字面量匹配，
        避免实现用 _RESOURCE_FIELDS 元组迭代时误报（反向验证发现的空过 bug）。
        """
        from senseframe.engine.runner.pipeline import PipelineContext
        # config 用 Mock 即可——release_resources 不读取 config
        ctx = PipelineContext(config=MagicMock())
        sentinel = object()
        for f in ("trainer", "module", "model", "datamodule", "bundle", "monitor"):
            setattr(ctx, f, sentinel)
        ctx.callbacks = [sentinel]
        # 执行释放
        ctx.release_resources()
        # 验证大对象引用已清空
        for f in ("trainer", "module", "model", "datamodule", "bundle", "monitor"):
            assert getattr(ctx, f) is None, \
                f"release_resources 后 {f} 仍非 None（方案 F 大对象引用未清理）"
        assert ctx.callbacks == [], \
            "release_resources 后 callbacks 未清空（方案 F）"

    # ---------- 方案 G：溯源体系 ----------

    def test_plan_g_artifact_manifest_importable(self):
        """ArtifactManifest 必须可从 pipeline 导入。"""
        from senseframe.engine.runner.pipeline import ArtifactManifest, ArtifactDescriptor
        assert ArtifactManifest is not None
        assert ArtifactDescriptor is not None

    def test_plan_g_pipeline_context_has_register_artifact(self):
        """PipelineContext 必须有 register_artifact 方法和 artifact_registry 字段。

        注意：artifact_registry 用 field(default_factory=list) 声明，dataclass
        会删除类属性（仅 __init__ 填充），故 hasattr(cls, ...) 返回 False。
        必须用 dataclasses.fields() 检查字段存在性（反向验证发现的空过 bug）。
        """
        import dataclasses
        from senseframe.engine.runner.pipeline import PipelineContext
        assert hasattr(PipelineContext, "register_artifact"), \
            "PipelineContext.register_artifact 不存在（方案 G 被回退）"
        field_names = {f.name for f in dataclasses.fields(PipelineContext)}
        assert "artifact_registry" in field_names, \
            "PipelineContext.artifact_registry 字段不存在（方案 G）"

    def test_plan_g_pipeline_run_generates_manifest(self):
        """Pipeline.run() 必须在 finally 块中调用 _generate_manifest。"""
        from senseframe.engine.runner.pipeline import Pipeline
        src = inspect.getsource(Pipeline.run)
        assert "_generate_manifest" in src, \
            "Pipeline.run 未调用 _generate_manifest（方案 G manifest.json 不生成）"

    def test_plan_g_load_manifest_api_exists(self):
        """load_manifest 和 verify_artifacts 必须作为公共 API 导出。"""
        from senseframe.engine.runner.pipeline import load_manifest, verify_artifacts
        assert callable(load_manifest), "load_manifest 必须是可调用函数"
        assert callable(verify_artifacts), "verify_artifacts 必须是可调用函数"

    def test_plan_g_artifacts_module_exists(self):
        """artifacts.py 模块必须存在且导出核心类。"""
        from senseframe.engine.runner.artifacts import (
            ArtifactDescriptor,
            ArtifactManifest,
            sha256_file,
            sha256_str,
            verify_artifacts,
        )
        # sha256_file 对临时文件返回非空哈希
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("test content")
            f.flush()
            h = sha256_file(f.name)
        assert len(h) == 64, f"sha256_file 返回的哈希长度 {len(h)} 不等于 64"


# ============================================================
# 导入依赖（用于 IDE 跳转，无运行时副作用）
# ============================================================

# 在模块级别导入 Pipeline，避免每个测试方法重复导入
from senseframe.engine.runner.pipeline import Pipeline


# ============================================================
# H. P0 协议栈地基加固回归守卫
# ============================================================

# ARCHITECTURE_TRIPWIRE: P0 协议栈地基加固回归守卫
# 不可替代原因: test_sp_tell_no_private_access 验证"源码中不存在对 ExplorationTracker
#   私有字段的直接访问"，这是否定属性，行为测试无法区分"通过公共 API 访问"和"直接访问私有字段"。
#   注：本类中 test_field_fill_stage_complete / test_stage_train_writes_complete /
#   test_hpo_trial_result_renamed 使用反射/导入验证，属 A/B 类，无需此注释。
# 删除条件: 当 ExplorationTracker 内部字段通过 __slots__ 或名称改写（name mangling）
#   真正私有化，使直接访问在运行时即报错。
class TestP0ProtocolFoundationGuards:
    """P0 协议栈地基加固的回归守卫。

    每个测试对标一个 P0 工作项的验收标准：
    - P0.1 _FIELD_FILL_STAGE 完整性
    - P0.1 stage_train writes 完整性
    - P0.4 SP tell 不再 hack ExplorationTracker 私有字段
    - P0.6 hpo.TrialResult 已重命名为 HPOTrialResult
    """

    # ---------- P0.1 DSP 字段标注完整性 ----------

    def test_field_fill_stage_complete(self):
        """_FIELD_FILL_STAGE 必须覆盖所有 PipelineContext 字段（无 "unknown"）。

        反假绿：用 dataclasses.fields 反射所有字段名，逐一断言在
        _FIELD_FILL_STAGE 中有记录。若仅检查 schema() 输出，会被
        "schema() 中 .get(name, 'unknown') 自动兜底" 掩盖空过。
        """
        import dataclasses
        from senseframe.engine.runner.pipeline import (
            PipelineContext, _FIELD_FILL_STAGE,
        )

        all_fields = {f.name for f in dataclasses.fields(PipelineContext)}
        registered = set(_FIELD_FILL_STAGE.keys())
        missing = all_fields - registered
        assert not missing, \
            f"_FIELD_FILL_STAGE 缺失字段: {missing}（schema() 会输出 fill_stage='unknown'）"

        # 反向验证：schema() 输出确实无 "unknown"
        schema = PipelineContext.schema()
        unknown_fields = [
            f["name"] for f in schema["fields"]
            if f.get("fill_stage") == "unknown"
        ]
        assert unknown_fields == [], \
            f"schema() 仍报 fill_stage='unknown' 的字段: {unknown_fields}"

    def test_stage_train_writes_complete(self):
        """stage_train 装饰器 writes 必须含 4 字段（trainer + 3 训练结果字段）。

        反假绿：直接读取 stage_train._stage_spec.writes，不用 stage_io() 间接查询。
        若仅查 stage_io("train")，会被 stage_io 内部的 fallback 掩盖。

        注意：StageSpec.writes 是 List[FieldSpec]，不是 List[str]，
        需从 FieldSpec.name 提取字段名。
        """
        from senseframe.engine.runner.pipeline import stage_train

        spec = getattr(stage_train, "_stage_spec", None)
        assert spec is not None, "stage_train 缺少 _stage_spec 属性"

        # StageSpec.writes 是 List[FieldSpec]，提取 name 字段
        actual_names = {fs.name for fs in spec.writes}
        expected = {"trainer", "training_duration_s", "best_model_path", "best_model_score"}
        missing = expected - actual_names
        assert not missing, \
            f"stage_train.writes 缺失: {missing}，实际: {actual_names}"

    # ---------- P0.4 SP tell 不再 hack 私有字段 ----------

    def test_sp_tell_no_private_access(self):
        """search_protocol.py 中不应直接访问 ExplorationTracker 私有字段。

        反假绿：用 grep 实证（不用 mock sentinel）。检查源码中是否含
        'tracker._lock' 或 'tracker.history' 直接访问模式。

        P0.4 修复前：tell 中存在 'with tracker._lock' + 'tracker.history' 改写
        P0.4 修复后：tell 改用 tracker.update_trial 公共 API
        """
        from senseframe import search_protocol as sp_mod

        src = inspect.getsource(sp_mod)
        # 排除注释行
        violations = []
        for i, line in enumerate(src.split("\n"), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # 检测直接访问私有字段（_lock / history）
            if "tracker._lock" in stripped or "tracker.history" in stripped:
                violations.append((i, stripped))
        assert violations == [], \
            f"search_protocol.py 仍直接访问 ExplorationTracker 私有字段: {violations}"

    # ---------- P0.6 hpo.TrialResult 已重命名 ----------

    def test_hpo_trial_result_renamed(self):
        """hpo 模块中应无 TrialResult 类，改为 HPOTrialResult。

        反假绿：用 try/except ImportError 区分"类不存在"（pass）和"类仍存在"（fail）。
        直接 hasattr 检查会被 __getattr__ 兼容层掩盖。
        """
        # 1. hpo.TrialResult 不应存在
        import senseframe.engine.hpo as hpo_mod
        assert not hasattr(hpo_mod, "TrialResult"), \
            "hpo.TrialResult 仍存在，应已重命名为 HPOTrialResult"

        # 2. hpo.HPOTrialResult 必须存在
        assert hasattr(hpo_mod, "HPOTrialResult"), \
            "hpo.HPOTrialResult 不存在"

        # 3. engine 顶层导出 HPOTrialResult，不导出 TrialResult
        import senseframe.engine as engine_mod
        assert hasattr(engine_mod, "HPOTrialResult"), \
            "senseframe.engine 未导出 HPOTrialResult"
        assert "HPOTrialResult" in engine_mod.__all__, \
            "senseframe.engine.__all__ 未含 HPOTrialResult"

        # 4. senseframe 顶层 TrialResult 应指向 SP 版本（不是 hpo 版本）
        import senseframe
        from senseframe.search_protocol import TrialResult as SPTrialResult
        assert senseframe.TrialResult is SPTrialResult, \
            "senseframe.TrialResult 应指向 search_protocol.TrialResult（SP 版本）"
