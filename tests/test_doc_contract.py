"""文档契约 CI 检查：从代码 dataclass fields / 函数签名自动派生 API 契约，
与文档中引用的字段名/参数名/枚举值对比，发现不一致即报错。

设计原则（建立"文档生成从代码自动派生"机制，避免再次漂移）：
- 不解析 markdown 全文，只检查文档中"代码契约相关"的关键片段
  （dataclass 字段列表 / 函数参数 / 错误码表 / stage 名 / 字段→stage 映射）
- 每个检查函数返回 (doc_path, line, expected, actual) 四元组列表
- 任一不一致 → pytest.fail，CI 阻断合并

覆盖范围：
1. TrainerConfig 字段名（不应出现 max_epochs）
2. DatasetSpec 字段名（应含 layout）
3. DataProfile 字段列表完整性
4. 错误码表完整性（应含 UNKNOWN_ERROR）
5. stage 名（不应带 stage_ 前缀）
6. ExperimentConfig 推荐构造方式（from_dict）
7. 已删除的常量不应出现在文档中（DEFAULT_DATA_ROOT / WiFi-CSI-Sensing-Benchmark 硬编码）
"""
from __future__ import annotations

import re
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT  # 文档分布在项目根（SKILL.md）和 commands/、reference/ 子目录


def _field_names(cls):
    """统一获取 dataclass / pydantic v2 BaseModel 的字段名集合。

    P1 演进（2026-07-18）：TrainerConfig/InputFeature/OutputFeature 已迁移到
    pydantic v2 BaseModel，用 model_fields 替代 dataclasses.fields。
    """
    if hasattr(cls, "model_fields"):  # pydantic v2
        return list(cls.model_fields.keys())
    return [f.name for f in dataclass_fields(cls)]

# 受检文档清单
DOC_FILES = [
    PROJECT_ROOT / "SKILL.md",
    PROJECT_ROOT / "commands" / "senseframe-full.md",
    PROJECT_ROOT / "reference" / "introspect.md",
    PROJECT_ROOT / "commands" / "senseframe-hpo.md",
    PROJECT_ROOT / "commands" / "senseframe-train.md",
    PROJECT_ROOT / "reference" / "config_schema.md",
]


def _read_doc(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"文档不存在: {path}")
    return path.read_text(encoding="utf-8")


def _doc_lines(path: Path) -> list:
    if not path.exists():
        pytest.skip(f"文档不存在: {path}")
    return path.read_text(encoding="utf-8").splitlines()


# ============================================================
# 1. TrainerConfig 字段名契约
# ============================================================
def test_trainer_config_no_max_epochs_in_docs():
    """文档不应出现 max_epochs（TrainerConfig 字段是 epochs）。"""
    from senseframe.engine.config import TrainerConfig
    actual_fields = set(_field_names(TrainerConfig))
    assert "max_epochs" not in actual_fields, "TrainerConfig 不应有 max_epochs 字段"
    assert "epochs" in actual_fields, "TrainerConfig 应有 epochs 字段"

    violations = []
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            continue
        for i, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
            # 排除注释行（说明"旧文档用 max_epochs"等）
            if line.strip().startswith("#") or "旧文档" in line or "旧逻辑" in line:
                continue
            # 检查 max_epochs=<> 或 max_epochs: 等用法（非引用历史）
            if re.search(r"\bmax_epochs\b\s*[=:]", line):
                violations.append((doc_path.name, i, line.strip()))
    if violations:
        pytest.fail(
            "文档中发现了 max_epochs 用法（应为 epochs）：\n"
            + "\n".join(f"  {n}:{ln}: {l}" for n, ln, l in violations)
        )


# ============================================================
# 2. DatasetSpec 字段名契约
# ============================================================
def test_dataset_spec_has_layout_field():
    """DatasetSpec 应有 layout 字段（P0-1.7 修复后新增）。"""
    from senseframe.registry import DatasetSpec
    actual_fields = {f.name for f in dataclass_fields(DatasetSpec)}
    assert "layout" in actual_fields, "DatasetSpec 应有 layout 字段"


# ============================================================
# 3. DataProfile 字段列表完整性
# ============================================================
def test_dataprofile_fields_complete_in_introspect_md():
    """introspect.md 中列出的 DataProfile 字段应与 dataclass 完全一致。"""
    from senseframe.core.profiler import DataProfile
    actual_fields = {f.name for f in dataclass_fields(DataProfile)}

    doc_path = PROJECT_ROOT / "reference" / "introspect.md"
    content = _read_doc(doc_path)
    # 匹配 `DataProfile` 实例字段：`xxx`, `yyy`, ...
    m = re.search(r"`DataProfile` 实例字段[：:]\s*(.+?)。", content)
    if not m:
        pytest.fail("introspect.md 中未找到 DataProfile 实例字段列表")
    listed = set(re.findall(r"`(\w+)`", m.group(1)))

    missing = actual_fields - listed
    extra = listed - actual_fields
    if missing or extra:
        msg = []
        if missing:
            msg.append(f"文档缺失字段: {sorted(missing)}")
        if extra:
            msg.append(f"文档多余字段: {sorted(extra)}")
        pytest.fail(
            "introspect.md 的 DataProfile 字段列表与 dataclass 不一致：\n"
            + "\n".join(f"  {m}" for m in msg)
            + f"\n  实际 dataclass 字段: {sorted(actual_fields)}"
        )


# ============================================================
# 4. 错误码表完整性
# ============================================================
def test_error_codes_complete_in_skill_md():
    """SKILL.md 错误码表应包含代码中所有 error_code。

    注：ERROR_CODES 字典中的 "OK" 是成功状态码（无对应异常类），
    不应出现在错误码表中，故排除。
    """
    from senseframe.schemas import ERROR_CODES
    expected_codes = set(ERROR_CODES.keys()) - {"OK"}  # OK 不是错误码

    doc_path = PROJECT_ROOT / "SKILL.md"
    content = _read_doc(doc_path)
    # 匹配错误码表中的 `CODE` 行
    listed = set(re.findall(r"\|\s*`([A-Z_]+)`\s*\|", content))

    missing = expected_codes - listed
    if missing:
        pytest.fail(
            f"SKILL.md 错误码表缺失: {sorted(missing)}\n"
            f"  文档列出: {sorted(listed)}\n"
            f"  代码定义: {sorted(expected_codes)}"
        )


# ============================================================
# 5. stage 名不应带 stage_ 前缀
# ============================================================
def test_no_stage_prefix_in_docs():
    """文档中 stage_io / failed_stage 等引用不应使用 stage_xxx 前缀。"""
    # 实际 stage 名（从 pipeline 装饰器派生）
    from senseframe.engine.runner.pipeline import Pipeline
    actual_stages = set()
    for name in dir(Pipeline):
        attr = getattr(Pipeline, name, None)
        # stage 方法通常以 stage_ 开头，但 stage 名是装饰器参数
    # 直接硬编码已知 stage 名（与 @stage(name=...) 一致）
    known_stages = {"validate", "preflight", "resolve", "load", "build",
                    "probe_vram", "train", "eval", "export", "analyze"}

    violations = []
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            continue
        for i, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
            # 匹配 stage_io("stage_xxx") 或 失败 stage: stage_xxx
            m = re.search(r'stage_io\(\s*"(stage_\w+)"\s*\)', line)
            if m:
                violations.append((doc_path.name, i, line.strip(),
                                   f"应用 stage_io(\"{m.group(1).removeprefix('stage_')}\")"))
            m = re.search(r"失败 stage:\s*(stage_\w+)", line)
            if m:
                violations.append((doc_path.name, i, line.strip(),
                                   f"应用 失败 stage: {m.group(1).removeprefix('stage_')}"))
            # 模拟 stage_xxx 失败
            if re.search(r"模拟\s*stage_\w+\s*失败", line):
                violations.append((doc_path.name, i, line.strip(),
                                   "stage 名不应带 stage_ 前缀"))
    if violations:
        pytest.fail(
            "文档中发现了带 stage_ 前缀的 stage 名引用：\n"
            + "\n".join(f"  {n}:{ln}: {l} ({s})" for n, ln, l, s in violations)
        )


# ============================================================
# 6. ExperimentConfig 推荐构造方式
# ============================================================
def test_experiment_config_from_dict_in_docs():
    """senseframe-full.md 应推荐 ExperimentConfig.from_dict() 构造方式。"""
    doc_path = PROJECT_ROOT / "commands" / "senseframe-full.md"
    content = _read_doc(doc_path)
    # 应出现 from_dict 调用
    if "ExperimentConfig.from_dict" not in content:
        pytest.fail(
            "senseframe-full.md 未推荐 ExperimentConfig.from_dict() 构造方式。"
            "应使用 from_dict(config_dict) + validate() 替代直接 dataclass 构造。"
        )


# ============================================================
# 7. 已删除的常量不应出现在文档中
# ============================================================
def test_no_deleted_constants_in_docs():
    """文档不应引用已删除的常量/硬编码。"""
    forbidden_patterns = [
        # DEFAULT_DATA_ROOT 已删除（路径解耦后改为 scene.data_root 必填）
        (r"\bDEFAULT_DATA_ROOT\b", "DEFAULT_DATA_ROOT 常量已删除，改用 scene.data_root"),
        # WiFi-CSI-Sensing-Benchmark 硬编码已删除（改用 SENSEFRAME_SENSEFI_PATH env）
        # 仅检查"路径硬编码"用法，不检查"修复说明"注释
        (r'["\']WiFi-CSI-Sensing-Benchmark["\']', "WiFi-CSI-Sensing-Benchmark 路径硬编码已删除，改用 SENSEFRAME_SENSEFI_PATH env"),
        # /Data 后缀已删除（实际 CSI_DATASETS/ 下直接是各数据集子目录）
        (r'CSI_DATASETS["\']?\s*/\s*["\']?Data["\']?', "/Data 后缀已删除"),
    ]

    violations = []
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            continue
        for i, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
            # 排除注释行（说明"已删除"等）
            stripped = line.strip()
            if stripped.startswith("#") or "已删除" in line or "旧文档" in line or "旧逻辑" in line:
                continue
            for pattern, reason in forbidden_patterns:
                if re.search(pattern, line):
                    violations.append((doc_path.name, i, line.strip(), reason))
    if violations:
        pytest.fail(
            "文档中引用了已删除的常量/硬编码：\n"
            + "\n".join(f"  {n}:{ln}: {l} ({r})" for n, ln, l, r in violations)
        )


# ============================================================
# 8. 字段→stage 映射完整性（introspect.md）
# ============================================================
def test_field_stage_mapping_in_introspect_md():
    """introspect.md 的字段→stage 映射表应与代码 _FIELD_FILL_STAGE 一致。"""
    from senseframe.engine.runner.pipeline import _FIELD_FILL_STAGE
    expected_fields = set(_FIELD_FILL_STAGE.keys())

    doc_path = PROJECT_ROOT / "reference" / "introspect.md"
    content = _read_doc(doc_path)
    # 匹配映射表中的字段名（表格行 | field_name | ... |）
    # 简化：匹配反引号包裹的字段名
    # 实际表格格式可能多样，这里只检查"已知字段"是否被提及
    listed = set(re.findall(r"`(\w+)`\s*$", content, re.MULTILINE))
    # 实际更稳健：检查 _FIELD_FILL_STAGE 的所有 key 是否在文档中出现
    missing = []
    for field in expected_fields:
        if f"`{field}`" not in content:
            missing.append(field)
    if missing:
        pytest.fail(
            f"introspect.md 字段→stage 映射表缺失字段: {sorted(missing)}\n"
            f"  代码 _FIELD_FILL_STAGE 定义了 {len(expected_fields)} 个字段"
        )


# ============================================================
# 9. SceneConfig.data_root 必填契约
# ============================================================
def test_scene_config_data_root_required_in_docs():
    """文档应说明 scene.data_root 必填（路径解耦后的契约）。"""
    doc_path = PROJECT_ROOT / "commands" / "senseframe-full.md"
    content = _read_doc(doc_path)
    # 应出现 data_root 必填说明
    if "data_root" not in content:
        pytest.fail(
            "senseframe-full.md 未提及 scene.data_root。"
            "路径解耦后 data_root 是必填字段，文档应说明提供方式（YAML/CLI/env 三选一）。"
        )


# ============================================================
# 10. SENSEFRAME_SENSEFI_PATH env 契约
# ============================================================
def test_sensefi_path_env_in_docs():
    """文档应说明 SENSEFRAME_SENSEFI_PATH 环境变量（SenseFi 路径解耦后的契约）。"""
    # SKILL.md 或 reference/ 下应提及此 env
    docs_to_check = [
        PROJECT_ROOT / "SKILL.md",
        PROJECT_ROOT / "reference" / "datasets_and_models.md",
        PROJECT_ROOT / "reference" / "scene_development.md",
    ]
    found = False
    for doc_path in docs_to_check:
        if not doc_path.exists():
            continue
        if "SENSEFRAME_SENSEFI_PATH" in doc_path.read_text(encoding="utf-8"):
            found = True
            break
    if not found:
        pytest.fail(
            "文档未提及 SENSEFRAME_SENSEFI_PATH 环境变量。"
            "SenseFi 路径解耦后，框架不猜测路径，调用者必须设置此 env。"
            "应在 SKILL.md 或 reference/datasets_and_models.md 中说明。"
        )


# ============================================================
# 11. postprocess.py 参数契约（P0-1.5 路径安全修复后仅 --output-dir）
# ============================================================
def test_postprocess_no_deleted_params_in_docs():
    """文档不应再出现 postprocess.py 的 --models-dir / --result-dir / --eval-script 参数。

    P0-1.5 路径安全修复后，postprocess.py 仅接受 --output-dir，
    所有产物均在 output_dir 内，manifest 存相对路径。
    """
    forbidden_params = [
        (r"--models-dir\b", "--models-dir 参数已删除（P0-1.5 后 postprocess 仅接受 --output-dir）"),
        (r"--result-dir\b", "--result-dir 参数已删除（P0-1.5 后 postprocess 仅接受 --output-dir）"),
        (r"--eval-script\b", "--eval-script 参数已删除（P0-1.5 后 eval.py 自动生成到 output_dir/eval.py）"),
    ]

    violations = []
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            continue
        for i, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), 1):
            # 排除注释行（说明"已删除"等）
            stripped = line.strip()
            if stripped.startswith("#") or "已删除" in line or "旧文档" in line or "旧逻辑" in line:
                continue
            for pattern, reason in forbidden_params:
                if re.search(pattern, line):
                    violations.append((doc_path.name, i, line.strip(), reason))
    if violations:
        pytest.fail(
            "文档中引用了已删除的 postprocess.py 参数：\n"
            + "\n".join(f"  {n}:{ln}: {l} ({r})" for n, ln, l, r in violations)
        )


# ============================================================
# 12. metadata.config 完整性契约（stage_export 合并声明式配置 + 路由解析值）
# ============================================================
def test_metadata_config_completeness_in_docs():
    """文档应说明 metadata.config 是完整配置快照（含 data_root + epochs 等复现必需字段）。"""
    doc_path = PROJECT_ROOT / "SKILL.md"
    content = _read_doc(doc_path)
    # 步骤 10 后处理说明中应提及 metadata.config 含 data_root
    required_mentions = ["metadata.config", "data_root"]
    missing = [k for k in required_mentions if k not in content]
    if missing:
        pytest.fail(
            f"SKILL.md 未说明 metadata.config 含 {missing}。"
            "metadata.config 是完整配置快照（experiment_config_to_dict + ctx.resolved 合并），"
            "供实验复现与推理脚本生成使用。应在步骤 10（后处理与导出）中说明。"
        )


def test_metadata_config_contains_repro_fields():
    """代码契约：experiment_config_to_dict 输出应含复现必需字段。

    metadata.config = {**experiment_config_to_dict(ctx.config), **ctx.resolved}，
    因此 experiment_config_to_dict 的输出字段即 metadata.config 的声明式配置部分。
    复现必需字段（epochs/seed/data_root/learning_mode/...）必须全部在其中。
    """
    from senseframe.engine.config import ExperimentConfig, SceneConfig, TrainerConfig
    from senseframe.engine.runner.resolver import experiment_config_to_dict

    # 构造最小合法配置（data_root 仅作占位符，不触发文件访问）
    import tempfile
    from pathlib import Path
    config = ExperimentConfig(
        scene=SceneConfig(
            name="test", dataset="UT_HAR_data", model_id="ResNet18",
            learning_mode="supervised", data_root=str(Path(tempfile.gettempdir()) / "sf_test_data"),
        ),
        input_features=[],
        output_features=[],
        trainer=TrainerConfig(),
    )
    d = experiment_config_to_dict(config)

    # 复现必需字段（来自 TrainerConfig + SceneConfig）
    required_fields = [
        "epochs", "seed", "deterministic", "learning_rate", "batch_size",
        "optimizer", "weight_decay", "early_stopping", "scheduler",
        "data_root", "model_id", "dataset", "learning_mode",
    ]
    missing = [f for f in required_fields if f not in d]
    if missing:
        pytest.fail(
            f"experiment_config_to_dict 输出缺失复现必需字段: {missing}。"
            "这些字段通过 experiment_config_to_dict(ctx.config) + ctx.resolved 合并 "
            "进入 metadata.config，缺失会导致实验无法复现。"
        )


# ============================================================
# 13. Stage IO 声明一致性（reads/writes 字段合法性）
# ============================================================
def test_stage_io_reads_writes_consistency():
    """stage 装饰器的 reads/writes 声明应与 _FIELD_FILL_STAGE 字段对齐。

    每个 @stage 装饰器声明的 reads/writes 中的字段名都应在
    _FIELD_FILL_STAGE 中有定义。此测试检查"声明字段是否合法"，
    不检查"声明是否完整"（完整性检查需要 AST 分析，太复杂）。
    """
    from senseframe.engine.runner.pipeline import _FIELD_FILL_STAGE, Pipeline

    known_fields = set(_FIELD_FILL_STAGE.keys())

    # 通过 Pipeline.default().stages_with_spec() 获取所有 stage 的 StageSpec
    pipeline = Pipeline.default()
    specs = pipeline.stages_with_spec()

    unknown_fields = []
    for spec in specs:
        for field_spec in spec.reads:
            if field_spec.name not in known_fields:
                unknown_fields.append((spec.name, "reads", field_spec.name))
        for field_spec in spec.writes:
            if field_spec.name not in known_fields:
                unknown_fields.append((spec.name, "writes", field_spec.name))

    if unknown_fields:
        pytest.fail(
            "stage 装饰器声明了未在 _FIELD_FILL_STAGE 中定义的字段：\n"
            + "\n".join(
                f"  stage='{s}', direction={d}, field='{f}'"
                for s, d, f in unknown_fields
            )
            + f"\n  已知字段({len(known_fields)}个): {sorted(known_fields)}"
        )


# ============================================================
# 14. data_hash 计算与写入 manifest 契约
# ============================================================
def test_data_hash_computed_in_pipeline():
    """stage_load 应计算 data_hash，_generate_manifest 应写入 manifest。

    旧逻辑：manifest.data_hash 恒为空字符串，未计算数据集内容哈希，
    导致数据集变更/损坏无法被溯源检测。

    修复后契约：
    - PipelineContext 有 data_hash 字段
    - stage_load 中计算 ctx.data_hash（元数据哈希：路径+大小+mtime）
    - _generate_manifest 从 ctx.data_hash 读取（不再恒定空字符串）

    拆分背景：原 pipeline.py 拆分为 pipeline/ 包，
    stage_load 位于 pipeline/stages/load.py，
    _generate_manifest 位于 pipeline/stages/export.py。
    """
    load_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "load.py"
    export_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "export.py"
    if not load_path.exists():
        pytest.skip(f"pipeline/stages/load.py 不存在: {load_path}")
    if not export_path.exists():
        pytest.skip(f"pipeline/stages/export.py 不存在: {export_path}")
    load_source = load_path.read_text(encoding="utf-8")
    export_source = export_path.read_text(encoding="utf-8")

    # 1. 验证 data_hash 字段存在于代码中
    if "data_hash" not in load_source:
        pytest.fail("pipeline/stages/load.py 未包含 data_hash 字段定义")

    # 2. 验证 stage_load 中计算 data_hash（ctx.data_hash 赋值）
    if "ctx.data_hash" not in load_source:
        pytest.fail(
            "pipeline/stages/load.py 未在 stage_load 中计算 ctx.data_hash。"
            "stage_load 应在数据加载后计算数据集元数据哈希并赋值给 ctx.data_hash。"
        )

    # 3. 验证 _generate_manifest 引用 ctx.data_hash（不再恒定空字符串）
    manifest_start = export_source.find("def _generate_manifest")
    if manifest_start == -1:
        pytest.fail("pipeline/stages/export.py 未找到 _generate_manifest 函数")
    manifest_section = export_source[manifest_start:]
    if "ctx.data_hash" not in manifest_section:
        pytest.fail(
            "pipeline/stages/export.py 的 _generate_manifest 未从 ctx.data_hash 读取 data_hash，"
            "manifest.data_hash 仍为空字符串，数据集变更/损坏无法被溯源检测。"
        )


# ============================================================
# 15. best_epoch 持久化契约（Part 2）
# ============================================================
def test_best_epoch_persisted_in_metadata():
    """stage_export 应将 best_epoch/best_model_path/best_model_score/epoch_utilization 写入 metadata。

    旧逻辑：stage_train 写入 ctx.best_epoch，stage_eval 消费（内存中通），
    但 stage_export 未持久化到 metadata.json，pipeline_checkpoint.json 也缺失，
    导致下游消费者（generate_inference）和断点续跑无法获取 best checkpoint 信息。

    修复后契约（Part 2）：
    - stage_export 的 metadata dict 含 best_epoch/best_model_path/best_model_score/epoch_utilization
    - _serialize_stage_outputs 的 simple_fields 含 best_epoch

    拆分背景：原 pipeline.py 拆分为 pipeline/ 包，
    stage_export 位于 pipeline/stages/export.py，
    _serialize_stage_outputs 位于 pipeline/runtime.py。
    """
    export_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "export.py"
    runtime_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "runtime.py"
    if not export_path.exists():
        pytest.skip(f"pipeline/stages/export.py 不存在: {export_path}")
    if not runtime_path.exists():
        pytest.skip(f"pipeline/runtime.py 不存在: {runtime_path}")
    export_source = export_path.read_text(encoding="utf-8")
    runtime_source = runtime_path.read_text(encoding="utf-8")

    # 验证 metadata dict 含 4 个字段
    export_start = export_source.find("def stage_export")
    if export_start == -1:
        pytest.fail("pipeline/stages/export.py 未找到 stage_export 函数")
    export_section = export_source[export_start:]
    for field in ["best_epoch", "best_model_path", "best_model_score", "epoch_utilization"]:
        if field not in export_section:
            pytest.fail(
                f"stage_export 未将 {field} 写入 metadata dict。"
                f"best checkpoint 溯源信息必须持久化到 metadata.json。"
            )

    # 验证 _serialize_stage_outputs 含 best_epoch
    serialize_start = runtime_source.find("def _serialize_stage_outputs")
    if serialize_start == -1:
        pytest.fail("pipeline/runtime.py 未找到 _serialize_stage_outputs 方法")
    serialize_section = runtime_source[serialize_start:]
    if "best_epoch" not in serialize_section:
        pytest.fail(
            "_serialize_stage_outputs 的 simple_fields 未包含 best_epoch。"
            "pipeline_checkpoint.json 必须持久化 best_epoch 供断点续跑使用。"
        )


# ============================================================
# 16. analyze_training_result 用 .get() 访问 entry（Part 3 风险推演 R2）
# ============================================================
def test_analyze_training_result_uses_get():
    """analyze_training_result 对 training_log entry 的访问必须用 .get() 而非 []。

    Part 3 引入 phase 字段后，epoch 0 的 entry 可能无 val_loss 键（to_dict 省略 None 字段），
    final_eval 的 entry 可能无 train_loss 键。用 [] 访问会 KeyError。
    .get() 是防御性编程契约。

    拆分背景：原 pipeline.py 拆分为 pipeline/ 包，
    analyze_training_result 位于 pipeline/stages/train.py。
    """
    pipeline_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "train.py"
    if not pipeline_path.exists():
        pytest.skip(f"pipeline/stages/train.py 不存在: {pipeline_path}")
    source = pipeline_path.read_text(encoding="utf-8")

    func_start = source.find("def analyze_training_result")
    if func_start == -1:
        pytest.fail("pipeline/stages/train.py 未找到 analyze_training_result 函数")
    func_section = source[func_start:source.find("\ndef ", func_start + 1)]

    # 检查是否有 entry["xxx"] 形式的访问（不允许，注释和 docstring 除外）
    import re
    bracket_access = []
    in_docstring = False
    for line in func_section.splitlines():
        stripped = line.strip()
        # 跳过注释行
        if stripped.startswith("#"):
            continue
        # 跳过 docstring 内容（简单状态机）
        if '"""' in stripped:
            if stripped.count('"""') == 2:
                continue  # 单行 docstring
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        m = re.findall(r'entry\["(\w+)"\]', line)
        if m:
            bracket_access.extend(m)
    if bracket_access:
        pytest.fail(
            f"analyze_training_result 用 [] 访问 entry 字段: {bracket_access}。"
            f"必须用 .get() 访问，因为 phase=train_only/final_eval 的 entry 可能缺失部分字段。"
        )


# ============================================================
# 17. epochs 预算公式契约（Part 1 + 方案 B 去静态化）
# ============================================================
def test_epochs_budget_formula():
    """_compute_epochs_budget 应为单公式，无模型范式系数，无 None 回退。

    方案 B 去静态化后：
    - _compute_epochs_budget(None) 应 raise ValueError（不再回退 100）
    - get_default_epochs 无 n_samples 时应 raise ValueError（不再查静态表）
    - _EPOCHS_TABLE / _SCENE_EPOCHS 应为空（静态表已删除）
    """
    from senseframe.registry import (
        _compute_epochs_budget, get_default_epochs,
        _EPOCHS_TABLE, _SCENE_EPOCHS,
    )

    # 验证签名不含 model_id
    import inspect
    sig = inspect.signature(_compute_epochs_budget)
    if "model_id" in sig.parameters:
        pytest.fail(
            "_compute_epochs_budget 不应接受 model_id 参数。"
            "模型差异由 Early Stopping 实时吸收，无需预测。"
        )

    # 验证公式行为
    assert _compute_epochs_budget(3977) == 37, f"UT_HAR(3977) 应为 37，实际 {_compute_epochs_budget(3977)}"
    assert _compute_epochs_budget(5000) == 30, f"Widar(5000) 应为 30，实际 {_compute_epochs_budget(5000)}"
    assert _compute_epochs_budget(700) == 214, f"NTU-Fi_HAR(700) 应为 214，实际 {_compute_epochs_budget(700)}"
    assert _compute_epochs_budget(100) == 300, f"小数据集(100) 应为 300（上限），实际 {_compute_epochs_budget(100)}"
    assert _compute_epochs_budget(50000) == 30, f"大数据集(50000) 应为 30（下限），实际 {_compute_epochs_budget(50000)}"

    # 方案 B：None 应 raise（不再回退 100）
    with pytest.raises(ValueError, match="n_samples required"):
        _compute_epochs_budget(None)
    with pytest.raises(ValueError, match="n_samples required"):
        _compute_epochs_budget(0)

    # 方案 B：get_default_epochs 无 n_samples 应 raise
    with pytest.raises(ValueError, match="requires n_samples"):
        get_default_epochs("ResNet18", "UT_HAR_data")

    # 方案 B：静态表应为空（set_scene_epochs/set_default_epochs 已废弃为 no-op）
    assert len(_EPOCHS_TABLE) == 0, f"_EPOCHS_TABLE 应为空（方案 B 去静态化），实际 {len(_EPOCHS_TABLE)} 条"
    assert len(_SCENE_EPOCHS) == 0, f"_SCENE_EPOCHS 应为空（方案 B 去静态化），实际 {len(_SCENE_EPOCHS)} 条"

    # 方案 B：get_default_epochs 有 n_samples 应返回动态预算
    assert get_default_epochs("ResNet18", "UT_HAR_data", n_samples=3977) == 37
    assert get_default_epochs("ResNet18", "UT_HAR_data", n_samples=3977, scene_name="wifi_csi") == 37


# ============================================================
# 18. auto_lr_find 字段契约（Part 4）
# ============================================================
def test_auto_lr_find_in_trainer_config():
    """TrainerConfig 应含 auto_lr_find 字段（Part 4: 自动 LR 标定）。"""
    from senseframe.engine.config import TrainerConfig
    actual_fields = set(_field_names(TrainerConfig))
    assert "auto_lr_find" in actual_fields, (
        "TrainerConfig 应含 auto_lr_find 字段。"
        "Part 4 新增自动 LR 标定能力（Lightning LR Range Test），默认 False。"
    )

    # 验证默认值
    tc = TrainerConfig()
    assert tc.auto_lr_find is False, "auto_lr_find 默认应为 False（显式启用）"


# ============================================================
# 19. stage_probe_vram 契约（方案 B：动态显存探测）
# ============================================================
def test_stage_probe_vram_contract():
    """stage_probe_vram 应注册到 Pipeline.default()，且 vram_probe_result 写入 metadata。

    方案 B 契约：
    - Pipeline.default() 的 stages 含 ("probe_vram", stage_probe_vram)，位于 build 和 train 之间
    - PipelineContext 含 vram_probe_result 字段
    - _FIELD_FILL_STAGE 含 vram_probe_result → "stage_probe_vram" 映射
    - stage_export 的 metadata.resource 含 vram_probe 键
    - _NON_SERIALIZABLE_STAGES 含 probe_vram（依赖 model/datamodule 对象引用）
    """
    from senseframe.engine.runner.pipeline import (
        Pipeline, PipelineContext, _FIELD_FILL_STAGE,
        _NON_SERIALIZABLE_STAGES, stage_probe_vram,
    )
    from dataclasses import fields as dataclass_fields

    # 1. Pipeline.default() 含 probe_vram stage，且位于 build 和 train 之间
    pipeline = Pipeline.default()
    stage_names = [n for n, _ in pipeline.stages]
    assert "probe_vram" in stage_names, (
        "Pipeline.default() 应含 probe_vram stage（方案 B 动态显存探测）"
    )
    build_idx = stage_names.index("build")
    probe_idx = stage_names.index("probe_vram")
    train_idx = stage_names.index("train")
    assert build_idx < probe_idx < train_idx, (
        f"probe_vram 应位于 build({build_idx}) 和 train({train_idx}) 之间，"
        f"实际位置: {probe_idx}"
    )

    # 2. PipelineContext 含 vram_probe_result 字段
    field_names = {f.name for f in dataclass_fields(PipelineContext)}
    assert "vram_probe_result" in field_names, (
        "PipelineContext 应含 vram_probe_result 字段（stage_probe_vram 写入）"
    )

    # 3. _FIELD_FILL_STAGE 含 vram_probe_result 映射
    assert _FIELD_FILL_STAGE.get("vram_probe_result") == "stage_probe_vram", (
        "_FIELD_FILL_STAGE 应含 vram_probe_result → stage_probe_vram 映射"
    )

    # 4. _NON_SERIALIZABLE_STAGES 含 probe_vram
    assert "probe_vram" in _NON_SERIALIZABLE_STAGES, (
        "_NON_SERIALIZABLE_STAGES 应含 probe_vram（依赖 model/datamodule 对象引用，"
        "resume 时必须重跑）"
    )

    # 5. stage_probe_vram 的 stage_io 声明
    spec = stage_probe_vram._stage_spec
    assert "vram_probe_result" in [f.name for f in spec.writes], (
        "stage_probe_vram 的 writes 应含 vram_probe_result"
    )
    # reads 应含探测必需的输入字段
    required_reads = {"model", "datamodule", "resolved", "report"}
    actual_reads = {f.name for f in spec.reads}
    missing_reads = required_reads - actual_reads
    assert not missing_reads, (
        f"stage_probe_vram 的 reads 缺失: {missing_reads}"
    )

    # 6. stage_export 的 metadata.resource 含 vram_probe 键
    pipeline_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "stages" / "export.py"
    source = pipeline_path.read_text(encoding="utf-8")
    export_start = source.find("def stage_export")
    export_section = source[export_start:]
    if '"vram_probe": ctx.vram_probe_result' not in export_section and \
       "'vram_probe': ctx.vram_probe_result" not in export_section:
        pytest.fail(
            "stage_export 的 metadata.resource 应含 vram_probe 键，"
            "值来自 ctx.vram_probe_result（stage_probe_vram 写入）。"
        )


# ============================================================
# 20. P3-8：SKILL.md API 签名一致性
# ============================================================
def test_skill_md_api_signatures():
    """P3-8：校验 SKILL.md 中 API 示例的参数与实际签名一致。

    旧问题：SKILL.md 中 save_skill 示例使用了不存在的 version 参数，
    误导用户调用。CI 应阻断此类文档-实现漂移。

    覆盖范围：
    - save_skill：实际签名 (name, code, description, tags, source_path)
    - load_skill：实际签名 (name, version)
    - search_skills：实际签名 (query, top_k)
    """
    import inspect
    from senseframe.skills import save_skill, load_skill, search_skills

    skill_md_path = PROJECT_ROOT / "SKILL.md"
    content = _read_doc(skill_md_path)

    # 收集三个函数的合法参数集
    valid_params = {
        "save_skill": set(inspect.signature(save_skill).parameters.keys()),
        "load_skill": set(inspect.signature(load_skill).parameters.keys()),
        "search_skills": set(inspect.signature(search_skills).parameters.keys()),
    }

    violations = []

    # 校验 func(name=value, ...) 形式的调用
    # 匹配模式：捕获函数名和参数列表
    call_pattern = re.compile(
        r'\b(save_skill|load_skill|search_skills)\s*\(\s*([^)]+?)\s*\)',
        re.MULTILINE | re.DOTALL,
    )
    for match in call_pattern.finditer(content):
        func_name = match.group(1)
        args_str = match.group(2)
        # 仅提取 name=value 形式的参数名（位置参数不校验）
        param_names = re.findall(r'(\w+)\s*=', args_str)
        for param in param_names:
            if param not in valid_params[func_name]:
                violations.append(
                    (func_name, param, sorted(valid_params[func_name]),
                     match.start())
                )

    # 校验表格中的签名声明：`func(param1, param2, ...)`
    sig_pattern = re.compile(
        r'`(save_skill|load_skill|search_skills)\(([^)]*)\)`'
    )
    for match in sig_pattern.finditer(content):
        func_name = match.group(1)
        params_str = match.group(2)
        # 提取参数名（去掉 =default 和类型注解）
        declared_params = []
        for token in params_str.split(','):
            token = token.strip()
            if not token:
                continue
            # 取 = 前的部分（去掉默认值）
            name = token.split('=')[0].strip()
            # 去掉类型注解（如 name: str）
            name = name.split(':')[0].strip()
            if name:
                declared_params.append(name)
        for param in declared_params:
            if param not in valid_params[func_name]:
                violations.append(
                    (func_name, param, sorted(valid_params[func_name]),
                     match.start())
                )

    if violations:
        msg_lines = []
        for func_name, param, valid, pos in violations:
            # 找到所在行号
            line_no = content[:pos].count('\n') + 1
            msg_lines.append(
                f"  SKILL.md:{line_no} {func_name} 使用了不存在的参数 '{param}'。"
                f"合法参数: {valid}"
            )
        pytest.fail(
            "SKILL.md 中 API 示例与实际签名不一致（P3-8 契约）：\n"
            + "\n".join(msg_lines)
        )


# ============================================================
# 21. P5 P3-12：TrainerConfig 默认值与 config_schema.md 一致性
# ============================================================
def test_trainer_config_defaults_match_doc():
    """config_schema.md 中 TrainerConfig 表格的默认值应与代码 dataclass 默认值一致。

    P5 P3-12：防止文档默认值与代码漂移。覆盖字段名完整性和标量默认值一致性。
    """
    from senseframe.engine.config import TrainerConfig

    tc = TrainerConfig()
    code_fields = {name: getattr(tc, name) for name in _field_names(TrainerConfig)}

    doc_path = PROJECT_ROOT / "reference" / "config_schema.md"
    content = _read_doc(doc_path)

    # 定位 TrainerConfig 表格（## TrainerConfig 到下一个 ## 之间）
    m = re.search(r"## TrainerConfig.*?\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if not m:
        pytest.fail("config_schema.md 未找到 TrainerConfig 章节")
    table = m.group(1)

    # 解析表格行：| `field` | type | 必填 | default | desc |
    doc_defaults = {}
    for line in table.splitlines():
        m = re.match(r"\|\s*`(\w+)`\s*\|[^|]+\|[^|]+\|\s*([^|]+?)\s*\|", line)
        if m:
            field_name = m.group(1)
            default_str = m.group(2).strip()
            doc_defaults[field_name] = default_str

    # 1. 字段完整性：代码字段应在 doc 中
    missing_in_doc = set(code_fields.keys()) - set(doc_defaults.keys())
    if missing_in_doc:
        pytest.fail(
            f"config_schema.md TrainerConfig 表格缺失字段: {sorted(missing_in_doc)}。"
            f"代码 TrainerConfig 共 {len(code_fields)} 个字段，doc 仅列出 {len(doc_defaults)} 个。"
        )

    # 2. 默认值一致性（对标量字段做比对）
    def _format(v):
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        return str(v)

    def _parse_float(s):
        """从文档字符串中提取数值，支持 1e-4 / 0.0001 / `1e-4` 等形式。"""
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None

    mismatches = []
    for name, code_val in code_fields.items():
        doc_val = doc_defaults.get(name, "")
        if isinstance(code_val, (list, tuple)):
            continue  # 容器类型跳过
        # 数值字段：浮点归一化比较（1e-4 == 0.0001）
        if isinstance(code_val, (int, float)) and not isinstance(code_val, bool):
            doc_float = _parse_float(doc_val)
            if doc_float is not None and abs(doc_float - float(code_val)) < 1e-9:
                continue
            mismatches.append((name, _format(code_val), doc_val))
            continue
        expected = _format(code_val)
        if expected not in doc_val and f"`{expected}`" not in doc_val:
            mismatches.append((name, expected, doc_val))

    if mismatches:
        pytest.fail(
            "config_schema.md TrainerConfig 默认值与代码不一致：\n"
            + "\n".join(f"  {n}: 代码={exp}, doc={act}" for n, exp, act in mismatches)
        )


# ============================================================
# 22. 文档与代码一致性自动校验（设计文档 0.7.2 节）
# ============================================================
def _extract_cli_commands_via_ast() -> list:
    """从 senseframe/cli.py 的 cmd_map 字典 AST 解析提取顶层 CLI 子命令列表。

    使用 AST 而非反射，避免调用 main() 触发 argparse。
    cmd_map 是 main() 函数内的 dict literal，键为字符串（命令名），
    值为 _cmd_xxx 函数引用。通过 ast.Assign + targets[0].id == "cmd_map" 精确定位。
    """
    import ast
    cli_path = PROJECT_ROOT / "senseframe" / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "cmd_map"
                and isinstance(node.value, ast.Dict)):
            return sorted([
                k.value for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ])
    return []


def _extract_cli_table_from_doc(content: str) -> set:
    r"""从 markdown 表格中提取 CLI 命令名（反引号包裹的 `cmd`）。

    匹配表格行 `| `cmd` | ... |` 格式。
    仅扫描 "## CLI" 章节，避免误匹配其他表格（如 PipelineContext 字段表）。
    """
    # 定位 "## CLI" 章节（直到下一个 ## 标题或文件末尾）
    section_match = re.search(r'^##\s+CLI\s*$', content, re.MULTILINE)
    if not section_match:
        return set()
    section_start = section_match.end()
    # 找到下一个 ## 标题
    next_section = re.search(r'^##\s+', content[section_start:], re.MULTILINE)
    section_end = (section_start + next_section.start()) if next_section else len(content)
    section = content[section_start:section_end]

    commands = set()
    for line in section.splitlines():
        m = re.match(r"\|\s*`([a-z][a-z0-9-]*)`\s*\|", line)
        if m:
            commands.add(m.group(1))
    return commands


class TestDocCodeSync:
    """文档与代码一致性自动校验（MCP 自省协议契约保障）。

    设计文档 0.7.2 节要求：扩展 test_doc_contract.py 自动校验 stage 数/CLI 数/路由级别。
    长期机制：MCP senseframe://introspect Resource 暴露的 schema 与文档同源，
    文档漂移即 MCP 契约漂移，CI 阻断。
    """

    def test_stage_count_matches_docs(self):
        """Pipeline.default() 的 stage 数与所有文档声称一致。"""
        from senseframe.engine.runner.pipeline import Pipeline
        actual_stages = [name for name, _ in Pipeline.default().stages]
        actual_count = len(actual_stages)

        # 1. SKILL.md "# N stages: a → b → c" 校验（数量 + 列表）
        skill_md = (PROJECT_ROOT / "SKILL.md").read_text(encoding="utf-8")
        m = re.search(r'#\s*(\d+)\s*stages?:\s*([^\n]+)', skill_md)
        assert m, "SKILL.md 应包含 '# N stages: ...' 格式的 stage 声明"
        doc_count = int(m.group(1))
        doc_list_str = m.group(2).strip()
        doc_stages = [s.strip() for s in re.split(r'→|->', doc_list_str)]
        assert doc_count == actual_count, (
            f"SKILL.md 声称 {doc_count} 个 stage，实际 {actual_count} 个。"
            f"实际列表：{actual_stages}"
        )
        assert doc_stages == actual_stages, (
            f"SKILL.md stage 列表 {doc_stages} 与实际 {actual_stages} 不一致"
        )

        # 2. README.md "N 个 Stage" 校验
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        m = re.search(r'(\d+)\s*个\s*Stage', readme)
        assert m, "README.md 应包含 'N 个 Stage' 声明"
        assert int(m.group(1)) == actual_count, (
            f"README.md 声称 {m.group(1)} 个 Stage，实际 {actual_count} 个"
        )

        # 3. runtime.py docstring "默认 pipeline（N 个 stage）" 校验
        runtime_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "runtime.py"
        runtime_src = runtime_path.read_text(encoding="utf-8")
        m = re.search(r'默认 pipeline（(\d+)\s*个\s*stage）', runtime_src)
        assert m, "runtime.py Pipeline.default() docstring 应包含 '默认 pipeline（N 个 stage）'"
        assert int(m.group(1)) == actual_count, (
            f"runtime.py docstring 声称 {m.group(1)} 个 stage，实际 {actual_count} 个"
        )

        # 4. pipeline/__init__.py docstring "N 个顶层 stage" 校验
        init_path = PROJECT_ROOT / "senseframe" / "engine" / "runner" / "pipeline" / "__init__.py"
        init_src = init_path.read_text(encoding="utf-8")
        m = re.search(r'(\d+)\s*个\s*顶层\s*stage', init_src)
        assert m, "pipeline/__init__.py docstring 应包含 'N 个顶层 stage'"
        assert int(m.group(1)) == actual_count, (
            f"pipeline/__init__.py docstring 声称 {m.group(1)} 个顶层 stage，实际 {actual_count} 个"
        )

    def test_cli_count_matches_docs(self):
        """CLI 子命令数与 cli.py 实际注册数一致。"""
        actual_cli = _extract_cli_commands_via_ast()
        actual_count = len(actual_cli)

        # 1. cli.py docstring "CLI 接口：N 个子命令" 校验
        cli_src = (PROJECT_ROOT / "senseframe" / "cli.py").read_text(encoding="utf-8")
        m = re.search(r'CLI 接口：(\d+)\s*个\s*子命令', cli_src)
        assert m, "cli.py docstring 应包含 'CLI 接口：N 个子命令'"
        assert int(m.group(1)) == actual_count, (
            f"cli.py docstring 声称 {m.group(1)} 个子命令，实际 {actual_count} 个。"
            f"实际列表：{actual_cli}"
        )

        # 2. SKILL.md CLI 表格：set equality
        skill_md = (PROJECT_ROOT / "SKILL.md").read_text(encoding="utf-8")
        skill_cli = _extract_cli_table_from_doc(skill_md)
        # 排除可能的非命令表格行（如 | Command | Purpose | 表头）
        skill_cli -= {"command"}  # 表头小写形式
        assert skill_cli == set(actual_cli), (
            f"SKILL.md CLI 表 {sorted(skill_cli)} 与实际 {sorted(actual_cli)} 不一致。"
            f"缺失：{sorted(set(actual_cli) - skill_cli)}，"
            f"多余：{sorted(skill_cli - set(actual_cli))}"
        )

        # 3. README.md CLI 表格：set equality
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        readme_cli = _extract_cli_table_from_doc(readme)
        readme_cli -= {"command"}
        assert readme_cli == set(actual_cli), (
            f"README.md CLI 表 {sorted(readme_cli)} 与实际 {sorted(actual_cli)} 不一致。"
            f"缺失：{sorted(set(actual_cli) - readme_cli)}，"
            f"多余：{sorted(readme_cli - set(actual_cli))}"
        )

    def test_route_levels_match_docs(self):
        """路由级别数与 routing.RESOURCE_ROUTES 注册数一致。"""
        from senseframe.routing import RESOURCE_ROUTES
        actual_levels = set(RESOURCE_ROUTES.keys())
        actual_count = len(actual_levels)

        # README.md "N 级路由" / "N 级路由表" 校验
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        matches = re.findall(r'(\d+)\s*级\s*路由', readme)
        assert matches, "README.md 应包含 'N 级路由' 声明"
        for m_count in matches:
            doc_count = int(m_count)
            assert doc_count == actual_count, (
                f"README.md 声称 {doc_count} 级路由，实际 {actual_count} 级。"
                f"实际级别：{sorted(actual_levels)}"
            )


