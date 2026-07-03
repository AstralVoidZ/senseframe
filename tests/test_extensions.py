"""senseframe.extensions 模块测试（load_extension）。

覆盖基础加载、技能检索复用、auto_persist 自动入库（T1）、sandbox 沙箱（T1）。

注意：auto_persist 与 search_skill 操作全局技能库，用唯一技能名（uuid）避免冲突，
测试后清理。
"""

import logging
import uuid

import pytest

from senseframe.extensions import load_extension
from senseframe.security import SecurityError
from senseframe.skills import (
    get_skill_library,
    list_skills,
    save_skill,
)


# 合法扩展：含 register_loss 调用 + 自定义符号
EXT_CODE = '''
@register_loss("test_ext_loss")
def _test_ext_loss(alpha=0.25, gamma=2.0, **kw):
    return None

MY_CONSTANT = 42

def my_helper():
    return "hello"
'''

# 含危险调用 os.remove 的扩展（仅定义函数不调用，避免真实执行）
DANGEROUS_EXT_CODE = '''
import os

def dangerous_func():
    os.remove("/nonexistent/path/for/sandbox/test")
'''

# 安全扩展
SAFE_EXT_CODE = '''
SAFE_CONSTANT = 100

def safe_helper():
    return "safe"
'''


# ============================================================
# TestLoadExtensionBasic
# ============================================================

class TestLoadExtensionBasic:
    """基础加载行为。"""

    def test_load_valid_extension(self, tmp_path):
        ext_path = tmp_path / "valid_ext.py"
        ext_path.write_text(EXT_CODE, encoding="utf-8")
        # auto_persist=False 避免污染全局技能库
        module = load_extension(str(ext_path), auto_persist=False)
        assert module is not None
        # 扩展定义的符号可访问
        assert module.MY_CONSTANT == 42
        assert module.my_helper() == "hello"

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_extension(str(tmp_path / "nonexistent.py"))

    def test_no_path_no_search_skill_raises(self):
        with pytest.raises(ValueError):
            load_extension(path=None, search_skill=None)


# ============================================================
# TestSearchSkill
# ============================================================

class TestSearchSkill:
    """技能检索复用通道（RFC-002 阶段 M）。"""

    def test_search_skill_reuse(self, tmp_path):
        skill_name = f"test_search_{uuid.uuid4().hex[:8]}"
        save_skill(
            name=skill_name,
            code=EXT_CODE,
            description="test extension for search reuse",
            tags=["test", "extension"],
        )
        try:
            module = load_extension(search_skill=skill_name)
            assert module is not None
            # 复用的代码符号可访问
            assert module.MY_CONSTANT == 42
        finally:
            get_skill_library().remove(skill_name, force=True)

    def test_search_skill_no_hit_no_path_raises(self, monkeypatch):
        # hash-based 嵌入对任意查询都可能返回非零相似度，故 mock search_skills 返回空
        # 以可靠验证"未命中且 path=None 抛 ValueError"
        import senseframe.skills as skills_mod
        monkeypatch.setattr(skills_mod, "search_skills", lambda query, top_k=5: [])
        with pytest.raises(ValueError):
            load_extension(search_skill="zzz_nonexistent_query_xyz_12345_unique")


# ============================================================
# TestAutoPersist (T1)
# ============================================================

class TestAutoPersist:
    """auto_persist 自动入库（RFC-002 阶段 T）。"""

    def test_auto_persist_true_default(self, tmp_path):
        skill_name = f"test_autopersist_{uuid.uuid4().hex[:8]}"
        ext_path = tmp_path / f"{skill_name}.py"
        ext_path.write_text(EXT_CODE, encoding="utf-8")
        try:
            # auto_persist=True 为默认
            load_extension(str(ext_path))
            assert skill_name in list_skills()
        finally:
            get_skill_library().remove(skill_name, force=True)

    def test_auto_persist_false(self, tmp_path):
        skill_name = f"test_no_persist_{uuid.uuid4().hex[:8]}"
        ext_path = tmp_path / f"{skill_name}.py"
        ext_path.write_text(EXT_CODE, encoding="utf-8")
        try:
            load_extension(str(ext_path), auto_persist=False)
            assert skill_name not in list_skills()
        finally:
            get_skill_library().remove(skill_name, force=True)

    def test_auto_persist_skill_name_var(self, tmp_path):
        file_stem = f"test_autopersist_file_{uuid.uuid4().hex[:8]}"
        custom_name = f"test_custom_skill_{uuid.uuid4().hex[:8]}"
        code = f'__skill_name__ = "{custom_name}"\n' + EXT_CODE
        ext_path = tmp_path / f"{file_stem}.py"
        ext_path.write_text(code, encoding="utf-8")
        try:
            load_extension(str(ext_path))
            assert custom_name in list_skills()
            assert file_stem not in list_skills()
        finally:
            get_skill_library().remove(custom_name, force=True)
            get_skill_library().remove(file_stem, force=True)


# ============================================================
# TestSandbox (T1)
# ============================================================

class TestSandbox:
    """sandbox 沙箱最小化（RFC-002 阶段 T）。"""

    def test_sandbox_off_loads_normally(self, tmp_path):
        ext_path = tmp_path / "dangerous_off.py"
        ext_path.write_text(DANGEROUS_EXT_CODE, encoding="utf-8")
        # off 模式不扫描，正常加载
        module = load_extension(
            str(ext_path), sandbox="off", auto_persist=False
        )
        assert module is not None

    def test_sandbox_soft_loads_dangerous_with_warning(self, tmp_path):
        # senseframe logger propagate=False，caplog 捕获不到，直接挂 handler
        security_logger = logging.getLogger("senseframe.security")
        records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _CaptureHandler(level=logging.WARNING)
        security_logger.addHandler(handler)
        try:
            ext_path = tmp_path / "dangerous_soft.py"
            ext_path.write_text(DANGEROUS_EXT_CODE, encoding="utf-8")
            module = load_extension(
                str(ext_path), sandbox="soft", auto_persist=False
            )
        finally:
            security_logger.removeHandler(handler)
        # 正常加载不抛异常
        assert module is not None
        # 记录 warning（静态扫描发现 os.remove）
        security_warnings = [
            r for r in records if "os.remove" in r.getMessage()
        ]
        assert len(security_warnings) > 0

    def test_sandbox_strict_raises_security_error(self, tmp_path):
        ext_path = tmp_path / "dangerous_strict.py"
        ext_path.write_text(DANGEROUS_EXT_CODE, encoding="utf-8")
        with pytest.raises(SecurityError):
            load_extension(
                str(ext_path), sandbox="strict", auto_persist=False
            )

    def test_sandbox_soft_safe_no_warning(self, tmp_path):
        # senseframe logger propagate=False，caplog 捕获不到，直接挂 handler
        security_logger = logging.getLogger("senseframe.security")
        records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _CaptureHandler(level=logging.WARNING)
        security_logger.addHandler(handler)
        try:
            ext_path = tmp_path / "safe_soft.py"
            ext_path.write_text(SAFE_EXT_CODE, encoding="utf-8")
            module = load_extension(
                str(ext_path), sandbox="soft", auto_persist=False
            )
        finally:
            security_logger.removeHandler(handler)
        assert module is not None
        # 无危险调用 warning
        security_warnings = [
            r for r in records
            if "Dangerous call" in r.getMessage()
            or "blocked by sandbox" in r.getMessage()
        ]
        assert len(security_warnings) == 0
