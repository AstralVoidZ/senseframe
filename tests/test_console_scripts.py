"""console_scripts 注册测试。

验证 pyproject.toml 注册了 senseframe 命令（v2 次要差距修复）。
"""
from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from pathlib import Path


class TestConsoleScripts:
    """验证 pyproject.toml [project.scripts] 段。"""

    @property
    def _pyproject(self):
        path = Path(__file__).parent.parent / "pyproject.toml"
        with open(path, "rb") as f:
            return tomllib.load(f)

    def test_scripts_section_exists(self):
        """pyproject.toml 应含 [project.scripts] 段。"""
        project = self._pyproject["project"]
        assert "scripts" in project, "pyproject.toml 缺少 [project.scripts] 段"

    def test_senseframe_command_registered(self):
        """应注册 senseframe 命令指向 senseframe.cli:main。"""
        project = self._pyproject["project"]
        scripts = project.get("scripts", {})
        assert "senseframe" in scripts
        assert scripts["senseframe"] == "senseframe.cli:main"

    def test_cli_has_main_entry(self):
        """senseframe.cli 模块应有 main 函数。"""
        from senseframe.cli import main
        assert callable(main)
