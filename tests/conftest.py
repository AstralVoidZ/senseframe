"""
pytest 配置与 fixtures（R-fix 重构后骨架）。

旧测试套件已删除（数据依赖、hang、与重构脱节）。
迭代完成后重新设计测试系统时，在此骨架上扩展。

测试分层策略（待实现）：
- unit: 纯逻辑单元测试，无外部依赖，秒级
- integration: 模块间集成，可用 tmp_path 造数据
- e2e: 端到端，需真实数据集，标记 @pytest.mark.e2e

标记注册（供未来使用）：
- e2e: 端到端测试（需数据集）
- slow: 耗时测试（可能 hang）
- requires_data: 依赖真实数据目录

路径安全规范（P5 P2-14）：
- 禁止硬编码 /tmp/... 或 /home/user/... 等 Unix 路径（跨平台失败）
- 需要临时目录时使用 pytest 内置 tmp_path fixture
- debug_*.py 是开发期调试脚本（不进入 pytest 收集），不受此规范约束
"""

import os
import sys
from pathlib import Path

import pytest

# bootstrap：senseframe 可导入前的必要本地推导
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

# 测试分层重构（plan_test_refactor_v1.md）：
# 让 tests/ 自身可作为包导入，使 tests/fakes/ 与 tests/unit/l*/conftest.py
# 中的 `from tests.fakes.xxx import` 生效。不创建 tests/__init__.py 以避免
# 影响 pytest rootdir 收集语义；改为在 sys.path 中加入 tests/ 父目录并确保
# tests 包可被 import。
_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT.parent))

# 单一数据源：bootstrap 后从 senseframe.common.paths 导入 PROJECT_ROOT
from senseframe.common.paths import PROJECT_ROOT  # noqa: E402

# data_root 从 SENSEFRAME_DATA_ROOT env 读（框架不猜测路径）。
# 未设置 env 时 DATA_ROOT=None，requires_data 测试自动跳过。
_env_data_root = os.environ.get("SENSEFRAME_DATA_ROOT")
DATA_ROOT = Path(_env_data_root) if _env_data_root else None


def pytest_configure(config):
    """注册自定义标记。"""
    config.addinivalue_line("markers", "e2e: 端到端测试（需数据集，默认跳过）")
    config.addinivalue_line("markers", "slow: 耗时测试（默认跳过，-m slow 启用）")
    config.addinivalue_line("markers", "requires_data: 依赖真实数据目录")
    # 测试分层重构（plan_test_refactor_v1.md）：4 层 marker
    config.addinivalue_line("markers", "l1_contract: L1 外部协议契约测试（锚点：外部协议/库 API）")
    config.addinivalue_line("markers", "l2_orchestration: L2 编排 spec 契约测试（锚点：项目内 RFC/设计文档）")
    config.addinivalue_line("markers", "l3_algorithm: L3 算法行为测试（锚点：论文/官方实现）")
    config.addinivalue_line("markers", "l4_regression: L4 回归测试（锚点：bug 编号 + 修复 commit）")


def pytest_collection_modifyitems(config, items):
    """默认跳过 e2e / slow / requires_data，使用 -m 显式启用。"""
    selected = config.getoption("-m") or ""
    has_data = DATA_ROOT is not None and DATA_ROOT.exists()

    skip_e2e = pytest.mark.skip(reason="e2e 测试，使用 -m e2e 显式启用")
    skip_slow = pytest.mark.skip(reason="slow 测试，使用 -m slow 显式启用")
    skip_data = pytest.mark.skip(reason="无数据目录，使用 -m requires_data 显式启用")

    for item in items:
        kw = item.keywords
        if "e2e" in kw and "e2e" not in selected:
            item.add_marker(skip_e2e)
        if "slow" in kw and "slow" not in selected:
            item.add_marker(skip_slow)
        if "requires_data" in kw and not has_data:
            item.add_marker(skip_data)
