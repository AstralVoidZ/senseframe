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
"""

import os
import sys
from pathlib import Path

import pytest

# bootstrap：senseframe 可导入前的必要本地推导
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

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


@pytest.fixture
def project_root() -> Path:
    """项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def data_root() -> Path:
    """数据集根目录（从 SENSEFRAME_DATA_ROOT env 读，未设置则 raise）。"""
    if DATA_ROOT is None:
        pytest.skip("SENSEFRAME_DATA_ROOT 未设置，跳过依赖数据的测试")
    return DATA_ROOT
