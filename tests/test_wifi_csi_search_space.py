"""wifi_csi 场景搜索空间扩展测试。

验证 get_search_space() 暴露 optimizer/scheduler/epochs/gradient_clip_val/early_stopping，
让 HPO 可搜索这些 ML 关键参数（v2 差距 1 修复）。
"""
from __future__ import annotations

import pytest

from senseframe.scenes.wifi_csi.container import WiFiCSIContainer


class TestWifiCsiSearchSpaceExtension:
    """验证搜索空间包含 ML 关键参数。"""

    @pytest.fixture
    def container(self):
        return WiFiCSIContainer()

    @pytest.fixture
    def search_space(self, container):
        return container.get_search_space(model_id="ResNet18", dataset_name="UT_HAR_data")

    def test_optimizer_exposed(self, search_space):
        """optimizer 应在搜索空间中，值为 categorical。"""
        assert "optimizer" in search_space.params
        spec = search_space.params["optimizer"]
        assert spec["type"] == "categorical"
        assert set(spec["values"]) >= {"adam", "adamw", "sgd"}

    def test_scheduler_exposed(self, search_space):
        """scheduler 应在搜索空间中，值为 categorical（含 None）。"""
        assert "scheduler" in search_space.params
        spec = search_space.params["scheduler"]
        assert spec["type"] == "categorical"
        assert None in spec["values"] or "none" in spec["values"]

    def test_epochs_exposed(self, search_space):
        """epochs 应在搜索空间中，值为 int 范围。"""
        assert "epochs" in search_space.params
        spec = search_space.params["epochs"]
        assert spec["type"] == "int"
        assert spec["low"] > 0
        assert spec["high"] > spec["low"]

    def test_gradient_clip_val_exposed(self, search_space):
        """gradient_clip_val 应在搜索空间中，值为 float 范围（含 None 表示不裁剪）。"""
        assert "gradient_clip_val" in search_space.params
        spec = search_space.params["gradient_clip_val"]
        # 允许 categorical（含 None）或 float 范围
        assert spec["type"] in ("categorical", "float")

    def test_early_stopping_exposed(self, search_space):
        """early_stopping（patience）应在搜索空间中。"""
        assert "early_stopping" in search_space.params
        spec = search_space.params["early_stopping"]
        # 允许 categorical（含 None 表示不启用）或 int 范围
        assert spec["type"] in ("categorical", "int")

    def test_existing_params_preserved(self, search_space):
        """原有参数（learning_rate/batch_size/weight_decay）不应丢失。"""
        assert "learning_rate" in search_space.params
        assert "batch_size" in search_space.params
        assert "weight_decay" in search_space.params
