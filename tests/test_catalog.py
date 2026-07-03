"""senseframe.scenes.wifi_csi.catalog 模块测试。

覆盖技术目录的枚举、查询、pipeline/augment 推荐与增强判定。
"""

import pytest

from senseframe.scenes.wifi_csi.catalog import (
    CATALOG,
    list_techniques,
    get_technique,
    list_by_category,
    list_categories,
    get_applicable_techniques,
    is_augment,
    suggest_pipeline,
    suggest_augment,
)


# ============================================================
# TestCatalog
# ============================================================

class TestCatalog:
    """技术目录基础查询。"""

    def test_list_techniques_count(self):
        names = list_techniques()
        assert len(names) == 13

    def test_list_techniques_matches_catalog(self):
        assert list_techniques() == [t["name"] for t in CATALOG]

    def test_get_technique_hampel_fields(self):
        t = get_technique("hampel")
        assert isinstance(t, dict)
        assert t["name"] == "hampel"
        assert t["category"] == "denoising"
        assert "description" in t
        assert "applicable" in t
        assert "params" in t
        assert t["implemented"] is True

    def test_get_technique_nonexistent_raises(self):
        with pytest.raises(KeyError):
            get_technique("不存在")

    def test_list_by_category_denoising(self):
        items = list_by_category("denoising")
        assert len(items) > 0
        for t in items:
            assert t["category"] == "denoising"

    def test_list_categories_sorted_unique(self):
        cats = list_categories()
        # 去重
        assert len(cats) == len(set(cats))
        # 排序
        assert cats == sorted(cats)
        # 五个类别
        assert cats == [
            "augmentation",
            "denoising",
            "feature_engineering",
            "phase",
            "time_frequency",
        ]

    def test_get_applicable_techniques_ntu(self):
        items = get_applicable_techniques("NTU-Fi_HAR")
        assert len(items) > 0
        for t in items:
            assert "NTU-Fi_HAR" in t["applicable"]


# ============================================================
# TestSuggestPipeline
# ============================================================

class TestSuggestPipeline:
    """pipeline 推荐（排除增强类）。"""

    def test_suggest_pipeline_non_empty_no_augment(self):
        names = suggest_pipeline("NTU-Fi_HAR")
        assert len(names) > 0
        for n in names:
            assert is_augment(n) is False

    def test_suggest_pipeline_filter_category(self):
        names = suggest_pipeline("NTU-Fi_HAR", categories=["denoising"])
        assert len(names) > 0
        for n in names:
            t = get_technique(n)
            assert t["category"] == "denoising"

    def test_suggest_pipeline_all_implemented(self):
        names = suggest_pipeline("NTU-Fi_HAR")
        assert len(names) > 0
        for n in names:
            assert get_technique(n)["implemented"] is True


# ============================================================
# TestSuggestAugment
# ============================================================

class TestSuggestAugment:
    """增强推荐。"""

    def test_suggest_augment_non_empty(self):
        names = suggest_augment("NTU-Fi_HAR")
        assert len(names) > 0

    def test_suggest_augment_all_augmentation(self):
        names = suggest_augment("NTU-Fi_HAR")
        assert len(names) > 0
        for n in names:
            t = get_technique(n)
            assert t["category"] == "augmentation"


# ============================================================
# TestIsAugment
# ============================================================

class TestIsAugment:
    """增强判定。"""

    def test_is_augment_true(self):
        assert is_augment("time_jitter") is True

    def test_is_augment_false(self):
        assert is_augment("hampel") is False

    def test_is_augment_nonexistent_false(self):
        assert is_augment("不存在") is False
