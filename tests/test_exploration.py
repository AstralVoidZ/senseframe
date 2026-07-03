"""senseframe.exploration 模块测试。

覆盖 ExplorationTracker（含 R1 feedback 参数与 feedback 感知 recommend_next）
与 SearchSpaceMap。feedback 测试是 R1 闭环的核心验证，覆盖 5 种 status。
"""

import pytest

from senseframe.exploration import ExplorationTracker, SearchSpaceMap


# ============================================================
# TestExplorationTracker
# ============================================================

class TestExplorationTracker:
    """ExplorationTracker 的试验记录、查询、统计、持久化。"""

    def test_add_trial_returns_trial_id(self):
        t = ExplorationTracker()
        tid = t.add_trial(strategy={"loss": "focal"}, result={"val_accuracy": 0.8})
        assert tid is not None
        assert isinstance(tid, str)

    def test_add_trial_with_feedback(self):
        """R1：记录带 feedback 的试验。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "focal"},
            result={"val_accuracy": 0.8},
            feedback={"status": "overfitting", "suggestions": ["augment"]},
        )
        trial = t.list_trials()[0]
        assert trial["feedback"]["status"] == "overfitting"

    def test_list_trials_all(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        t.add_trial(strategy={"a": 2}, result={"val_accuracy": 0.8})
        assert len(t.list_trials()) == 2

    def test_list_trials_filter_by_status(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        t.add_trial(strategy={"a": 2}, result=None)  # pending
        completed = t.list_trials(status="completed")
        assert len(completed) == 1
        pending = t.list_trials(status="pending")
        assert len(pending) == 1

    def test_get_trial_by_id(self):
        t = ExplorationTracker()
        tid = t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        got = t.get_trial(tid)
        assert got is not None
        assert got["trial_id"] == tid
        assert t.get_trial("nonexistent") is None

    def test_best_trial_max(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        t.add_trial(strategy={"a": 2}, result={"val_accuracy": 0.9})
        best = t.best_trial(metric="val_accuracy")
        assert best["result"]["val_accuracy"] == 0.9

    def test_best_trial_min(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_loss": 0.5})
        t.add_trial(strategy={"a": 2}, result={"val_loss": 0.3})
        best = t.best_trial(metric="val_loss", mode="min")
        assert best["result"]["val_loss"] == 0.3

    def test_best_trial_no_data(self):
        t = ExplorationTracker()
        assert t.best_trial() is None

    def test_explored_strategies_dedup(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"loss": "focal"}, result={"val_accuracy": 0.7})
        t.add_trial(strategy={"loss": "focal"}, result={"val_accuracy": 0.8})
        t.add_trial(strategy={"loss": "ce"}, result={"val_accuracy": 0.6})
        strategies = t.explored_strategies()
        assert len(strategies) == 2

    def test_last_feedback(self):
        """R1：返回最近一次试验的 feedback。"""
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        t.add_trial(
            strategy={"a": 2},
            result={"val_accuracy": 0.8},
            feedback={"status": "overfitting"},
        )
        fb = t.last_feedback()
        assert fb is not None
        assert fb["status"] == "overfitting"

    def test_last_feedback_none_when_no_feedback(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        assert t.last_feedback() is None

    def test_coverage(self):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        t.add_trial(strategy={"a": 2}, result={"val_accuracy": 0.8})
        t.add_trial(strategy={"a": 3}, result=None)  # pending
        cov = t.coverage()
        assert cov["total_trials"] == 3
        assert cov["completed"] == 2
        assert cov["pending"] == 1
        assert cov["failed"] == 0
        assert cov["unique_strategies"] == 3

    def test_save_load_roundtrip(self, tmp_path):
        t = ExplorationTracker()
        t.add_trial(strategy={"a": 1}, result={"val_accuracy": 0.7})
        path = tmp_path / "history.json"
        t.save(path)
        t2 = ExplorationTracker.load(path)
        assert len(t2.list_trials()) == 1
        assert t2.get_trial("trial_0000") is not None


# ============================================================
# TestRecommendNext (R1 feedback 感知排序核心)
# ============================================================

class TestRecommendNext:
    """recommend_next 的 feedback 感知排序（R1 闭环核心）。

    必须覆盖 5 种 feedback status：overfitting / numerical_instability /
    underfitting / converged / success。
    """

    def test_no_feedback_returns_compatibility_candidates(self):
        """无 feedback 时返回兼容性矩阵候选。"""
        t = ExplorationTracker()
        recs = t.recommend_next(task_type="classification")
        # 应包含 loss/metric 组合
        has_loss_metric = any(
            "loss" in r["strategy"] and "metric" in r["strategy"]
            for r in recs
        )
        assert has_loss_metric

    def test_overfitting_feedback_recommends_augment(self):
        """R1 闭环核心：overfitting feedback 时首条推荐应包含 augment 策略。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "cross_entropy"},
            result={"val_accuracy": 0.7},
            feedback={"status": "overfitting"},
        )
        recs = t.recommend_next(task_type="classification")
        assert len(recs) > 0
        first = recs[0]["strategy"]
        # 首条推荐应包含 augment
        assert "transform" in first
        assert "augment" in first["transform"]

    def test_numerical_instability_feedback_recommends_stable_loss_or_clip(self):
        """R1：numerical_instability feedback 时首条推荐应含稳定 loss 或 gradient_clip_val。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "focal"},
            result={"val_accuracy": 0.7},
            feedback={"status": "numerical_instability"},
        )
        recs = t.recommend_next(task_type="classification")
        assert len(recs) > 0
        first = recs[0]["strategy"]
        stable_losses = {"smooth_l1", "mae", "cross_entropy"}
        has_stable_loss = first.get("loss") in stable_losses
        has_clip = "gradient_clip_val" in first
        assert has_stable_loss or has_clip

    def test_underfitting_feedback_recommends_strong_loss_or_epochs(self):
        """R1：underfitting feedback 时首条推荐应含更强 loss 或 epochs_scale。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "cross_entropy"},
            result={"val_accuracy": 0.3},
            feedback={"status": "underfitting"},
        )
        recs = t.recommend_next(task_type="classification")
        assert len(recs) > 0
        first = recs[0]["strategy"]
        strong_losses = {"focal", "cross_entropy_weighted"}
        has_strong_loss = first.get("loss") in strong_losses
        has_epochs = "epochs_scale" in first
        assert has_strong_loss or has_epochs

    def test_converged_feedback_recommends_transform_pipeline(self):
        """R1：converged feedback 时推荐应含 transform pipeline。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "cross_entropy"},
            result={"val_accuracy": 0.95},
            feedback={"status": "converged"},
        )
        recs = t.recommend_next(task_type="classification")
        # 推荐中应含 transform pipeline
        has_pipeline = any(
            "transform" in r["strategy"]
            and "pipeline" in r["strategy"].get("transform", {})
            for r in recs
        )
        assert has_pipeline

    def test_success_feedback_recommends_lr_scale(self):
        """R1：success feedback 时推荐应含 lr_scale 微调。"""
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "cross_entropy"},
            result={"val_accuracy": 0.92},
            feedback={"status": "success"},
        )
        recs = t.recommend_next(task_type="classification")
        has_lr_scale = any("lr_scale" in r["strategy"] for r in recs)
        assert has_lr_scale

    def test_top_k(self):
        t = ExplorationTracker()
        recs = t.recommend_next(task_type="classification", top_k=1)
        assert len(recs) <= 1

    def test_explored_strategies_excluded(self):
        t = ExplorationTracker()
        t.add_trial(
            strategy={"loss": "cross_entropy", "metric": "accuracy"},
            result={"val_accuracy": 0.7},
        )
        recs = t.recommend_next(task_type="classification")
        for r in recs:
            assert r["strategy"] != {"loss": "cross_entropy", "metric": "accuracy"}


# ============================================================
# TestSearchSpaceMap
# ============================================================

class TestSearchSpaceMap:
    """SearchSpaceMap 的 overview / techniques / compatible_strategies / coverage。"""

    def test_overview(self):
        tracker = ExplorationTracker()
        m = SearchSpaceMap(tracker)
        ov = m.overview(task_type="classification")
        assert "techniques" in ov
        assert "compatible_strategies" in ov
        assert "exploration_coverage" in ov
        assert "recommendations" in ov

    def test_techniques_overview(self):
        m = SearchSpaceMap()
        to = m.techniques_overview()
        assert "available" in to
        if to["available"]:
            assert "categories" in to
            assert "total_techniques" in to

    def test_compatible_strategies(self):
        m = SearchSpaceMap()
        cs = m.compatible_strategies(task_type="classification")
        assert cs["available"] is True
        assert "losses" in cs
        assert "metrics" in cs
        assert "activations" in cs
        assert "cross_entropy" in cs["losses"]

    def test_coverage_report(self):
        tracker = ExplorationTracker()
        tracker.add_trial(
            strategy={"transform": {"pipeline": ["hampel"]}},
            result={"val_accuracy": 0.7},
        )
        m = SearchSpaceMap(tracker)
        cr = m.coverage_report()
        assert "trials" in cr
        assert "transform_techniques_total" in cr
        assert "transform_strategies_explored" in cr
        assert cr["transform_strategies_explored"] == 1
