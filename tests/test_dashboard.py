"""senseframe.observability.ExplorationDashboard 与 ExplorationTracker
action_log/feedback_trace 测试。

覆盖 RFC-002 阶段 V 新增的探索过程可视化与 feedback → recommended → adopted
追溯链路。
"""

import pytest

from senseframe.observability import ExplorationDashboard
from senseframe.exploration import ExplorationTracker


def _build_tracker_with_feedback():
    """构造带 feedback 的 tracker，用于 dashboard 测试。

    最后一次试验的 feedback 为 numerical_instability，可稳定生成 2 条
    feedback 感知推荐（稳定 loss + 梯度裁剪），便于验证 adopted/unadopted。
    """
    tracker = ExplorationTracker()
    tracker.add_trial(
        strategy={"loss": "focal", "lr": 0.001},
        result={"val_accuracy": 0.80, "val_loss": 0.55},
        feedback={"status": "overfitting", "suggestions": ["数据增强"]},
    )
    tracker.add_trial(
        strategy={"loss": "cross_entropy", "lr": 0.01},
        result={"val_accuracy": 0.85, "val_loss": 0.40},
        feedback={"status": "numerical_instability", "suggestions": ["降低 lr"]},
    )
    return tracker


# ============================================================
# TestActionLog
# ============================================================
class TestActionLog:
    """action_log：recommend_next / log_adoption / 持久化。"""

    def test_recommend_next_logs_recommended(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        assert len(recs) > 0
        # action_log 应有 status="recommended" 的记录
        recommended = [e for e in tracker.action_log if e["status"] == "recommended"]
        assert len(recommended) > 0

    def test_log_adoption_records_adopted(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        rec_id = recs[0]["recommendation_id"]
        tracker.log_adoption(rec_id, actual_strategy=recs[0]["strategy"])
        adopted = [e for e in tracker.action_log if e["status"] == "adopted"]
        assert len(adopted) == 1
        assert adopted[0]["recommendation_id"] == rec_id

    def test_save_load_action_log_roundtrip(self, tmp_path):
        tracker = _build_tracker_with_feedback()
        tracker.recommend_next()
        path = tmp_path / "action.json"
        tracker.save(path)
        loaded = ExplorationTracker.load(path)
        assert len(loaded.action_log) == len(tracker.action_log)
        # 每条记录的 status 与 recommendation_id 一致
        for a, b in zip(tracker.action_log, loaded.action_log):
            assert a["status"] == b["status"]
            assert a["recommendation_id"] == b["recommendation_id"]


# ============================================================
# TestFeedbackTrace
# ============================================================
class TestFeedbackTrace:
    """feedback_trace：feedback → recommended → adopted 追溯链路。"""

    def test_trace_non_empty_with_feedback_recommend_adopt(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        tracker.log_adoption(recs[0]["recommendation_id"], recs[0]["strategy"])
        traces = tracker.feedback_trace()
        assert len(traces) > 0

    def test_trace_fields(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        tracker.log_adoption(recs[0]["recommendation_id"], recs[0]["strategy"])
        traces = tracker.feedback_trace()
        assert len(traces) > 0
        for tr in traces:
            assert "feedback_status" in tr
            assert "recommended_strategy" in tr
            assert "adopted_strategy" in tr
            assert "adopted" in tr

    def test_unadopted_recommendation(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        # numerical_instability 生成 2 条推荐，仅采纳第一条
        assert len(recs) >= 2
        tracker.log_adoption(recs[0]["recommendation_id"], recs[0]["strategy"])
        traces = tracker.feedback_trace()
        # 至少有一条未采纳
        unadopted = [tr for tr in traces if not tr["adopted"]]
        assert len(unadopted) > 0
        assert all(tr["adopted_strategy"] is None for tr in unadopted)


# ============================================================
# TestExplorationDashboard
# ============================================================
class TestExplorationDashboard:
    """ExplorationDashboard：覆盖率/对比/追溯/渲染。"""

    def test_coverage_report(self):
        tracker = _build_tracker_with_feedback()
        tracker.recommend_next()
        dashboard = ExplorationDashboard(tracker)
        cov = dashboard.coverage_report()
        assert "total_trials" in cov
        assert "completed" in cov
        assert "feedback_distribution" in cov
        assert cov["total_trials"] == 2
        assert cov["completed"] == 2

    def test_trial_comparison(self):
        tracker = _build_tracker_with_feedback()
        dashboard = ExplorationDashboard(tracker)
        comparison = dashboard.trial_comparison()
        assert len(comparison) > 0
        for t in comparison:
            assert "trial_id" in t
            assert "val_accuracy" in t
            assert "feedback" in t

    def test_trial_comparison_with_ids(self):
        tracker = _build_tracker_with_feedback()
        dashboard = ExplorationDashboard(tracker)
        comparison = dashboard.trial_comparison(trial_ids=["trial_0000"])
        assert len(comparison) == 1
        assert comparison[0]["trial_id"] == "trial_0000"

    def test_feedback_trace_delegates(self):
        tracker = _build_tracker_with_feedback()
        recs = tracker.recommend_next()
        tracker.log_adoption(recs[0]["recommendation_id"], recs[0]["strategy"])
        dashboard = ExplorationDashboard(tracker)
        traces = dashboard.feedback_trace()
        assert traces == tracker.feedback_trace()

    def test_render_text(self):
        tracker = _build_tracker_with_feedback()
        tracker.recommend_next()
        dashboard = ExplorationDashboard(tracker)
        out = dashboard.render(format="text")
        assert isinstance(out, str)
        assert len(out) > 0
        assert "Dashboard" in out

    def test_render_markdown(self):
        tracker = _build_tracker_with_feedback()
        dashboard = ExplorationDashboard(tracker)
        out = dashboard.render(format="markdown")
        assert isinstance(out, str)
        assert len(out) > 0
        assert "|" in out

    def test_render_html(self):
        tracker = _build_tracker_with_feedback()
        dashboard = ExplorationDashboard(tracker)
        out = dashboard.render(format="html")
        assert isinstance(out, str)
        assert len(out) > 0
        assert "<table" in out or "<div" in out

    def test_render_invalid_format_raises(self):
        tracker = _build_tracker_with_feedback()
        dashboard = ExplorationDashboard(tracker)
        with pytest.raises(ValueError):
            dashboard.render(format="invalid")
