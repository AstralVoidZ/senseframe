"""senseframe.engine.runner.pipeline 的 feedback 闭环测试（R1+R2）。

仅测试 analyze_training_result 纯函数，不依赖 Lightning Trainer。
闭合探索-反馈回路：eval 结果 → 失败分类 + 改进建议。
"""

import pytest

from senseframe.engine.runner.pipeline import analyze_training_result


# ============================================================
# TestAnalyzeTrainingResult
# ============================================================

class TestAnalyzeTrainingResult:
    """analyze_training_result 失败分类与建议。"""

    def test_numerical_instability_on_nan(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": float("nan")},
            training_log=[],
            early_stopped=False,
        )
        assert result["status"] == "numerical_instability"

    def test_underfitting_low_val_acc(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": 0.3},
            training_log=[],
            early_stopped=False,
            task_type="classification",
        )
        assert result["status"] == "underfitting"

    def test_overfitting_large_gap(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": 0.80},
            training_log=[{"train_accuracy": 0.98, "val_accuracy": 0.80}],
            early_stopped=False,
            task_type="classification",
        )
        assert result["status"] == "overfitting"

    def test_converged_early_stopped(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": 0.85},
            training_log=[{"train_accuracy": 0.88, "val_accuracy": 0.85}],
            early_stopped=True,
            task_type="classification",
        )
        assert result["status"] == "converged"

    def test_success_normal_completion(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": 0.85},
            training_log=[{"train_accuracy": 0.88, "val_accuracy": 0.85}],
            early_stopped=False,
            task_type="classification",
        )
        assert result["status"] == "success"

    def test_returns_three_fields(self):
        result = analyze_training_result(
            final_eval={"val_accuracy": 0.85},
            training_log=[],
            early_stopped=False,
        )
        assert set(result.keys()) == {"status", "diagnosis", "suggestions"}


# ============================================================
# TestFeedbackStructure
# ============================================================

# 每种 status 的代表性输入（final_eval, training_log, early_stopped, task_type, expected_status）
_STATUS_CASES = [
    (
        {"val_accuracy": float("nan")},
        [],
        False,
        "classification",
        "numerical_instability",
    ),
    (
        {"val_accuracy": 0.3},
        [],
        False,
        "classification",
        "underfitting",
    ),
    (
        {"val_accuracy": 0.80},
        [{"train_accuracy": 0.98, "val_accuracy": 0.80}],
        False,
        "classification",
        "overfitting",
    ),
    (
        {"val_accuracy": 0.85},
        [{"train_accuracy": 0.88, "val_accuracy": 0.85}],
        True,
        "classification",
        "converged",
    ),
    (
        {"val_accuracy": 0.85},
        [{"train_accuracy": 0.88, "val_accuracy": 0.85}],
        False,
        "classification",
        "success",
    ),
]


class TestFeedbackStructure:
    """每种 status 的结构化反馈完整性。"""

    @pytest.mark.parametrize(
        "final_eval, training_log, early_stopped, task_type, expected_status",
        _STATUS_CASES,
    )
    def test_suggestions_non_empty_list(
        self, final_eval, training_log, early_stopped, task_type, expected_status
    ):
        result = analyze_training_result(
            final_eval, training_log, early_stopped, task_type=task_type
        )
        assert result["status"] == expected_status
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0

    @pytest.mark.parametrize(
        "final_eval, training_log, early_stopped, task_type, expected_status",
        _STATUS_CASES,
    )
    def test_diagnosis_non_empty_string(
        self, final_eval, training_log, early_stopped, task_type, expected_status
    ):
        result = analyze_training_result(
            final_eval, training_log, early_stopped, task_type=task_type
        )
        assert result["status"] == expected_status
        assert isinstance(result["diagnosis"], str)
        assert len(result["diagnosis"]) > 0
