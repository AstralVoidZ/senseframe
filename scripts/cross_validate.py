"""
Phase 3: 交叉验证。

用新 run_experiment 跑 T1-T12 等价配置，对比 baseline_report.json。

用法：
    python scripts/cross_validate.py

输出：
    tests/new_report.json      — 新实现的结果
    tests/diff_report.json     — 对比差异报告
"""

import json
import sys
import traceback
from pathlib import Path

# bootstrap：senseframe 可导入前的必要本地推导
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from senseframe.common.paths import PROJECT_ROOT, resolve_data_root  # noqa: E402
from senseframe.engine.config import ExperimentConfig  # noqa: E402
from senseframe.engine.runner import run_experiment  # noqa: E402


# data_root 从 SENSEFRAME_DATA_ROOT env 读（框架不猜测路径，未设置则 raise）
DATA_ROOT = resolve_data_root()
OUTPUT_BASE = "results_new"
BASELINE_PATH = PROJECT_ROOT / "tests" / "baseline_report.json"
NEW_REPORT_PATH = PROJECT_ROOT / "tests" / "new_report.json"
DIFF_REPORT_PATH = PROJECT_ROOT / "tests" / "diff_report.json"


def _make_config(test_id, model_id, dataset, **overrides):
    """构建 ExperimentConfig，与基准配置等价。"""
    scene = {
        "name": "wifi_csi",
        "dataset": dataset,
        "model_id": model_id,
        "learning_mode": overrides.get("learning_mode", "supervised"),
        "data_root": str(DATA_ROOT),
    }
    # 将额外参数放入 params
    params = {}
    # 默认 metrics 与基准一致
    if "metrics" not in overrides:
        params["metrics"] = ["accuracy", "macro_f1", "micro_f1"]
    else:
        params["metrics"] = overrides.pop("metrics")
    if "self_supervised_epochs" in overrides:
        params["self_supervised_epochs"] = overrides.pop("self_supervised_epochs")
    if "gpu" in overrides:
        params["gpu"] = overrides.pop("gpu")
    if "resume" in overrides:
        params["resume"] = overrides.pop("resume")
    if params:
        scene["params"] = params

    config_dict = {
        "scene": scene,
        "input_features": [{"name": "csi", "type": "csi", "shape": [1]}],
        "output_features": [{"name": "y", "type": "category", "num_classes": 2}],
        "trainer": {
            "epochs": overrides.get("epochs", 50),
            "learning_rate": overrides.get("learning_rate", 1e-3),
            "batch_size": overrides.get("batch_size", 64),
            "optimizer": overrides.get("optimizer", "adam"),
            "seed": overrides.get("seed", 42),
            "deterministic": overrides.get("deterministic", False),
            "max_time": overrides.get("max_time", "00:00:30:00"),
        },
        "output_dir": overrides.get("output_dir", f"{OUTPUT_BASE}/{test_id}"),
    }

    if "early_stopping" in overrides:
        config_dict["trainer"]["early_stopping"] = overrides["early_stopping"]

    return ExperimentConfig.from_dict(config_dict)


def _record_output(test_id, config, out):
    """记录单个测试的输出。"""
    record = {
        "test_id": test_id,
        "status": out.status,
    }

    if out.status == "success":
        record["final_eval"] = out.final_eval
        record["training"] = {
            "epochs_trained": out.training["epochs_trained"],
            "early_stopped": out.training["early_stopped"],
            "best_val_loss": out.training.get("best_val_loss"),
        }
        record["output_dir"] = out.output_dir

        out_dir = Path(out.output_dir)
        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            record["metadata"] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

        record["files"] = sorted(
            str(p.relative_to(out_dir))
            for p in out_dir.rglob("*")
            if p.is_file()
        )

        log_path = out_dir / "training_log.jsonl"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            record["training_log_lines"] = len(lines)
    else:
        record["error"] = out.error

    return record


def _compare(baseline, new):
    """对比基准和新实现的结果。"""
    diff = {
        "test_id": baseline["test_id"],
        "baseline_status": baseline["status"],
        "new_status": new["status"],
        "status_match": baseline["status"] == new["status"],
    }

    if baseline["status"] == "success" and new["status"] == "success":
        # 对比 final_eval
        b_eval = baseline.get("final_eval", {})
        n_eval = new.get("final_eval", {})
        eval_diff = {}
        for k in set(list(b_eval.keys()) + list(n_eval.keys())):
            b_v = b_eval.get(k)
            n_v = n_eval.get(k)
            if b_v is not None and n_v is not None:
                eval_diff[k] = {
                    "baseline": b_v,
                    "new": n_v,
                    "delta": round(n_v - b_v, 6),
                    "match": abs(n_v - b_v) < 1e-6,
                }
            else:
                eval_diff[k] = {"baseline": b_v, "new": n_v, "match": False}
        diff["final_eval_diff"] = eval_diff
        diff["final_eval_match"] = all(
            v.get("match", False) for v in eval_diff.values()
        )

        # 对比 training
        b_tr = baseline.get("training", {})
        n_tr = new.get("training", {})
        diff["epochs_match"] = b_tr.get("epochs_trained") == n_tr.get("epochs_trained")
        diff["early_stopped_match"] = b_tr.get("early_stopped") == n_tr.get("early_stopped")

        # 对比文件结构
        b_files = set(baseline.get("files", []))
        n_files = set(new.get("files", []))
        diff["files_match"] = b_files == n_files
        if not diff["files_match"]:
            diff["files_only_in_baseline"] = sorted(b_files - n_files)
            diff["files_only_in_new"] = sorted(n_files - b_files)

        # 对比 training_log 行数
        diff["log_lines_match"] = (
            baseline.get("training_log_lines") == new.get("training_log_lines")
        )

    return diff


def main():
    """跑 T1-T12 等价配置，对比基准。"""
    # T1-T12 配置（与 generate_baseline.py 等价）
    tests = [
        ("T1", _make_config("T1", "MLP", "UT_HAR_data", epochs=3)),
        ("T2", _make_config("T2", "LeNet", "UT_HAR_data", epochs=3)),
        ("T3", _make_config("T3", "GRU", "UT_HAR_data", epochs=2)),
        ("T4", _make_config("T4", "MLP", "NTU-Fi_HAR", epochs=2)),
        ("T5", _make_config("T5", "MLP", "Widar", epochs=2)),
        ("T6", _make_config("T6", "MLP", "NTU-Fi_HAR", epochs=2,
                            learning_mode="self_supervised",
                            self_supervised_epochs=2,
                            metrics=["accuracy", "macro_f1"])),
        ("T7_run1", _make_config("T7_run1", "MLP", "UT_HAR_data", epochs=2,
                                 deterministic=True)),
        ("T7_run2", _make_config("T7_run2", "MLP", "UT_HAR_data", epochs=2,
                                 deterministic=True)),
        ("T8", _make_config("T8", "MLP", "UT_HAR_data", epochs=10,
                            early_stopping=2)),
        ("T9", _make_config("T9", "MLP", "UT_HAR_data", epochs=1)),
        ("T10", _make_config("T10", "NonExistentModel", "UT_HAR_data", epochs=1)),
        ("T11", _make_config("T11", "MLP", "UT_HAR_data", epochs=2,
                            metrics=["accuracy", "macro_f1", "micro_f1",
                                     "weighted_f1", "macro_precision", "macro_recall"])),
        ("T12_phase1", _make_config("T12_phase1", "MLP", "UT_HAR_data", epochs=1)),
    ]

    # T9: 覆盖 data_root 为不存在路径
    tests[9] = ("T9", _make_config("T9", "MLP", "UT_HAR_data", epochs=1))
    tests[9][1].scene.data_root = str(DATA_ROOT / "NonExistentPath")

    report = {
        "description": "New implementation results for cross-validation",
        "generator": "scripts/cross_validate.py",
        "implementation": "run_experiment (Stage 8 decoupled)",
        "data_root": str(DATA_ROOT),
        "results": [],
    }

    print(f"=== Phase 3: 交叉验证（{len(tests)} 个测试）===")
    print(f"实现: run_experiment (Stage 8 解耦版)")
    print()

    for test_id, config in tests:
        print(f"[{test_id}] 运行中... model={config.scene.model_id}, "
              f"dataset={config.scene.dataset}, mode={config.scene.learning_mode}")

        try:
            out = run_experiment(config)
            record = _record_output(test_id, config, out)
            report["results"].append(record)

            if out.status == "success":
                print(f"  -> 成功: epochs={out.training['epochs_trained']}, "
                      f"final_eval={out.final_eval}")
            else:
                print(f"  -> 预期错误: {out.error}")
        except Exception as e:
            print(f"  -> 异常: {e}")
            traceback.print_exc()
            report["results"].append({
                "test_id": test_id,
                "status": "exception",
                "error": str(e),
            })

        print()

    # T12 Phase 2: resume 续训
    t12_p1 = next((r for r in report["results"] if r["test_id"] == "T12_phase1"), None)
    if t12_p1 and t12_p1["status"] == "success":
        print("[T12_phase2] resume 续训...")
        ckpt_dir = Path(t12_p1["output_dir"]) / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            config_t12_p2 = _make_config("T12_phase2", "MLP", "UT_HAR_data", epochs=2)
            config_t12_p2.scene.params = config_t12_p2.scene.params or {}
            config_t12_p2.scene.params["resume"] = str(ckpts[0])
            try:
                out = run_experiment(config_t12_p2)
                record = _record_output("T12_phase2", config_t12_p2, out)
                report["results"].append(record)
                if out.status == "success":
                    print(f"  -> 成功: epochs={out.training['epochs_trained']}")
                else:
                    print(f"  -> 失败: {out.error}")
            except Exception as e:
                print(f"  -> 异常: {e}")
        else:
            print("  -> 跳过: 无 checkpoint")
        print()

    # 保存新报告
    NEW_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"=== 新报告已保存: {NEW_REPORT_PATH} ===")

    # 对比基准
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        b_results = {r["test_id"]: r for r in baseline["results"]}
        n_results = {r["test_id"]: r for r in report["results"]}

        diffs = []
        for tid in sorted(set(list(b_results.keys()) + list(n_results.keys()))):
            b = b_results.get(tid)
            n = n_results.get(tid)
            if b and n:
                diffs.append(_compare(b, n))
            elif b:
                diffs.append({"test_id": tid, "baseline_status": b["status"], "new_status": "missing"})
            elif n:
                diffs.append({"test_id": tid, "baseline_status": "missing", "new_status": n["status"]})

        diff_report = {
            "description": "Cross-validation diff: baseline (run_training) vs new (run_experiment)",
            "total_tests": len(diffs),
            "summary": {
                "status_match": sum(1 for d in diffs if d.get("status_match")),
                "final_eval_match": sum(1 for d in diffs if d.get("final_eval_match")),
                "files_match": sum(1 for d in diffs if d.get("files_match")),
                "log_lines_match": sum(1 for d in diffs if d.get("log_lines_match")),
            },
            "diffs": diffs,
        }

        DIFF_REPORT_PATH.write_text(
            json.dumps(diff_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"=== 对比报告已保存: {DIFF_REPORT_PATH} ===")
        print()
        print("=== 交叉验证汇总 ===")
        s = diff_report["summary"]
        print(f"状态匹配: {s['status_match']}/{diff_report['total_tests']}")
        print(f"指标匹配: {s['final_eval_match']}/{diff_report['total_tests']}")
        print(f"文件匹配: {s['files_match']}/{diff_report['total_tests']}")
        print(f"日志行数匹配: {s['log_lines_match']}/{diff_report['total_tests']}")
    else:
        print(f"警告: 基准报告不存在 ({BASELINE_PATH})，跳过对比")


if __name__ == "__main__":
    main()
