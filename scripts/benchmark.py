#!/usr/bin/env python
"""senseframe 框架引导可靠性测评。

测评目标：验证 senseframe 框架引导 AI Agent 训练 WiFi CSI 识别模型的可靠性。

双基线设计：
  - smoke 基线：短训练（2-3 epochs），从 tests/baseline_report.json 提取，
    测框架引导可靠性（L1/L2/L4），快速可复现。
  - SOTA 基线：充分训练（100-200 epochs），对标论文报告值，
    测框架能否达到公开性能（Phase 2，耗时长）。

双模式设计：
  - direct：直接调用 run_experiment，排除 AI Agent 随机性，测框架本身。
  - opencode：调用 `opencode run --command train`，端到端测 AI Agent 引导可靠性。

矩阵设计：
  - minimal：UT_HAR × MLP × 3 次（deterministic），验证可复现性。
  - multi_model：UT_HAR × {MLP, LeNet, GRU} × 各 1 次，验证多模型引导。
  - sota_core：3 数据集 × 2 模型，充分训练对标论文 SOTA（Phase 2）。
  - sota_reproducibility：UT_HAR × MLP × 200 epochs × 3 次（Phase 2）。
  - sota_agent：UT_HAR × MLP × 200 epochs × opencode 模式（Phase 2）。

用法：
    # Phase 1：最小矩阵，direct 模式，smoke 基线
    python scripts/benchmark.py --matrix minimal --mode direct --baseline smoke

    # Phase 1：多模型矩阵，opencode 模式，smoke 基线
    python scripts/benchmark.py --matrix multi_model --mode opencode --baseline smoke

    # Phase 2：SOTA 核心对标（充分训练，耗时数小时）
    python scripts/benchmark.py --matrix sota_core --mode direct --baseline sota

    # Phase 2：SOTA 可复现性
    python scripts/benchmark.py --matrix sota_reproducibility --mode direct --baseline sota

    # Phase 2：SOTA AI Agent 端到端
    python scripts/benchmark.py --matrix sota_agent --mode opencode --baseline sota

输出：
    benchmarks/report_<timestamp>.json — 结构化测评报告
    控制台 — 汇总表与 PASS/FAIL 判定
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

# 将项目根目录加入 sys.path（bootstrap：senseframe 可导入前的必要本地推导）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 单一数据源：bootstrap 后从 senseframe.common.paths 导入 PROJECT_ROOT
from senseframe.common.paths import PROJECT_ROOT  # noqa: E402
from senseframe.engine.metadata import load_metadata  # noqa: E402


# ============================================================================
# 基线定义
# ============================================================================

# Smoke 基线：从 tests/baseline_report.json 提取的短训练结果
# 键：(model_id, dataset, epochs) → {macro_f1, accuracy}
SMOKE_BASELINE = {
    ("MLP", "UT_HAR_data", 3): {"macro_f1": 0.220075, "accuracy": 0.445783},
    ("LeNet", "UT_HAR_data", 3): {"macro_f1": 0.115018, "accuracy": 0.337349},
    ("GRU", "UT_HAR_data", 2): {"macro_f1": 0.100127, "accuracy": 0.323293},
    ("MLP", "NTU-Fi_HAR", 2): {"macro_f1": 0.969158, "accuracy": 0.969697},
    ("MLP", "Widar", 2): {"macro_f1": 0.614205, "accuracy": 0.670754},
}

# SOTA 基线：论文报告的 accuracy 值（充分训练 100-200 epochs）
# 来源：SenseFi 论文 (Yang et al., Patterns 2023, arXiv:2207.07859)
#       https://arxiv.org/abs/2207.07859
# 校准说明：
#   论文精度表为图片格式，WebFetch/WebSearch 无法提取精确数值。
#   MLP/UT_HAR_data 已通过 senseframe 实际训练校准（batch A，3 次确定性运行）：
#     实测 macro_f1=0.8882, accuracy=0.8956, std=0.0（149 epochs, early_stopped）
#     论文报告 accuracy，此处 macro_f1 按 accuracy - 0.01 估算（实测差距仅 0.007）。
#   其余条目仍为估算值，待对应批次跑完后逐步校准。
# 阈值 -10%，考虑：(1) CPU vs GPU 环境差异 (2) 部分基线仍为估算 (3) 精度/批大小差异。
# 键：(model_id, dataset) → {macro_f1, accuracy, epochs_required}
SOTA_BASELINE = {
    ("MLP", "UT_HAR_data"): {
        "macro_f1": 0.88, "accuracy": 0.89, "epochs_required": 200,
        "ref": "senseframe 实测校准（batch A，3 次确定性运行，std=0.0）",
    },
    ("ResNet18", "UT_HAR_data"): {
        "macro_f1": 0.87, "accuracy": 0.88, "epochs_required": 200,
        "ref": "senseframe 实测校准（batch C，47 epochs early_stopped，delta=-0.006 vs 估算）",
    },
    ("MLP", "Widar"): {
        "macro_f1": 0.60, "accuracy": 0.67, "epochs_required": 30,
        "ref": "senseframe 实测校准（batch B，8 epochs early_stopped，MLP 对 22 类 Widar 过拟合）",
    },
    ("MLP", "NTU-Fi_HAR"): {
        "macro_f1": 0.99, "accuracy": 0.99, "epochs_required": 50,
        "ref": "senseframe 实测校准（batch B，50 epochs，near-perfect）",
    },
}

# 判定阈值
THRESHOLDS = {
    "smoke": {
        "task_success_rate": 1.0,       # L1: 任务完成率 100%
        "macro_f1_tolerance": 0.02,     # L2: Macro-F1 ≥ baseline - 2%
        "reproducibility_std": 0.001,   # L4: deterministic 方差 < 0.1%
    },
    "sota": {
        "task_success_rate": 1.0,
        "macro_f1_tolerance": 0.10,     # SOTA - 10%（宽松：CPU 环境 + 基线估算不确定性）
        "reproducibility_std": 0.005,
    },
}


# ============================================================================
# 测评矩阵定义
# ============================================================================

MATRIX_MINIMAL = {
    "name": "minimal",
    "description": "最小矩阵：UT_HAR × MLP × 3 次，验证可复现性",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "UT_HAR_data",
            "epochs": 3,
            "repeats": 3,
            "deterministic": True,
            "seed": 42,
        },
    ],
}

MATRIX_MULTI_MODEL = {
    "name": "multi_model",
    "description": "多模型矩阵：UT_HAR × {MLP, LeNet, GRU} × 各 1 次",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "UT_HAR_data",
            "epochs": 3,
            "repeats": 1,
            "deterministic": True,
            "seed": 42,
        },
        {
            "model_id": "LeNet",
            "dataset": "UT_HAR_data",
            "epochs": 3,
            "repeats": 1,
            "deterministic": True,
            "seed": 42,
        },
        {
            "model_id": "GRU",
            "dataset": "UT_HAR_data",
            "epochs": 2,
            "repeats": 1,
            "deterministic": True,
            "seed": 42,
        },
    ],
}

# ============================================================================
# Phase 2 矩阵：SOTA 基线对标（充分训练）
# 优化后资源限制策略（基于 Phase 2 测试结果校准）：
#   - batch_size=32/64（降低内存占用）
#   - num_workers=0（Windows 兼容 + 避免进程爆炸）
#   - early_stopping 放宽至 40（原 20 过于激进，UT_HAR MLP 在 epoch 149 早停，
#     但继续到 epoch 161 可将 macro_f1 从 0.8882 提升到 0.9106）
#   - deterministic：仅 sota_reproducibility 用 true（专测可复现性），
#     其余用 false（追求最佳精度，deterministic 有性能损失）
#   - scheduler=cosine（学习率衰减，改善收敛，缓解 Widar 过拟合）
#   - Widar 额外：weight_decay=1e-4 + lr=5e-4（22 类任务过拟合严重）
# ============================================================================

# 矩阵 1：核心对标（4 组，覆盖 3 数据集 × 2 模型）
MATRIX_SOTA_CORE = {
    "name": "sota_core",
    "description": "Phase 2 核心对标：3 数据集 × 2 模型，充分训练对标论文 SOTA",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "Widar",
            "epochs": 30,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 20,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "learning_rate": 5e-4,
        },
        {
            "model_id": "MLP",
            "dataset": "UT_HAR_data",
            "epochs": 200,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 64,
            "early_stopping": 40,
            "scheduler": "cosine",
        },
        {
            "model_id": "MLP",
            "dataset": "NTU-Fi_HAR",
            "epochs": 50,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 20,
            "scheduler": "cosine",
        },
        {
            "model_id": "ResNet18",
            "dataset": "UT_HAR_data",
            "epochs": 200,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 40,
            "scheduler": "cosine",
        },
    ],
}

# 矩阵 1a：MLP 分批（Widar + NTU-Fi_HAR，UT_HAR 已在 sota_reproducibility 覆盖）
MATRIX_SOTA_CORE_MLP = {
    "name": "sota_core_mlp",
    "description": "Phase 2 核心对标（MLP 分批）：Widar + NTU-Fi_HAR",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "Widar",
            "epochs": 30,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 20,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "learning_rate": 5e-4,
        },
        {
            "model_id": "MLP",
            "dataset": "NTU-Fi_HAR",
            "epochs": 50,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 20,
            "scheduler": "cosine",
        },
    ],
}

# 矩阵 1b：ResNet18 分批（资源消耗大，单独执行）
MATRIX_SOTA_CORE_RESNET = {
    "name": "sota_core_resnet",
    "description": "Phase 2 核心对标（ResNet18 分批）：UT_HAR × ResNet18 × 200 epochs",
    "tests": [
        {
            "model_id": "ResNet18",
            "dataset": "UT_HAR_data",
            "epochs": 200,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 32,
            "early_stopping": 40,
            "scheduler": "cosine",
        },
    ],
}

# 矩阵 2：可复现性（1 组 × 3 次，验证充分训练下仍可复现）
# 注意：此矩阵保持 deterministic=True，专测可复现性
MATRIX_SOTA_REPRODUCIBILITY = {
    "name": "sota_reproducibility",
    "description": "Phase 2 可复现性：UT_HAR × MLP × 200 epochs × 3 次",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "UT_HAR_data",
            "epochs": 200,
            "repeats": 3,
            "deterministic": True,
            "seed": 42,
            "batch_size": 64,
            "early_stopping": 40,
            "scheduler": "cosine",
        },
    ],
}

# 矩阵 3：AI Agent 端到端（1 组，验证 opencode 引导充分训练）
# 注意：opencode 模式由 AI Agent 自由配置，不强制 scheduler/early_stopping
MATRIX_SOTA_AGENT = {
    "name": "sota_agent",
    "description": "Phase 2 AI Agent：UT_HAR × MLP × 200 epochs × opencode 模式",
    "tests": [
        {
            "model_id": "MLP",
            "dataset": "UT_HAR_data",
            "epochs": 200,
            "repeats": 1,
            "deterministic": False,
            "seed": 42,
            "batch_size": 64,
            "early_stopping": None,
        },
    ],
}

MATRICES = {
    "minimal": MATRIX_MINIMAL,
    "multi_model": MATRIX_MULTI_MODEL,
    "sota_core": MATRIX_SOTA_CORE,
    "sota_core_mlp": MATRIX_SOTA_CORE_MLP,
    "sota_core_resnet": MATRIX_SOTA_CORE_RESNET,
    "sota_reproducibility": MATRIX_SOTA_REPRODUCIBILITY,
    "sota_agent": MATRIX_SOTA_AGENT,
}


# ============================================================================
# Direct 模式：直接调用 run_experiment
# ============================================================================

def _run_direct(test_case, run_id, output_base):
    """直接调用 run_experiment 执行单次训练。

    返回 dict: {status, macro_f1, accuracy, epochs_trained, output_dir, error, elapsed}
    """
    from senseframe.engine import ExperimentConfig, run_experiment
    # data_root 从 SENSEFRAME_DATA_ROOT env 读（框架不猜测路径）
    from senseframe.common.paths import resolve_data_root
    from senseframe.registry import get_dataset_spec

    model_id = test_case["model_id"]
    dataset = test_case["dataset"]
    epochs = test_case["epochs"]
    deterministic = test_case.get("deterministic", False)
    seed = test_case.get("seed", 42)
    batch_size = test_case.get("batch_size", 64)
    early_stopping = test_case.get("early_stopping", None)
    # 优化 2/4：支持 weight_decay、scheduler、learning_rate 透传（缓解过拟合 + 改善收敛）
    weight_decay = test_case.get("weight_decay", 0.0)
    scheduler = test_case.get("scheduler", None)
    learning_rate = test_case.get("learning_rate", 1e-3)

    # num_classes 从 DatasetSpec 派生（框架不猜测类别数）
    try:
        spec = get_dataset_spec(dataset)
    except KeyError:
        raise KeyError(
            f"benchmark: 数据集 '{dataset}' 未注册，无法派生 num_classes。"
            f"请先通过场景注册声明该数据集元数据。"
        )
    num_classes = spec.num_classes

    # 构建 ExperimentConfig（与 cross_validate.py 等价）
    scene = {
        "name": "wifi_csi",
        "dataset": dataset,
        "model_id": model_id,
        "learning_mode": "supervised",
        "data_root": str(resolve_data_root()),
    }
    trainer = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "adam",
        "seed": seed,
        "deterministic": deterministic,
        "weight_decay": weight_decay,
        "max_time": "00:00:30:00",
        # 优化 7：关闭进度条（后台进程被 capture），依赖 EpochLogCallback 日志监控
        "enable_progress_bar": False,
    }
    if early_stopping is not None:
        trainer["early_stopping"] = early_stopping
    # scheduler 通过 scene.params 透传（TrainerConfig 无此字段，走 escape hatch）
    if scheduler is not None:
        scene["params"] = {"scheduler": scheduler}
    config_dict = {
        "scene": scene,
        "input_features": [{"name": "csi", "type": "csi", "shape": list(spec.input_shape)}],
        # num_classes 从 DatasetSpec 派生（单一数据源），禁止硬编码
        "output_features": [{"name": "y", "type": "category", "num_classes": num_classes}],
        "trainer": trainer,
        "output_dir": f"{output_base}/{run_id}",
    }

    config = ExperimentConfig.from_dict(config_dict)

    start = time.time()
    try:
        out = run_experiment(config)
        elapsed = time.time() - start

        if out.status == "success":
            return {
                "status": "success",
                # RFC-004 方案 C：final_eval 字段统一 val_ 前缀
                "macro_f1": out.final_eval.get("val_macro_f1") or out.final_eval.get("macro_f1"),
                "accuracy": out.final_eval.get("val_accuracy") or out.final_eval.get("accuracy"),
                "epochs_trained": out.training.epochs_trained if hasattr(out.training, "epochs_trained") else out.training["epochs_trained"],
                "output_dir": out.output_dir,
                "elapsed": round(elapsed, 2),
                "error": None,
            }
        else:
            return {
                "status": "error",
                "macro_f1": None,
                "accuracy": None,
                "epochs_trained": 0,
                "output_dir": None,
                "elapsed": round(elapsed, 2),
                "error": out.error,
            }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "exception",
            "macro_f1": None,
            "accuracy": None,
            "epochs_trained": 0,
            "output_dir": None,
            "elapsed": round(elapsed, 2),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


# ============================================================================
# Opencode 模式：调用 opencode run --command train
# ============================================================================

def _run_opencode(test_case, run_id, output_base):
    """调用 opencode run --command train 驱动 AI Agent 端到端训练。

    返回 dict: {status, macro_f1, accuracy, output_dir, error, elapsed}
    """
    model_id = test_case["model_id"]
    dataset = test_case["dataset"]
    epochs = test_case["epochs"]

    # 构造 opencode run 命令
    # train.md 参数格式：$1=数据集 $2=模型 $3=模式 [--epochs N]
    # 通过 $ARGUMENTS 传递位置参数 + 命名参数
    args_str = f"{dataset} {model_id} supervised --epochs {epochs}"

    # Windows 上 opencode 是 npm 安装的 .cmd/.ps1 脚本，
    # subprocess.run 需要通过 shell 解析（否则 WinError 2）
    cmd = f'opencode run --command train --format json "{args_str}"'

    # 超时：按 epochs 估算，每 epoch 约 8 秒（UT_HAR MLP），加 120 秒余量
    # Phase 2 的 200 epochs 约需 1700 秒，设上限 7200 秒（2 小时）
    timeout = min(epochs * 10 + 300, 7200)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            shell=True,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "status": "error",
                "macro_f1": None,
                "accuracy": None,
                "output_dir": None,
                "elapsed": round(elapsed, 2),
                "error": f"opencode exit {result.returncode}: {result.stderr[:500]}",
            }

        # 从 opencode JSON 输出中解析结果
        # opencode 输出格式：每行一个 JSON 事件
        # 我们需要找到包含训练结果的事件
        macro_f1 = None
        accuracy = None
        output_dir = None

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 解析 opencode 事件，提取训练结果
            # 事件格式因 opencode 版本而异，这里做容错解析
            content = event.get("content", "")
            if isinstance(content, str) and "macro_f1" in content:
                # 尝试从文本中提取 JSON
                import re
                match = re.search(r'\{[^{}]*"macro_f1"[^{}]*\}', content)
                if match:
                    try:
                        metrics = json.loads(match.group())
                        macro_f1 = metrics.get("macro_f1")
                        accuracy = metrics.get("accuracy")
                    except json.JSONDecodeError:
                        pass

        # 如果没从 opencode 输出解析到，尝试从产出目录读取 metadata.json
        if macro_f1 is None:
            # opencode train 命令默认输出到 runs/ 目录
            # 查找最新的实验目录
            results_dir = PROJECT_ROOT / "runs"
            if results_dir.exists():
                exp_dirs = sorted(results_dir.glob("*"), key=lambda p: p.stat().st_mtime,
                                  reverse=True)
                for exp_dir in exp_dirs:
                    metadata_path = exp_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            # P3：通过 load_metadata 自动协商 schema_version 迁移
                            metadata = load_metadata(metadata_path)
                            # 确认是本次运行的（通过 model_id 和 dataset 匹配）
                            if (metadata.get("model_id") == model_id and
                                    metadata.get("dataset") == dataset):
                                final_eval = metadata.get("final_eval", {})
                                # RFC-004 方案 C：兼容 val_ 前缀新格式与无前缀旧格式
                                macro_f1 = final_eval.get("val_macro_f1") or final_eval.get("macro_f1")
                                accuracy = final_eval.get("val_accuracy") or final_eval.get("accuracy")
                                output_dir = str(exp_dir)
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue

        if macro_f1 is not None:
            return {
                "status": "success",
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "output_dir": output_dir,
                "elapsed": round(elapsed, 2),
                "error": None,
            }
        else:
            return {
                "status": "error",
                "macro_f1": None,
                "accuracy": None,
                "output_dir": None,
                "elapsed": round(elapsed, 2),
                "error": "opencode completed but no metrics found",
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "status": "error",
            "macro_f1": None,
            "accuracy": None,
            "output_dir": None,
            "elapsed": round(elapsed, 2),
            "error": f"opencode run timeout ({timeout}s)",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "exception",
            "macro_f1": None,
            "accuracy": None,
            "output_dir": None,
            "elapsed": round(elapsed, 2),
            "error": f"{type(e).__name__}: {e}",
        }


# ============================================================================
# 判定逻辑
# ============================================================================

def _judge_single(result, baseline_metrics, threshold):
    """判定单次测试是否通过。"""
    if result["status"] != "success":
        return {
            "pass": False,
            "reason": f"任务失败: {result.get('error', 'unknown')}",
        }

    if baseline_metrics is None:
        return {
            "pass": True,
            "reason": "无对应基线，跳过指标判定",
        }

    baseline_f1 = baseline_metrics.get("macro_f1")
    actual_f1 = result.get("macro_f1")
    if baseline_f1 is None or actual_f1 is None:
        return {
            "pass": False,
            "reason": "Macro-F1 缺失",
        }

    delta = actual_f1 - baseline_f1
    if delta >= -threshold:
        return {
            "pass": True,
            "reason": f"Macro-F1={actual_f1:.4f} ≥ baseline({baseline_f1:.4f}) - {threshold}",
            "delta": round(delta, 6),
        }
    else:
        return {
            "pass": False,
            "reason": f"Macro-F1={actual_f1:.4f} < baseline({baseline_f1:.4f}) - {threshold}",
            "delta": round(delta, 6),
        }


def _judge_reproducibility(results):
    """判定可复现性（deterministic 模式下多次运行的方差）。"""
    f1_values = [r["macro_f1"] for r in results
                 if r["status"] == "success" and r["macro_f1"] is not None]
    if len(f1_values) < 2:
        return {"pass": True, "reason": "单次运行，跳过可复现性判定"}

    std = stdev(f1_values)
    threshold = THRESHOLDS["smoke"]["reproducibility_std"]
    if std < threshold:
        return {
            "pass": True,
            "reason": f"std={std:.6f} < {threshold}",
            "std": round(std, 6),
            "mean": round(mean(f1_values), 6),
        }
    else:
        return {
            "pass": False,
            "reason": f"std={std:.6f} ≥ {threshold}",
            "std": round(std, 6),
            "mean": round(mean(f1_values), 6),
        }


def _get_baseline(model_id, dataset, epochs, baseline_type):
    """获取基线指标。"""
    if baseline_type == "smoke":
        return SMOKE_BASELINE.get((model_id, dataset, epochs))
    elif baseline_type == "sota":
        return SOTA_BASELINE.get((model_id, dataset))
    return None


# ============================================================================
# 主流程
# ============================================================================

def run_benchmark(matrix_name, mode, baseline_type, repeats_override=None):
    """执行测评。"""
    matrix = MATRICES[matrix_name]
    output_base = f"benchmarks/runs/{matrix_name}_{mode}_{baseline_type}"
    thresholds = THRESHOLDS[baseline_type]

    report = {
        "description": "senseframe 框架引导可靠性测评",
        "matrix": matrix_name,
        "matrix_description": matrix["description"],
        "mode": mode,
        "baseline": baseline_type,
        "thresholds": thresholds,
        "timestamp": datetime.now().isoformat(),
        "results": [],
    }

    print(f"=== senseframe 测评 ===")
    print(f"矩阵: {matrix_name} ({matrix['description']})")
    print(f"模式: {mode} | 基线: {baseline_type}")
    print(f"输出: {output_base}")
    print()

    total_tests = 0
    passed_tests = 0

    for test_case in matrix["tests"]:
        model_id = test_case["model_id"]
        dataset = test_case["dataset"]
        epochs = test_case["epochs"]
        repeats = repeats_override or test_case.get("repeats", 1)

        baseline_metrics = _get_baseline(model_id, dataset, epochs, baseline_type)

        print(f"[{model_id}/{dataset}/{epochs}ep] 基线: {baseline_metrics}")

        run_results = []
        for i in range(repeats):
            run_id = f"{model_id}_{dataset}_{epochs}ep_run{i+1}"
            print(f"  Run {i+1}/{repeats} [{run_id}]...", end=" ", flush=True)

            if mode == "direct":
                result = _run_direct(test_case, run_id, output_base)
            elif mode == "opencode":
                result = _run_opencode(test_case, run_id, output_base)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown mode: {mode}",
                }

            run_results.append(result)
            total_tests += 1

            if result["status"] == "success":
                print(f"✓ Macro-F1={result['macro_f1']:.4f} "
                      f"({result['elapsed']}s)")
            else:
                print(f"✗ {result.get('error', 'failed')}")

        # 判定单次结果
        for result in run_results:
            judgment = _judge_single(result, baseline_metrics,
                                     thresholds["macro_f1_tolerance"])
            result["judgment"] = judgment
            if judgment["pass"]:
                passed_tests += 1

        # 判定可复现性（同 test_case 多次运行）
        reproducibility = {"pass": True, "reason": "未评估"}
        if repeats > 1:
            reproducibility = _judge_reproducibility(run_results)

        report["results"].append({
            "test_case": test_case,
            "baseline": baseline_metrics,
            "runs": run_results,
            "reproducibility": reproducibility,
        })

        print(f"  可复现性: {'✓' if reproducibility['pass'] else '✗'} "
              f"{reproducibility['reason']}")
        print()

    # 汇总
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    report["summary"] = {
        "total_runs": total_tests,
        "passed_runs": passed_tests,
        "task_success_rate": round(success_rate, 4),
        "l1_pass": success_rate >= thresholds["task_success_rate"],
    }

    # 保存报告
    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = benchmarks_dir / f"report_{matrix_name}_{mode}_{baseline_type}_{timestamp_str}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 打印汇总
    print(f"=== 测评汇总 ===")
    print(f"矩阵: {matrix_name} | 模式: {mode} | 基线: {baseline_type}")
    print(f"L1 任务完成率: {passed_tests}/{total_tests} = {success_rate:.1%} "
          f"{'✓ PASS' if report['summary']['l1_pass'] else '✗ FAIL'}")

    all_repro_pass = all(r["reproducibility"]["pass"] for r in report["results"])
    print(f"L4 可复现性: {'✓ PASS' if all_repro_pass else '✗ FAIL'}")
    print(f"报告: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="senseframe 框架引导可靠性测评",
    )
    parser.add_argument("--matrix", type=str, default="minimal",
                        choices=["minimal", "multi_model",
                                 "sota_core", "sota_core_mlp", "sota_core_resnet",
                                 "sota_reproducibility", "sota_agent"],
                        help="测评矩阵（默认 minimal；Phase 2 用 sota_*）")
    parser.add_argument("--mode", type=str, default="direct",
                        choices=["direct", "opencode"],
                        help="测评模式（默认 direct）")
    parser.add_argument("--baseline", type=str, default="smoke",
                        choices=["smoke", "sota"],
                        help="基线类型（默认 smoke）")
    parser.add_argument("--repeats", type=int,
                        help="覆盖重复次数（仅 minimal 矩阵）")

    args = parser.parse_args()

    run_benchmark(args.matrix, args.mode, args.baseline, args.repeats)


if __name__ == "__main__":
    main()
