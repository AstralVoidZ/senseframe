#!/usr/bin/env python
"""训练后处理：拷贝模型到 models/ + 生成推理脚本 eval.py 到项目根。

用法：
    python postprocess.py --output-dir runs/MLP_UT_HAR_data_20240101_120000_123
    python postprocess.py --output-dir runs/... --models-dir models --result-dir result

功能：
    1. 读取训练输出目录的 metadata.json 和 model.pth
    2. 拷贝 model.pth 到 models/ 目录（带语义命名）
    3. 调用 generate_inference.py 生成推理脚本 eval.py 到项目根
    4. 创建 result/ 目录（用于存放后续推理结果）
    5. Phase 2.2d：生成 artifact_manifest.json（产物清单 + 校验和）
    6. 输出结构化 JSON 摘要

产物职责：
    - models/    存放模型权重 + 元数据
    - eval.py    推理脚本入口（项目根，运行时 --output 指定结果到 result/）
    - result/    存放推理运行产生的结果 JSON
    - artifact_manifest.json  产物清单（含路径/大小/SHA256，便于校验与追溯）

本脚本不侵入 senseframe 框架，仅做训练后的产物整理。
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _file_sha256(path: Path) -> str:
    """计算文件 SHA256 校验和（分块读取，支持大文件）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_artifact_entry(path: Path, base_dir: Path, kind: str,
                          compute_hash: bool = True) -> dict:
    """构建单个产物的 manifest 条目。"""
    # is_relative_to 为 Python 3.9+，此处做兼容回退
    try:
        rel = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else None
    except AttributeError:
        try:
            rel = str(path.relative_to(base_dir))
        except ValueError:
            rel = None
    entry = {
        "kind": kind,
        "path": str(path.resolve()),
        "relative_path": rel,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if compute_hash and path.exists() and path.is_file():
        entry["sha256"] = _file_sha256(path)
    return entry


def generate_artifact_manifest(
    output_dir: Path,
    dest_model_path: Path,
    dest_metadata_path: Path,
    eval_script_path: Path,
    metadata: dict,
) -> Path:
    """
    Phase 2.2d：生成产物清单 artifact_manifest.json。

    清单记录训练 + 后处理产生的所有关键产物，含路径、大小、SHA256 校验和，
    便于产物完整性校验、版本追溯与下游工具消费。

    Args:
        output_dir: 训练输出目录
        dest_model_path: 拷贝后的模型权重路径
        dest_metadata_path: 拷贝后的 metadata 路径
        eval_script_path: 生成的推理脚本路径
        metadata: 训练 metadata dict

    Returns:
        manifest 文件路径
    """
    artifacts = []

    # 训练输出目录内的产物
    for name, kind in [
        ("model.pth", "model_weights"),
        ("metadata.json", "metadata"),
        ("config.yaml", "config"),
        ("training_log.jsonl", "training_log"),
    ]:
        p = output_dir / name
        if p.exists():
            artifacts.append(_build_artifact_entry(p, output_dir, kind))

    # checkpoints 目录
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        for ckpt in ckpt_dir.glob("*.ckpt"):
            artifacts.append(_build_artifact_entry(ckpt, output_dir, "checkpoint"))

    # 后处理产物
    if dest_model_path.exists():
        artifacts.append(_build_artifact_entry(
            dest_model_path, _PROJECT_ROOT, "deployed_model"))
    if dest_metadata_path.exists():
        artifacts.append(_build_artifact_entry(
            dest_metadata_path, _PROJECT_ROOT, "deployed_metadata"))
    if eval_script_path.exists():
        artifacts.append(_build_artifact_entry(
            eval_script_path, _PROJECT_ROOT, "inference_script"))

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "model_id": metadata.get("model_id"),
        "dataset": metadata.get("dataset"),
        "learning_mode": metadata.get("learning_mode"),
        "training_output_dir": str(output_dir.resolve()),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }

    manifest_path = output_dir / "artifact_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    return manifest_path


def postprocess(output_dir: str, models_dir: str = "models",
                result_dir: str = "result", eval_script: str = "eval.py") -> dict:
    """训练后处理。

    Args:
        output_dir: 训练输出目录（含 metadata.json + model.pth）
        models_dir: 模型拷贝目标目录
        result_dir: 推理结果存放目录（仅创建，不写入）
        eval_script: 推理脚本输出路径（默认: eval.py，项目根）

    Returns:
        dict: 后处理结果摘要
    """
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"
    model_path = output_path / "model.pth"

    # 校验必需文件
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {output_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth not found in {output_dir}")

    # 读取 metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_id = metadata.get("model_id", "unknown")
    dataset = metadata.get("dataset", "unknown")
    learning_mode = metadata.get("learning_mode", "supervised")

    # 构建语义化文件名
    model_filename = f"{model_id}_{dataset}_{learning_mode}.pth"

    # 1. 拷贝模型到 models/
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    dest_model_path = models_path / model_filename
    shutil.copy2(model_path, dest_model_path)

    # 2. 拷贝 metadata 到 models/（同名 .json）
    dest_metadata_path = models_path / f"{model_id}_{dataset}_{learning_mode}.json"
    shutil.copy2(metadata_path, dest_metadata_path)

    # 3. 生成推理脚本（路径由参数指定，默认项目根 eval.py）
    eval_script_path = Path(eval_script)
    eval_script_path.parent.mkdir(parents=True, exist_ok=True)
    from scripts.generate_inference import generate_inference_script
    generate_inference_script(str(metadata_path), str(eval_script_path))

    # 4. 创建 result/ 目录（用于存放后续推理结果）
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    # 5. Phase 2.2d：生成产物清单 artifact_manifest.json
    manifest_path = generate_artifact_manifest(
        output_path, dest_model_path, dest_metadata_path,
        eval_script_path, metadata,
    )

    # 6. 输出摘要
    summary = {
        "status": "success",
        "model_id": model_id,
        "dataset": dataset,
        "learning_mode": learning_mode,
        "training_output_dir": str(output_path),
        "model_file": str(dest_model_path),
        "model_metadata": str(dest_metadata_path),
        "eval_script": str(eval_script_path),
        "result_dir": str(result_path),
        "artifact_manifest": str(manifest_path),
        "usage": {
            "batch_eval": f"python eval.py --checkpoint {dest_model_path} --batch-eval",
            "single_sample": f"python eval.py --checkpoint {dest_model_path} --single-sample --input sample.npy",
            "result_output": f"python eval.py --checkpoint {dest_model_path} --batch-eval --output {result_dir}/eval_result.json",
        },
        "final_eval": metadata.get("final_eval", {}),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        prog="postprocess",
        description="训练后处理：拷贝模型到 models/ + 生成 eval.py 到项目根",
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="训练输出目录（含 metadata.json + model.pth）")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="模型拷贝目标目录（默认: models）")
    parser.add_argument("--result-dir", type=str, default="result",
                        help="推理结果存放目录（默认: result）")
    parser.add_argument("--eval-script", type=str, default="eval.py",
                        help="推理脚本输出路径（默认: eval.py，项目根）")

    args = parser.parse_args()

    try:
        summary = postprocess(args.output_dir, args.models_dir, args.result_dir,
                              args.eval_script)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    except FileNotFoundError as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
