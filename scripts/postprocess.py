#!/usr/bin/env python
"""训练后处理：将模型/元数据/推理脚本统一整理到 output_dir 内，生成产物清单。

设计原则（P0-1.5 路径安全修复）：
- 所有后处理产物均在 output_dir 内（deployed/ 子目录 + output_dir/eval.py + output_dir/result/）
- manifest.json 中存储相对 output_dir 的相对路径，禁止绝对路径
- 框架不为"受信任脚本"开后门，path_safe 策略对 postprocess 一视同仁

用法：
    python postprocess.py --output-dir runs/MLP_UT_HAR_data_20240101_120000_123

产物布局（全部位于 output_dir 内）：
    output_dir/
    ├── metadata.json           （训练时生成）
    ├── model.pth               （训练时生成）
    ├── manifest.json           （训练时生成 + postprocess 追加）
    ├── eval.py                 （postprocess 生成）
    ├── deployed/
    │   ├── {model}_{dataset}_{mode}.pth
    │   └── {model}_{dataset}_{mode}.json
    └── result/                 （推理结果存放目录，仅创建）
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# 将项目根目录加入 sys.path（bootstrap：senseframe 可导入前的必要本地推导）
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


def generate_artifact_manifest(
    output_dir: Path,
    dest_model_path: Path,
    dest_metadata_path: Path,
    eval_script_path: Path,
    metadata: dict,
) -> Path:
    """追加后处理产物到框架 manifest.json（统一 schema A，相对路径存储）。

    P0-1.5 修复：所有产物均在 output_dir 内，path 字段存储相对 output_dir 的相对路径，
    禁止绝对路径。复用框架 ArtifactManifest.save() 写回，schema 与训练阶段一致。
    """
    from senseframe.engine.runner.artifacts import (
        ArtifactDescriptor, ArtifactManifest, sha256_file,
    )

    manifest_path = output_dir / "manifest.json"

    # P5 P3-6：manifest 加载失败时 raise 而非静默降级。
    # 旧代码用空字段构造 manifest，丢失 run_id/config_hash/data_hash 等溯源元数据，
    # 破坏 verify_artifacts 链路。
    try:
        manifest = ArtifactManifest.load(manifest_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"postprocess: 训练阶段 manifest.json 不存在: {manifest_path}。"
            f"请确认 output_dir 指向已完成训练的输出目录。"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"postprocess: 加载 manifest.json 失败: {type(e).__name__}: {e}。"
            f"manifest 可能已损坏，请检查 {manifest_path} 或重新运行训练。"
        ) from e

    # 2. 追加后处理产物（相对路径存储，无 output_dir 外路径）
    # 幂等：同名产物已存在则跳过
    existing_names = {a.name for a in manifest.artifacts}

    postprocess_artifacts = [
        (dest_model_path, "deployed_model", "model",
         {"format": "state_dict", "deployed": True}),
        (dest_metadata_path, "deployed_metadata", "metadata",
         {"deployed": True}),
        (eval_script_path, "inference_script", "log",
         {"format": "python", "entry": "eval.py"}),
    ]

    for path, name, kind, content_schema in postprocess_artifacts:
        if name in existing_names:
            continue
        if not path.exists():
            continue
        # P0-1.5: 强制相对路径，禁止绝对路径存储
        try:
            rel_path = path.relative_to(output_dir)
        except ValueError:
            # 不应发生：所有产物应在 output_dir 内
            import logging
            logging.getLogger(__name__).error(
                f"generate_artifact_manifest: artifact {name} path {path} "
                f"is outside output_dir {output_dir}, skipping"
            )
            continue
        try:
            desc = ArtifactDescriptor(
                name=name,
                path=str(rel_path),  # 相对 output_dir 的相对路径
                kind=kind,
                producer_stage="postprocess",
                content_hash=sha256_file(path),
                size_bytes=path.stat().st_size,
                content_schema=content_schema,
            )
            manifest.artifacts.append(desc)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"generate_artifact_manifest: failed to append {name}: {e}"
            )

    # 3. 用框架 API 写回（统一 schema A）
    return manifest.save(output_dir)


def postprocess(output_dir: str) -> dict:
    """训练后处理：所有产物整理到 output_dir 内。

    Args:
        output_dir: 训练输出目录（含 metadata.json + model.pth），产物也写入此目录

    Returns:
        dict: 后处理结果摘要（路径均为相对 output_dir 的相对路径）
    """
    output_path = Path(output_dir).resolve()
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

    # 1. 拷贝模型到 output_dir/deployed/
    deployed_dir = output_path / "deployed"
    deployed_dir.mkdir(parents=True, exist_ok=True)
    dest_model_path = deployed_dir / model_filename
    shutil.copy2(model_path, dest_model_path)

    # 2. 拷贝 metadata 到 output_dir/deployed/（同名 .json）
    dest_metadata_path = deployed_dir / f"{model_id}_{dataset}_{learning_mode}.json"
    shutil.copy2(metadata_path, dest_metadata_path)

    # 3. 生成推理脚本到 output_dir/eval.py
    eval_script_path = output_path / "eval.py"
    from scripts.generate_inference import generate_inference_script
    generate_inference_script(str(metadata_path), str(eval_script_path))

    # 4. 创建 output_dir/result/ 目录（用于存放后续推理结果）
    result_path = output_path / "result"
    result_path.mkdir(parents=True, exist_ok=True)

    # 5. 生成产物清单 manifest.json（追加后处理产物，相对路径）
    manifest_path = generate_artifact_manifest(
        output_path, dest_model_path, dest_metadata_path,
        eval_script_path, metadata,
    )

    # 6. 输出摘要（相对路径，便于跨机器复用）
    summary = {
        "status": "success",
        "model_id": model_id,
        "dataset": dataset,
        "learning_mode": learning_mode,
        "training_output_dir": str(output_path),
        "model_file": str(dest_model_path.relative_to(output_path)),
        "model_metadata": str(dest_metadata_path.relative_to(output_path)),
        "eval_script": str(eval_script_path.relative_to(output_path)),
        "result_dir": str(result_path.relative_to(output_path)),
        "manifest": str(manifest_path.relative_to(output_path)),
        "usage": {
            "batch_eval": f"python eval.py --checkpoint deployed/{model_filename} --batch-eval",
            "single_sample": f"python eval.py --checkpoint deployed/{model_filename} --single-sample --input sample.npy",
            "result_output": f"python eval.py --checkpoint deployed/{model_filename} --batch-eval --output result/eval_result.json",
        },
        "final_eval": metadata.get("final_eval", {}),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        prog="postprocess",
        description="训练后处理：将模型/元数据/推理脚本整理到 output_dir 内",
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="训练输出目录（含 metadata.json + model.pth，产物也写入此目录）")

    args = parser.parse_args()

    try:
        summary = postprocess(args.output_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    except (FileNotFoundError, RuntimeError) as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
