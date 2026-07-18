#!/usr/bin/env python
"""根据训练 metadata 生成推理脚本。

用法：
    python generate_inference.py --metadata runs/MLP_UT_HAR_data_20240101_120000_123/metadata.json
    python generate_inference.py --metadata metadata.json --output result/infer.py

功能：
    1. 读取训练输出的 metadata.json（含 model_id/dataset/num_classes/normalization）
    2. 生成独立的推理脚本，支持：
       - --batch-eval：批量评测整个测试集，对照真值打分
       - --single-sample：单样本推理
       - --test-data：指定测试集路径
       - --checkpoint：指定模型权重路径
    3. 推理脚本自包含，可独立运行
"""

import argparse
import json
import sys
from pathlib import Path

# bootstrap：senseframe 可导入前的必要本地推导
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from senseframe.engine.metadata import load_metadata  # noqa: E402

# 推理脚本模板
# 使用 $VAR 占位符避免 f-string 冲突，运行时用 str.replace 注入
_INFERENCE_TEMPLATE = '''#!/usr/bin/env python
"""推理脚本：加载训练好的模型进行预测或批量评测。

由 generate_inference.py 自动生成，基于训练 metadata：
  - model_id: $__MODEL_ID__
  - dataset: $__DATASET__
  - num_classes: $__NUM_CLASSES__
  - learning_mode: $__LEARNING_MODE__
  - normalization: $__NORMALIZATION__

用法：
    # 批量评测测试集（对照真值打分，结果到 result/eval_result.json）
    python eval.py --checkpoint models/model.pth --batch-eval

    # 单样本推理
    python eval.py --checkpoint models/model.pth --single-sample --input sample.npy

    # 指定测试集路径
    python eval.py --checkpoint models/model.pth --batch-eval --test-data /path/to/test_data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 向上搜索 senseframe 包所在的项目根，不硬编码 eval.py 的位置
# eval.py 可能位于项目根、子目录或任意部署目录，只要能找到 senseframe 包即可
_PROJECT_ROOT = None
_search = Path(__file__).resolve().parent
for _candidate in [_search] + list(_search.parents):
    if (_candidate / "senseframe" / "__init__.py").exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    # 回退：当前文件所在目录（向后兼容）
    _PROJECT_ROOT = _search
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from senseframe.scenes import get_scene  # noqa: E402
from senseframe.common import load_checkpoint_flexible  # noqa: E402


# 训练时记录的元数据
MODEL_ID = "$__MODEL_ID__"
DATASET = "$__DATASET__"
NUM_CLASSES = $__NUM_CLASSES__
LEARNING_MODE = "$__LEARNING_MODE__"
NORMALIZATION = $__NORMALIZATION__
DATA_ROOT = "$__DATA_ROOT__"


def load_model(checkpoint_path: str):
    """加载模型权重，返回已加载 state_dict 的模型实例。

    使用 load_checkpoint_flexible 兼容 Lightning checkpoint（含 state_dict 顶层 key
    + model. 前缀）与裸 state_dict 两种格式。旧代码直接 torch.load +
    model.load_state_dict(state_dict) 会在加载 Lightning .ckpt 时触发
    `Unexpected key(s) in state_dict: "epoch", "global_step", ...` 错误。
    """
    scene = get_scene("wifi_csi")
    model = scene.build_model_for_dataset(
        MODEL_ID, DATASET, NUM_CLASSES, learning_mode=LEARNING_MODE,
    )
    # weights_only=False 兼容 Lightning ckpt 的 callbacks 字段（含 Python 对象）
    load_checkpoint_flexible(
        checkpoint_path, model, map_location="cpu", weights_only=False,
    )
    model.eval()
    return model


def load_test_dataset(test_data_path: str = None):
    """加载测试集，返回 (test_dataset, dataset_name)。"""
    scene = get_scene("wifi_csi")
    if test_data_path:
        # 自定义测试集路径：尝试用场景容器加载指定路径的数据
        # 注意：senseframe 的 CSIDataModule 按数据集名加载，自定义路径需用户自行适配
        print(f"警告：自定义测试集路径 {test_data_path} 需手动适配数据加载逻辑")
        print(f"当前使用默认数据集 {DATASET} 的测试集")
    _, test_ds = scene.load_dataset(DATASET, root=DATA_ROOT,
                                     learning_mode="supervised")
    return test_ds


def batch_evaluate(model, test_dataset, batch_size: int = 64):
    """批量评测：对整个测试集预测，对照真值计算指标。

    返回 dict: {accuracy, macro_f1, confusion_matrix, predictions}
    """
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # CSI 数据集的 batch 格式：(x, y) 或 dict
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
            else:
                raise ValueError(f"不支持的 batch 格式: {type(batch)}")

            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    accuracy = (all_preds == all_labels).mean()

    # macro F1
    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_preds),
        "predictions": all_preds.tolist(),
        "ground_truth": all_labels.tolist(),
    }


def single_sample_predict(model, sample_input):
    """单样本推理：输入单个样本，返回预测类别和置信度。

    Args:
        model: 已加载的模型
        sample_input: numpy array 或 tensor，形状匹配模型输入

    Returns:
        dict: {predicted_class, confidence, logits}
    """
    if isinstance(sample_input, np.ndarray):
        sample_input = torch.from_numpy(sample_input).float()

    # 添加 batch 维度
    if sample_input.dim() == len(__import__("senseframe").scenes.get_scene("wifi_csi").get_dataset_info(DATASET).get("input_shape", [270, 3])):
        sample_input = sample_input.unsqueeze(0)

    with torch.no_grad():
        logits = model(sample_input)
        probs = torch.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class].item()

    return {
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "logits": logits[0].tolist(),
        "probabilities": probs[0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        prog="infer",
        description=f"推理脚本 (model={MODEL_ID}, dataset={DATASET})",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径 (.pth)")
    parser.add_argument("--batch-eval", action="store_true",
                        help="批量评测整个测试集")
    parser.add_argument("--single-sample", action="store_true",
                        help="单样本推理")
    parser.add_argument("--input", type=str,
                        help="单样本推理的输入文件路径 (.npy)")
    parser.add_argument("--test-data", type=str,
                        help="测试集路径（默认用训练时的数据集）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量评测的 batch size")
    parser.add_argument("--output", type=str, default="result/eval_result.json",
                        help="结果输出文件路径 (.json)，默认 result/eval_result.json")

    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint)
    print(f"模型加载完成: {MODEL_ID} ({DATASET}, {NUM_CLASSES} classes)")

    result = {}

    if args.batch_eval:
        print("批量评测模式")
        test_ds = load_test_dataset(args.test_data)
        print(f"测试集大小: {len(test_ds)}")
        result = batch_evaluate(model, test_ds, args.batch_size)
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Micro F1: {result['micro_f1']:.4f}")
        print(f"样本数: {result['num_samples']}")

    elif args.single_sample:
        print("单样本推理模式")
        if not args.input:
            print("错误：单样本推理需要 --input 参数", file=sys.stderr)
            sys.exit(1)
        sample = np.load(args.input)
        result = single_sample_predict(model, sample)
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")

    else:
        print("错误：请指定 --batch-eval 或 --single-sample", file=sys.stderr)
        sys.exit(1)

    # 输出结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"结果已保存到: {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
'''


def generate_inference_script(metadata_path: str, output_path: str = None) -> str:
    """根据 metadata.json 生成推理脚本。

    Args:
        metadata_path: metadata.json 文件路径
        output_path: 输出推理脚本路径，None 则输出到 stdout

    Returns:
        生成的推理脚本内容
    """
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # P3：通过 load_metadata 自动协商 schema_version 迁移
    metadata = load_metadata(metadata_file)

    # 提取关键字段
    model_id = metadata.get("model_id", "Unknown")
    dataset = metadata.get("dataset", "Unknown")
    # num_classes 必填：框架不猜测数据集类别数，metadata 缺失则 raise
    if "num_classes" not in metadata:
        raise KeyError(
            "metadata 中缺少 num_classes，无法生成推理脚本。"
            "请确保训练时 metadata 完整记录了 num_classes。"
        )
    num_classes = metadata["num_classes"]
    learning_mode = metadata.get("learning_mode", "supervised")
    normalization = metadata.get("normalization")

    # data_root 必填：从 metadata.config 读取，缺失则 raise（框架不猜测路径）
    config_dict = metadata.get("config", {})
    data_root = config_dict.get("data_root")
    if not data_root:
        raise KeyError(
            "metadata.config 中缺少 data_root，无法生成推理脚本。"
            "请确保训练时 metadata.config 完整记录了 data_root。"
        )

    # 填充模板
    script = _INFERENCE_TEMPLATE
    script = script.replace("$__MODEL_ID__", str(model_id))
    script = script.replace("$__DATASET__", str(dataset))
    script = script.replace("$__NUM_CLASSES__", str(num_classes))
    script = script.replace("$__LEARNING_MODE__", str(learning_mode))
    script = script.replace("$__DATA_ROOT__", str(data_root))
    # normalization 为 None 时用 Python 的 None，非 None 时用 dict 字面量
    if normalization is None:
        script = script.replace("$__NORMALIZATION__", "None")
    else:
        script = script.replace("$__NORMALIZATION__", repr(normalization))

    # 输出
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(script)
        print(f"推理脚本已生成: {output_path}", file=sys.stderr)
    else:
        print(script)

    return script


def main():
    parser = argparse.ArgumentParser(
        prog="generate_inference",
        description="根据训练 metadata 生成推理脚本",
    )
    parser.add_argument("--metadata", type=str, required=True,
                        help="训练输出的 metadata.json 路径")
    parser.add_argument("--output", type=str, default="eval.py",
                        help="输出推理脚本路径（默认: eval.py，项目根）")

    args = parser.parse_args()
    generate_inference_script(args.metadata, args.output)


if __name__ == "__main__":
    main()
