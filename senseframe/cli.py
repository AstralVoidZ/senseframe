"""
CLI 接口：7 个子命令，所有输出为结构化 JSON。

命令：
- probe:         探测硬件资源
- list-models:   列出可用模型（可按数据集/范式过滤）
- list-datasets: 列出可用数据集
- list-scenes:   列出已注册的场景容器
- paradigms:     列出 SOTA 范式
- recommend:     根据资源推荐可用模型
- experiment:    执行声明式实验（YAML → ExperimentConfig → run_pipeline）
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .engine import ExperimentConfig, run_hpo
from .engine.runner import run_experiment
from .observability import setup_logging
from .registry import DATASET_INFO, list_datasets, list_models
from .routing import ResourceProbe, ResourceRouter
from .scenes import list_scenes, activate_lazy_scenes

_logger = logging.getLogger(__name__)


# ============================================================
# SOTA 范式知识库（Phase R6：从 paradigms.py 内联）
# 预埋业界最优 WiFi CSI 动作识别训练范式，供 CLI paradigms 子命令使用。
# ============================================================

@dataclass
class Paradigm:
    """单个训练范式的结构化描述。"""

    name: str
    category: str
    description: str
    applicable_models: List[str]
    best_for_datasets: List[str]
    resource_level: str  # minimal / low / medium / high / extreme
    expected_accuracy_range: str
    training_time: str
    key_papers: List[str] = field(default_factory=list)
    config_template: Dict[str, Any] = field(default_factory=dict)
    selection_criteria: str = ""
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "applicable_models": self.applicable_models,
            "best_for_datasets": self.best_for_datasets,
            "resource_level": self.resource_level,
            "expected_accuracy_range": self.expected_accuracy_range,
            "training_time": self.training_time,
            "key_papers": self.key_papers,
            "config_template": self.config_template,
            "selection_criteria": self.selection_criteria,
            "pros": self.pros,
            "cons": self.cons,
        }


PARADIGM_REGISTRY: Dict[str, Paradigm] = {
    "traditional_ml": Paradigm(
        name="traditional_ml",
        category="传统机器学习",
        description="使用 SVM / Random Forest / KNN 等传统算法，配合手工特征（方差、均值、频域特征）进行分类。适合作为基线或资源极度受限场景。",
        applicable_models=["MLP"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi_HAR", "Widar"],
        resource_level="minimal",
        expected_accuracy_range="60%-80%",
        training_time="秒级",
        key_papers=[
            "Wang et al., 'Understanding and Using the WiFi Sensing Channel', 2024",
        ],
        config_template={
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "device": "cpu",
        },
        selection_criteria="资源极度受限（纯CPU、内存<2GB）或需要快速基线对比时选择。",
        pros=["训练极快", "资源需求极低", "可解释性强"],
        cons=["精度上限低", "需手工特征工程", "泛化能力弱"],
    ),
    "cnn": Paradigm(
        name="cnn",
        category="卷积神经网络",
        description="将 CSI 数据图像化后使用 CNN 提取空间特征。LeNet 适合轻量场景，ResNet18/50/101 逐级提升精度但增加资源消耗。当前 WiFi CSI HAR 最主流的范式。",
        applicable_models=["LeNet", "ResNet18", "ResNet50", "ResNet101"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        resource_level="medium",
        expected_accuracy_range="80%-95%",
        training_time="分钟级",
        key_papers=[
            "Ma et al., 'WiFi Sensing with Channel State Information: A Survey', 2022",
            "Li et al., 'AutoFi: Automated WiFi Human Activity Recognition', 2023",
        ],
        config_template={
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
        },
        selection_criteria="通用首选。ResNet18 是精度/速度最佳平衡点；ResNet50/101 适合追求极致精度且有 GPU 的场景。",
        pros=["精度高", "特征提取能力强", "成熟稳定"],
        cons=["ResNet50/101 需 GPU", "对时序信息建模弱", "数据量小时易过拟合"],
    ),
    "rnn": Paradigm(
        name="rnn",
        category="循环神经网络",
        description="利用 RNN / GRU / LSTM / BiLSTM 对 CSI 时序序列建模，捕捉动作的时间动态。适合需要时间维度信息的场景。",
        applicable_models=["RNN", "GRU", "LSTM", "BiLSTM"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        resource_level="medium",
        expected_accuracy_range="75%-90%",
        training_time="分钟级（RNN 可能需要小时级，因 epoch 数大）",
        key_papers=[
            "Wang et al., 'A Deep Learning Approach for Activity Recognition', 2017",
        ],
        config_template={
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
        },
        selection_criteria="当动作的时间模式是关键区分因素时选择。GRU/LSTM 通常优于原始 RNN；BiLSTM 适合需要双向上下文的场景。",
        pros=["时序建模能力强", "适合序列数据", "GRU/LSTM 训练稳定"],
        cons=["RNN 训练慢（epoch 多）", "长序列梯度消失", "并行度低"],
    ),
    "hybrid": Paradigm(
        name="hybrid",
        category="CNN+RNN 混合",
        description="CNN 提取空间特征 + RNN 建模时序动态，时空联合建模。CNN+GRU 是代表方法，兼顾空间和时间信息。",
        applicable_models=["CNN+GRU"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        resource_level="medium",
        expected_accuracy_range="82%-93%",
        training_time="分钟级",
        key_papers=[
            "Li et al., 'AutoFi: Automated WiFi Human Activity Recognition', 2023",
        ],
        config_template={
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
        },
        selection_criteria="当单一 CNN 或 RNN 无法满足精度要求，且需要同时捕捉空间和时序特征时选择。",
        pros=["时空联合建模", "精度通常优于单一 CNN/RNN", "结构灵活"],
        cons=["参数量较大", "训练时间比单一模型长", "调参复杂度增加"],
    ),
    "transformer": Paradigm(
        name="transformer",
        category="Transformer / 注意力机制",
        description="使用 ViT 等 Transformer 架构，通过自注意力机制捕捉 CSI 数据的全局依赖关系。适合大数据集和高精度需求场景。",
        applicable_models=["ViT"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        resource_level="high",
        expected_accuracy_range="85%-96%",
        training_time="分钟级至小时级",
        key_papers=[
            "Dosovitskiy et al., 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', 2021",
            "Li et al., 'AutoFi: Automated WiFi Human Activity Recognition', 2023",
        ],
        config_template={
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "device": "cuda",
        },
        selection_criteria="数据量充足、有 GPU 资源、追求最高精度时选择。小数据集上可能不如 CNN。",
        pros=["全局注意力建模", "大数据集上精度最高", "可扩展性强"],
        cons=["需 GPU", "小数据集易过拟合", "训练资源消耗大"],
    ),
    "self_supervised": Paradigm(
        name="self_supervised",
        category="自监督学习",
        description="AutoFi 范式：先在无标注数据上自监督预训练（EntLoss: KL+EH+HE+KDE），再在有标注数据上监督微调。适合标注数据稀缺的场景。",
        applicable_models=["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
                           "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"],
        best_for_datasets=["NTU-Fi_HAR", "NTU-Fi-HumanID"],
        resource_level="high",
        expected_accuracy_range="85%-95%",
        training_time="小时级（两阶段：100 epoch 预训练 + 300 epoch 微调）",
        key_papers=[
            "Li et al., 'AutoFi: Automated WiFi Human Activity Recognition', 2023",
        ],
        config_template={
            "learning_mode": "self_supervised",
            "epochs": 100,
            "supervised_epochs": 300,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 1.5e-6,
        },
        selection_criteria="标注数据稀缺、有大量未标注 CSI 数据时选择。两阶段训练资源消耗较大。",
        pros=["利用无标注数据", "标注数据少时优势明显", "特征学习能力强"],
        cons=["两阶段训练耗时长", "资源消耗大", "需大量无标注数据"],
    ),
    "cross_domain": Paradigm(
        name="cross_domain",
        category="跨域泛化",
        description="通过域适应（Domain Adaptation）或元学习（Meta-Learning）提升模型在新环境、新用户、新位置的泛化能力。解决 WiFi CSI 的环境依赖问题。",
        applicable_models=[],
        best_for_datasets=["UT_HAR_data", "NTU-Fi_HAR", "Widar"],
        resource_level="high",
        expected_accuracy_range="70%-88%（跨域场景）",
        training_time="小时级",
        key_papers=[
            "Zeng et al., 'A Survey of WiFi Sensing: From Theory to Applications', 2024",
        ],
        config_template={
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "domain_adaptation": True,
        },
        selection_criteria="模型需要部署到新环境（不同房间/用户/设备位置）时选择。当前框架未实现，作为范式预埋。",
        pros=["跨域泛化能力强", "减少环境重新标注成本", "实际部署价值高"],
        cons=["实现复杂", "需源域和目标域数据", "当前框架未覆盖"],
    ),
    "lightweight": Paradigm(
        name="lightweight",
        category="轻量化模型",
        description="通过模型压缩（知识蒸馏、剪枝、量化）或高效架构（EfficientFi）降低模型大小和推理延迟，适合边缘设备部署。",
        applicable_models=["MLP", "LeNet"],
        best_for_datasets=["UT_HAR_data", "NTU-Fi_HAR"],
        resource_level="low",
        expected_accuracy_range="75%-90%",
        training_time="分钟级",
        key_papers=[
            "Wang et al., 'EfficientFi: Efficient WiFi Human Activity Recognition', 2023",
        ],
        config_template={
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "device": "cpu",
            "quantization": True,
        },
        selection_criteria="目标平台为边缘设备（树莓派、嵌入式设备）或对推理延迟有严格要求时选择。",
        pros=["模型小、推理快", "适合边缘部署", "能耗低"],
        cons=["精度有损失", "需额外压缩流程", "当前框架未实现蒸馏/剪枝"],
    ),
    "foundation_model": Paradigm(
        name="foundation_model",
        category="基础模型 / 预训练大模型",
        description="在大规模 CSI 数据上预训练通用感知模型，通过微调适配各种下游任务。代表 WiFi CSI 感知的前沿方向。",
        applicable_models=[],
        best_for_datasets=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        resource_level="extreme",
        expected_accuracy_range="90%-98%（预期）",
        training_time="天级（预训练）",
        key_papers=[
            "Wang et al., 'Understanding and Using the WiFi Sensing Channel', 2024",
            "Zeng et al., 'A Survey of WiFi Sensing: From Theory to Applications', 2024",
        ],
        config_template={
            "pretrain_epochs": 1000,
            "finetune_epochs": 100,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "device": "cuda",
        },
        selection_criteria="有大规模预训练数据和充足算力时选择。当前为前沿研究方向，框架未实现。",
        pros=["通用感知能力强", "少样本学习", "跨任务迁移"],
        cons=["预训练成本极高", "需大规模数据", "当前框架未覆盖"],
    ),
}


def list_paradigms(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出所有范式，可按类别过滤。"""
    if category:
        return [p.to_dict() for p in PARADIGM_REGISTRY.values() if p.category == category]
    return [p.to_dict() for p in PARADIGM_REGISTRY.values()]


def get_paradigm(name: str) -> Optional[Paradigm]:
    """查询单个范式详情。"""
    return PARADIGM_REGISTRY.get(name)


def _print_json(data: Any):
    """输出 JSON 到 stdout。"""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _cmd_probe(args):
    """探测硬件资源。"""
    # P1-a 修复：激活 lazy scenes，确保 available_models 返回完整模型列表
    # （与 _cmd_list_models / _cmd_recommend 对齐，其他命令都已调用）
    activate_lazy_scenes()
    report = ResourceProbe.probe()
    route_level = ResourceRouter.route(report)
    route_config = ResourceRouter.get_route_config(route_level)
    available_models = ResourceRouter.filter_models(route_level)
    # RFC-004 方案 D：同时输出 lightning_params（输出契约层），便于用户理解实际 Lightning 参数
    lightning_params = ResourceRouter.to_lightning_params(route_config)

    _print_json({
        "resource": report.to_dict(),
        "route_level": route_level,
        "route_config": route_config,  # 内部表示（调试用）
        "lightning_params": lightning_params,  # 输出契约层（accelerator/devices/precision）
        "available_models": available_models,
    })


def _cmd_list_models(args):
    """列出可用模型。"""
    activate_lazy_scenes()
    models = list_models(
        dataset=args.dataset,
        paradigm=args.paradigm,
        enabled_only=not args.all,
    )
    _print_json({"count": len(models), "models": models})


def _cmd_list_datasets(args):
    """列出可用数据集。"""
    activate_lazy_scenes()
    datasets = list_datasets()
    _print_json({"count": len(datasets), "datasets": datasets})


def _cmd_list_scenes(args):
    """列出已注册的场景容器。"""
    scenes = list_scenes()
    scene_list = []
    unavailable = []
    for name, meta in scenes.items():
        # list_scenes 在场景激活失败时附加 "_unavailable" key（List[str]），
        # 此处单独提取并跳过（不是 SceneMeta）
        if name == "_unavailable":
            unavailable = list(meta) if isinstance(meta, list) else []
            continue
        scene_list.append({
            "name": meta.name,
            "supported_tasks": meta.supported_tasks,
            "supported_models": meta.supported_models,
            "supported_datasets": meta.supported_datasets,
            "requires_custom_dataloader": meta.requires_custom_dataloader,
            # Phase 6.3：补全能力声明字段，供 Agent 程序化查询
            "supported_learning_modes": meta.supported_learning_modes,
            "input_shape_hint": meta.input_shape_hint,
        })
    _print_json({"count": len(scene_list), "scenes": scene_list,
                 "unavailable": unavailable})


def _cmd_paradigms(args):
    """列出 SOTA 范式。"""
    paradigms = list_paradigms(category=args.category)
    _print_json({"count": len(paradigms), "paradigms": paradigms})


def _has_factory(model_id: str, dataset: str) -> bool:
    """检查模型是否绑定了指定数据集的工厂。"""
    from .registry import resolve_factory
    try:
        resolve_factory(model_id, dataset)
        return True
    except KeyError:
        # 预期情况：模型未注册或无匹配工厂
        return False
    except Exception as e:
        # 意外情况：注册表内部错误
        _logger.warning("Unexpected error checking factory for '%s/%s': %s",
                        model_id, dataset, e, exc_info=True)
        return False


def _cmd_recommend(args):
    """根据资源推荐可用模型。"""
    activate_lazy_scenes()
    report = ResourceProbe.probe()
    route_level = ResourceRouter.route(report)
    available = ResourceRouter.filter_models(route_level)

    # 按数据集过滤（通过 resolve_factory 检测是否绑定了工厂）
    if args.dataset:
        available = [
            m for m in available
            if _has_factory(m, args.dataset)
        ]

    # 获取模型详情
    from .registry import MODEL_TABLE
    recommendations = []
    for model_id in available:
        info = MODEL_TABLE[model_id].copy()
        info["model_id"] = model_id
        if args.dataset:
            # 方案 B：epochs 完全动态，需 n_samples。从 dataset spec 获取。
            # 无 n_samples 时 get_default_epochs raise → 降级为 None，CLI 不崩溃。
            try:
                from .registry import get_default_epochs, get_dataset_spec, is_dataset_registered
                n_samples = None
                if is_dataset_registered(args.dataset):
                    n_samples = get_dataset_spec(args.dataset).n_samples
                info["default_epochs"] = get_default_epochs(
                    model_id, args.dataset, n_samples=n_samples)
            except (KeyError, ValueError) as e:
                _logger.warning(
                    f"get_default_epochs failed for ({model_id}, {args.dataset}): {e}; "
                    f"default_epochs set to None"
                )
                info["default_epochs"] = None
        recommendations.append(info)

    # 按优先级排序
    priority = args.priority or "balanced"
    if priority == "accuracy":
        recommendations.sort(key=lambda x: x.get("estimated_params_m", 0), reverse=True)
    elif priority == "speed":
        recommendations.sort(key=lambda x: x.get("estimated_params_m", 0))
    elif priority == "memory":
        recommendations.sort(key=lambda x: x.get("estimated_vram_mb", 0))

    _print_json({
        "resource": report.to_dict(),
        "route_level": route_level,
        "priority": priority,
        "count": len(recommendations),
        "recommendations": recommendations,
    })


def _build_unified_report(
    report: Dict[str, Any], config: ExperimentConfig, route_config: Dict[str, Any],
) -> Dict[str, Any]:
    """P3：将分散的 report dict 聚合为 PreflightReport 统一输出。

    在 _cmd_dry_run 的所有 return 路径调用，确保输出格式一致：
    - 保留原 report 中的分类字段（config_semantics / dynamic_validation 等）
    - 新增统一 checks 数组（所有 CheckResult 聚合）
    - 新增 summary（total/passed/warnings/errors 计数）
    - 新增 env_snapshot（环境快照，维度 8）
    """
    from .engine.runner.preflight import PreflightReport, build_env_snapshot, CheckResult
    from .engine.runner.resolver import experiment_config_to_dict

    preflight_rpt = PreflightReport(
        status=report.get("status", "ok"),
        env_snapshot=build_env_snapshot(
            route_config, experiment_config_to_dict(config),
        ),
        plan=report.get("plan", {}),
    )
    # 添加静态检查项（从 report["checks"] 转换）
    static_checks = [
        CheckResult(
            name=c["name"], ok=c["ok"], severity="info" if c["ok"] else "error",
            detail=c.get("detail"),
        )
        for c in report.get("checks", [])
    ]
    preflight_rpt.add_category("static", static_checks)
    # 添加 P1-P2 分类检查
    for cat_name in ("config_semantics", "dependency_contract", "reproducibility", "resource_contract", "training_contract", "data_contract"):
        cat_checks_raw = report.get(cat_name, [])
        if cat_checks_raw:
            cat_checks = [
                CheckResult(
                    name=c["name"], ok=c["ok"], severity=c.get("severity", "info"),
                    detail=c.get("detail"), error_code=c.get("error_code"),
                    remediation=c.get("remediation"),
                )
                for c in cat_checks_raw
            ]
            preflight_rpt.add_category(cat_name, cat_checks)
    # 添加动态校验（model_contract 归入 dynamic_validation）
    dyn_val = report.get("dynamic_validation", {})
    if dyn_val and "checks" in dyn_val:
        dyn_checks = [
            CheckResult(
                name=c["name"], ok=c["ok"], severity=c.get("severity", "info"),
                detail=c.get("detail"), error_code=c.get("error_code"),
                remediation=c.get("remediation"),
            )
            for c in dyn_val["checks"]
        ]
        preflight_rpt.add_category("model_contract", dyn_checks)

    # 输出统一报告（保留原 report 字段 + 新增 checks/summary/env_snapshot）
    unified = preflight_rpt.to_dict()
    # 保留 dynamic_validation 的 detail（to_dict 中 model_contract 转为 dynamic_validation）
    if dyn_val and dyn_val.get("detail"):
        unified["dynamic_validation"]["detail"] = dyn_val["detail"]
    # 保留 dynamic_validation 的 skipped/failed 状态及 error/error_code 字段
    # 修复 bug：旧逻辑只覆写 "skipped"，"failed" 状态被 to_dict 的 all([]) 错误设为 "passed"
    # 且 error/error_code 字段被 to_dict 丢弃
    if dyn_val and dyn_val.get("status") in ("skipped", "failed"):
        unified["dynamic_validation"] = dyn_val
    # 保留 blocked_reason
    if report.get("blocked_reason"):
        unified["blocked_reason"] = report["blocked_reason"]
    return unified


def _run_dynamic_validation(config: ExperimentConfig) -> dict:
    """P0-2 + P1：轻量动态校验（含模型契约校验）。

    取代旧 Pipeline.run(dry_run=True)（启动 Lightning Trainer + trainer.validate()），
    改为直接调用 scene API 构建模型 + 加载数据，做最小化前向校验。

    设计原则：
    - 不启动 Lightning Trainer（无 trainer.fit / trainer.validate 开销）
    - 不初始化 CUDA 上下文（在 CPU 上前向，避免 CUDA 副作用）
    - 不产生训练产物（无 checkpoint / 日志 / manifest）
    - 开销 < 2 秒

    P1 增强：调用 validate_model_contract 返回结构化 checks 数组，
    含 param_count_reasonable / forward_pass / output_shape_match / backward_pass。

    Returns:
        dict: {"status": "passed"/"failed", "checks": [...], "detail": ..., "error_code": ...}
    """
    import torch
    from .scenes import activate_lazy_scenes, get_scene
    from .engine.runner.preflight import validate_model_contract

    try:
        activate_lazy_scenes()
        scene = get_scene(config.scene.name)

        # 准备 scene_kwargs（与 stage_load 一致）
        scene_kwargs = {}
        if config.scene.params:
            # P5 P3-4 选项 B：入口校验 scene.params 已知键类型，捕获类型污染
            from .schemas import validate_scene_params
            validate_scene_params(config.scene.params)
            scene_kwargs["params"] = config.scene.params

        # 1. 获取 dataset_info（num_classes）
        scene_info = scene.get_dataset_info(config.scene.dataset, **scene_kwargs)
        num_classes = scene_info["num_classes"]

        # 2. 加载数据集（取 1 batch 样本）
        bundle = scene.load_dataset(
            config.scene.dataset, config.scene.data_root,
            learning_mode=config.scene.learning_mode,
            **scene_kwargs,
        )

        # 取 train_dataset 第一个样本
        # DatasetBundle 属性：train/test/val/unsupervised/supervised_finetune
        # supervised 模式用 train，self_supervised 模式用 supervised_finetune（有标签）
        train_ds = getattr(bundle, "train", None)
        if train_ds is None:
            train_ds = getattr(bundle, "supervised_finetune", None)
        if train_ds is None:
            train_ds = getattr(bundle, "test", None)
        if train_ds is None or len(train_ds) == 0:
            return {
                "status": "failed",
                "error": "train_dataset 为空或不存在（bundle.train/supervised_finetune/test 均为 None）",
                "error_code": "DATA_EMPTY",
                "checks": [],
            }

        n_samples = len(train_ds)
        sample = train_ds[0]
        if isinstance(sample, (list, tuple)):
            x_sample = sample[0]
            y_sample = sample[1] if len(sample) >= 2 else None
        elif isinstance(sample, dict):
            x_sample = sample.get("x") or sample.get("input") or list(sample.values())[0]
            y_sample = sample.get("y") or sample.get("label")
        else:
            x_sample = sample
            y_sample = None

        # 修复：应用 scene 的 transform 到 sample
        # 旧逻辑直接用 raw sample 做前向传播，未应用 transform（归一化/reshape/stride），
        # 导致 NTU-Fi_HAR/Widar 等需要 transform 的数据集前向传播失败（shape/dtype 不匹配）。
        # UT_HAR_data 不受影响（tensor_loader 直接返回正确 shape，无 transform）。
        try:
            tc = scene.get_transforms(config.scene.dataset, **scene_kwargs)
            if tc.train_transform is not None and y_sample is not None:
                x_sample, y_sample = tc.train_transform(x_sample, y_sample)
        except Exception as te:
            _logger.warning(
                "dynamic_validation: transform 应用失败，使用 raw sample: %s", te
            )

        # P1-1 修复：遍历 train_ds 取所有标签，计算 class_distribution 和 imbalance_ratio
        # 仅取 y（item[1]），不加载 x，开销小；供 _cmd_dry_run 构造完整 DataProfile
        class_distribution = {}
        imbalance_ratio = None
        if config.scene.learning_mode != "self_supervised":
            try:
                import numpy as _np
                labels = []
                for idx in range(n_samples):
                    try:
                        item = train_ds[idx]
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            labels.append(item[1])
                    except Exception:
                        pass
                if labels:
                    y_arr = _np.array(labels)
                    unique, counts = _np.unique(y_arr, return_counts=True)
                    class_distribution = {str(int(u)): int(c) for u, c in zip(unique, counts)}
                    imbalance_ratio = float(max(counts)) / max(float(min(counts)), 1.0)
            except Exception:
                pass

        # 扩展到 batch_size（模拟真实 batch 形状）
        batch_size = min(config.trainer.batch_size, 4)  # dry-run 用小 batch 加速
        x = x_sample.unsqueeze(0).repeat(batch_size, *[1] * x_sample.dim())

        # 3. 构建模型
        model = scene.build_model_for_dataset(
            config.scene.model_id, config.scene.dataset, num_classes,
            learning_mode=config.scene.learning_mode,
            **scene_kwargs,
        )

        # P1-2：模型契约校验（参数量 + 前向 + 输出形状 + backward）
        model_checks = validate_model_contract(model, num_classes, x)
        checks = [c.to_dict() for c in model_checks]

        # 判定整体状态：有 error 级失败则 failed
        has_error = any(
            not c["ok"] and c["severity"] == "error"
            for c in checks
        )

        n_params = sum(p.numel() for p in model.parameters())

        return {
            "status": "failed" if has_error else "passed",
            "checks": checks,
            "detail": {
                "n_samples": n_samples,
                "n_params": n_params,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "class_distribution": class_distribution,
                "imbalance_ratio": imbalance_ratio,
            },
            "error_code": (
                next((c["error_code"] for c in checks if not c["ok"] and c["error_code"]), None)
            ),
        }

    except Exception as e:
        _logger.error(f"dynamic validation failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "DYNAMIC_VALIDATION_ERROR",
            "checks": [],
        }


def _cmd_dry_run(config: ExperimentConfig, static_only: bool = False) -> dict:
    """
    Phase 6.3：预检模式（--dry-run）。

    不实际执行训练，仅完成启动前检查并输出报告：
    1. 配置 schema 校验
    2. 场景/数据集/模型注册校验
    3. 硬件资源探测 + 路由
    4. 启动前预检（数据存在性、显存、磁盘空间）
    5. 输出"将要执行"的训练计划摘要

    P0-1（2026-07-12）：static_only 默认改为 False（默认启用动态校验）。
    旧默认 True 导致 --dry-run 总跳过动态校验（dynamic_validation: skipped），
    被 Agent 评分扣分。现在 --dry-run 默认启用动态前向校验，
    用户可通过 --static-only 显式跳过。

    P0-2：动态校验使用轻量前向（forward + backward 1 step），
    不启动 Lightning Trainer，无训练副作用，开销 < 2 秒。

    Args:
        config: ExperimentConfig 实例
        static_only: True 时仅执行静态校验，跳过动态前向校验（默认 False）

    Returns:
        预检报告 dict（含 status: ok/blocked + 各检查项结果 + dynamic_validation）
    """
    from .engine.runner.resolver import experiment_config_to_dict
    from .engine.runner.preflight import preflight_check as _preflight_check
    from .scenes import get_scene, has_scene

    report: Dict[str, Any] = {"status": "ok", "checks": []}

    def _check(name: str, ok: bool, detail: Any = None):
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            report["status"] = "blocked"

    # 1. 配置校验
    try:
        config.validate()
        _check("config_validation", True)
    except ValueError as e:
        _check("config_validation", False, str(e))
        return report

    # 2. 场景注册校验
    if not has_scene(config.scene.name):
        _check("scene_registered", False,
               f"Scene '{config.scene.name}' not registered")
        return report
    _check("scene_registered", True)

    scene = get_scene(config.scene.name)
    meta = scene.meta()

    # 3. 数据集/模型支持校验
    # Phase 8.1：CustomContainer 的 supported_datasets 动态由 manifest 决定
    is_custom = (config.scene.name == "custom")
    if not is_custom:
        if config.scene.dataset not in meta.supported_datasets:
            _check("dataset_supported", False,
                   f"'{config.scene.dataset}' not in {meta.supported_datasets}")
        else:
            _check("dataset_supported", True)
    else:
        # CustomContainer：尝试加载 manifest 验证 dataset 名称匹配
        try:
            from .scenes.custom.container import _load_manifest_cached
            manifest_path = config.scene.params.get("manifest_path") if config.scene.params else None
            if manifest_path is None:
                _check("dataset_supported", False,
                       "CustomContainer 需要 scene.params.manifest_path")
            else:
                manifest = _load_manifest_cached(manifest_path)
                if config.scene.dataset != manifest.name:
                    _check("dataset_supported", False,
                           f"dataset '{config.scene.dataset}' != manifest.name '{manifest.name}'")
                else:
                    _check("dataset_supported", True, {"manifest": manifest.name})
        except Exception as e:
            _check("dataset_supported", False, f"manifest 加载失败: {e}")

    if config.scene.model_id not in meta.supported_models:
        _check("model_supported", False,
               f"'{config.scene.model_id}' not in {meta.supported_models}")
    else:
        _check("model_supported", True)

    # 4. 学习模式支持校验（Phase 6.2 字段）
    if config.scene.learning_mode not in meta.supported_learning_modes:
        _check("learning_mode_supported", False,
               f"'{config.scene.learning_mode}' not in {meta.supported_learning_modes}")
    else:
        _check("learning_mode_supported", True)

    # 5. 硬件资源探测 + 路由
    try:
        resource = ResourceProbe.probe()
        route_level = ResourceRouter.route(resource)
        route_config = ResourceRouter.get_route_config(route_level)
        _check("resource_probe", True, {
            "has_cuda": resource.has_cuda,
            "gpu_name": resource.gpu_name,
            "gpu_free_vram_mb": resource.gpu_free_vram_mb,
            "route_level": route_level,
        })
    except Exception as e:
        _check("resource_probe", False, str(e))
        return report

    # 6. 启动前预检（数据存在性、显存、磁盘）
    try:
        config_dict = experiment_config_to_dict(config)
        model_info = scene.get_model_info(config.scene.model_id)
        _preflight_check(
            config_dict, model_info, resource, config.scene.dataset,
            scene_name=config.scene.name,
            scene_params=config.scene.params,
        )
        _check("preflight", True, {
            "data_root": config_dict.get("data_root"),
            "estimated_vram_mb": model_info.get("estimated_vram_mb"),
        })
    except FileNotFoundError as e:
        _check("preflight", False, f"DATA_NOT_FOUND: {e}")
    except RuntimeError as e:
        msg = str(e).lower()
        if "vram" in msg:
            _check("preflight", False, f"PREFLIGHT_ERROR (VRAM): {e}")
        elif "disk" in msg:
            _check("preflight", False, f"PREFLIGHT_ERROR (Disk): {e}")
        else:
            _check("preflight", False, str(e))
    except Exception as e:
        _check("preflight", False, str(e))

    # 7. 训练计划摘要
    # RFC-004 方案 D：输出契约分离 — 用 to_lightning_params() 转换 route_config，
    # 不直接暴露内部表示（route_config 无 accelerator 字段，是 to_lightning_params 的派生）
    lightning_params = ResourceRouter.to_lightning_params(route_config)
    report["plan"] = {
        "scene": config.scene.name,
        "dataset": config.scene.dataset,
        "model_id": config.scene.model_id,
        "learning_mode": config.scene.learning_mode,
        "epochs": config.trainer.epochs,
        "batch_size": config.trainer.batch_size,
        "learning_rate": config.trainer.learning_rate,
        "optimizer": config.trainer.optimizer,
        "device": route_config.get("device"),  # 原始设备（内部表示，供调试）
        # lightning_params 是输出契约层（accelerator/devices/precision 派生自 device）
        "accelerator": lightning_params["accelerator"],
        "devices": lightning_params["devices"],
        "precision": lightning_params["precision"],
    }

    # P1-1：配置语义校验（跨字段逻辑约束）
    from .engine.runner.preflight import (
        validate_config_semantics, validate_dependency_contract,
        validate_resource_contract, validate_reproducibility,
        validate_data_contract,
    )
    config_checks = validate_config_semantics(config.trainer)
    report["config_semantics"] = [c.to_dict() for c in config_checks]
    # 有 error 级失败则更新 report status
    config_errors = [c for c in config_checks if not c.ok and c.severity == "error"]
    if config_errors:
        report["status"] = "blocked"
        report["blocked_reason"] = f"config_semantics: {[c.name for c in config_errors]}"

    # P2-2：依赖契约校验（logger/export/deterministic 依赖）
    export_formats = getattr(config, "export_formats", None) or []
    dep_checks = validate_dependency_contract(config, export_formats)
    report["dependency_contract"] = [c.to_dict() for c in dep_checks]
    dep_errors = [c for c in dep_checks if not c.ok and c.severity == "error"]
    if dep_errors:
        report["status"] = "blocked"
        report["blocked_reason"] = f"dependency_contract: {[c.name for c in dep_errors]}"

    # P2-4：可复现性检查（seed/deterministic/版本记录）
    # 注意：传入 resource（ResourceReport 实例），不是 report（dict）
    # getattr(report, "has_cuda", False) 对 dict 返回 False，会导致 vram_sufficient
    # 错误显示"CPU 模式" + num_workers_reasonable 使用 cpu_count 默认值 1
    repro_checks = validate_reproducibility(config, resource)
    report["reproducibility"] = [c.to_dict() for c in repro_checks]

    # P2-3：资源契约校验（显存/num_workers/训练规模估算）
    # dry-run 模式下 vram_probe_result 为 None（probe stage 未运行）
    resource_checks = validate_resource_contract(
        resource, route_config,
        vram_probe_result=None,  # dry-run 不运行 probe
        n_samples=0, batch_size=config.trainer.batch_size,
    )
    report["resource_contract"] = [c.to_dict() for c in resource_checks]

    # P0-2: 轻量动态校验（forward + backward 1 step，不启动 Lightning Trainer）
    # 静态校验未通过或显式 static_only 模式时跳过；否则执行轻量前向校验。
    if static_only:
        report["dynamic_validation"] = {
            "status": "skipped",
            "reason": "static_only mode",
        }
        return _build_unified_report(report, config, route_config)

    if report["status"] != "ok":
        report["dynamic_validation"] = {
            "status": "skipped",
            "reason": "static checks blocked",
        }
        return _build_unified_report(report, config, route_config)

    _logger.info("dry-run: lightweight dynamic validation (forward + backward 1 step)")
    # P0-2（2026-07-12）：取代旧 Pipeline.run(dry_run=True)（启动 Lightning Trainer +
    # trainer.validate()，开销 ~10s + 产生临时产物）。改为 _run_dynamic_validation：
    # 直接调用 scene API 构建模型 + 加载数据，CPU 上做 forward + backward 1 step，
    # 不启动 Trainer，不初始化 CUDA，不产生训练产物，开销 < 2s。
    # P1：_run_dynamic_validation 内部调用 validate_model_contract 返回结构化 checks。
    dyn_result = _run_dynamic_validation(config)
    report["dynamic_validation"] = dyn_result

    # 修复 bug：dynamic_validation 失败时更新顶层 status
    # 旧逻辑：dynamic_validation 失败时不更新 report["status"]，仅 config_semantics/
    # dependency_contract 失败时才设为 "blocked"，导致顶层 status 仍为 "ok" 误导用户
    if dyn_result.get("status") == "failed":
        report["status"] = "blocked"
        report["blocked_reason"] = (
            f"dynamic_validation: {dyn_result.get('error_code', 'unknown')} - "
            f"{dyn_result.get('error', 'unknown error')}"
        )

    # P1-3：训练契约校验（loss/metrics 与 task_type 一致性）
    # 仅在动态校验成功（获取到 num_classes）后执行
    if dyn_result.get("status") == "passed" and dyn_result.get("detail"):
        from .engine.runner.preflight import validate_training_contract
        from .core.task import TaskSpec
        num_classes = dyn_result["detail"].get("num_classes", 0)
        n_samples = dyn_result["detail"].get("n_samples", 0)

        # P2-2 修复：用真实 n_samples 重跑 config_semantics，
        # 让 batch_size_within_dataset 检查真正生效（首次调用 n_samples=0 导致检查被禁用）
        config_checks = validate_config_semantics(config.trainer, n_samples=n_samples)
        report["config_semantics"] = [c.to_dict() for c in config_checks]
        config_errors = [c for c in config_checks if not c.ok and c.severity == "error"]
        if config_errors:
            report["status"] = "blocked"
            report["blocked_reason"] = f"config_semantics: {[c.name for c in config_errors]}"

        task_spec = TaskSpec.classification(num_classes=num_classes)
        # 从 config 提取训练参数
        t = config.trainer
        metrics = config.scene.params.get("metrics", ["accuracy"]) if config.scene.params else ["accuracy"]
        loss_name = "cross_entropy"  # classification 默认
        training_checks = validate_training_contract(
            task_spec, loss_name, metrics,
            t.optimizer, t.scheduler, t.epochs, t.early_stopping,
        )
        report["training_contract"] = [c.to_dict() for c in training_checks]

        # P2-1：数据契约校验（基于 DataProfile）
        # P1-1 修复：从 dynamic_validation detail 读取 class_distribution/imbalance_ratio
        # （_run_dynamic_validation 已遍历 train_ds 计算标签分布）
        from .core.profiler import DataProfile
        dyn_detail = dyn_result.get("detail", {})
        profile = DataProfile(
            n_samples=n_samples,
            modality="csi" if config.scene.name == "wifi_csi" else "unknown",
            class_distribution=dyn_detail.get("class_distribution", {}),
            imbalance_ratio=dyn_detail.get("imbalance_ratio"),
            missing_rate=0.0,
        )
        data_checks = validate_data_contract(
            profile, task_spec, config.scene.learning_mode,
        )
        report["data_contract"] = [c.to_dict() for c in data_checks]

    # P3：PreflightReport 统一聚合 + 环境快照（维度 8）
    return _build_unified_report(report, config, route_config)


def _cmd_export(args):
    """
    Phase 7.1：独立模型导出命令。

    基于训练输出的 metadata.json + model.pth 导出多种格式。
    """
    from .export import export_from_metadata, SUPPORTED_FORMATS

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        # P5 P3-13：错误输出到 stderr；P5 P2-13：code 映射到标准 ERROR_CODES
        print(json.dumps({
            "error": f"Metadata file not found: {args.metadata}",
            "code": "DATA_NOT_FOUND",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(json.dumps({
            "error": f"Checkpoint file not found: {args.checkpoint}",
            "code": "CHECKPOINT_ERROR",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            print(json.dumps({
                "error": f"Unsupported format '{fmt}'. Supported: {SUPPORTED_FORMATS}",
                "code": "CONFIG_VALIDATION_ERROR",
            }, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)

    # Phase 12.2：CLI 输出激活参数
    output_activation = getattr(args, "output_activation", None)
    if output_activation and output_activation not in (
        "none", "softmax", "sigmoid", "tanh", "relu"
    ):
        from .export import list_supported_activations
        print(json.dumps({
            "error": f"Unsupported output_activation '{output_activation}'. "
                     f"Supported: {list_supported_activations()}",
            "code": "CONFIG_VALIDATION_ERROR",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    try:
        result = export_from_metadata(
            metadata_path=str(metadata_path),
            checkpoint_path=str(checkpoint_path),
            output_dir=args.output_dir,
            formats=formats,
            validate=args.validate,
            output_activation=output_activation,
        )
        _print_json(result.to_dict())
        if result.errors:
            # P4-3：导出错误（如 onnx 包缺失）输出顶层 error JSON 到 stderr，
            # 与其他错误路径（metadata/checkpoint/format 不存在）的输出格式一致。
            # 旧代码仅 exit(1) 无额外输出，自动化测试无法区分"成功但有部分错误"
            # 与"完全失败"，且错误埋在 result.errors 子字段中不易发现。
            error_msg = "; ".join(f"{k}: {v}" for k, v in result.errors.items())
            print(json.dumps({
                "error": f"Export completed with errors: {error_msg}",
                "code": "SAVE_ERROR",
                "format_errors": result.errors,
            }, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "code": "UNKNOWN_ERROR",
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


def _cmd_predict(args):
    """
    Phase 8.3：批量推理命令。

    从训练产物（model.pth + metadata.json）加载模型，对样本列表执行推理，
    输出 JSON 格式的预测结果。

    样本输入格式（--samples 指向的 JSON 文件）：
        [
            {"path": "data/sample1.npy"},
            {"path": "data/sample2.npy"}
        ]

    输出格式（--output 指向的 JSON 文件，或 stdout）：
        [
            {"path": "...", "label": 3, "label_name": "walk", "confidence": 0.92},
            ...
        ]
    """
    from .inference import predict

    # 校验 model + metadata
    model_path = Path(args.model)
    if not model_path.exists():
        print(json.dumps({
            "error": f"Model file not found: {args.model}",
            "code": "MODEL_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(json.dumps({
            "error": f"Metadata file not found: {args.metadata}",
            "code": "METADATA_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    # 加载样本列表
    samples_path = Path(args.samples)
    if not samples_path.exists():
        print(json.dumps({
            "error": f"Samples file not found: {args.samples}",
            "code": "SAMPLES_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    try:
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(json.dumps({
            "error": f"Invalid JSON in samples file: {e}",
            "code": "INVALID_SAMPLES_JSON",
        }, ensure_ascii=False))
        sys.exit(1)

    if not isinstance(samples, list):
        print(json.dumps({
            "error": "Samples file must contain a JSON array",
            "code": "INVALID_SAMPLES_FORMAT",
        }, ensure_ascii=False))
        sys.exit(1)

    # 执行推理
    try:
        results = predict(
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            samples=samples,
            output_format="dict",
            include_logits=args.include_logits,
            device=args.device,
        )
    except Exception as e:
        print(json.dumps({
            "error": f"Inference failed: {e}",
            "code": type(e).__name__,
        }, ensure_ascii=False))
        sys.exit(1)

    # 输出结果
    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(json.dumps({
            "status": "success",
            "num_samples": len(results),
            "output_file": args.output,
        }, ensure_ascii=False))
    else:
        print(output_json)


def _cmd_experiment(args):
    """
    执行声明式实验（新入口）。

    流程：
    1. 加载 YAML 配置文件 → dict
    2. ExperimentConfig.from_dict(dict) → 校验 schema
    3. 若 --hpo 启用 → run_hpo(config)
       否则 → run_experiment(config)
    4. 输出结构化 JSON

    YAML 配置示例：
        scene:
          name: wifi_csi
          dataset: UT_HAR_data
          model_id: ResNet18
        input_features:
          - name: csi
            type: csi
            shape: [270, 3]
        output_features:
          - name: action
            type: category
            num_classes: 7
        trainer:
          epochs: 50
          batch_size: 64
    """
    # 配置日志级别
    setup_logging(level=args.log_level, log_file=args.log_file)

    # 加载 YAML 配置
    if not args.config:
        print(json.dumps({
            "error": "--config is required for experiment command",
            "code": "MISSING_CONFIG",
        }, ensure_ascii=False))
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(json.dumps({
            "error": f"Config file not found: {args.config}",
            "code": "CONFIG_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        print(json.dumps({
            "error": "Config file must contain a YAML mapping at top level",
            "code": "INVALID_CONFIG_FORMAT",
        }, ensure_ascii=False))
        sys.exit(1)

    # 解析为 ExperimentConfig（暂不 validate，CLI 覆盖 + env fallback 后再校验）
    try:
        config = ExperimentConfig.from_dict(config_dict)
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "code": "CONFIG_VALIDATION_ERROR",
        }, ensure_ascii=False))
        sys.exit(1)

    # CLI 覆盖（可选）
    if args.scene:
        config.scene.name = args.scene
    if args.dataset:
        config.scene.dataset = args.dataset
    if args.model:
        config.scene.model_id = args.model
    if args.data_root:
        config.scene.data_root = args.data_root
    # env fallback：YAML/CLI 都未提供 data_root 时，从 SENSEFRAME_DATA_ROOT 读
    if not config.scene.data_root:
        import os as _os
        _env_root = _os.environ.get("SENSEFRAME_DATA_ROOT")
        if _env_root:
            config.scene.data_root = _env_root
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.trainer.learning_rate = args.learning_rate
    if args.output_dir:
        config.output_dir = args.output_dir

    # CLI 覆盖 + env fallback 完成后，统一 validate
    try:
        config.validate()
    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "code": "CONFIG_VALIDATION_ERROR",
        }, ensure_ascii=False))
        sys.exit(1)

    # s2: export_formats 已纳入 ExperimentConfig schema，直接赋值
    if args.export_formats:
        from .export import SUPPORTED_FORMATS
        formats = [f.strip() for f in args.export_formats.split(",") if f.strip()]
        for fmt in formats:
            if fmt not in SUPPORTED_FORMATS:
                print(json.dumps({
                    "error": f"Unsupported export format '{fmt}'. Supported: {SUPPORTED_FORMATS}",
                    "code": "UNSUPPORTED_FORMAT",
                }, ensure_ascii=False))
                sys.exit(1)
        config.export_formats = formats

    # Phase 6.3：--dry-run 预检模式（不实际训练，仅输出检查报告）
    if args.dry_run:
        # P0-1（2026-07-12）：--dry-run 默认启用动态前向校验（static_only=False）。
        # 旧逻辑 --dry-run 默认纯静态（dynamic_validation: skipped）被 Agent 扣分。
        # 现改为：
        #   --dry-run              → static_only=False（默认启用动态校验）
        #   --dry-run --static-only → static_only=True（跳过动态校验）
        static_only = getattr(args, "static_only", False)
        report = _cmd_dry_run(config, static_only=static_only)
        _print_json(report)
        # P2-3：动态校验失败也视为 dry-run 失败（非零退出码）
        dyn_status = report.get("dynamic_validation", {}).get("status", "")
        if report["status"] != "ok" or dyn_status == "failed":
            sys.exit(1)
        sys.exit(0)

    # HPO 模式
    if args.hpo:
        config.hpo.enabled = True
        if args.hpo_trials is not None:
            config.hpo.n_trials = args.hpo_trials
        if args.hpo_metric:
            config.hpo.metric = args.hpo_metric
        if args.hpo_direction:
            config.hpo.direction = args.hpo_direction
        # HPO 前重新校验
        config.validate()
        result = run_hpo(config)
        _print_json(result.to_dict())
    else:
        # 单次实验
        # Phase 7.2：自愈重试
        use_retry = args.retry or getattr(config, "retry", False)
        if args.no_retry:
            use_retry = False

        if use_retry:
            from .retry import run_experiment_with_retry
            retry_result = run_experiment_with_retry(
                config=config,
                run_fn=run_experiment,
            )
            output = retry_result.final_output
        else:
            output = run_experiment(config)

        _print_json(output.to_dict())
        if output.status == "error":
            sys.exit(1)


def _cmd_exploration(args):
    """RFC-002 阶段 W：探索状态管理子命令。

    P3: 路径自动发现——优先显式 --exploration-file，否则扫描 --output-dir 下最新的 exploration.json。
    """
    from pathlib import Path
    from .exploration import ExplorationTracker, SearchSpaceMap

    # P3: 路径解析
    explicit_file = getattr(args, 'exploration_file', None)
    if explicit_file:
        # 显式指定路径
        file_path = Path(explicit_file)
        if not file_path.exists():
            print(f"No exploration history found at {file_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # 自动发现：扫描 output_dir 下最新的 exploration.json
        output_dir = Path(getattr(args, 'output_dir', '.'))
        candidates = sorted(
            output_dir.rglob("exploration.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"No exploration.json found under {output_dir}", file=sys.stderr)
            sys.exit(1)
        file_path = candidates[0]
        print(f"Auto-discovered: {file_path}", file=sys.stderr)

    tracker = ExplorationTracker.load(file_path)
    action = args.explore_action

    if action == "list":
        trials = tracker.list_trials(status=args.status)
        _print_json({"count": len(trials), "trials": trials})
    elif action == "recommend":
        recs = tracker.recommend_next(task_type=args.task_type, top_k=args.top_k)
        _print_json({"count": len(recs), "recommendations": recs})
    elif action == "coverage":
        _print_json(tracker.coverage())
    elif action == "map":
        space_map = SearchSpaceMap(tracker)
        overview = space_map.overview(task_type=args.task_type, dataset=args.dataset)
        _print_json(overview)
    elif action == "dashboard":
        from .observability import ExplorationDashboard
        dashboard = ExplorationDashboard(tracker)
        # dashboard 输出文本（非 JSON）
        print(dashboard.render(format=args.format))


def _cmd_monitor(args):
    """P2: 训练实时监控。

    读取 <output_dir>/training_log.jsonl（IncrementalLogWriter 输出），
    解析为 TrainingMonitor 并渲染实时指标曲线。
    """
    from pathlib import Path
    import json as _json
    from .observability import TrainingMonitor

    output_dir = Path(args.output_dir)
    log_path = output_dir / "training_log.jsonl"
    if not log_path.exists():
        print(f"No training log found at {log_path}", file=sys.stderr)
        sys.exit(1)

    monitor = TrainingMonitor()
    for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            entry = _json.loads(line)
            monitor.on_epoch_end(entry)
        except _json.JSONDecodeError:
            continue

    print(monitor.render_text())


def _cmd_serve(args):
    """P3: 启动推理服务。

    加载训练输出目录中的模型（model.onnx 或 model.pth + metadata.json），
    通过 FastAPI 暴露 /predict /predict/batch /health /info HTTP 端点。
    """
    from .serving import InferenceServer

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(json.dumps({
            "error": f"Output directory not found: {args.output_dir}",
            "code": "OUTPUT_DIR_NOT_FOUND",
        }, ensure_ascii=False))
        sys.exit(1)

    server = InferenceServer(args.output_dir, device=args.device)
    server.start(host=args.host, port=args.port)


def _cmd_skills(args):
    """RFC-002 阶段 W：技能库管理子命令。"""
    from .skills import list_skills, search_skills, load_skill, get_skill_library

    action = args.skills_action

    if action == "list":
        names = list_skills()
        _print_json({"count": len(names), "skills": names})
    elif action == "search":
        results = search_skills(args.query, top_k=args.top_k)
        output = [
            {
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
                "version": s.version,
            }
            for s in results
        ]
        _print_json({"count": len(output), "results": output})
    elif action == "show":
        skill = load_skill(args.name)
        if skill is None:
            print(json.dumps({
                "error": f"Skill '{args.name}' not found",
                "code": "SKILL_NOT_FOUND",
            }, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
        _print_json({
            "name": skill.name,
            "description": skill.description,
            "tags": skill.tags,
            "version": skill.version,
            "validated": skill.validated,
            "depends_on": skill.depends_on,
            "code": skill.code[:500],
        })
    elif action == "remove":
        try:
            success = get_skill_library().remove(args.name, force=args.force)
        except ValueError as e:
            # 依赖错误：含依赖错误信息
            print(json.dumps({
                "error": str(e),
                "code": "SKILL_HAS_DEPENDENTS",
                "name": args.name,
            }, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)
        if success:
            _print_json({"status": "success", "name": args.name, "force": args.force})
        else:
            print(json.dumps({
                "error": f"Skill '{args.name}' not found",
                "code": "SKILL_NOT_FOUND",
            }, ensure_ascii=False), file=sys.stderr)
            sys.exit(1)


def _cmd_catalog(args):
    """RFC-002 阶段 W：技术目录查询子命令。"""
    import importlib
    from .scenes import list_scenes, has_scene

    action = args.catalog_action

    if action == "list":
        scene_filter = args.scene
        if scene_filter is not None and not has_scene(scene_filter):
            print(f"Scene '{scene_filter}' not registered", file=sys.stderr)
            sys.exit(1)
        scenes_to_query = [scene_filter] if scene_filter else [
            k for k in list_scenes().keys() if k != "_unavailable"
        ]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        total = 0
        for scene_name in scenes_to_query:
            # 直接导入场景的 catalog 模块（兼容延迟加载场景如 wifi_csi）
            try:
                mod = importlib.import_module(
                    f".scenes.{scene_name}.catalog", package="senseframe"
                )
                catalog = getattr(mod, "CATALOG", None)
            except (ImportError, AttributeError):
                catalog = None
            if not catalog:
                continue
            for entry in catalog:
                cat = entry.get("category", "other")
                grouped.setdefault(cat, []).append({
                    "name": entry.get("name"),
                    "description": entry.get("description", ""),
                    "applicable": entry.get("applicable", []),
                    "implemented": entry.get("implemented", False),
                    "params": entry.get("params", {}),
                    "scene": scene_name,
                })
                total += 1

        if total == 0:
            print("No catalog available", file=sys.stderr)
            sys.exit(1)
        _print_json({"count": total, "categories": grouped})
    elif action in ("pipeline", "augment"):
        scene_name = args.scene
        dataset = args.dataset
        if not has_scene(scene_name):
            print(f"Scene '{scene_name}' not registered", file=sys.stderr)
            sys.exit(1)
        try:
            mod = importlib.import_module(
                f".scenes.{scene_name}.catalog", package="senseframe"
            )
            func_name = "suggest_pipeline" if action == "pipeline" else "suggest_augment"
            func = getattr(mod, func_name)
        except (ImportError, AttributeError):
            print(f"No catalog available for scene '{scene_name}'", file=sys.stderr)
            sys.exit(1)
        result = func(dataset)
        _print_json({"dataset": dataset, "scene": scene_name, action: result})


def _cmd_create_scene(args):
    """s3：创建场景包脚手架。

    从 senseframe/scenes/_template/ 复制模板到目标目录，并替换占位符：
        {SCENE_NAME}       → scene_name（snake_case，如 my_scene）
        {SCENE_NAME_CLASS} → 类名（如 MyScene）
    """
    import shutil

    scene_name = args.scene_name
    # M1 修复：scene_name 用于构造路径，必须校验防路径穿越
    # 拒绝含 / \ .. 的输入，限制为 [a-z][a-z0-9_]* 标识符
    from .common.path_safe import sanitize_path_component
    try:
        scene_name = sanitize_path_component(scene_name)
    except ValueError as e:
        print(f"Invalid scene name: {e}", file=sys.stderr)
        sys.exit(1)
    # snake_case → PascalCase：my_scene -> MyScene
    class_name = "".join(p.capitalize() for p in scene_name.split("_"))

    # 模板目录
    template_dir = Path(__file__).parent / "scenes" / "_template"
    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}", file=sys.stderr)
        sys.exit(1)

    # 目标目录（默认 scenes/<name>，可通过 --output 指定父目录）
    if args.output:
        target_dir = Path(args.output) / scene_name
    else:
        target_dir = Path("scenes") / scene_name
    if target_dir.exists():
        print(f"Scene directory already exists: {target_dir}", file=sys.stderr)
        sys.exit(1)

    # 复制模板
    shutil.copytree(template_dir, target_dir)

    # 替换占位符：必须先替换长占位符 {SCENE_NAME_CLASS}，否则
    # {SCENE_NAME} 会误伤 {SCENE_NAME_CLASS}（后者包含前者作为前缀）
    replacements = [
        ("{SCENE_NAME_CLASS}", class_name),
        ("{SCENE_NAME}", scene_name),
    ]
    for py_file in target_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for old, new in replacements:
            content = content.replace(old, new)
        py_file.write_text(content, encoding="utf-8")

    print(f"Scene '{scene_name}' created at {target_dir}")
    print(f"  Class: {class_name}Container")
    print(f"  Next steps:")
    print(f"    1. Implement load_dataset / build_model_for_dataset in {target_dir}/container.py")
    print(f"    2. Add transforms to {target_dir}/transforms.py")
    print(f"    3. Add catalog entries to {target_dir}/catalog.py")
    print(f"    4. Register scene in senseframe/scenes/__init__.py")


def main():
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        prog="senseframe",
        description="SenseFrame: WiFi CSI 动作识别训练框架",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # probe
    subparsers.add_parser("probe", help="探测硬件资源")

    # list-models
    p_models = subparsers.add_parser("list-models", help="列出可用模型")
    p_models.add_argument("--dataset", type=str, help="按数据集过滤")
    p_models.add_argument("--paradigm", type=str, help="按范式过滤")
    p_models.add_argument("--all", action="store_true", help="包含未启用的模型")

    # list-datasets
    subparsers.add_parser("list-datasets", help="列出可用数据集")

    # list-scenes
    subparsers.add_parser("list-scenes", help="列出已注册的场景容器")

    # paradigms
    p_para = subparsers.add_parser("paradigms", help="列出 SOTA 范式")
    p_para.add_argument("--category", type=str, help="按类别过滤")

    # recommend
    p_rec = subparsers.add_parser("recommend", help="根据资源推荐可用模型")
    p_rec.add_argument("--dataset", type=str, required=True, help="目标数据集")
    p_rec.add_argument("--priority", choices=["accuracy", "speed", "memory", "balanced"],
                       default="balanced", help="推荐优先级")

    # Phase 7.1：export 独立导出命令
    p_export = subparsers.add_parser(
        "export",
        help="基于 metadata + checkpoint 导出多格式模型",
    )
    p_export.add_argument("--metadata", type=str, required=True,
                          help="训练输出的 metadata.json 路径")
    p_export.add_argument("--checkpoint", type=str, required=True,
                          help="模型权重路径 (.pth)")
    p_export.add_argument("--formats", type=str, default="onnx,torchscript,state_dict",
                          help="导出格式，逗号分隔（默认 onnx,torchscript,state_dict）")
    p_export.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "exports"),
                          help="导出目录（默认 <project_root>/exports/）")
    p_export.add_argument("--validate", action="store_true",
                          help="Phase 8.4：导出后验证精度对齐（需 onnxruntime 验证 ONNX）")
    # Phase 12.2：输出激活
    p_export.add_argument(
        "--output-activation", type=str, default=None,
        choices=[None, "none", "softmax", "sigmoid", "tanh", "relu"],
        help="Phase 12.2：输出激活包装（默认从 metadata.task_spec 推断）",
    )

    # Phase 8.3：predict 批量推理命令
    p_predict = subparsers.add_parser(
        "predict",
        help="基于训练产物执行批量推理，输出 JSON 预测结果",
    )
    p_predict.add_argument("--model", type=str, required=True,
                           help="模型权重路径 (.pth)")
    p_predict.add_argument("--metadata", type=str, required=True,
                           help="训练输出的 metadata.json 路径")
    p_predict.add_argument("--samples", type=str, required=True,
                           help="样本列表 JSON 文件路径（[{'path': '...'}, ...]）")
    p_predict.add_argument("--output", type=str, default=None,
                           help="输出 JSON 文件路径（默认 stdout）")
    p_predict.add_argument("--device", type=str, default="cpu",
                           choices=["cpu", "cuda"], help="推理设备")
    p_predict.add_argument("--include-logits", action="store_true",
                           help="在结果中包含原始 logits")

    # experiment（声明式 YAML 配置入口）
    p_exp = subparsers.add_parser(
        "experiment",
        help="执行声明式实验（YAML 配置 → ExperimentConfig）",
    )
    p_exp.add_argument("--config", type=str, required=True,
                       help="YAML 配置文件路径")
    # CLI 覆盖（可选）
    p_exp.add_argument("--scene", type=str, help="覆盖 scene.name")
    p_exp.add_argument("--dataset", type=str, help="覆盖 scene.dataset")
    p_exp.add_argument("--model", type=str, help="覆盖 scene.model_id")
    p_exp.add_argument("--data-root", type=str, default=None,
                       help="覆盖 scene.data_root（数据根目录，必填：YAML/CLI/env 三选一）")
    p_exp.add_argument("--epochs", type=int, help="覆盖 trainer.epochs")
    p_exp.add_argument("--batch-size", type=int, help="覆盖 trainer.batch_size")
    p_exp.add_argument("--learning-rate", type=float, help="覆盖 trainer.learning_rate")
    p_exp.add_argument("--output-dir", type=str, help="覆盖 output_dir")
    # Phase 6.3：预检模式（不实际训练，仅检查配置/资源/数据）
    p_exp.add_argument("--dry-run", action="store_true",
                       help="预检模式：仅校验配置/资源/数据，不执行训练")
    # P0-1（2026-07-12）：--dry-run 默认启用动态前向校验（forward + backward 1 step），
    # 不启动 Lightning Trainer，无训练副作用。用户可通过 --static-only 显式跳过
    # 动态校验（如纯配置检查场景）。取代旧 --dynamic（语义反转，避免两个互斥标志）。
    p_exp.add_argument("--static-only", action="store_true",
                       help="dry-run 时跳过动态前向校验（默认启用动态校验；仅 --dry-run 时生效）")
    # Phase 7.1：训练后多格式导出
    p_exp.add_argument("--export-formats", type=str, default=None,
                       help="训练后导出模型格式，逗号分隔，如 onnx,torchscript")
    # Phase 7.2：自愈重试
    p_exp.add_argument("--retry", action="store_true", default=False,
                       help="启用自愈重试（OOM 自动降 batch_size，瞬时 IO 错误重试）")
    p_exp.add_argument("--no-retry", action="store_true", default=False,
                       help="显式禁用重试（覆盖配置文件中的 retry 设置）")
    # HPO 选项
    p_exp.add_argument("--hpo", action="store_true",
                       help="启用超参搜索（覆盖 hpo.enabled=True）")
    p_exp.add_argument("--hpo-trials", type=int, help="HPO trial 数量")
    p_exp.add_argument("--hpo-metric", type=str, help="HPO 优化指标名")
    p_exp.add_argument("--hpo-direction", type=str,
                       choices=["minimize", "maximize"], help="HPO 优化方向")
    # 日志控制
    p_exp.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARN", "ERROR"], help="日志级别")
    p_exp.add_argument("--log-file", type=str, help="日志文件路径")

    # RFC-002 阶段 W：exploration 探索状态管理
    p_explore = subparsers.add_parser("exploration", help="探索状态管理")
    explore_sub = p_explore.add_subparsers(dest="explore_action", required=True)
    explore_sub.add_parser("list", help="列出试验")
    explore_sub.add_parser("recommend", help="推荐下一步")
    explore_sub.add_parser("coverage", help="覆盖率统计")
    explore_sub.add_parser("map", help="搜索空间地图")
    explore_sub.add_parser("dashboard", help="可视化仪表盘")
    p_explore.add_argument("--exploration-file", type=str, default=None,
                           help="显式指定 exploration.json 路径（否则自动发现）")
    p_explore.add_argument("--output-dir", type=str, default=".",
                           help="自动发现 exploration.json 的搜索目录（默认当前目录）")
    p_explore.add_argument("--task-type", type=str, default=None)
    p_explore.add_argument("--top-k", type=int, default=5)
    p_explore.add_argument("--status", type=str, default=None)
    p_explore.add_argument("--dataset", type=str, default=None)
    p_explore.add_argument("--format", choices=["text", "markdown", "html"], default="text")

    # RFC-002 阶段 W：skills 技能库管理
    p_skills = subparsers.add_parser("skills", help="技能库管理")
    skills_sub = p_skills.add_subparsers(dest="skills_action", required=True)
    skills_sub.add_parser("list", help="列出所有技能")
    p_skills_search = skills_sub.add_parser("search", help="检索技能")
    p_skills_search.add_argument("query", type=str, help="检索查询")
    p_skills_search.add_argument("--top-k", type=int, default=5)
    p_skills_show = skills_sub.add_parser("show", help="显示技能详情")
    p_skills_show.add_argument("name", type=str, help="技能名")
    p_skills_remove = skills_sub.add_parser("remove", help="移除技能")
    p_skills_remove.add_argument("name", type=str, help="技能名")
    p_skills_remove.add_argument("--force", action="store_true", help="强制移除")

    # RFC-002 阶段 W：catalog 技术目录查询
    p_catalog = subparsers.add_parser("catalog", help="技术目录查询")
    catalog_sub = p_catalog.add_subparsers(dest="catalog_action", required=True)
    p_catalog_list = catalog_sub.add_parser("list", help="列出技术目录")
    p_catalog_list.add_argument("--scene", type=str, default=None)
    p_catalog_pipeline = catalog_sub.add_parser("pipeline", help="推荐 pipeline")
    p_catalog_pipeline.add_argument("--dataset", type=str, required=True)
    p_catalog_pipeline.add_argument("--scene", type=str, default="wifi_csi")
    p_catalog_augment = catalog_sub.add_parser("augment", help="推荐增强")
    p_catalog_augment.add_argument("--dataset", type=str, required=True)
    p_catalog_augment.add_argument("--scene", type=str, default="wifi_csi")

    # P2：训练实时监控
    p_monitor = subparsers.add_parser("monitor", help="训练实时监控")
    p_monitor.add_argument("output_dir", type=str, help="训练输出目录")

    # P3：推理服务
    p_serve = subparsers.add_parser(
        "serve",
        help="启动推理服务（加载训练输出目录的模型，暴露 HTTP 推理 API）",
    )
    p_serve.add_argument("output_dir", type=str,
                         help="训练输出目录（含 model.onnx 或 model.pth + metadata.json）")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    p_serve.add_argument("--port", type=int, default=8000, help="监听端口")
    p_serve.add_argument("--device", type=str, default="cpu",
                         choices=["cpu", "cuda"], help="推理设备")

    # s3：场景包脚手架
    p_create = subparsers.add_parser(
        "create-scene",
        help="创建场景包脚手架（从模板生成场景包目录）",
    )
    p_create.add_argument("scene_name", type=str,
                          help="场景名（snake_case，如 my_scene）")
    p_create.add_argument("--output", type=str, default=None,
                          help="输出父目录（默认 scenes/，目录名固定为 scene_name）")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 分发命令
    cmd_map = {
        "probe": _cmd_probe,
        "list-models": _cmd_list_models,
        "list-datasets": _cmd_list_datasets,
        "list-scenes": _cmd_list_scenes,
        "paradigms": _cmd_paradigms,
        "recommend": _cmd_recommend,
        "export": _cmd_export,
        "predict": _cmd_predict,
        "experiment": _cmd_experiment,
        # RFC-002 阶段 W：新能力子命令
        "exploration": _cmd_exploration,
        "skills": _cmd_skills,
        "catalog": _cmd_catalog,
        # P2：训练实时监控
        "monitor": _cmd_monitor,
        # P3：推理服务
        "serve": _cmd_serve,
        # s3：场景包脚手架
        "create-scene": _cmd_create_scene,
    }

    handler = cmd_map.get(args.command)
    if handler is None:
        print(json.dumps({"error": f"Unknown command: {args.command}", "code": "UNKNOWN_COMMAND"}))
        sys.exit(1)

    try:
        handler(args)
    except Exception as e:
        # A4: 错误上下文（traceback 落盘 + JSON 摘要）
        tb = traceback.format_exc()
        print(json.dumps({
            "error": str(e),
            "code": type(e).__name__,
            "traceback": tb,
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
