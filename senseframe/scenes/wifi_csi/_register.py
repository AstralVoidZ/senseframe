"""
WiFi CSI 场景注册：将模型、数据集、归一化策略注册到全局注册表。

仅在场景激活时调用（延迟注册），不阻塞框架其他功能。
SenseFi 代码库缺失时给出清晰提示。

修复（拆分 register）：原 register() 把元数据注册（步骤 1-3 + 7）与
工厂绑定（步骤 4-6，依赖 SenseFi）混在一起，SenseFi 缺失时步骤 1-3 已执行
留下"幽灵模型"（list_models 返回 11 个无工厂绑定的模型元数据）。
拆分为 register_metadata()（不依赖 SenseFi）+ register_factories()（依赖 SenseFi），
register() 仍保留为两者顺序调用的入口。
"""

from ...registry import (
    ZScoreStrategy,
    bind_scene_factory,
    register_dataset,
    register_model,
    register_normalization,
)

_wifi_csi_registered = False
_wifi_csi_metadata_registered = False


def is_registered() -> bool:
    return _wifi_csi_registered


def register_metadata() -> None:
    """注册元数据（不依赖 SenseFi）：模型规格 + 数据集规格 + 归一化 + epoch 表。

    SenseFi 缺失时可独立调用，让 list_models / list_datasets / get_default_epochs
    等元数据查询正常工作，工厂绑定失败时不会留下"幽灵模型"。
    """
    global _wifi_csi_metadata_registered
    if _wifi_csi_metadata_registered:
        return

    # 1) 模型元数据（11 个）
    _MODEL_META = [
        ("MLP",       "traditional_ml", False, 512,  0.3,  1e-3),
        ("LeNet",     "cnn",            False, 1024, 0.5,  1e-3),
        ("ResNet18",  "cnn",            False, 2048, 11.2, 1e-3),
        ("ResNet50",  "cnn",            True,  4096, 23.5, 1e-3),
        ("ResNet101", "cnn",            True,  6144, 42.5, 1e-3),
        ("RNN",       "rnn",            False, 1024, 1.2,  1e-3),
        ("GRU",       "rnn",            False, 1024, 1.5,  1e-3),
        ("LSTM",      "rnn",            False, 1536, 2.0,  1e-3),
        ("BiLSTM",    "rnn",            False, 2048, 4.0,  1e-3),
        ("CNN+GRU",   "hybrid",         False, 2048, 5.5,  1e-3),
        ("ViT",       "transformer",    True,  4096, 8.0,  1e-3),
    ]
    for model_id, paradigm, requires_gpu, vram, params_m, lr in _MODEL_META:
        register_model(
            model_id,
            paradigm=paradigm,
            enabled=True,
            requires_gpu=requires_gpu,
            estimated_vram_mb=vram,
            estimated_params_m=params_m,
            default_lr=lr,
        )

    # 2) 数据集注册（4 个，含加载描述）
    # layout: nested=类别子目录, flat=扁平结构
    _DATASET_META = [
        # UT_HAR: data/label 子目录内扁平 .npy 文件
        ("UT_HAR_data",    7,  (1, 250, 90),  ("UT_HAR",),              "npy",  "tensor",  "flat"),
        # NTU-Fi-HumanID: train_amp/test_amp 下按类别子目录组织（如 test_amp/015/*.mat）
        ("NTU-Fi-HumanID", 14, (3, 114, 500), ("NTU-Fi-HumanID",),      "mat",  "csi_mat", "nested"),
        # NTU-Fi_HAR: train_amp/test_amp 下类别子目录（box/circle/clean/fall/run/walk）
        # NTU-Fi_HAR 与 NTU-Fi-HumanID 目录结构相同，均按类别子目录组织（nested layout）
        ("NTU-Fi_HAR",     6,  (3, 114, 500), ("NTU-Fi_HAR",),          "mat",  "csi_mat", "nested",
         "", "NTU-Fi-HumanID"),  # unsupervised_source="", supervised_source="NTU-Fi-HumanID"
        # Widar: train/test 子目录下类别子目录 <class_dir>/<sample>.csv
        ("Widar",          22, (22, 20, 20),  ("Widardata", "Widar"),   "csv",  "csv_folder", "nested"),
    ]
    # 修复根因（P2）：训练集样本数，供 get_default_epochs 动态计算推荐 epochs。
    # 值为各数据集训练集已知样本数（来自数据集规格）。
    # get_default_epochs 命中静态 _EPOCHS 表时取 min(静态, 动态)，避免
    # 硬编码 200 epoch 与实测 19 epoch 早停严重失配。
    _DATASET_N_SAMPLES = {
        "UT_HAR_data":    3977,   # UT-HAR 训练集样本数
        "NTU-Fi-HumanID": 1500,   # NTU-Fi-HumanID 训练集样本数（约）
        "NTU-Fi_HAR":     700,    # NTU-Fi_HAR 训练集样本数（约）
        "Widar":          5000,   # Widar 训练集样本数（约）
    }
    # P2-3 修复：各数据集 val split 策略
    # UT_HAR_data 原始 .npy 含 X_val/y_val，has_native_val=True
    # 其余数据集无原生 val，从 train 自动划分 10%（val_split_ratio=0.1）
    _DATASET_VAL_STRATEGY = {
        "UT_HAR_data":    {"has_native_val": True,  "val_split_ratio": None},
        "NTU-Fi-HumanID": {"has_native_val": False, "val_split_ratio": 0.1},
        "NTU-Fi_HAR":     {"has_native_val": False, "val_split_ratio": 0.1},
        "Widar":          {"has_native_val": False, "val_split_ratio": 0.1},
    }
    for entry in _DATASET_META:
        name, num_classes, input_shape, dir_names, file_format, loader_type, layout = entry[:7]
        val_strategy = _DATASET_VAL_STRATEGY.get(name, {})
        kwargs = dict(
            num_classes=num_classes,
            input_shape=input_shape,
            dir_names=dir_names,
            file_format=file_format,
            loader_type=loader_type,
            layout=layout,
            n_samples=_DATASET_N_SAMPLES.get(name),
            has_native_val=val_strategy.get("has_native_val", False),
            val_split_ratio=val_strategy.get("val_split_ratio"),
        )
        if len(entry) > 7:
            kwargs["unsupervised_source"] = entry[7]
        if len(entry) > 8:
            kwargs["supervised_source"] = entry[8]
        register_dataset(name, **kwargs)

    # 3) 归一化策略
    # P2-2 上策修复：提取共享归一化常数为模块级变量，显式声明 NTU-Fi_HAR 与
    # NTU-Fi-HumanID 共享同一组统计量（同源 CSI 采集设备/环境）。旧代码在两处
    # register_normalization 调用中各自硬编码相同数值，隐式耦合且易在维护时失同步。
    _NTU_FI_MEAN = 42.3199
    _NTU_FI_STD = 4.9802
    _ntu_fi_norm = ZScoreStrategy(_NTU_FI_MEAN, _NTU_FI_STD)
    register_normalization("NTU-Fi_HAR",     _ntu_fi_norm)
    register_normalization("NTU-Fi-HumanID", _ntu_fi_norm)
    register_normalization("Widar",          ZScoreStrategy(0.0025,  0.0119))
    # P1 修复：UT_HAR_data 归一化（从 X_train 全局统计量计算）
    # 旧代码未注册，get_normalization 回退 IdentityStrategy（no-op），导致原始幅值进入模型
    register_normalization("UT_HAR_data",    ZScoreStrategy(17.6529, 5.9034))

    # 7) epochs 已彻底去静态化（方案 B）：
    # 删除 _EPOCHS 静态表（原 44 条目），epochs 完全由 _compute_epochs_budget(n_samples)
    # 动态计算 + Early Stopping 实时控制。静态表无法预测最优 epochs，且与动态预算的
    # min 组合语义混乱。业界共识：epochs 是"预算上限"，Early Stopping 决定停止点。

    _wifi_csi_metadata_registered = True


def register_factories() -> None:
    """注册工厂绑定（依赖 SenseFi）：导入模型类 + 绑定工厂。

    SenseFi 缺失时 raise ImportError，调用方需捕获并降级处理。
    要求 register_metadata() 已调用（否则元数据未注册，工厂绑定无意义）。
    """
    if not _wifi_csi_metadata_registered:
        register_metadata()

    # 4) 导入 SenseFi 模型（延迟导入，缺失时给出清晰提示）
    # SenseFi 代码库路径由调用者通过 SENSEFRAME_SENSEFI_PATH 环境变量显式提供，
    # 框架不猜测、不探测候选目录名。未设置则 raise ImportError。
    import os
    import sys

    sensefi_path = os.environ.get("SENSEFRAME_SENSEFI_PATH")
    if not sensefi_path:
        raise ImportError(
            "WiFi CSI 场景需要 SenseFi 代码库，但 SENSEFRAME_SENSEFI_PATH 环境变量未设置。\n"
            "请设置：export SENSEFRAME_SENSEFI_PATH=/path/to/WiFi-CSI-Sensing-Benchmark\n"
            "如果不需要 WiFi CSI 场景，可以使用其他场景（generic/custom）。"
        )
    sensefi_path = os.path.abspath(sensefi_path)
    if sensefi_path not in sys.path:
        sys.path.insert(0, sensefi_path)

    try:
        from UT_HAR_model import (
            UT_HAR_MLP, UT_HAR_LeNet, UT_HAR_ResNet18, UT_HAR_ResNet50, UT_HAR_ResNet101,
            UT_HAR_RNN, UT_HAR_GRU, UT_HAR_LSTM, UT_HAR_BiLSTM, UT_HAR_CNN_GRU, UT_HAR_ViT,
        )
        from NTU_Fi_model import (
            NTU_Fi_MLP, NTU_Fi_LeNet, NTU_Fi_ResNet18, NTU_Fi_ResNet50, NTU_Fi_ResNet101,
            NTU_Fi_RNN, NTU_Fi_GRU, NTU_Fi_LSTM, NTU_Fi_BiLSTM, NTU_Fi_CNN_GRU, NTU_Fi_ViT,
        )
        from widar_model import (
            Widar_MLP, Widar_LeNet, Widar_ResNet18, Widar_ResNet50, Widar_ResNet101,
            Widar_RNN, Widar_GRU, Widar_LSTM, Widar_BiLSTM, Widar_CNN_GRU, Widar_ViT,
        )
        from self_supervised_model import (
            MLP_Parrallel, CNN_Parrallel, ResNet18_Parrallel, ResNet50_Parrallel, ResNet101_Parrallel,
            RNN_Parrallel, GRU_Parrallel, LSTM_Parrallel, BiLSTM_Parrallel, CNN_GRU_Parrallel, ViT_Parrallel,
        )
    except ImportError as e:
        raise ImportError(
            f"WiFi CSI 场景需要 SenseFi 代码库，但导入失败: {e}\n"
            f"SENSEFRAME_SENSEFI_PATH={sensefi_path}\n"
            f"请确保该目录存在并包含模型文件（UT_HAR_model.py / NTU_Fi_model.py / "
            f"widar_model.py / self_supervised_model.py）。"
        ) from e

    # 5) 监督模式工厂绑定（统一签名：factory(num_classes=None)）
    _SUPERVISED_FACTORIES = {
        "UT_HAR_data": {
            "MLP": UT_HAR_MLP, "LeNet": UT_HAR_LeNet,
            "ResNet18": UT_HAR_ResNet18, "ResNet50": UT_HAR_ResNet50, "ResNet101": UT_HAR_ResNet101,
            "RNN": UT_HAR_RNN, "GRU": UT_HAR_GRU, "LSTM": UT_HAR_LSTM, "BiLSTM": UT_HAR_BiLSTM,
            "CNN+GRU": UT_HAR_CNN_GRU, "ViT": UT_HAR_ViT,
        },
        "NTU-Fi-HumanID": {
            "MLP": NTU_Fi_MLP, "LeNet": NTU_Fi_LeNet,
            "ResNet18": NTU_Fi_ResNet18, "ResNet50": NTU_Fi_ResNet50, "ResNet101": NTU_Fi_ResNet101,
            "RNN": NTU_Fi_RNN, "GRU": NTU_Fi_GRU, "LSTM": NTU_Fi_LSTM, "BiLSTM": NTU_Fi_BiLSTM,
            "CNN+GRU": NTU_Fi_CNN_GRU, "ViT": NTU_Fi_ViT,
        },
        "NTU-Fi_HAR": {
            "MLP": NTU_Fi_MLP, "LeNet": NTU_Fi_LeNet,
            "ResNet18": NTU_Fi_ResNet18, "ResNet50": NTU_Fi_ResNet50, "ResNet101": NTU_Fi_ResNet101,
            "RNN": NTU_Fi_RNN, "GRU": NTU_Fi_GRU, "LSTM": NTU_Fi_LSTM, "BiLSTM": NTU_Fi_BiLSTM,
            "CNN+GRU": NTU_Fi_CNN_GRU, "ViT": NTU_Fi_ViT,
        },
        "Widar": {
            "MLP": Widar_MLP, "LeNet": Widar_LeNet,
            "ResNet18": Widar_ResNet18, "ResNet50": Widar_ResNet50, "ResNet101": Widar_ResNet101,
            "RNN": Widar_RNN, "GRU": Widar_GRU, "LSTM": Widar_LSTM, "BiLSTM": Widar_BiLSTM,
            "CNN+GRU": Widar_CNN_GRU, "ViT": Widar_ViT,
        },
    }

    def _make_factory(cls, dataset: str, model_id: str):
        """统一工厂签名：factory(num_classes=None)。

        - UT_HAR_data: cls() 无参构造（num_classes 硬编码在模型中）
        - ViT: cls(num_classes=num_classes) 关键字参数
        - 其他: cls(num_classes) 位置参数
        """
        if dataset == "UT_HAR_data":
            return lambda num_classes=None: cls()
        elif model_id == "ViT":
            return lambda num_classes=None: cls(num_classes=num_classes)
        else:
            return lambda num_classes=None: cls(num_classes)

    for dataset, factories in _SUPERVISED_FACTORIES.items():
        for model_id, cls in factories.items():
            factory = _make_factory(cls, dataset, model_id)
            bind_scene_factory(
                "wifi_csi", model_id, dataset, factory,
                learning_mode="supervised",
            )

    # 6) 自监督模式工厂绑定（仅 NTU-Fi_HAR 数据集支持）
    _SELFSUP_FACTORIES = {
        "MLP": MLP_Parrallel, "LeNet": CNN_Parrallel,
        "ResNet18": ResNet18_Parrallel, "ResNet50": ResNet50_Parrallel, "ResNet101": ResNet101_Parrallel,
        "RNN": RNN_Parrallel, "GRU": GRU_Parrallel, "LSTM": LSTM_Parrallel, "BiLSTM": BiLSTM_Parrallel,
        "CNN+GRU": CNN_GRU_Parrallel, "ViT": ViT_Parrallel,
    }
    for model_id, cls in _SELFSUP_FACTORIES.items():
        bind_scene_factory(
            "wifi_csi", model_id, "NTU-Fi_HAR",
            lambda num_classes=None, _cls=cls: _cls(),
            learning_mode="self_supervised",
        )


def register() -> None:
    """注册 WiFi CSI 场景的全部模型、数据集、归一化策略 + 工厂绑定。

    等价于 register_metadata() + register_factories()。
    SenseFi 缺失时 register_factories() 会 raise ImportError，
    但 register_metadata() 已完成，list_models / get_default_epochs 等元数据查询正常。
    """
    global _wifi_csi_registered
    if _wifi_csi_registered:
        return

    register_metadata()
    register_factories()

    _wifi_csi_registered = True
