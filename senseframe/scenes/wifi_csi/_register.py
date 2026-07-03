"""
WiFi CSI 场景注册：将模型、数据集、归一化策略注册到全局注册表。

仅在场景激活时调用（延迟注册），不阻塞框架其他功能。
SenseFi 代码库缺失时给出清晰提示。
"""

from ...registry import (
    ZScoreStrategy,
    bind_scene_factory,
    register_dataset,
    register_model,
    register_normalization,
    set_scene_epochs,
)

_wifi_csi_registered = False


def is_registered() -> bool:
    return _wifi_csi_registered


def register() -> None:
    """注册 WiFi CSI 场景的全部模型、数据集、归一化策略。"""
    global _wifi_csi_registered
    if _wifi_csi_registered:
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
    _DATASET_META = [
        ("UT_HAR_data",    7,  (1, 250, 90),  ("UT_HAR",),              "npy",  "tensor"),
        ("NTU-Fi-HumanID", 14, (3, 114, 500), ("NTU-Fi-HumanID",),      "mat",  "csi_mat"),
        ("NTU-Fi_HAR",     6,  (3, 114, 500), ("NTU-Fi_HAR",),          "mat",  "csi_mat",
         "", "NTU-Fi-HumanID"),  # unsupervised_source="", supervised_source="NTU-Fi-HumanID"
        ("Widar",          22, (22, 20, 20),  ("Widardata", "Widar"),   "csv",  "csv_folder"),
    ]
    for entry in _DATASET_META:
        name, num_classes, input_shape, dir_names, file_format, loader_type = entry[:6]
        kwargs = dict(
            num_classes=num_classes,
            input_shape=input_shape,
            dir_names=dir_names,
            file_format=file_format,
            loader_type=loader_type,
        )
        if len(entry) > 6:
            kwargs["unsupervised_source"] = entry[6]
        if len(entry) > 7:
            kwargs["supervised_source"] = entry[7]
        register_dataset(name, **kwargs)

    # 3) 归一化策略
    register_normalization("NTU-Fi_HAR",     ZScoreStrategy(42.3199, 4.9802))
    register_normalization("NTU-Fi-HumanID", ZScoreStrategy(42.3199, 4.9802))
    register_normalization("Widar",          ZScoreStrategy(0.0025,  0.0119))

    # 4) 导入 SenseFi 模型（延迟导入，缺失时给出清晰提示）
    try:
        import sys
        from pathlib import Path

        SENSEFI_PATH = Path(__file__).resolve().parents[3] / "CSI_DATASETS" / "WiFi-CSI-Sensing-Benchmark"
        if str(SENSEFI_PATH) not in sys.path:
            sys.path.insert(0, str(SENSEFI_PATH))

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
            f"请确保 CSI_DATASETS/WiFi-CSI-Sensing-Benchmark/ 目录存在并包含模型文件。\n"
            f"如果不需要 WiFi CSI 场景，可以使用其他场景（generic/custom）。"
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

    # 7) 默认 epoch 表
    _EPOCHS = {
        ("MLP", "UT_HAR_data"): 200, ("LeNet", "UT_HAR_data"): 200,
        ("ResNet18", "UT_HAR_data"): 200, ("ResNet50", "UT_HAR_data"): 200, ("ResNet101", "UT_HAR_data"): 200,
        ("RNN", "UT_HAR_data"): 3000, ("GRU", "UT_HAR_data"): 200,
        ("LSTM", "UT_HAR_data"): 200, ("BiLSTM", "UT_HAR_data"): 200,
        ("CNN+GRU", "UT_HAR_data"): 200, ("ViT", "UT_HAR_data"): 200,
        ("MLP", "NTU-Fi-HumanID"): 50, ("LeNet", "NTU-Fi-HumanID"): 50,
        ("ResNet18", "NTU-Fi-HumanID"): 50, ("ResNet50", "NTU-Fi-HumanID"): 50, ("ResNet101", "NTU-Fi-HumanID"): 50,
        ("RNN", "NTU-Fi-HumanID"): 75, ("GRU", "NTU-Fi-HumanID"): 50,
        ("LSTM", "NTU-Fi-HumanID"): 50, ("BiLSTM", "NTU-Fi-HumanID"): 50,
        ("CNN+GRU", "NTU-Fi-HumanID"): 200, ("ViT", "NTU-Fi-HumanID"): 50,
        ("MLP", "NTU-Fi_HAR"): 30, ("LeNet", "NTU-Fi_HAR"): 30,
        ("ResNet18", "NTU-Fi_HAR"): 30, ("ResNet50", "NTU-Fi_HAR"): 30, ("ResNet101", "NTU-Fi_HAR"): 30,
        ("RNN", "NTU-Fi_HAR"): 70, ("GRU", "NTU-Fi_HAR"): 30,
        ("LSTM", "NTU-Fi_HAR"): 30, ("BiLSTM", "NTU-Fi_HAR"): 30,
        ("CNN+GRU", "NTU-Fi_HAR"): 100, ("ViT", "NTU-Fi_HAR"): 30,
        ("MLP", "Widar"): 30, ("LeNet", "Widar"): 100,
        ("ResNet18", "Widar"): 100, ("ResNet50", "Widar"): 100, ("ResNet101", "Widar"): 100,
        ("RNN", "Widar"): 500, ("GRU", "Widar"): 200,
        ("LSTM", "Widar"): 200, ("BiLSTM", "Widar"): 200,
        ("CNN+GRU", "Widar"): 200, ("ViT", "Widar"): 200,
    }
    for (model_id, dataset), epochs in _EPOCHS.items():
        set_scene_epochs("wifi_csi", model_id, dataset, epochs)

    _wifi_csi_registered = True
