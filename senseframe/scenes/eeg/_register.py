"""EEG 场景延迟注册：首次访问时实例化容器。

P1.2 落地：对齐 wifi_csi 的延迟注册模式。
EEG 场景无外部依赖（模型内置 PyTorch 实现），但仍走延迟注册路径。
"""

_registered = False


def is_registered() -> bool:
    return _registered


def register() -> None:
    """注册 EEG 场景的模型工厂绑定。

    EEG 场景的模型类在 senseframe/scenes/eeg/models.py 内置，
    无外部依赖，所以 register() 只做工厂绑定。
    容器 build_model_for_dataset 直接从 MODEL_REGISTRY 查找。
    """
    global _registered
    if _registered:
        return

    _registered = True
