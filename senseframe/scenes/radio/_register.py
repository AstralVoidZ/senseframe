"""Radio 场景延迟注册：首次访问时实例化容器。

P1.2 落地：对齐 wifi_csi 的延迟注册模式。
Radio 场景无外部依赖（模型内置 PyTorch 实现），但仍走延迟注册路径，
保证框架核心 import 时不触发场景子包导入。
"""

_registered = False


def is_registered() -> bool:
    return _registered


def register() -> None:
    """注册 Radio 场景的模型工厂绑定。

    Radio 场景的模型类在 senseframe/scenes/radio/models.py 内置，
    无外部依赖（不依赖 SenseFi 等），所以 register() 只做工厂绑定。
    """
    global _registered
    if _registered:
        return

    # Radio 场景模型工厂绑定走 MODEL_REGISTRY（按 dataset 选择模型类），
    # 容器 build_model_for_dataset 直接从 MODEL_REGISTRY 查找，
    # 无需走 registry.bind_scene_factory 全局注册表。

    _registered = True
