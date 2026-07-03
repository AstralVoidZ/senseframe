"""
场景注册表：通过名称获取场景容器实例。

使用方式：
    from senseframe.scenes import get_scene, list_scenes, register_scene

    # 获取已注册场景
    scene = get_scene("wifi_csi")

    # 列出所有场景
    print(list_scenes())

    # 注册自定义场景
    register_scene("my_scene", MySceneContainer())
"""

from typing import Any, Dict, List

from .base import DefaultConfig, SceneContainer, SceneMeta, SearchSpace, TransformConfig


# ============================================================
# 延迟注册场景声明
# ============================================================
# 延迟注册的场景：首次访问时才实例化容器，但元数据可静态声明
_LAZY_SCENES: Dict[str, SceneMeta] = {}


def declare_lazy_scene(name: str, meta: SceneMeta) -> None:
    """声明延迟注册场景的元数据（不实例化容器）。

    允许 list_scenes/has_scene 在场景未激活时返回元数据，
    避免在框架核心层硬编码具体场景信息。
    """
    _LAZY_SCENES[name] = meta


def _ensure_lazy_scene(name: str) -> None:
    """触发延迟注册场景的实例化。"""
    if name in _REGISTRY:
        return
    if name == "wifi_csi":
        from ._wifi_csi_lazy import LazyWiFiCSIContainer
        register_scene("wifi_csi", LazyWiFiCSIContainer())

# ============================================================
# 注册表
# ============================================================
# RFC-002 阶段 K：支持 duck typing，不强制继承 SceneContainer
_REGISTRY: Dict[str, Any] = {}

# 场景容器必需实现的方法
_REQUIRED_SCENE_METHODS = ("meta", "load_dataset", "build_model_for_dataset", "get_dataset_info")


def _validate_scene_capabilities(obj: Any) -> None:
    """验证场景容器具备必需方法（duck typing 校验）。"""
    missing = [m for m in _REQUIRED_SCENE_METHODS if not callable(getattr(obj, m, None))]
    if missing:
        raise TypeError(
            f"Scene container missing required methods: {missing}. "
            f"Must implement: {_REQUIRED_SCENE_METHODS}"
        )


def register_scene(name: str, container: Any, *, overwrite: bool = False) -> None:
    """注册场景容器。

    RFC Phase A：统一重复注册行为，默认不覆盖（overwrite=False），
    与 register_model/register_dataset 一致。
    Agent 重新注册同名场景时需显式 overwrite=True。

    RFC-002 阶段 K：支持 duck typing，不强制继承 SceneContainer。
    只需实现 4 个必需方法：meta / load_dataset / build_model_for_dataset / get_dataset_info。

    Args:
        name: 场景名称
        container: 场景容器实例（继承 SceneContainer 或鸭子类型）
        overwrite: True 时覆盖已注册的同名场景；False 时已存在则 raise
    """
    if not overwrite and name in _REGISTRY:
        raise ValueError(f"Scene '{name}' already registered")
    _validate_scene_capabilities(container)
    _REGISTRY[name] = container


def get_scene(name: str) -> SceneContainer:
    """获取已注册的场景容器（支持延迟注册）。"""
    if name not in _REGISTRY and name in _LAZY_SCENES:
        _ensure_lazy_scene(name)
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys()) + list(_LAZY_SCENES.keys())
        raise ValueError(f"Unknown scene: '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_scenes() -> Dict[str, SceneMeta]:
    """列出所有已注册场景及其元数据（含延迟注册场景）。

    防御性查询：若 _REGISTRY 中某容器的 .meta() 抛异常（如外部依赖缺失
    导致半激活容器），回退到 _LAZY_SCENES 中的静态元数据，不阻塞其他
    场景的列举。这是 list_scenes 作为纯查询函数应有的健壮性
    （CQS 合规：不修改注册表，但容忍容器自身的实例化失败）。
    """
    result: Dict[str, SceneMeta] = {}
    for name, container in _REGISTRY.items():
        try:
            result[name] = container.meta()
        except Exception:
            # 容器实例化失败（如 wifi_csi 缺 SenseFi 依赖）：
            # 回退到 _LAZY_SCENES 静态元数据，若也不存在则跳过
            if name in _LAZY_SCENES:
                result[name] = _LAZY_SCENES[name]
    # 延迟注册场景：即使未激活也列出其元数据
    for name, meta in _LAZY_SCENES.items():
        if name not in result:
            result[name] = meta
    return result


def has_scene(name: str) -> bool:
    """检查场景是否已注册（含延迟注册场景）。"""
    return name in _REGISTRY or name in _LAZY_SCENES


def activate_lazy_scenes() -> None:
    """激活所有延迟注册场景。

    入口点契约：任何查询 registry 的入口点（CLI/scripts/external API）
    必须先调用此函数显式激活场景。CQS 合规改造后，getter 不再有副作用，
    激活责任转移到调用方。

    逐个触发延迟场景的实例化（调用 get_scene 触发 _ensure_lazy_scene）。
    某个场景因依赖缺失而激活失败时，静默跳过，不影响其他场景。

    激活验证：注册容器后立即调用 .meta() 验证可正常实例化。若 .meta()
    抛异常（如外部依赖缺失），从 _REGISTRY 回滚移除，使该场景保持
    "仅 _LAZY_SCENES 元数据可用"状态，避免 list_scenes() 后续触发半激活
    容器的 .meta() 而抛异常（反向验证发现的方案 B 缺陷）。

    供 CLI list-models / list-datasets 等需要在场景上下文外查询注册表的入口调用，
    确保 wifi_csi 等延迟场景的模型/数据集元数据已注册到全局注册表。

    See: RFC-004 方案 B — 入口点显式激活契约
    """
    for name in list(_LAZY_SCENES.keys()):
        if name not in _REGISTRY:
            try:
                container = get_scene(name)
                # 验证容器可正常实例化：调用 .meta() 触发完整激活
                container.meta()
            except Exception:
                # 激活失败：回滚 _REGISTRY，保留 _LAZY_SCENES 元数据
                _REGISTRY.pop(name, None)


# ============================================================
# 自动注册内置场景
# ============================================================
def _register_builtin_scenes() -> None:
    """注册内置场景容器。

    WiFi CSI 场景延迟注册：仅在首次 get_scene("wifi_csi") 时触发，
    避免 SenseFi 代码库缺失时阻塞框架其他功能。
    """
    from .generic.container import GenericContainer

    if "generic" not in _REGISTRY:
        register_scene("generic", GenericContainer())

    from .custom.container import CustomContainer

    if "custom" not in _REGISTRY:
        register_scene("custom", CustomContainer())

    from .detection.container import DetectionContainer

    if "detection" not in _REGISTRY:
        register_scene("detection", DetectionContainer())

    # WiFi CSI 延迟注册：声明元数据但不实例化容器
    declare_lazy_scene("wifi_csi", SceneMeta(
        name="wifi_csi",
        supported_tasks=["classification", "self_supervised"],
        supported_models=["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
                          "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"],
        supported_datasets=["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"],
        requires_custom_dataloader=True,
        supported_learning_modes=["supervised", "self_supervised"],
    ))


_register_builtin_scenes()


__all__ = [
    "SceneContainer",
    "SceneMeta",
    "DefaultConfig",
    "SearchSpace",
    "TransformConfig",
    "register_scene",
    "get_scene",
    "list_scenes",
    "has_scene",
    "activate_lazy_scenes",
]
