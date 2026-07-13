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

import logging
import threading
from typing import Any, Dict, List, Optional

from .base import DefaultConfig, SceneContainer, SceneMeta, SearchSpace, TransformConfig

_logger = logging.getLogger(__name__)


# ============================================================
# 线程安全保护（问题 4.7）
# ============================================================
# 用 RLock 避免同线程递归死锁（如 _ensure_lazy_scene 调用 register_scene）
# 锁粒度最小化：只锁 _REGISTRY / _LAZY_SCENES / _LAZY_PROXIES 的字典读写，
# 不锁外部代码（如 register_factories 内的 import、container.meta() 等）
_SCENES_LOCK = threading.RLock()


# ============================================================
# 延迟注册场景声明
# ============================================================
# 延迟注册的场景：首次访问时才实例化容器，但元数据可静态声明
_LAZY_SCENES: Dict[str, SceneMeta] = {}

# 问题 4.8：LazySceneContainer proxy 缓存
# 同 name 只创建一次 proxy，避免 _REGISTRY 内容器实例漂移
_LAZY_PROXIES: Dict[str, Any] = {}


def declare_lazy_scene(name: str, meta: SceneMeta) -> None:
    """声明延迟注册场景的元数据（不实例化容器）。

    允许 list_scenes/has_scene 在场景未激活时返回元数据，
    避免在框架核心层硬编码具体场景信息。
    """
    with _SCENES_LOCK:
        _LAZY_SCENES[name] = meta


def _ensure_lazy_scene(name: str) -> None:
    """触发延迟注册场景的实例化。

    修复（问题 4.8）：缓存 proxy 实例，同 name 只创建一次
    LazySceneContainer，避免 _REGISTRY 内容器实例漂移。

    修复（问题 4.7）：所有 _REGISTRY / _LAZY_PROXIES 读写均在 _SCENES_LOCK 内。
    import 在锁外执行（避免锁住外部代码）；register_scene 内部自带锁（RLock 可重入）。
    """
    # 快速路径：已注册直接返回
    with _SCENES_LOCK:
        if name in _REGISTRY:
            return
        proxy = _LAZY_PROXIES.get(name)

    if proxy is not None:
        # proxy 已缓存，直接注册（register_scene 内部加锁）
        _register_cached_proxy(name, proxy)
        return

    # 首次创建 proxy（import 在锁外执行，避免锁住外部代码）
    if name == "wifi_csi":
        from ._wifi_csi_lazy import LazyWiFiCSIContainer
        proxy = LazyWiFiCSIContainer()
    else:
        return

    # 缓存 proxy（双检：避免多线程同时创建不同 proxy 实例）
    with _SCENES_LOCK:
        if name in _LAZY_PROXIES:
            proxy = _LAZY_PROXIES[name]
        else:
            _LAZY_PROXIES[name] = proxy

    _register_cached_proxy(name, proxy)


def _register_cached_proxy(name: str, proxy: Any) -> None:
    """用缓存的 proxy 注册到 _REGISTRY（并发安全）。

    并发场景下另一线程可能已注册同名 scene，此时忽略 ValueError。
    """
    with _SCENES_LOCK:
        if name in _REGISTRY:
            return
    try:
        register_scene(name, proxy)
    except ValueError:
        # 并发场景：另一线程已注册同名 scene，忽略
        pass

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
    _validate_scene_capabilities(container)  # 纯验证（getattr），不调外部代码
    with _SCENES_LOCK:
        if not overwrite and name in _REGISTRY:
            raise ValueError(f"Scene '{name}' already registered")
        _REGISTRY[name] = container


def get_scene(name: str) -> SceneContainer:
    """获取已注册的场景容器（支持延迟注册）。

    修复（问题 4.7）：_REGISTRY / _LAZY_SCENES 读操作加锁。
    _ensure_lazy_scene 触发的外部 import 在锁外执行。
    """
    with _SCENES_LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
        is_lazy = name in _LAZY_SCENES
        if not is_lazy:
            available = list(_REGISTRY.keys()) + list(_LAZY_SCENES.keys())
            raise ValueError(f"Unknown scene: '{name}'. Available: {available}")
    # 触发延迟注册（import 在 _ensure_lazy_scene 内部锁外执行）
    _ensure_lazy_scene(name)
    with _SCENES_LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
        available = list(_REGISTRY.keys()) + list(_LAZY_SCENES.keys())
    raise ValueError(f"Unknown scene: '{name}'. Available: {available}")


def list_scenes(
    task_type: Optional[str] = None,
    learning_mode: Optional[str] = None,
    modality: Optional[str] = None,
    include_unavailable: bool = False,
) -> Dict[str, Any]:
    """列出所有已注册场景及其元数据（含延迟注册场景）。

    P5 P3-1：与 list_models/list_datasets 的过滤维度对称，支持按 task_type/learning_mode/modality 筛选。

    防御性查询：若 _REGISTRY 中某容器的 .meta() 抛异常（如外部依赖缺失
    导致半激活容器），回退到 _LAZY_SCENES 中的静态元数据，不阻塞其他
    场景的列举。这是 list_scenes 作为纯查询函数应有的健壮性
    （CQS 合规：不修改注册表，但容忍容器自身的实例化失败）。

    修复（问题 4.9）：异常不再静默吞，logger.warning 记录失败场景。
    失败场景名加入返回值的 "_unavailable" 列表（key 为 "_unavailable"，
    value 为 List[str]），供调用方感知哪些场景激活失败。
    注意：调用方迭代返回值时应跳过 "_unavailable" key（其 value 是 List[str]
    而非 SceneMeta）。

    Args:
        task_type: 按任务类型过滤（如 "classification"，None=不过滤）
        learning_mode: 按学习模式过滤（None=不过滤）
        modality: 按数据模态过滤（None=不过滤）
        include_unavailable: 是否包含激活失败的场景（默认 False 排除）
    """
    result: Dict[str, Any] = {}
    unavailable: List[str] = []
    # 快照注册表（锁内），container.meta() 在锁外执行（可能调用外部代码）
    with _SCENES_LOCK:
        items = list(_REGISTRY.items())
        lazy_items = list(_LAZY_SCENES.items())
    for name, container in items:
        try:
            result[name] = container.meta()
        except Exception as e:
            # 容器实例化失败（如 wifi_csi 缺 SenseFi 依赖）：
            # logger.warning 留痕（不再静默吞），回退到 _LAZY_SCENES 静态元数据
            _logger.warning(f"list_scenes: scene {name} activation failed: {e}")
            unavailable.append(name)
            with _SCENES_LOCK:
                if name in _LAZY_SCENES:
                    result[name] = _LAZY_SCENES[name]
    # 延迟注册场景：即使未激活也列出其元数据
    for name, meta in lazy_items:
        if name not in result:
            result[name] = meta
    # P5 P3-1：过滤维度
    def _matches(meta) -> bool:
        if task_type is not None and task_type not in (getattr(meta, "supported_tasks", None) or []):
            return False
        if learning_mode is not None and learning_mode not in (getattr(meta, "supported_learning_modes", None) or ["supervised"]):
            return False
        if modality is not None and getattr(meta, "modality", "unknown") != modality:
            return False
        return True
    result = {k: v for k, v in result.items() if k == "_unavailable" or _matches(v)}
    if unavailable:
        if include_unavailable:
            result["_unavailable"] = unavailable
        else:
            result.pop("_unavailable", None)
    return result


def has_scene(name: str) -> bool:
    """检查场景是否已注册（含延迟注册场景）。"""
    with _SCENES_LOCK:
        return name in _REGISTRY or name in _LAZY_SCENES


def activate_lazy_scenes() -> Dict[str, str]:
    """激活所有延迟注册场景。

    入口点契约：任何查询 registry 的入口点（CLI/scripts/external API）
    必须先调用此函数显式激活场景。CQS 合规改造后，getter 不再有副作用，
    激活责任转移到调用方。

    逐个触发延迟场景的实例化（调用 get_scene 触发 _ensure_lazy_scene）。
    某个场景因依赖缺失而激活失败时，跳过该场景，不影响其他场景。

    激活验证：注册容器后立即调用 .meta() 验证可正常实例化。若 .meta()
    抛异常（如外部依赖缺失），从 _REGISTRY 回滚移除，使该场景保持
    "仅 _LAZY_SCENES 元数据可用"状态，避免 list_scenes() 后续触发半激活
    容器的 .meta() 而抛异常（反向验证发现的方案 B 缺陷）。

    修复（异常分级）：
    - ImportError（如 SenseFi 缺失）：logger.warning + 降级，元数据可能已注册
      （依赖 register_metadata 拆分），list_models / get_default_epochs 仍可用。
    - 其他异常（TypeError/AttributeError/NameError 等代码 bug）：
      logger.error + 留 exc_info 痕迹，便于排查。
    旧逻辑 except Exception 静默吞所有异常，用户无法感知 wifi_csi 不可用。

    返回 dict: {scene_name: error_message}，激活成功的场景不在 dict 中。
    供 CLI list-models / list-datasets 等需要在场景上下文外查询注册表的入口调用，
    确保 wifi_csi 等延迟场景的模型/数据集元数据已注册到全局注册表。

    See: RFC-004 方案 B — 入口点显式激活契约
    """
    errors: Dict[str, str] = {}
    # 快照 _LAZY_SCENES / _REGISTRY keys（锁内），避免迭代时其他线程修改
    with _SCENES_LOCK:
        lazy_names = list(_LAZY_SCENES.keys())
        registered_names = set(_REGISTRY.keys())
    for name in lazy_names:
        if name in registered_names:
            continue
        try:
            container = get_scene(name)
            # 验证容器可正常实例化：调用 .meta() 触发完整激活
            container.meta()
        except ImportError as e:
            # 预期可恢复异常：外部依赖缺失，降级处理
            # register_metadata() 可能已完成，list_models 等元数据查询仍可用
            errors[name] = f"ImportError: {e}"
            _logger.warning(
                f"Scene '{name}' activation skipped (missing dependency): {e}"
            )
            # 回滚 _REGISTRY，保留 _LAZY_SCENES 元数据
            with _SCENES_LOCK:
                _REGISTRY.pop(name, None)
        except Exception as e:
            # 代码 bug：留 exc_info 痕迹，便于排查
            errors[name] = f"{type(e).__name__}: {e}"
            _logger.error(
                f"Unexpected error activating scene '{name}': {e}",
                exc_info=True,
            )
            with _SCENES_LOCK:
                _REGISTRY.pop(name, None)
    return errors


# ============================================================
# 自动注册内置场景
# ============================================================
def _register_builtin_scenes() -> None:
    """注册内置场景容器。

    WiFi CSI 场景延迟注册：仅在首次 get_scene("wifi_csi") 时触发，
    避免 SenseFi 代码库缺失时阻塞框架其他功能。

    修复（异常隔离）：旧逻辑 generic/custom/detection 任一导入失败会导致
    import senseframe 崩溃。改为逐个 try/except，单个场景失败不影响其他。
    """
    # generic 场景
    try:
        from .generic.container import GenericContainer
        if "generic" not in _REGISTRY:
            register_scene("generic", GenericContainer())
    except Exception as e:
        _logger.error(f"Failed to register 'generic' scene: {e}", exc_info=True)

    # custom 场景
    try:
        from .custom.container import CustomContainer
        if "custom" not in _REGISTRY:
            register_scene("custom", CustomContainer())
    except Exception as e:
        _logger.error(f"Failed to register 'custom' scene: {e}", exc_info=True)

    # detection 场景
    try:
        from .detection.container import DetectionContainer
        if "detection" not in _REGISTRY:
            register_scene("detection", DetectionContainer())
    except Exception as e:
        _logger.error(f"Failed to register 'detection' scene: {e}", exc_info=True)

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
