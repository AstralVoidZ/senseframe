"""ISP-8 + ISP-9 + ISP-10：``senseframe://scenes`` + capabilities + search-space。

3 个 Resource 端点：
- ISP-8:  senseframe://scenes                          — 场景目录（list_scenes）
- ISP-9:  senseframe://scenes/{name}/capabilities      — 场景能力声明（SceneMeta）
- ISP-10: senseframe://search-space/{scene}/{model_id} — 搜索空间（ParameterSpec 列表）

依赖：
- senseframe.scenes.list_scenes / get_scene
- senseframe.scenes.base.SceneMeta

注意：resources/ 不得 import tools/（AST 守卫测试钉死）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["scenes", "scene_capabilities", "search_space"]


async def scenes() -> dict[str, Any]:
    """ISP-8：场景目录（调用 list_scenes）。

    Returns:
        含 schema_version + scenes 列表（每条含 name + meta 字段）。
    """
    from senseframe.scenes import list_scenes

    raw = list_scenes()
    # list_scenes 返回 {name: SceneMeta | dict, ...}（可能含 _unavailable key）
    scenes_list: list[dict[str, Any]] = []
    for name, meta in raw.items():
        if name == "_unavailable":
            continue
        scenes_list.append({
            "name": name,
            "meta": _meta_to_dict(meta),
        })
    return {
        "schema_version": "1.0.0",
        "scenes": scenes_list,
    }


async def scene_capabilities(name: str) -> dict[str, Any]:
    """ISP-9：场景能力声明（SceneMeta）。

    Args:
        name: 场景名（路径参数，如 "wifi_csi" / "generic"）。

    Returns:
        含 name + capabilities 字段（SceneMeta 的所有字段）。
    """
    from senseframe.scenes import get_scene

    scene = get_scene(name)
    meta = scene.meta()
    return {
        "name": name,
        "capabilities": _meta_to_dict(meta),
    }


async def search_space(scene: str, model_id: str) -> dict[str, Any]:
    """ISP-10：搜索空间（ParameterSpec 列表）。

    Args:
        scene: 场景名（路径参数）。
        model_id: 模型 ID（路径参数）。

    Returns:
        含 scene + model_id + parameters 列表（每条含 ParameterSpec 字段）。
    """
    from senseframe.scenes import get_scene

    container = get_scene(scene)
    # SceneContainer.get_search_space 是可选方法（9 可选方法之一）
    search_space_obj = None
    if hasattr(container, "get_search_space"):
        try:
            search_space_obj = container.get_search_space(model_id=model_id)
        except TypeError:
            # 签名不匹配，尝试无参数调用
            try:
                search_space_obj = container.get_search_space()
            except Exception:
                search_space_obj = None
        except Exception:
            search_space_obj = None

    parameters: list[dict[str, Any]] = []
    if search_space_obj is not None:
        # search_space_obj 可能是 SearchSpace dataclass 或 dict
        if hasattr(search_space_obj, "to_dict"):
            parameters = search_space_obj.to_dict().get("parameters", [])
        elif isinstance(search_space_obj, dict):
            parameters = search_space_obj.get("parameters", [])
        elif hasattr(search_space_obj, "parameters"):
            for p in search_space_obj.parameters:
                if hasattr(p, "to_dict"):
                    parameters.append(p.to_dict())
                else:
                    parameters.append(dict(p))

    return {
        "scene": scene,
        "model_id": model_id,
        "parameters": parameters,
        "advisory": True,
    }


# ============================================================
# 辅助函数
# ============================================================


def _meta_to_dict(meta: Any) -> dict[str, Any]:
    """将 SceneMeta（dataclass 或 dict）转换为 dict。"""
    if hasattr(meta, "to_dict"):
        return meta.to_dict()
    if isinstance(meta, dict):
        return meta
    # dataclass.asdict 兜底
    try:
        from dataclasses import asdict
        return asdict(meta)
    except Exception:
        return {"name": str(meta)}
