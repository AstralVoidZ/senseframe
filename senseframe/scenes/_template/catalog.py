"""{SCENE_NAME} 场景的技术目录（自动生成模板）。"""
from typing import Any, Dict, List

CATALOG: List[Dict[str, Any]] = [
    # TODO: 添加技术目录条目
    # {
    #     "name": "example_transform",
    #     "category": "feature_engineering",
    #     "description": "示例变换原语",
    #     "applicable": ["{SCENE_NAME}_dataset"],
    #     "implemented": True,
    #     "params": {},
    # },
]


def list_techniques() -> List[str]:
    return [t["name"] for t in CATALOG]


def get_technique(name: str) -> Dict[str, Any]:
    for t in CATALOG:
        if t["name"] == name:
            return t
    raise KeyError(f"Technique '{name}' not found")


def list_categories() -> List[str]:
    return sorted(set(t["category"] for t in CATALOG))


def list_by_category(category: str) -> List[Dict[str, Any]]:
    return [t for t in CATALOG if t["category"] == category]


def get_applicable_techniques(dataset: str) -> List[Dict[str, Any]]:
    return [t for t in CATALOG if dataset in t.get("applicable", [])]


def is_augment(name: str) -> bool:
    tech = next((t for t in CATALOG if t["name"] == name), None)
    return tech is not None and tech.get("category") == "augmentation"


def suggest_pipeline(dataset: str, categories: List[str] = None) -> List[str]:
    techs = get_applicable_techniques(dataset)
    if categories:
        techs = [t for t in techs if t["category"] in categories]
    return [t["name"] for t in techs if t.get("implemented") and t["category"] != "augmentation"]


def suggest_augment(dataset: str) -> List[str]:
    techs = get_applicable_techniques(dataset)
    return [t["name"] for t in techs if t.get("implemented") and t["category"] == "augmentation"]
