"""
技能库：Agent 生成代码的持久化与检索复用。

设计理念（RFC-002 原则 3）：
- Agent 生成的代码持久化为技能（Voyager 范式）
- 验证通过的技能才入库，保证库质量
- 支持检索复用，避免重复生成
- 技能库是脚手架与外骨骼的桥梁

技能生命周期：
1. Agent 生成代码（通过 load_extension 或 save_skill）
2. 框架验证代码（通过 validator）
3. 验证通过则入库（持久化到磁盘）
4. 后续探索可检索复用（search/get）

存储格式：
- 每个技能一个 .py 文件 + .meta.json 元数据
- 默认存储路径：~/.senseframe/skills/
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 嵌入检索（RFC-002 原则 3：基于嵌入的语义检索）
# ============================================================
# 默认 hash-based 轻量嵌入：字符 n-gram + 词级 hash，无外部依赖
# 可选启用 sentence-transformers（若已安装）以获得更强语义能力
_EMBED_DIM = 256

_ST_MODEL = None
_ST_AVAILABLE = None


def _get_embedder():
    """延迟探测 sentence-transformers 可用性。"""
    global _ST_AVAILABLE, _ST_MODEL
    if _ST_AVAILABLE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            _ST_AVAILABLE = True
        except Exception:
            _ST_AVAILABLE = False
    return _ST_MODEL if _ST_AVAILABLE else None


def _embed_text(text: str, dim: int = _EMBED_DIM) -> List[float]:
    """文本嵌入（RFC-002 原则 3）。

    默认 hash-based 轻量嵌入：字符 3-gram + 词级 hash，L2 归一化。
    若 sentence-transformers 可用则用真正的语义嵌入。
    """
    st = _get_embedder()
    if st is not None:
        vec = st.encode(text, normalize_embeddings=True).tolist()
        return vec

    text = text.lower()
    vec = [0.0] * dim
    # 字符 3-gram（捕捉词形变化，如 classify/classification）
    for i in range(max(0, len(text) - 2)):
        gram = text[i:i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    # 词级 hash（增强完整词匹配，权重更高）
    for word in re.findall(r"\w+", text):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 2.0
    # L2 归一化
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度（向量已归一化时等价于点积）。"""
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    return sum(x * y for x, y in zip(a[:min_len], b[:min_len]))


@dataclass
class Skill:
    """技能：Agent 生成的可复用代码单元。"""
    name: str
    description: str
    code: str
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""
    # 验证状态
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    # RFC-002 阶段 M：依赖追踪（依赖的其他技能名）
    depends_on: List[str] = field(default_factory=list)
    # s2：来源扩展文件路径（auto_persist 时记录，便于追溯）
    source_path: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            code=d.get("code", ""),
            tags=d.get("tags", []),
            version=d.get("version", "1.0.0"),
            created_at=d.get("created_at", ""),
            validated=d.get("validated", False),
            validation_errors=d.get("validation_errors", []),
            depends_on=d.get("depends_on", []),
            source_path=d.get("source_path", ""),
        )


class SkillLibrary:
    """技能库：管理 Agent 生成的技能。

    支持持久化到磁盘、语义检索（基于关键词匹配）、版本管理。

    Usage:
        lib = SkillLibrary()
        lib.register(skill)
        results = lib.search("focal loss for imbalanced classification")
        skill = lib.get("my_focal_loss")
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """初始化技能库。

        Args:
            storage_dir: 存储目录，None 时用默认路径 ~/.senseframe/skills/
        """
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.senseframe/skills")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 内存索引：name -> Skill
        self._skills: Dict[str, Skill] = {}
        # RFC-002 原则 3：嵌入缓存（name -> embedding vector）
        self._embeddings: Dict[str, List[float]] = {}
        # RFC-002 阶段 M：历史版本（name -> [旧版本 Skill 列表]）
        self._versions: Dict[str, List[Skill]] = {}
        self._load_from_disk()

    def register(self, skill: Skill, *, validate: bool = True) -> bool:
        """入库技能。

        Args:
            skill: 技能对象
            validate: 是否验证代码（True 时尝试 exec 验证语法）

        Returns:
            True 入库成功，False 验证失败
        """
        if validate:
            try:
                compile(skill.code, f"<skill:{skill.name}>", "exec")
                skill.validated = True
                skill.validation_errors = []
            except SyntaxError as e:
                skill.validated = False
                skill.validation_errors = [f"语法错误: {e}"]
                return False

        # s2：同名同版本同代码去重（避免重复入库）
        if skill.name in self._skills:
            existing = self._skills[skill.name]
            if (existing.version == skill.version
                    and existing.code == skill.code):
                # 完全相同，跳过入库
                return True
            # 版本或代码不同，保留旧版本
            self._versions.setdefault(skill.name, []).append(self._skills[skill.name])
        self._skills[skill.name] = skill
        # RFC-002 原则 3：计算并缓存嵌入
        self._embeddings[skill.name] = _embed_text(
            f"{skill.name} {skill.description} {' '.join(skill.tags)}"
        )
        self._save_to_disk(skill)
        return True

    def get(self, name: str, version: Optional[str] = None) -> Optional[Skill]:
        """按名获取技能（RFC-002 阶段 M：支持版本回退）。

        Args:
            name: 技能名
            version: 版本号（None 返回最新版本）
        """
        if version is None:
            return self._skills.get(name)
        for skill in self._versions.get(name, []):
            if skill.version == version:
                return skill
        current = self._skills.get(name)
        if current and current.version == version:
            return current
        return None

    def list_skills(self) -> List[str]:
        """列出所有技能名。"""
        return sorted(self._skills.keys())

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Skill, float]]:
        """语义检索技能（带相关度分数，RFC-002 原则 3：基于嵌入的语义检索）。

        默认 hash-based 轻量嵌入（字符 n-gram + 词级 hash），
        若安装 sentence-transformers 则自动启用语义嵌入。
        比关键词匹配更强：能捕捉词形变化（classify/classification）和语义近似。

        Args:
            query: 查询字符串
            top_k: 返回前 K 个最相关技能

        Returns:
            (Skill, score) 元组列表，按 score 降序，仅含 score > 0 的项
        """
        if not self._skills:
            return []
        query_vec = _embed_text(query)
        scored: List[Tuple[float, Skill]] = []
        for name, skill in self._skills.items():
            emb = self._embeddings.get(name)
            if emb is None:
                continue
            score = _cosine_similarity(query_vec, emb)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(skill, score) for score, skill in scored[:top_k]]

    def search(self, query: str, top_k: int = 5) -> List[Skill]:
        """语义检索技能（RFC-002 原则 3：基于嵌入的语义检索）。

        默认 hash-based 轻量嵌入（字符 n-gram + 词级 hash），
        若安装 sentence-transformers 则自动启用语义嵌入。
        比关键词匹配更强：能捕捉词形变化（classify/classification）和语义近似。

        Args:
            query: 查询字符串
            top_k: 返回前 K 个最相关技能

        Returns:
            匹配的技能列表（按相关度降序）
        """
        return [skill for skill, _ in self.search_with_scores(query, top_k)]

    def update(self, skill) -> None:
        """更新已注册技能的属性（如 depends_on）。"""
        if skill.name not in self._skills:
            raise KeyError(f"Skill not found: {skill.name}")
        self._skills[skill.name] = skill

    def remove(self, name: str, *, force: bool = False) -> bool:
        """移除技能（RFC-002 阶段 M：检查依赖）。

        Args:
            name: 技能名
            force: True 时强制删除（忽略依赖），False 时若有依赖则拒绝

        Raises:
            ValueError: 有其他技能依赖此技能且 force=False
        """
        if name not in self._skills:
            return False
        # 检查是否有其他技能依赖此技能
        if not force:
            dependents = [
                n for n, s in self._skills.items()
                if name in s.depends_on
            ]
            if dependents:
                raise ValueError(
                    f"Cannot remove skill '{name}': depended on by {dependents}. "
                    f"Use force=True to remove anyway."
                )
        del self._skills[name]
        self._embeddings.pop(name, None)
        self._versions.pop(name, None)
        # 删除磁盘文件
        skill_file = self.storage_dir / f"{name}.py"
        meta_file = self.storage_dir / f"{name}.meta.json"
        for f in [skill_file, meta_file]:
            if f.exists():
                f.unlink()
        return True

    def _save_to_disk(self, skill: Skill) -> None:
        """持久化技能到磁盘。"""
        # 代码文件
        skill_file = self.storage_dir / f"{skill.name}.py"
        skill_file.write_text(skill.code, encoding="utf-8")
        # 元数据
        meta_file = self.storage_dir / f"{skill.name}.meta.json"
        meta_file.write_text(
            json.dumps(skill.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_from_disk(self) -> None:
        """从磁盘加载所有技能元数据。"""
        for meta_file in self.storage_dir.glob("*.meta.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                skill = Skill.from_dict(data)
                self._skills[skill.name] = skill
                self._embeddings[skill.name] = _embed_text(
                    f"{skill.name} {skill.description} {' '.join(skill.tags)}"
                )
            except (json.JSONDecodeError, KeyError):
                continue


# 全局默认技能库实例
_default_library: Optional[SkillLibrary] = None


def get_skill_library() -> SkillLibrary:
    """获取全局默认技能库实例。"""
    global _default_library
    if _default_library is None:
        _default_library = SkillLibrary()
    return _default_library


def save_skill(
    name: str,
    code: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    source_path: str = "",
    version: Optional[str] = None,
) -> bool:
    """保存代码为技能。

    Agent 生成代码后可直接调用此 API 入库，无需走文件中转。

    Args:
        name: 技能名
        code: 代码内容
        description: 技能描述
        tags: 标签列表
        source_path: 来源扩展文件路径（s2：便于追溯）
        version: 技能版本号（None 时由 Skill 默认值 "1.0.0" 决定）

    Returns:
        True 入库成功，False 验证失败
    """
    lib = get_skill_library()
    kwargs = dict(
        name=name,
        description=description,
        code=code,
        tags=tags or [],
        source_path=source_path,
    )
    if version is not None:
        kwargs["version"] = version
    skill = Skill(**kwargs)
    return lib.register(skill)


def load_skill(name: str, version: Optional[str] = None) -> Optional[Skill]:
    """按名加载技能（RFC-002 阶段 M：支持版本回退）。"""
    return get_skill_library().get(name, version=version)


def search_skills(query: str, top_k: int = 5) -> List[Skill]:
    """检索技能库。"""
    return get_skill_library().search(query, top_k=top_k)


def search_skills_with_scores(
    query: str, top_k: int = 5
) -> List[Tuple[Skill, float]]:
    """检索技能库（带相关度分数）。

    返回 (Skill, score) 元组列表，按 score 降序，仅含 score > 0 的项。
    """
    return get_skill_library().search_with_scores(query, top_k=top_k)


def list_skills() -> List[str]:
    """列出所有技能名。"""
    return get_skill_library().list_skills()


__all__ = [
    "Skill",
    "SkillLibrary",
    "get_skill_library",
    "save_skill",
    "load_skill",
    "search_skills",
    "search_skills_with_scores",
    "list_skills",
]
