"""
RFC-004 方案 G：训练产物溯源体系 — Artifact Manifest 契约。

Pipeline 必须在结束时产出 manifest.json，记录所有产物的路径/hash/大小/生产者/内容契约。
这是训练溯源的唯一入口。

架构原则：Artifact Manifest 契约
- 每个产物有明确的逻辑名、路径、kind、生产者 stage、内容 hash、内容契约
- manifest.json 是产物清单的唯一入口，供训练后分析、复现、校验
- 缺失产物（config.yaml / metrics.csv / env_snapshot.json）由对应 stage 显式产出
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional


# ============================================================
# 产物描述符与清单
# ============================================================

@dataclass
class ArtifactDescriptor:
    """单个训练产物的描述符。"""

    name: str                            # 逻辑名（如 "model_weights"）
    path: str                            # 相对 output_dir 的路径
    kind: str                            # "model" / "metrics" / "config" / "log" / "profile" / "feedback" / "metadata"
    # 生产者 stage 名（如 stage_export / postprocess），表示哪个 stage 写入了该文件或生成了该数据。
    # 语义统一为"数据级 producer"：即产出该产物内容的 stage，而非文件级 producer
    # （文件可能由 postprocess 搬运，但内容由 stage_export 生成）。
    producer_stage: str
    content_hash: str                    # SHA256 文件哈希
    size_bytes: int                      # 文件大小（字节）
    content_schema: Dict[str, Any] = field(default_factory=dict)  # 内容契约（字段名/类型）

    # P1-1: 有效的 kind 枚举（与 verify_artifacts 中的 kind 映射对齐）
    VALID_KINDS: ClassVar[set] = {
        "model", "metadata", "log", "metrics", "config", "profile", "feedback",
    }

    def __post_init__(self):
        """P1-1: 字段类型校验，拦截无效 kind 与空 name。"""
        # kind 必须是有效枚举值
        if self.kind not in self.VALID_KINDS:
            raise ValueError(
                f"ArtifactDescriptor.kind must be one of {sorted(self.VALID_KINDS)}, "
                f"got {self.kind!r}"
            )
        # name 必须是非空字符串
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"name must be non-empty str, got {self.name!r}")
        # 注：name==kind 是合法设计（单实例类型直接复用类型名，如
        # name="metrics" kind="metrics"），不发出警告。
        # name 是实例级标识（find_artifact 用），kind 是类型分组标签
        # （list_by_kind 用）；单实例产物 name 复用 kind 名是自然命名习惯。

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactManifest:
    """训练产物清单 — 溯源的唯一入口。

    config_hash 语义（P1 修复）：
        应包含 experiment_config_to_dict(ctx.config) + ctx.resolved 的合并哈希，
        覆盖声明式配置（scene/trainer/hpo/output_dir 等用户声明值）和路由解析后
        实际生效值（device/batch_size/precision/learning_rate 等运行时路由填充值）。
        仅 hash ExperimentConfig 会漏掉 ctx.resolved 中的路由覆盖，导致同 config_hash
        的两次运行实际行为不同（如 device 不同），破坏溯源可比性。
    data_hash 语义（P1 修复）：
        数据集内容哈希，由 stage_load 计算，基于数据文件的 SHA256 + 文件大小 + 修改时间。
        空字符串表示未计算（如 dry-run / stage_load 未接入 compute_data_hash）。
    """

    run_id: str                          # 运行 ID（UUID）
    created_at: str                      # ISO8601 时间戳
    senseframe_version: str              # SenseFrame 版本号
    pipeline_version: str                # Pipeline checkpoint 版本号
    config_hash: str                     # ExperimentConfig + ctx.resolved 合并哈希（见类 docstring）
    data_hash: str                       # 数据集内容哈希（由 stage_load 调用 compute_data_hash 计算，空串=未计算）
    artifacts: List[ArtifactDescriptor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "senseframe_version": self.senseframe_version,
            "pipeline_version": self.pipeline_version,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }

    def save(self, output_dir: Path) -> Path:
        """保存 manifest.json 到 output_dir。

        Returns:
            manifest.json 的完整路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return manifest_path

    @classmethod
    def load(cls, path: Path) -> "ArtifactManifest":
        """从 manifest.json 加载清单。

        Args:
            path: manifest.json 路径，或包含 manifest.json 的目录
        """
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        artifacts = [
            ArtifactDescriptor(**a) for a in data.get("artifacts", [])
        ]
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            senseframe_version=data["senseframe_version"],
            pipeline_version=data["pipeline_version"],
            config_hash=data["config_hash"],
            data_hash=data.get("data_hash", ""),
            artifacts=artifacts,
        )

    def find(self, name: str) -> Optional[ArtifactDescriptor]:
        """按逻辑名查找产物。"""
        for a in self.artifacts:
            if a.name == name:
                return a
        return None

    def list_by_kind(self, kind: str) -> List[ArtifactDescriptor]:
        """按 kind 列出产物。"""
        return [a for a in self.artifacts if a.kind == kind]

    def list_by_producer(self, stage: str) -> List[ArtifactDescriptor]:
        """按生产者 stage 列出产物。"""
        return [a for a in self.artifacts if a.producer_stage == stage]


# ============================================================
# 哈希计算辅助
# ============================================================

def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """计算文件 SHA256 哈希（流式读取，支持大文件）。

    Args:
        path: 文件路径
        chunk_size: 每次读取的字节数（默认 64KB）

    Returns:
        16 进制哈希字符串
    """
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_str(text: str) -> str:
    """计算字符串 SHA256 哈希。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_data_hash(data_root: Path, file_pattern: str = "**/*") -> str:
    """计算数据集内容哈希（P1 新增：供 stage_load 填充 ArtifactManifest.data_hash）。

    性能策略：不读取文件内容做全量 SHA256（大数据集太慢），改为 hash
    "排序后的文件路径 + 文件大小 + 文件 mtime" 的拼接。
    这样可在秒级完成万级文件哈希，且能检测文件新增/删除/重命名/大小变更/
    修改时间变更，满足溯源可比性需求。

    Args:
        data_root: 数据根目录
        file_pattern: 文件匹配模式（默认 "**/*" 递归匹配所有文件）

    Returns:
        16 进制 SHA256 哈希字符串；data_root 不存在或无匹配文件时返回空字符串
    """
    data_root = Path(data_root)
    if not data_root.exists():
        return ""

    # 收集 (相对路径, 文件大小, mtime) 三元组
    entries: List[str] = []
    for f in data_root.glob(file_pattern):
        if not f.is_file():
            continue
        try:
            stat = f.stat()
        except OSError:
            # 文件在遍历过程中被删除/权限不足，跳过
            continue
        rel_path = str(f.relative_to(data_root))
        entries.append(f"{rel_path}|{stat.st_size}|{stat.st_mtime}")

    if not entries:
        return ""

    # 排序确保哈希稳定（不受文件系统遍历顺序影响）
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def verify_artifacts(output_dir: Path) -> Dict[str, bool]:
    """校验 output_dir 中所有产物的 hash，检测是否被篡改。

    安全策略（H2 修复）：
    - desc.path 来自 manifest.json（不可信外部输入）
    - 解析后路径必须位于 output_dir 内，否则标记 False（路径穿越拒绝）
    - 避免绝对路径右操作数被 pathlib 直接采用读取任意文件

    Args:
        output_dir: 训练输出目录（含 manifest.json）

    Returns:
        {产物名: hash 是否匹配}（路径逃逸的产物标记 False）
    """
    from ...common.path_safe import resolve_under

    output_dir = Path(output_dir)
    manifest = ArtifactManifest.load(output_dir)
    result: Dict[str, bool] = {}
    for desc in manifest.artifacts:
        # H2 修复：校验 desc.path 解析后仍在 output_dir 内
        try:
            artifact_path = resolve_under(output_dir, desc.path)
        except ValueError:
            # 路径逃逸 output_dir，标记为校验失败（不读取外部文件）
            result[desc.name] = False
            continue
        if not artifact_path.exists():
            result[desc.name] = False
            continue
        current_hash = sha256_file(artifact_path)
        result[desc.name] = current_hash == desc.content_hash
    return result


def verify_artifacts_recursive(
    output_dir: Path,
    max_depth: int = 3,
) -> Dict[str, Dict[str, bool]]:
    """递归校验 output_dir 及其子目录中所有 manifest.json 的产物 hash。

    P3-4 新增：用于 HPO 多 trial 场景。output_dir/ 下可能有 trial_0/、trial_1/
    等子目录，每个子目录有自己的 manifest.json。单 run 场景请用 verify_artifacts。

    Args:
        output_dir: 根输出目录
        max_depth: 最大递归深度（防止恶意目录结构导致的性能问题）

    Returns:
        {子目录相对路径: {产物名: hash 是否匹配}}
        根目录用 "." 表示
    """
    output_dir = Path(output_dir)
    results: Dict[str, Dict[str, bool]] = {}

    # 校验根目录
    root_manifest = output_dir / "manifest.json"
    if root_manifest.exists():
        results["."] = verify_artifacts(output_dir)

    # 递归发现子目录的 manifest.json
    if max_depth > 0:
        for manifest_path in output_dir.rglob("manifest.json"):
            if manifest_path.parent == output_dir:
                continue  # 跳过根目录（已处理）
            rel_dir = manifest_path.parent.relative_to(output_dir)
            # 深度检查
            if len(rel_dir.parts) > max_depth:
                continue
            try:
                results[str(rel_dir)] = verify_artifacts(manifest_path.parent)
            except Exception:
                results[str(rel_dir)] = {"_error": False}

    return results


__all__ = [
    "ArtifactDescriptor",
    "ArtifactManifest",
    "sha256_file",
    "sha256_str",
    "compute_data_hash",
    "verify_artifacts",
    "verify_artifacts_recursive",
]
