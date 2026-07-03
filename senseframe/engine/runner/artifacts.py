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
from typing import Any, Dict, List, Optional


# ============================================================
# 产物描述符与清单
# ============================================================

@dataclass
class ArtifactDescriptor:
    """单个训练产物的描述符。"""

    name: str                            # 逻辑名（如 "model_weights"）
    path: str                            # 相对 output_dir 的路径
    kind: str                            # "model" / "metrics" / "config" / "log" / "profile" / "feedback" / "metadata"
    producer_stage: str                  # 生产者 stage（如 "stage_export"）
    content_hash: str                    # SHA256 文件哈希
    size_bytes: int                      # 文件大小（字节）
    content_schema: Dict[str, Any] = field(default_factory=dict)  # 内容契约（字段名/类型）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactManifest:
    """训练产物清单 — 溯源的唯一入口。"""

    run_id: str                          # 运行 ID（UUID）
    created_at: str                      # ISO8601 时间戳
    senseframe_version: str              # SenseFrame 版本号
    pipeline_version: str                # Pipeline checkpoint 版本号
    config_hash: str                     # ExperimentConfig 的 SHA256
    data_hash: str                       # 数据集内容哈希（可选，空字符串表示未计算）
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


def verify_artifacts(output_dir: Path) -> Dict[str, bool]:
    """校验 output_dir 中所有产物的 hash，检测是否被篡改。

    Args:
        output_dir: 训练输出目录（含 manifest.json）

    Returns:
        {产物名: hash 是否匹配}
    """
    output_dir = Path(output_dir)
    manifest = ArtifactManifest.load(output_dir)
    result: Dict[str, bool] = {}
    for desc in manifest.artifacts:
        artifact_path = output_dir / desc.path
        if not artifact_path.exists():
            result[desc.name] = False
            continue
        current_hash = sha256_file(artifact_path)
        result[desc.name] = current_hash == desc.content_hash
    return result


__all__ = [
    "ArtifactDescriptor",
    "ArtifactManifest",
    "sha256_file",
    "sha256_str",
    "verify_artifacts",
]
