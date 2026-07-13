"""RFC-003 OP 持久化后端（P3.4.1）。

将 OP（Orchestration Protocol）从进程内存储升级为可持久化、可对接 K8s。

P3.4.1 提供：
- ``OrchestrationStore``：持久化后端协议（Protocol），定义 save/load/list/delete 四元接口
- ``FileOrchestrationStore``：文件系统默认实现，每个 PipelineRun 序列化为 ``{base_dir}/{run_id}.json``

设计要点：
- 仅依赖 ``orchestration.PipelineRun``（单向导入，不引入循环：orchestration.py 用
  ``TYPE_CHECKING`` 字符串注解引用 ``OrchestrationStore``，运行时不导入本模块）
- ``PipelineRun.to_dict / from_dict`` 完成序列化双方向（P3.4.1 前置依赖）
- 加载失败（文件损坏 / OSError）返回 None 或跳过，不抛异常（与 K8s Controller
  "status subresource 读取失败不应阻断 reconciliation" 同语义）
- ``list_runs`` 按 ``sorted(glob("*.json"))`` 顺序返回，保证多次运行结果稳定可重现

P4 扩展点：
- ``K8sOrchestrationStore``：以 K8s CRD（senseframe.io/v1 PipelineRun）为后端，
  save_run → PUT /apis/senseframe.io/v1/namespaces/{ns}/pipelineruns/{run_id}
  load_run → GET .../pipelineruns/{run_id}
  list_runs → LIST .../pipelineruns
  delete_run → DELETE .../pipelineruns/{run_id}
  （需 kopf / kubernetes-client 依赖 + K8s 集群验证，P3 不实现）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .orchestration import PipelineRun


@runtime_checkable
class OrchestrationStore(Protocol):
    """OP 持久化后端协议（P3.4.1）。

    任何实现以下四个方法的对象均满足此协议：
    - ``save_run(run)``：保存 / 覆盖单个 PipelineRun
    - ``load_run(run_id)``：加载单个 PipelineRun（不存在返回 None）
    - ``list_runs(pipeline_ref=None)``：列出所有 run（可按 pipeline_ref 过滤）
    - ``delete_run(run_id)``：删除单个 PipelineRun（不存在不报错）

    默认实现：``FileOrchestrationStore``（文件系统）。
    P4 扩展：``K8sOrchestrationStore``（K8s CRD 后端）。
    """

    def save_run(self, run: "PipelineRun") -> None: ...

    def load_run(self, run_id: str) -> Optional["PipelineRun"]: ...

    def list_runs(self, pipeline_ref: Optional[str] = None) -> List["PipelineRun"]: ...

    def delete_run(self, run_id: str) -> None: ...


class FileOrchestrationStore:
    """文件系统持久化后端（P3 默认实现，P3.4.1）。

    存储路径：``{base_dir}/{run_id}.json``
    内容：``PipelineRun.to_dict()`` 的 JSON 序列化（indent=2, ensure_ascii=False）

    行为约定：
    - ``save_run`` 自动创建 ``base_dir``（含父目录）
    - ``save_run`` 重复调用覆盖旧文件（幂等）
    - ``load_run`` 文件不存在返回 None；JSON 损坏 / OSError 返回 None（不抛异常）
    - ``list_runs`` 目录不存在返回空 list；按 ``sorted(glob("*.json"))`` 顺序
    - ``delete_run`` 文件不存在不报错（幂等）
    """

    def __init__(self, base_dir: Any):
        """Args:
            base_dir: 存储目录路径（str / Path）。``save_run`` 时自动创建。
        """
        self.base_dir = Path(base_dir)

    def save_run(self, run: "PipelineRun") -> None:
        """保存 / 覆盖单个 PipelineRun（P3.4.1）。

        自动创建 ``base_dir``（含父目录）。重复调用覆盖旧文件。
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{run.run_id}.json"
        path.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_run(self, run_id: str) -> Optional["PipelineRun"]:
        """加载单个 PipelineRun（P3.4.1）。

        文件不存在返回 None；JSON 损坏 / OSError 返回 None（不抛异常）。
        """
        # 延迟导入避免循环依赖（orchestration.py 不导入本模块，
        # 但本模块需要 PipelineRun.from_dict）
        from .orchestration import PipelineRun

        path = self.base_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return PipelineRun.from_dict(data)

    def list_runs(self, pipeline_ref: Optional[str] = None) -> List["PipelineRun"]:
        """列出所有 PipelineRun（P3.4.1）。

        Args:
            pipeline_ref: 可选过滤条件。None 时返回所有 run；
                          非 None 时仅返回 ``run.pipeline_ref == pipeline_ref`` 的 run。

        目录不存在返回空 list。按 ``sorted(glob("*.json"))`` 顺序返回，
        保证多次运行结果稳定可重现。
        """
        from .orchestration import PipelineRun

        if not self.base_dir.exists():
            return []
        runs: List["PipelineRun"] = []
        for p in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                run = PipelineRun.from_dict(data)
            except (json.JSONDecodeError, OSError):
                continue
            if pipeline_ref is None or run.pipeline_ref == pipeline_ref:
                runs.append(run)
        return runs

    def delete_run(self, run_id: str) -> None:
        """删除单个 PipelineRun（P3.4.1）。

        文件不存在不报错（幂等）。
        """
        path = self.base_dir / f"{run_id}.json"
        if path.exists():
            path.unlink()


__all__ = [
    "OrchestrationStore",
    "FileOrchestrationStore",
]
