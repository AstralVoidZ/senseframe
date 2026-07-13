"""
探索状态管理：让 Agent 看见完整搜索空间并引导下一步探索。

设计理念（RFC-002 原则 5）：
- 探索状态可见，Agent 看见完整搜索空间而非盲选
- 探索历史可回溯，避免重复探索
- 基于历史 + 兼容性矩阵推荐未探索的有效组合

控制论视角：RFC-002 指出"当前反馈回路是断裂的（开环），RFC-002 要求闭环"。
本模块是闭环的核心——把 eval 结果反馈给 Agent，引导 Agent 调整策略。

使用方式：
    from senseframe.exploration import ExplorationTracker

    tracker = ExplorationTracker()
    tracker.add_trial(strategy={"loss": "focal", "lr": 0.001}, result={"val_accuracy": 0.85})
    tracker.add_trial(strategy={"loss": "cross_entropy", "lr": 0.01}, result={"val_accuracy": 0.80})

    # 查询已探索
    trials = tracker.list_trials()

    # 推荐下一步
    suggestions = tracker.recommend_next(task_type="classification")
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExplorationTracker:
    """探索状态跟踪器：管理试验历史，引导下一步探索。

    闭环核心：record_trial 记录 → list_trials 回溯 → recommend_next 引导。
    """

    def __init__(self, history: Optional[List[Dict[str, Any]]] = None):
        self.history: List[Dict[str, Any]] = list(history) if history else []
        self.action_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_trial(
        self,
        strategy: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        trial_id: Optional[str] = None,
        parent_trial_id: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录一次探索试验。

        Args:
            strategy: 本次试验策略（如 {"loss": "focal", "lr": 0.001}）
            result: 试验结果（如 {"val_accuracy": 0.85}），None 表示未完成
            trial_id: 试验 ID（None 时自动生成）
            parent_trial_id: 父试验 ID（支持回溯分支）
            feedback: 结构化反馈（RFC-002 阶段 R：闭合探索-反馈回路），
                      形如 {"status": "overfitting", "suggestions": [...]}
                      feedback 会驱动 recommend_next 的优先级排序

        Returns:
            试验 ID
        """
        with self._lock:
            if trial_id is None:
                trial_id = f"trial_{len(self.history):04d}"
            entry = {
                "trial_id": trial_id,
                "parent_trial_id": parent_trial_id,
                "strategy": strategy,
                "result": result,
                "status": "completed" if result is not None else "pending",
                "timestamp": datetime.now().isoformat(),
            }
            if feedback is not None:
                entry["feedback"] = feedback
            self.history.append(entry)
            return trial_id

    def list_trials(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出试验（可按状态过滤）。

        Args:
            status: 过滤状态（completed/pending/failed），None 表示全部

        Returns:
            试验列表
        """
        if status is None:
            return list(self.history)
        return [t for t in self.history if t.get("status") == status]

    def update_trial(
        self,
        trial_id: str,
        result: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """更新 trial 结果（P0.3：SP tell 的公共 API）。

        替代 SP 中 `with tracker._lock` + `tracker.history` 直接改写的 hack，
        维持封装完整性。锁粒度与原 hack 一致（with self._lock）。

        Args:
            trial_id: 试验 ID
            result: 试验结果（如 {"value": 0.85, "intermediate_values": {...}}），
                    None 时不更新
            status: 新状态（completed/failed/pruned），None 时不更新
            feedback: 结构化反馈，None 时不更新

        Raises:
            KeyError: trial_id 不存在
        """
        with self._lock:
            for trial in self.history:
                if trial.get("trial_id") == trial_id:
                    if result is not None:
                        trial["result"] = result
                    if status is not None:
                        trial["status"] = status
                    if feedback is not None:
                        trial["feedback"] = feedback
                    return
            raise KeyError(f"trial not found: {trial_id}")

    def get_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 获取试验。"""
        for t in self.history:
            if t["trial_id"] == trial_id:
                return t
        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """返回 trial 历史的浅拷贝（P0.4：SP 公共 API）。

        替代外部直接访问 tracker.history，维持封装完整性。
        返回浅拷贝避免外部修改影响内部状态。

        Returns:
            trial 历史列表的浅拷贝
        """
        with self._lock:
            return list(self.history)

    def best_trial(self, metric: str = "val_accuracy", mode: str = "max") -> Optional[Dict[str, Any]]:
        """获取指定指标最优的试验。

        Args:
            metric: 指标名（如 "val_accuracy" / "val_loss"）
            mode: "max" 取最大，"min" 取最小

        Returns:
            最优试验，无可用数据时返回 None
        """
        candidates = [
            t for t in self.history
            if t.get("result") and metric in t["result"]
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t["result"][metric]) if mode == "max" \
            else min(candidates, key=lambda t: t["result"][metric])

    def explored_strategies(self) -> List[Dict[str, Any]]:
        """获取已探索的策略列表（去重）。"""
        seen = set()
        result = []
        for t in self.history:
            key = _strategy_key(t["strategy"])
            if key not in seen:
                seen.add(key)
                result.append(t["strategy"])
        return result

    def last_feedback(self) -> Optional[Dict[str, Any]]:
        """获取最近一次完成试验的 feedback（RFC-002 阶段 R）。

        recommend_next 据此调整优先级，闭合"训练→反馈→推荐"回路。
        """
        for t in reversed(self.history):
            if t.get("feedback") is not None:
                return t["feedback"]
        return None

    def recommend_next(
        self,
        task_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """推荐下一步值得探索的策略组合。

        RFC-002 阶段 R：feedback 感知排序。若最近试验有 feedback，
        按其 status（numerical_instability/underfitting/overfitting/converged/success）
        生成定向推荐并置于列表前列，闭合探索-反馈回路。

        Args:
            task_type: 任务类型（用于查询兼容性矩阵）
            top_k: 返回前 K 个推荐

        Returns:
            推荐策略列表，每项含 strategy + reason + priority（可选）
        """
        with self._lock:
            from .core.compatibility import get_compatible_losses, get_compatible_metrics

            explored = {_strategy_key(s) for s in self.explored_strategies()}
            recommendations: List[Dict[str, Any]] = []

            # RFC-002 阶段 R：feedback 驱动的定向推荐（高优先级，置于列表前列）
            feedback = self.last_feedback()
            # P1.7：记录触发推荐的 feedback_trial_id，供 feedback_trace 精确追溯
            feedback_trial_id = None
            if feedback is not None:
                for t in reversed(self.history):
                    if t.get("feedback") is not None:
                        feedback_trial_id = t.get("trial_id")
                        break
                for rec in _feedback_aware_recommendations(feedback, task_type, explored):
                    recommendations.append(rec)

            # 从兼容性矩阵生成候选 loss/metric 组合
            if task_type is not None:
                losses = get_compatible_losses(task_type)
                metrics = get_compatible_metrics(task_type)
                for loss in losses:
                    for metric in metrics:
                        strategy = {"loss": loss, "metric": metric}
                        key = _strategy_key(strategy)
                        if key not in explored:
                            recommendations.append({
                                "strategy": strategy,
                                "reason": f"未探索的兼容组合（task={task_type}）",
                            })

            # 补充：transform pipeline 方向（CSI 场景）
            try:
                from .scenes.wifi_csi.catalog import suggest_pipeline, suggest_augment
                for ds in ("NTU-Fi_HAR", "Widar", "UT_HAR_data"):
                    pipeline = suggest_pipeline(ds)
                    augment = suggest_augment(ds)
                    if pipeline:
                        strategy = {"transform": {"pipeline": pipeline[:3]}, "dataset": ds}
                        key = _strategy_key(strategy)
                        if key not in explored:
                            recommendations.append({
                                "strategy": strategy,
                                "reason": f"未探索的 CSI 信号处理 pipeline（{ds}）",
                            })
                    if augment:
                        strategy = {"transform": {"augment": augment}, "dataset": ds}
                        key = _strategy_key(strategy)
                        if key not in explored:
                            recommendations.append({
                                "strategy": strategy,
                                "reason": f"未探索的 CSI 数据增强组合（{ds}）",
                            })
            except ImportError:
                pass

            # P2: HPO 参数空间候选（未探索的数值超参组合）
            # 放在最后，通过较大 top_k 获取；统一 HPO 数值超参与策略空间搜索视图
            hpo_grid = [
                {"learning_rate": lr, "batch_size": bs}
                for lr in [1e-3, 1e-4, 1e-5]
                for bs in [16, 32, 64]
            ]
            for params in hpo_grid:
                key = _strategy_key(params)
                if key not in explored:
                    recommendations.append({
                        "strategy": params,
                        "reason": "未探索的 HPO 参数组合",
                    })

            # RFC-002 阶段 V：记录推荐到 action_log（status="recommended"，
            # 等待 log_adoption 标记 "adopted"），闭合 feedback → recommended → adopted 链路
            for rec in recommendations[:top_k]:
                rec_id = _strategy_key(rec["strategy"])
                rec["recommendation_id"] = rec_id
                self.action_log.append({
                    "recommendation_id": rec_id,
                    "recommended_strategy": rec["strategy"],
                    "reason": rec.get("reason", ""),
                    "priority": rec.get("priority", "normal"),
                    "status": "recommended",
                    "feedback_trial_id": feedback_trial_id,
                    "timestamp": datetime.now().isoformat(),
                })

            return recommendations[:top_k]

    def coverage(self) -> Dict[str, Any]:
        """探索覆盖率统计。

        Returns:
            含 total/explored/pending/failed/best 的统计字典
        """
        total = len(self.history)
        completed = sum(1 for t in self.history if t.get("status") == "completed")
        pending = sum(1 for t in self.history if t.get("status") == "pending")
        failed = sum(1 for t in self.history if t.get("result") is None and t.get("status") != "pending")
        unique_strategies = len(self.explored_strategies())
        return {
            "total_trials": total,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "unique_strategies": unique_strategies,
        }

    def save(self, path) -> None:
        """持久化探索历史到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"history": self.history, "action_log": self.action_log}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "ExplorationTracker":
        """从 JSON 文件加载探索历史。"""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        tracker = cls(history=data.get("history", []))
        tracker.action_log = data.get("action_log", [])
        return tracker

    def log_adoption(
        self,
        recommendation_id: str,
        actual_strategy: Dict[str, Any],
        reason: str = "",
    ) -> None:
        """记录 Agent 采纳推荐的动作（RFC-002 阶段 V）。

        闭合 feedback → recommended → adopted 的追溯链路。

        Args:
            recommendation_id: 推荐项的标识（用 recommend_next 返回项的 strategy 的 _strategy_key）
            actual_strategy: Agent 实际采用的策略
            reason: 采纳原因（可选）
        """
        with self._lock:
            self.action_log.append({
                "recommendation_id": recommendation_id,
                "actual_strategy": actual_strategy,
                "reason": reason,
                "status": "adopted",
                "timestamp": datetime.now().isoformat(),
            })

    def feedback_trace(self) -> List[Dict[str, Any]]:
        """返回 feedback → recommended → adopted 的追溯链路（RFC-002 阶段 V）。

        每条记录含 feedback_status / recommended_strategy / adopted_strategy / timestamp。
        """
        traces = []
        # 按 recommendation_id 关联 recommended 和 adopted
        adopted_map = {}
        for entry in self.action_log:
            if entry.get("status") == "adopted" or "actual_strategy" in entry:
                adopted_map[entry["recommendation_id"]] = entry

        for entry in self.action_log:
            if entry.get("status") == "recommended":
                rec_id = entry["recommendation_id"]
                adopted = adopted_map.get(rec_id)
                # P1.7：按 feedback_trial_id 精确查找触发此推荐的 feedback
                feedback_status = None
                fb_trial_id = entry.get("feedback_trial_id")
                if fb_trial_id:
                    for t in self.history:
                        if t.get("trial_id") == fb_trial_id and t.get("feedback") is not None:
                            feedback_status = t["feedback"].get("status")
                            break
                else:
                    # 向后兼容：旧 action_log 条目无 feedback_trial_id，回退到最近 feedback
                    for t in reversed(self.history):
                        if t.get("feedback") is not None:
                            feedback_status = t["feedback"].get("status")
                            break
                traces.append({
                    "feedback_status": feedback_status,
                    "recommended_strategy": entry.get("recommended_strategy"),
                    "adopted_strategy": adopted.get("actual_strategy") if adopted else None,
                    "adopted": adopted is not None,
                    "timestamp": entry.get("timestamp"),
                })
        return traces

    def submit_trial(self, strategy: Dict[str, Any], result=None, feedback=None) -> str:
        """s1: 提交试验（线程安全的 add_trial 别名，语义化）。"""
        return self.add_trial(strategy, result=result, feedback=feedback)

    def collect_results(self, trial_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """s1: 批量查询试验结果。"""
        with self._lock:
            return [self.get_trial(tid) for tid in trial_ids]


def _strategy_key(strategy: Dict[str, Any]) -> str:
    """生成策略的规范化键（用于去重比较）。"""
    return json.dumps(strategy, sort_keys=True, ensure_ascii=False)


# ============================================================
# feedback 感知推荐策略（RFC-002 阶段 R：闭合探索-反馈回路）
# ============================================================
# 按 feedback.status 生成定向推荐，优先级从高到低：
# - numerical_instability → 推荐数值稳定的 loss + 降低 lr
# - underfitting          → 推荐更强模型方向 + 提高 lr
# - overfitting           → 推荐数据增强 + 正则化
# - converged             → 推荐未探索的新方向
# - success               → 微调已成功策略
_FEEDBACK_STABLE_LOSSES = ["smooth_l1", "mae", "cross_entropy"]
_FEEDBACK_STRONG_LOSSES = ["focal", "cross_entropy_weighted"]
_FEEDBACK_AUGMENT_DATASETS = ("NTU-Fi_HAR", "Widar", "UT_HAR_data")


def _feedback_aware_recommendations(
    feedback: Dict[str, Any],
    task_type: Optional[str],
    explored: set,
) -> List[Dict[str, Any]]:
    """根据 feedback.status 生成定向推荐（高优先级）。

    返回的推荐已排除 explored 中的策略，每项含 priority 字段标识优先级。
    """
    status = feedback.get("status", "")
    recs: List[Dict[str, Any]] = []

    if status == "numerical_instability":
        # 优先推荐数值稳定的 loss + 降低 lr
        for loss in _FEEDBACK_STABLE_LOSSES:
            strategy = {"loss": loss, "lr_scale": 0.1}
            if _strategy_key(strategy) not in explored:
                recs.append({
                    "strategy": strategy,
                    "reason": "feedback: 数值不稳定 → 推荐稳定 loss + 降低 lr",
                    "priority": "high",
                })
                break
        # 梯度裁剪
        strategy = {"gradient_clip_val": 1.0}
        if _strategy_key(strategy) not in explored:
            recs.append({
                "strategy": strategy,
                "reason": "feedback: 数值不稳定 → 启用梯度裁剪",
                "priority": "high",
            })

    elif status == "underfitting":
        # 优先推荐更强 loss 方向 + 提高 lr
        for loss in _FEEDBACK_STRONG_LOSSES:
            strategy = {"loss": loss, "lr_scale": 2.0}
            if _strategy_key(strategy) not in explored:
                recs.append({
                    "strategy": strategy,
                    "reason": "feedback: 欠拟合 → 更强 loss + 提高 lr",
                    "priority": "high",
                })
                break
        # 增加训练轮数
        strategy = {"epochs_scale": 1.5}
        if _strategy_key(strategy) not in explored:
            recs.append({
                "strategy": strategy,
                "reason": "feedback: 欠拟合 → 增加训练轮数",
                "priority": "medium",
            })

    elif status == "overfitting":
        # 优先推荐数据增强
        try:
            from .scenes.wifi_csi.catalog import suggest_augment
            for ds in _FEEDBACK_AUGMENT_DATASETS:
                augment = suggest_augment(ds)
                if augment:
                    strategy = {"transform": {"augment": augment}, "dataset": ds}
                    if _strategy_key(strategy) not in explored:
                        recs.append({
                            "strategy": strategy,
                            "reason": f"feedback: 过拟合 → 数据增强（{ds}）",
                            "priority": "high",
                        })
                        break
        except ImportError:
            pass
        # 正则化
        for wd in [1e-4, 1e-3]:
            strategy = {"weight_decay": wd}
            if _strategy_key(strategy) not in explored:
                recs.append({
                    "strategy": strategy,
                    "reason": f"feedback: 过拟合 → 增大 weight_decay={wd}",
                    "priority": "medium",
                })
                break
        # dropout
        strategy = {"dropout": 0.5}
        if _strategy_key(strategy) not in explored:
            recs.append({
                "strategy": strategy,
                "reason": "feedback: 过拟合 → 启用 dropout",
                "priority": "medium",
            })

    elif status == "converged":
        # 推荐未探索的新方向（兼容性矩阵已在主流程处理，此处补充 transform pipeline）
        try:
            from .scenes.wifi_csi.catalog import suggest_pipeline
            for ds in _FEEDBACK_AUGMENT_DATASETS:
                pipeline = suggest_pipeline(ds)
                if pipeline:
                    strategy = {"transform": {"pipeline": pipeline[:3]}, "dataset": ds}
                    if _strategy_key(strategy) not in explored:
                        recs.append({
                            "strategy": strategy,
                            "reason": f"feedback: 已收敛 → 探索新 pipeline（{ds}）",
                            "priority": "medium",
                        })
                        break
        except ImportError:
            pass

    elif status == "success":
        # 微调已成功策略（lr 微调）
        for lr_scale in [0.5, 2.0]:
            strategy = {"lr_scale": lr_scale}
            if _strategy_key(strategy) not in explored:
                recs.append({
                    "strategy": strategy,
                    "reason": f"feedback: 成功 → 微调 lr（×{lr_scale}）",
                    "priority": "low",
                })
                break

    return recs


class SearchSpaceMap:
    """搜索空间地图（RFC-002 阶段 P）。

    聚合技术目录 + 兼容性矩阵 + 探索历史，输出完整搜索空间地图。
    让 Agent 看见完整空间而非盲选（RFC-002 原则 5：探索状态可见）。

    Usage:
        from senseframe.exploration import SearchSpaceMap
        m = SearchSpaceMap(tracker)
        map = m.overview(task_type="classification", dataset="NTU-Fi_HAR")
    """

    def __init__(self, tracker: Optional[ExplorationTracker] = None):
        self.tracker = tracker or ExplorationTracker()

    def overview(
        self,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
        scene: Optional[str] = None,
    ) -> Dict[str, Any]:
        """输出完整搜索空间地图。

        Args:
            task_type: 任务类型（过滤兼容策略）
            dataset: 数据集（过滤适用技术）
            scene: 限定场景名（None 时遍历所有有 catalog 的场景）

        Returns:
            含 techniques / compatible_strategies / coverage / recommendations 的地图
        """
        return {
            "techniques": self.techniques_overview(dataset, scene),
            "compatible_strategies": self.compatible_strategies(task_type),
            "exploration_coverage": self.tracker.coverage(),
            "recommendations": self.tracker.recommend_next(task_type),
        }

    def techniques_overview(
        self,
        dataset: Optional[str] = None,
        scene: Optional[str] = None,
    ) -> Dict[str, Any]:
        """技术目录概览（多场景聚合）。

        遍历所有已注册场景的 get_catalog()，聚合技术目录条目。
        支持按 dataset 过滤适用技术，按 scene 限定查询场景。

        Args:
            dataset: 数据集（过滤适用技术）
            scene: 限定场景名（None 时遍历所有有 catalog 的场景）

        Returns:
            含 available / categories / total_techniques 的概览字典
        """
        from .scenes import list_scenes, get_scene

        scenes_to_query = [scene] if scene else list(list_scenes().keys())
        all_categories: Dict[str, Any] = {}
        total = 0

        for scene_name in scenes_to_query:
            try:
                sc = get_scene(scene_name)
                catalog = sc.get_catalog() if hasattr(sc, "get_catalog") else None
                if not catalog:
                    continue
                # 聚合此场景的 catalog
                for entry in catalog:
                    cat = entry.get("category", "other")
                    all_categories.setdefault(cat, []).append({
                        "name": entry["name"],
                        "description": entry.get("description", ""),
                        "implemented": entry.get("implemented", False),
                        "applicable": dataset in entry.get("applicable", []) if dataset else True,
                        "scene": scene_name,
                    })
                    total += 1
            except Exception:
                continue

        return {
            "available": total > 0,
            "categories": all_categories,
            "total_techniques": total,
        }

    def compatible_strategies(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """兼容策略组合（兼容性矩阵）。"""
        try:
            from .core.compatibility import (
                get_compatible_losses,
                get_compatible_metrics,
                get_compatible_activations,
            )
        except ImportError:
            return {"available": False}

        if task_type is None:
            return {"available": True, "note": "指定 task_type 以查询兼容策略"}

        return {
            "available": True,
            "task_type": task_type,
            "losses": get_compatible_losses(task_type),
            "metrics": get_compatible_metrics(task_type),
            "activations": get_compatible_activations(task_type),
        }

    def coverage_report(self) -> Dict[str, Any]:
        """探索覆盖率报告。"""
        cov = self.tracker.coverage()
        try:
            from .scenes.wifi_csi.catalog import list_techniques
            total_techniques = len(list_techniques())
        except ImportError:
            total_techniques = 0

        explored_transforms = sum(
            1 for s in self.tracker.explored_strategies()
            if "transform" in s
        )
        return {
            "trials": cov,
            "transform_techniques_total": total_techniques,
            "transform_strategies_explored": explored_transforms,
        }


__all__ = ["ExplorationTracker", "SearchSpaceMap"]
