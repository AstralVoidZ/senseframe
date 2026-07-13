"""
可观测性工具：日志配置、耗时计时、增量日志持久化。

设计要点：
- 日志走 stderr，不污染 stdout 的 JSON 输出
- Timer 使用 perf_counter 高精度计时
- IncrementalLogWriter 每个 epoch 追加写入 JSONL 并 flush，防崩溃丢失
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# WSL2 检测提示去重标志：进程内只在首次 setup_logging 时打印一次
# 避免 CLI 子命令重复调用 setup_logging 导致 "WSL2 detected" 日志刷屏
_WSL2_WARNED = False

# 修复（5.17）：单次初始化标志——setup_logging 被多个模块独立调用时，
# 首次调用完成配置后，后续调用直接返回已配置的 logger（idempotent），
# 避免重复 close/clear handler 导致 FileHandler 句柄泄露 + 日志丢失。
_LOGGING_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    配置全局日志（单次初始化，重复调用 idempotent）。

    首次调用完成 handler 配置；后续调用直接返回已配置的 logger，
    不再重复 close/clear handler（避免 FileHandler 句柄泄露）。

    Args:
        level: DEBUG/INFO/WARN/ERROR
        log_file: 可选的日志文件路径

    Returns:
        配置好的 senseframe logger
    """
    global _LOGGING_CONFIGURED, _WSL2_WARNED

    logger = logging.getLogger("senseframe")

    # 单次初始化：已配置时直接返回，避免重复 reconfigure
    if _LOGGING_CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # RFC-005：先 close 旧 handler（释放文件句柄），再 clear（避免 FileHandler 句柄泄露）
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 日志走 stderr，stdout 保留给 JSON 输出
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False

    # RFC-004 方案 F：WSL2 环境检测 + 内存回收提示（一次性 info）
    # WSL2 autoMemoryReclaim=Gradual 异步缓慢回收，训练后内存占用可能持续偏高
    # 去重：进程内仅在首次 setup_logging 时打印，避免 CLI 子命令重复调用刷屏
    # P2-1：跨进程去重——CLI 每个子命令是新进程，进程内 _WSL2_WARNED 失效，
    # 用 ~/.senseframe/runtime_state.json 持久化跨进程状态。
    if not _WSL2_WARNED:
        # 函数内导入避免早期模块循环依赖（observability 是基础模块）
        from .common.runtime_state import get_state, set_state
        # 先检查进程内标志（同进程内去重），再检查跨进程状态（跨进程去重）
        if not get_state("wsl2_warning_shown", False):
            try:
                import platform
                if "microsoft" in platform.uname().release.lower():
                    logger.info(
                        "WSL2 detected. If memory is not released after training, "
                        "run 'sudo sh -c \"echo 1 > /proc/sys/vm/drop_caches\"' in WSL, "
                        "or set [wsl2] autoMemoryReclaim=dropcache in %USERPROFILE%\\.wslconfig"
                    )
                    # 提示后立即持久化，避免后续子命令重复打印
                    set_state("wsl2_warning_shown", True)
            except Exception:
                pass
        _WSL2_WARNED = True

    _LOGGING_CONFIGURED = True
    return logger


class Timer:
    """上下文管理器计时器，记录 elapsed 秒。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


class IncrementalLogWriter:
    """
    每个 epoch 结束追加写入 JSONL，防崩溃丢失。

    每次 write() 后立即 flush，保证进程意外终止时已写入的记录不丢失。

    RFC-005：补充上下文管理器协议 + __del__ 兜底，避免外部直接实例化时泄露。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        """追加一条记录到 JSONL 文件。"""
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """关闭文件句柄。"""
        if self._fh and not self._fh.closed:
            self._fh.close()
        self._fh = None

    # RFC-005：上下文管理器协议，支持 `with IncrementalLogWriter(p) as w:`
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        """GC 兜底：对象被回收时关闭文件句柄。"""
        try:
            self.close()
        except Exception:
            pass


class ExplorationDashboard:
    """探索过程可视化仪表盘（RFC-002 阶段 V）。

    .. deprecated:: 0.2.0
        自研渲染层。RFC-003 OBP-4 要求由 Grafana + Vega-Lite 接管渲染。
        保留向后兼容，新项目应使用 ``observability_otel.get_grafana_dashboard_template()`` + Vega-Lite。

    让 Agent 看见探索过程，而非读 JSON 文件。
    支持 text/html/markdown 三格式输出。

    Usage:
        from senseframe.observability import ExplorationDashboard
        from senseframe.exploration import ExplorationTracker

        tracker = ExplorationTracker()
        # ... 记录试验 ...
        dashboard = ExplorationDashboard(tracker)
        print(dashboard.render(format="text"))
    """

    def __init__(self, tracker):
        """Args: tracker: ExplorationTracker 实例"""
        self.tracker = tracker

    def coverage_report(self) -> Dict[str, Any]:
        """搜索空间覆盖率报告。"""
        cov = self.tracker.coverage()
        # 补充最优试验信息
        best = self.tracker.best_trial()
        if best:
            cov["best_trial"] = {
                "trial_id": best.get("trial_id"),
                "strategy": best.get("strategy"),
                "result": best.get("result"),
            }
        # feedback 状态分布
        feedbacks = [t.get("feedback", {}).get("status") for t in self.tracker.history if t.get("feedback")]
        from collections import Counter
        cov["feedback_distribution"] = dict(Counter(feedbacks))
        return cov

    def trial_comparison(self, trial_ids: Optional[List[str]] = None, top_n: int = 5) -> List[Dict[str, Any]]:
        """多 trial 指标对比表。

        Args:
            trial_ids: 指定 trial ID 列表（None 时取 top_n 最优）
            top_n: 未指定 trial_ids 时取前 N 个最优
        """
        if trial_ids:
            trials = [self.tracker.get_trial(tid) for tid in trial_ids]
            trials = [t for t in trials if t is not None]
        else:
            # 取 top_n 最优
            completed = [t for t in self.tracker.history if t.get("result")]
            # 按 val_accuracy 降序
            completed.sort(
                key=lambda t: t["result"].get("val_accuracy", t["result"].get("accuracy", 0)),
                reverse=True,
            )
            trials = completed[:top_n]

        return [
            {
                "trial_id": t.get("trial_id"),
                "strategy": t.get("strategy"),
                "val_accuracy": t.get("result", {}).get("val_accuracy") or t.get("result", {}).get("accuracy"),
                "val_loss": t.get("result", {}).get("val_loss"),
                "feedback": t.get("feedback", {}).get("status") if t.get("feedback") else None,
                "timestamp": t.get("timestamp"),
            }
            for t in trials
        ]

    def feedback_trace(self) -> List[Dict[str, Any]]:
        """feedback → action 追溯链路。"""
        return self.tracker.feedback_trace()

    def render(self, format: str = "text") -> str:
        """渲染仪表盘。

        Args:
            format: "text" / "html" / "markdown"
        """
        if format == "text":
            return self._render_text()
        elif format == "markdown":
            return self._render_markdown()
        elif format == "html":
            return self._render_html()
        else:
            raise ValueError(f"Unknown format: {format}. Supported: text/markdown/html")

    def _render_text(self) -> str:
        """文本格式渲染。"""
        lines = []
        lines.append("=" * 60)
        lines.append("SenseFrame Exploration Dashboard")
        lines.append("=" * 60)

        # 覆盖率
        cov = self.coverage_report()
        lines.append(f"\n[Coverage]")
        lines.append(f"  Total trials:     {cov['total_trials']}")
        lines.append(f"  Completed:        {cov['completed']}")
        lines.append(f"  Pending:         {cov['pending']}")
        lines.append(f"  Failed:          {cov['failed']}")
        lines.append(f"  Unique strategies: {cov['unique_strategies']}")
        if cov.get("best_trial"):
            bt = cov["best_trial"]
            lines.append(f"  Best trial:      {bt['trial_id']} (val_acc={bt['result'].get('val_accuracy', 'N/A')})")
        if cov.get("feedback_distribution"):
            lines.append(f"  Feedback dist:   {cov['feedback_distribution']}")

        # 试验对比
        lines.append(f"\n[Trial Comparison (top 5)]")
        lines.append(f"  {'trial_id':<12} {'val_acc':<10} {'val_loss':<10} {'feedback':<20}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*20}")
        for t in self.trial_comparison(top_n=5):
            va = t.get("val_accuracy")
            va_str = f"{va:.4f}" if va is not None else "N/A"
            vl = t.get("val_loss")
            vl_str = f"{vl:.4f}" if vl is not None else "N/A"
            fb = t.get("feedback") or "-"
            lines.append(f"  {t['trial_id']:<12} {va_str:<10} {vl_str:<10} {fb:<20}")

        # feedback 追溯
        traces = self.feedback_trace()
        if traces:
            lines.append(f"\n[Feedback → Action Trace]")
            for i, tr in enumerate(traces):
                fb = tr.get("feedback_status") or "-"
                adopted = "✓ adopted" if tr.get("adopted") else "✗ not adopted"
                lines.append(f"  [{i+1}] feedback={fb} → {adopted}")
                if tr.get("recommended_strategy"):
                    lines.append(f"      recommended: {tr['recommended_strategy']}")
                if tr.get("adopted_strategy"):
                    lines.append(f"      adopted:     {tr['adopted_strategy']}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _render_markdown(self) -> str:
        """Markdown 格式渲染。"""
        lines = []
        lines.append("# SenseFrame Exploration Dashboard")

        cov = self.coverage_report()
        lines.append(f"\n## Coverage\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total trials | {cov['total_trials']} |")
        lines.append(f"| Completed | {cov['completed']} |")
        lines.append(f"| Pending | {cov['pending']} |")
        lines.append(f"| Failed | {cov['failed']} |")
        lines.append(f"| Unique strategies | {cov['unique_strategies']} |")

        lines.append(f"\n## Trial Comparison (top 5)\n")
        lines.append(f"| trial_id | val_acc | val_loss | feedback |")
        lines.append(f"|----------|---------|----------|----------|")
        for t in self.trial_comparison(top_n=5):
            va = t.get("val_accuracy")
            va_str = f"{va:.4f}" if va is not None else "N/A"
            vl = t.get("val_loss")
            vl_str = f"{vl:.4f}" if vl is not None else "N/A"
            fb = t.get("feedback") or "-"
            lines.append(f"| {t['trial_id']} | {va_str} | {vl_str} | {fb} |")

        traces = self.feedback_trace()
        if traces:
            lines.append(f"\n## Feedback → Action Trace\n")
            for i, tr in enumerate(traces):
                fb = tr.get("feedback_status") or "-"
                adopted = "adopted" if tr.get("adopted") else "not adopted"
                lines.append(f"{i+1}. **feedback**: {fb} → **{adopted}**")
                if tr.get("recommended_strategy"):
                    lines.append(f"   - recommended: `{tr['recommended_strategy']}`")
                if tr.get("adopted_strategy"):
                    lines.append(f"   - adopted: `{tr['adopted_strategy']}`")

        return "\n".join(lines)

    def _render_html(self) -> str:
        """HTML 格式渲染。"""
        cov = self.coverage_report()
        html = ['<div class="senseframe-dashboard">']
        html.append('<h2>SenseFrame Exploration Dashboard</h2>')

        # Coverage
        html.append('<h3>Coverage</h3>')
        html.append('<table class="coverage">')
        for k, v in cov.items():
            if k in ("best_trial", "feedback_distribution"):
                html.append(f'<tr><td>{k}</td><td>{v}</td></tr>')
            else:
                html.append(f'<tr><td>{k}</td><td>{v}</td></tr>')
        html.append('</table>')

        # Trial Comparison
        html.append('<h3>Trial Comparison (top 5)</h3>')
        html.append('<table class="trials">')
        html.append('<tr><th>trial_id</th><th>val_acc</th><th>val_loss</th><th>feedback</th></tr>')
        for t in self.trial_comparison(top_n=5):
            va = t.get("val_accuracy")
            va_str = f"{va:.4f}" if va is not None else "N/A"
            vl = t.get("val_loss")
            vl_str = f"{vl:.4f}" if vl is not None else "N/A"
            fb = t.get("feedback") or "-"
            html.append(f'<tr><td>{t["trial_id"]}</td><td>{va_str}</td><td>{vl_str}</td><td>{fb}</td></tr>')
        html.append('</table>')

        html.append('</div>')
        return "\n".join(html)


class TrainingMonitor:
    """训练过程实时监控（RFC-002 阶段 P2）。

    让 Agent 和用户在训练过程中看到实时指标曲线，而非训练完才读 JSONL。

    Usage:
        from senseframe.observability import TrainingMonitor
        monitor = TrainingMonitor()
        # 训练中每个 epoch end 调用
        monitor.on_epoch_end({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_accuracy": 0.8})
        # 查询当前指标
        print(monitor.current_metrics())
        # 渲染 ASCII 曲线
        print(monitor.render_curve("val_loss"))
    """

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        """记录一个 epoch 结束时的指标。"""
        self.history.append(metrics)

    def current_metrics(self) -> Optional[Dict[str, Any]]:
        """查询当前 epoch 的指标。"""
        if not self.history:
            return None
        return self.history[-1]

    def best_metric(self, name: str, mode: str = "min") -> Optional[float]:
        """查询历史最优指标值。"""
        values = [m.get(name) for m in self.history if m.get(name) is not None]
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def history_df(self) -> List[Dict[str, Any]]:
        """返回指标历史（dict 列表，便于绘图）。"""
        return self.history.copy()

    def render_curve(self, metric_name: str, width: int = 50, height: int = 10) -> str:
        """渲染 ASCII 指标曲线（无外部依赖）。

        Args:
            metric_name: 指标名（如 "val_loss"）
            width: 曲线宽度
            height: 曲线高度
        """
        values = [m.get(metric_name) for m in self.history if m.get(metric_name) is not None]
        if len(values) < 2:
            return f"Not enough data for {metric_name} (need >=2 epochs, got {len(values)})"

        vmin, vmax = min(values), max(values)
        if vmax == vmin:
            vmax = vmin + 1.0  # 避免除零

        lines = []
        lines.append(f"{metric_name} curve ({len(values)} epochs):")
        lines.append(f"  range: [{vmin:.4f}, {vmax:.4f}]")

        # ASCII 曲线：每个 epoch 一个列，高度方向表示值
        for h in range(height, 0, -1):
            threshold = vmin + (vmax - vmin) * (h - 1) / (height - 1) if height > 1 else vmin
            row = []
            for v in values:
                if v >= threshold:
                    row.append("█")
                else:
                    row.append(" ")
            # 只显示前 width 个 epoch
            row_str = "".join(row[:width])
            # 右侧标尺
            if h == height or h == 1 or h == height // 2:
                label_val = vmin + (vmax - vmin) * (h - 1) / (height - 1) if height > 1 else vmin
                lines.append(f"{label_val:.4f} |{row_str}")
            else:
                lines.append(f"       |{row_str}")

        # x 轴
        lines.append(f"       +{'-' * min(len(values), width)}")
        lines.append(f"        epoch 1..{len(values)}")

        return "\n".join(lines)

    def render_text(self) -> str:
        """渲染完整训练状态文本。"""
        lines = []
        lines.append("=" * 60)
        lines.append("Training Monitor")
        lines.append("=" * 60)

        current = self.current_metrics()
        if current:
            lines.append(f"\n[Current Epoch {current.get('epoch', '?')}]")
            for k, v in current.items():
                if k != "epoch" and isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}")

        # 最优指标
        for metric in ["val_loss", "val_accuracy", "val_macro_f1"]:
            best = self.best_metric(metric, mode="max" if "accuracy" in metric or "f1" in metric else "min")
            if best is not None:
                lines.append(f"\n[Best {metric}: {best:.4f}]")

        # 曲线
        for metric in ["train_loss", "val_loss"]:
            curve = self.render_curve(metric)
            if "Not enough data" not in curve:
                lines.append(f"\n[{metric} curve]")
                lines.append(curve)

        lines.append("=" * 60)
        return "\n".join(lines)
