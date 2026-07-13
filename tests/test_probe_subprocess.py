"""Probe 子进程隔离测试。

覆盖子进程隔离方案的关键不变量：

1. probe_worker 模块可独立运行（--help 不报错）
2. _run_probe_in_subprocess 函数存在且可导入
3. _run_probe_in_subprocess 源码含 subprocess.run（子进程启动）
4. _run_probe_in_subprocess 源码含 JSON 解析（结果解析）
5. _run_probe_in_subprocess 源码含超时处理
6. _run_probe_in_subprocess 源码含错误 JSON 解析
7. probe_worker._do_probe 函数存在
8. probe_worker._do_probe 源码含 model.eval()（无副作用测量）
9. probe_worker._do_probe 源码含 torch.no_grad()（无梯度分配）
10. probe_worker._do_probe 源码含 max_memory_allocated（显存测量）
11. probe_worker.main 源码含 JSON stdout 输出
12. probe_worker.main 源码含 error JSON 输出
"""

import inspect
import subprocess
import sys

import pytest


# ============================================================
# 1. probe_worker 模块可独立运行
# ============================================================

class TestProbeWorkerRunnable:
    """probe_worker 应可作为独立模块运行。"""

    def test_probe_worker_help(self):
        """`python -m senseframe.engine.runner.probe_worker --help` 应退出码 0。"""
        result = subprocess.run(
            [sys.executable, "-m", "senseframe.engine.runner.probe_worker", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, \
            f"probe_worker --help 应退出码 0，实际 {result.returncode}。" \
            f"stderr: {result.stderr[:200]}"

    def test_probe_worker_missing_required_args(self):
        """缺少必需参数应退出码非 0（argparse 报错）。"""
        result = subprocess.run(
            [sys.executable, "-m", "senseframe.engine.runner.probe_worker"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, \
            "缺少必需参数应退出码非 0"


# ============================================================
# 2. _run_probe_in_subprocess 函数结构
# ============================================================

class TestRunProbeInSubprocessStructure:
    """_run_probe_in_subprocess 应正确启动子进程并解析结果。"""

    def test_function_exists(self):
        """_run_probe_in_subprocess 应可从 pipeline 导入。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        assert callable(_run_probe_in_subprocess)

    def test_source_contains_subprocess_run(self):
        """源码应含 subprocess.run（启动子进程）。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert "subprocess.run" in source, \
            "_run_probe_in_subprocess 应使用 subprocess.run 启动子进程"

    def test_source_contains_json_loads(self):
        """源码应含 json.loads（解析子进程输出）。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert "json.loads" in source, \
            "_run_probe_in_subprocess 应含 json.loads（解析子进程 JSON 输出）"

    def test_source_contains_timeout(self):
        """源码应含 timeout（子进程超时处理）。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert "timeout" in source, \
            "_run_probe_in_subprocess 应含 timeout（子进程超时处理）"

    def test_source_contains_error_handling(self):
        """源码应含 error 字段检查（子进程内部错误）。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert '"error"' in source or "'error'" in source, \
            "_run_probe_in_subprocess 应检查 error 字段（子进程内部错误）"

    def test_source_contains_probe_worker_module(self):
        """源码应含 probe_worker 模块引用。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert "probe_worker" in source, \
            "_run_probe_in_subprocess 应引用 probe_worker 模块"

    def test_source_contains_temp_file_cleanup(self):
        """源码应含临时文件清理。"""
        from senseframe.engine.runner.pipeline import _run_probe_in_subprocess
        source = inspect.getsource(_run_probe_in_subprocess)
        assert "os.unlink" in source or "os.remove" in source, \
            "_run_probe_in_subprocess 应清理临时文件（params-file）"


# ============================================================
# 3. probe_worker._do_probe 函数结构
# ============================================================

class TestProbeWorkerDoProbeStructure:
    """probe_worker._do_probe 应正确执行显存测量。"""

    def test_function_exists(self):
        """_do_probe 应可从 probe_worker 导入。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        assert callable(_do_probe)

    def test_source_contains_eval(self):
        """源码应含 model.eval()（无副作用测量，BN 不更新）。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "model.eval()" in source, \
            "_do_probe 应含 model.eval()（BN 不更新，无副作用）"

    def test_source_contains_no_grad(self):
        """源码应含 torch.no_grad()（无梯度分配）。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "torch.no_grad()" in source, \
            "_do_probe 应含 torch.no_grad()（无梯度分配）"

    def test_source_contains_max_memory_allocated(self):
        """源码应含 max_memory_allocated（显存测量）。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "max_memory_allocated" in source, \
            "_do_probe 应含 max_memory_allocated（峰值显存测量）"

    def test_source_contains_optimizer_state_calculation(self):
        """源码应含 optimizer state 静态计算。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "optimizer_state" in source, \
            "_do_probe 应含 optimizer state 静态计算"

    def test_source_contains_scene_model_rebuild(self):
        """源码应含 scene.build_model_for_dataset（子进程模型重建）。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "build_model_for_dataset" in source, \
            "_do_probe 应通过 scene.build_model_for_dataset 重建模型"

    def test_source_contains_scene_dataset_load(self):
        """源码应含 scene.load_dataset（子进程数据集加载）。"""
        from senseframe.engine.runner.probe_worker import _do_probe
        source = inspect.getsource(_do_probe)
        assert "load_dataset" in source, \
            "_do_probe 应通过 scene.load_dataset 加载数据集"


# ============================================================
# 4. probe_worker.main 函数结构
# ============================================================

class TestProbeWorkerMainStructure:
    """probe_worker.main 应正确解析参数和输出结果。"""

    def test_function_exists(self):
        """main 应可从 probe_worker 导入。"""
        from senseframe.engine.runner.probe_worker import main
        assert callable(main)

    def test_source_contains_json_output(self):
        """源码应含 json.dumps + print（JSON stdout 输出）。"""
        from senseframe.engine.runner.probe_worker import main
        source = inspect.getsource(main)
        assert "json.dumps" in source, \
            "main 应含 json.dumps（JSON 序列化）"
        assert "print(" in source, \
            "main 应含 print（stdout 输出）"

    def test_source_contains_error_output(self):
        """源码应含 error JSON 输出（异常处理）。"""
        from senseframe.engine.runner.probe_worker import main
        source = inspect.getsource(main)
        assert "error" in source, \
            "main 应含 error 字段输出（异常时输出 error JSON）"

    def test_source_contains_traceback(self):
        """源码应含 traceback 记录（调试辅助）。"""
        from senseframe.engine.runner.probe_worker import main
        source = inspect.getsource(main)
        assert "traceback" in source, \
            "main 应含 traceback 记录（异常调试辅助）"
