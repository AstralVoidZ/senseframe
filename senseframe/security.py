"""
RFC-002 阶段 T：沙箱最小化 — 验证即入库 + 安全护栏。

提供两层防护：
1. 静态扫描（DangerousCallGuard）：AST 解析扩展源码，识别危险调用
2. 运行时拦截（RuntimeGuard）：替换命名空间中的危险函数为受监控 wrapper

设计原则：
- 最小化：仅拦截不可逆/高风险操作（删除文件、系统命令、任意代码执行）
- 不阻塞合法用途：open(path, "r") 读模式不拦截，subprocess.run/Popen 白名单允许
- 双模式：soft 仅记录 warning，strict 抛 SecurityError 拒绝
- 无外部依赖：仅用标准库 ast/logging/dataclasses
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("senseframe.security")


class SecurityError(Exception):
    """沙箱违规异常。"""
    pass


@dataclass
class DangerousCall:
    """危险调用记录。"""
    node_type: str  # 如 "call"
    name: str       # 如 "os.remove"
    line: int       # 行号
    col: int        # 列号
    severity: str   # "high" / "medium"


# 危险函数清单（完整函数名 -> severity）
_DANGEROUS_FUNCS: Dict[str, str] = {
    "os.remove": "high",
    "os.unlink": "high",
    "os.rmdir": "high",
    "shutil.rmtree": "high",
    "os.system": "high",
}

# subprocess 白名单（常见合法用法）
_SUBPROCESS_WHITELIST = {"subprocess.run", "subprocess.Popen"}

# eval/exec 标记为 medium（可能是合法用途）
_EVAL_EXEC_FUNCS = {"eval": "medium", "exec": "medium"}


def _get_call_name(node: ast.Call) -> str:
    """提取 Call 节点的函数名。

    支持两种形式：
    - ast.Name(id='eval') -> "eval"
    - ast.Attribute(value=ast.Name(id='os'), attr='remove') -> "os.remove"
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        # 递归获取完整路径，如 os.path.join
        parts: List[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return ""


def _get_open_mode(call: ast.Call) -> str:
    """提取 open() 调用的 mode 参数值。默认 "r"。"""
    # open(file, mode, ...)
    if len(call.args) >= 2:
        mode_arg = call.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            return mode_arg.value
    # open(file, mode=...)
    for kw in call.keywords:
        if kw.arg == "mode":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return "r"


class DangerousCallGuard:
    """AST 静态扫描器：识别扩展源码中的危险调用。"""

    def scan(self, source: str) -> List[DangerousCall]:
        """扫描源代码，返回危险调用列表。

        Args:
            source: Python 源代码字符串

        Returns:
            危险调用列表，按 (line, col) 排序
        """
        dangers: List[DangerousCall] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # 语法错误由 exec 阶段处理，这里返回空
            return dangers

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _get_call_name(node)
            if not name:
                continue

            # 1. 危险函数清单（os.remove, shutil.rmtree, os.system 等）
            if name in _DANGEROUS_FUNCS:
                dangers.append(DangerousCall(
                    node_type="call",
                    name=name,
                    line=node.lineno,
                    col=node.col_offset,
                    severity=_DANGEROUS_FUNCS[name],
                ))
                continue

            # 2. subprocess.* （白名单外）
            if name.startswith("subprocess."):
                if name in _SUBPROCESS_WHITELIST:
                    continue  # 白名单允许
                dangers.append(DangerousCall(
                    node_type="call",
                    name=name,
                    line=node.lineno,
                    col=node.col_offset,
                    severity="medium",
                ))
                continue

            # 3. eval / exec
            if name in _EVAL_EXEC_FUNCS:
                dangers.append(DangerousCall(
                    node_type="call",
                    name=name,
                    line=node.lineno,
                    col=node.col_offset,
                    severity=_EVAL_EXEC_FUNCS[name],
                ))
                continue

            # 4. open(path, "w"/"wb"/"a"/"x") 写入模式
            if name == "open":
                mode = _get_open_mode(node)
                # 首字符判断读写：w/a/x 均为写入
                if mode and mode[0] in ("w", "a", "x"):
                    dangers.append(DangerousCall(
                        node_type="call",
                        name=f"open(mode={mode!r})",
                        line=node.lineno,
                        col=node.col_offset,
                        severity="medium",
                    ))
                continue

        dangers.sort(key=lambda d: (d.line, d.col))
        return dangers


class _ModuleProxy:
    """模块代理：拦截危险属性访问。

    避免修改原模块（全局副作用），通过代理在命名空间层面拦截。
    """

    # 各模块的危险属性 -> severity（None 表示白名单）
    _DANGEROUS_ATTRS: Dict[str, Dict[str, Any]] = {
        "os": {"remove": "high", "unlink": "high", "rmdir": "high", "system": "high"},
        "shutil": {"rmtree": "high"},
        "subprocess": {
            "call": "medium",
            "check_call": "medium",
            "check_output": "medium",
            "Popen": None,  # 白名单
            "run": None,    # 白名单
        },
    }

    def __init__(self, mod: Any, mode: str):
        object.__setattr__(self, "_mod", mod)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_mod_name", getattr(mod, "__name__", ""))

    def __getattr__(self, name: str) -> Any:
        mod = object.__getattribute__(self, "_mod")
        mode = object.__getattribute__(self, "_mode")
        mod_name = object.__getattribute__(self, "_mod_name")
        attr = getattr(mod, name)

        dangerous = self._DANGEROUS_ATTRS.get(mod_name, {})
        if name in dangerous:
            severity = dangerous[name]
            if severity is None:
                return attr  # 白名单放行
            msg = f"{mod_name}.{name}() blocked by sandbox (severity={severity})"
            if mode == "strict":
                raise SecurityError(msg)
            logger.warning(msg)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        mod = object.__getattribute__(self, "_mod")
        setattr(mod, name, value)


class RuntimeGuard:
    """运行时拦截器：替换命名空间中的危险函数为受监控 wrapper。

    轻量级方案（不使用 sys.settrace）：
    - 包装 __builtins__ 中的 open/eval/exec（创建副本，避免全局副作用）
    - 若命名空间包含 os/shutil/subprocess 模块，用 _ModuleProxy 包装
    """

    @staticmethod
    def wrap_namespace(ns: Dict[str, Any], mode: str = "soft") -> None:
        """将命名空间中的危险函数替换为 wrapper。

        Args:
            ns: 执行命名空间（会被原地修改）
            mode: "soft" 仅记录 warning，"strict" 抛 SecurityError
        """
        if mode not in ("soft", "strict"):
            return

        # 创建 __builtins__ 的副本（避免修改全局 builtins 模块）
        builtins = ns.get("__builtins__")
        if builtins is not None:
            if isinstance(builtins, dict):
                new_builtins = dict(builtins)
            else:
                # module 对象转 dict
                new_builtins = {
                    k: getattr(builtins, k)
                    for k in dir(builtins)
                    if not k.startswith("__")
                }
                for k in ("__name__", "__doc__", "__package__",
                          "__loader__", "__spec__"):
                    if hasattr(builtins, k):
                        new_builtins[k] = getattr(builtins, k)

            # 包装 open/eval/exec
            for fname, wrapper_factory in (
                ("open", RuntimeGuard._make_safe_open),
                ("eval", RuntimeGuard._make_safe_eval),
                ("exec", RuntimeGuard._make_safe_exec),
            ):
                original = new_builtins.get(fname)
                if original is None or getattr(original, "_senseframe_wrapped", False):
                    continue
                new_builtins[fname] = wrapper_factory(original, mode)

            ns["__builtins__"] = new_builtins

        # 包装命名空间中的 os/shutil/subprocess 模块（若已注入）
        for mod_name in ("os", "shutil", "subprocess"):
            mod = ns.get(mod_name)
            if mod is None or not hasattr(mod, "__name__"):
                continue
            if isinstance(mod, _ModuleProxy):
                continue  # 已包装
            ns[mod_name] = _ModuleProxy(mod, mode)

    @staticmethod
    def _make_safe_open(original_open: Any, mode: str) -> Any:
        """创建受监控的 open wrapper。"""
        def safe_open(file, mode_arg="r", *args, **kwargs):
            if mode_arg and mode_arg[0] in ("w", "a", "x"):
                msg = f"open({file!r}, mode={mode_arg!r}) write mode blocked by sandbox"
                if mode == "strict":
                    raise SecurityError(msg)
                logger.warning(msg)
            return original_open(file, mode_arg, *args, **kwargs)
        safe_open._senseframe_wrapped = True
        return safe_open

    @staticmethod
    def _make_safe_eval(original_eval: Any, mode: str) -> Any:
        """创建受监控的 eval wrapper。"""
        def safe_eval(expr, *args, **kwargs):
            msg = "eval() blocked by sandbox"
            if mode == "strict":
                raise SecurityError(msg)
            logger.warning(msg)
            return original_eval(expr, *args, **kwargs)
        safe_eval._senseframe_wrapped = True
        return safe_eval

    @staticmethod
    def _make_safe_exec(original_exec: Any, mode: str) -> Any:
        """创建受监控的 exec wrapper。"""
        def safe_exec(code, *args, **kwargs):
            msg = "exec() blocked by sandbox"
            if mode == "strict":
                raise SecurityError(msg)
            logger.warning(msg)
            return original_exec(code, *args, **kwargs)
        safe_exec._senseframe_wrapped = True
        return safe_exec


def check_extension_safety(
    source: str, mode: str = "soft"
) -> Tuple[bool, List[DangerousCall]]:
    """便捷函数：检查扩展源码安全性。

    Args:
        source: Python 源代码字符串
        mode: "soft" 仅记录，"strict" 有 high severity 时返回 False

    Returns:
        (是否安全, 危险调用列表)
        - soft 模式总是返回 True（仅记录 warning）
        - strict 模式有 high severity 调用时返回 False
        - off/其他模式不检查，返回 (True, [])
    """
    guard = DangerousCallGuard()
    dangers = guard.scan(source)

    if mode == "soft":
        for d in dangers:
            logger.warning(
                f"Dangerous call in extension: {d.name} at line {d.line}:{d.col} "
                f"(severity={d.severity})"
            )
        return True, dangers
    elif mode == "strict":
        high_dangers = [d for d in dangers if d.severity == "high"]
        for d in dangers:
            log_fn = logger.warning if d.severity == "high" else logger.info
            log_fn(
                f"Dangerous call in extension: {d.name} at line {d.line}:{d.col} "
                f"(severity={d.severity})"
            )
        if high_dangers:
            return False, dangers
        return True, dangers
    else:
        # off 或其他模式：不检查
        return True, []


__all__ = [
    "SecurityError",
    "DangerousCall",
    "DangerousCallGuard",
    "RuntimeGuard",
    "check_extension_safety",
]
