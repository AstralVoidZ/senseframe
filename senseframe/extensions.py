"""
RFC Phase F：代码注入 — load_extension API。

Agent 生成 Python 文件（如 my_extension.py），通过
``senseframe.load_extension("my_extension.py")`` 加载，在文件中调用
所有 ``register_*`` API 注册自定义策略（loss/metric/task_type/model/scene/normalization）。

设计决策（RFC §8.2 已决策，RFC-002 阶段 T 增量演进）：
- **RFC-002 阶段 T 起引入沙箱最小化**。``load_extension`` 通过 ``sandbox`` 参数
  控制：``off``（旧行为，直接 exec）/ ``soft``（默认，扫描 + 包装命名空间，
  仅记录 warning）/ ``strict``（high severity 危险调用抛 SecurityError 拒绝加载）。
  静态扫描见 :mod:`senseframe.security`。
- 加载时把 senseframe 公共 API 注入扩展命名空间，扩展文件无需显式 import
  即可直接调用 ``register_loss`` / ``register_task_type`` 等。

用法（Agent 视角）：

.. code-block:: python

    # 1. Agent 生成扩展文件
    ext = '''
    @register_loss("my_focal")
    def _focal(alpha=0.25, gamma=2.0, **kw):
        import torch.nn as nn
        return nn.CrossEntropyLoss(**kw)
    '''
    open("my_ext.py", "w").write(ext)

    # 2. 加载扩展（注册生效）
    import senseframe
    senseframe.load_extension("my_ext.py")

    # 3. 使用注册的策略
    from senseframe import get_loss
    loss = get_loss("my_focal")

扩展文件可访问的注入符号：
- 所有 senseframe 顶层导出（register_loss / register_task_type / register_model 等）
- ``senseframe`` 模块自身（便于 ``senseframe.register_*`` 形式调用）
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union

from .observability import setup_logging

logger = setup_logging()


def _build_extension_namespace() -> Dict[str, Any]:
    """构建扩展文件执行命名空间。

    注入 senseframe 公共 API，使扩展文件无需显式 import 即可调用
    register_* 系列 API。同时注入 ``__name__`` 与 ``__file__`` 供扩展
    文件内 ``__file__`` 相对路径解析使用。
    """
    # 延迟导入避免循环：在函数内 import senseframe 自身
    import senseframe

    ns: Dict[str, Any] = {"__name__": "senseframe_extension", "__builtins__": __builtins__}

    # 注入 senseframe 顶层所有公开符号
    for attr in dir(senseframe):
        if attr.startswith("_"):
            continue
        ns[attr] = getattr(senseframe, attr)

    # 同时注入 senseframe 模块本身，便于扩展用 senseframe.xxx 形式
    ns["senseframe"] = senseframe
    return ns


def load_extension(
    path: Union[str, Path, None] = None,
    *,
    name: Optional[str] = None,
    persist_as_skill: Optional[str] = None,
    skill_description: str = "",
    skill_tags: Optional[list] = None,
    search_skill: Optional[str] = None,
    auto_persist: bool = True,
    sandbox: str = "soft",
) -> ModuleType:
    """加载扩展 Python 文件并执行（RFC Phase F + RFC-002 阶段 H/M/T）。

    直接 exec 加载。扩展文件中可调用所有 ``register_*`` API 注册自定义策略，
    注册结果立即生效。

    RFC-002 阶段 H：支持 persist_as_skill 参数，加载成功后自动入库。
    RFC-002 阶段 M：支持 search_skill 参数，先检索技能库，命中则复用已有技能代码，
                    避免重复生成（Voyager 范式的检索复用闭环）。
    RFC-002 阶段 T：验证即入库 + 沙箱最小化。
                    - auto_persist=True 时，验证通过后自动入库为技能
                      （技能名派生：persist_as_skill > __skill_name__ > 文件 stem）
                    - sandbox 控制沙箱模式：off/soft/strict

    Args:
        path: 扩展文件路径（.py），可为 str/Path。search_skill 命中时可省略
        name: 扩展模块名（用于日志和 ``__name__``），None 时从文件名/技能名派生
        persist_as_skill: 若不为 None，加载成功后将代码持久化为技能入库
        skill_description: 技能描述（persist_as_skill 不为 None 时生效）
        skill_tags: 技能标签列表
        search_skill: 技能检索查询字符串（不为 None 时先检索技能库，命中则复用）
        auto_persist: RFC-002 阶段 T，默认 True。验证通过后自动入库为技能。
                      仅在 persist_as_skill 为 None 且来源为文件时触发。
        sandbox: RFC-002 阶段 T，沙箱模式。默认 "soft"。
                 - "off"：不扫描，直接 exec（旧行为）
                 - "soft"：扫描 + 包装命名空间，发现危险调用记录 warning 但继续
                 - "strict"：扫描发现 high severity 危险调用时抛 SecurityError

    Returns:
        ModuleType: 执行后的命名空间包装为模块对象，调用方可访问
        扩展文件中定义的任意符号。

    Raises:
        FileNotFoundError: 文件不存在且未命中技能
        ValueError: search_skill 未命中且 path 为 None
        SyntaxError: 文件/技能代码语法错误
        SecurityError: sandbox="strict" 且扩展含 high severity 危险调用
        Exception: 执行期间抛出的任意异常（原样向上传播）

    Example:
        >>> import senseframe
        >>> # 首次：生成 + 加载 + 入库
        >>> senseframe.load_extension("my_extension.py", persist_as_skill="my_ext",
        ...                           skill_description="自定义 focal loss")
        >>> # 后续：检索复用（无需重新生成文件）
        >>> senseframe.load_extension(search_skill="focal loss for classification")
        >>> # RFC-002 阶段 T：自动入库 + 沙箱
        >>> senseframe.load_extension("safe_ext.py")  # auto_persist=True, sandbox="soft"
        >>> senseframe.load_extension("dangerous_ext.py", sandbox="strict")  # 抛 SecurityError
    """
    # RFC-002 阶段 M：技能复用通道 — 先检索技能库
    source = None
    ext_name = name
    source_origin = "file"

    if search_skill is not None:
        from .skills import search_skills
        results = search_skills(search_skill, top_k=1)
        if results:
            skill = results[0]
            source = skill.code
            ext_name = ext_name or skill.name
            source_origin = f"skill:{skill.name}"
            logger.info(f"Skill '{skill.name}' found for query '{search_skill}', reusing")
        elif path is None:
            raise ValueError(
                f"No skill found for '{search_skill}' and no path provided. "
                f"Available skills: {search_skills.__module__}"
            )

    if source is None:
        if path is None:
            raise ValueError("Either path or search_skill must be provided")
        file_path = Path(path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Extension file not found: {file_path}")

        # M3 修复：扩展路径白名单校验
        # 通过环境变量 SENSEFRAME_EXTENSION_DIRS 配置允许的扩展目录（冒号分隔）
        # 配置后强制校验 file_path 必须位于其中一个目录内；未配置时记录 warning 但允许
        import os as _os
        allowed_dirs_str = _os.environ.get("SENSEFRAME_EXTENSION_DIRS", "")
        if allowed_dirs_str:
            sep = ";" if _os.name == "nt" else ":"
            allowed_dirs = [Path(d).resolve() for d in allowed_dirs_str.split(sep) if d.strip()]
            if allowed_dirs:
                from .common.path_safe import resolve_under
                in_allowed = False
                for ad in allowed_dirs:
                    try:
                        resolve_under(ad, file_path)
                        in_allowed = True
                        break
                    except ValueError:
                        continue
                if not in_allowed:
                    raise PermissionError(
                        f"Extension file not in allowed dirs (SENSEFRAME_EXTENSION_DIRS): {file_path}. "
                        f"Allowed: {allowed_dirs}"
                    )
        else:
            logger.warning(
                "Loading extension from %s without SENSEFRAME_EXTENSION_DIRS whitelist; "
                "set this env var to restrict extension paths", file_path
            )

        ext_name = ext_name or file_path.stem
        source = file_path.read_text(encoding="utf-8")
        file_path_str = str(file_path)
    else:
        file_path_str = f"<{source_origin}>"

    logger.info(f"Loading extension from {source_origin}: '{ext_name}'")

    # RFC-002 阶段 T：沙箱静态扫描（off 模式跳过）
    if sandbox != "off":
        from .security import check_extension_safety, SecurityError
        is_safe, dangers = check_extension_safety(source, mode=sandbox)
        if sandbox == "strict" and not is_safe:
            raise SecurityError(
                f"Extension '{ext_name}' contains dangerous calls: "
                f"{[(d.name, d.line, d.severity) for d in dangers]}"
            )

    # 构建命名空间，注入 senseframe 公共 API
    ns = _build_extension_namespace()
    ns["__name__"] = ext_name
    ns["__file__"] = file_path_str

    # RFC-002 阶段 T：运行时拦截 — 包装命名空间中的危险函数（off 模式跳过）
    if sandbox != "off":
        from .security import RuntimeGuard
        RuntimeGuard.wrap_namespace(ns, mode=sandbox)

    # exec 加载扩展代码
    cwd = os.getcwd()
    try:
        if path is not None and source_origin == "file":
            os.chdir(Path(path).resolve().parent)
        # 编译 + 执行（分离两步便于定位 SyntaxError 行号）
        code = compile(source, file_path_str, "exec")
        exec(code, ns)
    finally:
        os.chdir(cwd)

    # 包装为模块对象返回，调用方可访问扩展定义的符号
    module = ModuleType(ext_name)
    module.__dict__.update(ns)
    module.__file__ = file_path_str

    # RFC-002 阶段 H：持久化为技能（仅文件来源且未通过技能检索复用时）
    if persist_as_skill is not None and source_origin == "file":
        from .skills import save_skill
        success = save_skill(
            name=persist_as_skill,
            code=source,
            description=skill_description,
            tags=skill_tags or [],
        )
        if success:
            logger.info(f"Extension '{ext_name}' persisted as skill '{persist_as_skill}'")
        else:
            logger.warning(f"Extension '{ext_name}' skill validation failed, not persisted")

    # RFC-002 阶段 T：验证即入库（auto_persist）
    # 仅当 persist_as_skill 为 None（避免与显式入库重复）且来源为文件时触发
    if auto_persist and persist_as_skill is None and source_origin == "file":
        from .skills import save_skill
        # 技能名派生优先级：persist_as_skill（已排除）> __skill_name__ > ext_name
        skill_name = ext_name
        if "__skill_name__" in ns and isinstance(ns["__skill_name__"], str):
            skill_name = ns["__skill_name__"]
        success = save_skill(
            name=skill_name,
            code=source,
            description=skill_description,
            tags=skill_tags or [],
            source_path=file_path_str,  # s2：记录来源扩展文件路径
        )
        if success:
            logger.info(f"Extension '{ext_name}' auto-persisted as skill '{skill_name}'")
        else:
            logger.warning(
                f"Extension '{ext_name}' auto-persist skill validation failed, not persisted"
            )

    logger.info(f"Extension '{ext_name}' loaded successfully")
    return module


__all__ = ["load_extension"]
