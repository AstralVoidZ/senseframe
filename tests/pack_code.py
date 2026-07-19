"""SenseFrame SKILL 打包与部署工具（重构版）。

将 SenseFrame 打包成完备的 SKILL 包，部署到目标目录，
兼容 opencode / Claude Code / 通用 agents 三种 SKILL 配置格式。

支持全套环境准备：部署代码 + 创建 .venv + 安装依赖 + 解压数据集，开箱即测。
增量部署：基于 .senseframe_deploy.json 中的 env_state 字段，跳过未变化的步骤。

设计原则：
- 代码本体只部署一份（项目根），避免重复
- SKILL.md + reference/ 部署到三个 skills 目录（opencode/Claude/agents）
- 排除开发文件：README.md、tests/、docs/、__pycache__/、*.pyc、.pytest_cache/
- 部分面向对象：DeployContext 封装共享状态，WslExecutor 封装 WSL 操作，
  EnvState 封装增量状态；其余校验/部署/工具函数保持函数式

部署布局：
    <target>/
      senseframe/            # 核心 Python 包（可 import）
      scripts/               # CLI 脚本
      configs/               # 配置示例
      examples/              # 示例代码
      pyproject.toml         # 依赖清单与项目元数据（PEP 621）
      AGENTS.md              # opencode 项目入口（向上遍历发现）
      CLAUDE.md              # Claude Code 兼容入口
      .opencode/skills/senseframe/
        SKILL.md             # opencode 原生 skill
        reference/           # 参考文档（按需加载）
      .claude/skills/senseframe/
        SKILL.md             # Claude Code 兼容 skill
        reference/
      .agents/skills/senseframe/
        SKILL.md             # 通用 agents 兼容 skill
        reference/
      .opencode/commands/    # opencode 命令
      .claude/commands/      # Claude Code 命令
      .agents/commands/      # 通用 agents 命令
      .venv/                 # Python 虚拟环境（--venv/--full 时创建）
      resource/CSI_DATASETS/ # 解压后的数据集（--data/--full 时解压）
      setup.sh               # WSL 内环境准备脚本（--full 时生成）
      .senseframe_deploy.json  # 部署清单（含 env_state，用于增量更新与清理）
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# 配置常量
# ============================================================

# 源目录（SenseFrame 项目根）
SOURCE_DIR = Path(__file__).resolve().parent.parent

# 默认目标目录（../AdvanceTest/AutoML）
DEFAULT_TARGET = SOURCE_DIR.parent / "AdvanceTest" / "AutoML"

# SKILL 名称（必须与 SKILL.md frontmatter name 一致）
SKILL_NAME = "senseframe"

# 三个 skills 部署位置（相对于目标目录）
SKILL_DIRS = [
    ".opencode/skills/senseframe",
    ".claude/skills/senseframe",
    ".agents/skills/senseframe",
]

# 三个 commands 部署位置（相对于目标目录）
COMMAND_DIRS = [
    ".opencode/commands",
    ".claude/commands",
    ".agents/commands",
]

# 项目根部署的目录/文件（代码本体）
ROOT_PAYLOAD_DIRS = ["senseframe", "scripts", "configs", "examples", "schemas"]
ROOT_PAYLOAD_FILES = ["pyproject.toml"]

# skills 目录部署的文件（SKILL.md + reference/）
SKILL_PAYLOAD_FILES = ["SKILL.md"]
SKILL_PAYLOAD_DIRS = ["reference"]

# commands 源目录（项目根下的 commands/）
COMMAND_SOURCE_DIR = "commands"

# 排除模式（不打包开发文件）
EXCLUDE_PATTERNS = [
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache",
    "*.egg-info", ".git", ".gitignore", ".idea", ".vscode",
]

# opencode skill name 正则
SKILL_NAME_REGEX = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# description 长度限制（opencode 规范）
DESCRIPTION_MIN = 1
DESCRIPTION_MAX = 1024

# 部署清单文件名
DEPLOY_MANIFEST = ".senseframe_deploy.json"

# 文本文件扩展名（部署到 WSL/Linux 时需转换行尾）
TEXT_EXTENSIONS = {
    ".py", ".md", ".yaml", ".yml", ".txt", ".json",
    ".sh", ".toml", ".cfg", ".ini", ".xlf", ".csv",
}

# WSL UNC 路径前缀
WSL_PATH_MARKERS = ("wsl.localhost", "wsl$")

# 数据集 zip 默认相对路径与解压目标
DATASET_ZIP_REL = "resource/CSI_DATASETS.zip"
DATASET_DIR_REL = "resource/CSI_DATASETS"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DeployStats:
    """部署统计。"""
    files_copied: int = 0
    files_skipped: int = 0
    files_updated: int = 0
    dirs_created: int = 0
    bytes_copied: int = 0
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  文件复制: {self.files_copied}",
            f"  文件跳过(未变): {self.files_skipped}",
            f"  文件更新: {self.files_updated}",
            f"  目录创建: {self.dirs_created}",
            f"  字节数: {self.bytes_copied:,}",
        ]
        if self.errors:
            lines.append(f"  错误: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """校验结果。"""
    ok: bool
    message: str
    details: list = field(default_factory=list)


@dataclass
class DeployContext:
    """部署上下文：封装跨函数共享的部署参数。

    消除每函数 5-7 个参数的冗长签名，统一为 ctx 一个对象。
    """
    source: Path
    target: Path
    dry_run: bool = False
    force: bool = False
    clean_all: bool = False  # True 时清理含 .venv + resource/
    le_mode: str = "preserve"  # auto 解析后存实际值

    @property
    def is_wsl(self) -> bool:
        return WslExecutor.is_wsl_target(self.target)

    @property
    def wsl_path(self) -> Optional[str]:
        return WslExecutor.unc_to_wsl_path(self.target)


@dataclass
class EnvState:
    """环境准备状态（持久化到 .senseframe_deploy.json 的 env_state 字段）。

    用于增量判断：venv/deps/data 是否可跳过。
    """
    venv_created: bool = False
    venv_python_hash: str = ""        # .venv/bin/python 的 sha256
    requirements_hash: str = ""       # pyproject.toml 的 sha256（字段名保留以兼容已有 .senseframe_deploy.json）
    deps_installed_at: str = ""       # 依赖安装时间（ISO 格式）
    datasets_extracted: bool = False
    datasets_fingerprint: str = ""    # 解压后目录的指纹（顶层文件名 hash）

    @classmethod
    def load(cls, target: Path) -> "EnvState":
        """从部署清单加载状态；清单不存在或字段缺失返回默认值。"""
        manifest_path = target / DEPLOY_MANIFEST
        if not manifest_path.exists():
            return cls()
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            env = data.get("env_state") or {}
            return cls(
                venv_created=bool(env.get("venv_created", False)),
                venv_python_hash=str(env.get("venv_python_hash", "")),
                requirements_hash=str(env.get("requirements_hash", "")),
                deps_installed_at=str(env.get("deps_installed_at", "")),
                datasets_extracted=bool(env.get("datasets_extracted", False)),
                datasets_fingerprint=str(env.get("datasets_fingerprint", "")),
            )
        except Exception:
            return cls()

    def save(self, target: Path, manifest: dict) -> None:
        """将 env_state 写入部署清单（合并到现有 manifest）。"""
        manifest["env_state"] = asdict(self)
        (target / DEPLOY_MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- 增量判断 ----

    def should_skip_venv(self, ctx: "DeployContext", wsl: "WslExecutor") -> bool:
        """venv 是否可跳过创建：标记为已创建 + WSL 内 .venv/bin/python 可执行。"""
        if not self.venv_created or not ctx.is_wsl:
            return False
        return wsl.test_exec(".venv/bin/python", cwd=ctx.wsl_path)

    def should_skip_deps(self, ctx: "DeployContext") -> bool:
        """依赖是否可跳过安装：已安装 + pyproject.toml hash 未变。"""
        if not self.deps_installed_at:
            return False
        req_path = ctx.target / "pyproject.toml"
        if not req_path.exists():
            return False
        current_hash = "sha256:" + file_hash(req_path)
        return current_hash == self.requirements_hash

    def should_skip_data(self, ctx: "DeployContext") -> bool:
        """数据集是否可跳过解压：标记为已解压 + 目标目录存在且非空。"""
        if not self.datasets_extracted:
            return False
        data_dir = ctx.target / DATASET_DIR_REL
        return data_dir.exists() and any(data_dir.iterdir())

    def update_venv(self, ctx: "DeployContext", wsl: "WslExecutor") -> None:
        """标记 venv 已创建（用 python 路径的 sha256 作为指纹）。"""
        self.venv_created = True
        # 用 .venv/bin/python 路径字符串作指纹（实际存在性由 WSL test 保证）
        self.venv_python_hash = "sha256:" + hashlib.sha256(
            b".venv/bin/python").hexdigest()[:16]

    def update_deps(self, ctx: "DeployContext") -> None:
        """标记依赖已安装。"""
        req_path = ctx.target / "pyproject.toml"
        self.requirements_hash = "sha256:" + file_hash(req_path) if req_path.exists() else ""
        self.deps_installed_at = datetime.now().isoformat()

    def update_data(self, ctx: "DeployContext") -> None:
        """标记数据集已解压。"""
        data_dir = ctx.target / DATASET_DIR_REL
        if data_dir.exists():
            # 用顶层子目录名列表的 hash 作为指纹
            tops = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
            self.datasets_fingerprint = "sha256:" + hashlib.sha256(
                "|".join(tops).encode("utf-8")).hexdigest()[:16]
        self.datasets_extracted = True


# ============================================================
# 工具函数（无状态，保持函数式）
# ============================================================

def should_exclude(path: Path) -> bool:
    """检查路径是否匹配排除模式。"""
    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
        if path.is_dir() and name == pattern:
            return True
    return False


def is_text_file(path: Path) -> bool:
    """判断是否为文本文件（按扩展名）。"""
    return path.suffix.lower() in TEXT_EXTENSIONS


def normalize_line_endings(data: bytes, mode: str) -> bytes:
    """归一化行尾：lf / crlf / preserve。"""
    if mode == "preserve":
        return data
    if mode == "lf":
        return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if mode == "crlf":
        # 先统一为 lf，再转为 crlf
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return data.replace(b"\n", b"\r\n")
    return data


def detect_target_line_ending(target: Path) -> str:
    """根据目标路径自动检测行尾模式。

    WSL UNC 路径（\\\\wsl.localhost\\... / \\\\wsl$\\...）→ lf
    其他 → preserve
    """
    path_str = str(target)
    for marker in WSL_PATH_MARKERS:
        if path_str.lower().startswith(f"\\\\{marker}\\"):
            return "lf"
    return "preserve"


def file_hash(path: Path) -> str:
    """计算文件 sha256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def files_equal(src: Path, dst: Path, le_mode: str = "preserve") -> bool:
    """判断两个文件内容是否相同（考虑行尾转换）。

    9P 兼容：跨 WSL UNC 读取可能抛 OSError（符号链接/权限/缓存），
    捕获后视为"不等"，触发重复制（保守策略，保证正确性）。
    """
    if not dst.exists():
        return False
    try:
        if le_mode != "preserve" and is_text_file(src):
            src_data = normalize_line_endings(src.read_bytes(), le_mode)
            dst_data = dst.read_bytes()
            return src_data == dst_data
        if src.stat().st_size != dst.stat().st_size:
            return False
        return file_hash(src) == file_hash(dst)
    except OSError:
        return False


def copy_file(src: Path, dst: Path, le_mode: str = "preserve") -> None:
    """复制单个文件，支持行尾转换，保留 mtime。"""
    import os
    if le_mode != "preserve" and is_text_file(src):
        data = normalize_line_endings(src.read_bytes(), le_mode)
        dst.write_bytes(data)
        try:
            shutil.copystat(src, dst)
        except OSError:
            pass
        os.utime(dst, (src.stat().st_atime, src.stat().st_mtime))
    else:
        shutil.copy2(src, dst)


# ============================================================
# WSL 执行器（OOP：封装路径转换 + 命令执行 + 9P 缓存规避）
# ============================================================

class WslExecutor:
    """WSL 命令执行器。

    封装三个职责：
    1. UNC ↔ WSL 路径转换（静态方法）
    2. wsl.exe 命令执行（实例方法，自动截断长输出）
    3. 9P 缓存规避：用 WSL 内 test 命令检查文件存在性，避免 Windows 端
       Path.exists() 命中 9P 缓存旧状态（wsl.exe 退出后 9P 可能未同步）

    使用方式：
        wsl = WslExecutor(distro="Ubuntu", dry_run=False)
        wsl.run("python3 -m venv .venv", cwd="/home/user/proj")
        if wsl.test_exec(".venv/bin/python", cwd="/home/user/proj"):
            ...
    """

    def __init__(self, distro: str = "Ubuntu", dry_run: bool = False):
        self.distro = distro
        self.dry_run = dry_run

    @staticmethod
    def unc_to_wsl_path(unc_path: Path) -> Optional[str]:
        """将 WSL UNC 路径转为 WSL 内 POSIX 路径。

        \\\\wsl.localhost\\Ubuntu\\home\\user\\proj → /home/user/proj
        \\\\wsl$\\Ubuntu\\home\\user\\proj → /home/user/proj
        非 WSL UNC 路径返回 None。
        """
        path_str = str(unc_path)
        for marker in WSL_PATH_MARKERS:
            prefix = f"\\\\{marker}\\"
            if path_str.lower().startswith(prefix.lower()):
                rest = path_str[len(prefix):]
                parts = rest.split("\\", 1)
                if len(parts) == 2:
                    return "/" + parts[1].replace("\\", "/")
        return None

    @staticmethod
    def is_wsl_target(target: Path) -> bool:
        """判断目标目录是否在 WSL 文件系统中。"""
        return WslExecutor.unc_to_wsl_path(target) is not None

    def run(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        timeout: int = 600,
    ) -> tuple:
        """在 WSL 中执行 shell 命令。

        Args:
            cmd: shell 命令字符串
            cwd: 工作目录（WSL POSIX 路径，None 时用 $HOME）
            timeout: 超时秒数

        Returns:
            (returncode, stdout, stderr)
            dry_run 时返回 (0, "<dry-run>", "")
        """
        if cwd:
            full_cmd = f"cd {cwd} && {cmd}"
        else:
            full_cmd = cmd

        if self.dry_run:
            print(f"    [dry-run] wsl -d {self.distro} -- {full_cmd}")
            return (0, "<dry-run>", "")

        print(f"    $ wsl -d {self.distro} -- {full_cmd}")
        try:
            result = subprocess.run(
                ["wsl.exe", "-d", self.distro, "-e", "bash", "-c", full_cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.stdout:
                lines = result.stdout.rstrip().split("\n")
                if len(lines) > 40:
                    print(f"    [stdout 末 30 行 / 共 {len(lines)} 行]")
                    for line in lines[-30:]:
                        print(f"      {line}")
                else:
                    for line in lines:
                        print(f"      {line}")
            if result.stderr:
                for line in result.stderr.rstrip().split("\n")[-15:]:
                    print(f"    [stderr] {line}")
            return (result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT] 命令超时 ({timeout}s)")
            return (124, "", f"timeout after {timeout}s")
        except FileNotFoundError:
            print(f"    [ERROR] wsl.exe 不可用（非 Windows 或未安装 WSL）")
            return (127, "", "wsl.exe not found")

    def test_file(self, path: str, cwd: Optional[str] = None) -> bool:
        """WSL 内 test -f（规避 9P 缓存）。"""
        rc, _, _ = self.run(f"test -f {path}", cwd=cwd, timeout=10)
        return rc == 0

    def test_exec(self, path: str, cwd: Optional[str] = None) -> bool:
        """WSL 内 test -x（规避 9P 缓存）。"""
        rc, _, _ = self.run(f"test -x {path}", cwd=cwd, timeout=10)
        return rc == 0

    def test_dir(self, path: str, cwd: Optional[str] = None) -> bool:
        """WSL 内 test -d（规避 9P 缓存）。"""
        rc, _, _ = self.run(f"test -d {path}", cwd=cwd, timeout=10)
        return rc == 0


# ============================================================
# 校验函数（无状态，保持函数式）
# ============================================================

def parse_frontmatter(skill_md_path: Path) -> dict:
    """解析 SKILL.md 的 YAML frontmatter。"""
    content = skill_md_path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    fm_text = parts[1].strip()
    result = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def validate_source(source: Path) -> ValidationResult:
    """校验源目录完整性。"""
    details = []
    errors = []

    required = ["senseframe/__init__.py", "SKILL.md", "pyproject.toml"]
    for rel in required:
        if not (source / rel).exists():
            errors.append(f"缺少必需文件: {rel}")

    init_file = source / "senseframe" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        if "__version__" not in content:
            errors.append("senseframe/__init__.py 缺少 __version__")

    skill_md = source / "SKILL.md"
    if skill_md.exists():
        fm = parse_frontmatter(skill_md)
        name = fm.get("name", "")
        desc = fm.get("description", "")

        if not name:
            errors.append("SKILL.md frontmatter 缺少 name")
        elif not SKILL_NAME_REGEX.match(name):
            errors.append(f"SKILL.md name '{name}' 不符合正则 {SKILL_NAME_REGEX.pattern}")
        elif name != SKILL_NAME:
            errors.append(f"SKILL.md name '{name}' 与部署目录名 '{SKILL_NAME}' 不一致")

        if not desc:
            errors.append("SKILL.md frontmatter 缺少 description")
        elif not (DESCRIPTION_MIN <= len(desc) <= DESCRIPTION_MAX):
            errors.append(f"description 长度 {len(desc)} 不在 [{DESCRIPTION_MIN}, {DESCRIPTION_MAX}] 范围")

        details.append(f"SKILL.md frontmatter: name={name}, desc.length={len(desc)}")

    if errors:
        return ValidationResult(ok=False, message="; ".join(errors), details=details)
    return ValidationResult(ok=True, message="源目录校验通过", details=details)


def validate_skill_md_references(skill_md_path: Path, skill_dir: Path) -> ValidationResult:
    """校验 SKILL.md 中的 ./reference/ 相对路径引用是否存在。"""
    content = skill_md_path.read_text(encoding="utf-8")
    ref_pattern = re.compile(r"\]\((\./[^)]+)\)")
    missing = []
    for match in ref_pattern.finditer(content):
        rel_path = match.group(1)
        if rel_path.startswith("./reference/"):
            target = skill_md_path.parent / rel_path
            if not target.exists():
                missing.append(rel_path)
    if missing:
        return ValidationResult(ok=False, message=f"SKILL.md 引用的文件不存在: {missing}")
    return ValidationResult(ok=True, message="SKILL.md 引用校验通过")


def post_deploy_validate(target: Path) -> ValidationResult:
    """部署后校验。"""
    errors = []
    details = []

    senseframe_init = target / "senseframe" / "__init__.py"
    if not senseframe_init.exists():
        errors.append("部署后 senseframe/__init__.py 不存在")
    else:
        details.append("senseframe/__init__.py 存在")

    for skill_rel in SKILL_DIRS:
        skill_md = target / skill_rel / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"部署后 {skill_rel}/SKILL.md 不存在")
        else:
            fm = parse_frontmatter(skill_md)
            if fm.get("name") != SKILL_NAME:
                errors.append(f"{skill_rel}/SKILL.md name 不匹配")
            else:
                details.append(f"{skill_rel}/SKILL.md frontmatter OK")

    for entry in ["AGENTS.md", "CLAUDE.md"]:
        if not (target / entry).exists():
            errors.append(f"部署后 {entry} 不存在")
        else:
            details.append(f"{entry} 存在")

    if not (target / "pyproject.toml").exists():
        errors.append("部署后 pyproject.toml 不存在")

    for skill_rel in SKILL_DIRS:
        skill_md = target / skill_rel / "SKILL.md"
        if skill_md.exists():
            result = validate_skill_md_references(skill_md, target / skill_rel)
            if not result.ok:
                errors.append(f"{skill_rel}: {result.message}")

    cmd_src = SOURCE_DIR / COMMAND_SOURCE_DIR
    if cmd_src.exists():
        expected_cmds = [f.name for f in cmd_src.glob("*.md")]
        for cmd_dir_rel in COMMAND_DIRS:
            cmd_dir = target / cmd_dir_rel
            if not cmd_dir.exists():
                errors.append(f"部署后 {cmd_dir_rel}/ 不存在")
                continue
            for cmd_name in expected_cmds:
                cmd_file = cmd_dir / cmd_name
                if not cmd_file.exists():
                    errors.append(f"部署后 {cmd_dir_rel}/{cmd_name} 不存在")
                else:
                    fm = parse_frontmatter(cmd_file)
                    if not fm.get("description"):
                        errors.append(f"{cmd_dir_rel}/{cmd_name} 缺少 description")
                    else:
                        details.append(f"{cmd_dir_rel}/{cmd_name} frontmatter OK")

    if errors:
        return ValidationResult(ok=False, message="; ".join(errors), details=details)
    return ValidationResult(ok=True, message="部署后校验通过", details=details)


# ============================================================
# 部署函数（函数式，参数改为 ctx: DeployContext）
# ============================================================

def clean_target(ctx: DeployContext) -> None:
    """清理旧的 SenseFrame 部署。

    默认仅清理清单记录的代码文件，保留 .venv 与 resource/。
    ctx.clean_all=True 时额外清理 .venv 与 resource/CSI_DATASETS/。
    """
    manifest = ctx.target / DEPLOY_MANIFEST
    if not manifest.exists():
        print(f"  无部署清单，跳过清理")
        return

    try:
        records = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  部署清单解析失败: {e}，跳过清理")
        return

    # 先删清单记录的文件
    for f in records.get("files", []):
        fp = ctx.target / f
        if fp.exists() and fp.is_file():
            if not ctx.dry_run:
                fp.unlink()
            print(f"  删除文件: {f}")

    # 再删空目录（从深到浅）
    all_dirs = sorted(records.get("dirs", []), key=len, reverse=True)
    for d in all_dirs:
        dp = ctx.target / d
        if dp.exists() and dp.is_dir():
            try:
                if not ctx.dry_run:
                    dp.rmdir()
                print(f"  删除目录: {d}")
            except OSError:
                pass

    # clean_all 模式：额外清理 .venv 与 resource/CSI_DATASETS/
    if ctx.clean_all:
        venv_dir = ctx.target / ".venv"
        if venv_dir.exists():
            if not ctx.dry_run:
                shutil.rmtree(venv_dir, ignore_errors=True)
            print(f"  删除目录: .venv/")

        data_dir = ctx.target / DATASET_DIR_REL
        # 9P 兼容：rmtree 前先扫描并删除所有 symlink（含死链）
        # shutil.rmtree 在 Windows 端通过 9P 访问 WSL 时，对 symlink 处理有缺陷：
        # 死链 symlink 被 os.scandir 当作空目录，但 os.rmdir 删 symlink 失败（需 os.unlink），
        # ignore_errors=True 吞掉错误导致 symlink 残留
        if data_dir.exists() and not ctx.dry_run:
            for item in data_dir.rglob("*"):
                if item.is_symlink():
                    try:
                        item.unlink()
                        print(f"  删除 symlink: {item.relative_to(ctx.target)}")
                    except OSError as e:
                        print(f"  [WARN] 无法删除 symlink {item}: {e}")
            shutil.rmtree(data_dir, ignore_errors=True)
        if data_dir.exists():
            print(f"  删除目录: {DATASET_DIR_REL}/")

    if not ctx.dry_run:
        manifest.unlink()


def deploy_root_payload(ctx: DeployContext, stats: DeployStats) -> list:
    """部署代码本体到目标目录根。"""
    deployed_files = []

    for dir_name in ROOT_PAYLOAD_DIRS:
        src = ctx.source / dir_name
        if not src.exists():
            print(f"  跳过不存在的目录: {dir_name}")
            continue
        dst = ctx.target / dir_name
        if not ctx.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if should_exclude(item):
                continue
            rel = item.relative_to(src)
            dst_item = dst / rel
            if item.is_file():
                if dst_item.exists() and not ctx.force and files_equal(item, dst_item, ctx.le_mode):
                    stats.files_skipped += 1
                    continue
                if not ctx.dry_run:
                    dst_item.parent.mkdir(parents=True, exist_ok=True)
                    copy_file(item, dst_item, ctx.le_mode)
                stats.files_copied += 1
                stats.bytes_copied += item.stat().st_size
                deployed_files.append(str(dst_item.relative_to(ctx.target)))
            elif item.is_dir():
                if not ctx.dry_run:
                    dst_item.mkdir(parents=True, exist_ok=True)

    for file_name in ROOT_PAYLOAD_FILES:
        src = ctx.source / file_name
        if not src.exists():
            print(f"  跳过不存在的文件: {file_name}")
            continue
        dst = ctx.target / file_name
        if dst.exists() and not ctx.force and files_equal(src, dst, ctx.le_mode):
            stats.files_skipped += 1
            continue
        if not ctx.dry_run:
            copy_file(src, dst, ctx.le_mode)
        stats.files_copied += 1
        stats.bytes_copied += src.stat().st_size
        deployed_files.append(file_name)

    return deployed_files


def deploy_skills(ctx: DeployContext, stats: DeployStats) -> tuple:
    """部署 SKILL.md + reference/ 到三个 skills 目录。"""
    deployed_files = []
    deployed_dirs = []

    for skill_rel in SKILL_DIRS:
        skill_dst = ctx.target / skill_rel
        if not ctx.dry_run:
            skill_dst.mkdir(parents=True, exist_ok=True)
        deployed_dirs.append(skill_rel)

        for fname in SKILL_PAYLOAD_FILES:
            src = ctx.source / fname
            if not src.exists():
                continue
            dst = skill_dst / fname
            if dst.exists() and not ctx.force and files_equal(src, dst, ctx.le_mode):
                stats.files_skipped += 1
                continue
            if not ctx.dry_run:
                copy_file(src, dst, ctx.le_mode)
            stats.files_copied += 1
            stats.bytes_copied += src.stat().st_size
            deployed_files.append(str(dst.relative_to(ctx.target)))

        for dname in SKILL_PAYLOAD_DIRS:
            src = ctx.source / dname
            if not src.exists():
                continue
            dst = skill_dst / dname
            if not ctx.dry_run:
                dst.mkdir(parents=True, exist_ok=True)
            deployed_dirs.append(str(dst.relative_to(ctx.target)))
            for item in src.rglob("*"):
                if should_exclude(item):
                    continue
                rel = item.relative_to(src)
                dst_item = dst / rel
                if item.is_file():
                    if dst_item.exists() and not ctx.force and files_equal(item, dst_item, ctx.le_mode):
                        stats.files_skipped += 1
                        continue
                    if not ctx.dry_run:
                        dst_item.parent.mkdir(parents=True, exist_ok=True)
                        copy_file(item, dst_item, ctx.le_mode)
                    stats.files_copied += 1
                    stats.bytes_copied += item.stat().st_size
                    deployed_files.append(str(dst_item.relative_to(ctx.target)))
                elif item.is_dir():
                    if not ctx.dry_run:
                        dst_item.mkdir(parents=True, exist_ok=True)
                    deployed_dirs.append(str(dst_item.relative_to(ctx.target)))

    return deployed_files, deployed_dirs


def deploy_commands(ctx: DeployContext, stats: DeployStats) -> tuple:
    """部署 commands/ 到三个 commands 目录。"""
    deployed_files = []
    deployed_dirs = []

    cmd_src = ctx.source / COMMAND_SOURCE_DIR
    if not cmd_src.exists():
        print(f"  跳过: 源目录无 {COMMAND_SOURCE_DIR}/")
        return deployed_files, deployed_dirs

    cmd_files = sorted(cmd_src.glob("*.md"))
    if not cmd_files:
        print(f"  跳过: {COMMAND_SOURCE_DIR}/ 无 .md 文件")
        return deployed_files, deployed_dirs

    for cmd_dir_rel in COMMAND_DIRS:
        cmd_dst = ctx.target / cmd_dir_rel
        if not ctx.dry_run:
            cmd_dst.mkdir(parents=True, exist_ok=True)
        deployed_dirs.append(cmd_dir_rel)

        for cmd_file in cmd_files:
            dst = cmd_dst / cmd_file.name
            if dst.exists() and not ctx.force and files_equal(cmd_file, dst, ctx.le_mode):
                stats.files_skipped += 1
                continue
            if not ctx.dry_run:
                copy_file(cmd_file, dst, ctx.le_mode)
            stats.files_copied += 1
            stats.bytes_copied += cmd_file.stat().st_size
            deployed_files.append(str(dst.relative_to(ctx.target)))

    return deployed_files, deployed_dirs


def generate_agents_md(ctx: DeployContext) -> str:
    """生成 AGENTS.md（opencode 项目入口）。"""
    content = f"""# AGENTS.md

> 本文件是 AI agent 的项目入口说明。opencode 启动时从当前目录向上遍历查找此文件。

## 项目概述

SenseFrame 测试环境。SenseFrame 是 Agent 驱动的 AutoML 训练框架，
提供可编程原语库 + 执行底座 + 安全护栏。

## 目录结构

```
thepot/
├── AGENTS.md              # 本文件（opencode 入口）
├── CLAUDE.md              # Claude Code 兼容入口
├── senseframe/            # 核心 Python 包（可直接 import）
├── scripts/               # CLI 脚本
├── configs/               # 配置示例
├── examples/              # 示例代码
├── pyproject.toml         # Python 依赖与项目元数据（PEP 621）
├── resource/              # 资源输入
│   ├── CSI_DATASETS.zip   # 原始数据集 zip
│   └── CSI_DATASETS/      # 解压后的数据集
├── .venv/                 # Python 虚拟环境
├── .opencode/skills/senseframe/   # opencode skill
├── .claude/skills/senseframe/     # Claude Code skill
└── .agents/skills/senseframe/     # 通用 agents skill
```

## 使用 SenseFrame

### 1. 安装依赖

推荐使用 .venv 虚拟环境（已由 pack_code.py --venv 创建）：

```bash
source .venv/bin/activate
pip install -e '.[eeg,radio,dev]'
```

或直接执行环境准备脚本：`bash setup.sh`

### 2. 加载 SKILL

opencode/Claude Code 会自动发现 `senseframe` skill。AI 可通过 `skill` 工具加载：

```
skill({{ name: "senseframe" }})
```

### 3. 快速验证

```bash
python -c "import senseframe; print(senseframe.__version__)"
python -m senseframe.cli probe
python -m senseframe.cli list-models
```

### 4. 数据集路径

WiFi CSI 数据集已部署到 `resource/CSI_DATASETS/`。配置中 `data_root` 可指向此目录：

```yaml
data_root: ./resource/CSI_DATASETS
```

### 5. 运行实验

```bash
python -m senseframe.cli experiment --config configs/exp.yaml --dry-run
```

## 技术约束

- Python 3.x + PyTorch + PyTorch Lightning
- 数据集格式：WiFi CSI（NTU-Fi_HAR / UT_HAR_data / Widar3.0 等）
- 错误处理：基于 `error_code` 程序化决策，不字符串匹配
- 资源路由：CPU 选小模型，GPU 启用 mixed_precision

## 部署信息

- 部署时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 部署工具: pack_code.py
- SKILL 名称: {SKILL_NAME}
"""
    if not ctx.dry_run:
        data = normalize_line_endings(content.encode("utf-8"), ctx.le_mode)
        (ctx.target / "AGENTS.md").write_bytes(data)
    return "AGENTS.md"


def generate_claude_md(ctx: DeployContext) -> str:
    """生成 CLAUDE.md（Claude Code 兼容入口，内容同 AGENTS.md）。"""
    agents_md = ctx.target / "AGENTS.md"
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8")
        content = content.replace("# AGENTS.md", "# CLAUDE.md")
    else:
        content = f"# CLAUDE.md\n\nSenseFrame 测试环境。详见 AGENTS.md。\n"
    if not ctx.dry_run:
        data = normalize_line_endings(content.encode("utf-8"), ctx.le_mode)
        (ctx.target / "CLAUDE.md").write_bytes(data)
    return "CLAUDE.md"


def write_manifest(
    ctx: DeployContext,
    files: list,
    dirs: list,
    stats: DeployStats,
    env_state: Optional[EnvState] = None,
) -> dict:
    """写入部署清单（含 env_state）。返回 manifest dict。"""
    manifest = {
        "skill_name": SKILL_NAME,
        "deployed_at": datetime.now().isoformat(),
        "source": str(ctx.source),
        "files": sorted(set(files)),
        "dirs": sorted(set(dirs)),
        "stats": {
            "files_copied": stats.files_copied,
            "files_skipped": stats.files_skipped,
            "files_updated": stats.files_updated,
            "bytes_copied": stats.bytes_copied,
        },
        "env_state": asdict(env_state) if env_state else {},
    }
    if not ctx.dry_run:
        (ctx.target / DEPLOY_MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return manifest


# ============================================================
# 环境准备（venv + 依赖 + 数据集；增量部署）
# ============================================================

def setup_venv(ctx: DeployContext, wsl: WslExecutor, env_state: EnvState) -> bool:
    """在目标目录创建 .venv 虚拟环境（WSL 内 python3 -m venv）。

    增量：env_state 标记为已创建 + WSL 内 .venv/bin/python 可执行 → 跳过。

    Args:
        ctx: 部署上下文
        wsl: WSL 执行器
        env_state: 环境状态（增量判断 + 状态更新）

    Returns:
        True 成功/跳过，False 失败
    """
    if not ctx.is_wsl:
        print(f"  [SKIP] 非 WSL 目标，跳过 venv 创建（请在目标平台手动创建）")
        return False

    # 增量判断
    if not ctx.force and env_state.should_skip_venv(ctx, wsl):
        print(f"  [OK] .venv 已存在且可用，跳过创建（增量）")
        return True

    print(f"  → 在 WSL 中创建 .venv（distro={wsl.distro}）...")
    rc, _, _ = wsl.run("python3 -m venv .venv", cwd=ctx.wsl_path, timeout=120)
    if rc != 0:
        print(f"  [FAIL] venv 创建失败 (rc={rc})")
        return False

    # 9P 缓存延迟修复：用 WSL 内 test 验证创建结果
    if not wsl.test_exec(".venv/bin/python", cwd=ctx.wsl_path):
        print(f"  [FAIL] venv 创建后 .venv/bin/python 不存在")
        return False

    env_state.update_venv(ctx, wsl)
    print(f"  [OK] .venv 创建成功")
    return True


def install_requirements(
    ctx: DeployContext,
    wsl: WslExecutor,
    env_state: EnvState,
    timeout: int = 1800,
) -> bool:
    """在 .venv 中安装 pyproject.toml 依赖。

    增量：env_state 标记已安装 + pyproject.toml hash 未变 → 跳过。

    Args:
        ctx: 部署上下文
        wsl: WSL 执行器
        env_state: 环境状态
        timeout: pip install 超时秒数

    Returns:
        True 成功/跳过，False 失败
    """
    if not ctx.is_wsl:
        print(f"  [SKIP] 非 WSL 目标，跳过依赖安装")
        return False

    # 9P 缓存延迟修复：用 WSL 内 test 检查 venv 是否存在
    if not wsl.test_exec(".venv/bin/python", cwd=ctx.wsl_path):
        print(f"  [FAIL] .venv 不存在，请先 --venv")
        return False

    # 增量判断
    if not ctx.force and env_state.should_skip_deps(ctx):
        print(f"  [OK] 依赖已安装且 pyproject.toml 未变，跳过安装（增量）")
        return True

    # PEP 621 安装：pip install . 安装核心依赖；可选项通过 extras 按需安装
    # 默认装 [eeg,radio,dev]：覆盖 SenseFrame 部署场景的常用依赖
    print(f"  → 在 .venv 中安装依赖（pip install -e .[eeg,radio,dev]）...")
    print(f"    超时 {timeout}s（PyTorch 等大包下载可能较慢）")
    rc, _, _ = wsl.run(
        ".venv/bin/pip install --upgrade pip && "
        ".venv/bin/pip install -e '.[eeg,radio,dev]'",
        cwd=ctx.wsl_path, timeout=timeout,
    )
    if rc != 0:
        print(f"  [FAIL] 依赖安装失败 (rc={rc})")
        print(f"  提示: 可在 WSL 内手动排查: cd {ctx.wsl_path} && .venv/bin/pip install -e '.[eeg,radio,dev]'")
        return False

    env_state.update_deps(ctx)
    print(f"  [OK] 依赖安装完成")
    return True


def verify_senseframe_import(ctx: DeployContext, wsl: WslExecutor) -> bool:
    """验证 senseframe 在 .venv 中可正常 import。"""
    if not ctx.is_wsl:
        print(f"  [SKIP] 非 WSL 目标，跳过 import 验证")
        return False

    print(f"  → 验证 senseframe 可 import...")
    rc, _, _ = wsl.run(
        ".venv/bin/python -c 'import senseframe; "
        "print(\"version:\", senseframe.__version__)'",
        cwd=ctx.wsl_path, timeout=60,
    )
    if rc != 0:
        print(f"  [FAIL] senseframe import 失败 (rc={rc})")
        return False
    print(f"  [OK] senseframe import 成功")
    return True


def extract_datasets(ctx: DeployContext, env_state: EnvState) -> bool:
    """解压数据集 zip 到 target/resource/CSI_DATASETS/，递归解压子 zip。

    增量：env_state 标记已解压 + 目标目录非空 → 跳过。

    Args:
        ctx: 部署上下文
        env_state: 环境状态

    Returns:
        True 成功/跳过，False 失败
    """
    zip_path = ctx.target / DATASET_ZIP_REL
    extracted_dir = ctx.target / DATASET_DIR_REL

    # 增量判断
    if not ctx.force and env_state.should_skip_data(ctx):
        print(f"  [OK] {DATASET_DIR_REL}/ 已解压且未变，跳过（增量）")
        return True

    if not zip_path.exists():
        print(f"  [WARN] {DATASET_ZIP_REL} 不存在，跳过解压")
        print(f"  提示: 请手动放置 {DATASET_ZIP_REL} 到目标目录后重新运行 --data")
        return False

    print(f"  → 解压 {DATASET_ZIP_REL} → {DATASET_DIR_REL}/...")
    if ctx.dry_run:
        print(f"    [dry-run] 将解压 {zip_path.stat().st_size:,} 字节")
        return True

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            top_names = sorted({n.split("/")[0] for n in zf.namelist() if n})
            print(f"    zip 内顶层: {top_names}")
            extracted_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(extracted_dir)
        print(f"  [OK] 顶层解压完成 → {extracted_dir}")

        # 递归解压子 zip
        sub_zips = list(extracted_dir.rglob("*.zip"))
        if sub_zips:
            print(f"  → 递归解压 {len(sub_zips)} 个子 zip...")
            for sub_zip in sub_zips:
                # 扁平化解压：直接解压到 sub_zip 所在目录，不创建额外同名子目录
                # 旧逻辑 sub_dir = sub_zip.parent / sub_zip.stem 会创建 Widardata/，
                # 但 zip 内顶层已是 Widardata/，导致 Widardata/Widardata/ 嵌套
                target_dir = sub_zip.parent
                print(f"    → {sub_zip.name} → {target_dir.name}/ (扁平化解压)")
                try:
                    with zipfile.ZipFile(sub_zip, "r") as szf:
                        szf.extractall(target_dir)
                    sub_zip.unlink()
                    print(f"    [OK] {sub_zip.name} 解压完成 + 删除 zip")
                except Exception as e:
                    print(f"    [WARN] {sub_zip.name} 解压失败: {e}（保留 zip）")
            print(f"  [OK] 子 zip 递归解压完成")
        else:
            print(f"  [OK] 无子 zip，跳过递归解压")

        # 扁平化同名嵌套目录（如 UT_HAR/UT_HAR/ → UT_HAR/）
        # 子 zip 内顶层目录名与外层目录名相同时产生的嵌套
        _flatten_nested_same_name_dirs(extracted_dir)

        # 清理残留空目录（如 zip 内 __MACOSX / 空的 Data 目录）
        _cleanup_empty_dirs(extracted_dir)

        if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
            print(f"  [FAIL] 解压后 {DATASET_DIR_REL}/ 仍为空")
            return False

        env_state.update_data(ctx)
        return True
    except Exception as e:
        print(f"  [FAIL] 解压失败: {e}")
        return False


def _flatten_nested_same_name_dirs(root: Path) -> None:
    """扁平化同名嵌套目录（如 UT_HAR/UT_HAR/ → UT_HAR/）。

    子 zip 解压后，若 extracted_dir/<name>/ 下存在同名子目录 <name>/，
    将内层内容上移一层，删除空的内层目录。

    Args:
        root: 解压根目录（如 resource/CSI_DATASETS/）
    """
    for child in list(root.iterdir()):
        if not child.is_dir():
            continue
        inner = child / child.name
        if inner.exists() and inner.is_dir():
            # 检测到同名嵌套：UT_HAR/UT_HAR/
            print(f"    [扁平化] {child.name}/{child.name}/ → {child.name}/")
            # 将 inner 内所有内容移动到 child
            for item in inner.iterdir():
                target = child / item.name
                if target.exists():
                    # 冲突时跳过（保留原文件），记录警告
                    print(f"    [WARN] 冲突跳过: {child.name}/{item.name} 已存在")
                    continue
                item.rename(target)
            try:
                inner.rmdir()
            except OSError:
                # 内层目录非空（有子目录残留），递归处理
                _flatten_nested_same_name_dirs(inner)
                try:
                    inner.rmdir()
                except OSError:
                    print(f"    [WARN] 无法删除嵌套目录: {inner}")


def _cleanup_empty_dirs(root: Path) -> None:
    """递归清理空目录（含 __MACOSX / 空的占位目录）。

    Args:
        root: 解压根目录
    """
    # 先删除 __MACOSX（macOS 打包元数据，无用）
    macosx = root / "__MACOSX"
    if macosx.exists():
        import shutil
        shutil.rmtree(macosx, ignore_errors=True)
        print(f"    [清理] 删除 __MACOSX/")

    # 先删除死链 symlink（Path.is_dir() 对死链返回 False，会被空目录清理逻辑跳过）
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            try:
                path.unlink()
                print(f"    [清理] 删除死链 symlink: {path.name}")
            except OSError:
                pass

    # 自底向上删除空目录
    changed = True
    while changed:
        changed = False
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                    changed = True
                except OSError:
                    pass


def generate_setup_sh(ctx: DeployContext) -> str:
    """生成 setup.sh 入口脚本，方便用户在 WSL 内重新执行环境准备。"""
    wsl_path = ctx.wsl_path or "$PWD"
    content = f"""#!/usr/bin/env bash
# SenseFrame 环境准备脚本（由 pack_code.py 生成）
# 在 WSL 内执行: bash setup.sh
set -e

echo "=== SenseFrame 环境准备 ==="
echo "工作目录: {wsl_path}"
cd "{wsl_path}"

# 1. 创建 .venv（如不存在）
if [ ! -x ".venv/bin/python" ]; then
    echo "[1/4] 创建 .venv..."
    python3 -m venv .venv
else
    echo "[1/4] .venv 已存在，跳过创建"
fi

# 2. 安装依赖
echo "[2/4] 安装依赖..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[eeg,radio,dev]'

# 3. 解压数据集（如 zip 存在且未解压）
if [ -f "resource/CSI_DATASETS.zip" ] && [ ! -d "resource/CSI_DATASETS" ]; then
    echo "[3/4] 解压数据集..."
    mkdir -p resource/CSI_DATASETS
    cd resource
    python3 -c "import zipfile; zipfile.ZipFile('CSI_DATASETS.zip').extractall('CSI_DATASETS')"
    cd CSI_DATASETS
    for sub in *.zip; do
        [ -f "$sub" ] || continue
        echo "  → 解压 $sub (扁平化)"
        python3 -c "import zipfile; zipfile.ZipFile('$sub').extractall('.')"
        rm -f "$sub"
    done
    # 扁平化同名嵌套目录（如 UT_HAR/UT_HAR/ → UT_HAR/）
    python3 -c "
import os, shutil
for d in os.listdir('.'):
    if os.path.isdir(d) and os.path.isdir(os.path.join(d, d)):
        inner = os.path.join(d, d)
        for item in os.listdir(inner):
            src = os.path.join(inner, item)
            dst = os.path.join(d, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            os.rmdir(inner)
        except OSError:
            pass
# 清理空目录
import pathlib
changed = True
while changed:
    changed = False
    for p in sorted(pathlib.Path('.').rglob('*'), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
                changed = True
            except OSError:
                pass
"
    cd "{wsl_path}"
elif [ -d "resource/CSI_DATASETS" ]; then
    echo "[3/4] resource/CSI_DATASETS/ 已存在，跳过解压"
else
    echo "[3/4] 无 resource/CSI_DATASETS.zip，跳过"
fi

# 4. 验证
echo "[4/4] 验证 import..."
.venv/bin/python -c "import senseframe; print('version:', senseframe.__version__)"

echo ""
echo "=== 环境准备完成 ==="
echo "激活 venv:  source .venv/bin/activate"
echo "快速验证:   python -m senseframe.cli probe"
echo "运行实验:   python -m senseframe.cli experiment --config configs/exp.yaml --dry-run"
"""
    fname = "setup.sh"
    if not ctx.dry_run:
        data = normalize_line_endings(
            content.encode("utf-8"),
            ctx.le_mode if ctx.le_mode != "preserve" else "lf",
        )
        (ctx.target / fname).write_bytes(data)
    return fname


# ============================================================
# 主流程
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（简化参数名，不向后兼容）。"""
    parser = argparse.ArgumentParser(
        description="将 SenseFrame 打包为 SKILL 包并部署到目标目录（支持全套环境准备 + 增量部署）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tests/pack_code.py                              # 部署到默认目标
  python tests/pack_code.py --dry                        # 预览不执行
  python tests/pack_code.py --force                      # 强制覆盖所有文件（忽略增量）
  python tests/pack_code.py --clean                      # 清理代码部署（保留 .venv + resource/）
  python tests/pack_code.py --clean-all                  # 清理所有（含 .venv + resource/）
  python tests/pack_code.py -t <target>                  # 自定义目标
  python tests/pack_code.py --le lf                      # 强制 LF 行尾

  # 全套环境准备（部署 + venv + 依赖 + 数据集），开箱即测
  python tests/pack_code.py -t '\\\\wsl.localhost\\Ubuntu\\home\\user\\proj --full

  # 分步执行（增量：已完成的步骤自动跳过）
  python tests/pack_code.py -t <target> --venv           # 仅创建 .venv
  python tests/pack_code.py -t <target> --deps           # 仅安装依赖（需先有 .venv）
  python tests/pack_code.py -t <target> --data           # 仅解压数据集
        """,
    )
    parser.add_argument("--target", "-t", type=Path, default=DEFAULT_TARGET,
                        help=f"目标目录（默认: {DEFAULT_TARGET}）")
    parser.add_argument("--source", "-s", type=Path, default=SOURCE_DIR,
                        help=f"源目录（默认: {SOURCE_DIR}）")
    parser.add_argument("--dry", "--dry-run", dest="dry", action="store_true",
                        help="预览模式，不实际写入文件")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖所有文件 + 忽略增量部署检查")
    parser.add_argument("--clean", action="store_true",
                        help="部署前清理代码文件（保留 .venv 与 resource/）")
    parser.add_argument("--clean-all", action="store_true",
                        help="部署前清理所有（含 .venv 与 resource/CSI_DATASETS/）")
    parser.add_argument("--le", "--line-ending", dest="le",
                        choices=["auto", "preserve", "lf", "crlf"], default="auto",
                        help="文本文件行尾: auto(WSL→LF,其他→preserve) | preserve | lf | crlf")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    env_group = parser.add_argument_group("环境准备（WSL 集成 + 增量部署）")
    env_group.add_argument("--venv", action="store_true",
                           help="创建 .venv（通过 wsl.exe 调用 WSL 内 python3）")
    env_group.add_argument("--deps", action="store_true",
                           help="在 .venv 中安装依赖（需先 --venv）")
    env_group.add_argument("--data", action="store_true",
                           help="解压 resource/CSI_DATASETS.zip（含子 zip 递归解压）")
    env_group.add_argument("--full", action="store_true",
                           help="全套: 部署代码 + venv + deps + data + setup.sh")
    env_group.add_argument("--distro", default="Ubuntu",
                           help="WSL 发行版名称（默认: Ubuntu）")
    env_group.add_argument("--timeout", type=int, default=1800,
                           help="pip install 超时秒数（默认: 1800 = 30 分钟）")

    return parser


def main(argv: list = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # 解析参数
    source = args.source.resolve()
    target = args.target.resolve()
    le_mode = detect_target_line_ending(target) if args.le == "auto" else args.le
    do_venv = args.venv or args.full
    do_deps = args.deps or args.full
    do_data = args.data or args.full
    do_full = args.full

    ctx = DeployContext(
        source=source, target=target,
        dry_run=args.dry, force=args.force,
        clean_all=args.clean_all, le_mode=le_mode,
    )
    wsl = WslExecutor(distro=args.distro, dry_run=args.dry)
    # --clean 时重置 env_state；否则加载已有状态用于增量判断
    env_state = EnvState() if (args.clean or args.clean_all) else EnvState.load(target)

    # 计算总步骤数（动态）
    total_steps = 6  # 基础部署 6 步
    if do_venv: total_steps += 1
    if do_deps: total_steps += 1
    if do_data: total_steps += 1
    if do_full: total_steps += 1
    if do_venv or do_deps: total_steps += 1

    step = 0

    print("=" * 60)
    print("SenseFrame SKILL 打包部署工具")
    print("=" * 60)
    print(f"源目录:   {ctx.source}")
    print(f"目标目录: {ctx.target}")
    print(f"行尾模式: {ctx.le_mode}")
    print(f"WSL 目标: {ctx.is_wsl}")
    if do_full:
        mode = f"{'dry' if ctx.dry_run else 'deploy'}{' + clean' if args.clean else ''}{' + clean-all' if args.clean_all else ''}{' + force' if ctx.force else ''} + FULL (venv + deps + data)"
    else:
        extras = []
        if do_venv: extras.append("venv")
        if do_deps: extras.append("deps")
        if do_data: extras.append("data")
        mode = f"{'dry' if ctx.dry_run else 'deploy'}{' + clean' if args.clean else ''}{' + clean-all' if args.clean_all else ''}{' + force' if ctx.force else ''}" + (f" + [{' + '.join(extras)}]" if extras else "")
    print(f"模式:     {mode}")
    print()

    # 1. 校验源目录
    step += 1
    print(f"[{step}/{total_steps}] 校验源目录...")
    result = validate_source(ctx.source)
    if not result.ok:
        print(f"  [FAIL] {result.message}")
        return 1
    print(f"  [OK] {result.message}")
    for d in result.details:
        print(f"       {d}")

    # 2. 校验 SKILL.md 引用
    step += 1
    print(f"\n[{step}/{total_steps}] 校验 SKILL.md 引用...")
    skill_md = ctx.source / "SKILL.md"
    ref_result = validate_skill_md_references(skill_md, ctx.source)
    if not ref_result.ok:
        print(f"  [WARN] {ref_result.message}")
        print("         （部署后需手动检查）")
    else:
        print(f"  [OK] {ref_result.message}")

    # 3. 创建目标目录
    step += 1
    print(f"\n[{step}/{total_steps}] 创建目标目录...")
    if not ctx.target.exists():
        if not ctx.dry_run:
            ctx.target.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] 创建 {ctx.target}")
    else:
        print(f"  [OK] 目标目录已存在: {ctx.target}")

    # 4. 清理旧部署
    step += 1
    if args.clean or args.clean_all:
        print(f"\n[{step}/{total_steps}] 清理旧部署{'（含 .venv + resource/）' if ctx.clean_all else ''}...")
        clean_target(ctx)
        print("  [OK] 清理完成")
    else:
        print(f"\n[{step}/{total_steps}] 跳过清理")

    # 5. 部署
    step += 1
    print(f"\n[{step}/{total_steps}] 部署文件...")
    stats = DeployStats()

    print("  → 部署代码本体到项目根...")
    root_files = deploy_root_payload(ctx, stats)

    print("  → 部署 SKILL.md + reference/ 到 .opencode/.claude/.agents/skills/...")
    skill_files, skill_dirs = deploy_skills(ctx, stats)

    print("  → 部署 commands/ 到 .opencode/.claude/.agents/commands/...")
    cmd_files, cmd_dirs = deploy_commands(ctx, stats)

    print("  → 生成 AGENTS.md / CLAUDE.md...")
    agents_md = generate_agents_md(ctx)
    claude_md = generate_claude_md(ctx)

    all_files = root_files + skill_files + cmd_files + [agents_md, claude_md]
    all_dirs = skill_dirs + cmd_dirs + [d for d in
        ["senseframe", "scripts", "configs", "examples", "schemas"]
        if (ctx.target / d).exists()]

    print(f"\n  部署统计:")
    print(stats.summary())

    # 6. 后置校验
    step += 1
    print(f"\n[{step}/{total_steps}] 后置校验...")
    if not ctx.dry_run:
        post_result = post_deploy_validate(ctx.target)
        if not post_result.ok:
            print(f"  [FAIL] {post_result.message}")
            for d in post_result.details:
                print(f"       {d}")
            return 2
        print(f"  [OK] {post_result.message}")
        for d in post_result.details:
            print(f"       {d}")
    else:
        print("  [SKIP] dry 模式跳过后置校验")

    # 7. 创建 .venv（如启用）
    if do_venv:
        step += 1
        print(f"\n[{step}/{total_steps}] 创建 .venv 虚拟环境...")
        if not setup_venv(ctx, wsl, env_state):
            if not ctx.dry_run:
                print("  [WARN] venv 创建失败，后续依赖安装将跳过")

    # 8. 安装依赖（如启用）
    if do_deps:
        step += 1
        print(f"\n[{step}/{total_steps}] 安装 Python 依赖...")
        install_requirements(ctx, wsl, env_state, args.timeout)

    # 9. 解压数据集（如启用）
    if do_data:
        step += 1
        print(f"\n[{step}/{total_steps}] 解压数据集...")
        extract_datasets(ctx, env_state)

    # 10. 生成 setup.sh（如 full）
    if do_full:
        step += 1
        print(f"\n[{step}/{total_steps}] 生成 setup.sh 入口脚本...")
        setup_sh = generate_setup_sh(ctx)
        if setup_sh not in all_files:
            all_files.append(setup_sh)
        print(f"  [OK] 生成 {setup_sh}")
        print(f"  提示: 在 WSL 内可执行 'bash setup.sh' 重新准备环境")

    # 11. 验证 import（如有 venv）
    if do_venv or do_deps:
        step += 1
        print(f"\n[{step}/{total_steps}] 验证 senseframe import...")
        verify_senseframe_import(ctx, wsl)

    # 写入清单（含 env_state）
    if not ctx.dry_run:
        manifest = write_manifest(ctx, all_files, all_dirs, stats, env_state)
        # 如果是 dry-run 之外、env_state 有变更，write_manifest 已写入；
        # 但若 env_state 在 write_manifest 之后更新（如 setup_venv 后），
        # 需要再次保存——这里统一在最后保存一次
        env_state.save(ctx.target, manifest)
    elif env_state and not args.clean and not args.clean_all:
        # dry-run 时也展示 env_state 供预览
        pass

    # 完成
    print("\n" + "=" * 60)
    if ctx.dry_run:
        print("[DRY] 预览完成，未实际写入文件")
    else:
        print("[DONE] SenseFrame 部署完成")
    print("=" * 60)

    # 下一步指引
    next_steps = ["下一步:"]
    if ctx.is_wsl and (do_venv or do_deps):
        next_steps.append(f"  wsl -d {args.distro} -- cd {ctx.wsl_path}")
        next_steps.append(f"  source .venv/bin/activate")
        next_steps.append(f"  python -c \"import senseframe; print(senseframe.__version__)\"")
    else:
        next_steps.append(f"  cd {ctx.target}")
        next_steps.append(f"  pip install -e '.[eeg,radio,dev]'")
        next_steps.append(f"  python -c \"import senseframe; print(senseframe.__version__)\"")
    next_steps.append("  # 启动 opencode 或 Claude Code TUI，AI 会自动发现 senseframe skill")
    print("\n" + "\n".join(next_steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
