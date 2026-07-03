"""SenseFrame SKILL 打包与部署工具。

将 SenseFrame 打包成完备的 SKILL 包，部署到目标目录，
兼容 opencode / Claude Code / 通用 agents 三种 SKILL 配置格式。

部署布局：
    <target>/
      senseframe/            # 核心 Python 包（可 import）
      scripts/               # CLI 脚本
      configs/               # 配置示例
      examples/              # 示例代码
      schemas/               # JSON Schema
      requirements.txt       # 依赖清单
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
      .senseframe_deploy.json  # 部署清单（用于增量更新与清理）

设计原则：
- 代码本体只部署一份（项目根），避免重复
- SKILL.md + reference/ 部署到三个 skills 目录（opencode/Claude/agents）
- SKILL.md 中 ./reference/xxx.md 相对路径在 skills 目录内有效
- SKILL.md 中 scripts/xxx、configs/xxx 路径基于项目根（AI 工作目录）
- 排除开发文件：README.md、tests/、docs/、__pycache__/、*.pyc、.pytest_cache/
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ============================================================
# 配置
# ============================================================

# 源目录（SenseFrame 项目根）
SOURCE_DIR = Path(__file__).resolve().parent.parent

# 默认目标目录（../AdvanceTest/AutoML）
DEFAULT_TARGET = SOURCE_DIR.parent / "AdvanceTest" / "AutoML"

# SKILL 名称（必须与 SKILL.md frontmatter name 一致，符合 opencode 正则）
SKILL_NAME = "senseframe"

# 三个 skills 部署位置（相对于目标目录）
SKILL_DIRS = [
    ".opencode/skills/senseframe",
    ".claude/skills/senseframe",
    ".agents/skills/senseframe",
]

# 三个 commands 部署位置（相对于目标目录）
# opencode 扫描 .opencode/command/ 和 .opencode/commands/（两者都行）
# Claude Code 用 .claude/commands/
# 通用 agents 用 .agents/commands/
COMMAND_DIRS = [
    ".opencode/commands",
    ".claude/commands",
    ".agents/commands",
]

# 项目根部署的目录/文件（代码本体）
ROOT_PAYLOAD_DIRS = ["senseframe", "scripts", "configs", "examples", "schemas"]
ROOT_PAYLOAD_FILES = ["requirements.txt"]

# skills 目录部署的文件（SKILL.md + reference/）
SKILL_PAYLOAD_FILES = ["SKILL.md"]
SKILL_PAYLOAD_DIRS = ["reference"]

# commands 源目录（项目根下的 commands/）
COMMAND_SOURCE_DIR = "commands"

# 排除模式（不打包开发文件）
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    "*.egg-info",
    ".git",
    ".gitignore",
    ".idea",
    ".vscode",
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

# WSL UNC 路径前缀（用于自动检测目标是否在 WSL 中）
WSL_PATH_MARKERS = ("wsl.localhost", "wsl$")


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


# ============================================================
# 工具函数
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
        # 目录匹配
        if path.is_dir() and name == pattern:
            return True
    return False


def is_text_file(path: Path) -> bool:
    """判断是否为文本文件（按扩展名）。"""
    return path.suffix.lower() in TEXT_EXTENSIONS


def normalize_line_endings(data: bytes, mode: str) -> bytes:
    """转换行尾。

    Args:
        data: 原始字节
        mode: "lf" → \\n, "crlf" → \\r\\n, "preserve" → 原样
    """
    if mode == "preserve":
        return data
    # 先统一为 LF，再按需转换
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if mode == "crlf":
        data = data.replace(b"\n", b"\r\n")
    return data


def detect_target_line_ending(target: Path) -> str:
    """自动检测目标应使用的行尾模式。

    WSL UNC 路径（\\\\wsl.localhost\\... 或 \\\\wsl$\\...）→ "lf"
    其他（Windows 本地路径）→ "preserve"
    """
    target_str = str(target).lower()
    for marker in WSL_PATH_MARKERS:
        if marker in target_str:
            return "lf"
    return "preserve"


def file_hash(path: Path) -> str:
    """计算文件内容的 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def files_equal(src: Path, dst: Path, le_mode: str = "preserve") -> bool:
    """判断两个文件内容是否相同（考虑行尾转换）。

    若 le_mode != "preserve" 且 src 是文本文件，则比较归一化后的内容，
    而非原始字节——否则 CRLF 源 + LF 目标会被误判为"未变"，导致跳过更新。
    """
    if not dst.exists():
        return False
    if le_mode != "preserve" and is_text_file(src):
        src_data = normalize_line_endings(src.read_bytes(), le_mode)
        dst_data = dst.read_bytes()
        return src_data == dst_data
    # 二进制文件或 preserve 模式：直接比大小 + hash
    if src.stat().st_size != dst.stat().st_size:
        return False
    return file_hash(src) == file_hash(dst)


def copy_file(src: Path, dst: Path, le_mode: str = "preserve") -> None:
    """复制单个文件，支持行尾转换。

    文本文件按 le_mode 转换行尾；二进制文件原样复制。
    保留源文件的 mtime（copystat 在跨文件系统时可能失败，用 utime 兜底）。
    """
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


def copy_file_or_dir(
    src: Path,
    dst: Path,
    stats: DeployStats,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """复制文件或目录，支持增量更新。"""
    if should_exclude(src):
        return

    if src.is_file():
        dst_file = dst / src.name if dst.is_dir() else dst
        if dst_file.exists() and not force:
            if files_equal(src, dst_file):
                stats.files_skipped += 1
                return
            stats.files_updated += 1
        else:
            stats.files_copied += 1

        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_file)
        stats.bytes_copied += src.stat().st_size

    elif src.is_dir():
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        stats.dirs_created += 1
        for item in src.iterdir():
            if should_exclude(item):
                continue
            copy_file_or_dir(item, dst / src.name if dst.is_dir() else dst,
                             stats, force, dry_run)


# ============================================================
# 校验
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

    # 必需文件/目录
    required = ["senseframe/__init__.py", "SKILL.md", "requirements.txt"]
    for rel in required:
        if not (source / rel).exists():
            errors.append(f"缺少必需文件: {rel}")

    # 校验 senseframe 可 import
    init_file = source / "senseframe" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        if "__version__" not in content:
            errors.append("senseframe/__init__.py 缺少 __version__")

    # 校验 SKILL.md frontmatter
    skill_md = source / "SKILL.md"
    if skill_md.exists():
        fm = parse_frontmatter(skill_md)
        name = fm.get("name", "")
        desc = fm.get("description", "")

        if not name:
            errors.append("SKILL.md frontmatter 缺少 name")
        elif not SKILL_NAME_REGEX.match(name):
            errors.append(
                f"SKILL.md name '{name}' 不符合正则 {SKILL_NAME_REGEX.pattern}"
            )
        elif name != SKILL_NAME:
            errors.append(
                f"SKILL.md name '{name}' 与部署目录名 '{SKILL_NAME}' 不一致"
            )

        if not desc:
            errors.append("SKILL.md frontmatter 缺少 description")
        elif not (DESCRIPTION_MIN <= len(desc) <= DESCRIPTION_MAX):
            errors.append(
                f"description 长度 {len(desc)} 不在 [{DESCRIPTION_MIN}, {DESCRIPTION_MAX}] 范围"
            )

        details.append(f"SKILL.md frontmatter: name={name}, desc.length={len(desc)}")

    if errors:
        return ValidationResult(ok=False, message="; ".join(errors), details=details)
    return ValidationResult(ok=True, message="源目录校验通过", details=details)


def validate_skill_md_references(skill_md_path: Path, skill_dir: Path) -> ValidationResult:
    """校验 SKILL.md 中的相对路径引用是否存在。

    SKILL.md 中 ./reference/xxx.md 引用相对于 SKILL.md 所在目录。
    scripts/xxx、configs/xxx 引用相对于项目根（此处不校验，因为部署后才有项目根）。
    """
    content = skill_md_path.read_text(encoding="utf-8")
    # 匹配 markdown 链接 [text](./reference/xxx.md) 和 [text](./scripts/xxx.py)
    ref_pattern = re.compile(r"\]\((\./[^)]+)\)")
    missing = []
    for match in ref_pattern.finditer(content):
        rel_path = match.group(1)
        # 只校验 ./reference/ 开头的（这些在 skills 目录内）
        if rel_path.startswith("./reference/"):
            target = skill_md_path.parent / rel_path
            if not target.exists():
                missing.append(rel_path)
    if missing:
        return ValidationResult(
            ok=False,
            message=f"SKILL.md 引用的文件不存在: {missing}",
        )
    return ValidationResult(ok=True, message="SKILL.md 引用校验通过")


# ============================================================
# 部署逻辑
# ============================================================

def clean_target(target: Path, dry_run: bool = False) -> None:
    """清理旧的 SenseFrame 部署（仅清理清单记录的文件）。"""
    manifest = target / DEPLOY_MANIFEST
    if not manifest.exists():
        print(f"  无部署清单，跳过清理")
        return

    try:
        records = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  部署清单解析失败: {e}，跳过清理")
        return

    # 先删文件
    for f in records.get("files", []):
        fp = target / f
        if fp.exists() and fp.is_file():
            if not dry_run:
                fp.unlink()
            print(f"  删除文件: {f}")

    # 再删空目录（从深到浅）
    all_dirs = sorted(records.get("dirs", []), key=len, reverse=True)
    for d in all_dirs:
        dp = target / d
        if dp.exists() and dp.is_dir():
            try:
                if not dry_run:
                    dp.rmdir()
                print(f"  删除目录: {d}")
            except OSError:
                pass  # 非空目录跳过

    if not dry_run:
        manifest.unlink()


def deploy_root_payload(
    source: Path,
    target: Path,
    stats: DeployStats,
    force: bool = False,
    dry_run: bool = False,
    le_mode: str = "preserve",
) -> list:
    """部署代码本体到目标目录根。"""
    deployed_files = []
    deployed_dirs = []

    # 目录
    for dir_name in ROOT_PAYLOAD_DIRS:
        src = source / dir_name
        if not src.exists():
            print(f"  跳过不存在的目录: {dir_name}")
            continue
        dst = target / dir_name
        if not dry_run:
            dst.mkdir(parents=True, exist_ok=True)
        deployed_dirs.append(dir_name)
        for item in src.rglob("*"):
            if should_exclude(item):
                continue
            rel = item.relative_to(src)
            dst_item = dst / rel
            if item.is_file():
                if dst_item.exists() and not force and files_equal(item, dst_item, le_mode):
                    stats.files_skipped += 1
                    continue
                if not dry_run:
                    dst_item.parent.mkdir(parents=True, exist_ok=True)
                    copy_file(item, dst_item, le_mode)
                stats.files_copied += 1
                stats.bytes_copied += item.stat().st_size
                deployed_files.append(str(dst_item.relative_to(target)))
            elif item.is_dir():
                if not dry_run:
                    dst_item.mkdir(parents=True, exist_ok=True)
                deployed_dirs.append(str(dst_item.relative_to(target)))

    # 文件
    for file_name in ROOT_PAYLOAD_FILES:
        src = source / file_name
        if not src.exists():
            print(f"  跳过不存在的文件: {file_name}")
            continue
        dst = target / file_name
        if dst.exists() and not force and files_equal(src, dst, le_mode):
            stats.files_skipped += 1
            continue
        if not dry_run:
            copy_file(src, dst, le_mode)
        stats.files_copied += 1
        stats.bytes_copied += src.stat().st_size
        deployed_files.append(file_name)

    return deployed_files


def deploy_skills(
    source: Path,
    target: Path,
    stats: DeployStats,
    force: bool = False,
    dry_run: bool = False,
    le_mode: str = "preserve",
) -> tuple:
    """部署 SKILL.md + reference/ 到三个 skills 目录。"""
    deployed_files = []
    deployed_dirs = []

    for skill_rel in SKILL_DIRS:
        skill_dst = target / skill_rel
        if not dry_run:
            skill_dst.mkdir(parents=True, exist_ok=True)
        deployed_dirs.append(skill_rel)

        # SKILL.md
        for fname in SKILL_PAYLOAD_FILES:
            src = source / fname
            if not src.exists():
                continue
            dst = skill_dst / fname
            if dst.exists() and not force and files_equal(src, dst, le_mode):
                stats.files_skipped += 1
                continue
            if not dry_run:
                copy_file(src, dst, le_mode)
            stats.files_copied += 1
            stats.bytes_copied += src.stat().st_size
            deployed_files.append(str(dst.relative_to(target)))

        # reference/
        for dname in SKILL_PAYLOAD_DIRS:
            src = source / dname
            if not src.exists():
                continue
            dst = skill_dst / dname
            if not dry_run:
                dst.mkdir(parents=True, exist_ok=True)
            deployed_dirs.append(str(dst.relative_to(target)))
            for item in src.rglob("*"):
                if should_exclude(item):
                    continue
                rel = item.relative_to(src)
                dst_item = dst / rel
                if item.is_file():
                    if dst_item.exists() and not force and files_equal(item, dst_item, le_mode):
                        stats.files_skipped += 1
                        continue
                    if not dry_run:
                        dst_item.parent.mkdir(parents=True, exist_ok=True)
                        copy_file(item, dst_item, le_mode)
                    stats.files_copied += 1
                    stats.bytes_copied += item.stat().st_size
                    deployed_files.append(str(dst_item.relative_to(target)))
                elif item.is_dir():
                    if not dry_run:
                        dst_item.mkdir(parents=True, exist_ok=True)
                    deployed_dirs.append(str(dst_item.relative_to(target)))

    return deployed_files, deployed_dirs


def deploy_commands(
    source: Path,
    target: Path,
    stats: DeployStats,
    force: bool = False,
    dry_run: bool = False,
    le_mode: str = "preserve",
) -> tuple:
    """部署 commands/ 到三个 commands 目录（opencode + claude code + agents）。

    commands 源目录: <source>/commands/*.md
    部署目标: <target>/.{opencode,claude,agents}/commands/*.md
    每个 .md 文件直接部署到 commands 目录（不需要子目录）。
    """
    deployed_files = []
    deployed_dirs = []

    cmd_src = source / COMMAND_SOURCE_DIR
    if not cmd_src.exists():
        print(f"  跳过: 源目录无 {COMMAND_SOURCE_DIR}/")
        return deployed_files, deployed_dirs

    # 获取所有 .md 文件
    cmd_files = sorted(cmd_src.glob("*.md"))
    if not cmd_files:
        print(f"  跳过: {COMMAND_SOURCE_DIR}/ 无 .md 文件")
        return deployed_files, deployed_dirs

    for cmd_dir_rel in COMMAND_DIRS:
        cmd_dst = target / cmd_dir_rel
        if not dry_run:
            cmd_dst.mkdir(parents=True, exist_ok=True)
        deployed_dirs.append(cmd_dir_rel)

        for cmd_file in cmd_files:
            dst = cmd_dst / cmd_file.name
            if dst.exists() and not force and files_equal(cmd_file, dst, le_mode):
                stats.files_skipped += 1
                continue
            if not dry_run:
                copy_file(cmd_file, dst, le_mode)
            stats.files_copied += 1
            stats.bytes_copied += cmd_file.stat().st_size
            deployed_files.append(str(dst.relative_to(target)))

    return deployed_files, deployed_dirs


def generate_agents_md(target: Path, dry_run: bool = False, le_mode: str = "preserve") -> str:
    """生成 AGENTS.md（opencode 项目入口）。

    用 write_bytes + 归一化行尾，避免 Windows write_text 自动加 CRLF
    导致 WSL 中 YAML frontmatter / markdown 解析问题。
    """
    content = f"""# AGENTS.md

> 本文件是 AI agent 的项目入口说明。opencode 启动时从当前目录向上遍历查找此文件。

## 项目概述

SenseFrame 测试环境。SenseFrame 是 Agent 驱动的 AutoML 训练框架，
提供可编程原语库 + 执行底座 + 安全护栏。

## 目录结构

```
AdvanceTest/AutoML/
├── AGENTS.md              # 本文件（opencode 入口）
├── CLAUDE.md              # Claude Code 兼容入口
├── senseframe/            # 核心 Python 包（可直接 import）
├── scripts/               # CLI 脚本
├── configs/               # 配置示例
├── examples/              # 示例代码
├── schemas/               # JSON Schema
├── requirements.txt       # Python 依赖
├── CSI_DATASETS/          # WiFi CSI 数据集
├── .opencode/skills/senseframe/   # opencode skill
├── .claude/skills/senseframe/     # Claude Code skill
└── .agents/skills/senseframe/     # 通用 agents skill
```

## 使用 SenseFrame

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

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

WiFi CSI 数据集已部署到 `CSI_DATASETS/`。配置中 `data_root` 可指向此目录：

```yaml
data_root: ./CSI_DATASETS
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
    if not dry_run:
        data = normalize_line_endings(content.encode("utf-8"), le_mode)
        (target / "AGENTS.md").write_bytes(data)
    return "AGENTS.md"


def generate_claude_md(target: Path, dry_run: bool = False, le_mode: str = "preserve") -> str:
    """生成 CLAUDE.md（Claude Code 兼容入口，内容同 AGENTS.md）。"""
    agents_md = target / "AGENTS.md"
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8")
        # 替换标题
        content = content.replace("# AGENTS.md", "# CLAUDE.md")
    else:
        content = f"# CLAUDE.md\n\nSenseFrame 测试环境。详见 AGENTS.md。\n"
    if not dry_run:
        data = normalize_line_endings(content.encode("utf-8"), le_mode)
        (target / "CLAUDE.md").write_bytes(data)
    return "CLAUDE.md"


def write_manifest(
    target: Path,
    files: list,
    dirs: list,
    stats: DeployStats,
    dry_run: bool = False,
) -> None:
    """写入部署清单。"""
    manifest = {
        "skill_name": SKILL_NAME,
        "deployed_at": datetime.now().isoformat(),
        "source": str(SOURCE_DIR),
        "files": sorted(set(files)),
        "dirs": sorted(set(dirs)),
        "stats": {
            "files_copied": stats.files_copied,
            "files_skipped": stats.files_skipped,
            "files_updated": stats.files_updated,
            "bytes_copied": stats.bytes_copied,
        },
    }
    if not dry_run:
        (target / DEPLOY_MANIFEST).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ============================================================
# 后置校验
# ============================================================

def post_deploy_validate(target: Path) -> ValidationResult:
    """部署后校验。"""
    errors = []
    details = []

    # 1. senseframe 可 import
    senseframe_init = target / "senseframe" / "__init__.py"
    if not senseframe_init.exists():
        errors.append("部署后 senseframe/__init__.py 不存在")
    else:
        details.append("senseframe/__init__.py 存在")

    # 2. SKILL.md 在三个位置都存在
    for skill_rel in SKILL_DIRS:
        skill_md = target / skill_rel / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"部署后 {skill_rel}/SKILL.md 不存在")
        else:
            # 校验 frontmatter
            fm = parse_frontmatter(skill_md)
            if fm.get("name") != SKILL_NAME:
                errors.append(f"{skill_rel}/SKILL.md name 不匹配")
            else:
                details.append(f"{skill_rel}/SKILL.md frontmatter OK")

    # 3. AGENTS.md / CLAUDE.md 存在
    for entry in ["AGENTS.md", "CLAUDE.md"]:
        if not (target / entry).exists():
            errors.append(f"部署后 {entry} 不存在")
        else:
            details.append(f"{entry} 存在")

    # 4. requirements.txt 存在
    if not (target / "requirements.txt").exists():
        errors.append("部署后 requirements.txt 不存在")

    # 5. 校验 SKILL.md 引用的 reference 文件存在
    for skill_rel in SKILL_DIRS:
        skill_md = target / skill_rel / "SKILL.md"
        if skill_md.exists():
            result = validate_skill_md_references(skill_md, target / skill_rel)
            if not result.ok:
                errors.append(f"{skill_rel}: {result.message}")

    # 6. 校验 commands 在三个位置都存在且有 frontmatter
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
                    # 校验 frontmatter 有 description
                    fm = parse_frontmatter(cmd_file)
                    if not fm.get("description"):
                        errors.append(f"{cmd_dir_rel}/{cmd_name} 缺少 description")
                    else:
                        details.append(f"{cmd_dir_rel}/{cmd_name} frontmatter OK")

    if errors:
        return ValidationResult(ok=False, message="; ".join(errors), details=details)
    return ValidationResult(ok=True, message="部署后校验通过", details=details)


# ============================================================
# 主流程
# ============================================================

def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        description="将 SenseFrame 打包为 SKILL 包并部署到目标目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tests/pack_code.py                    # 部署到默认目标 ../AdvanceTest/AutoML
  python tests/pack_code.py --dry-run          # 预览不执行
  python tests/pack_code.py --force            # 强制覆盖所有文件
  python tests/pack_code.py --clean            # 先清理旧部署
  python tests/pack_code.py --target /custom   # 自定义目标
  python tests/pack_code.py -t '\\\\wsl.localhost\\Ubuntu\\home\\user\\proj   # 部署到 WSL
  python tests/pack_code.py --line-ending lf   # 强制 LF 行尾
        """,
    )
    parser.add_argument(
        "--target", "-t",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"目标目录（默认: {DEFAULT_TARGET}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不实际写入文件",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖所有文件（不走增量更新）",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="部署前清理旧的 SenseFrame 部署",
    )
    parser.add_argument(
        "--line-ending",
        choices=["auto", "preserve", "lf", "crlf"],
        default="auto",
        help="文本文件行尾: auto(WSL→LF,其他→preserve) | preserve | lf | crlf",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--source", "-s",
        type=Path,
        default=SOURCE_DIR,
        help=f"源目录（默认: {SOURCE_DIR}）",
    )

    args = parser.parse_args(argv)
    source: Path = args.source.resolve()
    target: Path = args.target.resolve()

    # 解析行尾模式
    if args.line_ending == "auto":
        le_mode = detect_target_line_ending(target)
    else:
        le_mode = args.line_ending

    print("=" * 60)
    print("SenseFrame SKILL 打包部署工具")
    print("=" * 60)
    print(f"源目录:   {source}")
    print(f"目标目录: {target}")
    print(f"行尾模式: {le_mode}")
    print(f"模式:     {'dry-run' if args.dry_run else 'deploy'}"
          f"{' + clean' if args.clean else ''}"
          f"{' + force' if args.force else ''}")
    print()

    # 1. 校验源目录
    print("[1/6] 校验源目录...")
    result = validate_source(source)
    if not result.ok:
        print(f"  [FAIL] {result.message}")
        return 1
    print(f"  [OK] {result.message}")
    for d in result.details:
        print(f"       {d}")

    # 2. 校验 SKILL.md 引用
    print("\n[2/6] 校验 SKILL.md 引用...")
    skill_md = source / "SKILL.md"
    ref_result = validate_skill_md_references(skill_md, source)
    if not ref_result.ok:
        print(f"  [WARN] {ref_result.message}")
        print("         （部署后需手动检查）")
    else:
        print(f"  [OK] {ref_result.message}")

    # 3. 创建目标目录
    print("\n[3/6] 创建目标目录...")
    if not target.exists():
        if not args.dry_run:
            target.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] 创建 {target}")
    else:
        print(f"  [OK] 目标目录已存在: {target}")

    # 4. 清理旧部署
    if args.clean:
        print("\n[4/6] 清理旧部署...")
        clean_target(target, args.dry_run)
        print("  [OK] 清理完成")
    else:
        print("\n[4/6] 跳过清理")

    # 5. 部署
    print("\n[5/6] 部署文件...")
    stats = DeployStats()

    # 5a. 代码本体
    print("  → 部署代码本体到项目根...")
    root_files = deploy_root_payload(source, target, stats, args.force, args.dry_run, le_mode)

    # 5b. SKILL 到三个 skills 目录
    print("  → 部署 SKILL.md + reference/ 到 .opencode/.claude/.agents/skills/...")
    skill_files, skill_dirs = deploy_skills(source, target, stats, args.force, args.dry_run, le_mode)

    # 5c. Commands 到三个 commands 目录
    print("  → 部署 commands/ 到 .opencode/.claude/.agents/commands/...")
    cmd_files, cmd_dirs = deploy_commands(source, target, stats, args.force, args.dry_run, le_mode)

    # 5d. 生成 AGENTS.md / CLAUDE.md
    print("  → 生成 AGENTS.md / CLAUDE.md...")
    agents_md = generate_agents_md(target, args.dry_run, le_mode)
    claude_md = generate_claude_md(target, args.dry_run, le_mode)

    all_files = root_files + skill_files + cmd_files + [agents_md, claude_md]
    all_dirs = skill_dirs + cmd_dirs + [d for d in [
        "senseframe", "scripts", "configs", "examples", "schemas",
    ] if (target / d).exists()]

    # 5e. 写入清单
    write_manifest(target, all_files, all_dirs, stats, args.dry_run)

    print(f"\n  部署统计:")
    print(stats.summary())

    # 6. 后置校验
    print("\n[6/6] 后置校验...")
    if not args.dry_run:
        post_result = post_deploy_validate(target)
        if not post_result.ok:
            print(f"  [FAIL] {post_result.message}")
            for d in post_result.details:
                print(f"       {d}")
            return 2
        print(f"  [OK] {post_result.message}")
        for d in post_result.details:
            print(f"       {d}")
    else:
        print("  [SKIP] dry-run 模式跳过后置校验")

    # 完成
    print("\n" + "=" * 60)
    if args.dry_run:
        print("[DRY-RUN] 预览完成，未实际写入文件")
    else:
        print("[DONE] SenseFrame SKILL 部署完成")
    print("=" * 60)
    print(f"""
下一步:
  cd {target}
  pip install -r requirements.txt
  python -c "import senseframe; print(senseframe.__version__)"
  # 启动 opencode 或 Claude Code TUI，AI 会自动发现 senseframe skill
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
