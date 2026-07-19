#!/usr/bin/env python3
"""EEG 数据集下载脚本（PhysioNet eegmmidb 子集 + BCI Competition IV 2a 全量）。

数据规模：
- PhysioNet eegmmidb: 10 受试者 × 14 run = 140 个 .edf 文件，约 1.5GB
- BCI Competition IV 2a: 9 受试者 × 2 文件 (T+E) = 18 个 .mat 文件，约 770MB

下载源：
- PhysioNet: https://www.physionet.org/files/eegmmidb/1.0.0/（完全开放，HTTPS 直连）
- BCI Competition IV 2a: http://bnci-horizon-2020.eu/database/data-sets/001-2014/
  （BNCI Horizon 2020 镜像，.mat 格式；BBCI 官方下载链接已失效 404）

用法：
    python download_eeg_datasets.py
    python download_eeg_datasets.py --subjects 5      # 仅下载前 5 个受试者
    python download_eeg_datasets.py --skip-physionet  # 跳过 PhysioNet
    python download_eeg_datasets.py --skip-bci        # 跳过 BCI IV 2a
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List

# ============================================================
# 配置
# ============================================================
SENSEFRAME_DATA = Path(r"<SENSEFRAME_DATA_ROOT>")
PHYSIONET_DIR = SENSEFRAME_DATA / "eeg" / "physionet" / "eegmmidb"
BCI_IV_2A_DIR = SENSEFRAME_DATA / "eeg" / "bci_iv_2a"

PHYSIONET_BASE = "https://www.physionet.org/files/eegmmidb/1.0.0"
# PhysioNet eegmmidb: 109 受试者 × 14 run，每受试者 14 个 .edf
PHYSIONET_RUNS = list(range(1, 15))  # R01 - R14

# BCI Competition IV 2a: 9 受试者 (A01-A09)，每人 T (训练) + E (评估) 两个 .mat
# BNCI Horizon 2020 数据库（BCI Competition IV 2a 官方镜像，CC BY-ND 4.0 许可）
# BBCI 原站（https://www.bbci.de/competition/iv/download/）已 404 失效
BCI_IV_2A_BASE = "http://bnci-horizon-2020.eu/database/data-sets/001-2014"
BCI_IV_2A_SUBJECTS = [f"A0{i:02d}" for i in range(1, 10)]  # A01-A09
BCI_IV_2A_FILES = ["T", "E"]  # 训练 / 评估

# 颜色输出
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


# ============================================================
# 通用下载工具
# ============================================================
def download_file(url: str, dest: Path, timeout: float = 60.0,
                  max_retries: int = 3) -> bool:
    """下载单个文件（支持重试）。

    Args:
        url: 下载 URL
        dest: 目标路径
        timeout: 单次请求超时（秒）
        max_retries: 最大重试次数

    Returns:
        True 下载成功 / False 失败
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {_c('SKIP', _GRAY)} {dest.name} (已存在，{dest.stat().st_size//1024}KB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 SenseFrame-Downloader/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length", 0))
                with open(tmp_dest, "wb") as f:
                    downloaded = 0
                    chunk_size = 64 * 1024
                    last_print = time.time()
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0 and time.time() - last_print >= 1.0:
                            pct = downloaded * 100 // total
                            print(
                                f"\r  {_c('DOWN', _CYAN)} {dest.name} "
                                f"{downloaded//1024}KB/{total//1024}KB ({pct}%)",
                                end="", flush=True,
                            )
                            last_print = time.time()
            tmp_dest.rename(dest)
            size_kb = dest.stat().st_size // 1024
            print(f"\r  {_c('OK', _GREEN)} {dest.name} ({size_kb}KB){' '*30}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"\n  {_c('RETRY', _YELLOW)} {dest.name} attempt {attempt}/{max_retries}: {e}")
            if tmp_dest.exists():
                tmp_dest.unlink()
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            print(f"\n  {_c('FAIL', _RED)} {dest.name}: {e}")
            if tmp_dest.exists():
                tmp_dest.unlink()
            return False

    return False


# ============================================================
# PhysioNet eegmmidb 下载
# ============================================================
def download_physionet_eegmmidb(subjects: int = 10) -> tuple[int, int]:
    """下载 PhysioNet eegmmidb 数据集。

    Args:
        subjects: 下载前 N 个受试者（1-109）

    Returns:
        (成功数, 失败数)
    """
    print(f"\n{_c('=== PhysioNet eegmmidb ===', _CYAN)}")
    print(f"目标: 前 {subjects} 受试者 × 14 run = {subjects * 14} 个 .edf 文件")
    print(f"目录: {PHYSIONET_DIR}")

    n_ok, n_fail = 0, 0
    for sub_idx in range(1, subjects + 1):
        sub_dir = PHYSIONET_DIR / f"S{sub_idx:03d}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{_c(f'[受试者 {sub_idx:03d}/{subjects}]', _YELLOW)} {sub_dir}")

        for run in PHYSIONET_RUNS:
            url = f"{PHYSIONET_BASE}/S{sub_idx:03d}/S{sub_idx:03d}R{run:02d}.edf"
            dest = sub_dir / f"S{sub_idx:03d}R{run:02d}.edf"
            if download_file(url, dest):
                n_ok += 1
            else:
                n_fail += 1

    print(f"\n{_c('PhysioNet 汇总', _CYAN)}: 成功 {n_ok} / 失败 {n_fail}")
    return n_ok, n_fail


# ============================================================
# BCI Competition IV 2a 下载
# ============================================================
def download_bci_iv_2a() -> tuple[int, int]:
    """下载 BCI Competition IV 2a 数据集（9 受试者 × T+E = 18 .mat 文件）。

    下载源：BNCI Horizon 2020 数据库（.mat 格式，CC BY-ND 4.0）
    数据规格：22 EEG + 3 EOG 通道，250Hz，4 类运动想象

    Returns:
        (成功数, 失败数)
    """
    print(f"\n{_c('=== BCI Competition IV 2a ===', _CYAN)}")
    print(f"目标: 9 受试者 × 2 文件 (T+E) = 18 个 .mat 文件（约 770MB）")
    print(f"目录: {BCI_IV_2A_DIR}")

    n_ok, n_fail = 0, 0
    for subject in BCI_IV_2A_SUBJECTS:
        sub_dir = BCI_IV_2A_DIR / subject
        sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{_c(f'[受试者 {subject}]', _YELLOW)} {sub_dir}")

        for file_type in BCI_IV_2A_FILES:
            # BCI IV 2a 文件命名：A01T.mat, A01E.mat（BNCI Horizon 2020 镜像）
            filename = f"{subject}{file_type}.mat"
            url = f"{BCI_IV_2A_BASE}/{filename}"
            dest = sub_dir / filename
            if download_file(url, dest):
                n_ok += 1
            else:
                n_fail += 1

    print(f"\n{_c('BCI IV 2a 汇总', _CYAN)}: 成功 {n_ok} / 失败 {n_fail}")
    return n_ok, n_fail


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="EEG 数据集下载")
    parser.add_argument("--subjects", type=int, default=10,
                        help="PhysioNet 受试者数量 (1-109, 默认 10)")
    parser.add_argument("--skip-physionet", action="store_true",
                        help="跳过 PhysioNet eegmmidb 下载")
    parser.add_argument("--skip-bci", action="store_true",
                        help="跳过 BCI Competition IV 2a 下载")
    args = parser.parse_args()

    if not (1 <= args.subjects <= 109):
        print(f"{_c('ERROR', _RED)} --subjects 必须在 1-109 之间")
        sys.exit(1)

    total_ok, total_fail = 0, 0

    if not args.skip_physionet:
        ok, fail = download_physionet_eegmmidb(args.subjects)
        total_ok += ok
        total_fail += fail

    if not args.skip_bci:
        ok, fail = download_bci_iv_2a()
        total_ok += ok
        total_fail += fail

    print(f"\n{_c('=== 总结 ===', _CYAN)}")
    print(f"总成功: {_c(str(total_ok), _GREEN)}")
    print(f"总失败: {_c(str(total_fail), _RED)}")
    print(f"数据目录: {SENSEFRAME_DATA}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
