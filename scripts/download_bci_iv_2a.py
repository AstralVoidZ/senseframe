#!/usr/bin/env python3
"""BCI Competition IV 2a 下载脚本（BNCI Horizon 2020 镜像，.mat 格式）。

下载源：http://bnci-horizon-2020.eu/database/data-sets/001-2014/
BBCI 官方下载链接（https://www.bbci.de/competition/iv/download/）已 404 失效。

数据规模：9 受试者 × 2 文件 (T+E) = 18 个 .mat 文件，约 770MB
"""
from __future__ import annotations

import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ============================================================
# 配置
# ============================================================
DEST_ROOT = Path(r"<SENSEFRAME_DATA_ROOT>/eeg/bci_iv_2a")
BASE_URL = "http://bnci-horizon-2020.eu/database/data-sets/001-2014"
SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]  # A01-A09（BNCI 命名 2 位数）
FILE_TYPES = ["T", "E"]  # 训练 / 评估

# 模拟浏览器 UA（避免服务器对默认 urllib UA 限流）
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def download_file(url: str, dest: Path, max_retry: int = 5) -> bool:
    """下载单个文件，支持断点续传检测 + 重试。

    Returns:
        True 下载成功（或文件已存在），False 失败
    """
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"SKIP {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    for attempt in range(1, max_retry + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as resp:
                if resp.status != 200:
                    print(f"FAIL {dest.name} attempt {attempt}: HTTP {resp.status}")
                    time.sleep(15)
                    continue
                # 流式写入（避免大文件占内存）
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                # 写入完成后 rename
                tmp.replace(dest)
                size_mb = dest.stat().st_size / 1e6
                elapsed = time.time() - t0
                print(f"OK   {dest.name} ({size_mb:.1f} MB in {elapsed:.1f}s)")
                return True
        except urllib.error.HTTPError as e:
            print(f"FAIL {dest.name} attempt {attempt}: HTTP {e.code} {e.reason}")
        except Exception as e:
            msg = str(e)[:80]
            print(f"FAIL {dest.name} attempt {attempt}: {msg}")
        time.sleep(15)

    print(f"GIVEUP {dest.name} after {max_retry} retries")
    return False


def main() -> int:
    print("=== BCI Competition IV 2a 下载（BNCI Horizon 2020 镜像）===")
    print(f"目标: 9 受试者 × 2 文件 (T+E) = 18 个 .mat 文件")
    print(f"目录: {DEST_ROOT}")
    print()

    n_ok, n_fail = 0, 0
    for subj in SUBJECTS:
        subj_dir = DEST_ROOT / subj
        subj_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- 受试者 {subj} ---")

        for ftype in FILE_TYPES:
            fname = f"{subj}{ftype}.mat"
            url = f"{BASE_URL}/{fname}"
            dest = subj_dir / fname
            if download_file(url, dest):
                n_ok += 1
            else:
                n_fail += 1

    print()
    print(f"=== 下载汇总 ===")
    print(f"成功: {n_ok}")
    print(f"失败: {n_fail}")
    print(f"总计: {n_ok + n_fail}")

    # 列出所有 .mat 文件
    print()
    print("=== 已下载文件 ===")
    for f in sorted(DEST_ROOT.rglob("*.mat")):
        print(f"  {f.relative_to(DEST_ROOT)}: {f.stat().st_size / 1e6:.1f} MB")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
