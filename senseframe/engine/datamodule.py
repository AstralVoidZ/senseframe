"""
通用 DataModule 适配器：将 (train_ds, test_ds) 包装为 Lightning DataModule。

dataloader 参数与 CSIDataModule 保持一致，确保确定性：
- train: shuffle=True, drop_last=True, batch_size=batch_size
- val:   shuffle=False, batch_size=batch_size*2

支持自监督模式的三数据集（unsupervised, supervised, test）。

Phase 2.1a：支持通过 TransformConfig 应用数据变换
Phase 2.1b：支持 streaming 模式（IterableDataset）
Phase 2.1c：支持数据缓存（cache_dir）
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl


# ============================================================
# RFC-005：修复 PyTorch DataLoader worker pipe 泄露
# ============================================================
# PyTorch _MultiProcessingDataLoaderIter._shutdown_workers() 关闭 _worker_result_queue
# 和 _index_queues（18 pipe/iterator），但不 close worker 进程的 stdin/stdout/stderr/
# sentinel pipe（16 pipe/iterator），导致每次 run 泄露 (num_workers × 2) × 2 DataLoader
# = 32~64 个 pipe（num_workers=8 时 +48 pipe/run）。
#
# 本 patch 在原 _shutdown_workers 后调用 w.close() 关闭 worker 进程的 pipe。
# 幂等：_patch_applied 标志防止重复 patch。
import torch.utils.data.dataloader as _dl_mod

if hasattr(_dl_mod, "_MultiProcessingDataLoaderIter") and not getattr(
    _dl_mod, "_sf_pipe_leak_patch_applied", False
):
    _orig_shutdown_workers = _dl_mod._MultiProcessingDataLoaderIter._shutdown_workers
    _orig_clean_up_worker = _dl_mod._MultiProcessingDataLoaderIter._clean_up_worker

    def _patched_shutdown_workers(self) -> None:
        _orig_shutdown_workers(self)
        # close worker 进程的 stdin/stdout/stderr/sentinel pipe（+16 pipe/iterator）
        # _orig_shutdown_workers 已 join workers，此时 poll() != None，close 安全
        for _w in getattr(self, "_workers", []):
            try:
                if hasattr(_w, "close"):
                    _w.close()
            except Exception:
                pass

    @staticmethod
    def _patched_clean_up_worker(w) -> None:
        # w.close() 后 Process._closed=True，原 _clean_up_worker 调 w.is_alive() 会
        # 抛 ValueError("process object is closed")。此处对 closed process 安全跳过。
        if getattr(w, "_closed", False):
            return
        try:
            _orig_clean_up_worker(w)
        except (ValueError, OSError):
            pass

    _dl_mod._MultiProcessingDataLoaderIter._shutdown_workers = _patched_shutdown_workers
    _dl_mod._MultiProcessingDataLoaderIter._clean_up_worker = _patched_clean_up_worker
    _dl_mod._sf_pipe_leak_patch_applied = True


# ============================================================
# Phase 2.1a：变换包装器
# ============================================================
class _TransformWrapper(Dataset):
    """包装 Dataset，在 __getitem__ 后应用变换函数。

    变换函数签名：fn(x, y) -> (x, y)
    """

    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y


# ============================================================
# Phase 2.1b：流式数据集
# ============================================================
class _StreamingCSVDataset(IterableDataset):
    """流式 CSV 数据集，逐行读取避免全量加载。

    用于 GenericDataModule 的 streaming 模式，CSV 格式假设：
    - 最后一列为标签 y
    - 其余列为特征 x
    """

    def __init__(self, csv_path: str, label_col: int = -1,
                 transform: Optional[Callable] = None):
        self.csv_path = csv_path
        self.label_col = label_col
        self.transform = transform

    def __iter__(self):
        import csv
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    *features, label = row
                    x = torch.FloatTensor([float(v) for v in features])
                    y = int(float(label))
                    if self.transform is not None:
                        x, y = self.transform(x, y)
                    yield x, y
                except (ValueError, IndexError):
                    continue  # 跳过无效行


class GenericDataModule(pl.LightningDataModule):
    """
    通用 DataModule，包装场景容器返回的 Dataset。

    Args:
        train_dataset: 训练集
        test_dataset: 测试集
        val_dataset: 验证集（Phase 1.2d 新增，None 时回退到 test_dataset 保持向后兼容）
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
        pin_memory: 是否 pin_memory
        persistent_workers: 是否持久化工作进程
        learning_mode: "supervised" 或 "self_supervised"
        unsupervised_dataset: 自监督模式下的无监督数据集（Phase 1 用）
        supervised_dataset: 自监督模式下的监督微调数据集（Phase 2 用）
        train_transform: Phase 2.1a 训练阶段变换函数 fn(x, y) -> (x, y)
        eval_transform: Phase 2.1a 验证/测试阶段变换函数
        streaming: Phase 2.1b 是否流式模式（train_dataset/test_dataset 为 IterableDataset）
        cache_dir: Phase 2.1c 缓存目录，None 不缓存
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        learning_mode: str = "supervised",
        unsupervised_dataset: Optional[Dataset] = None,
        supervised_dataset: Optional[Dataset] = None,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.learning_mode = learning_mode
        self.streaming = streaming
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # RFC Phase C：支持自定义 collate_fn（消除封装死点 E1）
        self.collate_fn = collate_fn

        # Phase 2.1a：应用变换包装（流式模式不包装，由 IterableDataset 内部处理）
        if not streaming:
            if train_transform is not None:
                train_dataset = _TransformWrapper(train_dataset, train_transform)
                # Phase 9.1.1 修复：自监督无监督数据也需 train_transform 包装，
                # 否则 raw numpy（float64）进入 self_supervised encoder 时与 Float 权重 dtype 不匹配。
                if unsupervised_dataset is not None:
                    unsupervised_dataset = _TransformWrapper(unsupervised_dataset, train_transform)
            if eval_transform is not None:
                test_dataset = _TransformWrapper(test_dataset, eval_transform)
                if val_dataset is not None:
                    val_dataset = _TransformWrapper(val_dataset, eval_transform)
                if supervised_dataset is not None:
                    supervised_dataset = _TransformWrapper(supervised_dataset, train_transform)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # Phase 1.2d：支持独立验证集，None 时回退到 test_dataset（向后兼容）
        self.val_dataset = val_dataset if val_dataset is not None else test_dataset
        self.unsupervised_dataset = unsupervised_dataset
        self.supervised_dataset = supervised_dataset

        # RFC-005：缓存 DataLoader 实例，使 persistent_workers 的 iterator 可被 teardown 关闭
        # 不缓存时每次 train_dataloader() 返回新 DataLoader，旧 iterator 被 gc 后
        # persistent_workers 的 IPC pipe fd 未被 close()，导致 +24 pipe/run 泄露。
        self._dl_cache: Dict[str, "DataLoader"] = {}

        # Phase 2.1c：数据缓存
        self._cached = False
        if self.cache_dir is not None:
            self._maybe_load_cache()

    # ============================================================
    # Phase 2.1c：缓存机制
    # ============================================================
    def _cache_key(self) -> str:
        """生成缓存键（基于数据集类型与大小）。"""
        parts = []
        for name, ds in [("train", self.train_dataset), ("test", self.test_dataset)]:
            try:
                n = len(ds)
            except TypeError:
                n = -1  # IterableDataset 无 len
            parts.append(f"{name}:{type(ds).__name__}:{n}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    def _maybe_load_cache(self) -> None:
        """尝试从缓存加载，命中则跳过后续数据加载。"""
        cache_file = self.cache_dir / f"datamodule_{self._cache_key()}.pt"
        if cache_file.exists():
            try:
                cached = torch.load(cache_file, weights_only=False)
                self.train_dataset = cached["train"]
                self.test_dataset = cached["test"]
                self.val_dataset = cached.get("val", self.test_dataset)
                self._cached = True
            except Exception:
                pass  # 缓存损坏，忽略

    def save_cache(self) -> Optional[Path]:
        """Phase 2.1c：将当前数据集序列化到缓存目录。

        Returns:
            缓存文件路径，未启用缓存返回 None
        """
        if self.cache_dir is None or self._cached:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"datamodule_{self._cache_key()}.pt"
        try:
            torch.save({
                "train": self.train_dataset,
                "test": self.test_dataset,
                "val": self.val_dataset,
            }, cache_file)
            return cache_file
        except Exception:
            return None

    def train_dataloader(self):
        if "train" in self._dl_cache:
            return self._dl_cache["train"]
        if self.learning_mode == "self_supervised" and self.unsupervised_dataset:
            dl = DataLoader(
                self.unsupervised_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        elif self.streaming:
            # 流式模式：IterableDataset 不支持 shuffle/drop_last
            dl = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
            )
        else:
            dl = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
            )
        self._dl_cache["train"] = dl
        return dl

    def val_dataloader(self):
        if "val" in self._dl_cache:
            return self._dl_cache["val"]
        # Phase 1.2d：使用独立的 val_dataset（无独立 val 时回退到 test_dataset）
        if self.streaming:
            dl = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
            )
        else:
            dl = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
            )
        self._dl_cache["val"] = dl
        return dl

    def test_dataloader(self):
        if "test" in self._dl_cache:
            return self._dl_cache["test"]
        if self.streaming:
            dl = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
            )
        else:
            dl = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
            )
        self._dl_cache["test"] = dl
        return dl

    def supervised_dataloader(self):
        """自监督模式下监督微调阶段使用的数据加载器。"""
        if self.supervised_dataset is None:
            raise RuntimeError("supervised_dataloader only available in self_supervised mode")
        if "supervised" in self._dl_cache:
            return self._dl_cache["supervised"]
        dl = DataLoader(
            self.supervised_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )
        self._dl_cache["supervised"] = dl
        return dl

    def teardown(self, stage: Optional[str] = None) -> None:
        """RFC-005：释放缓存的 DataLoader 引用。

        persistent_workers=True 时，DataModule 缓存 DataLoader（使 persistent_workers
        正确复用 worker）。teardown 时清空缓存，释放 DataLoader 引用。
        worker 进程的 pipe 由模块级 patch _patched_shutdown_workers 关闭。
        """
        self._dl_cache.clear()


__all__ = ["GenericDataModule"]
