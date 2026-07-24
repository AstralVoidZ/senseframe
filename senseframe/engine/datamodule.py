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
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl

# 修复（5.10）：GenericDataModule 构造参数无日志，加模块级 logger
_logger = logging.getLogger(__name__)


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
    # torch 2.13+ 移除了 _clean_up_worker，仅存在时才 patch
    _orig_clean_up_worker = getattr(
        _dl_mod._MultiProcessingDataLoaderIter, "_clean_up_worker", None
    )

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

    if _orig_clean_up_worker is not None:
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

        _dl_mod._MultiProcessingDataLoaderIter._clean_up_worker = _patched_clean_up_worker

    _dl_mod._MultiProcessingDataLoaderIter._shutdown_workers = _patched_shutdown_workers
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


def _np_random_worker_init_fn(worker_id: int):
    """DataLoader worker 初始化函数：为每个 worker 设置独立的 np.random 种子。

    P3 上策的额外保障：PyTorch DataLoader 默认仅为 torch RNG 派生 per-worker 种子，
    不为 np.random 派生。此函数从 torch 的 per-worker 种子派生 np.random 种子，
    确保即使原语未使用注入的 rng（如第三方原语），worker 间 np.random 状态也不同。

    注意：ComposedTransform 持有的独立 Generator 是主要的随机性隔离机制；
    此函数是防御性兜底，处理未走注入路径的原语。
    """
    import numpy as np
    # torch.initial_seed() 返回 DataLoader 为当前 worker 派生的种子
    # （base_seed + worker_id，base_seed 由 torch.manual_seed 设置）
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


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
        supervised_transform: Optional[Callable] = None,
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
            # P2-2 上策：supervised_dataset 使用独立的 supervised_transform（若提供），
            # 否则回退到 train_transform。消除 supervised_dataset 与 train_transform 的
            # 隐式耦合，允许微调阶段使用与预训练不同的增强强度。
            # P2-1 修复：supervised_dataset 是 Phase 2 微调的训练数据，必须被包装；
            # 旧代码错误地放在 eval_transform 分支内，若 eval_transform=None 但
            # train_transform!=None（如 augment-without-pipeline 配置）会漏包装。
            if supervised_dataset is not None:
                sup_tf = supervised_transform if supervised_transform is not None else train_transform
                if sup_tf is not None:
                    supervised_dataset = _TransformWrapper(supervised_dataset, sup_tf)
            if eval_transform is not None:
                test_dataset = _TransformWrapper(test_dataset, eval_transform)
                if val_dataset is not None:
                    val_dataset = _TransformWrapper(val_dataset, eval_transform)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # Phase 1.2d：支持独立验证集，None 时回退到 test_dataset（向后兼容）
        # P2 修复：回退时无日志，early_stopping 会监控 test 指标而非 val，静默回退使行为不可观测。
        if val_dataset is not None:
            self.val_dataset = val_dataset
        else:
            self.val_dataset = test_dataset
            _logger.warning(
                "val dataset not available, falling back to test set; "
                "early_stopping will monitor test metrics"
            )
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

        # 修复（5.10）：GenericDataModule 构造参数无日志
        # 旧逻辑：setup()/__init__ 无日志，无法追踪 batch_size / num_workers /
        # persistent_workers 等关键参数（影响 DataLoader 行为 + 资源占用）。
        # 由于 GenericDataModule 无 setup() 方法，在 __init__ 末尾记录关键参数。
        _logger.info(
            "GenericDataModule constructed: batch_size=%d, num_workers=%d, "
            "pin_memory=%s, persistent_workers=%s, learning_mode=%s, "
            "streaming=%s, cache_dir=%s, has_collate_fn=%s",
            self.batch_size, self.num_workers,
            self.pin_memory, self.persistent_workers,
            self.learning_mode, self.streaming,
            str(self.cache_dir) if self.cache_dir else "None",
            self.collate_fn is not None,
        )

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
            except Exception as e:
                # 缓存是优化手段，读失败不影响正确性，debug 级别即可
                _logger.debug("Cache load failed (will reload data): %s", e)

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
        except Exception as e:
            # 缓存写失败可能影响下次启动性能，warning 级别
            _logger.warning("Cache save failed (next startup will reload data): %s", e)
            return None

    def _safe_dataloader(self, dataset, **dl_kwargs) -> "DataLoader":
        """创建 DataLoader，多进程启动失败时自动降级到 num_workers=0。

        兑现 routing.py 的注释承诺："如遇 multiprocessing 问题，回退到 0"。
        触发场景：Python 3.14+ Linux forkserver 模式 + python -c/REPL 无法
        重新导入 <stdin> 主模块；或 worker 用对象不可 pickle。

        Args:
            dataset: 传给 DataLoader 的 dataset
            **dl_kwargs: DataLoader 关键字参数（num_workers/pin_memory/...）

        Returns:
            DataLoader 实例（原配置或降级到 num_workers=0）
        """
        try:
            dl = DataLoader(dataset, **dl_kwargs)
            # num_workers>0 时触发一次迭代以提早发现 worker 启动失败
            if dl_kwargs.get("num_workers", 0) > 0:
                _ = next(iter(dl))
            return dl
        except (ConnectionResetError, RuntimeError, OSError) as e:
            _logger.warning(
                "DataLoader multi-process failed (%s), retry with num_workers=0", e
            )
            # 降级到 num_workers=0
            self.num_workers = 0
            self.persistent_workers = False
            dl_kwargs["num_workers"] = 0
            dl_kwargs["persistent_workers"] = False
            return DataLoader(dataset, **dl_kwargs)

    def train_dataloader(self):
        if "train" in self._dl_cache:
            return self._dl_cache["train"]
        if self.learning_mode == "self_supervised" and self.unsupervised_dataset:
            # 自监督数据集（unsupervised）返回标准 2-tuple (x, y)，
            # 增强在 SelfSupervisedModule._self_supervised_step 内部生成
            # （gaussian_noise x1/x2），而非由数据集返回三元组，故默认
            # collate 可正确批化，无需自定义 collate_fn。
            # 但仍透传 self.collate_fn（默认 None=默认 collate）以保持与
            # 监督分支一致，并允许用户注入 batch 级增强（如 mixup collate）。
            dl = self._safe_dataloader(
                self.unsupervised_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
            actual_dataset = self.unsupervised_dataset
        elif self.streaming:
            # 流式模式：IterableDataset 不支持 shuffle/drop_last
            dl = self._safe_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
            actual_dataset = self.train_dataset
        else:
            dl = self._safe_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
            actual_dataset = self.train_dataset
        # P1-3: 首次创建 dataloader 时打印摘要（缓存命中路径不重复打印）
        try:
            n_samples = len(actual_dataset) if actual_dataset is not None else 0
        except TypeError:
            n_samples = -1  # IterableDataset 无 len
        _logger.info(
            "DataLoader(train): dataset_samples=%d, batch_size=%d, "
            "num_workers=%d, pin_memory=%s, persistent_workers=%s",
            n_samples, self.batch_size, self.num_workers,
            self.pin_memory, self.persistent_workers,
        )
        self._dl_cache["train"] = dl
        return dl

    def val_dataloader(self):
        if "val" in self._dl_cache:
            return self._dl_cache["val"]
        # Phase 1.2d：使用独立的 val_dataset（无独立 val 时回退到 test_dataset）
        if self.streaming:
            dl = self._safe_dataloader(
                self.val_dataset,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
        else:
            dl = self._safe_dataloader(
                self.val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
        # P1-3: 首次创建 dataloader 时打印摘要
        try:
            n_samples = len(self.val_dataset) if self.val_dataset is not None else 0
        except TypeError:
            n_samples = -1
        _logger.info(
            "DataLoader(val): dataset_samples=%d, batch_size=%d, "
            "num_workers=%d, pin_memory=%s, persistent_workers=%s",
            n_samples, self.batch_size * 2, self.num_workers,
            self.pin_memory, self.persistent_workers,
        )
        self._dl_cache["val"] = dl
        return dl

    def test_dataloader(self):
        if "test" in self._dl_cache:
            return self._dl_cache["test"]
        if self.streaming:
            dl = self._safe_dataloader(
                self.test_dataset,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
        else:
            dl = self._safe_dataloader(
                self.test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=_np_random_worker_init_fn,
            )
        # P1-3: 首次创建 dataloader 时打印摘要
        try:
            n_samples = len(self.test_dataset) if self.test_dataset is not None else 0
        except TypeError:
            n_samples = -1
        _logger.info(
            "DataLoader(test): dataset_samples=%d, batch_size=%d, "
            "num_workers=%d, pin_memory=%s, persistent_workers=%s",
            n_samples, self.batch_size * 2, self.num_workers,
            self.pin_memory, self.persistent_workers,
        )
        self._dl_cache["test"] = dl
        return dl

    def supervised_dataloader(self):
        """自监督模式下监督微调阶段使用的数据加载器。"""
        if self.supervised_dataset is None:
            raise RuntimeError("supervised_dataloader only available in self_supervised mode")
        if "supervised" in self._dl_cache:
            return self._dl_cache["supervised"]
        dl = self._safe_dataloader(
            self.supervised_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=_np_random_worker_init_fn,
        )
        # P1-3: 首次创建 dataloader 时打印摘要
        try:
            n_samples = len(self.supervised_dataset) if self.supervised_dataset is not None else 0
        except TypeError:
            n_samples = -1
        _logger.info(
            "DataLoader(supervised): dataset_samples=%d, batch_size=%d, "
            "num_workers=%d, pin_memory=%s, persistent_workers=%s",
            n_samples, self.batch_size, self.num_workers,
            self.pin_memory, self.persistent_workers,
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
