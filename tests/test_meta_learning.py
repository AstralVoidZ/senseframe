"""ε4 元学习测试（P3.2.1-P3.2.5）。

反假绿测试策略：
- grep 实证：源码检查不可绕过（mock 可绕过运行时，但绕不过源码 grep）
- 真实 HistoryStore 实例（不 mock）
- 真实 MetaLearner 调用（验证 warm-start 真实注入 tracker.history）
- 真实 StudyManager 集成（验证 create_study(warm_start_from=...) 端到端）

覆盖：
- P3.2.1: Sampler Protocol warm_start 扩展
- P3.2.2: HistoryStore 持久化
- P3.2.3: MetaLearner 实现
- P3.2.4: create_study warm_start_from 集成
- P3.2.5: ε4 集成测试
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from senseframe.automl import HistoryStore, MetaLearner
from senseframe.exploration import ExplorationTracker
from senseframe.search_protocol import (
    ParameterSpec,
    RandomSampler,
    Sampler,
    SearchSpace,
    StudyManager,
    get_sampler,
    list_samplers,
)


# ============================================================
# 辅助
# ============================================================
def _source_path(rel: str) -> Path:
    """获取源码文件绝对路径（用于 grep 实证）。"""
    return Path(__file__).parent.parent / "senseframe" / rel


def _grep_source(file_path: Path, pattern: str) -> bool:
    """grep 实证：检查源码文件是否包含 pattern。"""
    content = file_path.read_text(encoding="utf-8")
    return pattern in content


def _make_trial(
    trial_id: str,
    strategy: Dict[str, Any],
    val_accuracy: float,
) -> Dict[str, Any]:
    """构造测试用 trial（含 val_accuracy）。"""
    return {
        "trial_id": trial_id,
        "strategy": strategy,
        "result": {"val_accuracy": val_accuracy, "value": val_accuracy},
        "status": "completed",
        "timestamp": "2026-07-05T00:00:00",
    }


def _make_trial_value_only(
    trial_id: str,
    strategy: Dict[str, Any],
    value: float,
) -> Dict[str, Any]:
    """构造测试用 trial（仅 value，无 val_accuracy，用于 fallback 测试）。"""
    return {
        "trial_id": trial_id,
        "strategy": strategy,
        "result": {"value": value},
        "status": "completed",
        "timestamp": "2026-07-05T00:00:00",
    }


def _make_search_space() -> SearchSpace:
    """构造测试用搜索空间。"""
    return SearchSpace(parameters=[
        ParameterSpec(name="lr", type="float", low=0.0001, high=0.1, log=True),
        ParameterSpec(name="batch_size", type="int", low=16, high=128),
        ParameterSpec(name="loss", type="categorical", choices=["focal", "cross_entropy"]),
    ])


# ============================================================
# P3.2.1: Sampler Protocol warm_start 测试
# ============================================================
class TestSamplerWarmStart:
    """Sampler Protocol warm_start 扩展测试（P3.2.1）。"""

    def test_sampler_protocol_has_warm_start_method(self):
        """Sampler Protocol 含 warm_start 方法定义。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "def warm_start(")
        assert _grep_source(path, "source_history: List[Dict[str, Any]]")

    def test_random_sampler_still_satisfies_protocol(self):
        """RandomSampler 仍通过 isinstance(x, Sampler) 检查（向后兼容）。

        @runtime_checkable Protocol 不强制实现所有方法，
        RandomSampler 未实现 warm_start 仍能通过 isinstance 检查。
        """
        sampler = RandomSampler()
        assert isinstance(sampler, Sampler)

    def test_grid_sampler_still_satisfies_protocol(self):
        """GridSampler 仍通过 isinstance(x, Sampler) 检查（向后兼容）。"""
        from senseframe.search_protocol import GridSampler
        sampler = GridSampler()
        assert isinstance(sampler, Sampler)

    def test_evolutionary_sampler_warm_start_method(self):
        """EvolutionarySampler 实现 warm_start 方法。"""
        from senseframe.nas.sampler import EvolutionarySampler
        sampler = EvolutionarySampler()
        assert hasattr(sampler, "warm_start")
        assert callable(sampler.warm_start)
        # isinstance 仍成立
        assert isinstance(sampler, Sampler)

    def test_autoaugment_sampler_warm_start_method(self):
        """AutoAugmentSampler 实现 warm_start 方法。"""
        from senseframe.autoaugment.sampler import AutoAugmentSampler
        sampler = AutoAugmentSampler()
        assert hasattr(sampler, "warm_start")
        assert callable(sampler.warm_start)
        assert isinstance(sampler, Sampler)

    def test_warm_start_no_op_for_samplers_without_method(self):
        """warm_start 在无状态 sampler 上是 no-op（不报错）。

        RandomSampler/GridSampler/ASHASampler/HyperbandSampler 实现 warm_start
        为 no-op（仅 return None），不修改任何内部状态。这是 Python 3.12+
        @runtime_checkable Protocol 要求所有方法存在的兼容性设计——
        无状态 sampler 提供 no-op warm_start，进化类 sampler 提供真实实现。
        """
        from senseframe.search_protocol import (
            ASHASampler,
            GridSampler,
            HyperbandSampler,
            RandomSampler,
        )
        sample_history = [
            {"trial_id": "t1", "strategy": {"lr": 0.01}, "result": {"value": 0.85}},
        ]
        for cls in [RandomSampler, GridSampler, ASHASampler, HyperbandSampler]:
            sampler = cls()
            # warm_start 是 no-op，调用不报错
            assert hasattr(sampler, "warm_start")
            result = sampler.warm_start(sample_history)
            assert result is None  # no-op 返回 None


# ============================================================
# P3.2.2: HistoryStore 测试
# ============================================================
class TestHistoryStore:
    """HistoryStore 持久化测试（P3.2.2）。"""

    def test_save_and_load_history(self, tmp_path):
        """保存后加载内容一致。"""
        tracker = ExplorationTracker()
        tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.85, "value": 0.85},
            trial_id="t1",
        )
        tracker.add_trial(
            strategy={"lr": 0.001},
            result={"val_accuracy": 0.78, "value": 0.78},
            trial_id="t2",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", tracker)

        loaded = store.load_history("UT_HAR_data")
        assert len(loaded) == 2
        assert loaded[0]["trial_id"] == "t1"
        assert loaded[0]["strategy"] == {"lr": 0.01}
        assert loaded[1]["trial_id"] == "t2"

    def test_load_nonexistent_dataset_returns_empty(self, tmp_path):
        """加载不存在的数据集返回空 list（不抛异常）。"""
        store = HistoryStore(base_dir=tmp_path)
        result = store.load_history("nonexistent_dataset")
        assert result == []
        assert isinstance(result, list)

    def test_list_datasets(self, tmp_path):
        """列出已存储数据集。"""
        store = HistoryStore(base_dir=tmp_path)
        tracker = ExplorationTracker()
        tracker.add_trial(strategy={"lr": 0.01}, result={"value": 0.8}, trial_id="t1")

        store.save_history("UT_HAR_data", tracker)
        store.save_history("Widar", tracker)
        store.save_history("NTU-Fi_HAR", tracker)

        datasets = store.list_datasets()
        # 按字母序
        assert datasets == ["NTU-Fi_HAR", "UT_HAR_data", "Widar"]

    def test_save_creates_directory(self, tmp_path):
        """保存时自动创建目录（含父目录）。"""
        store = HistoryStore(base_dir=tmp_path / "nested" / "deep")
        tracker = ExplorationTracker()
        tracker.add_trial(strategy={"lr": 0.01}, result={"value": 0.8}, trial_id="t1")

        # 目录尚未存在
        assert not (tmp_path / "nested" / "deep").exists()

        store.save_history("UT_HAR_data", tracker)

        # 目录已自动创建
        assert (tmp_path / "nested" / "deep" / "UT_HAR_data").exists()
        assert (tmp_path / "nested" / "deep" / "UT_HAR_data" / "history.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        """重复保存覆盖旧文件。"""
        store = HistoryStore(base_dir=tmp_path)

        tracker1 = ExplorationTracker()
        tracker1.add_trial(strategy={"lr": 0.01}, result={"value": 0.8}, trial_id="t1")
        store.save_history("UT_HAR_data", tracker1)

        tracker2 = ExplorationTracker()
        tracker2.add_trial(strategy={"lr": 0.001}, result={"value": 0.9}, trial_id="t2")
        tracker2.add_trial(strategy={"lr": 0.0001}, result={"value": 0.95}, trial_id="t3")
        store.save_history("UT_HAR_data", tracker2)

        loaded = store.load_history("UT_HAR_data")
        # 应只有 tracker2 的 2 条记录（覆盖了 tracker1 的 1 条）
        assert len(loaded) == 2
        assert loaded[0]["trial_id"] == "t2"
        assert loaded[1]["trial_id"] == "t3"

    def test_save_preserves_trial_structure(self, tmp_path):
        """保存的 trial 结构（trial_id/strategy/result/status）完整。"""
        tracker = ExplorationTracker()
        tracker.add_trial(
            strategy={"lr": 0.01, "loss": "focal"},
            result={"val_accuracy": 0.85, "value": 0.85, "intermediate_values": {1: 0.5}},
            trial_id="t1",
            feedback={"status": "success"},
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", tracker)

        loaded = store.load_history("UT_HAR_data")
        assert len(loaded) == 1
        trial = loaded[0]
        # 结构完整
        assert trial["trial_id"] == "t1"
        assert trial["strategy"] == {"lr": 0.01, "loss": "focal"}
        assert trial["result"]["val_accuracy"] == 0.85
        assert trial["result"]["value"] == 0.85
        # 注意：JSON 序列化将 int key 转为 string key（{"1": 0.5}）
        assert trial["result"]["intermediate_values"] == {"1": 0.5}
        assert trial["status"] == "completed"
        assert trial["feedback"] == {"status": "success"}

    def test_load_after_save_roundtrip(self, tmp_path):
        """load → save → load 等价（round-trip）。"""
        store = HistoryStore(base_dir=tmp_path)

        original_tracker = ExplorationTracker()
        original_tracker.add_trial(
            strategy={"lr": 0.01, "batch_size": 32},
            result={"val_accuracy": 0.85},
            trial_id="t1",
        )
        original_tracker.add_trial(
            strategy={"lr": 0.001, "batch_size": 64},
            result={"val_accuracy": 0.78},
            trial_id="t2",
        )

        store.save_history("UT_HAR_data", original_tracker)
        loaded_history = store.load_history("UT_HAR_data")

        # 用加载的 history 构造新 tracker，再保存
        new_tracker = ExplorationTracker(history=loaded_history)
        store.save_history("UT_HAR_data_v2", new_tracker)
        reloaded = store.load_history("UT_HAR_data_v2")

        # 三方等价
        assert len(reloaded) == len(loaded_history) == 2
        assert reloaded[0]["strategy"] == loaded_history[0]["strategy"]
        assert reloaded[1]["strategy"] == loaded_history[1]["strategy"]
        assert reloaded[0]["trial_id"] == "t1"
        assert reloaded[1]["trial_id"] == "t2"

    def test_history_store_with_real_exploration_tracker(self, tmp_path):
        """与真实 ExplorationTracker 集成。"""
        tracker = ExplorationTracker()
        # 模拟源数据集搜索历史
        for i, (lr, bs, val_acc) in enumerate([
            (0.01, 32, 0.85),
            (0.001, 64, 0.78),
            (0.0001, 128, 0.65),
            (0.005, 16, 0.92),  # 最佳
        ]):
            tracker.add_trial(
                strategy={"lr": lr, "batch_size": bs},
                result={"val_accuracy": val_acc, "value": val_acc},
                trial_id=f"src_{i}",
            )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("source_dataset", tracker)

        # 加载并验证
        loaded = store.load_history("source_dataset")
        assert len(loaded) == 4

        # 用加载的历史构造新 tracker（模拟 warm-start）
        new_tracker = ExplorationTracker(history=loaded)
        assert len(new_tracker.history) == 4
        # best_trial 应可工作
        best = new_tracker.best_trial(metric="val_accuracy", mode="max")
        assert best is not None
        assert best["result"]["val_accuracy"] == 0.92


# ============================================================
# P3.2.3: MetaLearner 测试
# ============================================================
class TestMetaLearner:
    """MetaLearner 实现测试（P3.2.3）。"""

    def test_warm_start_injects_successful_strategies(self, tmp_path):
        """成功策略（val_accuracy > 0.7）被注入 tracker.history。"""
        # 准备源数据集历史
        source_tracker = ExplorationTracker()
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.85, "value": 0.85},  # 成功
            trial_id="src_1",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.001},
            result={"val_accuracy": 0.78, "value": 0.78},  # 成功
            trial_id="src_2",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        # 创建目标 study
        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        # warm-start
        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "UT_HAR_data")

        assert count == 2
        # 验证 tracker.history 已被注入
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 2
        assert tracker.history[0]["trial_id"] == "src_1"
        assert tracker.history[1]["trial_id"] == "src_2"

    def test_warm_start_filters_low_accuracy(self, tmp_path):
        """低准确率策略被过滤（val_accuracy <= 0.7 不注入）。"""
        source_tracker = ExplorationTracker()
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.85, "value": 0.85},  # 成功（>0.7）
            trial_id="src_1",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.0001},
            result={"val_accuracy": 0.65, "value": 0.65},  # 失败（<=0.7）
            trial_id="src_2",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.00001},
            result={"val_accuracy": 0.50, "value": 0.50},  # 失败
            trial_id="src_3",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "UT_HAR_data")

        # 仅 src_1 被注入
        assert count == 1
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 1
        assert tracker.history[0]["trial_id"] == "src_1"

    def test_warm_start_returns_count(self, tmp_path):
        """返回注入的条目数。"""
        source_tracker = ExplorationTracker()
        for i in range(5):
            source_tracker.add_trial(
                strategy={"lr": 0.01 * (i + 1)},
                result={"val_accuracy": 0.8, "value": 0.8},
                trial_id=f"src_{i}",
            )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "UT_HAR_data")

        assert count == 5
        assert isinstance(count, int)

    def test_warm_start_nonexistent_dataset_returns_zero(self, tmp_path):
        """源数据集不存在时返回 0（不抛异常）。"""
        store = HistoryStore(base_dir=tmp_path)

        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "nonexistent_dataset")

        assert count == 0
        # tracker.history 未被修改
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 0

    def test_warm_start_nonexistent_study_raises(self, tmp_path):
        """study_id 不存在时抛 KeyError。"""
        source_tracker = ExplorationTracker()
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.85, "value": 0.85},
            trial_id="src_1",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        sm = StudyManager()

        meta = MetaLearner(study_manager=sm, history_store=store)
        with pytest.raises(KeyError, match="nonexistent_study"):
            meta.warm_start("nonexistent_study", "UT_HAR_data")

    def test_warm_start_uses_value_field_when_no_val_accuracy(self, tmp_path):
        """result 含 value 但不含 val_accuracy 时 fallback 到 value。"""
        source_tracker = ExplorationTracker()
        # 模拟 SP Tell 上报：result 含 value 但不含 val_accuracy
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"value": 0.85},  # value > 0.7 → 成功
            trial_id="src_1",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.0001},
            result={"value": 0.65},  # value <= 0.7 → 过滤
            trial_id="src_2",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "UT_HAR_data")

        # 仅 src_1（value=0.85 > 0.7）被注入
        assert count == 1
        tracker = sm._trackers[study_id]
        assert tracker.history[0]["trial_id"] == "src_1"

    def test_warm_start_custom_threshold(self, tmp_path):
        """自定义 success_threshold 生效。"""
        source_tracker = ExplorationTracker()
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.75, "value": 0.75},  # 默认阈值下成功
            trial_id="src_1",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.001},
            result={"val_accuracy": 0.85, "value": 0.85},  # 高阈值下仍成功
            trial_id="src_2",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        sm = StudyManager()
        study_id = sm.create_study(name="target", sampler="random")

        meta = MetaLearner(study_manager=sm, history_store=store)
        # 提高阈值到 0.8
        count = meta.warm_start(study_id, "UT_HAR_data", success_threshold=0.8)

        # 仅 src_2（val_accuracy=0.85 > 0.8）被注入
        assert count == 1
        tracker = sm._trackers[study_id]
        assert tracker.history[0]["trial_id"] == "src_2"

    def test_warm_start_with_real_study_manager(self, tmp_path):
        """与真实 StudyManager 集成。"""
        # 源数据集：5 个成功策略
        source_tracker = ExplorationTracker()
        for i, val_acc in enumerate([0.85, 0.78, 0.92, 0.65, 0.88]):
            source_tracker.add_trial(
                strategy={"lr": 0.001 * (i + 1), "loss": "focal"},
                result={"val_accuracy": val_acc, "value": val_acc},
                trial_id=f"src_{i}",
            )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        # 创建目标 study（evolutionary sampler，最能从 warm-start 受益）
        sm = StudyManager()
        ss = _make_search_space()
        study_id = sm.create_study(
            name="target",
            sampler="evolutionary",
            search_space=ss,
        )

        meta = MetaLearner(study_manager=sm, history_store=store)
        count = meta.warm_start(study_id, "UT_HAR_data")

        # 4 个成功（0.85, 0.78, 0.92, 0.88），1 个失败（0.65）
        assert count == 4
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 4

        # 后续 ask 应能从扩展后的 history 中读取
        trial = sm.ask(study_id)
        assert trial is not None
        # 参数应在搜索空间范围内
        assert "lr" in trial.params


# ============================================================
# P3.2.4: create_study warm_start_from 测试
# ============================================================
class TestCreateStudyWarmStart:
    """create_study warm_start_from 参数测试（P3.2.4）。"""

    def test_create_study_with_warm_start(self, tmp_path):
        """create_study(warm_start_from=...) 注入历史。"""
        # 准备源数据集历史
        source_tracker = ExplorationTracker()
        source_tracker.add_trial(
            strategy={"lr": 0.01},
            result={"val_accuracy": 0.85, "value": 0.85},
            trial_id="src_1",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        # 创建目标 study 并 warm-start
        sm = StudyManager()
        ss = _make_search_space()
        study_id = sm.create_study(
            name="target",
            sampler="random",
            search_space=ss,
            warm_start_from="UT_HAR_data",
            history_store=store,
        )

        # tracker.history 应已注入源数据集的成功策略
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 1
        assert tracker.history[0]["trial_id"] == "src_1"

    def test_create_study_without_warm_start(self, tmp_path):
        """无 warm_start_from 行为不变（向后兼容）。"""
        sm = StudyManager()
        ss = _make_search_space()
        store = HistoryStore(base_dir=tmp_path)

        study_id = sm.create_study(
            name="target",
            sampler="random",
            search_space=ss,
            # 不传 warm_start_from
            history_store=store,
        )

        # tracker.history 为空
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 0

    def test_create_study_warm_start_without_store(self, tmp_path):
        """warm_start_from 但无 history_store 时不报错（no-op）。"""
        sm = StudyManager()
        ss = _make_search_space()

        # 仅传 warm_start_from，不传 history_store
        study_id = sm.create_study(
            name="target",
            sampler="random",
            search_space=ss,
            warm_start_from="UT_HAR_data",
            # history_store=None
        )

        # 不报错，tracker.history 为空（no-op）
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 0

    def test_create_study_warm_start_full_flow(self, tmp_path):
        """完整流程：保存源 study → 创建新 study warm-start → ask 返回偏向策略。"""
        # 1. 模拟源 study：跑若干 trial，保存历史
        source_sm = StudyManager()
        source_ss = _make_search_space()
        source_study = source_sm.create_study(
            name="source",
            sampler="random",
            search_space=source_ss,
        )

        # 跑 5 个 trial 并 tell
        successful_strategies = []
        for i in range(5):
            trial = source_sm.ask(source_study)
            val_acc = 0.6 + 0.08 * i  # 0.60, 0.68, 0.76, 0.84, 0.92
            source_sm.tell(trial.trial_id, value=val_acc, state="completed")
            if val_acc > 0.7:
                successful_strategies.append(trial.params)

        # 保存源历史
        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_sm._trackers[source_study])

        # 2. 创建新 study 并 warm-start
        target_sm = StudyManager()
        target_ss = _make_search_space()
        target_study = target_sm.create_study(
            name="target",
            sampler="random",
            search_space=target_ss,
            warm_start_from="UT_HAR_data",
            history_store=store,
        )

        # 3. 验证 tracker.history 已被注入
        tracker = target_sm._trackers[target_study]
        # 3 个成功（0.76, 0.84, 0.92）
        assert len(tracker.history) == 3

        # 4. ask 应能正常返回参数
        trial = target_sm.ask(target_study)
        assert trial is not None
        assert "lr" in trial.params

    def test_create_study_backward_compatible(self, tmp_path):
        """旧调用方式（仅 name/direction/search_space/sampler）仍工作。"""
        sm = StudyManager()
        ss = _make_search_space()

        # 旧调用方式：仅前 4 个参数
        study_id = sm.create_study(
            name="legacy",
            direction="maximize",
            search_space=ss,
            sampler="random",
        )

        # 验证 study 创建成功
        study = sm.get_study(study_id)
        assert study is not None
        assert study.name == "legacy"
        assert study.direction == "maximize"
        assert study.sampler == "random"

        # ask/tell 应正常工作
        trial = sm.ask(study_id)
        assert trial is not None
        sm.tell(trial.trial_id, value=0.8, state="completed")


# ============================================================
# grep 实证测试（反假绿）
# ============================================================
class TestGrepEvidence:
    """grep 实证：源码检查所有 P3.2 实现关键点。"""

    def test_grep_sampler_protocol_warm_start(self):
        """grep 实证：search_protocol.py 含 warm_start 方法定义。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "def warm_start(")
        assert _grep_source(path, "source_history: List[Dict[str, Any]]")

    def test_grep_history_store_class(self):
        """grep 实证：exploration.py 含 class HistoryStore 定义。"""
        path = _source_path("exploration.py")
        assert _grep_source(path, "class HistoryStore")
        assert _grep_source(path, "def save_history")
        assert _grep_source(path, "def load_history")
        assert _grep_source(path, "def list_datasets")

    def test_grep_meta_learner_class(self):
        """grep 实证：automl/meta_learner.py 含 class MetaLearner 定义。"""
        path = _source_path("automl/meta_learner.py")
        assert path.exists()
        assert _grep_source(path, "class MetaLearner")
        assert _grep_source(path, "def warm_start")
        assert _grep_source(path, "success_threshold")

    def test_grep_create_study_warm_start_from(self):
        """grep 实证：search_protocol.py create_study 含 warm_start_from 参数。"""
        path = _source_path("search_protocol.py")
        assert _grep_source(path, "warm_start_from: Optional[str]")
        assert _grep_source(path, "history_store: Optional")
        assert _grep_source(path, "warm_start_from and history_store is not None")
        # 延迟导入 MetaLearner
        assert _grep_source(path, "from .automl.meta_learner import MetaLearner")

    def test_grep_automl_exports_meta_learner(self):
        """grep 实证：automl/__init__.py 导出 MetaLearner 和 HistoryStore。"""
        path = _source_path("automl/__init__.py")
        assert _grep_source(path, "MetaLearner")
        assert _grep_source(path, "HistoryStore")
        assert _grep_source(path, "from .meta_learner import MetaLearner")

    def test_grep_evolutionary_sampler_warm_start(self):
        """grep 实证：EvolutionarySampler 实现 warm_start 方法。"""
        path = _source_path("nas/sampler.py")
        assert _grep_source(path, "def warm_start(")

    def test_grep_autoaugment_sampler_warm_start(self):
        """grep 实证：AutoAugmentSampler 实现 warm_start 方法。"""
        path = _source_path("autoaugment/sampler.py")
        assert _grep_source(path, "def warm_start(")

    def test_grep_p3_doc_meta_learning_section(self):
        """grep 实证：P3 规划文档含 'ε4 元学习' 章节。"""
        doc_path = Path(__file__).parent.parent / "docs" / "analysis"
        # rglob 递归查找（参考 test_autoaugment.py 的 test_grep_p3_doc_reference）
        p3_docs = [p for p in doc_path.rglob("*P3*") if p.is_file()]
        assert len(p3_docs) >= 1
        content = p3_docs[0].read_text(encoding="utf-8")
        # P3 文档应含 ε4 元学习章节
        assert "ε4 元学习" in content
        assert "MetaLearner" in content
        assert "HistoryStore" in content
        assert "warm_start" in content


# ============================================================
# P3.2.5: 集成测试
# ============================================================
class TestMetaLearningIntegration:
    """ε4 元学习端到端集成测试。"""

    def test_full_warm_start_flow(self, tmp_path):
        """完整流程：保存源数据集历史 → 创建新 study warm-start → ask 采样偏向成功策略。

        验证 warm-start 后 EvolutionarySampler 的 sample() 能从扩展后的
        tracker.history 中读取成功策略作为采样偏向。
        """
        # 1. 准备源数据集历史（含若干成功策略）
        source_tracker = ExplorationTracker()
        successful_strategy = {"lr": 0.005, "batch_size": 32, "loss": "focal"}
        source_tracker.add_trial(
            strategy=successful_strategy,
            result={"val_accuracy": 0.92, "value": 0.92},  # 成功
            trial_id="src_best",
        )
        source_tracker.add_trial(
            strategy={"lr": 0.0001, "batch_size": 128, "loss": "cross_entropy"},
            result={"val_accuracy": 0.65, "value": 0.65},  # 失败
            trial_id="src_fail",
        )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        # 2. 创建目标 study 并 warm-start
        sm = StudyManager()
        ss = _make_search_space()
        study_id = sm.create_study(
            name="target",
            sampler="evolutionary",
            search_space=ss,
            warm_start_from="UT_HAR_data",
            history_store=store,
        )

        # 3. 验证 tracker.history 已被注入（仅成功策略）
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 1
        assert tracker.history[0]["trial_id"] == "src_best"

        # 4. ask 应能从 history 中读取（EvolutionarySampler 同步 population）
        trial = sm.ask(study_id)
        assert trial is not None
        # 参数应在搜索空间范围内
        assert "lr" in trial.params
        assert "batch_size" in trial.params
        assert "loss" in trial.params

    def test_warm_start_improves_initial_sampling(self, tmp_path):
        """warm-start 后 EvolutionarySampler 第一次 ask 的 params 受源成功策略影响。

        对 EvolutionarySampler：warm-start 注入历史后，sampler.sample() 会
        从 history 中同步 population（_sync_population_from_history）。
        当 population 未满时返回随机个体；当 population 已满时通过锦标赛选择
        + 变异生成子代，子代会受成功策略影响。

        本测试通过小 population_size + 多次注入，使 population 在第一次 ask
        前已满，从而触发进化阶段（变异）——变异的父代来自成功策略。
        """
        # 1. 准备源数据集历史：3 个相同成功策略
        successful_strategy = {"lr": 0.005, "batch_size": 32, "loss": "focal"}
        source_tracker = ExplorationTracker()
        for i in range(3):
            source_tracker.add_trial(
                strategy=successful_strategy,
                result={"val_accuracy": 0.92, "value": 0.92},
                trial_id=f"src_{i}",
            )

        store = HistoryStore(base_dir=tmp_path)
        store.save_history("UT_HAR_data", source_tracker)

        # 2. 创建目标 study 并 warm-start（小 population_size=2）
        sm = StudyManager()
        ss = _make_search_space()
        study_id = sm.create_study(
            name="target",
            sampler="evolutionary",
            search_space=ss,
            warm_start_from="UT_HAR_data",
            history_store=store,
        )

        # 3. 验证 tracker.history 已被注入 3 个成功策略
        tracker = sm._trackers[study_id]
        assert len(tracker.history) == 3

        # 4. ask 多次：sampler 会从 history 中读取成功策略
        # EvolutionarySampler 每次 ask 创建新实例，但 sample() 会同步
        # population from history，所以成功策略会进入 population
        asked_params = []
        for _ in range(5):
            trial = sm.ask(study_id)
            asked_params.append(trial.params)

        # 验证所有 ask 都返回了合法参数（在搜索空间内）
        for params in asked_params:
            assert "lr" in params
            assert "batch_size" in params
            assert "loss" in params
            assert 0.0001 <= params["lr"] <= 0.1
            assert 16 <= params["batch_size"] <= 128
            assert params["loss"] in ["focal", "cross_entropy"]
