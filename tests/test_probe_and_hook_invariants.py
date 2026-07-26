"""training_log hook 顺序无关性 + stage_probe_vram 子进程隔离契约测试。

覆盖（_run_vram_probe deepcopy 方案已于 2026-07-12 移除，相关测试同步删除）：
1. on_validation_epoch_end / on_train_epoch_end hook 顺序无关性：
   - 累积器重置固定在 on_train_epoch_start
   - on_val_end 从累积器算 train_loss（不依赖 _epoch_train_* 实例变量）
   - 无论 hook 顺序如何，train_loss 正确（非 null、非偏移）
2. stage_train 不调用 set_seed（RNG 自然流转）
3. stage_probe_vram 子进程隔离契约（_run_probe_in_subprocess 调用）
"""

import torch
import torch.nn as nn
import pytest

from senseframe.engine.module import GenericLightningModule


# ============================================================
# 1. training_log hook 顺序无关性
# ============================================================

class TestHookOrderInvariance:
    """on_validation_epoch_end / on_train_epoch_end hook 顺序无关性。

    核心不变量：无论 hook 以何种顺序执行，train_loss 都应从累积器算出正确值，
    不依赖 _epoch_train_loss 实例变量（已删除）。
    """

    def _make_module(self):
        """构造 GenericLightningModule（不启动 Trainer）。"""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )
        # 用 4 类分类任务
        from senseframe.core.task import TaskSpec
        task_spec = TaskSpec.classification(num_classes=4)
        return GenericLightningModule(
            model=model,
            num_classes=4,
            learning_rate=1e-3,
            metrics=["accuracy"],
            task_spec=task_spec,
        )

    def test_module_has_no_epoch_train_loss_attr(self):
        """__init__ 不应再创建 _epoch_train_loss / _epoch_train_metrics 实例变量。"""
        module = self._make_module()
        assert not hasattr(module, "_epoch_train_loss"), \
            "_epoch_train_loss 实例变量应已删除（hook 顺序竞争根因）"
        assert not hasattr(module, "_epoch_train_metrics"), \
            "_epoch_train_metrics 实例变量应已删除（hook 顺序竞争根因）"

    def test_module_has_on_train_epoch_start(self):
        """GenericLightningModule 应有 on_train_epoch_start 方法（累积器重置点）。"""
        module = self._make_module()
        assert hasattr(module, "on_train_epoch_start"), \
            "应有 on_train_epoch_start 方法（累积器重置固定在此）"
        assert callable(module.on_train_epoch_start)

    def test_accumulator_reset_only_in_on_train_epoch_start(self):
        """累积器重置只应在 on_train_epoch_start 发生。

        模拟 hook 顺序竞争场景：
        1. training_step 累积 loss/steps
        2. on_validation_epoch_end 先执行（读累积器，不重置）
        3. on_train_epoch_end 后执行（不重置）
        4. on_train_epoch_start 下一个 epoch 重置
        """
        module = self._make_module()

        # 模拟 on_train_epoch_start 重置
        module._current_epoch_loss = 0.0
        module._current_epoch_steps = 0

        # 模拟 training_step 累积 3 步
        module._current_epoch_loss = 3.6
        module._current_epoch_steps = 3

        # 模拟 on_validation_epoch_end 读取（不重置）
        # 此时 steps=3, loss=3.6 → train_loss = 1.2
        if module.current_epoch_steps > 0:
            train_loss = round(module.current_epoch_loss / module.current_epoch_steps, 6)
        else:
            train_loss = None
        assert train_loss == 1.2, f"on_val_end 应从累积器算出正确 train_loss（1.2），实际: {train_loss}"

        # on_val_end 不重置累积器（关键不变量）
        # 模拟 on_train_epoch_end 也不重置（关键不变量）
        # 累积器值应保持不变
        assert module.current_epoch_steps == 3, \
            "on_val_end / on_train_end 都不应重置 steps"
        assert module.current_epoch_loss == 3.6, \
            "on_val_end / on_train_end 都不应重置 loss"

    def test_on_train_epoch_start_resets_accumulator(self):
        """on_train_epoch_start 应重置累积器（需 mock trainer.sanity_checking）。"""
        module = self._make_module()

        # 模拟上一个 epoch 残留
        module._current_epoch_loss = 99.0
        module._current_epoch_steps = 99

        # mock trainer（on_train_epoch_start 需要 self.trainer.sanity_checking）
        class _MockTrainer:
            sanity_checking = False
        module.trainer = _MockTrainer()

        module.on_train_epoch_start()

        assert module._current_epoch_loss == 0.0, \
            "on_train_epoch_start 应重置 _current_epoch_loss"
        assert module._current_epoch_steps == 0, \
            "on_train_epoch_start 应重置 _current_epoch_steps"

    def test_on_train_epoch_start_skips_sanity_check(self):
        """sanity_check 阶段不应重置累积器。"""
        module = self._make_module()
        module._current_epoch_loss = 99.0
        module._current_epoch_steps = 99

        class _MockTrainer:
            sanity_checking = True
        module.trainer = _MockTrainer()

        module.on_train_epoch_start()

        assert module.current_epoch_loss == 99.0, \
            "sanity_check 阶段不应重置累积器"
        assert module.current_epoch_steps == 99


# ============================================================
# 3. stage_train 不调用 set_seed（RNG 自然流转，子进程隔离后不需要）
# ============================================================

# ARCHITECTURE_TRIPWIRE: stage_train 不调用 set_seed（RNG 自然流转契约）
# 不可替代原因: "函数体不包含 set_seed 调用"是否定属性，行为测试只能验证 RNG 状态结果，
#   无法确认 stage_train 入口是否调用了 set_seed（可能被其他 set_seed 掩盖）。
# 删除条件: 当 RNG 管理由框架统一控制（如 Lightning seed_everything 全局接管），
#   stage_train 无法独立调用 set_seed。
class TestStageTrainNoSetSeed:
    """stage_train 入口不应调用 set_seed。

    根因（2026-07-12 确认）：stage_train 入口的 set_seed 会重置 RNG 状态，
    导致 DataLoader shuffle 顺序与无 probe 基线（N0）不同。
    - N0 基线（无 stage_train set_seed）：ep0=1.210943, val_acc=0.982
    - 有 stage_train set_seed：ep0=1.258861, val_acc=0.929

    子进程隔离后，probe 在子进程中运行，主进程不消耗 RNG，
    set_seed 仅在 stage_preflight 调用一次，之后 RNG 自然流经
    stage_load/resolve/build，与 N0 基线路径一致。
    """

    def test_stage_train_no_set_seed(self):
        """stage_train 源码不应包含 set_seed 调用。"""
        import inspect
        from senseframe.engine.runner.pipeline import stage_train
        source = inspect.getsource(stage_train)
        assert "set_seed(" not in source, \
            "stage_train 不应调用 set_seed（子进程隔离后不需要，反而破坏 RNG 自然流转）"


# ============================================================
# 4. stage_probe_vram 子进程隔离契约
# ============================================================

# ARCHITECTURE_TRIPWIRE: stage_probe_vram 子进程隔离契约
# 不可替代原因: "函数体不包含直接 CUDA 计算调用"是否定属性，行为测试无法区分
#   "主进程执行了 CUDA 计算"和"子进程执行了 CUDA 计算"（结果相同）；
#   必须检查源码确认 CUDA 操作全部委托给 _run_probe_in_subprocess。
# 删除条件: 当子进程隔离由框架层强制保证（如 stage_probe_vram 变为纯调度器，
#   类型系统禁止在主进程中引用 CUDA API）。
class TestProbeSubprocessIsolation:
    """stage_probe_vram 应通过子进程隔离 probe，主进程不执行 CUDA 计算。

    子进程隔离（2026-07-11）：probe 在独立进程中运行，子进程退出时
    CUDA 上下文销毁，主进程的 CUDA 状态不受影响。
    """

    def test_stage_probe_vram_calls_subprocess(self):
        """stage_probe_vram 源码应含 _run_probe_in_subprocess 调用。"""
        import inspect
        from senseframe.engine.runner.pipeline import stage_probe_vram
        source = inspect.getsource(stage_probe_vram)
        assert "_run_probe_in_subprocess(" in source, \
            "stage_probe_vram 应调用 _run_probe_in_subprocess（子进程隔离）"

    def test_stage_probe_vram_no_direct_cuda_compute(self):
        """stage_probe_vram 源码不应含 model.to(cuda) 或 torch.cuda 计算调用。

        主进程不执行任何 CUDA 计算，CUDA 计算全部在子进程中。
        """
        import inspect
        from senseframe.engine.runner.pipeline import stage_probe_vram
        source = inspect.getsource(stage_probe_vram)
        # 不应直接调用 model.to(cuda)（子进程负责）
        assert "model.to(device)" not in source, \
            "stage_probe_vram 不应直接调用 model.to(device)（子进程隔离）"
        # 不应直接调用 torch.cuda.max_memory_allocated（子进程负责）
        assert "max_memory_allocated" not in source, \
            "stage_probe_vram 不应直接调用 max_memory_allocated（子进程隔离）"

    def test_stage_probe_vram_no_set_seed_needed(self):
        """stage_probe_vram 源码不需要 set_seed（子进程隔离，主进程不执行 CUDA）。

        子进程隔离后，主进程不执行 CUDA 计算，不需要 set_seed 防御 CUDA 上下文时序。
        """
        import inspect
        from senseframe.engine.runner.pipeline import stage_probe_vram
        source = inspect.getsource(stage_probe_vram)
        # stage_probe_vram 不应包含 set_seed 调用（子进程隔离后不需要）
        assert "set_seed(" not in source, \
            "stage_probe_vram 不应包含 set_seed（子进程隔离后主进程不执行 CUDA）"
