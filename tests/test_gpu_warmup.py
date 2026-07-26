"""子进程隔离方案：stage_train 不依赖 GPU warmup 且不调用 set_seed 的契约测试。

子进程隔离方案（2026-07-11）最终决策：
- probe 在独立子进程中运行，子进程退出时 CUDA 上下文销毁
- 主进程 CUDA 状态完全干净，trainer.fit() 首步就是首次 CUDA 计算
- 不需要 GPU warmup 统一初始化时序（warmup 自身会改变 cuBLAS/cuDNN
  算法选择，导致 val_acc 从 0.982 降到 0.929）
- 不需要在 stage_train 入口 set_seed（子进程隔离后主进程不消耗 RNG，
  set_seed 反而重置 RNG 导致 DataLoader shuffle 顺序与无 probe 基线不同，
  实测 ep0 从 1.210943 变为 1.258861）

本文件验证：
1. _gpu_warmup 函数已从 pipeline 中移除
2. stage_train 不调用 _gpu_warmup
3. stage_train 不调用 set_seed（RNG 由 stage_preflight 的 set_seed 自然流转）
"""

import inspect

from senseframe.engine.runner.pipeline import stage_train


# ============================================================
# 1. _gpu_warmup 已移除
# ============================================================

# ARCHITECTURE_TRIPWIRE: _gpu_warmup 函数已从 pipeline 中移除
# 不可替代原因: "模块中不存在某函数"是否定属性，行为测试无法验证"某函数从未被定义过"；
#   必须通过反射/源码检查确认函数已删除。
# 删除条件: 当 pipeline 模块结构稳定且 CI linter 禁止引入 _gpu_warmup 命名时。
class TestGpuWarmupRemoved:
    """_gpu_warmup 函数应已从 pipeline 模块移除。"""

    def test_gpu_warmup_not_in_pipeline(self):
        """pipeline 模块不应再有 _gpu_warmup 函数。"""
        from senseframe.engine.runner import pipeline
        assert not hasattr(pipeline, "_gpu_warmup"), \
            "_gpu_warmup 应已移除（子进程隔离后不需要 warmup）"


# ============================================================
# 2. stage_train 不调用 warmup
# ============================================================

# ARCHITECTURE_TRIPWIRE: stage_train 不调用 _gpu_warmup（子进程隔离后不需要 warmup）
# 不可替代原因: "函数体不包含 _gpu_warmup 调用"是否定属性，行为测试无法区分
#   "未调用 warmup"和"调用了 warmup 但无副作用"；必须检查源码文本。
# 删除条件: 当 _gpu_warmup 函数从代码库中彻底删除且 CI 禁止重新引入时。
class TestStageTrainNoWarmup:
    """stage_train 不应调用 _gpu_warmup 或包含 warmup 相关逻辑。"""

    def test_stage_train_no_gpu_warmup_call(self):
        """stage_train 源码不应含 _gpu_warmup 调用。"""
        source = inspect.getsource(stage_train)
        assert "_gpu_warmup(" not in source, \
            "stage_train 不应调用 _gpu_warmup（子进程隔离后不需要）"

    def test_stage_train_no_warmup_log(self):
        """stage_train 源码不应含 warmup 完成日志。"""
        source = inspect.getsource(stage_train)
        assert "GPU warmup 完成" not in source, \
            "stage_train 不应含 warmup 日志（warmup 已移除）"


# ============================================================
# 3. stage_train 不调用 set_seed（RNG 自然流转）
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

    def test_stage_train_no_set_seed_call(self):
        """stage_train 源码不应含 set_seed 调用。"""
        source = inspect.getsource(stage_train)
        assert "set_seed(" not in source, \
            "stage_train 不应调用 set_seed（子进程隔离后不需要，反而破坏 RNG 自然流转）"
