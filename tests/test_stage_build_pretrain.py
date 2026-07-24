"""stage_build pretrain_checkpoint 消费测试。

验证 ctx.pretrain_checkpoint 非空时 stage_build 加载权重到 ctx.model（HIGH 1 修复）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import pytest


class TestLoadCheckpointBackboneFormat:
    """验证 load_checkpoint_flexible 支持 backbone_state_dict 格式。"""

    def test_backbone_state_dict_format_loaded(self, tmp_path):
        """自定义 MAE checkpoint（含 backbone_state_dict key）应被识别并加载。"""
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model = nn.Linear(4, 2)
        backbone_state = model.state_dict()

        ckpt_path = tmp_path / "pretrain.pt"
        torch.save(
            {"backbone_state_dict": backbone_state, "best_psnr": 19.4, "final_epoch": 50},
            ckpt_path,
        )

        target_model = nn.Linear(4, 2)
        result = load_checkpoint_flexible(ckpt_path, target_model, strict=True)

        assert result["source_format"] == "backbone_state_dict"
        assert result["num_keys_loaded"] == len(backbone_state)
        # 权重确实加载
        for k in backbone_state:
            assert torch.equal(target_model.state_dict()[k], backbone_state[k])

    def test_backbone_state_dict_strict_false_for_cross_modal(self, tmp_path):
        """跨模态迁移（patch_embedder 已替换）应支持 strict=False。"""
        from senseframe.common.checkpoint import load_checkpoint_flexible

        # 源模型有 5 个 key，目标模型只有 3 个（模拟 patch_embedder 替换）
        class SourceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embedder = nn.Linear(4, 8)
                self.encoder = nn.Linear(8, 8)

        class TargetModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 输入/输出维度都不同，确保 patch_embedder 全部 key 被 shape 过滤跳过
                # （仅改输入维度会让 bias shape 仍匹配从而被加载）
                self.patch_embedder = nn.Linear(2, 16)
                self.encoder = nn.Linear(8, 8)

        source = SourceModel()
        target = TargetModel()
        # 保存 patch_embedder 初始权重，用于断言 strict=False 时被跳过
        patch_embedder_init = {
            k: v.clone() for k, v in target.patch_embedder.state_dict().items()
        }
        ckpt_path = tmp_path / "cross_modal.pt"
        torch.save({"backbone_state_dict": source.state_dict()}, ckpt_path)

        result = load_checkpoint_flexible(ckpt_path, target, strict=False)
        assert result["source_format"] == "backbone_state_dict"
        # encoder 权重应匹配，patch_embedder 因 shape 不同被跳过
        assert torch.equal(target.encoder.state_dict()["weight"], source.encoder.state_dict()["weight"])
        # patch_embedder 应因 shape 不匹配被跳过，保持初始随机值
        for k, v in patch_embedder_init.items():
            assert torch.equal(target.patch_embedder.state_dict()[k], v), \
                f"patch_embedder.{k} 应因 shape 不匹配被跳过，但权重被修改"


def _make_build_ctx(tmp_path, pretrain_checkpoint=None):
    """构造 mock PipelineContext 供 stage_build 测试用。

    封装 TestStageBuildPretrainConsumption 两个测试共用的 ctx mock 构造代码。
    目标模型（nn.Linear(10, 7)）可通过 ``ctx.scene.build_model_for_dataset.return_value``
    取回用于断言。

    Args:
        tmp_path: pytest tmp_path fixture，用于 output_dir / data_root
        pretrain_checkpoint: ctx.pretrain_checkpoint 值，默认 None（向后兼容）

    Returns:
        构造好的 MagicMock ctx，target_model 已挂到 scene.build_model_for_dataset。
    """
    target_model = nn.Linear(10, 7)

    ctx = MagicMock()
    ctx.learning_mode = "supervised"
    ctx.config.scene.data_root = str(tmp_path)
    ctx.config.scene.task_spec = None
    ctx.config.trainer.early_stopping = None
    ctx.config.trainer.early_stopping_monitor = "val_loss"
    ctx.config.extra_callbacks = None
    ctx.config.module_factory = None
    ctx.config.datamodule_factory = None
    ctx.model_id = "MLP"
    ctx.dataset = "UT_HAR_data"
    ctx.num_classes = 7
    ctx.feature_spec = MagicMock(feature_dim=10)
    ctx.scene_info = {"input_shape": [1, 250, 90], "n_features": 10}
    ctx.route_config = {"max_epochs": 50}
    ctx.resolved = {"learning_rate": 0.001, "weight_decay": 0.0, "optimizer": "adam",
                    "scheduler": None, "batch_size": 32, "num_workers": 0,
                    "metrics": ["accuracy", "macro_f1"], "logger": "csv",
                    "pin_memory": False, "persistent_workers": False}
    ctx.bundle = MagicMock(train=MagicMock(), val=MagicMock(), test=MagicMock())
    ctx.scene = MagicMock()
    ctx.scene.get_transforms.return_value = MagicMock(
        train_transform=None, eval_transform=None)
    ctx.scene.build_model_for_dataset.return_value = target_model
    ctx.scene_kwargs = {}
    ctx.data_profile = None
    ctx.output_dir = tmp_path
    ctx.log_writer = MagicMock()
    ctx.extra = None
    ctx.pruner = None
    ctx.trial_id = None
    ctx.pretrain_checkpoint = pretrain_checkpoint
    ctx.config.trainer.epochs = 50
    ctx.intermediate_values = {}
    return ctx


class TestStageBuildPretrainConsumption:
    """验证 stage_build 消费 ctx.pretrain_checkpoint。"""

    def test_pretrain_checkpoint_loaded_into_model(self, tmp_path):
        """ctx.pretrain_checkpoint 非空时，stage_build 加载权重到 ctx.model。"""
        from senseframe.engine.runner.pipeline.stages import build as build_module

        # 构造 pretrain checkpoint（backbone_state_dict 格式），权重已知
        source_model = nn.Linear(10, 7)
        model_state = source_model.state_dict()
        ckpt_path = tmp_path / "pretrain.pt"
        torch.save({"backbone_state_dict": model_state, "best_psnr": 19.4}, ckpt_path)

        ctx = _make_build_ctx(tmp_path, pretrain_checkpoint=str(ckpt_path))
        target_model = ctx.scene.build_model_for_dataset.return_value

        # GenericDataModule / GenericLightningModule 是函数级导入，
        # 必须在源模块上 patch；build_logger 是模块级导入，patch.object 即可。
        with patch("senseframe.engine.datamodule.GenericDataModule", MagicMock()), \
             patch("senseframe.engine.module.GenericLightningModule", MagicMock()), \
             patch.object(build_module, "build_logger", MagicMock()):
            build_module.stage_build(ctx)

        # 验证权重已加载（model 权重与 checkpoint 一致）
        for k in model_state:
            assert torch.equal(target_model.state_dict()[k], model_state[k]), \
                f"权重 key '{k}' 未从 pretrain_checkpoint 加载"

    def test_no_pretrain_checkpoint_skips_loading(self, tmp_path):
        """ctx.pretrain_checkpoint=None 时，不加载 checkpoint（向后兼容）。"""
        from senseframe.engine.runner.pipeline.stages import build as build_module

        ctx = _make_build_ctx(tmp_path)  # pretrain_checkpoint 默认 None
        original_model = ctx.scene.build_model_for_dataset.return_value
        original_state = {k: v.clone() for k, v in original_model.state_dict().items()}

        with patch("senseframe.engine.datamodule.GenericDataModule", MagicMock()), \
             patch("senseframe.engine.module.GenericLightningModule", MagicMock()), \
             patch.object(build_module, "build_logger", MagicMock()):
            build_module.stage_build(ctx)

        # 权重未被修改（保持随机初始化）
        for k, v in original_state.items():
            assert torch.equal(original_model.state_dict()[k], v), \
                f"权重 key '{k}' 不应被修改（无 pretrain_checkpoint）"

    def test_pretrain_checkpoint_not_found_falls_back_to_scratch(self, tmp_path):
        """checkpoint 文件不存在时，stage_build 不中断，从零训练。"""
        from senseframe.engine.runner.pipeline.stages import build as build_module

        ctx = _make_build_ctx(tmp_path, pretrain_checkpoint=str(tmp_path / "nonexistent.pt"))
        original_model = ctx.scene.build_model_for_dataset.return_value
        original_state = {k: v.clone() for k, v in original_model.state_dict().items()}

        with patch("senseframe.engine.datamodule.GenericDataModule", MagicMock()), \
             patch("senseframe.engine.module.GenericLightningModule", MagicMock()), \
             patch.object(build_module, "build_logger", MagicMock()):
            build_module.stage_build(ctx)  # 应不抛异常

        # 权重未被修改（从零训练）
        for k, v in original_state.items():
            assert torch.equal(original_model.state_dict()[k], v), \
                f"权重 key '{k}' 不应被修改（checkpoint 不存在，从零训练）"
