"""P3 阶段 9：CSIFoundationModel + MAE 自监督预训练单元测试。

测试原则：
- 全部用合成数据，快速运行（< 30 秒）
- 真实 forward pass / 真实 MAE 流程
- Protocol 契约：isinstance + runtime_checkable
- MAE 原理验证：mask ratio / visible 数量 / ids_restore 还原顺序 / loss 仅 masked
"""
from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from senseframe.automl.peft_builder import PEFTModel
from senseframe.core.foundation_model import (
    PEFTConfig,
    PretrainConfig,
    SensingFoundationModel,
)
from senseframe.scenes.wifi_csi.foundation_model import (
    CSIFoundationModel,
    CSIPatchEmbedder,
)


# ============================================================
# 辅助
# ============================================================
def _make_small_model(
    input_shape: Tuple[int, int] = (3, 64),
    **kwargs,
) -> CSIFoundationModel:
    """构造小模型用于快速测试。"""
    defaults = dict(
        d_model=32,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        patch_len=8,
        decoder_dim=16,
    )
    defaults.update(kwargs)
    return CSIFoundationModel(input_shape=input_shape, **defaults)


# ============================================================
# TestCSIFoundationModelConstruction
# ============================================================
class TestCSIFoundationModelConstruction:
    """验证构造与 Protocol 实现。"""

    def test_construct_default(self):
        model = CSIFoundationModel(input_shape=(3, 64))
        assert isinstance(model, nn.Module)
        # 默认 patch_len=16，64/16=4
        assert model.n_patches == 4

    def test_model_id_property(self):
        model = _make_small_model()
        assert model.model_id == "csi-mae-base"

    def test_modality_property(self):
        model = _make_small_model()
        assert model.modality == "csi"

    def test_satisfies_protocol(self):
        model = _make_small_model()
        assert isinstance(model, SensingFoundationModel), (
            "CSIFoundationModel 应满足 SensingFoundationModel Protocol"
        )

    def test_construct_with_custom_input_shape(self):
        """不同 (C, L) 形状都能构造。"""
        shapes_and_patch_lens = [
            ((1, 64), 8),
            ((3, 80), 16),
            ((22, 40), 8),
            ((6, 32), 8),
        ]
        for shape, pl in shapes_and_patch_lens:
            model = CSIFoundationModel(input_shape=shape, patch_len=pl)
            assert model.patch_embedder.n_patches == shape[1] // pl
            assert model.patch_embedder.C == shape[0]
            assert model.patch_embedder.L == shape[1]

    def test_construct_invalid_d_model_n_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            CSIFoundationModel(input_shape=(3, 64), d_model=30, n_heads=4)

    def test_construct_invalid_patch_len_raises(self):
        with pytest.raises(ValueError, match="divisible by patch_len"):
            CSIFoundationModel(input_shape=(3, 64), patch_len=7)


# ============================================================
# TestCSIPatchEmbedder
# ============================================================
class TestCSIPatchEmbedder:
    """验证 patch 切分与投影。"""

    def test_patch_embed_output_shape(self):
        embedder = CSIPatchEmbedder(
            input_shape=(3, 64), patch_len=8, d_model=32
        )
        x = torch.randn(4, 3, 64)
        out = embedder(x)
        assert out.shape == (4, 8, 32), f"got {out.shape}"

    def test_patch_count_correct(self):
        embedder = CSIPatchEmbedder(
            input_shape=(3, 64), patch_len=8, d_model=32
        )
        assert embedder.n_patches == 8
        embedder2 = CSIPatchEmbedder(
            input_shape=(1, 250), patch_len=10, d_model=16
        )
        assert embedder2.n_patches == 25

    def test_to_patches_shape(self):
        """to_patches 返回未投影的原始 patch 展平值（重建 target 用）。"""
        embedder = CSIPatchEmbedder(
            input_shape=(3, 64), patch_len=8, d_model=32
        )
        x = torch.randn(4, 3, 64)
        patches = embedder.to_patches(x)
        # patch_len * C = 8 * 3 = 24
        assert patches.shape == (4, 8, 24), f"got {patches.shape}"

    def test_to_patches_preserves_values(self):
        """to_patches 展平后的值与原始 x 一致（仅 reshape，不改值）。"""
        embedder = CSIPatchEmbedder(
            input_shape=(2, 16), patch_len=8, d_model=4
        )
        x = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
             [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
              90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]],
        ])
        patches = embedder.to_patches(x)
        # 第 0 个 patch = x[0, :8, :2] 展平
        expected_patch_0 = torch.tensor([
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0,
            5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0, 80.0,
        ])
        assert torch.allclose(patches[0, 0], expected_patch_0), (
            f"patch values mismatch: {patches[0, 0]} vs {expected_patch_0}"
        )


# ============================================================
# TestRandomMasking
# ============================================================
class TestRandomMasking:
    """验证 MAE random_masking 实现。"""

    def test_mask_ratio_correct(self):
        """mask 数量 ≈ n_patches * mask_ratio（int 截断）。"""
        model = _make_small_model()  # n_patches=8
        x = torch.randn(4, 8, 32)
        _, mask, _ = model.random_masking(x, mask_ratio=0.5)
        # int(8 * 0.5) = 4 masked
        mask_count = mask.sum(dim=1)
        for c in mask_count:
            assert c.item() == 4, f"expected 4 masked, got {c.item()}"

    def test_mask_ratio_0_75(self):
        model = _make_small_model()  # n_patches=8
        x = torch.randn(2, 8, 32)
        _, mask, _ = model.random_masking(x, mask_ratio=0.75)
        # int(8 * 0.75) = 6 masked, 2 visible
        assert mask.sum(dim=1).max().item() == 6

    def test_visible_patches_count(self):
        model = _make_small_model()
        x = torch.randn(4, 8, 32)
        x_visible, _, _ = model.random_masking(x, mask_ratio=0.5)
        assert x_visible.shape == (4, 4, 32), f"got {x_visible.shape}"

    def test_mask_is_boolean_or_01(self):
        """mask 值只能是 0.0 或 1.0。"""
        model = _make_small_model()
        x = torch.randn(2, 8, 32)
        _, mask, _ = model.random_masking(x, mask_ratio=0.5)
        unique = set(mask.unique().tolist())
        assert unique.issubset({0.0, 1.0}), f"mask values not in {{0, 1}}: {unique}"

    def test_ids_restore_restores_order(self):
        """按 ids_restore 还原后，visible patches 回到原位置。"""
        model = _make_small_model()
        torch.manual_seed(123)
        x = torch.randn(2, 8, 32)
        x_visible, mask, ids_restore = model.random_masking(x, mask_ratio=0.5)

        B, L, D = x_visible.shape
        N = ids_restore.shape[1]
        # 拼接 visible + 占位（masked 位置用 0）
        full_shuffled = torch.cat(
            [x_visible, torch.zeros(B, N - L, D)], dim=1
        )
        # 按 ids_restore 还原顺序
        restored = torch.gather(
            full_shuffled,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D),
        )
        # 在 visible 位置（mask==0），restored 应等于原始 x
        visible_mask = (mask == 0)
        for b in range(B):
            vis_pos = visible_mask[b]
            assert torch.allclose(
                restored[b][vis_pos], x[b][vis_pos], atol=1e-6
            ), "ids_restore 未能将 visible patches 还原到原位置"

    def test_mask_ratio_zero_all_visible(self):
        """mask_ratio=0 时全部 visible，mask 全为 0。"""
        model = _make_small_model()
        x = torch.randn(2, 8, 32)
        x_visible, mask, _ = model.random_masking(x, mask_ratio=0.0)
        assert x_visible.shape == (2, 8, 32)
        assert mask.sum().item() == 0


# ============================================================
# TestEncode
# ============================================================
class TestEncode:
    """验证特征提取。"""

    def test_encode_output_shape(self):
        model = _make_small_model()
        x = torch.randn(4, 3, 64)
        out = model.encode(x)
        # encode 返回 (B, n_patches, d_model)
        assert out.shape == (4, 8, 32), f"got {out.shape}"

    def test_encode_different_batch_size(self):
        model = _make_small_model()
        for B in [1, 2, 8]:
            x = torch.randn(B, 3, 64)
            out = model.encode(x)
            assert out.shape == (B, 8, 32)

    def test_encode_no_gradient(self):
        """eval + no_grad 下输出不携带梯度。"""
        model = _make_small_model()
        model.eval()
        x = torch.randn(2, 3, 64)
        with torch.no_grad():
            out = model.encode(x)
        assert not out.requires_grad

    def test_forward_equals_encode(self):
        """forward 调用 encode（供 PEFTBuilder 注入时使用）。"""
        model = _make_small_model()
        model.eval()
        x = torch.randn(2, 3, 64)
        with torch.no_grad():
            out_forward = model(x)
            out_encode = model.encode(x)
        assert torch.allclose(out_forward, out_encode)


# ============================================================
# TestPretrainMAE
# ============================================================
class TestPretrainMAE:
    """验证 MAE 自监督预训练流程。"""

    def test_pretrain_runs_one_epoch(self):
        """1 epoch pretrain 跑通，loss 是有限值。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(8, 3, 64)
        config = PretrainConfig(epochs=1, mask_ratio=0.5, learning_rate=1e-3)
        model.pretrain([(x,)], config)
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            loss = model._mae_forward_loss(x, mask_ratio=0.5)
        assert torch.isfinite(loss), f"loss not finite: {loss}"

    def test_pretrain_loss_decreases(self):
        """结构化数据上 pretrain 后 loss 应下降（验证训练有效）。"""
        torch.manual_seed(0)
        model = _make_small_model()
        # 结构化数据：常量 + 小噪声，模型可学会预测均值
        x = torch.ones(16, 3, 64) + 0.1 * torch.randn(16, 3, 64)

        def avg_loss(model, n_trials=5):
            model.eval()
            losses = []
            with torch.no_grad():
                torch.manual_seed(42)
                for _ in range(n_trials):
                    losses.append(
                        model._mae_forward_loss(x, mask_ratio=0.5).item()
                    )
            return sum(losses) / len(losses)

        loss_before = avg_loss(model)
        config = PretrainConfig(epochs=5, mask_ratio=0.5, learning_rate=1e-3)
        model.pretrain([(x,)], config)
        loss_after = avg_loss(model)
        assert loss_after < loss_before, (
            f"loss 应下降: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_pretrain_mask_ratio_0_75(self):
        """mask_ratio=0.75 也能跑通。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(8, 3, 64)
        config = PretrainConfig(epochs=1, mask_ratio=0.75, learning_rate=1e-3)
        model.pretrain([(x,)], config)
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            loss = model._mae_forward_loss(x, mask_ratio=0.75)
        assert torch.isfinite(loss), f"loss not finite: {loss}"

    def test_pretrain_with_dataloader(self):
        """unlabeled_data 是 DataLoader 时也能跑通（每 batch 是 (x, y) tuple）。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(16, 3, 64)
        y = torch.zeros(16, dtype=torch.long)  # 自监督忽略 y
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        config = PretrainConfig(epochs=1, mask_ratio=0.5, learning_rate=1e-3)
        model.pretrain(loader, config)
        # 跑通即可
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            loss = model._mae_forward_loss(x, mask_ratio=0.5)
        assert torch.isfinite(loss)

    def test_pretrain_with_tensor_only_batches(self):
        """batch 是裸 tensor（非 tuple）时也能正确处理。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(8, 3, 64)
        config = PretrainConfig(epochs=1, mask_ratio=0.5, learning_rate=1e-3)
        model.pretrain([x], config)  # list of tensor
        model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            loss = model._mae_forward_loss(x, mask_ratio=0.5)
        assert torch.isfinite(loss)


# ============================================================
# TestGetPEFTModule
# ============================================================
class TestGetPEFTModule:
    """验证 PEFT 模块构建。"""

    def test_get_peft_module_lora(self):
        """LoRA 返回 PEFTModel，lora_modules 非空。"""
        model = _make_small_model()
        peft_config = PEFTConfig(
            peft_method="lora",
            peft_rank=8,
            peft_alpha=1,
            peft_target_modules="query_value",
        )
        peft_model = model.get_peft_module(peft_config)
        assert isinstance(peft_model, PEFTModel)
        assert peft_model.peft_method == "lora"
        assert len(peft_model.lora_modules) > 0, (
            "lora_modules 不应为空（encoder/decoder 含 query/value Linear）"
        )

    def test_get_peft_module_full(self):
        """full 方法返回 PEFTModel，无 PEFT 模块注入，backbone 全可训练。"""
        model = _make_small_model()
        peft_config = PEFTConfig(peft_method="full")
        peft_model = model.get_peft_module(peft_config)
        assert isinstance(peft_model, PEFTModel)
        assert peft_model.peft_method == "full"
        assert len(peft_model.lora_modules) == 0
        for p in peft_model.backbone.parameters():
            assert p.requires_grad, "full 方法 backbone 参数应全可训练"

    def test_get_peft_module_does_not_modify_original(self):
        """get_peft_module 深拷贝，原模型权重未被冻结。"""
        model = _make_small_model()
        peft_config = PEFTConfig(
            peft_method="lora",
            peft_rank=8,
            freeze_backbone=True,
        )
        _ = model.get_peft_module(peft_config)
        # 原模型所有参数仍可训练
        for p in model.parameters():
            assert p.requires_grad, (
                "原模型参数不应被 PEFT 构建冻结（深拷贝保护）"
            )

    def test_get_peft_module_lora_forward_works(self):
        """LoRA 注入后 forward 仍能跑通，输出 shape 一致。"""
        model = _make_small_model()
        peft_config = PEFTConfig(peft_method="lora", peft_rank=8, peft_alpha=1)
        peft_model = model.get_peft_module(peft_config)
        x = torch.randn(2, 3, 64)
        out = peft_model(x)
        assert out.shape == (2, 8, 32), f"got {out.shape}"


# ============================================================
# TestCrossDatasetTransfer
# ============================================================
class TestCrossDatasetTransfer:
    """轻量验证跨数据集 / 端到端流程。"""

    def test_encode_works_for_different_input_shape(self):
        """不同 (C, L) 输入都能 encode（shape-agnostic）。"""
        cases = [
            ((1, 64), 8),    # 类 UT_HAR 风格
            ((3, 80), 16),   # 类 NTU-Fi 风格
            ((6, 32), 8),
        ]
        for shape, pl in cases:
            model = CSIFoundationModel(
                input_shape=shape,
                d_model=32,
                n_heads=4,
                n_encoder_layers=2,
                n_decoder_layers=1,
                patch_len=pl,
                decoder_dim=16,
            )
            x = torch.randn(2, *shape)
            out = model.encode(x)
            assert out.shape == (2, shape[1] // pl, 32), (
                f"shape={shape}, pl={pl}: got {out.shape}"
            )

    def test_pretrain_then_encode(self):
        """先 pretrain 1 epoch 再 encode，验证流程闭环。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(8, 3, 64)
        config = PretrainConfig(epochs=1, mask_ratio=0.5, learning_rate=1e-3)
        model.pretrain([(x,)], config)
        # pretrain 后 encode 可用
        model.eval()
        with torch.no_grad():
            features = model.encode(x)
        assert features.shape == (8, 8, 32), f"got {features.shape}"
        assert torch.isfinite(features).all(), "features 应为有限值"

    def test_pretrain_then_peft_module(self):
        """先 pretrain 再 get_peft_module，验证完整迁移流程。"""
        torch.manual_seed(0)
        model = _make_small_model()
        x = torch.randn(8, 3, 64)
        # 1. pretrain
        pretrain_config = PretrainConfig(
            epochs=1, mask_ratio=0.5, learning_rate=1e-3
        )
        model.pretrain([(x,)], config=pretrain_config)
        # 2. PEFT 构建（深拷贝保护预训练权重）
        peft_config = PEFTConfig(peft_method="lora", peft_rank=8)
        peft_model = model.get_peft_module(peft_config)
        # 3. PEFT 模型 forward 可用
        peft_model.eval()
        with torch.no_grad():
            out = peft_model(x)
        assert out.shape == (8, 8, 32)
        # 4. 原模型仍可 encode（深拷贝未污染）
        model.eval()
        with torch.no_grad():
            base_out = model.encode(x)
        assert base_out.shape == (8, 8, 32)
