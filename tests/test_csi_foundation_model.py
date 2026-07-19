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


# ============================================================
# TestReplacePatchEmbedder
# ============================================================
class TestReplacePatchEmbedder:
    """验证 replace_patch_embedder 跨模态迁移行为（B5/B6 跨场景迁移核心 API）。

    测试原则（对齐 reference/p3_lessons_learned.md 教训 10）：
    - modality-specific 模块（patch_embedder/pos_embed/decoder_pos_embed/decoder_proj）必须重新初始化
    - modality-agnostic 模块（encoder/encoder_norm/decoder_embed/decoder/decoder_norm/mask_token）必须保留
    - 替换后 backbone 能 forward 新模态 shape 输入
    - 替换后 n_patches 与新模态一致
    """
    # CSI 模态测试配置：input_shape=(3, 64), patch_len=8 → patch_len*C=24, n_patches=8
    # EEG 模态测试配置：input_shape=(4, 48), patch_len=8 → patch_len*C=32, n_patches=6
    # 两个模态 patch_len*C 与 n_patches 都不同，能严格验证替换生效
    CSI_SHAPE = (3, 64)
    CSI_PATCH_LEN = 8
    EEG_SHAPE = (4, 48)
    EEG_PATCH_LEN = 8

    def _make_csi_model(self) -> CSIFoundationModel:
        """构建 CSI 维度的小模型（模拟 CSI MAE 预训练后的 backbone）。"""
        return _make_small_model(
            input_shape=self.CSI_SHAPE,
            patch_len=self.CSI_PATCH_LEN,
        )

    def test_replace_changes_patch_embedder_proj_dim(self):
        """替换后 patch_embedder.proj 输入维度 = new_patch_len * new_C。"""
        model = self._make_csi_model()
        old_proj_weight = model.patch_embedder.proj.weight
        # CSI: proj 输入维度 = 8 * 3 = 24
        assert old_proj_weight.shape[1] == self.CSI_PATCH_LEN * self.CSI_SHAPE[0]

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        new_proj_weight = model.patch_embedder.proj.weight
        # EEG: proj 输入维度 = 8 * 4 = 32
        assert new_proj_weight.shape[1] == self.EEG_PATCH_LEN * self.EEG_SHAPE[0], (
            f"替换后 proj 输入维度应为 {self.EEG_PATCH_LEN * self.EEG_SHAPE[0]}, "
            f"实际 {new_proj_weight.shape[1]}"
        )

    def test_replace_changes_n_patches(self):
        """替换后 n_patches = new_L / new_patch_len。"""
        model = self._make_csi_model()
        assert model.n_patches == self.CSI_SHAPE[1] // self.CSI_PATCH_LEN  # 64/8=8

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        assert model.n_patches == self.EEG_SHAPE[1] // self.EEG_PATCH_LEN, (
            f"替换后 n_patches 应为 {self.EEG_SHAPE[1] // self.EEG_PATCH_LEN}, "
            f"实际 {model.n_patches}"
        )

    def test_replace_reinitializes_pos_embed_shape(self):
        """替换后 pos_embed shape 变为新 n_patches。"""
        model = self._make_csi_model()
        old_pos_embed = model.pos_embed
        assert old_pos_embed.shape == (1, 8, 32)  # CSI: n_patches=8, d_model=32

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        new_pos_embed = model.pos_embed
        # EEG: n_patches=6, d_model=32
        assert new_pos_embed.shape == (1, 6, 32), (
            f"替换后 pos_embed shape 应为 (1, 6, 32), 实际 {new_pos_embed.shape}"
        )

    def test_replace_reinitializes_decoder_pos_embed_shape(self):
        """替换后 decoder_pos_embed shape 变为新 n_patches。"""
        model = self._make_csi_model()
        assert model.decoder_pos_embed.shape == (1, 8, 16)  # CSI: n_patches=8, decoder_dim=16

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        assert model.decoder_pos_embed.shape == (1, 6, 16), (
            f"替换后 decoder_pos_embed shape 应为 (1, 6, 16), "
            f"实际 {model.decoder_pos_embed.shape}"
        )

    def test_replace_reinitializes_decoder_proj_dim(self):
        """替换后 decoder_proj 输出维度 = new_patch_len * new_C。"""
        model = self._make_csi_model()
        # CSI: decoder_proj 输出维度 = 8 * 3 = 24
        assert model.decoder_proj.weight.shape[0] == self.CSI_PATCH_LEN * self.CSI_SHAPE[0]

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        # EEG: decoder_proj 输出维度 = 8 * 4 = 32
        assert model.decoder_proj.weight.shape[0] == self.EEG_PATCH_LEN * self.EEG_SHAPE[0], (
            f"替换后 decoder_proj 输出维度应为 {self.EEG_PATCH_LEN * self.EEG_SHAPE[0]}, "
            f"实际 {model.decoder_proj.weight.shape[0]}"
        )

    def test_replace_preserves_encoder_weights(self):
        """替换后 encoder 权重保持不变（modality-agnostic 核心验证）。"""
        model = self._make_csi_model()
        # 记录替换前 encoder 第一层权重
        old_encoder_weight = model.encoder[0].attn.query.weight.clone()
        old_encoder_norm_weight = model.encoder_norm.weight.clone()

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)

        new_encoder_weight = model.encoder[0].attn.query.weight
        new_encoder_norm_weight = model.encoder_norm.weight
        assert torch.equal(old_encoder_weight, new_encoder_weight), (
            "替换后 encoder[0].attn.query.weight 应保持不变（modality-agnostic）"
        )
        assert torch.equal(old_encoder_norm_weight, new_encoder_norm_weight), (
            "替换后 encoder_norm.weight 应保持不变（modality-agnostic）"
        )

    def test_replace_preserves_decoder_main_body_weights(self):
        """替换后 decoder 主体（decoder_embed/decoder/decoder_norm）权重保持。"""
        model = self._make_csi_model()
        old_decoder_embed_weight = model.decoder_embed.weight.clone()
        old_decoder_weight = model.decoder[0].attn.query.weight.clone()
        old_decoder_norm_weight = model.decoder_norm.weight.clone()
        old_mask_token = model.mask_token.clone()

        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)

        assert torch.equal(old_decoder_embed_weight, model.decoder_embed.weight), (
            "替换后 decoder_embed.weight 应保持不变"
        )
        assert torch.equal(old_decoder_weight, model.decoder[0].attn.query.weight), (
            "替换后 decoder[0].attn.query.weight 应保持不变"
        )
        assert torch.equal(old_decoder_norm_weight, model.decoder_norm.weight), (
            "替换后 decoder_norm.weight 应保持不变"
        )
        assert torch.equal(old_mask_token, model.mask_token), (
            "替换后 mask_token 应保持不变"
        )

    def test_replace_then_forward_new_modality(self):
        """替换后能 forward 新模态 shape 输入，输出 shape 正确。"""
        model = self._make_csi_model()
        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        model.eval()

        # EEG 输入: (B=2, C=4, L=48)
        x_eeg = torch.randn(2, *self.EEG_SHAPE)
        with torch.no_grad():
            out = model.encode(x_eeg)
        # 输出: (B=2, n_patches=6, d_model=32)
        assert out.shape == (2, 6, 32), (
            f"替换后 encode 输出 shape 应为 (2, 6, 32), 实际 {out.shape}"
        )
        assert torch.isfinite(out).all(), "encode 输出应为有限值"

    def test_replace_then_mae_forward_loss_new_modality(self):
        """替换后 MAE loss 在新模态输入上可计算（验证 decoder 链路完整）。"""
        model = self._make_csi_model()
        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)
        model.eval()

        x_eeg = torch.randn(4, *self.EEG_SHAPE)
        with torch.no_grad():
            torch.manual_seed(42)
            loss = model._mae_forward_loss(x_eeg, mask_ratio=0.5)
        assert torch.isfinite(loss), f"MAE loss 应为有限值, 实际 {loss}"
        assert loss.item() > 0, "MAE loss 应为正"

    def test_replace_invalid_shape_raises(self):
        """无效 input_shape（L 不能被 patch_len 整除）抛 ValueError。"""
        model = self._make_csi_model()
        # L=50, patch_len=8 → 50 % 8 != 0
        with pytest.raises(ValueError, match="divisible by patch_len"):
            model.replace_patch_embedder((4, 50), new_patch_len=8)

    def test_replace_invalid_shape_dim_raises(self):
        """无效 input_shape（不是 (C, L) 元组）抛 ValueError。"""
        model = self._make_csi_model()
        with pytest.raises(ValueError, match="input_shape must be"):
            model.replace_patch_embedder((4, 48, 3), new_patch_len=8)  # 3D

    def test_cross_modal_pretrain_then_replace_then_encode(self):
        """端到端跨模态迁移流程：CSI pretrain → replace → EEG forward。

        验证 B5/B6 跨场景迁移的完整链路：
        1. 用 CSI 维度构建 backbone + MAE 预训练（encoder 权重应变化）
        2. replace_patch_embedder 切换到 EEG 维度
        3. EEG 维度输入能 forward，输出 shape 正确
        4. encoder 权重在 replace 前后保持不变（modality-agnostic 保留）
        """
        torch.manual_seed(0)
        model = self._make_csi_model()

        # 1. 记录 pretrain 前 encoder 权重
        encoder_weight_before_pretrain = (
            model.encoder[0].attn.query.weight.clone()
        )

        # 2. CSI 维度 MAE 预训练
        x_csi = torch.randn(8, *self.CSI_SHAPE)
        pretrain_config = PretrainConfig(
            epochs=2, mask_ratio=0.5, learning_rate=1e-3
        )
        model.pretrain([(x_csi,)], config=pretrain_config)

        # 3. 验证 pretrain 确实改变了 encoder 权重
        encoder_weight_after_pretrain = model.encoder[0].attn.query.weight
        assert not torch.equal(encoder_weight_before_pretrain, encoder_weight_after_pretrain), (
            "pretrain 后 encoder 权重应发生变化（验证预训练真的在训练）"
        )

        # 4. 记录 replace 前 encoder 权重
        encoder_weight_before_replace = encoder_weight_after_pretrain.clone()

        # 5. replace_patch_embedder 切换到 EEG 维度
        model.replace_patch_embedder(self.EEG_SHAPE, self.EEG_PATCH_LEN)

        # 6. 验证 replace 后 encoder 权重保持不变
        encoder_weight_after_replace = model.encoder[0].attn.query.weight
        assert torch.equal(encoder_weight_before_replace, encoder_weight_after_replace), (
            "replace_patch_embedder 后 encoder 权重应保持不变（核心：modality-agnostic 保留）"
        )

        # 7. EEG 维度输入能 forward
        model.eval()
        x_eeg = torch.randn(4, *self.EEG_SHAPE)
        with torch.no_grad():
            out = model.encode(x_eeg)
        assert out.shape == (4, 6, 32), (
            f"跨模态迁移后 EEG 输入 encode 输出 shape 应为 (4, 6, 32), 实际 {out.shape}"
        )
        assert torch.isfinite(out).all()
