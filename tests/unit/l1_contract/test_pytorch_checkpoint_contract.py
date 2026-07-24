"""L1 契约测试：PyTorch checkpoint API 契约。

锚点来源：PyTorch 官方 API（torch.save / torch.load / state_dict / load_state_dict）。
- PyTorch torch.save/torch.load: https://pytorch.org/docs/stable/generated/torch.save.html
  - torch.load 支持 weights_only 参数（PyTorch 2.0+ 安全加载）
  - torch.load 不限制文件扩展名（.pt / .ckpt / .pth 均可）
- PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
  - state_dict(): 返回包含模型参数的 dict
  - load_state_dict(): 加载参数到模型
- Lightning checkpoint 格式:
  https://lightning.ai/docs/pytorch/stable/common/checkpointing.html
  - 顶层含 state_dict / epoch / global_step / pytorch-lightning_version 等字段
  - state_dict 内 key 可能带 "model." 前缀（GenericLightningModule 结构）

验证目标：
- load_checkpoint_flexible 支持 .pt / .ckpt 扩展名
- state_dict() / load_state_dict() 往返一致
- torch.load 支持 weights_only 参数
- Lightning checkpoint 含 state_dict 字段（Lightning 官方格式）
"""
from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn


@pytest.mark.l1_contract
class TestPytorchCheckpointContract:
    """验证 checkpoint 加载符合 PyTorch / Lightning 官方 API 契约。"""

    # ============================================================
    # PyTorch nn.Module state_dict / load_state_dict 契约
    # ============================================================

    def test_state_dict_load_state_dict_roundtrip(self):
        """L1 anchor: nn.Module.state_dict() / load_state_dict() 往返一致。

        锚点：PyTorch 官方 API
        (https://pytorch.org/docs/stable/generated/torch.nn.Module.html)。
        state_dict() 返回参数 dict，load_state_dict() 加载参数，
        往返后两个模型的参数应完全一致。
        """
        model_a = nn.Linear(10, 5)
        model_b = nn.Linear(10, 5)

        # 初始权重应不同（随机初始化）
        sd_a = model_a.state_dict()
        sd_b = model_b.state_dict()
        assert not torch.equal(sd_a["weight"], sd_b["weight"]), (
            "两个随机初始化的模型权重应不同"
        )

        # 往返: model_a.state_dict() → model_b.load_state_dict()
        model_b.load_state_dict(sd_a)

        # 验证往返后权重一致
        sd_b_after = model_b.state_dict()
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b_after[key]), (
                f"state_dict 往返后 key '{key}' 权重不一致"
            )

    def test_state_dict_returns_dict_of_tensors(self):
        """L1 anchor: nn.Module.state_dict() 返回 dict[str, Tensor]。

        锚点：PyTorch 官方 API。
        state_dict() 返回 OrderedDict，key 是参数名（str），value 是 Tensor。
        """
        model = nn.Linear(10, 5)
        sd = model.state_dict()

        assert isinstance(sd, dict), (
            f"state_dict() 必须返回 dict，实际 {type(sd).__name__}"
        )
        for key, value in sd.items():
            assert isinstance(key, str), (
                f"state_dict key 必须是 str，实际 {type(key).__name__}"
            )
            assert isinstance(value, torch.Tensor), (
                f"state_dict['{key}'] 必须是 torch.Tensor，"
                f"实际 {type(value).__name__}"
            )

    # ============================================================
    # torch.save / torch.load 契约
    # ============================================================

    def test_torch_save_load_roundtrip_bare_state_dict(self, tmp_path):
        """L1 anchor: torch.save / torch.load 往返一致（裸 state_dict 格式）。

        锚点：PyTorch 官方 API
        (https://pytorch.org/docs/stable/generated/torch.save.html)。
        torch.save(state_dict, path) → torch.load(path) 返回相同 dict。
        torch.load 不限制文件扩展名（.pt / .ckpt / .pth 均可）。
        """
        model = nn.Linear(10, 5)
        sd = model.state_dict()

        # 测试 .pt 扩展名
        pt_path = tmp_path / "model.pt"
        torch.save(sd, pt_path)
        loaded = torch.load(pt_path, weights_only=True)

        assert isinstance(loaded, dict), "torch.load 应返回 dict"
        for key in sd:
            assert key in loaded, f"torch.load 后缺少 key '{key}'"
            assert torch.equal(sd[key], loaded[key]), (
                f"torch.load 后 key '{key}' 权重不一致"
            )

        # 测试 .ckpt 扩展名（PyTorch torch.load 不限制扩展名）
        ckpt_path = tmp_path / "model.ckpt"
        torch.save(sd, ckpt_path)
        loaded_ckpt = torch.load(ckpt_path, weights_only=True)
        for key in sd:
            assert torch.equal(sd[key], loaded_ckpt[key]), (
                f"torch.load .ckpt 后 key '{key}' 权重不一致"
            )

    def test_torch_load_supports_weights_only_parameter(self):
        """L1 anchor: torch.load 支持 weights_only 参数，锚点：PyTorch 2.0+ 安全加载。

        锚点：PyTorch 官方文档
        (https://pytorch.org/docs/stable/generated/torch.load.html)。
        PyTorch 2.0+ 引入 weights_only 参数，True 时仅反序列化 tensor，
        防止任意代码执行（安全加载）。
        """
        sig = inspect.signature(torch.load)
        assert "weights_only" in sig.parameters, (
            "torch.load 必须支持 weights_only 参数（PyTorch 2.0+ 安全加载）"
        )
        # 验证 weights_only 是关键字参数（有默认值）
        param = sig.parameters["weights_only"]
        assert param.default is False or param.default is None, (
            f"torch.load weights_only 默认值应为 False 或 None，"
            f"实际 {param.default!r}"
        )

    # ============================================================
    # load_checkpoint_flexible 契约
    # ============================================================

    def test_load_checkpoint_flexible_loads_bare_state_dict(self, tmp_path):
        """L1 anchor: load_checkpoint_flexible 加载裸 state_dict（torch.save 格式）。

        锚点：PyTorch 官方 API + senseframe.common.checkpoint 设计。
        裸 state_dict = torch.save(model.state_dict(), path) 的输出。
        load_checkpoint_flexible 应识别此格式并正确加载。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model_src = nn.Linear(10, 5)
        model_dst = nn.Linear(10, 5)

        # 初始权重应不同
        assert not torch.equal(
            model_src.state_dict()["weight"], model_dst.state_dict()["weight"]
        ), "两个随机初始化的模型权重应不同"

        # 保存裸 state_dict
        ckpt_path = tmp_path / "bare.pt"
        torch.save(model_src.state_dict(), ckpt_path)

        # 加载
        info = load_checkpoint_flexible(ckpt_path, model_dst, weights_only=True)

        # 验证 source_format
        assert info["source_format"] == "bare_state_dict", (
            f"裸 state_dict 应识别为 'bare_state_dict'，"
            f"实际 {info['source_format']!r}"
        )

        # 验证权重已加载
        for key in model_src.state_dict():
            assert torch.equal(
                model_src.state_dict()[key], model_dst.state_dict()[key]
            ), f"加载后 key '{key}' 权重不一致"

    def test_load_checkpoint_flexible_loads_lightning_checkpoint(self, tmp_path):
        """L1 anchor: load_checkpoint_flexible 加载 Lightning checkpoint 格式。

        锚点：Lightning checkpoint 官方格式
        (https://lightning.ai/docs/pytorch/stable/common/checkpointing.html)。
        Lightning checkpoint 顶层含 "state_dict" key，内含模型权重。
        load_checkpoint_flexible 应识别此格式并正确加载。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model_src = nn.Linear(10, 5)
        model_dst = nn.Linear(10, 5)

        # 构造 Lightning checkpoint 格式（含 "state_dict" 顶层 key）
        lightning_ckpt = {
            "epoch": 10,
            "global_step": 100,
            "pytorch-lightning_version": "2.0.0",
            "state_dict": model_src.state_dict(),
            "callbacks": [],
            "optimizer_states": [],
        }

        ckpt_path = tmp_path / "lightning.ckpt"
        torch.save(lightning_ckpt, ckpt_path)

        # 加载（weights_only=False 因含 callbacks 等 Python 对象）
        info = load_checkpoint_flexible(ckpt_path, model_dst, weights_only=False)

        # 验证 source_format
        assert info["source_format"] == "lightning", (
            f"Lightning checkpoint 应识别为 'lightning'，"
            f"实际 {info['source_format']!r}"
        )

        # 验证权重已加载
        for key in model_src.state_dict():
            assert torch.equal(
                model_src.state_dict()[key], model_dst.state_dict()[key]
            ), f"加载后 key '{key}' 权重不一致"

    def test_load_checkpoint_flexible_strips_model_prefix(self, tmp_path):
        """L1 anchor: load_checkpoint_flexible 剥离 "model." 前缀（Lightning GenericLightningModule）。

        锚点：Lightning checkpoint 格式 + GenericLightningModule 结构。
        GenericLightningModule 将裸模型存为 self.model，导致 state_dict key 带
        "model." 前缀。load_checkpoint_flexible 应剥离此前缀后加载。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model_src = nn.Linear(10, 5)
        model_dst = nn.Linear(10, 5)

        # 构造带 "model." 前缀的 Lightning checkpoint
        prefixed_state_dict = {
            f"model.{key}": value for key, value in model_src.state_dict().items()
        }
        lightning_ckpt = {
            "epoch": 5,
            "global_step": 50,
            "state_dict": prefixed_state_dict,
        }

        ckpt_path = tmp_path / "prefixed.ckpt"
        torch.save(lightning_ckpt, ckpt_path)

        # 加载
        info = load_checkpoint_flexible(ckpt_path, model_dst, weights_only=False)

        # 验证 source_format 和前缀剥离
        assert info["source_format"] == "lightning", (
            f"带前缀的 Lightning checkpoint 应识别为 'lightning'，"
            f"实际 {info['source_format']!r}"
        )
        assert info["stripped_prefix"] == "model.", (
            f"应剥离 'model.' 前缀，实际 {info['stripped_prefix']!r}"
        )

        # 验证权重已正确加载（前缀剥离后 key 应匹配 model_dst.state_dict()）
        for key in model_src.state_dict():
            assert torch.equal(
                model_src.state_dict()[key], model_dst.state_dict()[key]
            ), f"前缀剥离后加载，key '{key}' 权重不一致"

    def test_load_checkpoint_flexible_raises_on_missing_file(self, tmp_path):
        """L1 anchor: load_checkpoint_flexible 对不存在文件抛 FileNotFoundError。

        锚点：Python 官方异常语义 + PyTorch torch.load 行为。
        文件不存在时应抛 FileNotFoundError（而非静默返回空）。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        missing_path = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            load_checkpoint_flexible(missing_path, nn.Linear(10, 5))

    def test_load_checkpoint_flexible_accepts_pt_and_ckpt_extensions(self, tmp_path):
        """L1 anchor: load_checkpoint_flexible 支持 .pt / .ckpt 扩展名。

        锚点：PyTorch torch.load 不限制扩展名 + Lightning .ckpt 有特殊格式。
        torch.load 可加载任意扩展名文件，.pt 是 PyTorch 惯例，
        .ckpt 是 Lightning checkpoint 惯例。load_checkpoint_flexible 应两者都支持。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model_src = nn.Linear(10, 5)

        # 测试 .pt 扩展名（裸 state_dict）
        pt_path = tmp_path / "model.pt"
        torch.save(model_src.state_dict(), pt_path)
        model_pt = nn.Linear(10, 5)
        info_pt = load_checkpoint_flexible(pt_path, model_pt, weights_only=True)
        assert info_pt["source_format"] == "bare_state_dict"

        # 验证权重正确加载
        for key in model_src.state_dict():
            assert torch.equal(
                model_src.state_dict()[key], model_pt.state_dict()[key]
            ), f".pt 加载后 key '{key}' 权重不一致"

        # 测试 .ckpt 扩展名（Lightning 格式）
        ckpt_path = tmp_path / "model.ckpt"
        lightning_ckpt = {
            "state_dict": model_src.state_dict(),
            "epoch": 0,
        }
        torch.save(lightning_ckpt, ckpt_path)
        model_ckpt = nn.Linear(10, 5)
        info_ckpt = load_checkpoint_flexible(ckpt_path, model_ckpt, weights_only=False)
        assert info_ckpt["source_format"] == "lightning"

        # 验证权重正确加载
        for key in model_src.state_dict():
            assert torch.equal(
                model_src.state_dict()[key], model_ckpt.state_dict()[key]
            ), f".ckpt 加载后 key '{key}' 权重不一致"

    def test_lightning_checkpoint_format_has_required_fields(self, tmp_path):
        """L1 anchor: Lightning checkpoint 含 state_dict 字段（Lightning 官方格式）。

        锚点：Lightning checkpoint 官方格式
        (https://lightning.ai/docs/pytorch/stable/common/checkpointing.html#contents)。
        Lightning checkpoint 顶层 dict 必须含 "state_dict" key（模型权重）。
        load_checkpoint_flexible 依赖此字段识别 Lightning 格式。
        """
        from senseframe.common.checkpoint import load_checkpoint_flexible

        model = nn.Linear(10, 5)

        # Lightning 官方 checkpoint 格式
        lightning_ckpt = {
            "epoch": 10,
            "global_step": 100,
            "pytorch-lightning_version": "2.0.0",
            "state_dict": model.state_dict(),
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        }

        ckpt_path = tmp_path / "full_lightning.ckpt"
        torch.save(lightning_ckpt, ckpt_path)

        # 验证 load_checkpoint_flexible 能识别 Lightning 格式
        model_dst = nn.Linear(10, 5)
        info = load_checkpoint_flexible(ckpt_path, model_dst, weights_only=False)

        assert info["source_format"] == "lightning", (
            "含 state_dict 字段的 checkpoint 应识别为 Lightning 格式"
        )
        assert info["num_keys_loaded"] > 0, (
            "Lightning checkpoint 应加载至少 1 个 key"
        )

        # 验证权重正确加载
        for key in model.state_dict():
            assert torch.equal(
                model.state_dict()[key], model_dst.state_dict()[key]
            ), f"Lightning checkpoint 加载后 key '{key}' 权重不一致"
