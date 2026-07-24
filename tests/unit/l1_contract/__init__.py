"""L1 外部协议契约测试。

锚点来源：外部协议规范 / 外部库官方 API / 标准公式。
- MCP spec（Model Context Protocol）
- pydantic v2 文档
- PyTorch Lightning 官方 API
- PyTorch 官方 API（torch.load / state_dict）
- SenseFrame 自有 search_protocol.py Protocol 定义
- K8s CRD 状态机范式 + 设计文档 0.6 节
- CloudEvents 1.0 规范
- PSNR 标准定义 10·log10(MAX²/MSE)

禁止：源码自身常量作为断言目标（自证断言）。
"""
