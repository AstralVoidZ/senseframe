"""L2 编排 spec 契约测试。

锚点来源：项目内独立 spec 文档（非源码自身）。
- RFC Phase D（Pipeline 9 stage 设计）
- 设计文档 0.6 节（PipelineRun 5 状态 7 转换 + 3 幂等短路）
- RFC-003（AutoML ε1 设计）
- Lightning Callback lifecycle（外部库 API，但编排行为由项目 spec 定义）

禁止：源码自身常量作为断言目标（如 len(VALID_TRANSITIONS) == 7 不引用 spec）。
"""
