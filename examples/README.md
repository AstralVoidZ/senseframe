# SenseFrame AutoML 测试 PROMPT 集

本目录存放用于测试 SenseFrame 的 Agent PROMPT。每个 PROMPT 设计为一种训练场景，
要求 Agent 在执行训练的同时，以多领域专家身份对 SenseFrame 进行边训练边审视，
最终输出多维度测试报告。

## 设计哲学

**不是为测试而测试，而是为发现真实问题而测试。**

每个 PROMPT 都要求 Agent：
1. 真实执行训练流程（不是 mock，不是 dry-run）
2. 边执行边以专家身份审视框架行为
3. 发现问题必须记录，不得掩盖
4. 输出结构化测试报告，含问题清单与改进建议

## 文件清单

| 文件 | 难度 | 场景 | 耗时 | 覆盖特性 |
|------|------|------|------|----------|
| [01_simple_training_prompt.md](./01_simple_training_prompt.md) | 简单 | 单模型监督训练 | 5-15 min | 声明式路径、配置校验、预检、基础训练、错误码 |
| [02_medium_hpo_prompt.md](./02_medium_hpo_prompt.md) | 中等 | HPO 超参搜索 + 多模型对比 | 15-40 min | HPO、Ask-Tell、ExplorationTracker、资源路由、技能库 |
| [03_hard_full_loop_prompt.md](./03_hard_full_loop_prompt.md) | 困难 | 完整闭环 + 自监督 + 断点续跑 | 30-90 min | 自监督两阶段、Pipeline 编排、断点续跑、自省协议全量、导出 |

## 自省协议规范

自省协议是本测试集的核心创新。要求 Agent 在训练的每个阶段后，
以四个专家视角审视 SenseFrame 的行为，并填写自省卡。

### 四个专家视角

1. **AI/Agent 视角**：Agent 操控性
   - API 是否对 Agent 友好（可程序化分支、结构化输出、无歧义）
   - 错误信息是否足够 Agent 自主决策
   - 自省 API 是否真正减少源码阅读
   - 探索-反馈回路是否闭合

2. **机器学习视角**：训练科学性
   - 默认策略（loss/metric/normalization）是否合理
   - 模型选择与数据特征是否匹配
   - 评估指标是否充分
   - 过拟合/欠拟合检测是否有效
   - 超参搜索空间是否合理

3. **AutoML 视角**：自动化程度
   - 端到端自动化程度（从意图到产物）
   - 资源路由是否真正基于硬件
   - 错误恢复能力（重试、断点续跑）
   - 探索历史是否可复用
   - 技能库是否真正减少重复工作

4. **WiFi CSI 信号处理视角**：领域适配
   - CSI 数据预处理是否正确（reshape、normalization、stride）
   - 模型架构是否适合 CSI 信号特征（时序、空间、多天线）
   - 数据集契约是否准确（NTU-Fi / UT-HAR / Widar3.0 差异）
   - 自监督预训练对 CSI 是否有意义

### 自省卡模板

每个 stage 执行后填写：

```markdown
## 自省卡：stage_<name>

### 执行事实
- stage: <name>
- 耗时: <s>
- 成功/失败: <status>
- 产出字段: <list>

### AI/Agent 视角
- API 友好度: [1-5]
- 观察: <具体观察，不是空话>
- 问题: <如无写"无">

### 机器学习视角
- 策略合理性: [1-5]
- 观察: <具体观察>
- 问题: <如无写"无">

### AutoML 视角
- 自动化程度: [1-5]
- 观察: <具体观察>
- 问题: <如无写"无">

### WiFi CSI 视角
- 领域适配: [1-5]
- 观察: <具体观察>
- 问题: <如无写"无">
```

### 自省纪律

- **必须基于执行事实**：不得臆测，每条观察必须引用具体日志/输出/字段
- **问题不得掩盖**：发现的问题必须记录，即使训练最终成功
- **评分要有区分度**：不得全部给 5 分，发现问题必须扣分
- **改进建议要具体**：不得写"优化体验"等空话，必须指出改哪个 API/字段/文档

## 测试报告规范

每个 PROMPT 执行完毕后，Agent 必须输出测试报告，结构如下：

```markdown
# SenseFrame 测试报告：<场景名>

## 执行摘要
- 测试时间: <YYYY-MM-DD HH:MM>
- 环境: <WSL2 / GPU 型号 / Python / PyTorch 版本>
- 场景: <简单/中等/困难>
- 最终状态: <成功/失败/部分成功>
- 耗时: <min>

## 训练结果
- 数据集: <name>
- 模型: <name>
- 最终指标: <val_accuracy / val_loss>
- 训练轮数: <actual / planned>
- 是否早停: <yes/no>

## 自省卡汇总

### 各 stage 评分矩阵
| Stage | AI/Agent | ML | AutoML | CSI | 平均 |
|-------|----------|----|--------|-----|------|
| validate | x | x | x | x | x.x |
| preflight | x | x | x | x | x.x |
| ... | | | | | |

### 关键发现（按严重度排序）
1. [严重] <问题描述 + 复现步骤 + 影响>
2. [中等] <问题描述 + 复现步骤 + 影响>
3. [轻微] <问题描述 + 复现步骤 + 影响>

## 多维度分析

### 架构维度
- <观察 + 证据>

### API 易用性维度
- <观察 + 证据>

### 文档完整性维度
- <观察 + 证据>

### 错误处理维度
- <观察 + 证据>

### 可观测性维度
- <观察 + 证据>

## 改进建议（按优先级排序）
1. [P0] <具体建议 + 影响范围 + 预期收益>
2. [P1] <具体建议 + 影响范围 + 预期收益>
3. [P2] <具体建议 + 影响范围 + 预期收益>

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐 / 谨慎推荐 / 不推荐>
- 一句话总结: <...>
```

## 使用方法

### 1. 环境准备（WSL2）

```bash
cd <DEPLOY_ROOT>  # 或你的部署目录
pip install -r requirements.txt
python -c "import senseframe; print(senseframe.__version__)"
python -m senseframe.cli probe
```

### 2. 加载 SKILL

在 opencode / Claude Code TUI 中：

```
skill({ name: "senseframe" })
```

### 3. 执行 PROMPT

将对应 PROMPT 文件内容粘贴给 Agent，或用 `@文件路径` 引用。
Agent 会按 PROMPT 指令执行训练并输出测试报告。

### 4. 推荐执行顺序

1. 先跑 `01_simple` 验证基础环境
2. 再跑 `02_medium` 验证 HPO 与资源路由
3. 最后跑 `03_hard` 验证完整闭环

每个 PROMPT 独立可执行，也可串行执行以累积测试报告。
