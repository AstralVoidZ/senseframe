---
description: "SenseFrame 中等场景：HPO 超参搜索 + 多模型对比 + 技能库"
subtask: false
---

<!--

command: /senseframe-hpo
场景: 中等 (15-40 min)
目的: 验证 Ask-Tell 搜索协议 + ExplorationTracker + 资源路由 + 技能库

用法:
  /senseframe-hpo                        # 默认 NTU-Fi_HAR + 5 trials
  /senseframe-hpo NTU-Fi_HAR 8           # 自定义数据集 + trial 数

$1 = 数据集名 (默认 NTU-Fi_HAR)
$2 = trial 数 (默认 5)
-->

# SenseFrame HPO 超参搜索测试

## Role

你是 AI Agent、机器学习、AutoML、WiFi CSI 信号处理四领域专家。
本测试验证 SenseFrame 的探索-反馈回路、资源路由、技能库等自动化能力。

## Objective

用 SenseFrame 命令式路径完成 HPO 超参搜索：
- 数据集: `$1`（默认 `NTU-Fi_HAR`）
- 试验数: 至少 `$2` 次（默认 5）
- 候选模型: ResNet18 + 至少 1 个对比模型
- 搜索维度: learning_rate + batch_size + optimizer

**关键**: 编码前必须先用自省 API 查询字段契约，不得盲猜 API。

## Context

- 工作目录: `.`（已部署 SenseFrame）
- SKILL `senseframe` 已加载
- 数据集: `resource/CSI_DATASETS/`（如未部署，可改 `--data-root` 或软链；NTU-Fi-HAR / UT-HAR / Widar3.0）
- GPU: 8GB 显存，HPO 必须串行（不得并发 trial）
- **报告输出目录**: `report/`（不存在时自动创建）
- **报告命名格式**: `report/hpo_<dataset>_<trials>_<YYYYMMDD_HHMMSS>.md`
  - 例：`report/hpo_NTU-Fi_HAR_5_20260706_143025.md`
  - 时间戳取 HPO 开始时刻，避免重名覆盖

## Execution Protocol

### Step 1: 环境与资源路由探测

**Do**:
```bash
python -m senseframe.cli probe
python -m senseframe.cli list-models --dataset $1
python -m senseframe.cli recommend --dataset $1 --priority balanced
```

**Output**: 可用模型清单 + 资源路由推荐
**Gate**: probe/recommend 输出结构化

**Introspect**:
- [ ] recommend 是否基于硬件资源选模型？还是固定推荐？
- [ ] 路由级别（L0-L4）是否合理？
- [ ] CPU/GPU 切换是否改变策略？（用 `CUDA_VISIBLE_DEVICES=""` 测试）

### Step 2: 自省 API 查询字段契约

**Do**: 编码前必须先查询契约，不得盲猜。

**⚠️ 方案 B 入口点契约**：在 Python 脚本中调用任何 registry 查询 API
（`list_models`/`list_datasets`/`context_schema` 等自省 API 内部可能查询
registry）前，必须先调用 `sf.activate_lazy_scenes()` 显式激活延迟场景。
CQS 合规改造后 getter 不再有自动注册副作用，否则 wifi_csi 等延迟场景的
模型/数据集元数据不在 registry 中。

```python
import senseframe as sf

# 方案 B：先激活延迟场景，再查询 registry
sf.activate_lazy_scenes()

# 查询 PipelineContext 字段契约
sf.context_schema()

# 查询 stage IO
# 注意：stage 名不带 "stage_" 前缀（如 "train"/"eval"），带前缀会返回 not found
sf.stage_io("train")
sf.stage_io("eval")

# 查询 pipeline DAG
sf.pipeline_graph()

# 查询数据集契约
sf.data_bundle_schema()
sf.data_profile_schema()
```

**Output**: 字段契约信息
**Gate**: 自省 API 全部可调用，输出结构化

**Introspect**:
- [ ] schema() 输出是否完整准确？
- [ ] stage_io reads/writes 是否与实际行为一致？
- [ ] 自省 API 是否真正减少源码阅读？

### Step 3: 编写并执行 HPO 脚本

**Do**: 基于自省 API 契约编写 HPO 脚本。

```python
import senseframe as sf
from senseframe.search_protocol import StudyManager, SearchSpace, ParameterSpec
from senseframe.exploration import ExplorationTracker

# 1. 搜索空间
space = SearchSpace(parameters=[
    ParameterSpec(name="lr", type="float", low=1e-4, high=1e-2, log=True),
    ParameterSpec(name="batch_size", type="int", low=8, high=64, step=8),
    ParameterSpec(name="optimizer", type="categorical",
                  choices=["adam", "sgd", "adamw"]),
])

# 2. 创建 Study
sm = StudyManager()
study_id = sm.create_study(
    name="hpo_test", direction="maximize",
    search_space=space, sampler="random",
)

# 3. Ask-Tell 循环（至少 $2 次）
tracker = ExplorationTracker()
for i in range($2):
    trial = sm.ask(study_id)
    config = build_config_from_trial(trial, model="ResNet18")
    output = sf.run_experiment(config)
    val_acc = output.final_eval.get("val_accuracy", 0.0)
    sm.tell(trial.trial_id, value=val_acc)
    tracker.record_trial(trial_id=trial.trial_id, strategy=trial.params, result={"val_accuracy": val_acc})

# 4. 查看最优
best = sm.best_trial(study_id)
print(f"Best: value={best.value}, params={best.params}")

# 5. 方案 G：验证每个 trial 的产物溯源
from pathlib import Path
for i in range($2):
    trial_dir = Path(f"runs/trial_{i}")  # 按实际 output_dir 调整
    manifest_path = trial_dir / "manifest.json"
    if manifest_path.exists():
        # verify_artifacts 接受 output_dir（含 manifest.json），返回 {产物名: hash 是否匹配}
        report = sf.verify_artifacts(trial_dir)
        verified = sum(1 for ok in report.values() if ok)
        print(f"trial_{i}: {verified}/{len(report)} verified")
```

**关键要求**:
- 每个 trial 必须真实训练（禁止 mock）
- trial 失败必须记录，不得从报告中删除
- 至少对比 2 个模型（ResNet18 + 另一个）

**Output**: HPO 结果表（每个 trial 的 params + val_accuracy + 耗时）
**Gate**: 至少完成 $2 次真实 trial

**Introspect — 每次 trial 后**:
- [ ] Ask-Tell 是否符合 Optuna 语义？
- [ ] SearchSpace 三种类型（float/int/categorical）是否可用？
- [ ] Sampler 是否可插拔？
- [ ] best_trial 是否正确返回最优？
- [ ] **方案 E 验证**：HPO 搜索空间是否覆盖了 `lr/batch_size/optimizer`？是否**避免重复搜索**已最优化的默认维度（`weight_decay=1e-4`/`early_stopping=5`/`scheduler="cosine"`）？默认值是否在 trial 中作为 baseline 出现？
- [ ] **方案 F 验证**：每次 trial 结束后 `ctx.trainer/module/model` 是否为 None（`release_resources` 已执行）？串行 trial 间内存是否稳定不增长？WSL2 环境是否出现内存回收提示？
- [ ] **方案 G 验证**：每个 trial 是否在 `output_dir` 生成 `manifest.json`？`verify_artifacts()` 是否全部通过？跨 trial 的 manifest 是否可通过 `load_manifest()` 加载对比？

### Step 4: ExplorationTracker 验证

**Do**:
```python
# 查看探索历史
history = tracker.history
print(f"Total trials: {len(history)}")

# 测试 recommend_next
recommendations = tracker.recommend_next()
print(f"Recommendations: {recommendations}")

# 测试 parent_trial_id 回溯
if len(history) > 1:
    print(f"Parent chain: {history[-1].parent_trial_id}")
```

**Output**: 探索历史 + 推荐结果
**Gate**: history 持久化，recommend_next 可调用

**Introspect**:
- [ ] 探索历史是否持久化？
- [ ] record_trial 是否真正记录策略与结果？
- [ ] recommend_next 是否给出合理建议？
- [ ] parent_trial_id 回溯是否可用？

### Step 5: 技能库测试

**Do**:
```python
# 保存最优策略为技能
sf.save_skill(
    name="hpo_best",
    description="HPO 最优策略",
    code=f"# lr={best.params['lr']}, bs={best.params['batch_size']}",
    version="1.0.0",
)

# 检索
matches = sf.search_skills(query="HPO")
all_skills = sf.list_skills()

# 加载
skill = sf.load_skill("hpo_best")
```

**Output**: 技能保存/检索/加载结果
**Gate**: save/search/load 全部成功

**Introspect**:
- [ ] save_skill 是否持久化？存储位置？
- [ ] search_skills 检索质量如何？
- [ ] load_skill 是否恢复策略？
- [ ] 技能库是否真正减少重复工作？

### Step 6: 多模型对比

**Do**: 用相同数据集、搜索空间对比至少 2 个模型。

**Output**: 模型对比表（模型 × 最优 val_accuracy × 耗时）
**Gate**: 至少 2 个模型对比完成

**Introspect**:
- [ ] 模型注册是否统一？
- [ ] 不同模型输入形状适配是否自动处理？
- [ ] recommend 推荐的模型是否与实际表现一致？

## Introspection Protocol

四视角评分，每步必填：

1. **AI/Agent** [1-5]: API 可程序化？error_code 结构化？自省 API 价值？
2. **ML** [1-5]: 搜索空间合理？模型对比公平？过拟合检测有效？
3. **AutoML** [1-5]: 端到端自动化？探索回路闭合？技能库复用？
4. **CSI** [1-5]: CSI 数据特征匹配？模型架构适配？

**纪律**: 基于执行事实；问题不掩盖；评分有区分度；建议指向具体 API。

## Output Contract

执行完毕后，**必须**将报告写入 `report/hpo_<dataset>_<trials>_<YYYYMMDD_HHMMSS>.md`，
并在 stdout 输出报告路径。报告内容必须包含以下章节，**禁止省略**任何章节。

```markdown
# SenseFrame 测试报告：HPO 超参搜索

## 报告元数据
- 报告路径: `report/hpo_<dataset>_<trials>_<YYYYMMDD_HHMMSS>.md`
- 生成时间: <ISO8601>
- 测试命令: `/senseframe-hpo <dataset> <trials>`
- 框架版本: <senseframe.__version__>

## 执行摘要
- 环境: Python <version> | torch <version> | CUDA <available/version>
- 硬件: <CPU/GPU 型号 + 显存>
- 数据集: <$1> | trials: <$2> | 候选模型: <列表>
- 搜索空间: lr=[low, high], batch_size=[low, high, step], optimizer=<choices>
- 状态: <成功/失败/部分成功> | 总耗时: <min>

## 资源路由与模型清单（Step 1 产物）
- probe 输出: <device/CPU 核数/显存>
- list-models 返回（数据集 <$1> 下的全部模型）:
  | model_id | paradigm | default_epochs | enabled |
  |----------|----------|---------------|---------|
  | <name>   | <type>   | <N>           | ✓/✗     |
- recommend 输出: <推荐模型 + 路由级别 + 理由>
- GPU vs CPU 路由对比（如执行）: <差异说明>

## 字段契约查询（Step 2 产物）
- context_schema 关键字段: <列出 5-10 个关键字段 + fill_stage + type>
- stage_io("train") reads: <list>, writes: <list>
- stage_io("eval") reads: <list>, writes: <list>
- pipeline_graph: <stage 依赖关系文字版>
- data_bundle_schema: <字段列表>
- data_profile_schema: <字段列表>
- 自省 API 调用是否全部成功: <yes/no + 失败列表>

## HPO 结果详情（Step 3 产物）
### 完整 trial 结果表
| Trial | model | lr | batch_size | optimizer | val_accuracy | val_macro_f1 | val_loss | 耗时(s) | epochs | 早停 | 状态 | output_dir |
|-------|-------|----|-----------|-----------|-------------|--------------|----------|---------|--------|------|------|-----------|
| 1 | ResNet18 | <val> | <val> | <val> | <val> | <val> | <val> | <val> | <N> | <yes/no> | ok/fail | <path> |
| 2 | <其他> | ... | | | | | | | | | | |
| ... | | | | | | | | | | | | |

### 最优 trial
- best_trial: trial_idx=<N>, value=<val_accuracy>, params=<dict>
- study_id: <id>
- 完整搜索历史: <N> 个 trial（<success> 成功 / <fail> 失败）

### Trial 失败详情（如有失败）
| Trial | 失败原因 | error_msg | output_dir |
|-------|---------|-----------|-----------|
| <N> | <异常类型> | <完整错误信息> | <path> |

### 资源监控
- 串行 trial 间内存稳定性: <stable/growing>（如 growing 给出 delta 值）
- WSL2 内存回收提示: <yes/no + 次数>
- 方案 F 验证: trainer/module/model 释放: <yes/no>

## 产物溯源校验（Step 3 方案 G）
每个 trial 的 manifest 校验结果：
| Trial | output_dir | artifacts 总数 | verified 数 | 失败产物 |
|-------|-----------|---------------|------------|---------|
| 1 | <path> | <N> | <N> | <list or 无> |
| 2 | | | | |
| ... | | | | |

## ExplorationTracker 验证（Step 4 产物）
- 历史记录数: <N>
- 历史记录示例（最近 3 条）:
  | trial_id | strategy | result | parent_trial_id |
  |----------|----------|--------|-----------------|
  | <id> | <params> | <val_acc> | <parent or None> |
- recommend_next 输出: <recommendation>
- 持久化路径: <如 .senseframe/exploration.json>

## 技能库测试（Step 5 产物）
- 保存技能: name=<>, version=<>, 存储路径=<>
- 检索结果: query="HPO", 匹配数=<N>, 匹配项=<list>
- 加载技能: name=<>, 加载是否成功=<yes/no>
- 技能库存储位置: <如 .senseframe/skills/>

## 多模型对比（Step 6 产物）
| 模型 | 最优 val_accuracy | 最优 trial params | 平均耗时(s) | 平均 epochs | 备注 |
|------|-------------------|------------------|------------|------------|------|
| ResNet18 | <val> | <lr=X, bs=Y> | <val> | <N> | <如"收敛稳定"> |
| <其他> | <val> | | | | <如"早期过拟合"> |

对比结论: <哪个模型在该数据集表现更好 + 原因分析>

## 自省评分矩阵
| 阶段 | AI/Agent | ML | AutoML | CSI | 平均 | 关键扣分原因 |
|------|----------|----|--------|-----|------|------------|
| 资源路由 | x | x | x | x | x.x | <如"recommend 固定推荐"> |
| 自省API | x | x | x | x | x.x | <如"stage_io 不准"> |
| HPO执行 | x | x | x | x | x.x | <如"Sampler 不可插拔"> |
| ExplorationTracker | x | x | x | x | x.x | <如"recommend_next 空"> |
| 技能库 | x | x | x | x | x.x | <如"search 质量差"> |
| 多模型 | x | x | x | x | x.x | <如"输入形状未自动适配"> |

## 关键发现（按严重度排序）
每个发现必须包含：复现命令 + 实际输出 + 期望输出 + 影响范围 + 严重度

1. **[严重]** <问题标题>
   - 复现命令: `<完整命令>`
   - 实际输出: `<粘贴实际输出>`
   - 期望输出: `<应该是什么>`
   - 根因分析: <代码位置 + 逻辑错误>
   - 影响: <对哪些功能/用户/场景有影响>
   - 建议修复: <具体修复方向>

2. **[中等/轻微]** <问题标题>
   - ...（同上格式）

## 改进建议（按优先级排序）
每条建议必须含：优先级 + 具体修改点 + 影响文件/模块 + 预期收益

1. **[P0]** <建议>
   - 修改文件: `<file:line>`
   - 修改内容: <具体改什么>
   - 预期收益: <修复后效果>

2. **[P1/P2]** <建议>
   - ...

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
- 下一步建议: <如"扩展搜索空间到 weight_decay / 切换 Bayesian sampler">
```

## Constraints

- 禁止伪造 trial：每个 trial 必须真实训练
- 禁止跳过自省 API：编码前必须查询字段契约
- 禁止掩盖失败：trial 失败必须记录
- 禁止全 5 分：评分必须有区分度，扣分必须填"关键扣分原因"
- 禁止省略章节：报告必须含全部章节，无内容时填"无"并说明原因
- 报告必须落盘到 `report/` 目录，禁止仅输出到 stdout
- GPU 8GB 限制：HPO 必须串行，不得并发 trial
