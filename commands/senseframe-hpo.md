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
- 数据集: `CSI_DATASETS/`（NTU-Fi-HAR / UT-HAR / Widar3.0）
- GPU: 8GB 显存，HPO 必须串行（不得并发 trial）

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
sf.stage_io("stage_train")
sf.stage_io("stage_eval")

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
    tracker.record_trial(trial_id=trial.trial_id, strategy=trial.params, result=val_acc)

# 4. 查看最优
best = sm.best_trial(study_id)
print(f"Best: value={best.value}, params={best.params}")

# 5. 方案 G：验证每个 trial 的产物溯源
from pathlib import Path
for i in range($2):
    trial_dir = Path(f"runs/trial_{i}")  # 按实际 output_dir 调整
    manifest_path = trial_dir / "manifest.json"
    if manifest_path.exists():
        m = sf.load_manifest(manifest_path)
        report = sf.verify_artifacts(m)
        print(f"trial_{i}: {report['verified']}/{report['total']} verified")
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
recommendation = tracker.recommend_next()
print(f"Recommendation: {recommendation}")

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

```markdown
# SenseFrame 测试报告：HPO 超参搜索

## 执行摘要
- 环境: <Python/torch/CUDA>
- 数据集: <$1> | trials: <$2>
- 状态: <成功/失败> | 耗时: <min>

## HPO 结果表
| Trial | lr | batch_size | optimizer | val_accuracy | 耗时(s) | 状态 |
|-------|----|-----------|-----------|-------------|---------|------|
| 1 | ... | ... | ... | ... | ... | ok/fail |
| ... | | | | | | |

## 多模型对比表
| 模型 | 最优 val_accuracy | 平均耗时 | 备注 |
|------|-------------------|---------|------|
| ResNet18 | ... | ... | ... |
| <其他> | ... | ... | ... |

## 探索历史
<trial 序列 + parent_trial_id 关系>

## 自省评分矩阵
| 阶段 | AI/Agent | ML | AutoML | CSI | 平均 |
|------|----------|----|--------|-----|------|
| 资源路由 | x | x | x | x | x.x |
| 自省API | x | x | x | x | x.x |
| HPO执行 | x | x | x | x | x.x |
| ExplorationTracker | x | x | x | x | x.x |
| 技能库 | x | x | x | x | x.x |
| 多模型 | x | x | x | x | x.x |

## 关键发现
1. [严重/中等/轻微] <问题 + 复现步骤 + 影响>

## 改进建议
1. [P0/P1/P2] <具体建议>

## 结论
- 综合评分: <x.x / 5.0>
- 推荐度: <推荐/谨慎推荐/不推荐>
- 一句话总结: <...>
```

## Constraints

- 禁止伪造 trial：每个 trial 必须真实训练
- 禁止跳过自省 API：编码前必须查询字段契约
- 禁止掩盖失败：trial 失败必须记录
- 禁止全 5 分：评分必须有区分度
- GPU 8GB 限制：HPO 必须串行，不得并发 trial
