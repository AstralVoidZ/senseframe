"""SenseFrame 闭环演示（RFC-002 阶段 s3）。

最小可运行示例：合成数据 → 训练 → feedback → recommend_next → 可视化。

注意：本演示脚本中训练流程为注释占位，如需真实训练请参考
scripts/run_experiment.py 或使用 `python -m senseframe.cli experiment` 命令。

运行：
    python examples/closed_loop_demo.py

依赖：
    torch, pytorch_lightning, senseframe
"""
import csv
import random
import sys
from pathlib import Path
from tempfile import mkdtemp

# 确保能导入项目根目录下的 senseframe 包（直接运行本脚本时生效）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def generate_synthetic_data(csv_path: Path, n_samples=120, n_features=10, n_classes=3):
    """生成合成 CSV 数据（特征与标签有相关性，便于模型学习）。"""
    random.seed(42)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(n_features)] + ["label"])
        for _ in range(n_samples):
            features = [random.gauss(0, 1) for _ in range(n_features)]
            label = random.randint(0, n_classes - 1)
            # 让前几个特征与标签相关
            for i in range(min(3, n_features)):
                features[i] += label * (i + 1) * 0.5
            writer.writerow(features + [label])


def main():
    print("=" * 60)
    print("SenseFrame 闭环演示")
    print("=" * 60)

    # 1. 准备数据
    data_dir = Path(mkdtemp())
    csv_path = data_dir / "synthetic.csv"
    generate_synthetic_data(csv_path)
    print(f"\n[1] 合成数据已生成: {csv_path}")

    # 2. 构造 config 并训练
    try:
        from senseframe.engine.config import ExperimentConfig
        # 构造最小 config（根据实际 API 调整）
        # config = ExperimentConfig(...)
        # result = run_experiment(config)
        print("[2] 训练流程（需根据 ExperimentConfig 实际结构调整）")
    except Exception as e:
        print(f"[2] 训练跳过（{e}）")

    # 3. 探索状态管理
    from senseframe.exploration import ExplorationTracker, SearchSpaceMap
    tracker = ExplorationTracker()
    # tracker.add_trial(strategy={"loss": "cross_entropy"}, result={"val_accuracy": 0.75})
    print("[3] ExplorationTracker 已初始化")

    # 4. 推荐下一步
    recs = tracker.recommend_next(task_type="classification", top_k=3)
    print(f"[4] recommend_next 返回 {len(recs)} 条推荐:")
    for i, rec in enumerate(recs):
        print(f"    {i+1}. {rec.get('strategy', {})} — {rec.get('reason', '')}")

    # 5. 搜索空间地图
    space_map = SearchSpaceMap(tracker)
    overview = space_map.overview()
    tech = overview.get("techniques", {})
    total = tech.get("total_techniques", 0)
    print(f"[5] 搜索空间: {total} 个技术可用")

    # 6. 技能库
    from senseframe.skills import list_skills, search_skills
    skills = list_skills()
    print(f"[6] 技能库: {len(skills)} 个技能")
    if skills:
        results = search_skills("classification loss", top_k=2)
        print(f"    检索 'classification loss': {len(results)} 条命中")

    # 7. 可视化
    from senseframe.observability import ExplorationDashboard
    dashboard = ExplorationDashboard(tracker)
    print("\n[7] Dashboard:")
    print(dashboard.render(format="text"))

    print("=" * 60)
    print("演示完成。")
    print("实际使用时：1) run_experiment 训练 → 2) feedback 自动生成 → 3) recommend_next 推荐方向 → 4) 调整策略再训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
