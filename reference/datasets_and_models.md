# 数据集与模型支持表

## 环境变量（必填）

WiFi CSI 场景依赖外部 SenseFi 代码库，框架不猜测路径，调用者必须显式提供：

| 环境变量 | 用途 | 示例 |
|----------|------|------|
| `SENSEFRAME_DATA_ROOT` | 数据集根目录（含 UT_HAR/、NTU-Fi_HAR/ 等子目录） | `/path/to/CSI_DATASETS` |
| `SENSEFRAME_SENSEFI_PATH` | SenseFi 代码库根目录（含 UT_HAR_model.py 等） | `/path/to/WiFi-CSI-Sensing-Benchmark` |

未设置时：`SENSEFRAME_DATA_ROOT` → 训练启动报错；`SENSEFRAME_SENSEFI_PATH` → 场景激活报 ImportError。

## WiFi CSI 场景支持的数据集

| 数据集名 | 目录名 | 类别数 | 输入形状 | loader | layout | 用途 |
|----------|--------|--------|----------|--------|--------|------|
| `UT_HAR_data` | UT_HAR | 7 | (1, 250, 90) | npy (tensor) | flat | 监督训练 |
| `NTU-Fi-HumanID` | NTU-Fi-HumanID | 14 | (3, 114, 500) | mat (csi_mat) | nested | 监督训练 / 自监督微调 |
| `NTU-Fi_HAR` | NTU-Fi_HAR | 6 | (3, 114, 500) | mat (csi_mat) | nested | 自监督预训练（无监督数据） |
| `Widar` | Widardata / Widar | 22 | (22, 20, 20) | csv (csv_folder) | nested | 监督训练 |

**字段说明**：
- `类别数`：监督训练时的分类数；自监督预训练阶段（NTU-Fi_HAR）无监督标签，类别数仅用于元信息
- `loader`：数据加载器类型，决定文件格式与解析方式
- `layout`：目录结构声明（`nested`=类别子目录 / `flat`=扁平结构），loader 按 spec 声明 glob，不探测不 fallback
- `输入形状`：不含 batch 维；runner 自动补 batch 维

**数据集部署注意事项**：
- `UT_HAR_data`：原始发布文件扩展名为 `.csv` 但内容实为 NumPy 二进制（`.npy`）。部署时需将 `UT_HAR/data/*.csv` 和 `UT_HAR/label/*.csv` 重命名为 `.npy`，与 `DatasetSpec.file_format="npy"` 声明一致。preflight 阶段（含 `--dry-run`）会基于 `file_format + layout` 递归 glob 检查声明一致性，扩展名不匹配时直接报错。
- 框架不探测实际文件格式、不 fallback 到其他扩展名——数据集部署是调用者职责，框架按 `DatasetSpec` 声明工作。

**归一化常数**：
- `NTU-Fi_HAR` / `NTU-Fi-HumanID`：mean=42.3199, std=4.9802
- `Widar`：mean=0.0025, std=0.0119
- `UT_HAR_data`：无归一化

**自监督模式约束**：
- `dataset` 必须为 `NTU-Fi_HAR`（Stage 1 无监督预训练数据源）
- Stage 2 微调自动使用 `NTU-Fi-HumanID`（14 类），由 `DatasetSpec.supervised_source` 声明
- `output_features.num_classes` 必须与 `DatasetSpec.supervised_source` 派生的类别数一致（不再硬编码覆盖）

## 支持的模型

| 模型 ID | 范式 | 需 GPU | 显存估计 | 参数量 | 默认学习率 |
|---------|------|--------|----------|--------|------------|
| `MLP` | traditional_ml | 否 | 512MB | 0.3M | 1e-3 |
| `LeNet` | cnn | 否 | 1024MB | 0.5M | 1e-3 |
| `ResNet18` | cnn | 否 | 2048MB | 11.2M | 1e-3 |
| `ResNet50` | cnn | 是 | 4096MB | 23.5M | 1e-3 |
| `ResNet101` | cnn | 是 | 6144MB | 42.5M | 1e-3 |
| `RNN` | rnn | 否 | 1024MB | 1.2M | 1e-3 |
| `GRU` | rnn | 否 | 1024MB | 1.5M | 1e-3 |
| `LSTM` | rnn | 否 | 1536MB | 2.0M | 1e-3 |
| `BiLSTM` | rnn | 否 | 2048MB | 4.0M | 1e-3 |
| `CNN+GRU` | hybrid | 否 | 2048MB | 5.5M | 1e-3 |
| `ViT` | transformer | 是 | 4096MB | 8.0M | 1e-3 |

## 默认 Epochs 表

`(model_id, dataset)` → 默认 epochs：

| 模型 \ 数据集 | UT_HAR_data | NTU-Fi-HumanID | NTU-Fi_HAR | Widar |
|---------------|-------------|----------------|------------|-------|
| MLP | 200 | 50 | 30 | 30 |
| LeNet | 200 | 50 | 30 | 100 |
| ResNet18 | 200 | 50 | 30 | 100 |
| ResNet50 | 200 | 50 | 30 | 100 |
| ResNet101 | 200 | 50 | 30 | 100 |
| RNN | 3000 | 75 | 70 | 500 |
| GRU | 200 | 50 | 30 | 200 |
| LSTM | 200 | 50 | 30 | 200 |
| BiLSTM | 200 | 50 | 30 | 200 |
| CNN+GRU | 200 | 200 | 100 | 200 |
| ViT | 200 | 50 | 30 | 200 |

**注意**：路由级别会 cap 实际 epochs：
- `cpu_minimal`：max_epochs=50
- `cpu_standard`：max_epochs=200
- `gpu_*`：无上限（使用配置值）

## 模型选择建议

### 按资源环境

- **CPU 环境**：选 MLP / LeNet / RNN / GRU / LSTM / BiLSTM / CNN+GRU / ResNet18（均不 require_gpu）
- **入门 GPU（<4GB）**：避免 ResNet50 / ResNet101 / ViT（需 GPU 且显存大）
- **标准 GPU（4-8GB）**：全部模型可用
- **高端 GPU（≥8GB）**：全部模型可用，ViT / ResNet101 可充分训练

### 按精度优先级

- **最高精度**：ResNet101 > ResNet50 > ViT > ResNet18 > CNN+GRU > BiLSTM > LSTM > GRU > RNN > LeNet > MLP
- **最快训练**：MLP > LeNet > RNN > GRU > LSTM > BiLSTM > CNN+GRU > ResNet18 > ViT > ResNet50 > ResNet101
- **最低显存**：MLP > LeNet > RNN > GRU > LSTM > BiLSTM > ResNet18 > CNN+GRU > ViT > ResNet50 > ResNet101

### 按学习模式

- **监督模式**：全部 11 个模型可用，全部 4 个数据集可用
- **自监督模式**：全部 11 个模型可用，但 `dataset` 必须为 `NTU-Fi_HAR`
