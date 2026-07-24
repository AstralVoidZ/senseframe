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

## EEG 场景支持的数据集

EEG 场景封装脑电信号分类，支持监督 + 自监督两种学习模式，模态为 `eeg`。无真实数据文件时自动回退到 stub（随机样本，形状与真实数据一致）；真实加载器基于 `mne`。

| 数据集名 | 类别数 | 类别 | 输入形状 | 通道数 | 采样率 | 时长 | 文件格式 | 真实加载器 |
|----------|--------|------|----------|--------|--------|------|----------|------------|
| `BCI_Competition_IV_2a` | 4 | left_hand / right_hand / feet / tongue | (22, 1000) | 22 | 250Hz | 4.0s | mat（BNCI Horizon 2020 镜像，兼容 .gdf） | BCICompetitionIV2aDataset |
| `PhysioNet_MI` | 2 | left_hand / right_hand | (64, 480) | 64 | 160Hz | 3.0s | edf | PhysioNetEegmmidbDataset |

**字段说明**：
- `输入形状`：不含 batch 维，格式为 `(C, T)`（通道 × 时间采样点）
- `文件格式`：`BCI_Competition_IV_2a` 优先加载 `.mat`（BNCI Horizon 2020 镜像），回退到 `.gdf`（BBCI 官方原格式）；`PhysioNet_MI` 加载 `.edf`
- 真实数据加载失败时（数据目录无 `S###` / `A##` 子目录）自动回退到 `StubEEGDataset`（512 样本随机数据）

**归一化**：EEG 信号按通道沿时间轴标准化（mean=0, std=1），类型 `per_channel_standardization`

**学习模式契约**：
- 监督模式：`train` / `val` / `test` 必填，`unsupervised` / `supervised_finetune` 为 None
- 自监督模式：`train` 为 None，`unsupervised`（无监督预训练）+ `supervised_finetune`（微调，从训练集划分 1/4）+ `val` / `test` 必填

**EEG 变换原语**：`normalize_eeg`（按通道标准化，默认 pipeline）、`bandpass_filter`（8-30Hz μ/β 带通）、`csp_features`（共空间模式特征 stub）、`time_freq`（STFT 时频表示）。自监督模式下预训练用强增强（train_transform），微调用弱增强（supervised_transform）。

## EEG 支持的模型

所有模型期望输入形状 `(B, C, T)`，监督模型输出 `(B, num_classes)`。

| 模型 ID | 范式 | 学习模式 | 需 GPU | 显存估计 | 参数量 | 默认学习率 |
|---------|------|----------|--------|----------|--------|------------|
| `EEGNet` | eeg_cnn | supervised | 否 | 512MB | 0.05M | 1e-3 |
| `DeepConvNet` | eeg_deep_cnn | supervised | 否 | 1024MB | 0.5M | 1e-3 |
| `TransformerEEG` | eeg_transformer | supervised | 是 | 2048MB | 1.0M | 5e-4 |
| `EEGLowEncoder` | eeg_self_supervised | self_supervised | 否 | 512MB | 0.1M | 1e-3 |

**模型说明**：
- `EEGNet`：紧凑型卷积分类器（Lawhern 2018），Temporal Conv + Depthwise Conv + Separable Conv
- `DeepConvNet`：4 层深度卷积分类器（Schirrmeister 2017）
- `TransformerEEG`：通道 embedding + TransformerEncoder + CLS token，patch_size=32，d_model=128
- `EEGLowEncoder`：基于 EEGNet 主体（去分类头）的自监督编码器，输出 `(B, feature_dim=128)`

**默认训练配置**：监督模式 epochs=100，自监督模式 epochs=50；learning_rate=1e-3；batch_size=64

## Radio 场景支持的数据集

Radio 场景封装无线电信号调制识别，仅支持监督学习模式，模态为 `iq`（IQ 数据）。真实数据需从 https://www.deepsig.ai/ 下载。

| 数据集名 | 类别数 | 输入形状 | 调制方式数 | SNR 范围 | 文件格式 | 真实加载器 |
|----------|--------|----------|------------|----------|----------|------------|
| `RadioML2016A` | 11 | (2, 128) | 11 | -20 ~ 18 dB（步长 2） | pkl | 未实现（仅 stub） |
| `RadioML2018` | 24 | (2, 1024) | 24 | -20 ~ 30 dB（步长 2） | h5 | RadioML2018Dataset |

**RadioML2016A 类别**：8PSK / AM-DSB / AM-SSB / BPSK / CPFSK / GFSK / PAM4 / QAM16 / QAM64 / QPSK / WBFM

**RadioML2018 类别**：16APSK / 32APSK / 64APSK / 128APSK / 16QAM / 32QAM / 64QAM / 128QAM / 256QAM / AM-SSB-WC / AM-SSB-SC / AM-DSB-WC / AM-DSB-SC / FM / GMSK / OQPSK / BPSK / QPSK / 8PSK / 16PSK / 32PSK / CPFSK / BFPM / PAM4

**字段说明**：
- `输入形状`：`(2, L)` — I/Q 双通道，每通道 L 个时间采样点
- `RadioML2018Dataset`：使用 h5py 流式读取（避免一次性载入 19GB 数据），Y/Z 载入内存做过滤；支持 `max_samples` / `snr_filter` / `modulation_filter` 参数
- `RadioML2016A`：真实 pickle 加载器尚未实现，仅 stub 模式可用

**归一化**：IQ 信号按样本沿时间轴标准化（mean=0, std=1），类型 `per_sample_standardization`

## Radio 支持的模型

所有模型期望输入形状 `(B, C, L)`，输出 `(B, num_classes)`。默认 `in_channels=2`（IQ 双通道）。

| 模型 ID | 范式 | 需 GPU | 显存估计 | 参数量 | 默认学习率 |
|---------|------|--------|----------|--------|------------|
| `CNN1D` | cnn1d | 否 | 512MB | 0.5M | 1e-3 |
| `ResNet1D` | resnet1d | 否 | 1024MB | 1.5M | 1e-3 |
| `Transformer1D` | transformer1d | 是 | 2048MB | 3.0M | 5e-4 |

**模型说明**：
- `CNN1D`：3 层 1D 卷积（kernel 7/5/3）+ BN + ReLU + MaxPool + GlobalAvgPool + Linear，hidden_channels=64
- `ResNet1D`：Stem Conv + 残差块 × 3（`_ResBlock1D`）+ GlobalAvgPool + Linear，hidden_channels=64
- `Transformer1D`：Patch embedding（patch_size=8）+ CLS token + 位置编码 + TransformerEncoder（d_model=128, n_heads=4, num_layers=4）

**默认训练配置**：epochs=50；learning_rate=1e-3；batch_size=128

**Radio 变换原语**：`iq_to_complex`（(2,L)→(1,L) 复数 magnitude）、`iq_to_spectrogram`（(2,L)→(2,F,T) STFT 时频图）、`normalize_iq`（按 IQ 通道标准化）。默认 pipeline：`["iq_to_complex", "normalize_iq"]`

## Detection 场景

Detection 场景是目标检测 stub 容器（Phase 12.5），用于验证非分类任务（DETECTION task_type）的端到端接入能力，模态为 `image`。仅含 stub 实现，不含真实检测算法（无 anchor / 真实 NMS 训练 / box regression）；真实检测任务应继承 `DetectionContainer` 并补全实现。

**支持的数据集**（均为 stub 随机样本）：

| 数据集名 | 类别数 | 输入形状 | 模态 |
|----------|--------|----------|------|
| `dummy_box` | 3 | (3, 64, 64) | image |
| `tiny_coco` | 80 | (3, 224, 224) | image |

**支持的模型**：
- `SimpleDetector`：最小检测器 stub，输入 `(B, C, H, W)`，输出 `{"bboxes": (B, 4), "logits": (B, num_classes)}`，仅 AdaptiveAvgPool + Linear，不含真实检测逻辑

**任务配置**：
- `task_type`：`detection`
- `loss`：`bce_with_logits`
- `metrics`：`["map"]`
- `output_activation`：`sigmoid`
- `bbox_format`：`cxcywh`（支持 cxcywh / xyxy / xywh，postprocess 自动转换）
- `nms_threshold`：0.5，`score_threshold`：0.05

**NMS 后处理**：`DetectionContainer.postprocess` 实现纯 torch NMS（无 torchvision 依赖），流程为 bbox 格式转换 → sigmoid 激活 → score 过滤 → NMS 去重

**默认训练配置**：epochs=50；batch_size=8；learning_rate=1e-3

**变换原语**（catalog.py + transforms.py）：
- 图像增强：`hsv_jitter`、`cutout`、`mixup`（batch 级）、`random_erasing`
- bbox 处理：`bbox_clip`（裁剪到图像边界）、`bbox_flip`（水平/垂直翻转）

## Generic 场景

Generic 场景是通用表格场景容器，支持任意 CSV / NumPy 数据集，模态为 `tabular`，`is_dynamic_dataset=True`（数据集列表由数据目录动态决定，非静态枚举）。不依赖 SenseFi 基准库，纯 PyTorch 实现，仅支持监督学习模式。

**数据集加载**（自动检测格式）：
- CSV 模式：`{root}/{dataset_name}.csv`，最后一列为标签（或通过 `label_column` 指定），按 `test_ratio` 随机分割 train/test
- NumPy 模式：`{root}/{dataset_name}/{x_train,y_train,x_test,y_test}.npy`，直接加载预分割数据

**`list_datasets(root)`**：扫描数据根目录，返回所有 `.csv` 文件名 + 含 `x_train.npy` 的子目录名

**支持的模型**：
- `GenericMLP`：`input → [Linear+ReLU+Dropout] × n_layers → Linear(output)`，默认 `hidden_dims=[128, 64]`，`dropout=0.1`，支持 `(batch, features)` 和 `(batch, 1, features)` 两种输入

**默认训练配置**：epochs=50；learning_rate=1e-3；batch_size=32；estimated_vram=256MB；estimated_params=0.1M；requires_gpu=False

**变换技术目录**（catalog.py，`applicable=["*"]` 适用于所有数据集）：
- 特征工程：`rolling_stats`（滑动窗口统计）、`fft_features`（FFT 频域特征）、`wavelet_decomp`（haar 小波分解）、`seasonal_decompose`（季节性分解去趋势）
- 数据增强：`jitter`（时域抖动）、`scaling`（幅度缩放）、`window_warp`（窗口切片插值）、`magnitude_warp`（幅度扭曲）

## Custom 场景

Custom 场景是基于 manifest 的零代码自定义数据集接入容器，模态为 `tabular`，`is_dynamic_dataset=True`。用户编写 manifest JSON/YAML 即可用 senseframe 训练自定义数据集，无需继承 `SceneContainer`、无需改源码。仅支持监督学习模式。

**使用方式**：在 YAML 配置中指定 `scene.params.manifest_path` 指向 manifest 文件，manifest 描述数据集元信息（name / num_classes / input_shape / file_format / normalization / samples / mat_key / label_map 等）。

**数据集与模型**：
- `supported_datasets`：动态返回，由 manifest.name 决定（`meta()` 返回空列表，由 runner 校验时二次确认）
- `supported_models`：`GenericMLP`（内置）+ 通过 `register_model` / `bind_model_factory` 注册的自定义模型

**模型构建**：`GenericMLP` 将 manifest 的 `input_shape` 展平为 `input_dim`（`np.prod(input_shape)`）；多维输入自动添加 flatten 变换以适配 MLP。已注册模型通过 `resolve_factory` 查找（场景级 → 全局级 → 通配符 `*`）。

**自动注册**：`load_dataset` 时将 manifest 数据集幂等注册到全局注册表，使 `get_model()` 等查询可用。

**归一化**：在 `ManifestDataset.__getitem__` 中完成（manifest 声明 `normalization="auto"` 时自动计算），容器 `normalize` 直接透传。

**默认训练配置**：epochs=50；learning_rate=1e-3；batch_size=32；estimated_vram=256MB；estimated_params=0.1M；requires_gpu=False

**技术目录**：custom 场景无静态技术目录，`get_catalog()` 返回 None（由 manifest 决定）。
