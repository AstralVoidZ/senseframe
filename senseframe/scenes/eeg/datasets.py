"""EEG 场景数据集：BCI Competition IV-2a / PhysioNet MI。

P1.2 落地：stub 实现，验证 SceneContainer.load_dataset 在 EEG 模态下的可移植性，
特别是自监督模式下 DatasetBundle 的 filling_rule 契约（unsupervised + supervised_finetune）。

数据集规格：
- BCI Competition IV-2a：4 类运动想象，22 通道，4s @ 250Hz = 1000 采样点
- PhysioNet MI：2 类运动想象，64 通道，3s @ 160Hz = 480 采样点

实现层次：
1. StubEEGDataset：随机样本 stub，用于无数据文件的契约验证
2. PhysioNetEegmmidbDataset：真实 PhysioNet eegmmidb .edf 加载器（基于 mne）
3. BCICompetitionIV2aDataset：真实 BCI Competition IV 2a .gdf 加载器（基于 mne）

真实加载器在数据文件存在时启用，否则自动回退到 stub。
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split


# ============================================================
# 数据集元数据
# ============================================================
DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "BCI_Competition_IV_2a": {
        "name": "BCI_Competition_IV_2a",
        "num_classes": 4,
        "classes": ["left_hand", "right_hand", "feet", "tongue"],
        "input_shape": (22, 1000),  # 22 通道, 4s @ 250Hz
        "modality": "eeg",
        "channels": 22,
        "sampling_rate": 250,
        "duration_s": 4.0,
        "file_format": "mat",  # BNCI Horizon 2020 镜像格式；也兼容 .gdf
    },
    "PhysioNet_MI": {
        "name": "PhysioNet_MI",
        "num_classes": 2,
        "classes": ["left_hand", "right_hand"],
        "input_shape": (64, 480),  # 64 通道, 3s @ 160Hz
        "modality": "eeg",
        "channels": 64,
        "sampling_rate": 160,
        "duration_s": 3.0,
        "file_format": "edf",
    },
}


# ============================================================
# Stub 数据集
# ============================================================
class StubEEGDataset(Dataset):
    """EEG 数据集 stub（无外部依赖）。

    用于契约验证：在没有真实 EEG 数据文件时，
    返回与真实数据集相同形状的随机样本。
    """
    def __init__(self, dataset_name: str, n_samples: int = 256, seed: int = 42):
        if dataset_name not in DATASET_INFO:
            raise ValueError(
                f"Unknown eeg dataset: {dataset_name}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        info = DATASET_INFO[dataset_name]
        self.info = info
        self.n_samples = n_samples

        rng = np.random.default_rng(seed)
        # EEG 信号：多通道时序，幅度约 10-100 μV
        self.x = torch.from_numpy(
            (rng.standard_normal((n_samples, *info["input_shape"])) * 50).astype(np.float32)
        )
        self.y = torch.from_numpy(
            rng.integers(0, info["num_classes"], (n_samples,)).astype(np.int64)
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ============================================================
# PhysioNet eegmmidb 真实数据集
# ============================================================
# 标准 64 通道顺序（国际 10-10 系统，PhysioNet eegmmidb 通用顺序）
# 该顺序覆盖 PhysioNet eegmmidb 所有 .edf 文件的标准电极位置
PHYSIONET_STANDARD_64_CHANNELS: List[str] = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8",
    "O1", "Oz", "O2", "Iz",
]


def _normalize_channel_name(name: str) -> str:
    """规范化通道名（去除 EDF 格式中的尾部填充点）。

    EDF 格式要求通道名至少 3 字符，不足则用 '.' 填充。
    例如 'C3..' → 'C3', 'Fc5.' → 'Fc5'。
    """
    return name.rstrip(".").capitalize() if name else name


class PhysioNetEegmmidbDataset(Dataset):
    """PhysioNet eegmmidb 真实 EEG 数据集。

    加载 PhysioNet EEG Motor Movement/Imagery Dataset 的 .edf 文件，
    实现 left hand / right hand 二分类运动想象任务。

    Args:
        root: 数据根目录（如 data/eeg/physionet/eegmmidb），内含 S001/S001R01.edf ...
        subjects: 加载的受试者编号列表（如 [1, 2, 3]），None=全部（1-109）
        runs: 加载的 run 列表，默认 (4, 8, 12)（motor imagery left/right hand）
        trial_tmin: trial 起始相对事件时间（秒），默认 0.0
        trial_tmax: trial 结束相对事件时间（秒），默认 3.0
        target_sf: 目标采样率，默认 160（PhysioNet 原生即 160Hz）

    输出：
        x: torch.float32, shape (64, n_times)
        y: torch.int64, 标量（0=left_hand, 1=right_hand）
    """
    def __init__(self, root: Union[str, Path],
                 subjects: Optional[Sequence[int]] = None,
                 runs: Sequence[int] = (4, 8, 12),
                 trial_tmin: float = 0.0,
                 trial_tmax: float = 3.0,
                 target_sf: float = 160.0):
        # 延迟导入 mne，避免在仅使用 stub 时强依赖
        import mne
        from mne.io import read_raw_edf as _read_raw_edf

        self.root = Path(root)
        self.runs = tuple(runs)
        self.trial_tmin = trial_tmin
        self.trial_tmax = trial_tmax
        self.target_sf = float(target_sf)

        if subjects is None:
            # 默认加载所有可用受试者（1-109）
            subjects = list(range(1, 110))
        self.subjects = list(subjects)

        # 标签映射：T1=left_hand=0, T2=right_hand=1（T0=baseline，丢弃）
        # mne.events_from_annotations 返回的 event_id dict 的 key 是 'T1'/'T2'
        # 我们只保留 T1 和 T2
        event_id_map = {"T1": 0, "T2": 1}

        x_list: List[np.ndarray] = []
        y_list: List[int] = []

        for subj in self.subjects:
            subj_dir = self.root / f"S{subj:03d}"
            if not subj_dir.exists():
                continue  # 受试者目录不存在则跳过

            for run in self.runs:
                edf_path = subj_dir / f"S{subj:03d}R{run:02d}.edf"
                if not edf_path.exists():
                    continue

                # 读取单个 run 的 .edf
                raw = _read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
                # 重采样到目标采样率
                if abs(raw.info["sfreq"] - self.target_sf) > 0.5:
                    raw.resample(self.target_sf, verbose="ERROR")

                # 对齐通道到标准 64 通道
                raw = self._align_physionet_channels(raw)

                # 提取事件
                events, event_id_dict = mne.events_from_annotations(
                    raw, verbose="ERROR"
                )
                # 构建 mne.Epochs 的 event_id 参数：只保留 T1/T2
                # event_id_dict 形如 {'T0': 1, 'T1': 2, 'T2': 3}
                # 注意 key 可能是 numpy.str_ 类型，需要统一为 str
                usable_event_id = {}
                for k, v in event_id_dict.items():
                    key = str(k)
                    if key in event_id_map:
                        usable_event_id[key] = v

                if not usable_event_id:
                    # 该 run 没有 T1/T2 事件（如 baseline run R01/R02），跳过
                    continue

                # 切分 epochs（reject=None 保留全部，baseline=None 不做基线校正）
                epochs = mne.Epochs(
                    raw, events, event_id=usable_event_id,
                    tmin=self.trial_tmin, tmax=self.trial_tmax,
                    baseline=None, reject=None, preload=True, verbose="ERROR",
                )
                data = epochs.get_data()  # (n_epochs, n_channels, n_times)

                # 标签：将 mne 内部 event_id 值映射回 0/1
                # epochs.events[:, -1] 是事件 code，对应 event_id_dict 的 value
                ev_code_to_label = {
                    v: event_id_map[str(k)]
                    for k, v in event_id_dict.items() if str(k) in event_id_map
                }
                labels = np.array(
                    [ev_code_to_label[c] for c in epochs.events[:, -1]],
                    dtype=np.int64,
                )

                x_list.append(data.astype(np.float32))
                y_list.extend(labels.tolist())

        if not x_list:
            raise RuntimeError(
                f"PhysioNet eegmmidb 加载失败：在 {self.root} 下未找到任何有效 trial。"
                f" 检查 subjects={self.subjects}, runs={self.runs}，"
                f"以及数据文件是否完整。"
            )

        # 拼接所有 run 的数据
        x_arr = np.concatenate(x_list, axis=0)
        y_arr = np.array(y_list, dtype=np.int64)

        # 验证输出形状：(n_trials, 64, n_times)
        expected_channels = DATASET_INFO["PhysioNet_MI"]["channels"]
        expected_n_times = int(self.target_sf * (self.trial_tmax - self.trial_tmin) + 1)
        # mne.Epochs 默认包含 tmin 到 tmax 的所有采样点（闭区间，n = (tmax-tmin)*sf + 1）
        # 任务要求 (64, 480) = 160 * 3 = 480，但 mne 默认输出 481（含端点）
        # 此处裁剪到目标长度，确保契约一致
        target_n_times = int(self.target_sf * (self.trial_tmax - self.trial_tmin))

        if x_arr.shape[1] != expected_channels:
            raise RuntimeError(
                f"PhysioNet 加载后通道数不匹配: 实际 {x_arr.shape[1]}, "
                f"期望 {expected_channels}"
            )
        if x_arr.shape[2] < target_n_times:
            raise RuntimeError(
                f"PhysioNet 加载后时间点不足: 实际 {x_arr.shape[2]}, "
                f"期望至少 {target_n_times}"
            )
        if x_arr.shape[2] != target_n_times:
            # 裁剪到目标长度（避免 mne 端点包含问题）
            x_arr = x_arr[:, :, :target_n_times]

        self.x = torch.from_numpy(x_arr)
        self.y = torch.from_numpy(y_arr)

    @staticmethod
    def _align_physionet_channels(raw):
        """对齐 PhysioNet .edf 通道到标准 64 通道顺序。

        策略：
        1. 仅保留 EEG 通道（剔除 EOG/ECG 等辅助通道）
        2. 规范化通道名（去除 EDF 尾部填充点）
        3. 若通道名集合与标准 64 通道一致，按标准顺序重排
        4. 若通道名一致但顺序不同，按标准顺序重排
        5. 若通道数 != 64，保留原序但报错由上层处理
        """
        import mne

        # 仅保留 EEG 类型通道
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        if len(eeg_picks) > 0 and len(eeg_picks) < len(raw.ch_names):
            raw.pick(eeg_picks)

        # 规范化通道名
        original_names = list(raw.ch_names)
        normalized_names = [_normalize_channel_name(n) for n in original_names]

        # 重命名通道（去除 EDF 尾部填充点）
        rename_map = {
            orig: norm for orig, norm in zip(original_names, normalized_names)
            if orig != norm
        }
        if rename_map:
            raw.rename_channels(rename_map)

        # 若通道名集合与标准 64 通道一致，按标准顺序重排
        actual_set = set(normalized_names)
        target_set = set(PHYSIONET_STANDARD_64_CHANNELS)
        if actual_set == target_set:
            raw.reorder_channels(PHYSIONET_STANDARD_64_CHANNELS)

        return raw

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ============================================================
# BCI Competition IV 2a 真实数据集
# ============================================================
class BCICompetitionIV2aDataset(Dataset):
    """BCI Competition IV 2a 真实 EEG 数据集。

    加载 BCI Competition IV Dataset 2a 的 .mat 文件（BNCI Horizon 2020 镜像），
    也兼容 .gdf 文件（BBCI 官方原格式）。
    实现 4 类运动想象（left_hand / right_hand / feet / tongue）分类。

    Args:
        root: 数据根目录（如 data/eeg/bci_iv_2a），内含 A01/A01T.mat, A01E.mat ...
              或 A01/A01T.gdf, A01E.gdf ...
        subjects: 加载的受试者列表（如 ['A01', 'A02']），None=全部（A01-A09）
        use_evaluation: 是否包含 E 文件（评估集），默认 True
        trial_tmin: trial 起始相对 trial 起点时间（秒），默认 2.0
                    （BNCI .mat 文件 trial 已包含 cue 前 2s + cue 后 4s = 6s，
                     此参数控制从 trial 起点偏移多少秒开始取数据）
        trial_tmax: trial 结束相对 trial 起点时间（秒），默认 6.0
        target_sf: 目标采样率，默认 250（原生即 250Hz）

    标签映射（BCI IV 2a 标准事件）：
        769 → 0 (left_hand)
        770 → 1 (right_hand)
        771 → 2 (feet)
        772 → 3 (tongue)

    输出：
        x: torch.float32, shape (22, n_times)
        y: torch.int64, 标量（0-3）
    """
    # BCI IV 2a 标准事件 ID（.gdf 格式用）
    # 768 = cue onset (Rejection)，769-772 = 类别 cue
    EVENT_ID_MAP = {769: 0, 770: 1, 771: 2, 772: 3}

    def __init__(self, root: Union[str, Path],
                 subjects: Optional[Sequence[str]] = None,
                 use_evaluation: bool = True,
                 trial_tmin: float = 0.0,
                 trial_tmax: float = 4.0,
                 target_sf: float = 250.0):
        self.root = Path(root)
        self.use_evaluation = use_evaluation
        self.trial_tmin = trial_tmin
        self.trial_tmax = trial_tmax
        self.target_sf = float(target_sf)

        if subjects is None:
            # BNCI Horizon 2020 镜像 .mat 命名是 A01-A09（2 位数）
            subjects = [f"A{i:02d}" for i in range(1, 10)]
        self.subjects = list(subjects)

        file_types = ["T"]
        if use_evaluation:
            file_types.append("E")

        x_list: List[np.ndarray] = []
        y_list: List[int] = []

        for subj in self.subjects:
            subj_dir = self.root / subj
            if not subj_dir.exists():
                continue

            for ftype in file_types:
                # 优先 .mat（BNCI Horizon 2020 镜像），回退 .gdf（BBCI 官方原格式）
                mat_path = subj_dir / f"{subj}{ftype}.mat"
                gdf_path = subj_dir / f"{subj}{ftype}.gdf"

                if mat_path.exists():
                    trials, labels = self._load_mat_file(mat_path)
                elif gdf_path.exists():
                    trials, labels = self._load_gdf_file(gdf_path)
                else:
                    continue

                if trials.shape[0] == 0:
                    continue

                x_list.append(trials.astype(np.float32))
                y_list.extend(labels.tolist())

        if not x_list:
            raise RuntimeError(
                f"BCI Competition IV 2a 加载失败：在 {self.root} 下未找到任何有效 trial。"
                f" 检查 subjects={self.subjects} 及数据文件是否完整。"
            )

        x_arr = np.concatenate(x_list, axis=0)
        y_arr = np.array(y_list, dtype=np.int64)

        # 验证并裁剪输出形状
        expected_channels = DATASET_INFO["BCI_Competition_IV_2a"]["channels"]
        target_n_times = int(self.target_sf * (self.trial_tmax - self.trial_tmin))

        if x_arr.shape[1] != expected_channels:
            raise RuntimeError(
                f"BCI IV 2a 加载后通道数不匹配: 实际 {x_arr.shape[1]}, "
                f"期望 {expected_channels}"
            )
        if x_arr.shape[2] < target_n_times:
            raise RuntimeError(
                f"BCI IV 2a 加载后时间点不足: 实际 {x_arr.shape[2]}, "
                f"期望至少 {target_n_times}"
            )
        if x_arr.shape[2] != target_n_times:
            x_arr = x_arr[:, :, :target_n_times]

        self.x = torch.from_numpy(x_arr)
        self.y = torch.from_numpy(y_arr)

    def _load_mat_file(self, mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """从 BNCI Horizon 2020 .mat 文件加载 trial 数据。

        BNCI Horizon 2020 .mat 文件结构（scipy.io.loadmat, struct_as_record=False）：
            data: (1, N_session) cell array
            data[0, i][0, 0]: mat_struct，字段：
                - X: (n_samples, n_channels) float64，连续 EEG（25 通道 = 22 EEG + 3 EOG）
                - trial: (n_trials, 1) uint8/uint16，cue onset 的 sample index
                - y: (n_trials, 1) uint8，标签 1-4（left/right/feet/tongue）
                - fs: 采样率（250Hz）
                - classes: 4 类名称

        仅有 trial 的 session（motor imagery session）才被处理。
        trial 切片：从 trial[i] + tmin*fs 到 trial[i] + tmax*fs，共 (tmax-tmin)*fs 个 sample。

        Args:
            mat_path: .mat 文件路径

        Returns:
            trials: (n_trials, n_channels, n_times) EEG 数据（仅前 22 个 EEG 通道）
            labels: (n_trials,) 标签数组（0-3）
        """
        from scipy.io import loadmat
        from scipy.io.matlab import mat_struct

        mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=False)
        data_cell = mat["data"]  # (1, N_session) object array

        out_trials: List[np.ndarray] = []
        out_labels: List[int] = []

        target_n_times = int((self.trial_tmax - self.trial_tmin) * self.target_sf)
        n_eeg_channels = DATASET_INFO["BCI_Competition_IV_2a"]["channels"]  # 22

        # 遍历所有 session
        for i in range(data_cell.shape[1]):
            inner = data_cell[0, i][0, 0]
            if not isinstance(inner, mat_struct):
                continue

            trial_idx = np.asarray(inner.trial).reshape(-1)
            y_arr = np.asarray(inner.y).reshape(-1)
            if trial_idx.size == 0 or y_arr.size == 0:
                continue  # baseline session，无 trial

            X = np.asarray(inner.X)  # (n_samples, n_channels)
            fs = float(np.asarray(inner.fs).ravel()[0])

            # 重采样判定
            need_resample = abs(fs - self.target_sf) > 0.5
            if need_resample:
                from scipy.signal import resample_poly
                up = int(self.target_sf)
                down = int(fs)
                X = resample_poly(X, up, down, axis=0)
                # trial sample index 也需要按重采样比例缩放
                trial_idx = (trial_idx * up / down).astype(np.int64)
                fs = self.target_sf

            t_start_offset = int(self.trial_tmin * fs)

            for j in range(len(trial_idx)):
                start = int(trial_idx[j]) + t_start_offset
                end = start + target_n_times
                if end > X.shape[0]:
                    continue  # 超出数据范围，跳过

                # 仅取前 22 个 EEG 通道，转置为 (n_channels, n_times)
                trial_seg = X[start:end, :n_eeg_channels].T  # (22, n_times)

                label = int(y_arr[j]) - 1  # 1-4 → 0-3
                if label not in (0, 1, 2, 3):
                    continue

                out_trials.append(trial_seg)
                out_labels.append(label)

        if not out_trials:
            return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        trials_arr = np.stack(out_trials, axis=0).astype(np.float32)  # (n_trials, 22, n_times)
        labels_arr = np.array(out_labels, dtype=np.int64)
        return trials_arr, labels_arr

    def _load_gdf_file(self, gdf_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """从 .gdf 文件加载 trial 数据（BBCI 官方原格式，保留兼容）。

        使用 mne.io.read_raw_gdf + mne.events_from_annotations + mne.Epochs 切分。

        Args:
            gdf_path: .gdf 文件路径

        Returns:
            trials: (n_trials, n_channels, n_times) EEG 数据
            labels: (n_trials,) 标签数组（0-3）
        """
        import mne
        from mne.io import read_raw_gdf as _read_raw_gdf

        raw = _read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
        if abs(raw.info["sfreq"] - self.target_sf) > 0.5:
            raw.resample(self.target_sf, verbose="ERROR")

        raw = self._align_bci_channels(raw)

        events, event_id_dict = mne.events_from_annotations(raw, verbose="ERROR")
        # event_id_dict 形如 {'769': 769, '770': 770, ...}（key 是字符串）
        # 只保留 769-772
        usable_event_id = {}
        for k, v in event_id_dict.items():
            try:
                code = int(k)
            except (ValueError, TypeError):
                continue
            if code in self.EVENT_ID_MAP:
                usable_event_id[str(code)] = v

        if not usable_event_id:
            return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        epochs = mne.Epochs(
            raw, events, event_id=usable_event_id,
            tmin=self.trial_tmin, tmax=self.trial_tmax,
            baseline=None, reject=None, preload=True, verbose="ERROR",
        )
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)

        # 标签映射
        ev_code_to_label = {}
        for k, v in event_id_dict.items():
            try:
                code = int(k)
            except (ValueError, TypeError):
                continue
            if code in self.EVENT_ID_MAP:
                ev_code_to_label[v] = self.EVENT_ID_MAP[code]

        labels = np.array(
            [ev_code_to_label[c] for c in epochs.events[:, -1]],
            dtype=np.int64,
        )
        return data, labels

    @staticmethod
    def _align_bci_channels(raw):
        """对齐 BCI IV 2a .gdf 通道到 22 个 EEG 通道。

        BCI IV 2a 标准的 22 个 EEG 通道命名（受 .gdf 格式限制可能含尾部空格）。
        策略：仅保留 EEG 类型通道，剔除 EOG 等辅助通道。
        """
        import mne

        eeg_picks = mne.pick_types(raw.info, eeg=True)
        if len(eeg_picks) > 0:
            raw.pick(eeg_picks)
        return raw

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ============================================================
# 数据集加载函数
# ============================================================
def _detect_real_data(dataset_name: str, root: str) -> bool:
    """检测数据根目录下是否存在真实 EEG 数据文件。

    判定规则（按目录结构而非单个文件）：
    - PhysioNet_MI: root 下存在至少一个 S### 子目录
    - BCI_Competition_IV_2a: root 下存在至少一个 A## 子目录
    """
    root_path = Path(root)
    if not root_path.exists():
        return False

    if dataset_name == "PhysioNet_MI":
        # PhysioNet: S001, S002, ... 子目录
        for child in root_path.iterdir():
            if child.is_dir() and child.name.startswith("S"):
                # 验证形如 S001 子目录下有 .edf 文件
                try:
                    if any(child.glob("*.edf")):
                        return True
                except OSError:
                    continue
        return False

    if dataset_name == "BCI_Competition_IV_2a":
        # BCI IV 2a: A01, A02, ... 子目录，优先 .mat（BNCI Horizon 2020 镜像）
        # 兼容 .gdf（BBCI 官方原格式）
        for child in root_path.iterdir():
            if child.is_dir() and child.name.startswith("A"):
                if any(child.glob("*.mat")) or any(child.glob("*.gdf")):
                    return True
        return False

    return False


def _build_real_dataset(dataset_name: str, root: str) -> Dataset:
    """根据数据集名构造真实 EEG 数据集实例。"""
    if dataset_name == "PhysioNet_MI":
        return PhysioNetEegmmidbDataset(root=root)
    if dataset_name == "BCI_Competition_IV_2a":
        return BCICompetitionIV2aDataset(root=root)
    raise ValueError(
        f"Unknown eeg dataset: {dataset_name}. "
        f"Available: {list(DATASET_INFO.keys())}"
    )


def load_eeg_dataset(dataset_name: str, root: str,
                     learning_mode: str = "supervised") -> Dict[str, Any]:
    """加载 EEG 数据集。

    Args:
        dataset_name: "BCI_Competition_IV_2a" 或 "PhysioNet_MI"
        root: 数据根目录
        learning_mode: "supervised" 或 "self_supervised"

    Returns:
        dict: {
            "info": 数据集元数据,
            "train": 监督训练集（supervised）或 None（self_supervised）,
            "unsupervised": 无监督预训练集（self_supervised）或 None,
            "supervised_finetune": 微调集（self_supervised）或 None,
            "val": 验证集,
            "test": 测试集,
        }
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Unknown eeg dataset: {dataset_name}. "
            f"Available: {list(DATASET_INFO.keys())}"
        )
    if learning_mode not in ("supervised", "self_supervised"):
        raise ValueError(
            f"EEG 场景不支持 learning_mode='{learning_mode}'，"
            f"仅支持 'supervised' 或 'self_supervised'"
        )

    info = DATASET_INFO[dataset_name]

    # 真实数据加载：检测数据目录结构是否就绪
    real_data_available = _detect_real_data(dataset_name, root)

    if not real_data_available:
        # Stub 模式：无真实数据文件时使用随机样本
        full_ds = StubEEGDataset(dataset_name, n_samples=512, seed=42)
    else:
        # 真实数据加载
        full_ds = _build_real_dataset(dataset_name, root)

    n = len(full_ds)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    if learning_mode == "self_supervised":
        # 自监督：unsupervised = train_ds 全部（无标签），
        # supervised_finetune = 从 train_ds 划分一小部分用于微调
        unsupervised_ds = train_ds
        n_finetune = max(1, n_train // 4)
        finetune_ds, _ = random_split(
            train_ds, [n_finetune, n_train - n_finetune],
            generator=torch.Generator().manual_seed(43),
        )
        return {
            "info": info,
            "train": None,
            "unsupervised": unsupervised_ds,
            "supervised_finetune": finetune_ds,
            "val": val_ds,
            "test": test_ds,
        }
    # supervised
    return {
        "info": info,
        "train": train_ds,
        "unsupervised": None,
        "supervised_finetune": None,
        "val": val_ds,
        "test": test_ds,
    }
