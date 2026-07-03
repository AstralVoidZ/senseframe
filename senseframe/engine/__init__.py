"""
senseframe.engine：通用训练引擎包。

引擎通过 senseframe.scenes.SceneContainer 与具体场景交互，
自身只负责通用训练流程编排，不感知领域细节。
"""

from .config import (
    ExperimentConfig,
    HPOConfig,
    InputFeature,
    OutputFeature,
    SceneConfig,
    TrainerConfig,
)
from .datamodule import GenericDataModule
from .hpo import (
    OPTUNA_AVAILABLE,
    HPOOutput,
    TrialResult,
    apply_params,
    extract_metric,
    run_hpo,
    sample_params,
)
from .module import GenericLightningModule
from .self_supervised import EntLoss, SelfSupervisedModule, gaussian_noise

__all__ = [
    "ExperimentConfig",
    "SceneConfig",
    "TrainerConfig",
    "HPOConfig",
    "InputFeature",
    "OutputFeature",
    "GenericLightningModule",
    "GenericDataModule",
    "SelfSupervisedModule",
    "EntLoss",
    "gaussian_noise",
    "run_hpo",
    "HPOOutput",
    "TrialResult",
    "apply_params",
    "extract_metric",
    "sample_params",
    "OPTUNA_AVAILABLE",
]
