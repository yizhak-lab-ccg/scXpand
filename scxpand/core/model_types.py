from dataclasses import dataclass
from typing import Any, Callable

from scxpand.autoencoders.ae_params import AutoEncoderParams
from scxpand.autoencoders.run_ae_train import run_ae_training
from scxpand.hyperopt.param_grids import (
    configure_ae_trial_params,
    configure_lightgbm_trial_params,
    configure_logistic_trial_params,
    configure_mlp_trial_params,
    configure_svm_trial_params,
)
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_training
from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.linear.linear_trainer import run_linear_training
from scxpand.mlp.mlp_params import MLPParam
from scxpand.mlp.run_mlp_train import run_mlp_training
from scxpand.util.classes import ModelType


@dataclass
class ModelSpec:
    config_func: Callable[[Any], dict]
    param_class: type
    runner: Callable[..., Any]
    default_save_dir: str


MODEL_TYPES: dict[ModelType, ModelSpec] = {
    ModelType.MLP: ModelSpec(
        config_func=configure_mlp_trial_params,
        param_class=MLPParam,
        runner=run_mlp_training,
        default_save_dir="results/mlp",
    ),
    ModelType.LOGISTIC: ModelSpec(
        config_func=configure_logistic_trial_params,
        param_class=LinearClassifierParam,
        runner=run_linear_training,
        default_save_dir="results/logistic",
    ),
    ModelType.SVM: ModelSpec(
        config_func=configure_svm_trial_params,
        param_class=LinearClassifierParam,
        runner=run_linear_training,
        default_save_dir="results/svm",
    ),
    ModelType.LIGHTGBM: ModelSpec(
        config_func=configure_lightgbm_trial_params,
        param_class=LightGBMParams,
        runner=run_lightgbm_training,
        default_save_dir="results/lightgbm",
    ),
    ModelType.AUTOENCODER: ModelSpec(
        config_func=configure_ae_trial_params,
        param_class=AutoEncoderParams,
        runner=run_ae_training,
        default_save_dir="results/autoencoder",
    ),
}
