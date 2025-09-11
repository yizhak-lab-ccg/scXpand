import optuna

from scxpand.lightgbm.lightgbm_params import BoostingType, MetricType, ObjectiveType
from scxpand.util.classes import LRSchedulerType, SamplerType


# Fixed choices for Optuna categorical parameter, stored as strings for persistence
AUX_CATEGORICAL_OPTIONS = (
    "none",
    "tissue_type",
    "imputed_labels",
    "tissue_type,imputed_labels",
)


def create_optimized_lr_scheduler_config(
    trial: optuna.Trial, lr_scheduler_type: str, n_epochs: int, prefix: str
) -> dict:
    """Create optimized learning rate scheduler configuration with hyperparameter search.

    Args:
        trial: Optuna trial object
        lr_scheduler_type: Type of learning rate scheduler
        n_epochs: Number of training epochs
        prefix: Prefix for parameter names (e.g., ``"mlp_"``, ``"ae_"``)

    Returns:
        Dictionary with scheduler-specific configuration
    """
    lr_scheduler_config = {"type": lr_scheduler_type}

    if lr_scheduler_type == LRSchedulerType.REDUCE_LR_ON_PLATEAU.value:
        lr_scheduler_config.update(
            {
                "factor": trial.suggest_float(f"{prefix}lr_factor", 0.1, 0.7, log=True),
                "patience": trial.suggest_int(f"{prefix}lr_patience", 3, 15),
                "min_lr": trial.suggest_float(f"{prefix}lr_min_lr", 1e-7, 1e-5, log=True),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.ONE_CYCLE_LR.value:
        lr_scheduler_config.update(
            {
                "warmup_ratio": trial.suggest_float(f"{prefix}lr_warmup_ratio", 0.05, 0.3),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.STEP_LR.value:
        lr_scheduler_config.update(
            {
                "step_size": trial.suggest_int(f"{prefix}lr_step_size", 5, 20),
                "gamma": trial.suggest_float(f"{prefix}lr_gamma", 0.1, 0.7, log=True),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.COSINE_ANNEALING_LR.value:
        lr_scheduler_config.update(
            {
                "T_max": trial.suggest_int(f"{prefix}lr_t_max", n_epochs, n_epochs * 2),
                "eta_min": trial.suggest_float(f"{prefix}lr_eta_min", 1e-7, 1e-5, log=True),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.CONSTANT_LR.value:
        # ConstantLR maintains the same learning rate throughout training
        lr_scheduler_config.update({})
    # NoScheduler needs no additional params

    return lr_scheduler_config


def configure_mlp_trial_params(trial: optuna.Trial) -> dict:
    """Suggest MLP parameters (prefixed for Optuna) and return them unprefixed."""
    use_log_transform = trial.suggest_categorical("mlp_use_log_transform", (True, False))
    n_epochs = 30
    init_learning_rate = trial.suggest_float("mlp_init_learning_rate", 1e-5, 1e-3, log=True)  # More conservative
    train_batch_size = trial.suggest_categorical("mlp_train_batch_size", (2048, 4096))
    dropout_rate = trial.suggest_float("mlp_dropout_rate", 0.1, 0.3)
    num_layers = trial.suggest_int("mlp_num_layers", 2, 5)
    weight_decay = trial.suggest_float("mlp_weight_decay", 1e-5, 1e-2, log=True)
    mask_rate = trial.suggest_float("mlp_mask_rate", 0.05, 0.3)
    noise_std = trial.suggest_float("mlp_noise_std", 1e-5, 1e-3, log=True)
    positives_weight = trial.suggest_float("mlp_positives_weight", 0.1, 10.0)
    use_soft_loss = trial.suggest_categorical("mlp_use_soft_loss", (True, False))
    sampler_type_str = trial.suggest_categorical(
        "mlp_sampler_type",
        (SamplerType.BALANCED_LABELS.value, SamplerType.BALANCED_TYPES.value, SamplerType.RANDOM.value),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Set inference batch size equal to train batch size
    inference_batch_size = train_batch_size

    cat_loss_weight = trial.suggest_float("mlp_cat_loss_weight", 0.1, 10.0, log=True)

    # Learning rate scheduler parameters
    lr_scheduler_type_str = trial.suggest_categorical(
        "mlp_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            LRSchedulerType.ONE_CYCLE_LR.value,
            LRSchedulerType.STEP_LR.value,
            LRSchedulerType.COSINE_ANNEALING_LR.value,
            LRSchedulerType.CONSTANT_LR.value,
            LRSchedulerType.NO_SCHEDULER.value,
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)

    # Adam optimizer parameters
    adam_beta1 = trial.suggest_float("mlp_adam_beta1", 0.8, 0.95)
    adam_beta2 = trial.suggest_float("mlp_adam_beta2", 0.95, 0.999)
    adam_betas = (adam_beta1, adam_beta2)

    lr_scheduler_config = create_optimized_lr_scheduler_config(trial, lr_scheduler_type, n_epochs, "mlp_")

    # Auxiliary categorical types for classification - use fixed choices directly
    aux_categorical_types_str = trial.suggest_categorical(
        "mlp_aux_categorical_types",
        AUX_CATEGORICAL_OPTIONS,
    )
    if aux_categorical_types_str == "none":
        aux_categorical_types = ()
    else:
        aux_categorical_types = tuple(aux_categorical_types_str.split(","))

    layer_units = []
    for i in range(num_layers):
        layer_name = f"mlp_layer_{i}_units"
        units = trial.suggest_categorical(layer_name, (512, 1024, 2048, 4096))
        layer_units.append(units)
    soft_loss_beta = 1.0
    soft_loss_start_epoch = -1
    if use_soft_loss:
        soft_loss_beta = trial.suggest_float("mlp_soft_loss_beta", 0.5, 10.0)
        soft_loss_start_epoch = trial.suggest_int("mlp_soft_loss_start_epoch", 0, n_epochs - 1)

    params = {
        "use_log_transform": use_log_transform,
        "n_epochs": n_epochs,
        "init_learning_rate": init_learning_rate,
        "train_batch_size": train_batch_size,
        "inference_batch_size": inference_batch_size,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "mask_rate": mask_rate,
        "noise_std": noise_std,
        "positives_weight": positives_weight,
        "soft_loss_beta": soft_loss_beta,
        "soft_loss_start_epoch": soft_loss_start_epoch,
        "layer_units": tuple(layer_units),
        "aux_categorical_types": aux_categorical_types,
        "sampler_type": sampler_type,
        "cat_loss_weight": cat_loss_weight,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_config": lr_scheduler_config,
        "adam_betas": adam_betas,
    }
    return params


def configure_ae_trial_params(trial: optuna.Trial) -> dict:
    """Suggest autoencoder parameters for Optuna trials.

    Optimizes both:
    - model_type: autoencoder architecture ('standard' or 'fork')
    - loss_type: loss function ('zinb', 'nb', 'mse')
    """
    use_log_transform = trial.suggest_categorical("ae_use_log_transform", (True, False))
    n_epochs = 30
    init_learning_rate = trial.suggest_float("ae_init_learning_rate", 1e-6, 1e-3, log=True)  # Ultra-conservative range
    train_batch_size = trial.suggest_categorical("ae_train_batch_size", (2048, 4096))
    sampler_type_str = trial.suggest_categorical(
        "ae_sampler_type",
        (SamplerType.BALANCED_LABELS.value, SamplerType.BALANCED_TYPES.value, SamplerType.RANDOM.value),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Set inference batch size equal to train batch size
    inference_batch_size = train_batch_size

    # Learning rate scheduler parameters
    lr_scheduler_type_str = trial.suggest_categorical(
        "ae_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            LRSchedulerType.ONE_CYCLE_LR.value,
            LRSchedulerType.STEP_LR.value,
            LRSchedulerType.COSINE_ANNEALING_LR.value,
            LRSchedulerType.CONSTANT_LR.value,
            LRSchedulerType.NO_SCHEDULER.value,
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)

    # Adam optimizer parameters
    adam_beta1 = trial.suggest_float("ae_adam_beta1", 0.8, 0.95)
    adam_beta2 = trial.suggest_float("ae_adam_beta2", 0.95, 0.999)
    adam_betas = (adam_beta1, adam_beta2)

    # LR scheduler config (type-specific parameters with optimization)
    lr_scheduler_config = create_optimized_lr_scheduler_config(trial, lr_scheduler_type, n_epochs, "ae_")

    # Architecture and loss type params
    model_type = trial.suggest_categorical("ae_model_type", ("standard", "fork"))  # autoencoder type
    loss_type = trial.suggest_categorical("ae_loss_type", ("mse", "nb", "zinb"))  # MSE first for stability
    latent_dim = trial.suggest_categorical("ae_latent_dim", (16, 32, 64, 128))

    # Loss weights
    recon_loss_weight = trial.suggest_float("ae_recon_loss_weight", 0.1, 10.0, log=True)
    cls_loss_weight = trial.suggest_float("ae_cls_loss_weight", 0.1, 10.0, log=True)
    cat_loss_weight = trial.suggest_float("ae_cat_loss_weight", 0.1, 10.0, log=True)

    # Data augmentation params
    mask_rate = trial.suggest_float("ae_mask_rate", 0.05, 0.5)
    noise_std = trial.suggest_float("ae_noise_std", 1e-5, 1e-3, log=True)
    positives_weight = trial.suggest_float("ae_positives_weight", 0.1, 10.0)

    # Encoder hidden dims
    encoder_num_layers = trial.suggest_int("ae_encoder_num_layers", 1, 3)
    encoder_hidden_dims = []
    for i in range(encoder_num_layers):
        layer_name = f"ae_encoder_layer_{i}_units"
        units = trial.suggest_categorical(layer_name, (32, 64, 128, 256, 512, 1024))
        encoder_hidden_dims.append(units)

    # Decoder hidden dims
    decoder_num_layers = trial.suggest_int("ae_decoder_num_layers", 1, 3)
    decoder_hidden_dims = []
    for i in range(decoder_num_layers):
        layer_name = f"ae_decoder_layer_{i}_units"
        units = trial.suggest_categorical(layer_name, (32, 64, 128, 256, 512, 1024))
        decoder_hidden_dims.append(units)

    # Classifier hidden dims
    classifier_num_layers = trial.suggest_int("ae_classifier_num_layers", 1, 2)
    classifier_hidden_dims = []
    for i in range(classifier_num_layers):
        layer_name = f"ae_classifier_layer_{i}_units"
        units = trial.suggest_categorical(layer_name, (16, 32, 64))
        classifier_hidden_dims.append(units)

    # Regularization params
    dropout_rate = trial.suggest_float("ae_dropout_rate", 0.1, 0.5)
    l1_lambda = trial.suggest_float("ae_l1_lambda", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("ae_weight_decay", 1e-6, 1e-3, log=True)

    # Auxiliary categorical types for classification
    aux_categorical_types_str = trial.suggest_categorical(
        "ae_aux_categorical_types",
        AUX_CATEGORICAL_OPTIONS,
    )
    if aux_categorical_types_str == "none":
        aux_categorical_types = ()
    else:
        aux_categorical_types = tuple(aux_categorical_types_str.split(","))

    # Soft loss parameters
    use_soft_loss = trial.suggest_categorical("ae_use_soft_loss", (True, False))
    soft_loss_beta = 1.0
    soft_loss_start_epoch = None
    if use_soft_loss:
        soft_loss_beta = trial.suggest_float("ae_soft_loss_beta", 0.5, 10.0)
        soft_loss_start_epoch = trial.suggest_int("ae_soft_loss_start_epoch", 0, n_epochs - 1)

    params = {
        "use_log_transform": use_log_transform,
        "n_epochs": n_epochs,
        "init_learning_rate": init_learning_rate,
        "train_batch_size": train_batch_size,
        "inference_batch_size": inference_batch_size,
        "sampler_type": sampler_type,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_config": lr_scheduler_config,
        "adam_betas": adam_betas,
        "model_type": model_type,  # autoencoder type
        "loss_type": loss_type,  # loss function type
        "latent_dim": latent_dim,
        "encoder_hidden_dims": tuple(encoder_hidden_dims),
        "decoder_hidden_dims": tuple(decoder_hidden_dims),
        "classifier_hidden_dims": tuple(classifier_hidden_dims),
        "dropout_rate": dropout_rate,
        "l1_lambda": l1_lambda,
        "weight_decay": weight_decay,
        "mask_rate": mask_rate,
        "noise_std": noise_std,
        "positives_weight": positives_weight,
        "recon_loss_weight": recon_loss_weight,
        "cls_loss_weight": cls_loss_weight,
        "cat_loss_weight": cat_loss_weight,
        "aux_categorical_types": aux_categorical_types,
        "soft_loss_beta": soft_loss_beta,
        "soft_loss_start_epoch": soft_loss_start_epoch,
    }
    if loss_type == "zinb":
        params["ridge_lambda"] = trial.suggest_float("ae_ridge_lambda", 1e-4, 1e-1, log=True)

    return params


def configure_logistic_trial_params(trial: optuna.Trial) -> dict:
    """Suggest Logistic Regression parameters (prefixed) and return them unprefixed."""
    use_log_transform = trial.suggest_categorical("logistic_use_log_transform", (True, False))
    # L2 and elasticnet are most stable for logistic regression with SGD
    penalty = trial.suggest_categorical("logistic_penalty", ("l2", "elasticnet"))
    alpha = trial.suggest_float("logistic_alpha", 1e-6, 1e-1, log=True)  # More conservative range
    n_epochs = 30
    tol = trial.suggest_float("logistic_tol", 1e-6, 1e-3, log=True)  # Tighter tolerance range
    class_weight = trial.suggest_categorical("logistic_class_weight", ("balanced", None))
    batch_size = trial.suggest_categorical("logistic_batch_size", (512, 1024, 2048))
    sampler_type_str = trial.suggest_categorical(
        "logistic_sampler_type",
        (SamplerType.BALANCED_LABELS.value, SamplerType.BALANCED_TYPES.value, SamplerType.RANDOM.value),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Learning rate and optimizer parameters - aligned with scikit-learn defaults
    init_learning_rate = trial.suggest_float("logistic_init_learning_rate", 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_categorical(
        "logistic_learning_rate", ("optimal", "constant", "invscaling", "adaptive")
    )
    # eta0 is only used when learning_rate is not 'optimal'
    eta0 = trial.suggest_float("logistic_eta0", 1e-3, 1.0, log=True)  # Better range for initial learning rate
    power_t = trial.suggest_float("logistic_power_t", 0.25, 0.75)  # More focused around default 0.5

    # Learning rate scheduler parameters
    lr_scheduler_type_str = trial.suggest_categorical(
        "logistic_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            LRSchedulerType.ONE_CYCLE_LR.value,
            LRSchedulerType.STEP_LR.value,
            LRSchedulerType.COSINE_ANNEALING_LR.value,
            LRSchedulerType.CONSTANT_LR.value,
            LRSchedulerType.NO_SCHEDULER.value,
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)
    lr_scheduler_config = create_optimized_lr_scheduler_config(trial, lr_scheduler_type, n_epochs, "logistic_")

    # SGD-specific parameters - aligned with scikit-learn recommendations
    warm_start = trial.suggest_categorical("logistic_warm_start", (True, False))
    average = trial.suggest_categorical("logistic_average", (True, False))
    # Note: n_iter_no_change and validation_fraction are only used with early_stopping=True
    # but we include them for potential future use or different training strategies
    n_iter_no_change = trial.suggest_int("logistic_n_iter_no_change", 3, 10)
    validation_fraction = trial.suggest_float("logistic_validation_fraction", 0.05, 0.2)
    # Note: early_stopping is always False when using partial_fit() - set in LinearClassifierParam

    # Additional regularization parameters
    fit_intercept = trial.suggest_categorical("logistic_fit_intercept", (True, False))
    shuffle = trial.suggest_categorical("logistic_shuffle", (True, False))

    # Data augmentation parameters
    mask_rate = trial.suggest_float("logistic_mask_rate", 0.0, 0.3)
    noise_std = trial.suggest_float("logistic_noise_std", 1e-5, 1e-3, log=True)
    soft_loss_beta = trial.suggest_float("logistic_soft_loss_beta", 0.5, 10.0)

    # l1_ratio is only used with elasticnet penalty
    l1_ratio = 0.0
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("logistic_l1_ratio", 0.1, 0.9)  # Avoid extremes 0.0 and 1.0

    params = {
        "model_type": "logistic",
        "use_log_transform": use_log_transform,
        "alpha": alpha,
        "penalty": penalty,
        "n_epochs": n_epochs,
        "tol": tol,
        "class_weight": class_weight,
        "l1_ratio": l1_ratio,
        "batch_size": batch_size,
        "sampler_type": sampler_type,
        "mask_rate": mask_rate,
        "noise_std": noise_std,
        "soft_loss_beta": soft_loss_beta,
        "init_learning_rate": init_learning_rate,
        "learning_rate": learning_rate,
        "eta0": eta0,
        "power_t": power_t,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_config": lr_scheduler_config,
        "warm_start": warm_start,
        "average": average,
        "n_iter_no_change": n_iter_no_change,
        "validation_fraction": validation_fraction,
        "fit_intercept": fit_intercept,
        "shuffle": shuffle,
    }
    return params


def configure_svm_trial_params(trial: optuna.Trial) -> dict:
    """Suggest SVM parameters (prefixed) and return them unprefixed."""
    use_log_transform = trial.suggest_categorical("svm_use_log_transform", (True, False))
    # For SVM (hinge loss), L1 penalty is not supported - only L2 and elasticnet
    penalty = trial.suggest_categorical("svm_penalty", ("l2", "elasticnet"))
    alpha = trial.suggest_float("svm_alpha", 1e-6, 1e-1, log=True)  # More conservative range for SVM
    n_epochs = 30
    tol = trial.suggest_float("svm_tol", 1e-6, 1e-3, log=True)  # Tighter tolerance range
    class_weight = trial.suggest_categorical("svm_class_weight", ("balanced", None))
    batch_size = trial.suggest_categorical("svm_batch_size", (512, 1024, 2048))
    sampler_type_str = trial.suggest_categorical(
        "svm_sampler_type",
        (SamplerType.BALANCED_LABELS.value, SamplerType.BALANCED_TYPES.value, SamplerType.RANDOM.value),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Learning rate and optimizer parameters - aligned with scikit-learn defaults
    init_learning_rate = trial.suggest_float("svm_init_learning_rate", 1e-5, 1e-2, log=True)
    learning_rate = trial.suggest_categorical("svm_learning_rate", ("optimal", "constant", "invscaling", "adaptive"))
    # eta0 is only used when learning_rate is not 'optimal'
    eta0 = trial.suggest_float("svm_eta0", 1e-3, 1.0, log=True)  # Better range for initial learning rate
    power_t = trial.suggest_float("svm_power_t", 0.25, 0.75)  # More focused around default 0.5

    # Learning rate scheduler parameters
    lr_scheduler_type_str = trial.suggest_categorical(
        "svm_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            LRSchedulerType.ONE_CYCLE_LR.value,
            LRSchedulerType.STEP_LR.value,
            LRSchedulerType.COSINE_ANNEALING_LR.value,
            LRSchedulerType.CONSTANT_LR.value,
            LRSchedulerType.NO_SCHEDULER.value,
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)
    lr_scheduler_config = create_optimized_lr_scheduler_config(trial, lr_scheduler_type, n_epochs, "svm_")

    # SGD-specific parameters - aligned with scikit-learn recommendations
    warm_start = trial.suggest_categorical("svm_warm_start", (True, False))
    average = trial.suggest_categorical("svm_average", (True, False))
    n_iter_no_change = trial.suggest_int("svm_n_iter_no_change", 3, 10)
    validation_fraction = trial.suggest_float("svm_validation_fraction", 0.05, 0.2)
    # Note: early_stopping is always False when using partial_fit() - set in LinearClassifierParam

    # Additional regularization parameters
    fit_intercept = trial.suggest_categorical("svm_fit_intercept", (True, False))
    shuffle = trial.suggest_categorical("svm_shuffle", (True, False))

    # Data augmentation parameters
    mask_rate = trial.suggest_float("svm_mask_rate", 0.0, 0.3)
    noise_std = trial.suggest_float("svm_noise_std", 1e-5, 1e-3, log=True)
    soft_loss_beta = trial.suggest_float("svm_soft_loss_beta", 0.5, 10.0)

    # l1_ratio is only used with elasticnet penalty
    l1_ratio = 0.0
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("svm_l1_ratio", 0.1, 0.9)  # Avoid extremes 0.0 and 1.0

    params = {
        "use_log_transform": use_log_transform,
        "model_type": "svm",
        "alpha": alpha,
        "penalty": penalty,
        "n_epochs": n_epochs,
        "tol": tol,
        "class_weight": class_weight,
        "l1_ratio": l1_ratio,
        "batch_size": batch_size,
        "sampler_type": sampler_type,
        "mask_rate": mask_rate,
        "noise_std": noise_std,
        "soft_loss_beta": soft_loss_beta,
        "init_learning_rate": init_learning_rate,
        "learning_rate": learning_rate,
        "eta0": eta0,
        "power_t": power_t,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_config": lr_scheduler_config,
        "warm_start": warm_start,
        "average": average,
        "n_iter_no_change": n_iter_no_change,
        "validation_fraction": validation_fraction,
        "fit_intercept": fit_intercept,
        "shuffle": shuffle,
    }
    return params


def configure_lightgbm_trial_params(trial: optuna.Trial) -> dict:
    """Suggest LightGBM parameters (prefixed) and return them unprefixed."""
    use_log_transform = trial.suggest_categorical("lgbm_use_log_transform", (True, False))
    use_zscore_norm = trial.suggest_categorical("lgbm_use_zscore_norm", (True, False))
    num_leaves = trial.suggest_int("lgbm_num_leaves", 15, 127)  # Original wider range for hyperopt to explore
    max_depth = trial.suggest_int("lgbm_max_depth", 3, 12)  # Original wider range
    learning_rate = trial.suggest_float("lgbm_learning_rate", 1e-3, 0.1, log=True)  # Original range
    n_estimators = trial.suggest_int("lgbm_n_estimators", 50, 300)  # Original range
    min_child_samples = trial.suggest_int("lgbm_min_child_samples", 5, 100)  # Original range
    reg_alpha = trial.suggest_float("lgbm_reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float("lgbm_reg_lambda", 1e-8, 10.0, log=True)
    class_weight = trial.suggest_categorical("lgbm_class_weight", ("balanced", None))
    # Use feature_fraction and bagging_fraction to simulate dropout/masking as in neural nets.
    # For example, feature_fraction=0.7 means 30% of features are randomly dropped per tree,
    # which is analogous to a 30% dropout rate in neural networks. The range 0.7-.95
    # corresponds to 5-30% masking, matching typical neural net dropout rates.
    feature_fraction = trial.suggest_float("lgbm_feature_fraction", 0.7, 0.95)
    bagging_fraction = trial.suggest_float("lgbm_bagging_fraction", 0.7, 0.95)
    # Original wider range - let hyperopt find the best values
    min_split_gain = trial.suggest_float("lgbm_min_split_gain", 0.0, 1.0)
    min_child_weight = trial.suggest_float("lgbm_min_child_weight", 1e-3, 10.0, log=True)

    # Additional LightGBM parameters
    boosting_type_str = trial.suggest_categorical("lgbm_boosting_type", ("gbdt", "dart", "goss"))
    objective_str = trial.suggest_categorical("lgbm_objective", ("binary",))
    metric_str = trial.suggest_categorical("lgbm_metric", ("binary_logloss", "auc"))

    # Convert strings to enum objects
    boosting_type = BoostingType(boosting_type_str)
    objective = ObjectiveType(objective_str)
    metric = MetricType(metric_str)

    params = {
        "use_log_transform": use_log_transform,
        "use_zscore_norm": use_zscore_norm,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_child_samples": min_child_samples,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "class_weight": class_weight,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "min_split_gain": min_split_gain,
        "min_child_weight": min_child_weight,
        "boosting_type": boosting_type,
        "objective": objective,
        "metric": metric,
    }
    return params
