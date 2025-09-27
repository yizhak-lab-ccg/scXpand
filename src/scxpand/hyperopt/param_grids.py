"""Hyperparameter configuration functions for Optuna optimization.

This module defines the parameter search spaces for all supported model types.
Each function suggests hyperparameters for a specific model architecture using
Optuna's trial.suggest_* methods with appropriate ranges and distributions.
"""

import optuna

from scxpand.lightgbm.lightgbm_params import BoostingType, MetricType, ObjectiveType
from scxpand.util.classes import LRSchedulerType, SamplerType

# Fixed choices for auxiliary categorical tasks, stored as strings for Optuna persistence
# These represent additional classification tasks that can be learned alongside the main task
AUX_CATEGORICAL_OPTIONS = (
    "none",  # No auxiliary tasks
    "tissue_type",  # Predict tissue type from gene expression
    "imputed_labels",  # Predict imputed cell type labels
    "tissue_type,imputed_labels",  # Both auxiliary tasks simultaneously
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
                "min_lr": trial.suggest_float(
                    f"{prefix}lr_min_lr", 1e-7, 1e-5, log=True
                ),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.ONE_CYCLE_LR.value:
        lr_scheduler_config.update(
            {
                "warmup_ratio": trial.suggest_float(
                    f"{prefix}lr_warmup_ratio", 0.05, 0.3
                ),
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
                "eta_min": trial.suggest_float(
                    f"{prefix}lr_eta_min", 1e-7, 1e-5, log=True
                ),
            }
        )
    elif lr_scheduler_type == LRSchedulerType.CONSTANT_LR.value:
        # ConstantLR maintains the same learning rate throughout training
        lr_scheduler_config.update({})

    return lr_scheduler_config


def configure_mlp_trial_params(trial: optuna.Trial) -> dict:
    """Configure MLP (Multi-layer Perceptron) hyperparameters for Optuna optimization.

    This function defines the search space for neural network hyperparameters including:
    - Architecture: Number of layers (2-5) and layer sizes (512-4096 units)
    - Training: Learning rate (1e-5 to 1e-3), batch size (2048 or 4096), 30 epochs
    - Regularization: Dropout (0.1-0.3), weight decay (1e-5 to 1e-2), data augmentation
    - Optimization: Adam optimizer with configurable learning rate schedulers
    - Advanced: Soft loss, auxiliary tasks, balanced sampling strategies

    Args:
        trial: Optuna trial object for parameter suggestion

    Returns:
        Dictionary of MLP parameters (unprefixed for model instantiation)
    """
    # Data preprocessing: log transform can help with gene expression data distribution
    use_log_transform = trial.suggest_categorical(
        "mlp_use_log_transform", (True, False)
    )

    # Training configuration: fixed 30 epochs for consistent comparison across trials
    # This provides sufficient training time while keeping optimization tractable
    n_epochs = 30

    # Learning rate: conservative range (1e-5 to 1e-3) based on neural network best practices
    # Log-uniform distribution allows exploration of different orders of magnitude
    # Lower bound prevents gradient explosion, upper bound ensures stable convergence
    init_learning_rate = trial.suggest_float(
        "mlp_init_learning_rate", 1e-5, 1e-3, log=True
    )

    # Batch size: larger batches (2048, 4096) for stable gradient estimates
    # These sizes balance memory usage with gradient stability for gene expression data
    train_batch_size = trial.suggest_categorical("mlp_train_batch_size", (2048, 4096))

    # Regularization: moderate dropout to prevent overfitting without losing capacity
    dropout_rate = trial.suggest_float("mlp_dropout_rate", 0.1, 0.3)

    # Architecture: 2-5 layers provides good expressiveness without excessive complexity
    num_layers = trial.suggest_int("mlp_num_layers", 2, 5)

    # Weight decay: L2 regularization with log-uniform distribution
    weight_decay = trial.suggest_float("mlp_weight_decay", 1e-5, 1e-2, log=True)

    # Data augmentation: masking and noise injection for robustness
    mask_rate = trial.suggest_float("mlp_mask_rate", 0.05, 0.3)  # 5-30% feature masking
    noise_std = trial.suggest_float(
        "mlp_noise_std", 1e-5, 1e-3, log=True
    )  # Gaussian noise

    # Class balancing: handle imbalanced datasets
    positives_weight = trial.suggest_float("mlp_positives_weight", 0.1, 10.0)

    # Soft loss: alternative loss function that can improve generalization
    use_soft_loss = trial.suggest_categorical("mlp_use_soft_loss", (True, False))
    # Sampling strategy: different approaches for handling class imbalance
    sampler_type_str = trial.suggest_categorical(
        "mlp_sampler_type",
        (
            SamplerType.BALANCED_LABELS.value,
            SamplerType.BALANCED_TYPES.value,
            SamplerType.RANDOM.value,
        ),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Inference configuration: match training batch size for consistency
    inference_batch_size = train_batch_size

    # Auxiliary task weight: balance between main task and auxiliary classification
    cat_loss_weight = trial.suggest_float("mlp_cat_loss_weight", 0.1, 10.0, log=True)

    # Learning rate scheduling: various strategies for adaptive learning rates
    lr_scheduler_type_str = trial.suggest_categorical(
        "mlp_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,  # Reduce on plateau
            LRSchedulerType.ONE_CYCLE_LR.value,  # One cycle policy
            LRSchedulerType.STEP_LR.value,  # Step decay
            LRSchedulerType.COSINE_ANNEALING_LR.value,  # Cosine annealing
            LRSchedulerType.CONSTANT_LR.value,  # Constant rate
            LRSchedulerType.NO_SCHEDULER.value,  # No scheduling
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)

    # Adam optimizer: momentum parameters for adaptive learning
    adam_beta1 = trial.suggest_float("mlp_adam_beta1", 0.8, 0.95)  # First moment decay
    adam_beta2 = trial.suggest_float(
        "mlp_adam_beta2", 0.95, 0.999
    )  # Second moment decay
    adam_betas = (adam_beta1, adam_beta2)

    # Generate scheduler-specific configuration with optimized parameters
    lr_scheduler_config = create_optimized_lr_scheduler_config(
        trial, lr_scheduler_type, n_epochs, "mlp_"
    )

    # Auxiliary tasks: additional classification objectives for multi-task learning
    aux_categorical_types_str = trial.suggest_categorical(
        "mlp_aux_categorical_types",
        AUX_CATEGORICAL_OPTIONS,
    )
    if aux_categorical_types_str == "none":
        aux_categorical_types = ()
    else:
        aux_categorical_types = tuple(aux_categorical_types_str.split(","))

    # Layer architecture: variable number of layers with configurable sizes
    layer_units = []
    for i in range(num_layers):
        layer_name = f"mlp_layer_{i}_units"
        units = trial.suggest_categorical(layer_name, (512, 1024, 2048, 4096))
        layer_units.append(units)

    # Soft loss parameters: only used when soft loss is enabled
    soft_loss_beta = 1.0
    soft_loss_start_epoch = -1
    if use_soft_loss:
        soft_loss_beta = trial.suggest_float("mlp_soft_loss_beta", 0.5, 10.0)
        soft_loss_start_epoch = trial.suggest_int(
            "mlp_soft_loss_start_epoch", 0, n_epochs - 1
        )

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
    """Configure Autoencoder hyperparameters for Optuna optimization.

    This function defines the search space for autoencoder hyperparameters including:
    - Architecture: Standard vs fork variants, encoder/decoder layers (1-3), latent dimensions (16-128)
    - Loss Functions: MSE (mean squared error), NB (negative binomial), ZINB (zero-inflated negative binomial)
    - Training: Learning rate (1e-6 to 1e-3), batch size (2048 or 4096), 30 epochs
    - Regularization: Dropout (0.1-0.5), L1/L2 regularization, data augmentation
    - Multi-task: Configurable loss weights for reconstruction, classification, and auxiliary tasks

    Args:
        trial: Optuna trial object for parameter suggestion

    Returns:
        Dictionary of autoencoder parameters (unprefixed for model instantiation)
    """
    # Data preprocessing: log transform for gene expression data normalization
    use_log_transform = trial.suggest_categorical("ae_use_log_transform", (True, False))

    # Training configuration: fixed 30 epochs for consistent comparison
    n_epochs = 30

    # Learning rate: ultra-conservative range (1e-6 to 1e-3) for autoencoder stability
    # Autoencoders can be sensitive to learning rate, especially with reconstruction loss
    # Lower bound prevents reconstruction instability, upper bound ensures convergence
    init_learning_rate = trial.suggest_float(
        "ae_init_learning_rate", 1e-6, 1e-3, log=True
    )

    # Batch size: larger batches for stable reconstruction learning
    train_batch_size = trial.suggest_categorical("ae_train_batch_size", (2048, 4096))

    # Sampling strategy: handle class imbalance in multi-task learning
    sampler_type_str = trial.suggest_categorical(
        "ae_sampler_type",
        (
            SamplerType.BALANCED_LABELS.value,
            SamplerType.BALANCED_TYPES.value,
            SamplerType.RANDOM.value,
        ),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Inference configuration: match training batch size
    inference_batch_size = train_batch_size

    # Learning rate scheduling: adaptive learning rates for complex autoencoder training
    lr_scheduler_type_str = trial.suggest_categorical(
        "ae_lr_scheduler_type",
        (
            LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,  # Reduce on plateau
            LRSchedulerType.ONE_CYCLE_LR.value,  # One cycle policy
            LRSchedulerType.STEP_LR.value,  # Step decay
            LRSchedulerType.COSINE_ANNEALING_LR.value,  # Cosine annealing
            LRSchedulerType.CONSTANT_LR.value,  # Constant rate
            LRSchedulerType.NO_SCHEDULER.value,  # No scheduling
        ),
    )
    lr_scheduler_type = LRSchedulerType(lr_scheduler_type_str)

    # Adam optimizer: momentum parameters for stable autoencoder training
    adam_beta1 = trial.suggest_float("ae_adam_beta1", 0.8, 0.95)  # First moment decay
    adam_beta2 = trial.suggest_float(
        "ae_adam_beta2", 0.95, 0.999
    )  # Second moment decay
    adam_betas = (adam_beta1, adam_beta2)

    # Generate scheduler-specific configuration with optimized parameters
    lr_scheduler_config = create_optimized_lr_scheduler_config(
        trial, lr_scheduler_type, n_epochs, "ae_"
    )

    # Architecture variants: different autoencoder designs
    model_type = trial.suggest_categorical(
        "ae_model_type", ("standard", "fork")
    )  # Standard vs fork architecture

    # Loss functions: different approaches for count data modeling
    loss_type = trial.suggest_categorical(
        "ae_loss_type", ("mse", "nb", "zinb")
    )  # MSE, Negative Binomial, Zero-inflated NB

    # Latent dimension: bottleneck size affects reconstruction quality vs compression
    latent_dim = trial.suggest_categorical("ae_latent_dim", (16, 32, 64, 128))

    # Multi-task loss weights: balance reconstruction vs classification objectives
    recon_loss_weight = trial.suggest_float(
        "ae_recon_loss_weight", 0.1, 10.0, log=True
    )  # Reconstruction loss weight
    cls_loss_weight = trial.suggest_float(
        "ae_cls_loss_weight", 0.1, 10.0, log=True
    )  # Classification loss weight
    cat_loss_weight = trial.suggest_float(
        "ae_cat_loss_weight", 0.1, 10.0, log=True
    )  # Auxiliary task loss weight

    # Data augmentation: improve robustness and prevent overfitting
    mask_rate = trial.suggest_float(
        "ae_mask_rate", 0.05, 0.5
    )  # 5-50% feature masking (higher range than MLP)
    noise_std = trial.suggest_float(
        "ae_noise_std", 1e-5, 1e-3, log=True
    )  # Gaussian noise injection
    positives_weight = trial.suggest_float(
        "ae_positives_weight", 0.1, 10.0
    )  # Class balancing

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
        soft_loss_start_epoch = trial.suggest_int(
            "ae_soft_loss_start_epoch", 0, n_epochs - 1
        )

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
    # Ridge regularization: L2 penalty on zero-inflation parameters (pi)
    # Only added to params when using ZINB loss, but ridge_lambda exists in AutoEncoderParams for all loss types
    if loss_type == "zinb":
        params["ridge_lambda"] = trial.suggest_float(
            "ae_ridge_lambda", 1e-4, 1e-1, log=True
        )

    return params


def configure_logistic_trial_params(trial: optuna.Trial) -> dict:
    """Configure Logistic Regression hyperparameters for Optuna optimization.

    This function defines the search space for logistic regression hyperparameters including:
    - Optimization: SGD with 30 epochs, learning rates 1e-5 to 1e-2 (log-uniform)
    - Regularization: L2 and elasticnet penalties (alpha: 1e-6 to 1e-1)
    - Scheduling: Multiple learning rate schedules (optimal, constant, invscaling, adaptive)
    - Data: Optional log transform, configurable batch sizes (512-2048)
    - Features: Warm start, averaging, early stopping, data augmentation

    Args:
        trial: Optuna trial object for parameter suggestion

    Returns:
        Dictionary of logistic regression parameters (unprefixed for model instantiation)
    """
    # Data preprocessing: log transform for gene expression data normalization
    use_log_transform = trial.suggest_categorical(
        "logistic_use_log_transform", (True, False)
    )

    # Regularization: L2 and elasticnet are most stable for logistic regression with SGD
    penalty = trial.suggest_categorical("logistic_penalty", ("l2", "elasticnet"))

    # Regularization strength: conservative range for stable convergence
    alpha = trial.suggest_float("logistic_alpha", 1e-6, 1e-1, log=True)

    # Training configuration: fixed 30 epochs for consistent comparison across model types
    # Provides sufficient training time while maintaining optimization efficiency
    n_epochs = 30

    # Convergence tolerance: tighter range for precise optimization
    tol = trial.suggest_float("logistic_tol", 1e-6, 1e-3, log=True)

    # Class balancing: handle imbalanced datasets
    class_weight = trial.suggest_categorical(
        "logistic_class_weight", ("balanced", None)
    )

    # Batch size: smaller batches for SGD stability
    batch_size = trial.suggest_categorical("logistic_batch_size", (512, 1024, 2048))
    # Sampling strategy: handle class imbalance in SGD training
    sampler_type_str = trial.suggest_categorical(
        "logistic_sampler_type",
        (
            SamplerType.BALANCED_LABELS.value,
            SamplerType.BALANCED_TYPES.value,
            SamplerType.RANDOM.value,
        ),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Learning rate configuration: aligned with scikit-learn SGD defaults
    # Range (1e-5 to 1e-2) balances convergence speed with stability for linear models
    init_learning_rate = trial.suggest_float(
        "logistic_init_learning_rate", 1e-5, 1e-2, log=True
    )

    # Learning rate scheduling: different strategies for adaptive learning
    learning_rate = trial.suggest_categorical(
        "logistic_learning_rate", ("optimal", "constant", "invscaling", "adaptive")
    )
    # optimal: adaptive learning rate based on regularization
    # constant: fixed learning rate
    # invscaling: inverse scaling with power_t
    # adaptive: adaptive learning rate that decreases on plateau

    # Initial learning rate: only used when learning_rate is not 'optimal'
    eta0 = trial.suggest_float("logistic_eta0", 1e-3, 1.0, log=True)

    # Power parameter: controls inverse scaling decay (default 0.5)
    power_t = trial.suggest_float("logistic_power_t", 0.25, 0.75)

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
    lr_scheduler_config = create_optimized_lr_scheduler_config(
        trial, lr_scheduler_type, n_epochs, "logistic_"
    )

    # SGD-specific parameters - aligned with scikit-learn recommendations
    warm_start = trial.suggest_categorical("logistic_warm_start", (True, False))
    average = trial.suggest_categorical("logistic_average", (True, False))
    # Note: n_iter_no_change and validation_fraction are included for completeness
    # but are not used in the current implementation since early_stopping=False with partial_fit()
    n_iter_no_change = trial.suggest_int("logistic_n_iter_no_change", 3, 10)
    validation_fraction = trial.suggest_float("logistic_validation_fraction", 0.05, 0.2)
    # Note: early_stopping is always False when using partial_fit() - set in model initialization

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
        l1_ratio = trial.suggest_float(
            "logistic_l1_ratio", 0.1, 0.9
        )  # Avoid extremes 0.0 and 1.0

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
    """Configure SVM (Support Vector Machine) hyperparameters for Optuna optimization.

    This function defines the search space for SVM hyperparameters including:
    - Optimization: SGD with 30 epochs, learning rates 1e-5 to 1e-2 (log-uniform)
    - Regularization: L2 and elasticnet penalties (alpha: 1e-6 to 1e-1)
    - Scheduling: Multiple learning rate schedules (optimal, constant, invscaling, adaptive)
    - Data: Optional log transform, configurable batch sizes (512-2048)
    - Features: Warm start, averaging, early stopping, data augmentation

    Note: SVM uses hinge loss, which only supports L2 and elasticnet penalties (not L1).

    Args:
        trial: Optuna trial object for parameter suggestion

    Returns:
        Dictionary of SVM parameters (unprefixed for model instantiation)
    """
    # Data preprocessing: log transform for gene expression data normalization
    use_log_transform = trial.suggest_categorical(
        "svm_use_log_transform", (True, False)
    )

    # Regularization: For SVM (hinge loss), L1 penalty is not supported - only L2 and elasticnet
    penalty = trial.suggest_categorical("svm_penalty", ("l2", "elasticnet"))

    # Regularization strength: conservative range for SVM stability
    alpha = trial.suggest_float("svm_alpha", 1e-6, 1e-1, log=True)

    # Training configuration: fixed 30 epochs for consistent comparison across model types
    # Provides sufficient training time while maintaining optimization efficiency
    n_epochs = 30

    # Convergence tolerance: tighter range for precise optimization
    tol = trial.suggest_float("svm_tol", 1e-6, 1e-3, log=True)

    # Class balancing: handle imbalanced datasets
    class_weight = trial.suggest_categorical("svm_class_weight", ("balanced", None))

    # Batch size: smaller batches for SGD stability
    batch_size = trial.suggest_categorical("svm_batch_size", (512, 1024, 2048))
    sampler_type_str = trial.suggest_categorical(
        "svm_sampler_type",
        (
            SamplerType.BALANCED_LABELS.value,
            SamplerType.BALANCED_TYPES.value,
            SamplerType.RANDOM.value,
        ),
    )
    sampler_type = SamplerType(sampler_type_str)

    # Learning rate and optimizer parameters - aligned with scikit-learn defaults
    # Range (1e-5 to 1e-2) balances convergence speed with stability for SVM with hinge loss
    init_learning_rate = trial.suggest_float(
        "svm_init_learning_rate", 1e-5, 1e-2, log=True
    )
    learning_rate = trial.suggest_categorical(
        "svm_learning_rate", ("optimal", "constant", "invscaling", "adaptive")
    )
    # eta0 is only used when learning_rate is not 'optimal'
    eta0 = trial.suggest_float(
        "svm_eta0", 1e-3, 1.0, log=True
    )  # Better range for initial learning rate
    power_t = trial.suggest_float(
        "svm_power_t", 0.25, 0.75
    )  # More focused around default 0.5

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
    lr_scheduler_config = create_optimized_lr_scheduler_config(
        trial, lr_scheduler_type, n_epochs, "svm_"
    )

    # SGD-specific parameters - aligned with scikit-learn recommendations
    warm_start = trial.suggest_categorical("svm_warm_start", (True, False))
    average = trial.suggest_categorical("svm_average", (True, False))
    # Note: n_iter_no_change and validation_fraction are included for completeness
    # but are not used in the current implementation since early_stopping=False with partial_fit()
    n_iter_no_change = trial.suggest_int("svm_n_iter_no_change", 3, 10)
    validation_fraction = trial.suggest_float("svm_validation_fraction", 0.05, 0.2)
    # Note: early_stopping is always False when using partial_fit() - set in model initialization

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
        l1_ratio = trial.suggest_float(
            "svm_l1_ratio", 0.1, 0.9
        )  # Avoid extremes 0.0 and 1.0

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
    """Configure LightGBM hyperparameters for Optuna optimization.

    This function defines the search space for gradient boosting hyperparameters including:
    - Tree Structure: Max depth (3-12), number of leaves (15-127), estimators (50-300)
    - Learning: Learning rate (1e-3 to 0.1), min child samples (5-100)
    - Regularization: Feature/bagging fractions (0.7-0.95), alpha/lambda regularization
    - Boosting: GBDT, DART, GOSS variants with binary objective
    - Preprocessing: Optional log transform and z-score normalization

    Args:
        trial: Optuna trial object for parameter suggestion

    Returns:
        Dictionary of LightGBM parameters (unprefixed for model instantiation)
    """
    # Data preprocessing: log transform for gene expression data normalization
    use_log_transform = trial.suggest_categorical(
        "lgbm_use_log_transform", (True, False)
    )

    # Additional normalization: z-score standardization
    use_zscore_norm = trial.suggest_categorical("lgbm_use_zscore_norm", (True, False))

    # Tree structure: balance between model complexity and overfitting
    num_leaves = trial.suggest_int(
        "lgbm_num_leaves", 15, 127
    )  # Number of leaves per tree
    max_depth = trial.suggest_int("lgbm_max_depth", 3, 12)  # Maximum tree depth

    # Learning configuration: conservative learning rate with sufficient iterations
    # Learning rate range (1e-3 to 0.1) prevents overfitting while ensuring convergence
    learning_rate = trial.suggest_float("lgbm_learning_rate", 1e-3, 0.1, log=True)
    # Number of boosting rounds: 50-300 provides good bias-variance tradeoff
    n_estimators = trial.suggest_int("lgbm_n_estimators", 50, 300)

    # Tree regularization: prevent overfitting on small samples
    min_child_samples = trial.suggest_int(
        "lgbm_min_child_samples", 5, 100
    )  # Minimum samples per leaf

    # L1 and L2 regularization: control model complexity
    reg_alpha = trial.suggest_float(
        "lgbm_reg_alpha", 1e-8, 10.0, log=True
    )  # L1 regularization
    reg_lambda = trial.suggest_float(
        "lgbm_reg_lambda", 1e-8, 10.0, log=True
    )  # L2 regularization

    # Class balancing: handle imbalanced datasets
    class_weight = trial.suggest_categorical("lgbm_class_weight", ("balanced", None))

    # Feature and bagging fractions: LightGBM's equivalent to neural network dropout
    # feature_fraction=0.7 means 30% of features are randomly dropped per tree
    # This is analogous to a 30% dropout rate in neural networks
    feature_fraction = trial.suggest_float(
        "lgbm_feature_fraction", 0.7, 0.95
    )  # Feature sampling ratio
    bagging_fraction = trial.suggest_float(
        "lgbm_bagging_fraction", 0.7, 0.95
    )  # Sample sampling ratio

    # Split criteria: minimum gain required for splits
    min_split_gain = trial.suggest_float(
        "lgbm_min_split_gain", 0.0, 1.0
    )  # Minimum split gain
    min_child_weight = trial.suggest_float(
        "lgbm_min_child_weight", 1e-3, 10.0, log=True
    )  # Minimum child weight

    # Boosting algorithm variants: different approaches to gradient boosting
    boosting_type_str = trial.suggest_categorical(
        "lgbm_boosting_type", ("gbdt", "dart", "goss")
    )
    # gbdt: Gradient Boosting Decision Tree (standard)
    # dart: Dropouts meet Multiple Additive Regression Trees (regularized)
    # goss: Gradient-based One-Side Sampling (efficient)

    # Objective and metric: binary classification setup
    objective_str = trial.suggest_categorical(
        "lgbm_objective", ("binary",)
    )  # Binary classification
    metric_str = trial.suggest_categorical(
        "lgbm_metric", ("binary_logloss", "auc")
    )  # Evaluation metrics

    # Convert strings to enum objects for type safety
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
