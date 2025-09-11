"""Parameters for linear classifier training."""

from dataclasses import dataclass, field

from scxpand.util.classes import (
    BaseParams,
    DataAugmentParams,
    LRSchedulerParams,
    LRSchedulerType,
    OptimizerParams,
    OptimizerType,
    SamplerType,
)


@dataclass
class LinearClassifierParam(BaseParams):
    """Parameters for linear classifier - aligned with scikit-learn SGDClassifier defaults."""

    use_log_transform: bool = True
    # Model type: 'logistic' for logistic regression, 'svm' for support vector machine
    model_type: str = "logistic"
    # Regularization parameter (alpha is inverse of C) - scikit-learn default: 0.0001
    alpha: float = 0.0001
    # Regularization type - scikit-learn default: 'l2'
    penalty: str = "l2"
    # Maximum number of iterations
    n_epochs: int = 1000
    # Class weight balancing - scikit-learn default: None
    class_weight: str | dict | None = None
    # Tolerance for stopping criterion - scikit-learn default: 1e-3
    tol: float = 1e-3
    # L1 ratio for elasticnet penalty - scikit-learn default: 0.15
    l1_ratio: float = 0.15
    # Random seed
    random_seed: int = 42
    # Batch size for processing
    batch_size: int = 2048
    # Early stopping patience
    early_stopping_patience: int = 5
    # Evaluation interval (every N epochs)
    eval_interval: int = 1
    # Training log interval (every N batches)
    train_log_interval: int = 10
    # Sampling strategy for batching
    sampler_type: SamplerType = SamplerType.RANDOM
    # Data augmentation parameters
    mask_rate: float = 0.0  # Rate at which to mask input features (set to 0)
    noise_std: float = 0.0  # Standard deviation of Gaussian noise to add
    soft_loss_beta: float | None = 1.0  # Scaling factor for soft labels (sigmoid scaling)

    # Learning rate and optimizer parameters for SGD - aligned with scikit-learn defaults
    init_learning_rate: float = 0.0001
    learning_rate: str = "optimal"  # scikit-learn default for classification
    eta0: float = 0.0  # scikit-learn default
    power_t: float = 0.5  # scikit-learn default

    # Learning rate scheduler parameters
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.CONSTANT_LR
    lr_scheduler_config: dict = field(
        default_factory=lambda: {
            "type": LRSchedulerType.CONSTANT_LR.value,
        }
    )

    # SGD-specific parameters - aligned with scikit-learn defaults
    max_iter: int = 1  # Set to 1 since we're using partial_fit in a loop
    warm_start: bool = False  # scikit-learn default
    average: bool = False  # scikit-learn default
    n_iter_no_change: int = 5  # scikit-learn default
    validation_fraction: float = 0.1  # scikit-learn default

    # Additional regularization parameters
    fit_intercept: bool = True  # scikit-learn default
    shuffle: bool = True  # scikit-learn default - important for SGD

    def get_dataset_params(self) -> DataAugmentParams:
        """Return a DataAugmentParams object with dataset-related parameters."""
        return DataAugmentParams(
            mask_rate=self.mask_rate,
            noise_std=self.noise_std,
            soft_loss_beta=self.soft_loss_beta,
        )

    def get_lr_scheduler_params(self) -> LRSchedulerParams:
        """Return a LRSchedulerParams object with learning rate scheduler parameters."""
        return LRSchedulerParams(
            lr_scheduler_type=self.lr_scheduler_type,
            lr_scheduler_config=self.lr_scheduler_config,
        )

    def get_optimizer_params(self) -> OptimizerParams:
        """Return an OptimizerParams object with optimizer parameters."""
        return OptimizerParams(
            optimizer_type=OptimizerType.SGD,  # Linear models use SGD
            adam_betas=(0.9, 0.999),  # Not used for SGD but required by interface
            weight_decay=0.0,  # Not used for SGD but required by interface
            max_grad_norm=float("inf"),  # Not used for SGD but required by interface
            init_learning_rate=self.init_learning_rate,
        )

    def get_model_type(self) -> str:
        """Return the model type identifier for this parameter class."""
        return self.model_type
