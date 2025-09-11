from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

from scxpand.util.classes import (
    BaseParams,
    DataAugmentParams,
    DataLoaderParams,
    LRSchedulerParams,
    LRSchedulerType,
    OptimizerParams,
    OptimizerType,
    SamplerType,
)


@dataclass
class AutoEncoderParams(BaseParams):
    """Configuration parameters for autoencoder training and architecture.

    Contains all hyperparameters needed to configure and train an autoencoder
    model for T-cell expansion prediction. Includes architecture settings,
    training parameters, regularization, and optimization settings.

    Architecture Parameters:
        latent_dim: Dimensionality of the latent embedding space.
        encoder_hidden_dims: Hidden layer sizes for the encoder network.
        decoder_hidden_dims: Hidden layer sizes for the decoder network.
        classifier_hidden_dims: Hidden layer sizes for the classification head.
        dropout_rate: Dropout probability for regularization.

    Training Parameters:
        n_epochs: Maximum number of training epochs.
        early_stopping_patience: Epochs to wait for improvement before stopping.
        train_batch_size: Batch size for training.
        inference_batch_size: Batch size for inference.

    Loss and Regularization:
        recon_loss_weight: Weight for reconstruction loss component.
        cls_loss_weight: Weight for classification loss component.
        ridge_lambda: L2 regularization coefficient.
        l1_lambda: L1 regularization coefficient for latent vectors.

    Example:
        >>> params = AutoEncoderParams(latent_dim=64, n_epochs=50)
        >>> # Customize for your dataset
        >>> params.encoder_hidden_dims = (128, 64)
        >>> params.init_learning_rate = 1e-4
    """

    use_log_transform: bool = True
    n_epochs: int = 10
    early_stopping_patience: int = 5
    init_learning_rate: float = 5e-5  #  low to prevent loss explosion
    ridge_lambda: float = 0.01  #  regularization for stability
    l1_lambda: float = 0.001  # Coefficient for L1 regularization on latent vector
    recon_loss_weight: float = 1.0  # Coefficient for weighting the reconstruction loss
    cls_loss_weight: float = 1.0  # Coefficient for weighting the classification loss
    cat_loss_weight: float = 1.0  # Coefficient for weighting the categorical loss
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    lr_scheduler_config: Dict[str, Any] | None = field(
        default_factory=lambda: {
            "type": LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
        }
    )
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.REDUCE_LR_ON_PLATEAU
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    train_batch_size: int = 2048
    inference_batch_size: int = 2048
    sampler_type: SamplerType = SamplerType.RANDOM
    latent_dim: int = 32
    encoder_hidden_dims: Tuple[int, ...] = (64,)
    decoder_hidden_dims: Tuple[int, ...] = (64,)
    classifier_hidden_dims: Tuple[int, ...] = (16,)
    dropout_rate: float = 0.1
    mask_rate: float = 0.1
    noise_std: float = 1e-4
    soft_loss_beta: float | None = 1.0
    soft_loss_start_epoch: int | None = None
    positives_weight: float = 1.0
    train_log_interval: int = 5
    random_seed: int = 42
    model_type: Literal["standard", "fork"] = "standard"
    loss_type: Literal["zinb", "nb", "mse"] = "mse"
    # Optional categorical types to be used as auxiliary classification targets (for example, tissue_type, imputed_labels, etc.)
    aux_categorical_types: Tuple[str, ...] = field(default_factory=tuple)

    def get_dataset_params(self) -> DataAugmentParams:
        return DataAugmentParams(
            mask_rate=self.mask_rate,
            noise_std=self.noise_std,
            soft_loss_beta=self.soft_loss_beta,
        )

    def get_optimizer_params(self) -> OptimizerParams:
        return OptimizerParams(
            optimizer_type=self.optimizer_type,
            adam_betas=self.adam_betas,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            init_learning_rate=self.init_learning_rate,
        )

    def get_lr_scheduler_params(self) -> LRSchedulerParams:
        return LRSchedulerParams(
            lr_scheduler_type=self.lr_scheduler_type,
            lr_scheduler_config=self.lr_scheduler_config,
        )

    def get_data_loader_params(self) -> DataLoaderParams:
        return DataLoaderParams(
            batch_size=self.train_batch_size,
            shuffle=True,
            sampler_type=self.sampler_type,
        )

    def needs_pi_head(self) -> bool:
        """Return True if the loss type requires pi (zero-inflation) parameter."""
        return self.loss_type.lower() == "zinb"

    def needs_theta_head(self) -> bool:
        """Return True if the loss type requires theta (dispersion) parameter."""
        return self.loss_type.lower() in ["nb", "zinb"]

    @classmethod
    def get_model_type(cls) -> str:
        """Return the model type identifier for this parameter class."""
        return "autoencoder"
