from dataclasses import dataclass, field

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
class MLPParam(BaseParams):
    use_log_transform: bool = True
    n_epochs: int = 10
    early_stopping_patience: int = 5
    init_learning_rate: float = 1e-4
    weight_decay: float = 5e-2
    max_grad_norm: float = 10.0
    lr_scheduler_config: dict[str, object] = field(
        default_factory=lambda: {
            "type": LRSchedulerType.REDUCE_LR_ON_PLATEAU.value,
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
        }
    )
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.REDUCE_LR_ON_PLATEAU
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    adam_betas: tuple = (0.9, 0.999)
    train_batch_size: int = 2048
    inference_batch_size: int = 2048
    sampler_type: SamplerType = SamplerType.RANDOM
    layer_units: tuple = (1024, 512, 256, 128)
    dropout_rate: float = 0.3
    mask_rate: float = 0.1
    noise_std: float = 1e-4
    soft_loss_beta: float | None = 1.0
    soft_loss_start_epoch: int | None = None
    positives_weight: float = 1.0
    train_log_interval: int = 5
    random_seed: int = 42
    # Optional categorical types to be used as auxiliary classification targets (for example, tissue_type, imputed_labels, etc.)
    aux_categorical_types: tuple[str, ...] = field(default_factory=tuple)
    # Loss weight for auxiliary categorical classification
    cat_loss_weight: float = 1.0

    def get_dataset_params(self) -> DataAugmentParams:
        """Return a DatasetParams object with dataset-related parameters."""
        return DataAugmentParams(
            mask_rate=self.mask_rate,
            noise_std=self.noise_std,
            soft_loss_beta=self.soft_loss_beta,
        )

    def get_optimizer_params(self) -> OptimizerParams:
        """Return an OptimizerParams object with optimizer-related parameters."""
        return OptimizerParams(
            optimizer_type=self.optimizer_type,
            adam_betas=self.adam_betas,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            init_learning_rate=self.init_learning_rate,
        )

    def get_lr_scheduler_params(self) -> LRSchedulerParams:
        """Return an LRSchedulerParams object with learning rate scheduler-related parameters."""
        return LRSchedulerParams(
            lr_scheduler_type=self.lr_scheduler_config["type"],
            lr_scheduler_config=self.lr_scheduler_config,
        )

    def get_data_loader_params(self) -> DataLoaderParams:
        """Return a DataLoaderParams object with data loader-related parameters."""
        return DataLoaderParams(
            batch_size=self.train_batch_size,
            shuffle=True,
            sampler_type=self.sampler_type,
        )

    @classmethod
    def get_model_type(cls) -> str:
        """Return the model type identifier for this parameter class."""
        return "mlp"
