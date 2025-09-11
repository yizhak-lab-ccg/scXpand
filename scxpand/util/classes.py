from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class BaseParams(ABC):
    """Abstract base class for all parameter classes.

    Provides a common interface for parameter classes with a shared get_model_type method.
    All parameter classes should inherit from this base class to ensure consistency.
    """

    @classmethod
    @abstractmethod
    def get_model_type(cls) -> str:
        """Return the model type identifier for this parameter class.

        Returns:
            String identifier for the model type (e.g., 'mlp', 'autoencoder', 'logistic', 'svm', etc.)
        """


class ModelType(str, Enum):
    """Enumeration of supported model types."""

    AUTOENCODER = "autoencoder"
    MLP = "mlp"
    LIGHTGBM = "lightgbm"
    LOGISTIC = "logistic"
    SVM = "svm"


class LRSchedulerType(str, Enum):
    REDUCE_LR_ON_PLATEAU = "ReduceLROnPlateau"
    ONE_CYCLE_LR = "OneCycleLR"
    STEP_LR = "StepLR"
    COSINE_ANNEALING_LR = "CosineAnnealingLR"
    CONSTANT_LR = "ConstantLR"
    NO_SCHEDULER = "NoScheduler"


class SamplerType(str, Enum):
    """Enumeration of supported sampler types."""

    BALANCED_LABELS = "balanced_labels"
    BALANCED_TYPES = "balanced_types"
    RANDOM = "random"


class OptimizerType(str, Enum):
    """Enumeration of supported optimizer types."""

    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"


def ensure_model_type(model_type: "ModelType | str") -> "ModelType":
    """Convert string to ModelType enum if needed, with validation."""
    if isinstance(model_type, str):
        try:
            return ModelType(model_type)
        except ValueError:
            valid_types = [m.value for m in ModelType]
            raise ValueError(f"model_type must be one of {valid_types}") from None
    return model_type


@dataclass
class DataAugmentParams:
    mask_rate: float = 0.0
    noise_std: float = 0.0
    soft_loss_beta: float = 1.0


@dataclass
class OptimizerParams:
    optimizer_type: OptimizerType
    adam_betas: tuple
    weight_decay: float
    max_grad_norm: float
    init_learning_rate: float


@dataclass
class LRSchedulerParams:
    lr_scheduler_type: LRSchedulerType
    lr_scheduler_config: dict


@dataclass
class DataLoaderParams:
    batch_size: int
    shuffle: bool
    sampler_type: SamplerType
