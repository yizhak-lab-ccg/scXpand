"""Training utility functions for scXpand models.

This module contains common utilities for training operations:
- Model validation and setup
- Training function routing
- Path and configuration validation
"""

from pathlib import Path
from typing import Any, Callable

from scxpand.core.model_types import MODEL_TYPES, ModelSpec
from scxpand.util.classes import ModelType, ensure_model_type
from scxpand.util.general_util import get_device
from scxpand.util.logger import get_logger


logger = get_logger()


def validate_and_setup_common(
    model_type: ModelType | str,
    data_path: str | None = None,
    model_path: str | None = None,
) -> tuple[ModelType, ModelSpec]:
    """Common validation and setup for all main functions.

    Args:
        model_type: Type of model to validate
        data_path: Path to data file (optional, for training/optimization)
        model_path: Path to model directory (optional, for prediction)

    Returns:
        Tuple of (validated_model_type, model_spec)
    """
    # Validate and convert model type
    model_type_enum = ensure_model_type(model_type)

    # Validate model is supported
    if model_type_enum not in MODEL_TYPES:
        supported_models = [m.value for m in ModelType]
        raise ValueError(f"model_type must be one of {supported_models}")

    # Validate data path if provided
    if data_path is not None:
        if not data_path:
            raise ValueError("data_path cannot be empty")
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not data_file.is_file():
            raise ValueError(f"Data path is not a file: {data_path}")

    # Validate model path if provided
    if model_path is not None:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not model_dir.is_dir():
            raise ValueError(f"Model path is not a directory: {model_path}")

    # Get model specification
    model_spec = MODEL_TYPES[model_type_enum]

    # Log common info
    logger.info(f"Using {model_type_enum.value} model")
    logger.info(f"Using device: {get_device()}")

    return model_type_enum, model_spec


def call_training_function(
    model_type: ModelType,
    train_fn: Callable,
    data_path: str,
    save_dir: str,
    prm: Any,
    resume: bool = False,
    num_workers: int = 0,
) -> dict[str, dict[str, float]]:
    """Call the appropriate training function with standardized parameters."""
    # Neural network models (autoencoder, mlp) use consistent signatures
    if model_type in (ModelType.AUTOENCODER, ModelType.MLP):
        return train_fn(
            data_path=data_path,
            base_save_dir=save_dir,
            prm=prm,
            resume=resume,
            num_workers=num_workers,
        )
    # LightGBM doesn't use resume or num_workers
    elif model_type == ModelType.LIGHTGBM:
        return train_fn(
            base_save_dir=save_dir,
            prm=prm,
            data_path=data_path,
        )
    # Linear models (logistic, svm) use base_save_dir, prm, data_path, num_workers
    elif model_type in (ModelType.LOGISTIC, ModelType.SVM):
        return train_fn(
            base_save_dir=save_dir,
            prm=prm,
            data_path=data_path,
            num_workers=num_workers,
        )
    else:
        model_value = model_type.value if hasattr(model_type, "value") else str(model_type)
        raise ValueError(f"Unknown model type for training: {model_value}")
