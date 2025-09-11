"""Core inference functions for scXpand models.

This module contains only the core inference functionality:
- Model loading
- Running inference on data
- Environment setup

Higher-level orchestration is handled by scxpand.core.prediction.
I/O operations are handled by scxpand.util.io.
Evaluation logic is handled by scxpand.core.evaluation.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import torch

from sklearn.base import BaseEstimator
from torch.nn import Module

from scxpand.autoencoders.ae_models import load_ae_model
from scxpand.autoencoders.ae_trainer import run_ae_inference
from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.lightgbm.run_lightgbm_ import run_lightgbm_inference
from scxpand.linear.sklearn_utils import load_sklearn_model, run_linear_inference
from scxpand.mlp.mlp_model import load_nn_model
from scxpand.mlp.mlp_trainer import run_mlp_inference
from scxpand.util.classes import ModelType, ensure_model_type
from scxpand.util.general_util import get_device
from scxpand.util.logger import get_logger


logger = get_logger()


def load_model(model_type: ModelType | str, model_path: Path, device: torch.device | str) -> BaseEstimator | Module:
    """Load a trained model from disk.

    Args:
        model_type: Type of model to load
        model_path: Path to directory containing the trained model
        device: Device to load the model onto (for neural networks)

    Returns:
        Loaded model (scikit-learn estimator or PyTorch module)

    Raises:
        ValueError: If model_type is not supported
    """
    # Ensure we have a ModelType enum
    try:
        model_type_enum = ensure_model_type(model_type)
    except ValueError:
        # Re-raise with the expected error message format for backward compatibility
        raise ValueError(f"Unsupported model_type for loading: '{model_type}'") from None

    logger.info(f"Loading {model_type_enum.value} model from {model_path}")

    if model_type_enum == ModelType.AUTOENCODER:
        model = load_ae_model(model_path=model_path, device=device)
    elif model_type_enum == ModelType.MLP:
        model = load_nn_model(results_path=model_path, device=device)
    elif model_type_enum in [ModelType.LIGHTGBM, ModelType.LOGISTIC, ModelType.SVM]:
        model = load_sklearn_model(results_path=model_path)
    else:
        raise ValueError(f"Unsupported model_type for loading: '{model_type_enum.value}'")

    # Validate that model was successfully loaded
    if model is None:
        raise RuntimeError("Model loading failed: model loader returned None")

    return model


def run_model_inference(
    model_type: ModelType | str,
    model: BaseEstimator | Module,
    data_format: DataFormat,
    adata: ad.AnnData | None = None,
    data_path: str | Path | None = None,
    device: torch.device | str | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
    eval_row_inds: np.ndarray | None = None,
) -> np.ndarray:
    """Run inference using a trained model.

    This is the internal inference function that handles the actual model execution.
    For the public API, use scxpand.run_inference() instead.

    Args:
        model_type: Type of model being used
        model: Trained model instance
        data_format: Data format specification for preprocessing
        adata: In-memory AnnData object (alternative to data_path)
        data_path: Path to data file (alternative to adata)
        device: Device for computation (neural networks only)
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        eval_row_inds: Specific cell indices to process (None for all)

    Returns:
        Array of prediction probabilities

    Raises:
        ValueError: If neither adata nor data_path is provided, or if model_type is unsupported
    """
    # Validate inputs
    if adata is None and data_path is None:
        raise ValueError("Either adata or data_path must be provided")

    # Ensure we have a ModelType enum
    try:
        model_type_enum = ensure_model_type(model_type)
    except ValueError:
        # Re-raise with the expected error message format for backward compatibility
        raise ValueError(f"Unsupported model_type for inference: '{model_type}'") from None

    logger.info(f"Running {model_type_enum.value} inference")

    # Route to appropriate inference function
    if model_type_enum == ModelType.AUTOENCODER:
        return run_ae_inference(
            model=model,
            batch_size=batch_size,
            data_format=data_format,
            adata=adata,
            data_path=data_path,
            device=device,
            num_workers=num_workers,
            eval_row_inds=eval_row_inds,
        )
    elif model_type_enum == ModelType.MLP:
        return run_mlp_inference(
            model=model,
            data_format=data_format,
            adata=adata,
            data_path=data_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_row_inds=eval_row_inds,
        )
    elif model_type_enum == ModelType.LIGHTGBM:
        return run_lightgbm_inference(
            model=model,
            data_format=data_format,
            adata=adata,
            data_path=data_path,
            eval_row_inds=eval_row_inds,
        )
    elif model_type_enum in [ModelType.LOGISTIC, ModelType.SVM]:
        return run_linear_inference(
            model=model,
            data_format=data_format,
            adata=adata,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_row_inds=eval_row_inds,
        )

    raise ValueError(f"Unsupported model_type for inference: '{model_type_enum.value}'")


def setup_inference_environment(
    model_type: ModelType | str, model_path: str | Path, device: str | None = None
) -> tuple[DataFormat, BaseEstimator | Module, torch.device | str]:
    """Setup the inference environment by loading data format, model, and determining device.

    Args:
        model_type: Type of model to setup
        model_path: Path to directory containing trained model and data format
        device: Device for computation (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detected

    Returns:
        Tuple of (data_format, model, device)

    Raises:
        FileNotFoundError: If data format file is not found
    """
    # Ensure we have a ModelType enum
    model_type_enum = ensure_model_type(model_type)
    model_path = Path(model_path)

    # Load data format
    data_format_path = model_path / "data_format.json"
    if not data_format_path.exists():
        raise FileNotFoundError(f"Data format file not found: {data_format_path}")

    logger.info(f"Loading data format from {data_format_path}")
    data_format = load_data_format(data_format_path)

    # Get device
    device = get_device()

    # Load model
    model = load_model(model_type=model_type_enum, model_path=model_path, device=device)

    logger.info(f"Inference environment ready: {model_type_enum.value} model on {device}")
    return data_format, model, device
