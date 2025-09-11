"""Internal inference API for pre-trained models.

This module provides internal functions for running inference with pre-trained models,
handling automatic model downloading and loading. For external use, use scxpand.run_inference().
"""

from pathlib import Path
from typing import Union

import anndata as ad

from scxpand.core.inference_results import InferenceResults, ModelInfo
from scxpand.core.prediction import run_prediction_pipeline
from scxpand.util.logger import get_logger
from scxpand.util.model_type import load_model_type

from .download_manager import download_pretrained_model
from .model_registry import get_pretrained_model_info


logger = get_logger()


def fetch_model_and_run_inference(
    model_name: str | None = None,
    model_url: str | None = None,
    data_path: Union[str, Path] | None = None,
    adata: ad.AnnData | None = None,
    save_path: Union[str, Path] | None = None,
    batch_size: int | None = None,
    num_workers: int = 4,
    eval_row_inds=None,
    device: str | None = None,
) -> InferenceResults:
    """Internal function for running inference with pre-trained models.

    This is an internal function that handles automatic model downloading, loading,
    and inference in a single call. Works with both file-based and in-memory data.
    For external use, use scxpand.run_inference() instead.

    Args:
        model_name: Name of pre-trained model from registry (alternative to model_url)
        model_url: Direct URL to model ZIP file (alternative to model_name)
        data_path: Path to input data file (h5ad format) - alternative to adata
        adata: In-memory AnnData object - alternative to data_path
        save_path: Directory to save prediction results (optional)
        batch_size: Batch size for inference (uses model default if None)
        num_workers: Number of workers for data loading
        eval_row_inds: Specific cell indices to evaluate (None for all cells)
        device: Device for computation (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detected

    Returns:
        InferenceResults: Structured results containing predictions, metrics (if available), and model info

    Raises:
        ValueError: If neither model_name nor model_doi provided, or neither data_path nor adata provided
        FileNotFoundError: If data_path does not exist

    Note:
        This is an internal function. For external use, use scxpand.run_inference() instead.
    """
    # Validate inputs
    if model_name is None and model_url is None:
        raise ValueError("Either model_name or model_url must be provided")

    if model_name is not None and model_url is not None:
        raise ValueError("Cannot specify both model_name and model_url. Use one or the other.")

    if adata is None and data_path is None:
        raise ValueError("Either adata or data_path must be provided")

    if data_path is not None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    # Handle model information and download
    if model_name is not None:
        # Use registry model
        model_info = get_pretrained_model_info(model_name)
        logger.info(f"Using registry model: {model_info.name}")
        logger.info(f"Model version: {model_info.version}")
        logger.info("Model type will be auto-detected from model_type.txt")

        # Download using registry
        model_path = download_pretrained_model(model_name=model_name)

        # Model type will be auto-detected from model_type.txt file
    else:
        # Use direct URL
        logger.info(f"Using direct URL: {model_url}")

        # Download using URL
        model_path = download_pretrained_model(model_url=model_url)

        # For URL models, model type will be auto-detected from model_type.txt file
        model_info = None

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = 1024  # Default batch size

    logger.info(f"Running inference with batch size: {batch_size}")

    # Run the unified prediction pipeline
    results = run_prediction_pipeline(
        model_path=model_path,
        data_path=data_path,
        adata=adata,
        save_path=save_path,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_row_inds=eval_row_inds,
        device=device,
    )

    # Load actual model type from model_type.txt file
    actual_model_type = load_model_type(model_path)

    # Create model info object
    if model_info is not None:
        # Registry model
        model_info_obj = ModelInfo(
            model_name=model_info.name,
            model_type=actual_model_type,
            version=model_info.version,
            source="registry",
        )
    else:
        # URL model
        model_info_obj = ModelInfo(
            model_type=actual_model_type,
            source="external_url",
            url=model_url,
        )

    logger.info("Inference completed successfully")
    return InferenceResults(
        predictions=results.predictions,
        metrics=results.metrics,
        model_info=model_info_obj,
    )
