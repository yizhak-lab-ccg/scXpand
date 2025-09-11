"""Main public inference API for scXpand models.

This module provides the primary public interface for running inference
with any type of scXpand model (local, registry, or URL-based).
"""

from pathlib import Path
from typing import Union

import anndata as ad

from scxpand.core.inference_results import InferenceResults
from scxpand.core.prediction import run_prediction_pipeline
from scxpand.pretrained import fetch_model_and_run_inference
from scxpand.util.logger import get_logger


logger = get_logger()

# Default model used when no model source is specified
DEFAULT_MODEL_NAME = "pan_cancer_autoencoder"


def run_inference(
    data_path: Union[str, Path] | None = None,
    adata: ad.AnnData | None = None,
    model_path: Union[str, Path] | None = None,
    model_name: str | None = None,
    model_url: str | None = None,
    save_path: Union[str, Path] | None = None,
    batch_size: int = 1024,
    num_workers: int = 4,
    eval_row_inds=None,
    device: str | None = None,
) -> InferenceResults:
    """Main public API for running inference with scXpand models.

    This is the primary entry point for running inference with any type of scXpand model.
    It automatically detects the model source and routes to the appropriate inference pipeline.
    Supports local models, registry models, and external models via URL.
    Metrics are automatically computed when ground truth labels are available in the data.

    Args:
        data_path: Path to input data file (h5ad format) - alternative to adata
        adata: In-memory AnnData object - alternative to data_path
        model_path: Path to local trained model directory (for local models)
        model_name: Name of pre-trained model from registry (for registry models)
        model_url: Direct URL to model ZIP file (for any external model)
        model_type: Type of model (auto-detected if not provided)
        save_path: Directory to save prediction results (optional)
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        eval_row_inds: Specific cell indices to evaluate (None for all cells, only supported for local models)
        device: Device for computation (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detected

    Returns:
        InferenceResults: Structured results containing predictions, metrics (if available), and model info

    Raises:
        ValueError: If model source is not specified or multiple sources are specified
        ValueError: If neither data_path nor adata is provided
        FileNotFoundError: If specified files do not exist

    Examples:
        >>> import scxpand
        >>>
        >>> # Local model inference
        >>> results = scxpand.run_inference(
        ...     data_path="my_data.h5ad", model_path="results/mlp"
        ... )
        >>> print(f"Generated {len(results.predictions)} predictions")
        >>> if results.has_metrics:
        ...     print(f"AUROC: {results.get_auroc():.3f}")
        >>>
        >>> # Registry model inference
        >>> results = scxpand.run_inference(
        ...     data_path="my_data.h5ad", model_name="pan_cancer_autoencoder"
        ... )
        >>> if results.has_model_info:
        ...     print(f"Model type: {results.model_info.model_type}")
        >>>
        >>> # Direct URL inference (seamless model sharing!)
        >>> results = scxpand.run_inference(
        ...     data_path="my_data.h5ad",
        ...     model_url="https://your-platform.com/model.zip",
        ... )
        >>>
        >>> # In-memory inference with any model type
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad("my_data.h5ad")
        >>> results = scxpand.run_inference(
        ...     adata=adata, model_name="pan_cancer_autoencoder"
        ... )
    """
    # Validate data input
    if adata is None and data_path is None:
        raise ValueError("Either adata or data_path must be provided")

    # Count how many model sources are specified
    model_sources = [model_path, model_name, model_url]
    specified_sources = [s for s in model_sources if s is not None]

    if len(specified_sources) == 0:
        # No model source specified, use default registry model
        model_name = DEFAULT_MODEL_NAME
        logger.info(f"No model source specified, using default registry model: {model_name}")
    elif len(specified_sources) > 1:
        raise ValueError("Cannot specify multiple model sources. Use only one of: model_path, model_name, or model_url")

    # Route to appropriate inference method
    if model_path is not None:
        # Local model inference
        logger.info(f"Using local model from: {model_path}")
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
        return results
    else:
        # Pre-trained model inference (registry or URL)
        if model_name is not None:
            logger.info(f"Using registry model: {model_name}")
        else:
            logger.info(f"Using URL model: {model_url}")

        results = fetch_model_and_run_inference(
            model_name=model_name,
            model_url=model_url,
            data_path=data_path,
            adata=adata,
            save_path=save_path,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_row_inds=eval_row_inds,
            device=device,
        )
        return results
