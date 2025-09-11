"""Core prediction orchestration for scXpand models.

This module contains the high-level prediction pipeline that coordinates:
- Model loading and setup
- Data preprocessing and inference
- Results evaluation and saving

It serves as the domain service layer that orchestrates different components
without being tightly coupled to any specific implementation.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np

from scxpand.core.evaluation import evaluate_predictions_and_save
from scxpand.core.inference_results import InferenceResults
from scxpand.util import inference_utils
from scxpand.util.io import ensure_directory_exists
from scxpand.util.logger import get_logger
from scxpand.util.model_type import infer_model_type_from_parameters


if TYPE_CHECKING:
    from scxpand.util.classes import ModelType

logger = get_logger()


def run_prediction_pipeline(
    model_path: str | Path,
    model_type: "ModelType | str | None" = None,
    adata: ad.AnnData | None = None,
    data_path: str | Path | None = None,
    save_path: str | Path | None = None,
    batch_size: int = 1024,
    num_workers: int = 0,
    eval_row_inds: np.ndarray | None = None,
    device: str | None = None,
) -> InferenceResults:
    """Complete prediction pipeline from model loading to evaluation.

    This is the main orchestration function that coordinates the entire
    prediction workflow. It follows the dependency inversion principle
    by depending on abstractions (interfaces) rather than concrete implementations.

    Args:
        model_path: Path to directory containing the trained model
        model_type: Type of model to use for prediction. If None, automatically
        detected from model_type.txt file in model_path
        adata: In-memory AnnData object (alternative to data_path)
        data_path: Path to data file (alternative to adata)
        save_path: Directory to save prediction results
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        eval_row_inds: Specific cell indices to evaluate (None for all)
        device: Device for computation (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detected

    Returns:
        InferenceResults: Structured results containing predictions and metrics (if available)

    Raises:
        ValueError: If neither adata nor data_path is provided
        FileNotFoundError: If model_path doesn't exist
    """
    # Validate inputs
    if adata is None and data_path is None:
        raise ValueError("Either adata or data_path must be provided")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Auto-detect model type if not provided
    if model_type is None:
        model_type = infer_model_type_from_parameters(model_path)
        logger.info(f"Loaded model type: {model_type}")

    # Setup save path
    if save_path is None:
        save_path = model_path / "predictions"
    else:
        save_path = Path(save_path)

    ensure_directory_exists(save_path)

    # Log pipeline start
    logger.info(f"Starting prediction pipeline for {model_type}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Save path: {save_path}")

    # Setup inference environment (load model, data format, device)
    data_format, model, device = inference_utils.setup_inference_environment(
        model_type=model_type, model_path=model_path, device=device
    )

    # Run inference
    logger.info("Running model inference...")
    y_pred_prob = inference_utils.run_model_inference(
        model_type=model_type,
        model=model,
        data_format=data_format,
        adata=adata,
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_row_inds=eval_row_inds,
    )

    logger.info(f"Inference completed. Generated {len(y_pred_prob)} predictions.")

    # Evaluate and save results if ground truth is available
    metrics_results = None
    logger.info("Evaluating predictions...")

    # Load data if not provided (for evaluation)
    if adata is None:
        adata = ad.read_h5ad(data_path, backed="r")

    # Get observation data for evaluation
    if eval_row_inds is not None:
        obs_df = adata.obs.iloc[eval_row_inds]
    else:
        obs_df = adata.obs

    # Run evaluation pipeline
    metrics_results = evaluate_predictions_and_save(
        y_pred_prob=y_pred_prob,
        obs_df=obs_df,
        model_type=model_type,
        save_path=save_path,
        eval_name="prediction",
        score_metric="harmonic_avg/AUROC",
    )

    logger.info("Prediction pipeline completed successfully")
    return InferenceResults(
        predictions=y_pred_prob,
        metrics=metrics_results,
        model_info=None,  # Local models don't have model_info
    )
