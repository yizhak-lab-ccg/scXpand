"""Core evaluation logic for scXpand models.

This module contains the domain logic for evaluating model predictions
and orchestrating the evaluation pipeline. It depends only on utilities
and has no dependencies on model-specific training code.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd

from scxpand.data_util.dataset import EXPANSION_COL
from scxpand.data_util.transforms import extract_is_expanded
from scxpand.util.general_util import format_float
from scxpand.util.io import save_predictions_to_csv
from scxpand.util.logger import get_logger
from scxpand.util.metrics import evaluate_and_save


if TYPE_CHECKING:
    from scxpand.util.classes import ModelType

logger = get_logger()


def evaluate_predictions_and_save(
    y_pred_prob: np.ndarray,
    obs_df: pd.DataFrame,
    model_type: "ModelType | str",
    save_path: Path,
    eval_name: str = "dev",
    score_metric: str = "harmonic_avg/AUROC",
    trial: optuna.Trial | None = None,
) -> dict:
    """Evaluate predictions against ground truth and save results.

    This is the main evaluation function that coordinates:
    1. Extracting ground truth labels from observation data
    2. Saving predictions to CSV file
    3. Computing and saving evaluation metrics
    4. Logging results

    Args:
        y_pred_prob: Predicted probabilities from model
        obs_df: DataFrame with cell metadata and ground truth labels
        model_type: Type of model used for predictions
        save_path: Directory to save evaluation results
        eval_name: Name for this evaluation (used in filenames)
        score_metric: Primary metric to optimize/report
        trial: Optional Optuna trial for hyperparameter optimization

    Returns:
        Dictionary containing evaluation metrics and scores, or empty dict if evaluation skipped
    """
    # Check if required columns are present for evaluation
    required_columns = [EXPANSION_COL]
    missing_columns = [col for col in required_columns if col not in obs_df.columns]

    if missing_columns:
        logger.info(f"Cannot evaluate metrics: missing required columns {missing_columns}")

        # Still save predictions even if we can't evaluate
        save_predictions_to_csv(
            predictions=y_pred_prob,
            obs_df=obs_df,
            model_type=model_type,
            save_path=save_path,
        )

        return {}

    # Extract ground truth labels
    y_true = extract_is_expanded(obs_df)

    # Save predictions to CSV file
    save_predictions_to_csv(
        predictions=y_pred_prob,
        obs_df=obs_df,
        model_type=model_type,
        save_path=save_path,
    )

    # Compute and save evaluation metrics
    results = evaluate_and_save(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        obs_df=obs_df,
        eval_name=eval_name,
        save_path=save_path,
        plots_dir=save_path / "plots",
        score_metric=score_metric,
        trial=trial,
    )

    # Log completion
    auroc = results.get("AUROC", "N/A")
    auroc_formatted = format_float(auroc) if isinstance(auroc, float) else auroc
    logger.info(f"Evaluation completed for {eval_name}. AUROC: {auroc_formatted}")

    return results
