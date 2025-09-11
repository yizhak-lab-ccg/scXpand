from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from scxpand.data_util.constants import IMPUTED_LABELS_COL, TISSUE_TYPE_COL
from scxpand.hyperopt.hyperopt_utils import report_optuna_trial_result
from scxpand.util.general_util import (
    metrics_dict_to_table,
    nested_dict_to_multiline_str,
    num2str,
    to_np,
)
from scxpand.util.logger import get_logger
from scxpand.util.plots import plot_roc_curve


if TYPE_CHECKING:
    import optuna


logger = get_logger()


# Helper function to safely compute the harmonic mean for a list of values.
# Returns 0 if any of the values is 0; if all values are NaN, returns NaN.
def safe_hmean(values: list[float]) -> float:
    non_nan = [val for val in values if not np.isnan(val)]
    if not non_nan:
        return np.nan
    if any(val == 0 for val in non_nan):
        return 0.0
    return len(non_nan) / np.sum([1.0 / val for val in non_nan])


# Helper function to compute basic metrics given labels and prediction probabilities.
def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    # Convert probabilities to binary predictions
    y_pred = np.where(y_pred_prob > threshold, 1, 0)
    acc = accuracy_score(y_true, y_pred)
    err = 1 - acc

    # Check for at least one sample of each class before computing AUROC
    if len(np.unique(y_true)) < 2:
        auroc = np.nan
    else:
        auroc = roc_auc_score(y_true=y_true, y_score=y_pred_prob)

    f1 = f1_score(y_true, y_pred, zero_division=np.nan)

    # Compute confusion matrix and derive FPR and FNR if possible
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    else:
        fpr, fnr = np.nan, np.nan

    # Compute the positive rate as the fraction of samples with label 1
    positives_rate = float(np.mean(y_true))

    # compute MSE
    rmse = np.sqrt(np.mean((y_true - y_pred_prob) ** 2))

    return {
        "error_rate": err,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "AUROC": auroc,
        "F1": f1,
        "RMSE": float(rmse),
        "positives_rate": positives_rate,
    }


# Main function to calculate overall and per-category metrics
def calculate_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred_prob: np.ndarray | torch.Tensor,
    obs_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Calculate comprehensive evaluation metrics for T-cell expansion prediction.

    Computes overall classification metrics plus stratified metrics by cell type
    and tissue type. Includes robust error handling for edge cases.

    Args:
        y_true: True binary labels (0=not expanded, 1=expanded).
        y_pred_prob: Predicted probabilities for expansion [0-1].
        obs_df: DataFrame with 'imputed_labels' and 'tissue_type' columns.
        threshold: Classification threshold for converting probabilities to predictions.

    Returns:
        Nested dictionary with 'overall' metrics and stratified metrics by
        tissue_type and imputed_labels combinations. Each contains:
        - AUROC: Area under ROC curve
        - F1: F1 score
        - error_rate: Classification error rate
        - false_positive_rate, false_negative_rate: FPR and FNR
        - RMSE: Root mean squared error
        - positives_rate: Fraction of positive samples


    """
    y_true = to_np(y_true)
    y_pred_prob = to_np(y_pred_prob)

    # Ensure the number of observations matches
    assert len(y_true) == len(y_pred_prob) == len(obs_df)

    # Compute overall metrics using the helper function
    overall = compute_basic_metrics(y_true, y_pred_prob, threshold)

    metric_names = list(overall.keys())

    # Compute per-category metrics for each unique (imputed_labels, tissue_type) pair
    per_category = {}

    # Check if stratification columns are available
    stratification_columns = [IMPUTED_LABELS_COL, TISSUE_TYPE_COL]
    available_stratification_columns = [col for col in stratification_columns if col in obs_df.columns]

    if len(available_stratification_columns) == len(stratification_columns):
        # Both columns available - compute stratified metrics
        imputed_labels = obs_df[IMPUTED_LABELS_COL].to_numpy()
        tissue_types = obs_df[TISSUE_TYPE_COL].to_numpy()

        # Get unique combinations of imputed_labels and tissue_type
        unique_combinations = {}
        for i in range(len(imputed_labels)):
            key = (str(imputed_labels[i]), str(tissue_types[i]))
            if key not in unique_combinations:
                unique_combinations[key] = []
            unique_combinations[key].append(i)

        # Calculate metrics for each category
        for (imp_label, tissue), idx_list in unique_combinations.items():
            indices = np.array(idx_list)
            y_true_grp = y_true[indices]
            y_pred_prob_grp = y_pred_prob[indices]
            grp_metrics = compute_basic_metrics(y_true_grp, y_pred_prob_grp, threshold)

            key = f"{imp_label}__{tissue}"
            per_category[key] = grp_metrics
    else:
        # Missing stratification columns - skip stratified metrics
        logger.warning(
            f"Cannot compute stratified metrics: missing columns {set(stratification_columns) - set(available_stratification_columns)}"
        )

    # Aggregate per-category metrics: arithmetic average and harmonic mean.
    avg_category = {}
    harm_category = {}

    for metric in metric_names:
        values = [
            group_metrics[metric] for group_metrics in per_category.values() if not np.isnan(group_metrics[metric])
        ]
        if values:
            avg_category[metric] = float(np.mean(values))
            harm_category[metric] = safe_hmean(values)
        else:
            avg_category[metric] = np.nan
            harm_category[metric] = np.nan

    results = overall.copy()
    results.update(per_category)
    results["average"] = avg_category
    results["harmonic_avg"] = harm_category

    return results


def evaluate_and_save(
    y_true: np.ndarray | torch.Tensor,
    y_pred_prob: np.ndarray | torch.Tensor,
    obs_df: pd.DataFrame,
    eval_name: str = "",
    save_path: Path | None = None,
    plots_dir: Path | None = None,
    threshold: float = 0.5,
    trial: "optuna.Trial | None" = None,
    score_metric: str = "harmonic_avg/AUROC",
    use_table_format: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate model performance, save results, optionally create plots, and report Optuna trial results if provided.

    Args:
        y_true: Ground truth labels [n_eval] (0 or 1)
        y_pred_prob: Predicted probabilities [n_eval] in range [0, 1]
        obs_df: DataFrame containing observation data
        eval_name: Name for the evaluation (used in file names)
        save_path: Path to save results
        plots_dir: Optional path to save plots
        threshold: Classification threshold
        trial: Optional Optuna trial for reporting
        score_metric: Metric key to report to Optuna
        use_table_format: If True, display metrics in table format

    Returns:
        dict: Dictionary containing the evaluation metrics
    """
    results = calculate_metrics(y_true=y_true, y_pred_prob=y_pred_prob, obs_df=obs_df, threshold=threshold)

    # Log metrics using the specified format
    if use_table_format:
        table_title = f"Evaluation Results ({eval_name})"
        table_output = metrics_dict_to_table(results, title=table_title)
        logger.info(table_output)
    else:
        logger.info(f"({eval_name}) AUROC: {num2str(results['AUROC'])}")

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f"results_{eval_name}.txt"
        with file_path.open("w") as f:
            if use_table_format:
                f.write(metrics_dict_to_table(results, title=f"Results ({eval_name})"))
            else:
                f.write(nested_dict_to_multiline_str(results))
        logger.info(f"({eval_name}) Results saved to {file_path}")

        # Convert results to pandas DataFrame for saving
        results_df = obs_df.copy()
        results_df["y_pred_prob"] = y_pred_prob
        results_df["y_pred"] = (y_pred_prob > threshold).astype(int)
        results_df["y_true"] = y_true
        csv_path = save_path / f"results_table_{eval_name}.csv"
        results_df.to_csv(csv_path)
        logger.info(f"({eval_name}) Per-cell meta-data saved to {csv_path}")

    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_roc_curve(
            labels=y_true,
            probs_pred=y_pred_prob,
            plot_save_dir=plots_dir,
            plot_name=f"roc_curve_{eval_name}",
        )
        logger.info(f"({eval_name}) Plots saved to {plots_dir}")

    if trial is not None:
        report_optuna_trial_result(trial=trial, results=results, score_metric=score_metric)

    return results


def get_score_from_nested_dict(nested_metrics_dict: dict, metric_name: str) -> float:
    """Extract a metric value from a nested dictionary.

    Args:
        nested_metrics_dict: A nested dictionary of metrics (e.g. {"dev": {"auc": 0.9}}).
        metric_name: A string of the metric name to get (e.g. "dev/auc").

    Returns:
        The score value (e.g. 0.9).
    """
    keys = metric_name.split("/")
    val = nested_metrics_dict
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return np.nan
    return val
