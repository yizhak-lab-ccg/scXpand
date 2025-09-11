import json

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import auc, roc_curve

from scxpand.util.general_util import to_np
from scxpand.util.logger import get_logger


logger = get_logger()
# Global plotting configuration
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "figure.titlesize": 16,
        "figure.figsize": (10, 8),
    }
)


def plot_roc_curve(
    labels,
    probs_pred,
    show_plot: bool = False,
    plot_save_dir: Path | None = None,
    plot_name: str = "roc_curve",
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> float:
    """Plot ROC curve for binary classification and calculate AUROC.

    Creates a publication-ready ROC curve plot showing model performance
    across all classification thresholds. Optionally saves the plot to disk.

    Args:
        labels: True binary labels (0 or 1).
        probs_pred: Predicted probabilities [0-1] from model.
        show_plot: Whether to display plot interactively.
        plot_save_dir: Directory to save plot. If None, plot is not saved.
        plot_name: Filename for saved plot (without extension).
        title: Plot title text.

    Returns:
        AUROC score (Area Under the ROC Curve).


    """
    probs_pred = to_np(probs_pred).astype(float)
    labels = to_np(labels).astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        logger.info(f"No positive labels found for {plot_name}. Returning NaN and skipping plot.")
        return np.nan

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=probs_pred)
    auroc = auc(fpr, tpr)

    if plot_save_dir or show_plot:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auroc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        if plot_save_dir:
            save_path = Path(plot_save_dir) / f"{plot_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close("all")

    return auroc


def plot_roc_curves_per_strata(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    obs_df: pd.DataFrame,
    strata_columns: list[str],
    show_plot: bool = True,
    plot_save_dir: Path | None = None,
    save_results: bool = True,
    max_cols: int = 2,
) -> dict[str, float]:
    """Plot ROC curves for each stratum in a grid of subplots and calculate AUROC scores.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_prob: Predicted probabilities [0-1] from model
        obs_df: DataFrame containing observation data with stratification columns
        strata_columns: List of column names to use for stratification.
        show_plot: Whether to display plots interactively
        plot_save_dir: Directory to save plots. If None, plots are not saved.
        save_results: Whether to save AUROC results to JSON file
        max_cols: Maximum number of columns in the plot grid

    Returns:
        Dictionary mapping stratum names to AUROC scores
    """

    # Check which columns are available
    available_cols = [col for col in strata_columns if col in obs_df.columns]
    if not available_cols:
        logger.warning(f"None of the stratification columns {strata_columns} are available. Skipping per-strata plots.")
        return {}

    if len(available_cols) < len(strata_columns):
        logger.warning(f"Some stratification columns missing: {set(strata_columns) - set(available_cols)}")

    # Create strata from available columns
    strata_df = obs_df[available_cols]
    strata = strata_df.astype(str).agg(" - ".join, axis=1)
    strata.index = obs_df.index

    unique_strata = strata.unique()
    auroc_per_strata = {}

    # Filter out strata with insufficient data upfront
    valid_strata = []
    for stratum_name in unique_strata:
        mask = strata == stratum_name
        labels_strata = y_true[mask]
        preds_strata = y_pred_prob[mask]

        # Skip if only one class or invalid predictions
        if len(np.unique(labels_strata)) >= 2 and not (np.isnan(preds_strata).any() or np.isinf(preds_strata).any()):
            valid_strata.append(stratum_name)

    if not valid_strata:
        logger.warning("No valid strata found for plotting. Skipping per-strata plots.")
        return {}

    # Calculate grid dimensions
    n_strata = len(valid_strata)
    n_cols = min(max_cols, n_strata)
    n_rows = (n_strata + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot grid if we have data to plot
    if plot_save_dir or show_plot:
        _fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # Handle single subplot case
        if n_strata == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

    # Plot ROC curve for each valid stratum
    for i, stratum_name in enumerate(valid_strata):
        mask = strata == stratum_name
        labels_strata = y_true[mask]
        preds_strata = y_pred_prob[mask]

        # Calculate ROC curve
        probs_pred = to_np(preds_strata).astype(float)
        labels = to_np(labels_strata).astype(int)

        fpr, tpr, _ = roc_curve(y_true=labels, y_score=probs_pred)
        auroc = auc(fpr, tpr)
        auroc_per_strata[stratum_name] = auroc

        # Plot on subplot if creating plots
        if plot_save_dir or show_plot:
            ax = axes[i]
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auroc:.2f}")
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve for {stratum_name}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

    # Hide empty subplots
    if plot_save_dir or show_plot:
        for i in range(n_strata, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if plot_save_dir:
            plot_save_dir = Path(plot_save_dir)
            plot_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = plot_save_dir / "roc_curves_per_strata.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()

        plt.close()

    # Save results if requested
    if save_results and plot_save_dir and auroc_per_strata:
        plot_save_dir = Path(plot_save_dir)
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        per_strata_results_path = plot_save_dir / "auroc_per_strata.json"
        with open(per_strata_results_path, "w") as f:
            json.dump(auroc_per_strata, f, indent=4)

    return auroc_per_strata
