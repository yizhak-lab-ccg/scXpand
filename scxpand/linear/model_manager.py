"""Model initialization and state management for linear models."""

from pathlib import Path

import numpy as np

from sklearn.linear_model import SGDClassifier

from scxpand.linear.linear_params import LinearClassifierParam
from scxpand.util.general_util import save_json_data
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import BEST_MODEL_INFO_FILE


logger = get_logger()


class ModelManager:
    """Handles model initialization and state management."""

    @staticmethod
    def initialize_model(prm: LinearClassifierParam, y_train: np.ndarray) -> SGDClassifier:
        """Initialize SGDClassifier with proper parameters."""
        logger.info("Initializing SGDClassifier...")

        sgd_class_weight_param = prm.class_weight
        if sgd_class_weight_param == "balanced_dict":
            sgd_class_weight_param = "balanced"
        elif sgd_class_weight_param == "None":
            sgd_class_weight_param = None
        elif sgd_class_weight_param == "balanced":
            from sklearn.utils.class_weight import compute_class_weight  # noqa: PLC0415

            unique_classes = np.unique(y_train)
            computed_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
            # Convert to dict format expected by SGDClassifier
            sgd_class_weight_param = dict(zip(unique_classes, computed_weights))

        loss = "log_loss" if prm.model_type == "logistic" else "hinge"

        model = SGDClassifier(
            loss=loss,
            penalty=prm.penalty,
            alpha=prm.alpha,
            l1_ratio=prm.l1_ratio,
            max_iter=prm.max_iter,  # Set to 1 since we're using partial_fit in a loop
            tol=prm.tol,
            class_weight=sgd_class_weight_param,
            random_state=prm.random_seed,
            warm_start=prm.warm_start,  # Enable warm start for incremental training
            learning_rate=prm.learning_rate,
            eta0=prm.eta0,
            power_t=prm.power_t,
            average=prm.average,
            n_iter_no_change=prm.n_iter_no_change,
            validation_fraction=prm.validation_fraction,
            early_stopping=False,  # Must be False when using partial_fit()
            fit_intercept=prm.fit_intercept,
            shuffle=prm.shuffle,
        )

        return model

    @staticmethod
    def save_model_state(
        model: SGDClassifier,
        current_score: float,
        epoch: int,
        dev_set_metrics: dict,
    ) -> dict:
        """Save the current model state as the best model."""
        return {
            "coef_": model.coef_.copy(),
            "intercept_": model.intercept_.copy(),
            "classes_": model.classes_.copy(),
            "n_features_in_": model.n_features_in_,
            "t_": model.t_,
            "epoch": epoch,
            "score": current_score,
            "dev_set_metrics": dev_set_metrics,
        }

    @staticmethod
    def load_model_state(
        model: SGDClassifier,
        best_model_state: dict,
        base_save_dir: Path,
        score_metric: str,
    ) -> None:
        """Load the best model state into the SGDClassifier.

        Args:
            model: SGDClassifier to load state into
            best_model_state: Dictionary containing the best model state
            base_save_dir: Directory containing saved model files
            score_metric: Metric name used for scoring
        """
        logger.info(
            f"Loading best model from epoch {best_model_state['epoch'] + 1} with score {best_model_state['score']:.4f}"
        )

        model.coef_ = best_model_state["coef_"]
        model.intercept_ = best_model_state["intercept_"]
        model.classes_ = best_model_state["classes_"]
        model.n_features_in_ = best_model_state["n_features_in_"]
        model.t_ = best_model_state["t_"]

        # Save best model info
        best_model_info = {
            "best_epoch": best_model_state["epoch"],
            "best_score": best_model_state["score"],
            "score_metric": score_metric,
        }
        save_json_data(data=best_model_info, save_path=base_save_dir / BEST_MODEL_INFO_FILE)
