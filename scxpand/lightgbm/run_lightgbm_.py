from pathlib import Path

import anndata as ad
import joblib
import lightgbm as lgb
import numpy as np
import optuna

from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight

from scxpand.core.evaluation import evaluate_predictions_and_save
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.prepare_data_for_train import prepare_data_for_training
from scxpand.data_util.transforms import extract_is_expanded, load_and_preprocess_data_numpy, preprocess_expression_data
from scxpand.lightgbm.lightgbm_params import LightGBMParams
from scxpand.util.classes import ModelType
from scxpand.util.general_util import save_params, set_seed
from scxpand.util.logger import get_logger
from scxpand.util.model_constants import SKLEARN_MODEL_FILE


logger = get_logger()


def _prepare_data_for_lightgbm_inference(
    data_format: DataFormat,
    adata: ad.AnnData | None = None,
    data_path: str | Path | None = None,
    eval_row_inds: np.ndarray | None = None,
) -> np.ndarray:
    """Prepare and preprocess data for LightGBM inference.

    Handles gene reordering to match the training data format and applies
    the same preprocessing pipeline used during training.

    Args:
        data_format: Data format specification for preprocessing
        adata: AnnData object containing gene expression data (alternative to data_path)
        data_path: Path to data file (alternative to adata)
        eval_row_inds: Indices of rows to evaluate (if None, uses all rows)

    Returns:
        Preprocessed expression matrix ready for model inference
    """
    if adata is not None:
        # In-memory data path
        logger.info(f"Input data has {adata.n_vars} genes, model expects {data_format.n_genes} genes")
        source_adata = adata
    else:
        # File-based data path
        adata_file = ad.read_h5ad(data_path, backed="r")
        logger.info(f"Input file has {adata_file.n_vars} genes, model expects {data_format.n_genes} genes")

        # Load into memory for gene reordering (backed mode doesn't support copy operations)
        source_adata = adata_file.to_memory()

        # Close the backed file
        if hasattr(adata_file, "isbacked") and adata_file.isbacked:
            adata_file.file.close()

    # Reorder genes to match the training data format
    adata_reordered = data_format.prepare_adata_for_training(source_adata, reorder_genes=True)
    logger.info(f"After gene reordering: {adata_reordered.n_vars} genes")

    # Extract the subset of data for evaluation
    if eval_row_inds is not None:
        X_raw = adata_reordered[eval_row_inds].X
    else:
        X_raw = adata_reordered.X

    # Apply the same preprocessing transformations that were used during training
    return preprocess_expression_data(X=X_raw, data_format=data_format)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights."""
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return [class_weights[yi] for yi in y]


def run_lightgbm_training(
    base_save_dir: str | Path,
    prm: LightGBMParams,
    data_path: str,
    dev_ratio: float = 0.2,
    trial: optuna.Trial | None = None,
    score_metric: str = "harmonic_avg/AUROC",
    resume: bool = False,  # noqa: ARG001
) -> dict[str, dict[str, float]]:
    """Train a LightGBM model for gene expression classification.

    Args:
        base_save_dir: Directory to save model and results
        prm: LightGBM parameters
        data_path: Path to data file
        dev_ratio: Ratio of data to use for validation
        trial: Optuna trial object for hyperparameter optimization
        score_metric: Metric to use for scoring
        resume: Whether to resume from existing checkpoint (not implemented for LightGBM)

    Returns:
        Dictionary containing evaluation results
    """
    base_save_dir = Path(base_save_dir)
    set_seed(prm.random_seed)
    save_params(params=prm, save_dir=base_save_dir)

    data_bundle = prepare_data_for_training(
        data_path=data_path,
        use_log_transform=prm.use_log_transform,
        use_zscore_norm=prm.use_zscore_norm,
        save_dir=base_save_dir,
        dev_ratio=dev_ratio,
        rand_seed=prm.random_seed,
        resume=False,
    )
    adata = data_bundle.adata
    data_format = data_bundle.data_format
    row_inds_train = data_bundle.row_inds_train
    row_inds_dev = data_bundle.row_inds_dev

    # Apply preprocessing to get training data
    X_train = load_and_preprocess_data_numpy(data_path=data_path, data_format=data_format, row_indices=row_inds_train)

    # Apply preprocessing to get validation data
    X_dev = load_and_preprocess_data_numpy(data_path=data_path, data_format=data_format, row_indices=row_inds_dev)

    # Extract labels
    y_train = extract_is_expanded(adata.obs.iloc[row_inds_train])

    if prm.class_weight == "balanced":
        sample_weights = compute_sample_weights(y_train)
    else:
        sample_weights = None

    logger.info("Training LightGBM model...")
    model = lgb.LGBMClassifier(
        num_leaves=prm.num_leaves,
        max_depth=prm.max_depth,
        learning_rate=prm.learning_rate,
        n_estimators=prm.n_estimators,
        min_child_samples=prm.min_child_samples,
        reg_alpha=prm.reg_alpha,
        reg_lambda=prm.reg_lambda,
        random_state=prm.random_seed,
        force_col_wise=prm.force_col_wise,
        deterministic=prm.deterministic,
        n_jobs=prm.n_jobs,
        feature_fraction=prm.feature_fraction,
        bagging_fraction=prm.bagging_fraction,
        min_split_gain=prm.min_split_gain,
        min_child_weight=prm.min_child_weight,
        boosting_type=prm.boosting_type.value,
        objective=prm.objective.value,
        verbose=prm.verbose,
    )
    model.fit(
        X=X_train,
        y=y_train,
        sample_weight=sample_weights,
    )
    # Predict on validation set
    logger.info("Predicting on validation set...")
    y_dev_pred_prob = model.predict_proba(X_dev)[:, 1]

    # Use utility function for evaluation and saving
    results = evaluate_predictions_and_save(
        y_pred_prob=y_dev_pred_prob,
        obs_df=adata.obs.iloc[row_inds_dev],
        model_type=ModelType.LIGHTGBM,
        save_path=base_save_dir,
        eval_name="dev",
        score_metric=score_metric,
        trial=trial,
    )
    logger.info("Saving model...")
    joblib.dump(model, base_save_dir / SKLEARN_MODEL_FILE)
    return results


def run_lightgbm_inference(
    model: BaseEstimator,
    data_format: DataFormat,
    adata: ad.AnnData | None = None,
    data_path: str | Path | None = None,
    eval_row_inds: np.ndarray | None = None,
) -> np.ndarray:
    """Run inference using a trained LightGBM model.

    Args:
        model: Trained LightGBM model
        data_format: Data format specification for preprocessing
        adata: AnnData object containing gene expression data (alternative to data_path)
        data_path: Path to data file (alternative to adata)
        eval_row_inds: Indices of rows to evaluate (if None, uses all rows)

    Returns:
        Array of prediction probabilities for the positive class
    """
    # Validate inputs
    if adata is None and data_path is None:
        raise ValueError("Either adata or data_path must be provided")

    # Prepare and preprocess data using utility function
    X = _prepare_data_for_lightgbm_inference(
        data_format=data_format,
        adata=adata,
        data_path=data_path,
        eval_row_inds=eval_row_inds,
    )

    logger.info(f"Running inference with {type(model).__name__} model...")

    # Check if the model uses hinge loss (SVM)
    has_predict_proba = hasattr(model, "predict_proba") and callable(model.predict_proba)
    is_hinge_loss = getattr(model, "loss", None) == "hinge"

    # For SVM (hinge loss), transform decision function to pseudo-probabilities
    if is_hinge_loss or not has_predict_proba:
        # Get decision function values
        decisions = model.decision_function(X)

        # Handle 1D or 2D decision function outputs
        if decisions.ndim == 1:
            # For binary classification, convert to probability using sigmoid
            predictions = 1 / (1 + np.exp(-decisions))
        else:
            # For multi-class, apply softmax transformation
            # Subtract max for numerical stability
            exp_scores = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Get the probability for positive class (assuming binary classification)
            predictions = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    else:
        # Get probability of the positive class (index 1)
        predictions = model.predict_proba(X)[:, 1]

    logger.info(f"Scikit-learn inference complete. Predictions shape: {predictions.shape}")
    return predictions
