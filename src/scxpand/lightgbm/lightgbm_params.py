from dataclasses import dataclass
from enum import Enum

from scxpand.util.classes import BaseParams


class BoostingType(str, Enum):
    """Enumeration of supported LightGBM boosting types."""

    GBDT = "gbdt"
    DART = "dart"
    GOSS = "goss"
    RF = "rf"


class ObjectiveType(str, Enum):
    """Enumeration of supported LightGBM objective types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    LAMBDARANK = "lambdarank"


class MetricType(str, Enum):
    """Enumeration of supported LightGBM metric types."""

    BINARY_LOGLOSS = "binary_logloss"
    MULTICLASS_LOGLOSS = "multi_logloss"
    RMSE = "rmse"
    MAE = "mae"
    AUC = "auc"
    NDCG = "ndcg"


@dataclass
class LightGBMParams(BaseParams):
    """Parameters for LightGBM model.

    Note: When max_depth > 0, num_leaves should be <= 2^max_depth to avoid overfitting.
    """

    # Use log transform for gene expression
    use_log_transform: bool = True
    # Use z-score normalization for gene expression
    use_zscore_norm: bool = True
    # Number of leaves in the tree (should be < 2^max_depth)
    num_leaves: int = 31
    # Maximum depth of the tree (-1 means no limit, but can cause overfitting)
    max_depth: int = -1
    # Learning rate for the model
    learning_rate: float = 0.1
    # Number of trees in the model
    n_estimators: int = 100
    # Minimum number of samples required to split an internal node
    min_child_samples: int = 20
    # L1 regularization term on weights
    reg_alpha: float = 0.0
    # L2 regularization term on weights
    reg_lambda: float = 0.0
    # Random seed for reproducibility (consistent with hyperopt system)
    random_seed: int = 42
    # Force column-wise splits for better performance with many features
    force_col_wise: bool = True
    # Ensure deterministic results
    deterministic: bool = True
    # Class weight balancing (sklearn-specific, not native LightGBM)
    class_weight: str | None = "balanced"
    # Number of jobs for parallel processing
    n_jobs: int = -1
    # Feature fraction for each tree
    feature_fraction: float = 1.0
    # Bagging fraction for tree building
    bagging_fraction: float = 1.0
    # Minimum gain to perform a split
    min_split_gain: float = 0.0
    # Minimum sum of instance weight (hessian) needed in a child
    min_child_weight: float = 1e-3
    # Boosting type ('gbdt', 'dart', 'goss')
    boosting_type: BoostingType = BoostingType.GBDT
    # Objective function (for sklearn API, 'binary' is automatically set for binary classification)
    objective: ObjectiveType = ObjectiveType.BINARY
    # Metric to use for evaluation during training
    metric: MetricType = MetricType.BINARY_LOGLOSS
    # Verbosity level (-1: silent, 0: warning, 1: info)
    verbose: int = -1

    @classmethod
    def get_model_type(cls) -> str:
        """Return the model type identifier for this parameter class."""
        return "lightgbm"
