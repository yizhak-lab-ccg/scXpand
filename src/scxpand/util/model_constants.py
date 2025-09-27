"""Constants for model file names and paths.

This module defines consistent naming conventions for model files across the codebase
to prevent inconsistencies and magic strings.
"""

# Model checkpoint file names
BEST_CHECKPOINT_FILE = "best_ckpt.pt"
LAST_CHECKPOINT_FILE = "last_ckpt.pt"

# Model configuration files
PARAMETERS_FILE = "parameters.json"
DATA_FORMAT_FILE = "data_format.json"
MODEL_TYPE_FILE = "model_type.txt"

# Results and metadata files
BEST_MODEL_INFO_FILE = "best_model_info.json"
BEST_MODEL_METRICS_FILE = "best_model_dev_set_metrics.json"
RUN_INFO_FILE = "run_info.json"
SUMMARY_INFO_FILE = "summary_info.json"
STUDY_INFO_FILE = "info.json"

# Data files
DATA_FORMAT_NPZ_FILE = "data_format.npz"
DATA_SPLITS_FILE = "data_splits.npz"

# Results files
RESULTS_DEV_FILE = "results_dev.txt"
RESULTS_TABLE_DEV_FILE = "results_table_dev.csv"

# Sklearn model files (for non-PyTorch models)
SKLEARN_MODEL_FILE = "model.joblib"
