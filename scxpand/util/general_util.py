import dataclasses
import json
import random
import subprocess
import time

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar

import numpy as np
import pandas as pd
import structlog
import torch

from tabulate import tabulate

from scxpand.util.classes import BaseParams
from scxpand.util.logger import get_logger
from scxpand.util.model_type import save_model_type


logger = get_logger()

T = TypeVar("T", bound=BaseParams)


def convert_enums_to_values(obj: Any) -> Any:
    """Recursively convert enum objects to their string values for JSON serialization and logging.

    Args:
        obj: Any object that might contain enums (dict, list, tuple, or individual values)

    Returns:
        Object with all enums converted to their .value strings
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {key: convert_enums_to_values(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        converted = [convert_enums_to_values(item) for item in obj]
        return type(obj)(converted)  # Preserve original type (list or tuple)
    else:
        return obj


def save_json_data(data: dict[str, Any], save_path: Path | str):
    """Save arbitrary dictionary data to a JSON file.

    Args:
        data: Dictionary of data to save
        save_path: Full path to the JSON file to create
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert enums to their string values for JSON serialization
    serializable_data = convert_enums_to_values(data)

    with save_path.open("w") as f:
        json.dump(serializable_data, f, indent=4)

    logger.info(f"Saved data to {save_path}")


def save_params(params: BaseParams, save_dir: Path | str):
    """Save parameters to a json file and save model type from parameter object.

    Args:
        params: Parameter object (must inherit from BaseParams and have get_model_type method)
        save_dir: Directory where to save the parameters
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert parameter object to dictionary
    params_dict = params.__dict__

    # Convert enums to their string values for JSON serialization
    serializable_params = convert_enums_to_values(params_dict)

    with (save_dir / "parameters.json").open("w") as f:
        json.dump(serializable_params, f, indent=4)

    logger.info(f"Saved parameters to {save_dir / 'parameters.json'}")
    logger.info(f"Parameters:\n{nested_dict_to_multiline_str(serializable_params)}")

    # Get model type from parameter object
    try:
        model_type = params.get_model_type()
        save_model_type(model_type, save_dir)
    except AttributeError:
        logger.warning(f"Parameter object {type(params)} does not have get_model_type method")
        logger.warning("Model type file will not be created")


def load_params(save_path: Path | str) -> dict:
    """Load model parameters from a saved JSON file.

    Loads hyperparameters and configuration from training results directory.

    Args:
        save_path: Path to directory containing 'parameters.json' file.

    Returns:
        Dictionary containing all saved parameters.

    Example:
        >>> params = load_params("results/model_001")
        >>> print(f"Learning rate: {params['init_learning_rate']}")
    """
    save_path = Path(save_path)
    with (save_path / "parameters.json").open("r") as f:
        return json.load(f)


# Add this helper function near the top of the file (after the imports)
def format_float(x: float, precision: int = 4, threshold: float = 1e-3) -> str:
    """Format a float number using fixed-point notation unless it is very small (but nonzero),
    in which case scientific notation is used.

    For scientific notation, trailing zeros in the significand and unnecessary zeros in the exponent
    are removed. For example, 5.000e-5 is formatted as 5e-5.

    Args:
        x (float): The float to format.
        precision (int): Number of digits for formatting.
        threshold (float): If abs(x) < threshold and x != 0, use scientific notation.

    Returns:
        str: The formatted float as a string.
    """
    if x != 0 and abs(x) < threshold:
        formatted = f"{x:.{precision}e}"  # e.g. "5.0000e-05"
        significand, exponent = formatted.split("e")
        # Remove trailing zeros and dot from the significand if necessary
        significand = significand.rstrip("0").rstrip(".")
        # Process the exponent to remove unnecessary leading zeros
        exp_sign = exponent[0]  # '+' or '-'
        exp_digits = exponent[1:].lstrip("0")
        if exp_digits == "":
            exp_digits = "0"
        return f"{significand}e{exp_sign}{exp_digits}"
    else:
        return f"{x:.{precision}f}"


def decisions_to_probabilities(decisions: np.ndarray) -> np.ndarray:
    """Convert raw *decision_function* scores to probability estimates.

    If *decisions* is 1-D (binary classifier) we apply a sigmoid. If it is
    2-D (multi-class) we apply a numerically-stable softmax **and return the
    probability of the positive / class-1 column** (or column-0 if it is the
    only one). This mirrors the logic used during training.
    """
    if decisions.ndim == 1:
        # Binary classification – logistic sigmoid
        return 1.0 / (1.0 + np.exp(-decisions))

    # Multi-class – softmax across columns
    exp_scores = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Positive class probability: assume class index 1 if it exists.
    return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]


def to_np(x):
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_numpy_array(x) -> np.ndarray:
    """Ensure the input is converted to a NumPy array.

    Handles PyTorch tensors, sparse matrices, and other array-like objects.

    Args:
        x: Array-like object (numpy array, torch tensor, sparse matrix, etc.)

    Returns:
        NumPy array
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif hasattr(x, "toarray"):  # Sparse matrices
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def copy_array_like(x, copy: bool = True):
    """Create a copy of an array-like object if requested.

    Args:
        x: Array-like object (numpy array, torch tensor, sparse matrix)
        copy: Whether to create a copy

    Returns:
        Copy or reference to the original array
    """
    if not copy:
        return x

    if isinstance(x, torch.Tensor):
        return x.clone()
    elif hasattr(x, "copy"):  # NumPy arrays and sparse matrices
        return x.copy()
    else:
        return x  # Fallback for unknown types


def compute_row_sums(X) -> np.ndarray:
    """Compute row sums for any array-like object, returning NumPy array.

    Args:
        X: Array-like object (numpy array, torch tensor, sparse matrix)

    Returns:
        NumPy array of row sums

    Raises:
        TypeError: If the input type doesn't support row sum computation
    """
    if hasattr(X, "sum"):
        if isinstance(X, torch.Tensor):
            return X.sum(dim=1).detach().cpu().numpy()
        else:  # NumPy array or sparse matrix
            row_sums = X.sum(axis=1)
            # Ensure flat NumPy array for sparse matrices
            return np.asarray(row_sums).flatten() if hasattr(row_sums, "flatten") else row_sums
    else:
        raise TypeError(f"Unsupported type: {type(X)}. Expected np.ndarray, torch.Tensor, or sparse matrix.")


def compute_scaling_factors(row_sums: np.ndarray, target_sum: float, dtype: type = np.float32) -> np.ndarray:
    """Compute row scaling factors for normalization.

    Args:
        row_sums: Sum of each row
        target_sum: Target sum for normalization
        dtype: Data type for the output array

    Returns:
        Array of scaling factors
    """
    scaling_factors = np.ones_like(row_sums, dtype=dtype)
    nonzero_mask = row_sums > 0
    scaling_factors[nonzero_mask] = target_sum / row_sums[nonzero_mask]
    return scaling_factors


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_new_version_path(save_path: Path | str) -> Path:
    """Create a versioned directory path to avoid overwriting existing results.

    If the target path already exists and contains files, creates a new versioned
    directory (e.g., 'results_v_1', 'results_v_2') to preserve existing data.

    Args:
        save_path: Desired save directory path.

    Returns:
        Path to use for saving (original path or new versioned path).

    Example:
        >>> path = get_new_version_path("results/experiment_1")
        >>> # If results/experiment_1 exists, returns results/experiment_1_v_1
        >>> print(f"Saving to: {path}")
    """
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
    # if exists and empty, return the same path:
    if save_path.is_dir() and not any(save_path.iterdir()):
        return save_path
    parent_dir = save_path.parent
    dir_name = save_path.name
    # find all existing dirs with the pattern: *_v_*
    existing_dirs = list(parent_dir.glob(f"{dir_name}_v_*"))
    # Get the version number of each:
    existing_versions = [int(d.name.split("_v_")[1]) for d in existing_dirs]
    # Find the next version number:
    max_version = np.max(existing_versions) if len(existing_versions) > 0 else -1
    new_version = max_version + 1
    version_path = parent_dir / f"{dir_name}_v_{new_version}"
    version_path.mkdir(parents=True, exist_ok=True)
    return version_path


def get_device() -> str:
    """Automatically detect and return the best available device for PyTorch.

    Checks for available hardware acceleration in order of preference:
    CUDA (NVIDIA) > MPS (Apple Silicon) > XPU (Intel) > CPU.

    Returns:
        Device string: 'cuda', 'mps', 'xpu', or 'cpu'.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        >>> model = model.to(device)
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"

    # Check for MPS (Apple Silicon GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    # Check for XPU (Intel GPU)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    # Check for MKLDNN (Intel CPU optimization)
    if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
        return "mkldnn"

    # Default to CPU
    return "cpu"


def floats_to_str(a, precision=5):
    """Convert a numeric float value to a string, with a given precision.

    If the input is a data structure, convert all float elements in it to strings.
    """
    if isinstance(a, float):
        return f"{a:.{precision}f}"
    if isinstance(a, np.ndarray):
        return floats_to_str(a.tolist(), precision)
    if isinstance(a, torch.Tensor):
        return floats_to_str(a.detach().cpu().numpy(), precision)
    if isinstance(a, tuple):
        return tuple(floats_to_str(elem, precision) for elem in a)
    if isinstance(a, list):
        return [floats_to_str(elem, precision) for elem in a]
    if isinstance(a, dict):
        return {key: floats_to_str(value, precision) for key, value in a.items()}

    return a


def get_elapsed_time_str(t0, t1):
    elapsed_time = t1 - t0
    elapsed_time_str = f"{elapsed_time.seconds // 3600} hours, {(elapsed_time.seconds % 3600) // 60} minutes, and {elapsed_time.seconds % 60} seconds."
    return elapsed_time_str


def time_to_str(t: datetime, fmt="%Y-%m-%d %H:%M:%S"):
    """Convert a datetime object to a string."""
    return t.strftime(fmt)


def get_local_time() -> datetime:
    return datetime.now().astimezone()


def get_utc_time() -> datetime:
    return datetime.now(timezone.utc)


def time_seconds_to_str(seconds: float) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{round(hours)}:{round(minutes)}:{round(seconds)}"
    # return f"{hours} hours, {minutes} minutes, and {round(seconds)} seconds."


def get_last_git_commit_link():
    try:
        # Get the Git commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        # Get the remote URL of the repository
        remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip().decode("utf-8")
        # Generate the Git link
        git_link = remote_url.replace(".git", "/commit/") + commit_hash
    except Exception as e:
        logger.info("Error: Failed to retrieve Git information.")
        logger.info(e)
    return git_link


def num2str(v: float) -> str:
    """Convert a number to a string, with a fixed number of decimal places.

    For floats, if their absolute value is small but nonzero, use scientific notation.
    """
    if isinstance(v, int):
        return f"{v}"
    return format_float(v, precision=4)


def compute_false_positive_rate(label: np.ndarray, prob_out: np.ndarray, threshold: float = 0.5) -> float:
    prob_out = (prob_out >= threshold).astype(int)
    label = label.astype(int)
    denominator = np.sum(label == 0)
    if denominator == 0:
        return 0
    return np.sum((label == 0) & (prob_out == 1)) / denominator.astype(float)


def compute_false_negative_rate(label: np.ndarray, prob_out: np.ndarray, threshold: float = 0.5) -> float:
    prob_out = (prob_out >= threshold).astype(int)
    label = label.astype(int)
    denominator = np.sum(label == 1)
    if denominator == 0:
        return 0
    return np.sum((label == 1) & (prob_out == 0)) / denominator.astype(float)


def compute_error_rate(label: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_pred >= threshold).astype(int)
    label = label.astype(int)
    return np.mean(label != y_pred)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def flatten_nested_dict(nested_dict: dict, parent_key: str = "") -> dict:
    """Convert a nested dictionary to a flattened dictionary with keys in the format 'key1/key2/...'.

    Args:
        nested_dict (dict): The dictionary to flatten.
        parent_key (str): A prefix for the keys (used in recursion). Defaults to an empty string.

    Returns:
        dict: A flattened dictionary where nested keys are concatenated by '/'.
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}/{key}" if parent_key else f"{key}"
        if isinstance(value, dict):
            flat_dict.update(flatten_nested_dict(value, new_key))
        else:
            flat_dict[new_key] = value
    return flat_dict


def nested_dict_to_flat_str(nested_scalars: dict, omit_keys: list[str] | None = None) -> str:
    """Flatten a nested dictionary into a string with , separated values.

    In case of a float, display it with 4 decimal places.
    """
    if omit_keys is None:
        omit_keys = []
    return ", ".join(
        [
            (f"{k}={format_float(v, precision=4)}" if isinstance(v, float) else f"{k}={v}")
            for k, v in nested_scalars.items()
            if k not in omit_keys
        ]
    )


def nested_dict_to_multiline_str(nested_scalars: dict, indent: int = 0, oneline_last_level: bool = True) -> str:
    """Recursively returns a string containing a nested dictionary of scalars in a hierarchically indented multi-line format.
    Float values are displayed with improved formatting.

    Args:
        nested_scalars (dict): The nested dictionary containing scalar values.
        indent (int): The current indentation level (used during recursion). Defaults to 0.
        oneline_last_level (bool): Whether to format the last level on a single line. Defaults to True.

    Returns:
        str: The formatted multi-line string.
    """
    lines = []
    #  if a flat dict, return a single line
    if oneline_last_level and all(isinstance(v, int | float | str) for v in nested_scalars.values()):
        return ", ".join(
            (f"{k}={format_float(v, precision=4)}" if isinstance(v, float) else f"{k}={v}")
            for k, v in nested_scalars.items()
        )
    else:
        for key, value in nested_scalars.items():
            if isinstance(value, dict):
                lines.append(" " * indent + f"{key}:")
                lines.append(nested_dict_to_multiline_str(value, indent=indent + 4))
            elif isinstance(value, float):
                lines.append(" " * indent + f"{key}: {format_float(value, precision=4)}")
            else:
                lines.append(" " * indent + f"{key}: {value}")
    return "\n".join(lines)


def log_nested_metrics(
    metrics: dict,
    logger_func,
    prefix: str = "",
    group: str = "validation",
    score_metric: str | None = None,
    epoch: int | None = None,
    use_table_format: bool = True,
) -> None:
    """Log nested metrics with hierarchical display and highlighted score metric.

    Args:
        metrics: Nested dictionary of metrics to log
        logger_func: Logger function (e.g., logger.info)
        prefix: Prefix for log messages
        group: Group name for the metrics (e.g., "validation", "test")
        score_metric: Key of the main score metric to highlight
        epoch: Optional epoch number to include in messages
        use_table_format: If True, display metrics in table format instead of hierarchical
    """
    # Create the main log message
    epoch_str = f" for epoch {epoch + 1}" if epoch is not None else ""

    # Highlight the main score metric if specified
    if score_metric:
        flat_metrics = flatten_nested_dict(metrics)
        if score_metric in flat_metrics:
            main_score = flat_metrics[score_metric]
            score_msg = f"{prefix}{group.capitalize()} metrics{epoch_str}: {score_metric} = {main_score:.4f}"
            logger_func(score_msg)
        else:
            logger_func(f"{prefix}Warning: Score metric '{score_metric}' not found in metrics")

    # Log the full structure
    if metrics:
        if use_table_format:
            table_title = f"{group.capitalize()} Metrics{epoch_str}"
            table_msg = f"{prefix}{metrics_dict_to_table(metrics, title=table_title)}"
            logger_func(table_msg)
        else:
            full_msg = (
                f"{prefix}{group.capitalize()} metrics details{epoch_str}:\n{nested_dict_to_multiline_str(metrics)}"
            )
            logger_func(full_msg)


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary, preserving the hierarchy in the keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        else:
            items.append((parent_key, k, v))
    return items


T = TypeVar("T", bound="BaseParams")


def load_and_override_params(
    param_class: Type[T],
    config_path: str | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
    **kwargs: Any,
) -> T:
    """Load parameters from config file or use defaults, then apply overrides.

    Args:
        param_class: The parameter dataclass to instantiate
        config_path: Optional path to JSON config file
        logger: Logger instance for logging changes
        **kwargs: Parameter overrides to apply

    Returns:
        The parameter object with overrides applied
    """
    # Load base parameters
    if config_path:
        prm = param_class(**load_params(config_path))
        if logger:
            logger.info(f"Loaded parameters from config: {config_path}")
    else:
        prm = param_class()
        if logger:
            logger.info("Using default parameters")

    # Apply overrides with logging
    if kwargs:
        # Get valid field names from the dataclass
        valid_fields = {field.name for field in dataclasses.fields(prm)}

        # Check for invalid field names
        invalid_fields = set(kwargs.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid parameter field(s): {sorted(invalid_fields)}. Valid fields are: {sorted(valid_fields)}"
            )

        # Apply overrides and log changes
        for field_name, new_value in kwargs.items():
            old_value = getattr(prm, field_name)
            if old_value != new_value:
                if logger:
                    logger.info(f"Parameter override: {field_name} = {new_value} (was {old_value})")
                setattr(prm, field_name, new_value)

    return prm


def metrics_dict_to_table(metrics: dict, title: str = "Metrics", precision: int = 4) -> str:
    """Convert nested metrics dictionary to a formatted table string using pandas.

    Args:
        metrics: Nested dictionary containing metrics data
        title: Title for the table
        precision: Number of decimal places for float values

    Returns:
        Formatted table as a string
    """
    if not metrics:
        return f"{title}: No metrics to display"

    # Separate overall metrics from category-specific metrics
    overall_metrics = {}
    category_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, dict):
            # This is a category-specific metric
            category_metrics[key] = value
        else:
            # This is an overall metric
            overall_metrics[key] = value

    # Build the output
    lines = []

    # Title
    lines.append(f"\n{title}")
    lines.append("=" * len(title))

    # Overall metrics section
    if overall_metrics:
        lines.append("\nOverall Metrics:")
        lines.append("-" * 15)

        # Sort overall metrics alphabetically
        sorted_overall = dict(sorted(overall_metrics.items()))

        # Create DataFrame for overall metrics
        overall_df = pd.DataFrame(list(sorted_overall.items()), columns=["Metric", "Value"])
        # Format float values
        overall_df["Value"] = overall_df["Value"].apply(
            lambda x: f"{x:.{precision}f}"
            if isinstance(x, float) and not np.isnan(x)
            else "nan"
            if isinstance(x, float) and np.isnan(x)
            else str(x)
        )

        # Use tabulate for prettier formatting
        table_str = tabulate(overall_df, headers="keys", tablefmt="simple", showindex=False)
        lines.append(table_str)
        lines.append("")  # Add newline after table

        # Category-specific metrics section
    if category_metrics:
        lines.append("")  # Just add empty line for spacing

        # Create DataFrame for category metrics
        category_df = pd.DataFrame.from_dict(category_metrics, orient="index")

        # Sort columns alphabetically
        category_df = category_df.reindex(sorted(category_df.columns), axis=1)

        # Format float values with specified precision
        for col in category_df.columns:
            category_df[col] = category_df[col].apply(
                lambda x: f"{x:.{precision}f}"
                if isinstance(x, float) and not np.isnan(x)
                else "nan"
                if isinstance(x, float) and np.isnan(x)
                else str(x)
            )

            # Use tabulate for prettier formatting with index as row labels
        table_str = tabulate(category_df, headers="keys", tablefmt="simple", showindex=True)
        lines.append(table_str)
        lines.append("")  # Add newline after table

    return "\n".join(lines)


def metrics_dict_to_dataframes(metrics: dict, precision: int = 4) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Convert nested metrics dictionary to pandas DataFrames for nice display in notebooks.

    Args:
        metrics: Nested dictionary containing metrics data
        precision: Number of decimal places for float values

    Returns:
        Tuple of (overall_df, category_df) where:
        - overall_df: DataFrame with overall metrics (Metric, Value columns)
        - category_df: DataFrame with category-specific metrics (categories as rows, metrics as columns)
        Either DataFrame can be None if no data exists for that category
    """
    if not metrics:
        return None, None

    # Separate overall metrics from category-specific metrics
    overall_metrics = {}
    category_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, dict):
            # This is a category-specific metric
            category_metrics[key] = value
        else:
            # This is an overall metric
            overall_metrics[key] = value

    # Create overall metrics DataFrame
    overall_df = None
    if overall_metrics:
        # Sort overall metrics alphabetically
        sorted_overall = dict(sorted(overall_metrics.items()))

        # Create DataFrame for overall metrics
        overall_df = pd.DataFrame(list(sorted_overall.items()), columns=["Metric", "Value"])

        # Format float values
        overall_df["Value"] = overall_df["Value"].apply(
            lambda x: f"{x:.{precision}f}"
            if isinstance(x, float) and not np.isnan(x)
            else "nan"
            if isinstance(x, float) and np.isnan(x)
            else str(x)
        )

    # Create category-specific metrics DataFrame
    category_df = None
    if category_metrics:
        # Create DataFrame for category metrics
        category_df = pd.DataFrame.from_dict(category_metrics, orient="index")

        # Sort columns alphabetically
        category_df = category_df.reindex(sorted(category_df.columns), axis=1)

        # Format float values with specified precision
        for col in category_df.columns:
            category_df[col] = category_df[col].apply(
                lambda x: f"{x:.{precision}f}"
                if isinstance(x, float) and not np.isnan(x)
                else "nan"
                if isinstance(x, float) and np.isnan(x)
                else str(x)
            )

    return overall_df, category_df


def log_inference_progress(
    current_iteration: int,
    total_iterations: int,
    start_time: float,
    log_interval: int = 20,
    logger_instance: structlog.stdlib.BoundLogger | None = None,
) -> None:
    """Log progress during inference or processing operations.

    Args:
        current_iteration: Current iteration number (0-indexed)
        total_iterations: Total number of iterations
        start_time: Start time of the process (from time.time())
        log_interval: Log every N iterations
        logger_instance: Logger object (default: module logger)
    """
    if logger_instance is None:
        logger_instance = get_logger()

    if (current_iteration + 1) % log_interval == 0:
        elapsed = time.time() - start_time
        progress = (current_iteration + 1) / total_iterations
        estimated_total = elapsed / progress if progress > 0 else 0
        logger_instance.info(f"Inference: {progress:.1%} | Time: {elapsed:.1f}s / ~{estimated_total:.1f}s")
