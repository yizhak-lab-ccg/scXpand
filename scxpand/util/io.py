"""File I/O operations for scXpand.

This module handles all file input/output operations including:
- Saving predictions to CSV files
- Loading evaluation indices
- File path validation and creation
- Multiprocessing-safe AnnData file operations with retry mechanisms

This module has no dependencies on model-specific code.
"""

import random
import time

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import anndata as ad
import numpy as np
import pandas as pd

from scxpand.util.logger import get_logger


if TYPE_CHECKING:
    from scxpand.util.classes import ModelType

logger = get_logger()


# Constants for retry mechanism (aligned with AnnData practices)
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 0.1
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_DELAY = 2.0
DEFAULT_CHUNK_SIZE = 6000  # AnnData's default chunk_size
STRATEGY_DELAY = 0.05


class IOSettings:
    """Settings for I/O operations, following AnnData patterns."""

    def __init__(self):
        self.max_retries = DEFAULT_MAX_RETRIES
        self.initial_delay = DEFAULT_INITIAL_DELAY
        self.backoff_factor = DEFAULT_BACKOFF_FACTOR
        self.max_delay = DEFAULT_MAX_DELAY
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.enable_retry_logging = True
        self.enable_jitter = True  # Prevents thundering herd in multiprocessing

    def __repr__(self):
        return (
            f"IOSettings(\n"
            f"    max_retries={self.max_retries},\n"
            f"    initial_delay={self.initial_delay},\n"
            f"    chunk_size={self.chunk_size},\n"
            f"    enable_retry_logging={self.enable_retry_logging},\n"
            f"    enable_jitter={self.enable_jitter}\n"
            f")"
        )


# Global settings instance (following AnnData pattern)
settings = IOSettings()

# HDF5 error patterns that indicate we should retry
HDF5_ERROR_PATTERNS = [
    "can't synchronously read data",
    "filter returned failure",
    "unable to read data",
    "hdf5 error",
    "file is already open",
    "concurrent access",
]


def is_hdf5_error(error: Exception) -> bool:
    """Check if an error is HDF5-related and should be retried."""
    error_msg = str(error).lower()
    return any(pattern in error_msg for pattern in HDF5_ERROR_PATTERNS)


def exponential_backoff_delay(
    attempt: int,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay for retry attempts with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Base delay for first retry
        backoff_factor: Multiplier for exponential growth
        max_delay: Maximum delay cap
        jitter: If True, adds randomization to prevent thundering herd

    Returns:
        Delay in seconds before next retry attempt
    """
    # Calculate base exponential delay
    delay = initial_delay * (backoff_factor**attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add jitter: randomize between 50% and 100% of calculated delay
        # This prevents multiple workers from retrying simultaneously
        jitter_factor = 0.5 + (random.random() * 0.5)  # Random float in [0.5, 1.0]
        delay *= jitter_factor

    return delay


def retry_hdf5_operation(
    operation: Callable[[], Any],
    max_retries: int | None = None,
    operation_name: str = "operation",
) -> Any:
    """Retry an operation that may fail due to HDF5 multiprocessing issues.

    Args:
        operation: Function to retry (should take no arguments)
        max_retries: Maximum number of retry attempts (uses settings default if None)
        operation_name: Name of operation for logging

    Returns:
        Result of the operation if successful

    Raises:
        Exception: The last exception if all retries fail
    """
    if max_retries is None:
        max_retries = settings.max_retries

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except OSError as e:
            last_exception = e

            if not is_hdf5_error(e) or attempt >= max_retries:
                raise e

            delay = exponential_backoff_delay(
                attempt,
                settings.initial_delay,
                settings.backoff_factor,
                settings.max_delay,
                jitter=settings.enable_jitter,
            )
            if settings.enable_retry_logging:
                logger.warning(
                    f"HDF5 error in {operation_name} on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
            time.sleep(delay)

    # This shouldn't be reached, but just in case
    raise last_exception


def read_adata_slice_direct(adata_obj, indices: np.ndarray) -> np.ndarray:
    """Direct slice reading - fastest when it works."""
    X_sparse = adata_obj.X[indices, :]
    return X_sparse.toarray() if hasattr(X_sparse, "toarray") else np.array(X_sparse)


def read_adata_slice_chunked(adata_obj, indices: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
    """Chunked reading - more robust for large slices."""
    if chunk_size is None:
        chunk_size = settings.chunk_size
    chunk_size = min(chunk_size, len(indices))
    results = []

    for i in range(0, len(indices), chunk_size):
        chunk_indices = indices[i : i + chunk_size]
        chunk_data = adata_obj.X[chunk_indices, :]
        chunk_dense = chunk_data.toarray() if hasattr(chunk_data, "toarray") else np.array(chunk_data)
        results.append(chunk_dense)

    return np.concatenate(results, axis=0)


def read_adata_slice_sequential(adata_obj, indices: np.ndarray) -> np.ndarray:
    """Sequential reading - slowest but most robust."""
    n_genes = adata_obj.n_vars
    results = np.zeros((len(indices), n_genes), dtype=np.float32)

    for i, idx in enumerate(indices):
        row_data = adata_obj.X[idx, :]
        results[i] = row_data.toarray().flatten() if hasattr(row_data, "toarray") else np.array(row_data).flatten()

    return results


def safe_read_adata_slice(adata_obj, indices: np.ndarray) -> np.ndarray:
    """Safely read a slice from AnnData with retry mechanism and fallback strategies.

    Args:
        adata_obj: AnnData object to read from
        indices: Row indices to read

    Returns:
        Dense numpy array with the requested data

    Raises:
        OSError: If all retry attempts and fallback strategies fail
    """
    # Define reading strategies in order of preference (fastest to most robust)
    strategies = [
        ("direct slice reading", lambda: read_adata_slice_direct(adata_obj, indices)),
        ("chunked reading", lambda: read_adata_slice_chunked(adata_obj, indices)),
        ("sequential reading", lambda: read_adata_slice_sequential(adata_obj, indices)),
    ]

    last_exception = None

    for strategy_name, strategy_func in strategies:
        try:
            logger.debug(f"Attempting {strategy_name} for {len(indices)} indices")
            return retry_hdf5_operation(strategy_func, operation_name=strategy_name)
        except Exception as e:
            last_exception = e
            logger.warning(f"{strategy_name} failed: {e}. Trying next strategy...")
            time.sleep(STRATEGY_DELAY)

    # If all strategies failed, raise the last exception
    logger.error(f"All read strategies failed for indices shape {indices.shape}")
    raise last_exception


def load_eval_indices(eval_row_inds_path: str | Path) -> np.ndarray:
    """Load evaluation indices from a file.

    Args:
        eval_row_inds_path: Path to file containing cell indices (one per line)

    Returns:
        Array of evaluation indices

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid data
    """
    eval_row_inds_path = Path(eval_row_inds_path)
    if not eval_row_inds_path.exists():
        raise FileNotFoundError(f"Evaluation indices file not found: {eval_row_inds_path}")

    logger.info(f"Loading evaluation indices from {eval_row_inds_path}")

    try:
        eval_indices = np.loadtxt(eval_row_inds_path, dtype=int)
        logger.info(f"Loaded {len(eval_indices)} evaluation indices")
        return eval_indices
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid data in evaluation indices file {eval_row_inds_path}: {e}") from e


def save_predictions_to_csv(
    predictions: np.ndarray,
    obs_df: pd.DataFrame,
    model_type: "ModelType | str",
    save_path: Path,
) -> None:
    """Save predictions to a CSV file.

    Args:
        predictions: Model predictions (probabilities)
        obs_df: DataFrame with cell metadata
        model_type: Type of model used for predictions
        save_path: Directory to save predictions

    Raises:
        ValueError: If predictions and obs_df have mismatched lengths
    """
    if len(predictions) != len(obs_df):
        raise ValueError(f"Predictions length ({len(predictions)}) doesn't match obs_df length ({len(obs_df)})")

    # Convert ModelType enum to string if needed
    model_type_str = model_type.value if hasattr(model_type, "value") else str(model_type)

    # Create predictions DataFrame
    predictions_df = obs_df.copy()
    predictions_df[f"{model_type_str}_prediction"] = predictions

    # Ensure save directory exists
    save_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    output_file = save_path / f"{model_type_str}_predictions.csv"
    predictions_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


def ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def open_adata_file_with_retry(data_path: str | Path) -> ad.AnnData:
    """Open AnnData file with retry mechanism."""
    return retry_hdf5_operation(lambda: ad.read_h5ad(data_path, backed="r"), operation_name=f"opening {data_path}")


def close_adata_file_safely(adata_obj: ad.AnnData) -> None:
    """Safely close AnnData file handle without raising exceptions."""
    try:
        if hasattr(adata_obj, "file") and adata_obj.file is not None:
            adata_obj.file.close()
            logger.debug("Closed AnnData file handle")
    except OSError as e:
        logger.warning(f"Error closing AnnData file handle: {e}")
        # Don't re-raise - closing errors shouldn't stop execution
    except Exception as e:
        logger.error(f"Unexpected error closing AnnData file handle: {e}")
        # Log critical errors but still don't re-raise to avoid crashes


@contextmanager
def open_adata_multiprocessing_safe(
    data_path: str | Path | None,
    adata: ad.AnnData | None = None,
    indices: np.ndarray | None = None,
):
    """Context manager for multiprocessing-safe AnnData file opening with retry mechanism.

    This function ensures that each worker process opens its own file handle
    to avoid the common "OSError: Can't synchronously read data" error when
    using PyTorch DataLoader with num_workers > 0. Includes robust retry
    mechanisms for HDF5-related errors.

    Args:
        data_path: Path to H5AD file. Required unless adata is provided.
        adata: In-memory AnnData object. Alternative to data_path.
        indices: Optional indices for the yielded data (passed through unchanged).

    Yields:
        Tuple of (AnnData object, indices) for batch access.

    Example:
        >>> with open_adata_multiprocessing_safe("data.h5ad") as (adata, indices):
        ...     X_data = safe_read_adata_slice(adata, cell_indices)

    Note:
        - If adata is provided, it's used directly (no file operations)
        - Each call opens a fresh file handle (safer for multiprocessing)
        - File handles are properly closed when the context exits
        - Includes retry mechanism for common HDF5 multiprocessing errors
    """
    if adata is not None:
        # Use provided in-memory AnnData object
        yield adata, indices
        return

    if data_path is None:
        raise ValueError("Either data_path or adata must be provided")

    # Each worker process opens its own file handle - this is the key to multiprocessing safety
    logger.debug(f"Opening AnnData file with backed='r' for worker-safe access: {data_path}")
    adata_obj = open_adata_file_with_retry(data_path)

    try:
        yield adata_obj, indices
    finally:
        close_adata_file_safely(adata_obj)
