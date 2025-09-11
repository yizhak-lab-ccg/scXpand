"""Statistical computation utilities for expression data.

This module provides efficient batch-based statistical computations for
preprocessing parameter estimation and sparse matrix operations.
"""

import time

from pathlib import Path

import numba
import numpy as np

from anndata import read_h5ad
from scipy.sparse import csr_matrix

from scxpand.util.logger import get_logger


logger = get_logger()


# ============================================================================
# Numba-Accelerated Sparse Matrix Operations
# ============================================================================


@numba.njit(cache=True)
def _csr_sum_and_squared_sum(data, indices, indptr, n_rows, n_cols):
    """Numba-accelerated sum and squared sum computation for CSR matrix batch."""
    sums = np.zeros(n_cols, dtype=np.float64)
    sq_sums = np.zeros(n_cols, dtype=np.float64)

    for row in range(n_rows):
        start = indptr[row]
        end = indptr[row + 1]
        for idx in range(start, end):
            col = indices[idx]
            val = data[idx]
            sums[col] += val
            sq_sums[col] += val * val

    return sums, sq_sums


@numba.njit(cache=True)
def _csr_row_scaling(data, indptr, scaling_factors, n_rows):
    """Numba-accelerated row scaling for CSR matrix."""
    for row in range(n_rows):
        start_idx = indptr[row]
        end_idx = indptr[row + 1]
        scale = scaling_factors[row]
        for idx in range(start_idx, end_idx):
            data[idx] *= scale


# ============================================================================
# Batch Statistical Computation
# ============================================================================


def compute_preprocessed_genes_means_stds(
    data_path: str | Path,
    row_inds: np.ndarray,
    batch_size: int = 200_000,
    target_sum: float = 1e4,
    use_log_transform: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-gene means and standard deviations after preprocessing.

    This function efficiently computes statistics on large datasets by processing
    in batches and applying the same preprocessing steps used during training.

    Args:
        data_path: Path to the AnnData file
        row_inds: Indices of rows to process (preferably sorted for speed)
        batch_size: Size of each batch for processing
        target_sum: Target sum for row normalization per cell
        use_log_transform: Whether to apply log1p transformation

    Returns:
        Tuple of (means, stds) arrays with shape [n_genes]
    """
    adata = read_h5ad(data_path, backed="r")
    start_time = time.time()

    logger.info(
        f"Computing per-gene mean and std for {len(row_inds)} rows after preprocessing "
        f"(row normalization with target_sum={target_sum}, log_transform={use_log_transform})"
    )

    n_rows = len(row_inds)
    n_genes = adata.shape[1]
    total_sum = np.zeros(n_genes, dtype=np.float64)
    total_squared_sum = np.zeros(n_genes, dtype=np.float64)
    total_count = 0

    n_batches = (n_rows + batch_size - 1) // batch_size

    # Pre-fetch X for faster slicing if possible
    X = adata.X

    for batch_idx, batch_start in enumerate(range(0, n_rows, batch_size)):
        batch_end = min(batch_start + batch_size, n_rows)
        batch_inds = row_inds[batch_start:batch_end]
        elapsed = time.time() - start_time
        logger.info(f"Processing batch {batch_idx + 1}/{n_batches}. Elapsed: {elapsed:.2f} seconds.")

        # Load batch as CSR matrix
        X_batch_csr = X[batch_inds, :]
        # Always convert to CSR matrix to ensure proper interface
        X_batch_csr = csr_matrix(X_batch_csr)

        # Apply preprocessing steps to match training pipeline
        # 1. Row normalization - apply directly on CSR matrix
        row_sums = np.array(X_batch_csr.sum(axis=1)).flatten()
        scaling_factors = np.ones_like(row_sums, dtype=float)
        np.divide(target_sum, row_sums, out=scaling_factors, where=row_sums > 0)

        # Apply scaling to CSR data
        _csr_row_scaling(X_batch_csr.data, X_batch_csr.indptr, scaling_factors, X_batch_csr.shape[0])

        # 2. Log transform if specified - apply directly on CSR data
        if use_log_transform:
            X_batch_csr.data = np.log1p(X_batch_csr.data)

        # Use Numba-accelerated sum and squared sum
        sums, sq_sums = _csr_sum_and_squared_sum(
            X_batch_csr.data, X_batch_csr.indices, X_batch_csr.indptr, X_batch_csr.shape[0], n_genes
        )

        total_sum += sums
        total_squared_sum += sq_sums
        total_count += X_batch_csr.shape[0]

    # Compute final statistics
    means = total_sum / total_count
    variances = total_squared_sum / total_count - (means**2)
    variances = np.maximum(variances, 0)  # Ensure non-negative variances
    stds = np.sqrt(variances).astype(np.float32)
    means = means.astype(np.float32)

    end_time = time.time()
    logger.info(f"compute_preprocessed_genes_means_stds took {end_time - start_time:.2f} seconds")

    # Clean up
    adata.file.close()
    del adata

    return means, stds
