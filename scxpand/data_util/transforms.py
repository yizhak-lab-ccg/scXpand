"""Core expression data transformation functions.

This module provides unified transformation functions that work with NumPy arrays,
PyTorch tensors, and sparse matrices for single-cell RNA expression data preprocessing.
"""

from copy import copy
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from scxpand.data_util.constants import DEFAULT_EPS, EXPANSION_COL
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.statistics import _csr_row_scaling
from scxpand.util.general_util import compute_row_sums, compute_scaling_factors, copy_array_like, ensure_numpy_array
from scxpand.util.logger import get_logger


DEFAULT_SIGMA_CLIP_FACTOR = 6.0  # Default factor for outlier clipping


logger = get_logger()

# Type aliases
ArrayLike = Union[np.ndarray, torch.Tensor, sp.spmatrix]

# Constants
EXPANSION_LABEL_TRUE = "expanded"


# ============================================================================
# Label Processing Functions
# ============================================================================


def extract_is_expanded(obs: pd.DataFrame | pd.Series | dict[str, pd.Series]) -> np.ndarray[int]:
    """Extract binary expansion labels from observation data.

    Converts expansion status to binary labels (1 for expanded, 0 for not expanded).
    Looks for 'expansion' column containing 'expanded' values.

    Args:
        obs: Observation data containing expansion information.
            Can be DataFrame with 'expansion' column, Series of expansion values,
            or dict with 'expansion' key.

    Returns:
        Binary array where 1 indicates expanded cells, 0 indicates non-expanded.

    Raises:
        KeyError: If 'expansion' column/key is not found in the data.

    Example:
        >>> labels = extract_is_expanded(adata.obs)
        >>> print(f"Found {labels.sum()} expanded cells out of {len(labels)}")
    """
    if isinstance(obs, pd.Series):
        # Handle Series case - obs should contain expansion data directly
        if EXPANSION_COL in obs.index:
            expansion_labels = obs[EXPANSION_COL]
        else:
            expansion_labels = obs  # Assume obs is the expansion series itself
    elif isinstance(obs, dict):
        if EXPANSION_COL not in obs:
            raise KeyError("'expansion' key not found in observation data dictionary")
        expansion_labels = obs[EXPANSION_COL].to_numpy()
    else:
        # DataFrame case
        if EXPANSION_COL not in obs.columns:
            raise KeyError(
                "'expansion' column not found in observation data. This column is required for evaluation metrics. "
            )
        expansion_labels = obs[EXPANSION_COL].to_numpy()

    if hasattr(expansion_labels, "to_numpy"):
        expansion_labels = expansion_labels.to_numpy()

    return (expansion_labels == EXPANSION_LABEL_TRUE).astype(int)


# ============================================================================
# Core Transformation Functions (Unified Interface)
# ============================================================================


def _convert_to_arrays(genes_mu: ArrayLike, genes_sigma: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert gene statistics to numpy arrays."""
    if isinstance(genes_mu, torch.Tensor):
        genes_mu = genes_mu.detach().cpu().numpy()
    if isinstance(genes_sigma, torch.Tensor):
        genes_sigma = genes_sigma.detach().cpu().numpy()
    return genes_mu, genes_sigma


def _compute_or_use_scaling_factors(X: ArrayLike, target_sum: float) -> np.ndarray:
    """Compute scaling factors dynamically."""
    # Use the clean utility function instead of verbose type checking
    row_sums = compute_row_sums(X)
    return compute_scaling_factors(row_sums, target_sum)


def _apply_scaling_factors_sparse(X_sparse: sp.spmatrix, scaling_factors: np.ndarray) -> sp.spmatrix:
    """Apply scaling factors to sparse matrix efficiently."""
    # Convert to CSR for efficient row scaling if needed
    if X_sparse.format == "csr":
        _csr_row_scaling(X_sparse.data, X_sparse.indptr, scaling_factors, X_sparse.shape[0])
        return X_sparse
    else:
        # Convert to CSR, apply scaling, then convert back
        X_csr = X_sparse.tocsr(copy=False)
        _csr_row_scaling(X_csr.data, X_csr.indptr, scaling_factors, X_csr.shape[0])
        return X_csr.asformat(X_sparse.format) if X_sparse.format != "csr" else X_csr


def _apply_scaling_factors_dense(X: ArrayLike, scaling_factors: np.ndarray) -> ArrayLike:
    """Apply scaling factors to dense arrays (NumPy or PyTorch)."""
    if isinstance(X, torch.Tensor):
        scaling_tensor = torch.from_numpy(scaling_factors).float().unsqueeze(1)
        scaling_tensor = scaling_tensor.to(X.device)
        X *= scaling_tensor
    else:  # NumPy array
        scaling_factors_2d = scaling_factors.reshape(-1, 1)
        X *= scaling_factors_2d
    return X


def _apply_row_normalization_sparse(X_sparse: sp.spmatrix, target_sum: float = 1e4, copy: bool = True) -> sp.spmatrix:
    """Apply row normalization to sparse matrix without converting to dense.

    Args:
        X_sparse: Sparse matrix to normalize
        target_sum: Target sum for each row
        copy: Whether to copy the matrix

    Returns:
        Normalized sparse matrix
    """
    X_sparse = copy_array_like(X_sparse, copy)
    scaling_factors = _compute_or_use_scaling_factors(X_sparse, target_sum)
    return _apply_scaling_factors_sparse(X_sparse, scaling_factors)


def _apply_log_transform_sparse(X_sparse: sp.spmatrix, copy: bool = True) -> sp.spmatrix:
    """Apply log1p transform to sparse matrix without converting to dense."""
    if copy:
        X_sparse = X_sparse.copy()  # Don't modify original

    # Use efficient in-place log1p transformation of non-zero data
    # This is already optimal - NumPy's log1p is vectorized and in-place
    np.log1p(X_sparse.data, out=X_sparse.data)

    return X_sparse


def _apply_zscore_sparse(
    X_sparse: sp.spmatrix,
    genes_mu: np.ndarray,
    genes_sigma: np.ndarray,
    eps: float,
    sigma_clip_factor: float = DEFAULT_SIGMA_CLIP_FACTOR,
) -> np.ndarray:
    """Apply z-score normalization to sparse matrix, returns dense due to mean subtraction."""
    # Z-score normalization requires subtracting mean from every element,
    # which makes the result dense anyway, so convert once at the end
    X_dense = X_sparse.toarray()

    X_dense -= genes_mu
    # Use constant small variance floor for numerical stability
    safe_sigma = genes_sigma + eps
    X_dense /= safe_sigma
    # Robust outlier clipping using sigma-based bounds (3-sigma rule)
    np.clip(X_dense, -sigma_clip_factor, sigma_clip_factor, out=X_dense)

    return X_dense


def apply_row_normalization(X: ArrayLike, target_sum: float = 1e4) -> ArrayLike:
    """Normalize each cell's total expression to target_sum (always in-place).

    Args:
        X: Expression matrix [n_cells, n_genes] - modified in place
        target_sum: Target sum for each cell after normalization

    Returns:
        Normalized expression matrix (same object as input, modified in-place)
    """
    # Handle sparse matrices separately
    if isinstance(X, sp.spmatrix):
        return _apply_row_normalization_sparse(X, target_sum, copy=False)

    # Compute scaling factors
    scaling_factors = _compute_or_use_scaling_factors(X, target_sum)

    # Apply scaling factors in-place
    return _apply_scaling_factors_dense(X, scaling_factors)


def apply_log_transform(X: ArrayLike, in_place: bool = True) -> ArrayLike:
    """Apply log1p transformation to expression data (log(x + 1)).

    Args:
        X: Row-normalized Expression matrix [n_cells, n_genes] with non-negative values
        in_place: Whether to modify X in place

    Returns:
        Log-transformed expression matrix (sparse matrices remain sparse)
    """
    if isinstance(X, sp.spmatrix):
        return _apply_log_transform_sparse(X, copy=not in_place)

    if isinstance(X, torch.Tensor):
        if in_place:
            X.log1p_()
            return X
        else:
            return torch.log1p(X)

    if isinstance(X, np.ndarray):
        if in_place:
            np.log1p(X, out=X)
            return X
        else:
            return np.log1p(X)

    raise TypeError(f"Unsupported type: {type(X)}. Expected np.ndarray, torch.Tensor, or sparse matrix.")


def apply_zscore_normalization(
    X: ArrayLike,
    genes_mu: ArrayLike,
    genes_sigma: ArrayLike,
    eps: float = DEFAULT_EPS,
    in_place: bool = True,
    sigma_clip_factor: float = DEFAULT_SIGMA_CLIP_FACTOR,
) -> ArrayLike:
    """Apply robust z-score normalization using precomputed gene statistics.

    Uses variance stabilization and outlier-resistant normalization following
    numerical computing best practices for genomics data.

    Args:
        X: Expression matrix [n_cells, n_genes]
        genes_mu: Per-gene means [n_genes]
        genes_sigma: Per-gene standard deviations [n_genes]
        eps: Small constant for numerical stability
        in_place: Whether to modify X in place (ignored for sparse matrices)
        sigma_clip_factor: Factor for robust outlier clipping (default DEFAULT_SIGMA_CLIP_FACTOR)

    Returns:
        Z-score normalized expression matrix (always dense due to mean subtraction)
    """
    # Convert gene statistics to numpy arrays
    genes_mu, genes_sigma = _convert_to_arrays(genes_mu, genes_sigma)

    if isinstance(X, sp.spmatrix):
        return _apply_zscore_sparse(X, genes_mu, genes_sigma, eps, sigma_clip_factor)

    if isinstance(X, torch.Tensor):
        if not in_place:
            X = X.clone()

        # Convert gene statistics to tensors
        genes_mu_tensor = torch.from_numpy(genes_mu).float().to(X.device)
        genes_sigma_tensor = torch.from_numpy(genes_sigma).float().to(X.device)

        X -= genes_mu_tensor
        # Use constant small variance floor for numerical stability
        safe_sigma = genes_sigma_tensor + eps
        X /= safe_sigma
        # Robust outlier clipping using sigma-based bounds (3-sigma rule)
        X.clamp_(-sigma_clip_factor, sigma_clip_factor)

        return X

    if isinstance(X, np.ndarray):
        if not in_place:
            X = X.copy()

        # Use efficient numpy operations with broadcasting
        np.subtract(X, genes_mu, out=X)  # In-place subtraction
        # Use constant small variance floor for numerical stability
        safe_sigma = genes_sigma + eps
        np.divide(X, safe_sigma, out=X)  # In-place division
        # Robust outlier clipping using sigma-based bounds (3-sigma rule)
        np.clip(X, -sigma_clip_factor, sigma_clip_factor, out=X)

        return X

    raise TypeError(f"Unsupported type: {type(X)}. Expected np.ndarray, torch.Tensor, or sparse matrix.")


def apply_inverse_zscore_normalization(
    X: torch.Tensor,
    genes_mu: np.ndarray,
    genes_sigma: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Apply inverse z-score normalization to recover original scale data."""
    genes_mu_tensor = torch.from_numpy(genes_mu).to(X.device, dtype=X.dtype)
    genes_sigma_tensor = torch.from_numpy(genes_sigma).to(X.device, dtype=X.dtype)
    # Use constant small variance floor for numerical stability (matching forward pass)
    safe_sigma = genes_sigma_tensor + eps
    return X * safe_sigma + genes_mu_tensor


def apply_inverse_log_transform(X: torch.Tensor) -> torch.Tensor:
    """Apply inverse log transform (expm1) to recover original scale data."""
    X_clamped = torch.clamp(X, min=-10.0, max=10.0)
    return torch.expm1(X_clamped)


# ============================================================================
# Complete Preprocessing Pipeline
# ============================================================================


def preprocess_expression_data(
    X: ArrayLike,
    data_format: DataFormat,
    eps: float = DEFAULT_EPS,
) -> ArrayLike:
    """Apply complete preprocessing pipeline to expression data.

    Pipeline: Row normalization → Log transform (optional) → Z-score normalization (optional)

    Returns the same type as input when possible (tensor in, tensor out).

    Args:
        X: Raw expression matrix [n_cells, n_genes] with non-negative values
        data_format: DataFormat containing preprocessing parameters
        eps: Small constant for numerical stability in z-score normalization

    Returns:
        Fully preprocessed expression matrix. Returns same type as input when possible:
        - torch.Tensor input → torch.Tensor output
        - numpy/sparse input → numpy.ndarray output
    """
    # Remember input type to preserve it in output
    input_is_tensor = isinstance(X, torch.Tensor)

    # Validate dimensions
    if X.shape[1] != len(data_format.genes_mu):
        raise ValueError(
            f"Number of genes in expression matrix ({X.shape[1]}) doesn't match "
            f"data_format ({len(data_format.genes_mu)}). This usually happens when you subset "
            f"AnnData to specific genes. Please use gene_subset parameter instead."
        )

    # Step 1: Row normalization (unified approach) - always in-place
    X = apply_row_normalization(X, target_sum=data_format.target_sum)

    # Step 2: Log transformation (optional)
    if data_format.use_log_transform:
        X = apply_log_transform(X, in_place=True)

    # Step 3: Z-score normalization (optional, produces dense array when applied)
    if data_format.use_zscore_norm:
        X = apply_zscore_normalization(
            X,
            genes_mu=data_format.genes_mu,
            genes_sigma=data_format.genes_sigma,
            eps=eps,
            in_place=True,
            sigma_clip_factor=DEFAULT_SIGMA_CLIP_FACTOR,
        )

    # Return same type as input when possible
    if input_is_tensor:
        # If input was tensor, return tensor
        if isinstance(X, torch.Tensor):
            return X  # Already a tensor
        else:
            return torch.from_numpy(ensure_numpy_array(X)).float()
    else:
        # If input was numpy/sparse, return numpy array
        return ensure_numpy_array(X)


def load_and_preprocess_data_numpy(
    data_path: str | Path,
    data_format: DataFormat,
    row_indices: np.ndarray | None = None,
    gene_subset: list[str] | list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Load and preprocess single-cell data using the full normalization pipeline.

    Efficiently loads data from disk and applies the complete preprocessing pipeline:
    row normalization → log transform → z-score normalization. This function is
    optimized for memory efficiency with large datasets.

    Args:
        data_path: Path to H5AD file containing single-cell expression data.
        data_format: DataFormat object with preprocessing parameters and gene statistics.
        row_indices: Specific cell indices to load. If None, loads all cells.
        gene_subset: Specific genes to subset after preprocessing.
        Can be gene names (str) or indices (int).

    Returns:
        Preprocessed expression matrix [n_cells, n_genes] as dense numpy array.
        Gene order matches data_format.gene_names (or gene_subset if provided).

    Example:
        >>> data_format = load_data_format("results/data_format.json")
        >>> X = load_and_preprocess_data_numpy("data.h5ad", data_format)
        >>> print(f"Loaded {X.shape[0]} cells, {X.shape[1]} genes")
    """
    adata = ad.read_h5ad(data_path, backed="r")

    # Validate gene count consistency
    if adata.n_vars != data_format.n_genes:
        raise ValueError(
            f"Number of genes in expression matrix ({adata.n_vars}) "
            f"does not match data_format.n_genes ({data_format.n_genes})"
        )

    if row_indices is None:
        row_indices = np.arange(adata.n_obs)

    # 1. Load raw data for ALL genes to correctly compute row sums for normalization.
    logger.debug(f"Loading all genes for {len(row_indices)} cells for row normalization...")
    X_raw_full = adata.X[row_indices, :]

    # 2. Make a writeable copy.
    X_processed = X_raw_full.copy()
    X_processed = X_processed.astype(np.float32)

    # 3. Apply row normalization and log transform on the full gene set.
    # These operations are efficient on both sparse and dense matrices.
    X_processed = apply_row_normalization(X_processed, target_sum=data_format.target_sum)
    if data_format.use_log_transform:
        X_processed = apply_log_transform(X_processed, in_place=True)

    # 4. Now, subset to the desired genes (if any) *after* row-level operations.
    if gene_subset is not None:
        subset_data_format, gene_indices_in_adata = _create_gene_subset_data_format(
            adata=adata, original_data_format=data_format, gene_subset=gene_subset
        )
        logger.debug(f"Subsetting to {len(gene_indices_in_adata)} genes after row-level operations.")
        X_processed = X_processed[:, gene_indices_in_adata]
    else:
        subset_data_format = data_format

    # 5. Apply z-score normalization (optional). This step will make the data dense if it's still sparse.
    if subset_data_format.use_zscore_norm:
        X_processed = apply_zscore_normalization(
            X_processed,
            genes_mu=subset_data_format.genes_mu,
            genes_sigma=subset_data_format.genes_sigma,
            eps=DEFAULT_EPS,
            in_place=True,
        )

    return ensure_numpy_array(X_processed)


def _create_gene_subset_data_format(
    adata: ad.AnnData, original_data_format: DataFormat, gene_subset: list[str] | list[int] | np.ndarray
) -> tuple[DataFormat, np.ndarray]:
    """Create a DataFormat object and gene indices for a subset of genes.

    Args:
        adata: AnnData object containing gene information
        original_data_format: Original DataFormat with all genes
        gene_subset: Genes to subset - can be names (str) or indices (int)

    Returns:
        Tuple of (subset_data_format, gene_indices_in_adata)
    """
    # Convert gene_subset to list if it's numpy array
    if isinstance(gene_subset, np.ndarray):
        gene_subset = gene_subset.tolist()

    # Validate that gene_subset is not empty
    if not gene_subset:
        raise ValueError("gene_subset cannot be empty")

    adata_gene_names = list(adata.var_names)
    data_format_gene_names = list(original_data_format.gene_names)

    # Handle gene names vs indices
    if isinstance(gene_subset[0], str):
        # Gene names provided - find indices in adata
        gene_names_subset = gene_subset

        missing_genes = [name for name in gene_names_subset if name not in adata_gene_names]
        if missing_genes:
            raise ValueError(
                f"Genes {missing_genes} not found in AnnData. "
                f"Available genes: {adata_gene_names[:5]}... (showing first 5)"
            )

        adata_gene_indices = [adata_gene_names.index(name) for name in gene_names_subset]
    else:
        # Gene indices provided - validate and convert to names
        gene_indices_subset = [int(idx) for idx in gene_subset]
        max_idx = len(adata_gene_names) - 1

        invalid_indices = [idx for idx in gene_indices_subset if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid gene indices {invalid_indices}. Valid range: 0 to {max_idx}")

        adata_gene_indices = gene_indices_subset
        gene_names_subset = [adata_gene_names[i] for i in gene_indices_subset]

    # Find corresponding indices in data_format
    missing_in_data_format = [name for name in gene_names_subset if name not in data_format_gene_names]
    if missing_in_data_format:
        raise ValueError(
            f"Genes {missing_in_data_format} from AnnData are not found in data_format. "
            "This suggests the data and normalization parameters are incompatible."
        )

    data_format_gene_indices = [data_format_gene_names.index(name) for name in gene_names_subset]

    # Create subset data_format
    subset_data_format = copy(original_data_format)
    subset_data_format.gene_names = gene_names_subset
    subset_data_format.n_genes = len(gene_names_subset)
    subset_data_format.genes_mu = original_data_format.genes_mu[data_format_gene_indices]
    subset_data_format.genes_sigma = original_data_format.genes_sigma[data_format_gene_indices]

    return subset_data_format, np.array(adata_gene_indices)
