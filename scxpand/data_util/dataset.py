from contextlib import contextmanager
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from scxpand.data_util.constants import (
    CLONE_ID_SIZE_COL,
    DEFAULT_EPS,
    EXPANSION_COL,
    IMPUTED_LABELS_COL,
    MEDIAN_CLONE_SIZE_COL,
    TISSUE_TYPE_COL,
)
from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.transforms import (
    apply_row_normalization,
    extract_is_expanded,
    preprocess_expression_data,
)
from scxpand.util.classes import DataAugmentParams
from scxpand.util.general_util import sigmoid
from scxpand.util.io import open_adata_multiprocessing_safe, safe_read_adata_slice
from scxpand.util.logger import get_logger


logger = get_logger()


def compute_soft_labels(obs_df: pd.DataFrame, dataset_params: DataAugmentParams) -> np.ndarray:
    """Compute soft labels for the training data.

    Args:
        obs_df: DataFrame containing observation data
        dataset_params: Data augmentation parameters containing soft_loss_beta
        prm: Param object with the model parameters

    Returns:
        y_soft: A NumPy array of soft labels in the range [0, 1], if y_soft > 0.5, the cell is expanded.
    """
    clone_id_size = obs_df[CLONE_ID_SIZE_COL].to_numpy()
    median_clone_size = obs_df[MEDIAN_CLONE_SIZE_COL].to_numpy()
    soft_loss_beta = dataset_params.soft_loss_beta

    # Ensure no division by zero
    safe_median_clone_size = np.where(median_clone_size > 0, median_clone_size, 1)
    ratio = clone_id_size / safe_median_clone_size

    # Apply sigmoid scaling
    EXPAND_RATIO_THRESH = 1.5
    y_soft = sigmoid(soft_loss_beta * (ratio - EXPAND_RATIO_THRESH)).astype(np.float32)

    # Handle any NaN values
    return np.nan_to_num(y_soft, nan=0.0)


def apply_pre_normalization_augmentations(
    X: torch.Tensor,
    mask_rate: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply pre-normalization augmentations to input tensor.

    These augmentations simulate missing data and should be applied to raw counts.

    Args:
        X: Input tensor to augment (raw counts)
        mask_rate: Rate at which to mask values (set to 0) to simulate missing genes
        generator: Optional PyTorch generator for reproducible randomness

    Returns:
        Augmented tensor with masked values
    """
    device = X.device

    if mask_rate > 0:
        mask = torch.rand(X.shape, device=device, generator=generator) < mask_rate
        X = X * (~mask).float()

    return X


def apply_post_normalization_augmentations(
    X: torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply post-normalization augmentations to input tensor.

    These augmentations add controlled noise to normalized data.

    Args:
        X: Input tensor to augment (normalized data)
        noise_std: Standard deviation of Gaussian noise to add
        generator: Optional PyTorch generator for reproducible randomness

    Returns:
        Augmented tensor with added noise
    """
    device = X.device

    if noise_std > 0:
        if generator is not None:
            # Use the generator with torch.randn for compatibility
            noise = torch.randn(X.shape, device=device, generator=generator) * noise_std
        else:
            # Fallback to randn_like without generator
            noise = torch.randn_like(X, device=device) * noise_std
        X = X + noise

    return X


class CellsDataset(Dataset):
    def __init__(
        self,
        data_format: DataFormat,
        row_inds: np.ndarray | None = None,
        dataset_params: DataAugmentParams | None = None,
        is_train: bool = True,
        data_path: str | Path | None = None,
        include_row_normalized_gene_counts: bool = False,
        adata: ad.AnnData | None = None,
    ) -> None:
        """PyTorch Dataset for single-cell expression data with preprocessing pipeline.

        Provides efficient batch loading with on-the-fly preprocessing including
        normalization, log transformation, and z-score standardization. Supports
        both file-based and in-memory data access.

        Args:
            data_format: DataFormat object containing preprocessing parameters.
            row_inds: Cell indices to include. If None, includes all cells.
            dataset_params: Data augmentation parameters. Only used during training.
            is_train: Whether this is training data (enables augmentation).
            data_path: Path to H5AD file. Required unless adata is provided.
            include_row_normalized_gene_counts: Include raw normalized counts in batches (useful for autoencoder training)
            adata: In-memory AnnData object. Alternative to data_path.


        """
        self.data_path = data_path
        self._adata = adata
        self.dataset_params = dataset_params if dataset_params is not None else DataAugmentParams()
        self.is_train = is_train
        self.data_format = data_format
        self.n_genes = data_format.n_genes
        self.aux_categorical_mappings = data_format.aux_categorical_mappings
        self.aux_categorical_types = list(data_format.aux_categorical_types)
        self.include_row_normalized_gene_counts = include_row_normalized_gene_counts
        self.row_inds = row_inds

        # Initialize data transformation components once during dataset creation
        self._initialize_data_transformation()

        # Load and preprocess data once during initialization
        self._preprocess_data()

        # Convert normalization parameters to tensors once during initialization (only if z-score normalization is enabled)
        if data_format.use_zscore_norm:
            self.genes_mu_tensor = torch.from_numpy(data_format.genes_mu).float()
            self.genes_sigma_tensor = torch.from_numpy(data_format.genes_sigma).float()
        else:
            self.genes_mu_tensor = None
            self.genes_sigma_tensor = None

        # Create a PyTorch generator for worker-specific randomness in augmentations
        # This will be reseeded in worker processes to ensure different random states
        self._torch_generator = None
        self._worker_seed_base = hash((id(self), self.dataset_params.mask_rate, self.dataset_params.noise_std))

    def _initialize_data_transformation(self):
        """Initialize data transformation components once during dataset creation."""
        # Use adata if provided, else load from file
        if self._adata is not None:
            raw_adata = self._adata
            opened_here = False
        else:
            raw_adata = ad.read_h5ad(self.data_path, backed="r")
            opened_here = True

        # Determine gene mapping and transformation strategy
        raw_genes = set(raw_adata.var_names)
        target_genes = set(self.data_format.gene_names)

        # Find gene overlap and create mapping
        self.gene_overlap = raw_genes.intersection(target_genes)
        self.missing_genes = target_genes - raw_genes
        self.extra_genes = raw_genes - target_genes

        # Create gene index mapping for efficient transformation
        raw_gene_to_idx = {gene: i for i, gene in enumerate(raw_adata.var_names)}

        # Check if gene transformation is needed
        self.needs_gene_transformation = (
            len(self.missing_genes) > 0
            or len(self.extra_genes) > 0
            or not raw_adata.var_names.equals(pd.Index(self.data_format.gene_names))
        )

        if self.needs_gene_transformation:
            # Create ordered indices for efficient gene selection
            self.gene_indices = []
            for target_gene in self.data_format.gene_names:
                if target_gene in raw_gene_to_idx:
                    self.gene_indices.append(raw_gene_to_idx[target_gene])
                else:
                    # Missing gene - will be filled with zeros
                    self.gene_indices.append(-1)
            self.gene_indices = np.array(self.gene_indices, dtype=np.int32)
        else:
            # No transformation needed - genes are already in correct order
            self.gene_indices = None

        # Store data format parameters for batch transformation
        self.target_sum = self.data_format.target_sum
        self.use_log_transform = self.data_format.use_log_transform
        self.use_zscore_norm = self.data_format.use_zscore_norm
        self.genes_mu = self.data_format.genes_mu
        self.genes_sigma = self.data_format.genes_sigma

        # Only close if we opened it
        if opened_here:
            raw_adata.file.close()
            del raw_adata

        if self.needs_gene_transformation:
            logger.info(
                f"Data transformation initialized: {len(self.gene_overlap)} overlapping genes, "
                f"{len(self.missing_genes)} missing genes, {len(self.extra_genes)} extra genes"
            )

    def _preprocess_data(self):
        """Load and preprocess data once during initialization."""
        # Use adata if provided, else load from file
        if self._adata is not None:
            raw_adata = self._adata
            opened_here = False
        else:
            raw_adata = ad.read_h5ad(self.data_path, backed="r")
            opened_here = True

        # Determine which columns are needed
        available_columns = set(raw_adata.obs.columns)

        if self.is_train:
            # For training, expansion is preferred but not absolutely required
            if EXPANSION_COL not in available_columns:
                logger.warning(f"Training mode but '{EXPANSION_COL}' column not found. Labels will not be computed.")
            # Start with required columns that exist in the data
            needed_columns = [
                col for col in [EXPANSION_COL, CLONE_ID_SIZE_COL, MEDIAN_CLONE_SIZE_COL] if col in available_columns
            ]
            # Add optional columns if available
            optional_cols = [col for col in [TISSUE_TYPE_COL, IMPUTED_LABELS_COL] if col in available_columns]
            needed_columns.extend(optional_cols)
        else:
            # For pure inference, we don't need any metadata columns - only gene expression is used
            # All columns are optional for compatibility (in case user wants to compute evaluation metrics)
            needed_columns = []
            optional_cols = [
                col
                for col in [
                    CLONE_ID_SIZE_COL,
                    MEDIAN_CLONE_SIZE_COL,
                    EXPANSION_COL,
                    TISSUE_TYPE_COL,
                    IMPUTED_LABELS_COL,
                ]
                if col in available_columns
            ]
            needed_columns.extend(optional_cols)

        if self.row_inds is None:
            self.row_inds = np.arange(raw_adata.shape[0])

        # Handle case where no metadata columns are needed for pure inference
        if needed_columns:
            self.obs_df = raw_adata.obs.iloc[self.row_inds][needed_columns].reset_index(drop=True)
        else:
            # Create empty dataframe with correct number of rows for pure inference
            self.obs_df = pd.DataFrame(index=range(len(self.row_inds)))

        # Only close if we opened it
        if opened_here:
            raw_adata.file.close()
            del raw_adata

        self.n_cells = len(self.row_inds)

        # Compute labels only when needed
        if self.is_train or EXPANSION_COL in self.obs_df.columns:
            # For training, we need labels. For inference, only compute if expansion column exists
            if EXPANSION_COL in self.obs_df.columns:
                self.y = torch.FloatTensor(extract_is_expanded(self.obs_df))
            else:
                # Training mode but no expansion column - this should not happen in practice
                # However, we'll allow it for partial data scenarios
                logger.warning(f"Training mode but '{EXPANSION_COL}' column not found. Labels will not be computed.")
                self.y = None
        else:
            # For inference without expansion column, don't compute labels at all
            self.y = None

        # Compute soft labels for training (only if required columns are available)
        if self.is_train and CLONE_ID_SIZE_COL in self.obs_df.columns and MEDIAN_CLONE_SIZE_COL in self.obs_df.columns:
            self.y_soft = torch.FloatTensor(compute_soft_labels(obs_df=self.obs_df, dataset_params=self.dataset_params))
        else:
            self.y_soft = None

    def transform_batch_data(self, X_raw: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        """Transform raw batch data according to data format requirements.

        Args:
            X_raw: Raw gene expression data tensor [batch_size, n_raw_genes]
            in_place: Whether to modify X_raw in-place when possible (faster)

        Returns:
            Transformed data tensor [batch_size, n_target_genes]
        """
        # Apply gene transformation if needed (native tensor operations)
        if self.needs_gene_transformation:
            X_tensor = self._apply_gene_transformation_tensor(X_raw)
        else:
            # No gene transformation needed - use tensor directly
            X_tensor = X_raw.clone() if not in_place else X_raw

        # Apply preprocessing steps (preprocess_expression_data preserves tensor input/output)
        return preprocess_expression_data(X=X_tensor, data_format=self.data_format, eps=DEFAULT_EPS)

    def _apply_gene_transformation_tensor(self, X_raw: torch.Tensor) -> torch.Tensor:
        """Apply gene transformation using native tensor operations (no conversions)."""
        batch_size = X_raw.shape[0]
        device = X_raw.device
        dtype = X_raw.dtype

        # Create output tensor with target gene count
        X_transformed = torch.zeros((batch_size, self.n_genes), dtype=dtype, device=device)

        # Vectorized mapping: find valid indices and copy columns efficiently
        valid_mask = self.gene_indices >= 0
        valid_target_indices = torch.from_numpy(np.where(valid_mask)[0]).to(device)
        valid_source_indices = torch.from_numpy(self.gene_indices[valid_mask]).to(device)

        # Copy all valid columns at once using advanced indexing
        X_transformed[:, valid_target_indices] = X_raw[:, valid_source_indices]

        return X_transformed

    def _get_worker_generator(self) -> torch.Generator:
        """Get a PyTorch generator seeded with worker-specific seed for reproducible augmentations."""
        if self._torch_generator is None:
            # Get worker info to create worker-specific seed
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # In multiprocessing mode - use worker ID for unique seed
                worker_seed = self._worker_seed_base + worker_info.id
            else:
                # Single process mode - use base seed
                worker_seed = self._worker_seed_base

            self._torch_generator = torch.Generator()
            self._torch_generator.manual_seed(worker_seed)

        return self._torch_generator

    @contextmanager
    def open_adata(self, indices: np.ndarray):
        """Context manager to yield (AnnData object, indices) for batch access.

        Uses the utility function for multiprocessing-safe file opening.
        """
        with open_adata_multiprocessing_safe(
            data_path=self.data_path,
            adata=self._adata,
            indices=indices,
        ) as (adata, returned_indices):
            yield adata, returned_indices

    def __getstate__(self):
        """Prepare object for pickling - exclude h5py objects."""
        state = self.__dict__.copy()
        # If _adata is a backed AnnData object, it contains unpicklable
        # file handles. Set it to None so it can be reloaded from data_path
        # in the worker process. In-memory AnnData objects are pickled as-is.
        if self._adata is not None and self._adata.isbacked:
            state["_adata"] = None
        return state

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> int:
        # Just return the index; data loading happens in collate_fn
        return idx


def encode_categorical_value(
    value: str | float,
    mapping: dict[str, int],
) -> tuple[int, bool]:
    """Encode a single categorical value to an index in the mapping.

    Args:
        value: The value to encode
        mapping: Dictionary mapping string values to indices

    Returns:
        Tuple of (index, valid) where valid is True if the mapping contains the value
    """
    str_val = str(value)
    idx = mapping.get(str_val, -1)
    return idx, idx >= 0


def encode_categorical_features_batch(
    obs_df: pd.DataFrame,
    categorical_features_types: list[str],
    categorical_mappings: dict[str, dict[str, int]],
) -> np.ndarray:
    """Encode categorical features into one-hot vectors.

    Args:
        obs_df: DataFrame containing observation data
        categorical_features_types: List of categorical feature names
        categorical_mappings: Dict mapping feature names to {category: index} dicts.

    Returns:
        2D numpy array of shape (batch_size, total_categorical_vector_length)
    """
    # Pre-calculate total vector length and allocate output array once
    total_length = sum(
        len(categorical_mappings[cat_type])
        for cat_type in categorical_features_types
        if cat_type in categorical_mappings and cat_type in obs_df.columns
    )
    batch_size = len(obs_df)
    result = np.zeros((batch_size, total_length), dtype=np.float32)

    # Track current position in output array
    current_pos = 0

    for cat_type in categorical_features_types:
        if cat_type not in categorical_mappings or cat_type not in obs_df.columns:
            continue

        mapping = categorical_mappings[cat_type]
        feature_length = len(mapping)

        # Convert values to strings once
        values = obs_df[cat_type].astype(str).to_numpy()
        valid_indices = np.array([mapping.get(val, -1) for val in values], dtype=np.int32)
        valid_mask = valid_indices >= 0

        # Use advanced indexing to set values
        row_indices = np.arange(batch_size)[valid_mask]
        col_indices = valid_indices[valid_mask] + current_pos
        result[row_indices, col_indices] = 1.0

        current_pos += feature_length

    return result


def get_dataloader_kwargs(num_workers: int, dataset: "CellsDataset") -> dict[str, object]:
    """Get common DataLoader keyword arguments.

    Args:
        num_workers: Number of worker processes
        dataset: Dataset to create loader for

    Returns:
        Dictionary of common DataLoader arguments
    """
    persistent_workers = num_workers > 0
    return {
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": partial(cells_collate_fn, dataset=dataset),
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
    }


def compute_categorical_targets_from_batch_obs(
    dataset: "CellsDataset",
    batch_obs: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Compute categorical targets directly from observation data using vectorized operations.

    Args:
        dataset: The dataset containing category mappings and metadata
        batch_obs: Dictionary of observation data for the current batch

    Returns:
        Dictionary mapping feature names to tensors containing categorical target indices
    """
    # If no categorical mappings, return empty dictionary
    if not dataset.aux_categorical_types:
        return {}

    # Initialize output dictionary
    categorical_targets = {}

    # For each categorical feature, map values directly to indices using vectorized operations
    for cat_type in dataset.aux_categorical_types:
        if cat_type not in dataset.aux_categorical_mappings or cat_type not in batch_obs:
            continue

        mapping = dataset.aux_categorical_mappings[cat_type]
        values = batch_obs[cat_type]

        # Convert values to strings once for consistent mapping lookup
        str_values = np.asarray([str(val) for val in values])

        indices = np.array([mapping.get(val, 0) for val in str_values], dtype=np.int64)

        # Create tensor directly from numpy array
        categorical_targets[cat_type] = torch.from_numpy(indices)

    return categorical_targets


def cells_collate_fn(batch_indices: list[int], dataset: "CellsDataset") -> dict[str, torch.Tensor]:
    """Collate function to efficiently create batches from the dataset using the new transformation system."""
    # Sort indices for efficient memory access
    sorted_indices = np.sort(batch_indices).tolist()

    # Get corresponding indices in the original AnnData object
    orig_adata_indices = dataset.row_inds[sorted_indices]

    # Get observation data for this batch efficiently
    batch_df = dataset.obs_df.iloc[sorted_indices]
    batch_obs = {k: batch_df[k].to_numpy() for k in batch_df.columns}

    # Use the dataset's context manager to get the correct AnnData object
    with dataset.open_adata(orig_adata_indices) as (raw_adata, indices):
        # Use safe reading with retry mechanism for HDF5 errors
        X_raw = safe_read_adata_slice(adata_obj=raw_adata, indices=indices)

    # Convert to tensor and apply pre-normalization augmentations if in training mode
    X_raw_tensor = torch.from_numpy(X_raw).float()
    if dataset.is_train:
        generator = dataset._get_worker_generator()
        X_raw_tensor = apply_pre_normalization_augmentations(
            X_raw_tensor, dataset.dataset_params.mask_rate, generator=generator
        )

    # Transform data using tensor-only pipeline
    X = dataset.transform_batch_data(X_raw_tensor)

    # Apply post-normalization augmentations (noise) if in training mode
    if dataset.is_train:
        generator = dataset._get_worker_generator()
        X = apply_post_normalization_augmentations(X, dataset.dataset_params.noise_std, generator=generator)

    # Build the batch dictionary
    batch = {"x": X}

    # Add labels only if they exist (for training or when expansion column is available)
    if dataset.y is not None:
        y = dataset.y[sorted_indices]
        batch["y"] = y

    # Add row-normalized gene counts (without any other modifications) to the batch if requested
    if dataset.include_row_normalized_gene_counts:
        # Use the original tensor before any augmentations
        X_raw_clean = torch.from_numpy(X_raw).float()
        x_row_normalized = apply_row_normalization(
            X=X_raw_clean.clone(),
            target_sum=dataset.data_format.target_sum,
        )
        batch["x_row_normalized_gene_counts"] = x_row_normalized

    if dataset.y_soft is not None:
        batch["y_soft"] = dataset.y_soft[sorted_indices]

    # Add categorical targets if requested
    if dataset.aux_categorical_mappings:
        batch["categorical_targets"] = compute_categorical_targets_from_batch_obs(
            dataset=dataset,
            batch_obs=batch_obs,
        )

    return batch
