"""Data format specification and preprocessing parameters for scXpand models."""

import json

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from anndata import AnnData
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix

from scxpand.data_util.statistics import compute_preprocessed_genes_means_stds
from scxpand.util.logger import get_logger


logger = get_logger()


class DataFormat(BaseModel):
    """Data format specification and preprocessing parameters for scXpand models.

    Contains all metadata and parameters needed to consistently preprocess
    single-cell expression data. Stores gene information, normalization
    parameters, and preprocessing settings used during model training.

    This class ensures that inference data is processed identically to training
    data by preserving gene ordering, normalization statistics, and preprocessing
    pipeline configuration.

    Attributes:
        n_genes: Number of genes in the dataset.
        gene_names: Ordered list of gene names.
        genes_mu: Per-gene means for z-score normalization.
        genes_sigma: Per-gene standard deviations for z-score normalization.
        use_log_transform: Whether to apply log1p transformation.
        use_zscore_norm: Whether to apply z-score normalization.
        target_sum: Target sum for row normalization (typically 10,000).
        aux_categorical_types: Tuple of categorical feature names to include as auxiliary targets.
        aux_categorical_mappings: Dictionary mapping categorical features to their integer encodings.
    """

    model_config = {"arbitrary_types_allowed": True}

    n_genes: int = Field(default_factory=int)
    gene_names: list[str] = Field(default_factory=list)
    genes_mu: np.ndarray = Field(default_factory=lambda: np.array([]))
    genes_sigma: np.ndarray = Field(default_factory=lambda: np.array([]))
    use_log_transform: bool = Field(default=False)
    use_zscore_norm: bool = Field(default=True)
    target_sum: float = Field(default=1e4)
    aux_categorical_types: tuple[str, ...] = Field(default_factory=tuple)
    aux_categorical_mappings: dict[str, dict[str, int]] = Field(default_factory=dict)

    def create_data_format(
        self,
        data_path: str | Path,
        adata: AnnData,
        row_inds_train: np.ndarray,
        batch_size: int = 500_000,
    ) -> None:
        """Create a DataFormat object based on the training set rows.

        This sets up the data format metadata including gene names, means, stds, and categorical mappings.

        Args:
            data_path: Path to the AnnData file. Required for efficient mean/std calculation.
            adata: AnnData object with the data.
            row_inds_train: indices of the training set rows (preferably sorted, for faster runtime).
            batch_size: The batch size to use for computing gene means and stds.
        """
        logger.info("Creating DataFormat object metadata")

        # Save the data format
        self.n_genes = adata.n_vars
        self.gene_names = adata.var_names.tolist()

        # Calculate means and stds with the same preprocessing steps as during training (only if z-score normalization is enabled)
        if self.use_zscore_norm:
            logger.info("Computing empirical mean and std per gene on the normalized training set")
            self.genes_mu, self.genes_sigma = compute_preprocessed_genes_means_stds(
                data_path=data_path,
                row_inds=row_inds_train,
                target_sum=self.target_sum,
                use_log_transform=self.use_log_transform,
                batch_size=batch_size,
            )
        else:
            logger.info("Skipping gene statistics computation (z-score normalization disabled)")
            # Set dummy values since some code might expect these arrays to exist
            self.genes_mu = np.zeros(self.n_genes, dtype=np.float32)
            self.genes_sigma = np.ones(self.n_genes, dtype=np.float32)

        self.aux_categorical_mappings = {}
        if self.aux_categorical_types:
            # Create categorical mappings based on the training set (category -> index) for auxiliary categorical targets
            train_obs = adata.obs.iloc[row_inds_train]
            for feature_type in self.aux_categorical_types:
                if feature_type in train_obs.columns:
                    unique_values = train_obs[feature_type].unique()
                    value_to_index = {str(val): i for i, val in enumerate(unique_values)}
                    self.aux_categorical_mappings[feature_type] = value_to_index
                else:
                    logger.warning(f"Categorical target type {feature_type} not in the AnnData object")
            logger.info(f"Auxiliary target categorical mappings: {self.aux_categorical_mappings}")

        logger.info(f"DataFormat created with {self.n_genes} genes")

    def save(self, save_path: Path) -> None:
        """Save the DataFormat object to a JSON file and numpy arrays to a .npz file."""
        # Prepare dict without numpy arrays
        data = self.model_dump(exclude={"genes_mu", "genes_sigma"})
        with save_path.open(mode="w") as f:
            json.dump(data, f, indent=2)
        # Save numpy arrays
        npz_path = save_path.with_suffix(".npz")
        np.savez_compressed(npz_path, genes_mu=self.genes_mu, genes_sigma=self.genes_sigma)
        logger.info(f"DataFormat saved to {save_path}")

    def _analyze_gene_differences(self, adata: AnnData) -> tuple[list[str], set[str], set[str]]:
        """Analyze differences between current genes and target genes.

        Returns:
            Tuple of (current_gene_names, missing_genes, superfluous_genes)
        """
        curr_gene_names = adata.var_names.tolist()
        missing_genes = set(self.gene_names) - set(curr_gene_names)
        superfluous_genes = set(curr_gene_names) - set(self.gene_names)

        logger.info(f"Number of genes in the AnnData object: {adata.n_vars}")
        logger.info(f"Number of missing genes: {len(missing_genes)}")
        logger.info(f"Number of superfluous genes: {len(superfluous_genes)}")

        return curr_gene_names, missing_genes, superfluous_genes

    def _reorder_genes_only(self, adata: AnnData, curr_gene_names: list[str]) -> AnnData:
        """Reorder genes when no genes need to be added or removed."""
        logger.info("Only reordering genes is needed")
        gene_to_idx = {gene: i for i, gene in enumerate(curr_gene_names)}
        ordered_indices = [gene_to_idx[gene] for gene in self.gene_names]
        new_adata = adata[:, ordered_indices].copy()

        # Verify correct ordering
        assert new_adata.var_names.tolist() == self.gene_names, "Gene reordering failed"
        return new_adata

    def convert_genes_expression_matrix(self, adata: AnnData) -> AnnData:
        """Reorder genes in AnnData to match the data format gene order.

        This function:
        - Reorders genes to match self.gene_names order
        - Adds missing genes as zero columns
        - Removes genes not in the data format
        - Converts X matrix to CSR format with float32 dtype

        Args:
            adata: AnnData object to reorder

        Returns:
            New AnnData object with genes reordered to match data format
        """
        curr_gene_names, missing_genes, superfluous_genes = self._analyze_gene_differences(adata)

        # If no changes needed, return
        if not missing_genes and not superfluous_genes and curr_gene_names == self.gene_names:
            logger.info("No gene changes needed, returning the original data")
            return adata
        else:
            logger.info("Gene order/content differences detected - reordering genes to match data format")

        # If only reordering is needed
        if not missing_genes and not superfluous_genes:
            return self._reorder_genes_only(adata, curr_gene_names)

        return self._handle_gene_differences(adata, curr_gene_names, missing_genes, superfluous_genes)

    def _handle_gene_differences(
        self, adata: AnnData, curr_gene_names: list[str], missing_genes: set[str], superfluous_genes: set[str]
    ) -> AnnData:
        """Handle cases where genes need to be added or removed."""
        logger.info("Handling missing or superfluous genes")

        # Get common genes between current and target gene sets
        common_genes = list(set(curr_gene_names) & set(self.gene_names))
        common_adata = adata[:, common_genes].copy()
        n_obs = adata.n_obs

        # Free memory by deleting original
        if len(superfluous_genes) > 0:
            logger.info(f"Identified {len(superfluous_genes)} superfluous genes for removal")
            del adata

        # Add missing genes if needed
        if missing_genes:
            common_adata = self._add_missing_genes(common_adata, missing_genes, n_obs)

        # Reorder genes to match data_format.gene_names
        return self._reorder_final_genes(common_adata, curr_gene_names, missing_genes, superfluous_genes)

    def _add_missing_genes(self, common_adata: AnnData, missing_genes: set[str], n_obs: int) -> AnnData:
        """Add missing genes as zero columns to the AnnData object."""
        missing_genes_list = list(missing_genes)
        logger.info(f"Adding {len(missing_genes_list)} missing genes")

        # Create a sparse matrix with zeros for missing genes
        missing_X = csr_matrix((n_obs, len(missing_genes_list)), dtype=np.float32)
        missing_var = pd.DataFrame(index=missing_genes_list)
        missing_adata = AnnData(X=missing_X, obs=common_adata.obs, var=missing_var)

        # Concatenate with existing data
        combined_adata = ad.concat(
            [common_adata, missing_adata],
            axis=1,  # Concatenate along the gene axis
            join="outer",
            merge="same",
        )

        # Free memory
        del missing_adata
        del missing_X
        del missing_var
        del common_adata

        # Set for next step
        common_adata = combined_adata
        del combined_adata

        return common_adata

    def _reorder_final_genes(
        self, common_adata: AnnData, curr_gene_names: list[str], missing_genes: set[str], superfluous_genes: set[str]
    ) -> AnnData:
        """Reorder genes to match the final data format order."""
        logger.info("Reordering genes to match data_format.gene_names")

        # Create a mapping from gene name to index
        gene_to_idx = {gene: i for i, gene in enumerate(common_adata.var_names)}

        # Verify all genes from data_format are in the mapping
        missing_from_mapping = set(self.gene_names) - set(gene_to_idx.keys())
        if missing_from_mapping:
            raise ValueError(f"Genes missing from mapping: {missing_from_mapping}")

        # Create a list of indices in the order of data_format.gene_names
        ordered_indices = [gene_to_idx[gene] for gene in self.gene_names]
        final_adata = common_adata[:, ordered_indices].copy()
        del common_adata

        # Verify that all genes are now present in the correct order
        assert final_adata.var_names.tolist() == self.gene_names, (
            "Final gene order does not match data_format.gene_names"
        )

        # Log the net change in gene count
        net_change = final_adata.n_vars - len(curr_gene_names)
        if net_change != 0:
            if net_change > 0:
                logger.info(
                    f"Net result: Added {net_change} genes (removed {len(superfluous_genes)} superfluous, added {len(missing_genes)} missing)"
                )
            else:
                logger.info(
                    f"Net result: Removed {abs(net_change)} genes (removed {len(superfluous_genes)} superfluous, added {len(missing_genes)} missing)"
                )
        else:
            logger.info(
                f"Net result: No change in gene count (removed {len(superfluous_genes)} superfluous, added {len(missing_genes)} missing)"
            )

        return final_adata

    def reorder_genes_to_match_format(self, adata: AnnData) -> AnnData:
        """Reorder genes in AnnData to match the data format gene order.

        This function reorders genes to match self.gene_names, adds missing genes
        as zero columns, and removes extra genes not in the data format.

        Args:
            adata: AnnData object to reorder

        Returns:
            New AnnData object with genes reordered to match data format
        """
        return self.convert_genes_expression_matrix(adata)

    def prepare_adata_for_training(self, adata: AnnData, *, reorder_genes: bool = False) -> AnnData:
        """Prepare AnnData object for training.

        Args:
            adata: AnnData object to prepare
            reorder_genes: If True, reorders genes to match the data format.
            If False, returns the AnnData unchanged.

        Returns:
            AnnData object, optionally with genes reordered to match data format.

        Note:
            - Gene reordering is typically only needed for inference with new data
            - During training, gene ordering is handled efficiently at batch level
            - Preprocessing (normalization, log transform, z-score) is performed
              on-the-fly during batch loading for memory efficiency
        """
        if reorder_genes:
            return self.reorder_genes_to_match_format(adata)
        return adata

    def __str__(self) -> str:
        """String representation of DataFormat with array summaries instead of full arrays."""

        # Helper function to format array info
        def format_array_info(arr: np.ndarray) -> str:
            if arr.size == 0:
                return "empty array"
            return f"array(shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f})"

        # Helper function to format gene names list
        def format_gene_names(names: list[str]) -> str:
            if not names:
                return "[]"
            if len(names) <= 5:
                return str(names)
            return f"[{names[0]}, {names[1]}, ..., {names[-2]}, {names[-1]}] (total: {len(names)})"

        # Helper function to format categorical mappings
        def format_categorical_mappings(mappings: dict[str, dict[str, int]]) -> str:
            if not mappings:
                return "{}"

            formatted_mappings = []
            for key, mapping in mappings.items():
                if len(mapping) <= 3:
                    formatted_mappings.append(f"'{key}': {mapping}")
                else:
                    items = list(mapping.items())
                    sample_items = items[:2] + [("...", "...")] + items[-1:]
                    formatted_mappings.append(f"'{key}': {dict(sample_items)} (total: {len(mapping)} categories)")

            return "{\n    " + ",\n    ".join(formatted_mappings) + "\n  }"

        return f"""DataFormat(
    n_genes={self.n_genes},
    gene_names={format_gene_names(self.gene_names)},
    genes_mu={format_array_info(self.genes_mu)},
    genes_sigma={format_array_info(self.genes_sigma)},
    use_log_transform={self.use_log_transform},
    use_zscore_norm={self.use_zscore_norm},
    target_sum={self.target_sum},
    aux_categorical_types={self.aux_categorical_types},
    aux_categorical_mappings={format_categorical_mappings(self.aux_categorical_mappings)}
)"""


def load_data_format(load_path: Path) -> DataFormat:
    """Load a DataFormat object from saved files.

    Loads normalization parameters and gene metadata from JSON and NPZ files
    created during model training.

    Args:
        load_path: Path to the JSON file (e.g., 'data_format.json').
                Expects corresponding NPZ file with same basename.

    Returns:
        DataFormat object containing preprocessing parameters and gene statistics.

    Example:
        >>> data_format = load_data_format(Path("results/data_format.json"))
        >>> print(f"Loaded {data_format.n_genes} genes")
    """
    logger.info(f"Loading data format from: {load_path}")
    with load_path.open(mode="r") as f:
        data = json.load(f)
    npz_path = load_path.with_suffix(".npz")
    with np.load(npz_path) as npz:
        data["genes_mu"] = npz["genes_mu"]
        data["genes_sigma"] = npz["genes_sigma"]
        logger.info(
            f"Loaded normalization statistics: mu array {npz['genes_mu'].shape}, sigma array {npz['genes_sigma'].shape}"
        )
    return DataFormat.model_validate(data)
