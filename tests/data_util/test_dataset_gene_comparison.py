"""Tests for gene order comparison optimization in dataset.py."""

import numpy as np
import pandas as pd

from anndata import AnnData

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.dataset import CellsDataset


class TestDatasetGeneComparison:
    """Test gene order comparison optimization in CellsDataset."""

    def test_gene_transformation_not_needed_same_order(self):
        """Test that gene transformation is not needed when genes are in the same order."""
        # Create test data
        n_cells, n_genes = 100, 50
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(
            {
                "expansion": ["expanded"] * 50 + ["not_expanded"] * 50,
                "clone_id_size": np.random.randint(1, 10, n_cells),
                "median_clone_size": np.random.randint(1, 5, n_cells),
                "tissue_type": ["tissue1"] * n_cells,
                "imputed_labels": ["label1"] * n_cells,
            }
        )
        var = pd.DataFrame(index=gene_names)
        adata = AnnData(X=X, obs=obs, var=var)

        # Create data format with same gene order
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names,
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            use_zscore_norm=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(data_format=data_format, adata=adata, is_train=False)

        # Should not need gene transformation
        assert not dataset.needs_gene_transformation
        assert dataset.gene_indices is None

    def test_gene_transformation_needed_different_order(self):
        """Test that gene transformation is needed when genes are in different order."""
        # Create test data
        n_cells, n_genes = 100, 50
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        reversed_gene_names = list(reversed(gene_names))

        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(
            {
                "expansion": ["expanded"] * 50 + ["not_expanded"] * 50,
                "clone_id_size": np.random.randint(1, 10, n_cells),
                "median_clone_size": np.random.randint(1, 5, n_cells),
                "tissue_type": ["tissue1"] * n_cells,
                "imputed_labels": ["label1"] * n_cells,
            }
        )
        var = pd.DataFrame(index=reversed_gene_names)  # Different order
        adata = AnnData(X=X, obs=obs, var=var)

        # Create data format with original gene order
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names,
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            use_zscore_norm=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(data_format=data_format, adata=adata, is_train=False)

        # Should need gene transformation
        assert dataset.needs_gene_transformation
        assert dataset.gene_indices is not None
        assert len(dataset.gene_indices) == n_genes

    def test_gene_transformation_needed_missing_genes(self):
        """Test that gene transformation is needed when some genes are missing."""
        # Create test data
        n_cells, n_genes = 100, 50
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        partial_gene_names = gene_names[:30]  # Missing some genes

        X = np.random.rand(n_cells, len(partial_gene_names))
        obs = pd.DataFrame(
            {
                "expansion": ["expanded"] * 50 + ["not_expanded"] * 50,
                "clone_id_size": np.random.randint(1, 10, n_cells),
                "median_clone_size": np.random.randint(1, 5, n_cells),
                "tissue_type": ["tissue1"] * n_cells,
                "imputed_labels": ["label1"] * n_cells,
            }
        )
        var = pd.DataFrame(index=partial_gene_names)
        adata = AnnData(X=X, obs=obs, var=var)

        # Create data format with all genes
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names,
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            use_zscore_norm=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(data_format=data_format, adata=adata, is_train=False)

        # Should need gene transformation
        assert dataset.needs_gene_transformation
        assert len(dataset.missing_genes) == 20  # 50 - 30 = 20 missing genes
        assert len(dataset.extra_genes) == 0

    def test_gene_transformation_needed_extra_genes(self):
        """Test that gene transformation is needed when there are extra genes."""
        # Create test data
        n_cells, n_genes = 100, 50
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        extended_gene_names = gene_names + [f"extra_gene_{i}" for i in range(10)]  # Extra genes

        X = np.random.rand(n_cells, len(extended_gene_names))
        obs = pd.DataFrame(
            {
                "expansion": ["expanded"] * 50 + ["not_expanded"] * 50,
                "clone_id_size": np.random.randint(1, 10, n_cells),
                "median_clone_size": np.random.randint(1, 5, n_cells),
                "tissue_type": ["tissue1"] * n_cells,
                "imputed_labels": ["label1"] * n_cells,
            }
        )
        var = pd.DataFrame(index=extended_gene_names)
        adata = AnnData(X=X, obs=obs, var=var)

        # Create data format with original genes only
        data_format = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names,
            genes_mu=np.zeros(n_genes, dtype=np.float32),
            genes_sigma=np.ones(n_genes, dtype=np.float32),
            use_log_transform=False,
            use_zscore_norm=False,
            target_sum=1e4,
        )

        # Create dataset
        dataset = CellsDataset(data_format=data_format, adata=adata, is_train=False)

        # Should need gene transformation
        assert dataset.needs_gene_transformation
        assert len(dataset.missing_genes) == 0
        assert len(dataset.extra_genes) == 10  # 10 extra genes

    def test_gene_comparison_performance(self):
        """Test that the optimized gene comparison works correctly."""
        # Create large gene lists
        n_genes = 1000
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        # Create pandas Index for efficient comparison
        pd_index = pd.Index(gene_names)

        # Test the optimized comparison
        result_optimized = pd_index.equals(pd.Index(gene_names))

        # Test the list comparison
        result_list = list(pd_index) == gene_names

        # Both should give the same result
        assert result_optimized == result_list
        assert result_optimized is True

        # Test with different order
        different_order = gene_names[::-1]
        result_optimized_diff = pd_index.equals(pd.Index(different_order))
        result_list_diff = list(pd_index) == different_order

        assert result_optimized_diff == result_list_diff
        assert result_optimized_diff is False
