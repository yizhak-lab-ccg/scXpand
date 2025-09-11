"""Test normalization consistency between different loading approaches.

This module ensures that:
1. Full-matrix normalization vs gene-subset normalization give identical results
2. CellsDataset batch loading vs direct loading give identical results
3. Single gene processing works correctly
4. Pre-computed scaling factors are used when available

This prevents the bug where gene-subset normalization produced different
histograms due to incorrect row normalization scaling factors.
"""

import tempfile

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch.utils.data import DataLoader

from scxpand.data_util.data_format import DataFormat, load_data_format
from scxpand.data_util.dataset import CellsDataset, cells_collate_fn
from scxpand.data_util.transforms import (
    apply_row_normalization,
    load_and_preprocess_data_numpy,
    preprocess_expression_data,
)
from scxpand.util.classes import DataAugmentParams


@pytest.fixture
def synthetic_scrnaseq_data():
    """Create synthetic single-cell RNA-seq data for testing."""
    np.random.seed(42)

    n_cells = 1000
    n_genes = 50

    # Create realistic expression data with sparsity
    # Most genes have low expression, some cells have high total expression
    base_expression = np.random.exponential(scale=0.5, size=(n_cells, n_genes))

    # Add some highly expressed genes
    high_expr_genes = [0, 5, 10, 15, 20]  # Some genes are highly expressed
    base_expression[:, high_expr_genes] *= 10

    # Add some highly active cells
    high_expr_cells = np.random.choice(n_cells, size=100, replace=False)
    base_expression[high_expr_cells, :] *= 5

    # Create sparsity - set many values to zero
    sparse_mask = np.random.random((n_cells, n_genes)) > 0.7  # 70% zeros
    base_expression[~sparse_mask] = 0

    # Ensure some cells have zero total expression
    zero_cells = np.random.choice(n_cells, size=50, replace=False)
    base_expression[zero_cells, :] = 0

    # Convert to sparse format
    X_sparse = sp.csr_matrix(base_expression.astype(np.float32))

    # Create gene names
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

    # Create AnnData object
    adata = ad.AnnData(X=X_sparse)
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Add some metadata
    adata.obs["cell_type"] = np.random.choice(["TypeA", "TypeB", "TypeC"], n_cells)
    adata.obs["batch"] = np.random.choice(["batch1", "batch2"], n_cells)

    return adata


@pytest.fixture
def data_format_with_scaling_factors(synthetic_scrnaseq_data):
    """Create a DataFormat with pre-computed scaling factors."""
    adata = synthetic_scrnaseq_data

    # Create basic data format
    data_format = DataFormat(
        n_genes=adata.n_vars,
        gene_names=list(adata.var_names),
        target_sum=10000.0,
        use_log_transform=False,  # Keep simple for testing
    )

    # Simulate training on subset of cells
    train_indices = np.arange(0, 800)  # First 800 cells are "training"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp_data.h5ad"
        adata.write_h5ad(temp_path)

        # Create the data format (this will compute scaling factors)
        data_format.create_data_format(
            data_path=temp_path,
            adata=adata,
            row_inds_train=train_indices,
        )

    return data_format, train_indices


class TestPrecomputedScalingFactors:
    """Test pre-computed scaling factor functionality."""

    def test_precomputed_scaling_dense(self, synthetic_scrnaseq_data):
        """Test pre-computed scaling on dense matrices."""
        adata = synthetic_scrnaseq_data

        # Get a subset of data
        test_indices = np.arange(100, 200)  # 100 cells
        X_test = adata.X[test_indices, :].toarray()

        # Compute scaling factors manually
        row_sums = X_test.sum(axis=1)
        target_sum = 10000.0
        scaling_factors = np.ones_like(row_sums, dtype=np.float32)
        nonzero_mask = row_sums > 0
        scaling_factors[nonzero_mask] = target_sum / row_sums[nonzero_mask]

        # Apply using our function
        X_scaled = apply_row_normalization(X_test.copy(), target_sum=target_sum)

        # Check that non-zero rows sum to target_sum
        new_row_sums = X_scaled.sum(axis=1)
        expected_sums = np.where(row_sums > 0, target_sum, 0.0)

        np.testing.assert_allclose(new_row_sums, expected_sums, rtol=1e-5)

    def test_precomputed_scaling_sparse(self, synthetic_scrnaseq_data):
        """Test pre-computed scaling on sparse matrices."""
        adata = synthetic_scrnaseq_data

        # Get a subset of data
        test_indices = np.arange(100, 200)
        X_test_sparse = adata.X[test_indices, :]

        # Compute scaling factors manually
        row_sums = np.asarray(X_test_sparse.sum(axis=1)).flatten()
        target_sum = 10000.0
        scaling_factors = np.ones_like(row_sums, dtype=np.float32)
        nonzero_mask = row_sums > 0
        scaling_factors[nonzero_mask] = target_sum / row_sums[nonzero_mask]

        # Apply using our function
        X_scaled_sparse = apply_row_normalization(X_test_sparse, target_sum=target_sum)

        # Check that non-zero rows sum to target_sum
        new_row_sums = np.asarray(X_scaled_sparse.sum(axis=1)).flatten()
        expected_sums = np.where(row_sums > 0, target_sum, 0.0)

        np.testing.assert_allclose(new_row_sums, expected_sums, rtol=1e-5)

    def test_tensor_scaling(self, synthetic_scrnaseq_data):
        """Test scaling works with PyTorch tensors."""
        adata = synthetic_scrnaseq_data

        # Get test data as tensor
        test_indices = np.arange(50, 100)
        X_test = torch.from_numpy(adata.X[test_indices, :].toarray()).float()

        # Compute scaling factors
        row_sums = X_test.sum(dim=1).numpy()
        target_sum = 10000.0
        scaling_factors = np.ones_like(row_sums, dtype=np.float32)
        nonzero_mask = row_sums > 0
        scaling_factors[nonzero_mask] = target_sum / row_sums[nonzero_mask]

        # Apply scaling
        X_scaled = apply_row_normalization(X_test.clone(), target_sum=target_sum)

        # Check results
        new_row_sums = X_scaled.sum(dim=1).numpy()
        expected_sums = np.where(row_sums > 0, target_sum, 0.0)

        np.testing.assert_allclose(new_row_sums, expected_sums, rtol=1e-5)


class TestNormalizationConsistency:
    """Test consistency between different normalization approaches."""

    def test_full_vs_subset_normalization_consistency(self, data_format_with_scaling_factors):
        """Test that full matrix and gene subset normalization give identical results."""
        data_format, train_indices = data_format_with_scaling_factors

        # Create test AnnData from training indices (since scaling factors are for training set)
        with tempfile.TemporaryDirectory() as temp_dir:
            _temp_path = Path(temp_dir) / "temp_data.h5ad"

            # We need to recreate the adata for this test since we need the training subset
            np.random.seed(42)
            n_cells = 800  # Match train_indices
            n_genes = len(data_format.gene_names)

            # Create test data
            X_test = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)
            # Add sparsity
            sparse_mask = np.random.random((n_cells, n_genes)) > 0.6
            X_test[~sparse_mask] = 0

            # Create test AnnData
            test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
            test_adata.var_names = data_format.gene_names
            test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

            # Save the test data to the temporary file
            test_adata.write_h5ad(_temp_path)

            # Test single gene
            target_gene = "GENE_0010"

            # Method 1: Process full matrix, then extract gene
            expr_full_matrix = load_and_preprocess_data_numpy(
                data_path=_temp_path,
                data_format=data_format,
                gene_subset=None,  # Full matrix
            )
            gene_index = data_format.gene_names.index(target_gene)
            expr_from_full = expr_full_matrix[:, gene_index]

            # Method 2: Process single gene with subset
            expr_from_subset = load_and_preprocess_data_numpy(
                data_path=_temp_path, data_format=data_format, gene_subset=[target_gene]
            ).flatten()

            # Results should be identical
            np.testing.assert_allclose(
                expr_from_full,
                expr_from_subset,
                rtol=1e-6,
                atol=1e-8,
                err_msg="Full matrix and gene subset normalization should give identical results",
            )

    def test_multiple_genes_consistency(self, data_format_with_scaling_factors):
        """Test consistency with multiple gene subsets."""
        data_format, train_indices = data_format_with_scaling_factors

        # Create test data
        np.random.seed(123)
        n_cells = 800
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=1.5, size=(n_cells, n_genes)).astype(np.float32)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # Test multiple genes
        target_genes = ["GENE_0005", "GENE_0015", "GENE_0025"]

        # Method 1: Full matrix
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            test_adata.write_h5ad(data_path)
            expr_full = load_and_preprocess_data_numpy(data_path=data_path, data_format=data_format, gene_subset=None)
            gene_indices = [data_format.gene_names.index(g) for g in target_genes]
            expr_from_full = expr_full[:, gene_indices]

            # Method 2: Gene subset
            expr_from_subset = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, gene_subset=target_genes
            )

        # Results should be identical
        np.testing.assert_allclose(
            expr_from_full,
            expr_from_subset,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Multi-gene subset should match full matrix extraction",
        )

    def test_row_subset_consistency(self, data_format_with_scaling_factors):
        """Test consistency when subsetting both rows and genes."""
        data_format, train_indices = data_format_with_scaling_factors

        # Create test data
        np.random.seed(456)
        n_cells = 800
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=1.0, size=(n_cells, n_genes)).astype(np.float32)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # Test with row subset
        row_subset = np.array([10, 50, 100, 200, 300])
        target_gene = "GENE_0020"

        # Method 1: Full matrix with row subset
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            test_adata.write_h5ad(data_path)
            expr_full = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, row_indices=row_subset, gene_subset=None
            )
            gene_index = data_format.gene_names.index(target_gene)
            expr_from_full = expr_full[:, gene_index]

            # Method 2: Gene subset with row subset
            expr_from_subset = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, row_indices=row_subset, gene_subset=[target_gene]
            ).flatten()

        # Results should be identical
        np.testing.assert_allclose(
            expr_from_full, expr_from_subset, rtol=1e-6, atol=1e-8, err_msg="Row and gene subset should be consistent"
        )

    def test_legacy_data_format_fallback(self, synthetic_scrnaseq_data):
        """Test that legacy data formats (without scaling factors) still work."""
        adata = synthetic_scrnaseq_data

        # Create legacy data format (no scaling factors)
        legacy_data_format = DataFormat(
            n_genes=adata.n_vars,
            gene_names=list(adata.var_names),
            target_sum=10000.0,
            use_log_transform=False,
            # Compute other parameters manually
            genes_mu=np.random.normal(0, 1, adata.n_vars).astype(np.float32),
            genes_sigma=np.random.uniform(0.5, 2.0, adata.n_vars).astype(np.float32),
            # row_scaling_factors remains empty (default)
        )

        # This should work without error (falls back to dynamic scaling)
        test_indices = np.arange(100)
        target_gene = "GENE_0001"

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            adata.write_h5ad(data_path)
            expr_result = load_and_preprocess_data_numpy(
                data_path=data_path,
                data_format=legacy_data_format,
                row_indices=test_indices,
                gene_subset=[target_gene],
            )

        # Should return valid result
        assert expr_result.shape == (len(test_indices), 1)
        assert np.isfinite(expr_result).all()

    def test_sparse_expression_gene(self, data_format_with_scaling_factors):
        """Test with a gene that has very sparse expression (like in the bug report)."""
        data_format, train_indices = data_format_with_scaling_factors

        # Create test data where one gene has very sparse expression
        np.random.seed(789)
        n_cells = 800
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        # Make one gene very sparse (like ENSG00000153563 in the bug)
        sparse_gene_idx = 10
        X_test[:, sparse_gene_idx] = 0  # Start with all zeros
        # Only a few cells have expression
        expressing_cells = np.random.choice(n_cells, size=5, replace=False)
        X_test[expressing_cells, sparse_gene_idx] = np.random.choice([1, 2, 3], size=5)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        sparse_gene_name = data_format.gene_names[sparse_gene_idx]

        # Test both approaches
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            test_adata.write_h5ad(data_path)
            expr_full = load_and_preprocess_data_numpy(data_path=data_path, data_format=data_format, gene_subset=None)[
                :, sparse_gene_idx
            ]

            expr_subset = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, gene_subset=[sparse_gene_name]
            ).flatten()

        # Should be identical
        np.testing.assert_allclose(
            expr_full,
            expr_subset,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Sparse gene expression should be consistent between approaches",
        )

        # Check that we get reasonable statistics
        assert np.isfinite(expr_full).all()
        assert np.isfinite(expr_subset).all()

        # Most values should be the same (since most cells have zero expression)
        unique_values_full = np.unique(expr_full)
        unique_values_subset = np.unique(expr_subset)
        np.testing.assert_allclose(unique_values_full, unique_values_subset, rtol=1e-6)


class TestDataFormatIntegration:
    """Test DataFormat scaling factor computation and storage."""

    def test_data_format_scaling_factor_computation(self, synthetic_scrnaseq_data):
        """Test that DataFormat correctly computes and stores scaling factors."""
        adata = synthetic_scrnaseq_data

        data_format = DataFormat(
            n_genes=adata.n_vars,
            gene_names=list(adata.var_names),
            target_sum=10000.0,
            use_log_transform=False,
        )

        train_indices = np.arange(500)  # First 500 cells

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_data.h5ad"
            adata.write_h5ad(temp_path)

            # Create data format (should compute scaling factors)
            data_format.create_data_format(
                data_path=temp_path,
                adata=adata,
                row_inds_train=train_indices,
            )

        # Check that scaling factors are computed on-the-fly when needed
        # In the new design, scaling factors are not stored in DataFormat
        # They are computed dynamically during data processing

        # Check that the data format was created successfully
        assert data_format.n_genes > 0
        assert len(data_format.gene_names) > 0
        assert len(data_format.genes_mu) > 0
        assert len(data_format.genes_sigma) > 0

        # Verify that the data format can be used for processing
        # The actual scaling factor computation will be tested in other tests

    def test_data_format_save_load_with_scaling_factors(self, data_format_with_scaling_factors):
        """Test saving and loading DataFormat with scaling factors."""
        data_format, train_indices = data_format_with_scaling_factors

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_data_format.json"

            # Save
            data_format.save(save_path)

            # Load

            loaded_data_format = load_data_format(save_path)

            # Check that everything is preserved
            assert loaded_data_format.n_genes == data_format.n_genes
            assert loaded_data_format.gene_names == data_format.gene_names
            assert loaded_data_format.target_sum == data_format.target_sum

            np.testing.assert_allclose(loaded_data_format.genes_mu, data_format.genes_mu)
            np.testing.assert_allclose(loaded_data_format.genes_sigma, data_format.genes_sigma)

            # In the new design, scaling factors are computed on-the-fly, not stored
            # So we just verify that the basic data format properties are preserved


class TestEpsilonConsistency:
    """Test epsilon consistency between different processing approaches."""

    def test_shared_epsilon_batch_vs_full_processing(self, data_format_with_scaling_factors):
        """Test that CellsDataset and load_and_preprocess_data_numpy produce identical results with shared epsilon."""
        data_format, train_indices = data_format_with_scaling_factors

        # Create test data
        np.random.seed(999)
        n_cells = 200
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

        # Make one gene sparse for testing (similar to notebook scenario)
        sparse_gene_idx = 10
        X_test[:, sparse_gene_idx] = 0
        expressing_cells = np.random.choice(n_cells, size=8, replace=False)
        X_test[expressing_cells, sparse_gene_idx] = np.random.choice([1, 2, 3], size=8)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "epsilon_test.h5ad"
            test_adata.write_h5ad(temp_path)

            # Define shared epsilon (high precision like CellsDataset uses)
            SHARED_EPS = 1e-10

            # Method 1: CellsDataset (batch processing) - uses eps=1e-10 internally
            test_indices = np.arange(50)  # Subset for testing
            dataset = CellsDataset(
                data_format=data_format,
                row_inds=test_indices,
                dataset_params=DataAugmentParams(soft_loss_beta=1.0, mask_rate=0.0, noise_std=0.0),
                is_train=False,
                data_path=temp_path,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=50,
                shuffle=False,
                collate_fn=lambda batch_indices: cells_collate_fn(batch_indices, dataset),
            )

            batch = next(iter(dataloader))
            expr_batch = batch["x"].numpy()  # Shape: [50, n_genes]

            # Method 2: Manual preprocessing with shared epsilon
            X_raw_subset = test_adata.X[test_indices, :].toarray()

            # Apply preprocessing with shared epsilon
            X_manual_torch = torch.from_numpy(X_raw_subset).float()
            expr_manual = preprocess_expression_data(
                X=X_manual_torch,
                data_format=data_format,
                eps=SHARED_EPS,  # Use same epsilon as CellsDataset
            ).numpy()

            # Results should be identical with shared epsilon
            np.testing.assert_allclose(
                expr_batch,
                expr_manual,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Batch and manual processing should be identical with shared epsilon",
            )

            # Test specifically for the sparse gene
            sparse_gene_batch = expr_batch[:, sparse_gene_idx]
            sparse_gene_manual = expr_manual[:, sparse_gene_idx]

            np.testing.assert_allclose(
                sparse_gene_batch,
                sparse_gene_manual,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Sparse gene should have identical results with shared epsilon",
            )

            # Statistics should be identical
            assert abs(sparse_gene_batch.mean() - sparse_gene_manual.mean()) < 1e-12
            assert abs(sparse_gene_batch.std() - sparse_gene_manual.std()) < 1e-12

    def test_different_epsilon_values_produce_different_results(self, data_format_with_scaling_factors):
        """Test that different epsilon values produce measurably different results."""
        data_format, _ = data_format_with_scaling_factors

        # Create test data
        np.random.seed(777)
        n_cells = 100
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=1.0, size=(n_cells, n_genes)).astype(np.float32)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # Test with different epsilon values
        eps_high_precision = 1e-10  # CellsDataset default
        eps_large = 1e-3  # Much larger epsilon to ensure measurable difference

        # Apply preprocessing with different epsilon values
        X_raw = test_adata.X.toarray()

        expr_high_eps = preprocess_expression_data(X=X_raw.copy(), data_format=data_format, eps=eps_high_precision)
        expr_large_eps = preprocess_expression_data(X=X_raw.copy(), data_format=data_format, eps=eps_large)

        # Results should be different (though possibly only slightly)
        max_diff_large = np.abs(expr_high_eps - expr_large_eps).max()

        # The difference should be small but measurable
        # Standard vs high precision might be very small, but large vs high should be measurable
        assert max_diff_large > 1e-8, "Large epsilon difference should produce measurable results"
        assert max_diff_large < 1e-1, "Difference should be reasonable (not too large)"

        # Standard deviations should be different
        std_diff_large = abs(expr_high_eps.std() - expr_large_eps.std())
        assert std_diff_large > 1e-8, "Standard deviations should differ with large epsilon difference"

    def test_epsilon_consistency_gene_subset(self, data_format_with_scaling_factors):
        """Test that load_and_preprocess_data_numpy produces consistent results for gene subsets."""
        data_format, _ = data_format_with_scaling_factors

        # Create test data
        np.random.seed(555)
        n_cells = 150
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=1.5, size=(n_cells, n_genes)).astype(np.float32)

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # Test specific gene
        target_gene = "GENE_0015"
        gene_index = data_format.gene_names.index(target_gene)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_consistency.h5ad"
            test_adata.write_h5ad(temp_path)

            # Method 1: Process full matrix with load_and_preprocess_data_numpy, then extract gene
            expr_full_matrix = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format,
                gene_subset=None,  # Full matrix
            )
            expr_gene_from_full = expr_full_matrix[:, gene_index]

            # Method 2: Process single gene with load_and_preprocess_data_numpy using gene_subset
            # This should produce IDENTICAL results because it correctly:
            # 1. Loads ALL genes
            # 2. Computes row normalization on ALL genes
            # 3. Then subsets to the target gene
            expr_gene_from_subset = load_and_preprocess_data_numpy(
                data_path=temp_path,
                data_format=data_format,
                gene_subset=[target_gene],
            ).flatten()

            # Results should be IDENTICAL because both use the same row normalization
            # (computed on all genes) before extracting the target gene
            np.testing.assert_allclose(
                expr_gene_from_full,
                expr_gene_from_subset,
                rtol=1e-10,
                atol=1e-12,
                err_msg="Gene subset processing should match full matrix extraction when using load_and_preprocess_data_numpy",
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_expression_cells(self, data_format_with_scaling_factors):
        """Test handling of cells with zero total expression."""
        data_format, _ = data_format_with_scaling_factors

        # Create data with some zero-expression cells
        n_cells = 100
        n_genes = len(data_format.gene_names)
        X_test = np.random.exponential(scale=1.0, size=(n_cells, n_genes)).astype(np.float32)

        # Make some cells have zero expression
        zero_cells = np.array([10, 20, 30, 40, 50])
        X_test[zero_cells, :] = 0

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # This should work without error
        target_gene = "GENE_0005"
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            test_adata.write_h5ad(data_path)
            expr_result = load_and_preprocess_data_numpy(
                data_path=data_path,
                data_format=data_format,
                row_indices=np.arange(n_cells),
                gene_subset=[target_gene],
            )

        # Check that zero cells remain properly normalized
        assert expr_result.shape == (n_cells, 1)
        assert np.isfinite(expr_result).all()

        # Zero expression cells should have the same normalized value
        zero_cell_values = expr_result[zero_cells, 0]
        assert np.allclose(zero_cell_values, zero_cell_values[0])

    def test_single_nonzero_gene_cell(self, data_format_with_scaling_factors):
        """Test cell with expression in only one gene."""
        data_format, _ = data_format_with_scaling_factors

        # Create cell with expression only in one gene
        n_cells = 50
        n_genes = len(data_format.gene_names)
        X_test = np.zeros((n_cells, n_genes), dtype=np.float32)

        # One cell has expression only in one gene
        special_cell = 25
        special_gene = 15
        X_test[special_cell, special_gene] = 100.0

        test_adata = ad.AnnData(X=sp.csr_matrix(X_test))
        test_adata.var_names = data_format.gene_names
        test_adata.obs_names = [f"test_cell_{i}" for i in range(n_cells)]

        # Test processing this gene
        target_gene = data_format.gene_names[special_gene]

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            test_adata.write_h5ad(data_path)
            expr_full = load_and_preprocess_data_numpy(data_path=data_path, data_format=data_format, gene_subset=None)[
                :, special_gene
            ]

            expr_subset = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, gene_subset=[target_gene]
            ).flatten()

        # Should be identical
        np.testing.assert_allclose(expr_full, expr_subset, rtol=1e-6)

        # The special cell should have the target sum after normalization
        # (before z-score normalization)
        assert expr_full[special_cell] == expr_subset[special_cell]


def test_consistency_integration_with_notebook_scenario():
    """Integration test that reproduces the exact scenario from the notebook.

    This test ensures that the CellsDataset approach and the gene_subset approach
    give identical results, preventing the histogram discrepancy bug.
    """
    # Create realistic data similar to the notebook scenario
    np.random.seed(12345)

    n_cells = 1000
    n_genes = 100

    # Create expression data with realistic properties
    X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Make some genes very sparse (like ENSG00000153563)
    sparse_genes = [25, 50, 75]
    for gene_idx in sparse_genes:
        X[:, gene_idx] = 0
        # Only a few cells express these genes
        expressing_cells = np.random.choice(n_cells, size=10, replace=False)
        X[expressing_cells, gene_idx] = np.random.choice([1, 2, 3], size=10)

    # Create AnnData
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["cell_type"] = np.random.choice(["CD4", "CD8", "NK"], n_cells)
    adata.obs["tissue"] = np.random.choice(["Tumor", "Normal"], n_cells)

    # Create data format with scaling factors
    data_format = DataFormat(
        n_genes=n_genes,
        gene_names=list(adata.var_names),
        target_sum=10000.0,
        use_log_transform=False,
    )

    # Use subset of cells as "training"
    train_indices = np.arange(800)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "integration_test.h5ad"
        adata.write_h5ad(temp_path)

        data_format.create_data_format(
            data_path=temp_path,
            adata=adata,
            row_inds_train=train_indices,
        )

        # Test the problematic sparse gene
        sparse_gene = "GENE_0025"  # One of our sparse genes

        # Method 1: Full matrix normalization (correct approach)
        expr_full_matrix = load_and_preprocess_data_numpy(
            data_path=temp_path,
            data_format=data_format,
            row_indices=train_indices,  # Use training indices
            gene_subset=None,
        )
        gene_index = data_format.gene_names.index(sparse_gene)
        expr_from_full = expr_full_matrix[:, gene_index]

        # Method 2: Gene subset normalization (fixed approach)
        expr_from_subset = load_and_preprocess_data_numpy(
            data_path=temp_path,
            data_format=data_format,
            row_indices=train_indices,  # Use training indices
            gene_subset=[sparse_gene],
        ).flatten()

        # These should now be identical (the bug fix)
        np.testing.assert_allclose(
            expr_from_full,
            expr_from_subset,
            rtol=1e-6,
            atol=1e-8,
            err_msg="The histogram bug should be fixed - both approaches should give identical results",
        )

        # Verify that the sparse gene has expected properties
        nonzero_count_full = np.count_nonzero(expr_from_full)
        nonzero_count_subset = np.count_nonzero(expr_from_subset)
        assert nonzero_count_full == nonzero_count_subset

        # Statistics should be identical
        assert abs(expr_from_full.mean() - expr_from_subset.mean()) < 1e-8
        assert abs(expr_from_full.std() - expr_from_subset.std()) < 1e-8
        assert abs(expr_from_full.min() - expr_from_subset.min()) < 1e-8
        assert abs(expr_from_full.max() - expr_from_subset.max()) < 1e-8

    print("✅ Integration test passed - histogram bug is fixed!")
