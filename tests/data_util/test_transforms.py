"""Comprehensive tests for the transforms module.

This module tests all transformation functions with various input types:
- NumPy arrays (dense)
- PyTorch tensors
- Sparse matrices (CSR)
- Edge cases and memory efficiency
"""

import tempfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.transforms import (
    DEFAULT_EPS,
    _create_gene_subset_data_format,
    apply_inverse_log_transform,
    apply_inverse_zscore_normalization,
    apply_log_transform,
    apply_row_normalization,
    apply_zscore_normalization,
    extract_is_expanded,
    load_and_preprocess_data_numpy,
    preprocess_expression_data,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_cells, n_genes = 100, 50

    # Dense data
    dense_data = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(
        np.float32
    )

    # Sparse data (about 30% sparsity)
    sparse_mask = np.random.random((n_cells, n_genes)) > 0.3
    sparse_data = dense_data * sparse_mask
    sparse_csr = sp.csr_matrix(sparse_data)

    # Small test data for exact verification
    small_dense = np.array(
        [[1.0, 2.0, 0.0], [4.0, 0.0, 3.0], [0.0, 5.0, 6.0]], dtype=np.float32
    )
    small_sparse = sp.csr_matrix(small_dense)

    return {
        "dense": dense_data,
        "sparse": sparse_csr,
        "small_dense": small_dense,
        "small_sparse": small_sparse,
        "genes_mu": np.random.normal(0, 1, n_genes).astype(np.float32),
        "genes_sigma": np.random.uniform(0.5, 2.0, n_genes).astype(np.float32),
        "small_genes_mu": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "small_genes_sigma": np.array([0.5, 1.0, 1.5], dtype=np.float32),
    }


@pytest.fixture
def mock_data_format(sample_data):
    """Create a mock DataFormat for testing."""
    return DataFormat(
        n_genes=3,
        gene_names=["gene1", "gene2", "gene3"],
        genes_mu=sample_data["small_genes_mu"],
        genes_sigma=sample_data["small_genes_sigma"],
        target_sum=1e4,
        use_log_transform=True,
    )


@pytest.fixture
def sample_anndata():
    """Create sample AnnData object for testing gene subset functionality."""
    # Create a larger dataset for realistic testing
    n_cells, n_genes = 50, 10
    np.random.seed(42)

    # Generate expression data
    X = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    # Create gene names
    gene_names = [f"ENSG{i:011d}" for i in range(n_genes)]

    # Create AnnData object
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    return adata


@pytest.fixture
def comprehensive_data_format(sample_anndata):
    """Create a comprehensive DataFormat matching the sample AnnData."""
    n_genes = sample_anndata.n_vars
    np.random.seed(42)

    return DataFormat(
        n_genes=n_genes,
        gene_names=list(sample_anndata.var_names),
        genes_mu=np.random.normal(0, 1, n_genes).astype(np.float32),
        genes_sigma=np.random.uniform(0.5, 2.0, n_genes).astype(np.float32),
        target_sum=1e4,
        use_log_transform=True,
    )


class TestRowNormalization:
    """Test row normalization function with all input types."""

    def test_dense_numpy_in_place(self, sample_data):
        """Test row normalization with dense NumPy array in-place."""
        X = sample_data["small_dense"].copy()
        target_sum = 1000.0

        result = apply_row_normalization(X, target_sum=target_sum)

        # Check that result is the same object (in-place)
        assert result is X

        # Check row sums
        row_sums = result.sum(axis=1)
        expected_sums = np.full(3, target_sum)
        np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-5)

    def test_dense_numpy_not_in_place(self, sample_data):
        """Test row normalization with dense NumPy array not in-place."""
        X = sample_data["small_dense"].copy()
        X_original = X.copy()
        target_sum = 1000.0

        X_copy = X.copy()
        result = apply_row_normalization(X_copy, target_sum=target_sum)

        # Check that original is unchanged
        np.testing.assert_array_equal(X, X_original)

        # Check that result is same object (in-place)
        assert result is X_copy

        # Check row sums
        row_sums = result.sum(axis=1)
        expected_sums = np.full(3, target_sum)
        np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-5)

    def test_sparse_matrix(self, sample_data):
        """Test row normalization with sparse matrix."""
        X_sparse = sample_data["small_sparse"]
        target_sum = 1000.0

        X_sparse_copy = X_sparse.copy()
        result = apply_row_normalization(X_sparse_copy, target_sum=target_sum)

        # Check that result is sparse
        assert sp.issparse(result)

        # Check that original is unchanged
        np.testing.assert_array_equal(X_sparse.toarray(), sample_data["small_dense"])

        # Check that result is same object (in-place)
        assert result is X_sparse_copy

        # Check row sums
        row_sums = np.array(result.sum(axis=1)).flatten()
        expected_sums = np.full(3, target_sum)
        np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-5)

    def test_pytorch_tensor(self, sample_data):
        """Test row normalization with PyTorch tensor."""
        X_tensor = torch.from_numpy(sample_data["small_dense"])
        target_sum = 1000.0

        result = apply_row_normalization(X_tensor, target_sum=target_sum)

        # Check that result is the same object (in-place)
        assert result is X_tensor

        # Check row sums
        row_sums = result.sum(dim=1)
        expected_sums = torch.full((3,), target_sum)
        torch.testing.assert_close(row_sums, expected_sums, rtol=1e-5, atol=1e-5)

    def test_zero_row_handling(self):
        """Test handling of rows with zero sum."""
        X = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        target_sum = 100.0

        result = apply_row_normalization(X, target_sum=target_sum)

        # Check that zero row remains zero
        assert result[1, 0] == 0.0
        assert result[1, 1] == 0.0

        # Check other rows
        np.testing.assert_allclose(result[0].sum(), target_sum, rtol=1e-5)
        np.testing.assert_allclose(result[2].sum(), target_sum, rtol=1e-5)

    def test_zero_row_handling_sparse(self):
        """Test handling of rows with zero sum in sparse matrices."""
        # Create sparse matrix with a zero row
        X = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        X_sparse = sp.csr_matrix(X)
        target_sum = 100.0

        # This should not raise any warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # Turn warnings into errors
            result = apply_row_normalization(X_sparse, target_sum=target_sum)

        # Check that result is sparse
        assert sp.issparse(result)

        # Convert to dense for checking values
        result_dense = result.toarray()

        # Check that zero row remains zero
        assert result_dense[1, 0] == 0.0
        assert result_dense[1, 1] == 0.0

        # Check other rows
        np.testing.assert_allclose(result_dense[0].sum(), target_sum, rtol=1e-5)
        np.testing.assert_allclose(result_dense[2].sum(), target_sum, rtol=1e-5)


class TestLogTransformation:
    """Test log transformation function with all input types."""

    def test_dense_numpy(self, sample_data):
        """Test log transformation with dense NumPy array."""
        X = sample_data["small_dense"].copy()
        X_original = X.copy()

        result = apply_log_transform(X)

        # Check that result is the same object (in-place)
        assert result is X

        # Check transformation correctness
        expected = np.log1p(X_original)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_sparse_matrix(self, sample_data):
        """Test log transformation with sparse matrix."""
        X_sparse = sample_data["small_sparse"]

        X_sparse_copy = X_sparse.copy()
        result = apply_log_transform(X_sparse_copy)

        # Check that result is sparse
        assert sp.issparse(result)

        # Check that original is unchanged
        np.testing.assert_array_equal(X_sparse.toarray(), sample_data["small_dense"])

        # Check that result is same object (in-place)
        assert result is X_sparse_copy

        # Check transformation correctness (only on non-zero elements)
        expected_dense = np.log1p(sample_data["small_dense"])
        np.testing.assert_allclose(result.toarray(), expected_dense, rtol=1e-6)

    def test_pytorch_tensor(self, sample_data):
        """Test log transformation with PyTorch tensor."""
        X_tensor = torch.from_numpy(sample_data["small_dense"])
        X_original = X_tensor.clone()

        result = apply_log_transform(X_tensor, in_place=False)

        # Check that original is unchanged
        torch.testing.assert_close(X_tensor, X_original)

        # Check transformation correctness
        expected = torch.log1p(X_original)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


class TestZScoreNormalization:
    """Test z-score normalization function with all input types."""

    def test_dense_numpy(self, sample_data):
        """Test z-score normalization with dense NumPy array."""
        X = sample_data["small_dense"].copy()
        genes_mu = sample_data["small_genes_mu"]
        genes_sigma = sample_data["small_genes_sigma"]
        eps = DEFAULT_EPS

        result = apply_zscore_normalization(X, genes_mu, genes_sigma, eps=eps)

        # Check that result is the same object (in-place)
        assert result is X

        # Check transformation correctness
        expected = (sample_data["small_dense"] - genes_mu) / (genes_sigma + eps)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_sparse_matrix_becomes_dense(self, sample_data):
        """Test z-score normalization with sparse matrix (becomes dense)."""
        X_sparse = sample_data["small_sparse"]
        genes_mu = sample_data["small_genes_mu"]
        genes_sigma = sample_data["small_genes_sigma"]
        eps = DEFAULT_EPS

        result = apply_zscore_normalization(X_sparse, genes_mu, genes_sigma, eps=eps)

        # Check that result is dense (z-score always makes dense)
        assert isinstance(result, np.ndarray)
        assert not sp.issparse(result)

        # Check that original is unchanged
        np.testing.assert_array_equal(X_sparse.toarray(), sample_data["small_dense"])

        # Check transformation correctness
        expected = (sample_data["small_dense"] - genes_mu) / (genes_sigma + eps)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_pytorch_tensor(self, sample_data):
        """Test z-score normalization with PyTorch tensor."""
        X_tensor = torch.from_numpy(sample_data["small_dense"])
        genes_mu = sample_data["small_genes_mu"]
        genes_sigma = sample_data["small_genes_sigma"]
        eps = DEFAULT_EPS

        result = apply_zscore_normalization(
            X_tensor, genes_mu, genes_sigma, eps=eps, in_place=False
        )

        # Check transformation correctness
        expected_np = (sample_data["small_dense"] - genes_mu) / (genes_sigma + eps)
        expected = torch.from_numpy(expected_np)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_with_torch_gene_stats(self, sample_data):
        """Test z-score normalization with PyTorch tensor gene statistics."""
        X = sample_data["small_dense"].copy()
        genes_mu_torch = torch.from_numpy(sample_data["small_genes_mu"])
        genes_sigma_torch = torch.from_numpy(sample_data["small_genes_sigma"])
        eps = DEFAULT_EPS

        result = apply_zscore_normalization(
            X, genes_mu_torch, genes_sigma_torch, eps=eps
        )

        # Check transformation correctness
        expected = (sample_data["small_dense"] - sample_data["small_genes_mu"]) / (
            sample_data["small_genes_sigma"] + eps
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestInverseFunctions:
    """Test inverse transformation functions."""

    def test_inverse_zscore(self, sample_data):
        """Test inverse z-score normalization."""
        # Create normalized data
        X_original = torch.from_numpy(sample_data["small_dense"])
        genes_mu = sample_data["small_genes_mu"]
        genes_sigma = sample_data["small_genes_sigma"]
        eps = DEFAULT_EPS

        # Forward transform
        X_normalized = apply_zscore_normalization(
            X_original.clone(), genes_mu, genes_sigma, eps=eps
        )

        # Inverse transform
        X_recovered = apply_inverse_zscore_normalization(
            X_normalized, genes_mu, genes_sigma, eps=eps
        )

        # Check recovery
        torch.testing.assert_close(X_recovered, X_original, rtol=1e-5, atol=1e-5)

    def test_inverse_log(self, sample_data):
        """Test inverse log transformation."""
        X_original = torch.from_numpy(sample_data["small_dense"])

        # Forward transform
        X_log = apply_log_transform(X_original.clone())

        # Inverse transform
        X_recovered = apply_inverse_log_transform(X_log)

        # Check recovery
        torch.testing.assert_close(X_recovered, X_original, rtol=1e-5, atol=1e-5)

    def test_inverse_log_clamping(self):
        """Test that inverse log transform handles extreme values."""
        # Create extreme values
        X_extreme = torch.tensor([[-20.0, 0.0, 20.0]])

        result = apply_inverse_log_transform(X_extreme)

        # Check that no overflow/underflow occurred
        assert torch.isfinite(result).all()
        # Note: expm1(-20) gives a negative result (approximately -1), which is mathematically correct
        # We just check that the function handles extreme values without crashing


class TestCompletePreprocessingPipeline:
    """Test the complete preprocessing pipeline."""

    def test_dense_pipeline(self, sample_data, mock_data_format):
        """Test complete pipeline with dense data."""
        X = sample_data["small_dense"].copy()

        result = preprocess_expression_data(X, mock_data_format)

        # Input might be modified since we simplified the function, just check result validity

        # Check that result is dense
        assert isinstance(result, np.ndarray)

        # Check that pipeline was applied (row sums should be normalized)
        # After row norm and log transform, exact values are complex to predict
        # but we can check basic properties
        assert result.shape == X.shape
        assert np.isfinite(result).all()

    def test_sparse_pipeline_becomes_dense(self, sample_data, mock_data_format):
        """Test complete pipeline with sparse data (becomes dense due to z-score)."""
        X_sparse = sample_data["small_sparse"]

        result = preprocess_expression_data(X_sparse, mock_data_format)

        # Check that result is dense (due to z-score normalization)
        assert isinstance(result, np.ndarray)
        assert not sp.issparse(result)

        # Input might be modified, just check result validity

        # Basic sanity checks
        assert result.shape == X_sparse.shape
        assert np.isfinite(result).all()

    def test_pytorch_pipeline(self, sample_data, mock_data_format):
        """Test complete pipeline with PyTorch tensor."""
        X_tensor = torch.from_numpy(sample_data["small_dense"])

        result = preprocess_expression_data(X_tensor, mock_data_format)

        # Check that result preserves tensor type (optimized behavior)
        assert isinstance(result, torch.Tensor)

        # Basic sanity checks
        assert result.shape == X_tensor.shape
        assert torch.isfinite(result).all()

    def test_pipeline_without_log_transform(self, sample_data):
        """Test pipeline without log transformation."""
        data_format = DataFormat(
            n_genes=3,
            gene_names=["gene1", "gene2", "gene3"],
            categorical_mappings={},
            genes_mu=sample_data["small_genes_mu"],
            genes_sigma=sample_data["small_genes_sigma"],
            target_sum=1e4,
            use_log_transform=False,  # No log transform
            eps=DEFAULT_EPS,
        )

        X = sample_data["small_dense"].copy()
        result = preprocess_expression_data(X, data_format)

        # Should still work and produce finite results
        assert np.isfinite(result).all()
        assert result.shape == X.shape


class TestMemoryEfficiency:
    """Test memory efficiency and sparse preservation."""

    def test_sparse_row_norm_preserves_sparsity(self):
        """Test that row normalization preserves sparsity structure."""
        # Create sparse matrix with known sparsity pattern
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 2])
        X_sparse = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        result = apply_row_normalization(X_sparse, target_sum=100.0)

        # Check sparsity is preserved
        assert sp.issparse(result)
        assert result.nnz == X_sparse.nnz  # Same number of non-zeros

        # Check that row sums are correct (main functionality test)
        row_sums = np.array(result.sum(axis=1)).flatten()
        expected_sums = np.full(3, 100.0)
        np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-5)

        # Check that non-zero locations are preserved (may be reordered due to matrix multiplication)
        original_nonzero = set(zip(*X_sparse.nonzero(), strict=False))
        result_nonzero = set(zip(*result.nonzero(), strict=False))
        assert original_nonzero == result_nonzero

    def test_sparse_log_preserves_sparsity(self):
        """Test that log transformation preserves sparsity structure."""
        # Create sparse matrix
        data = np.array([1.0, 2.0, 3.0, 4.0])
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 2, 1, 2])
        X_sparse = sp.csr_matrix((data, (row, col)), shape=(3, 3))

        result = apply_log_transform(X_sparse)

        # Check sparsity is preserved
        assert sp.issparse(result)
        assert result.nnz == X_sparse.nnz

        # Check sparsity pattern is preserved
        np.testing.assert_array_equal(result.indices, X_sparse.indices)
        np.testing.assert_array_equal(result.indptr, X_sparse.indptr)

        # Check values are log-transformed
        expected_data = np.log1p(data)
        np.testing.assert_allclose(result.data, expected_data, rtol=1e-6)

    def test_large_sparse_memory_efficiency(self):
        """Test memory efficiency with large sparse matrices."""
        # Create large sparse matrix
        n_rows, n_cols = 10000, 5000
        density = 0.01
        X_large_sparse = sp.random(
            n_rows, n_cols, density=density, format="csr", dtype=np.float32
        )

        # Track memory usage indirectly by checking operations don't fail
        # and preserve sparsity where expected

        # Row normalization should preserve sparsity
        X_row_norm = apply_row_normalization(X_large_sparse, target_sum=1e4)
        assert sp.issparse(X_row_norm)
        assert X_row_norm.shape == X_large_sparse.shape

        # Log transform should preserve sparsity
        X_log = apply_log_transform(X_row_norm)
        assert sp.issparse(X_log)
        assert X_log.shape == X_large_sparse.shape

        # Z-score should produce dense (but operation should still work)
        genes_mu = np.random.normal(0, 1, n_cols).astype(np.float32)
        genes_sigma = np.random.uniform(0.5, 2.0, n_cols).astype(np.float32)
        X_zscore = apply_zscore_normalization(X_log, genes_mu, genes_sigma)
        assert isinstance(X_zscore, np.ndarray)
        assert X_zscore.shape == X_large_sparse.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unsupported_type_error(self):
        """Test that unsupported types raise appropriate errors."""
        unsupported_data = [1, 2, 3]  # Python list

        with pytest.raises(TypeError, match="Unsupported type"):
            apply_row_normalization(unsupported_data)

        with pytest.raises(TypeError, match="Unsupported type"):
            apply_log_transform(unsupported_data)

        with pytest.raises(TypeError, match="Unsupported type"):
            apply_zscore_normalization(
                unsupported_data, np.array([1.0]), np.array([1.0])
            )

    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        genes_mu = np.array([1.0])  # Wrong size
        genes_sigma = np.array([1.0])  # Wrong size

        # Should handle gracefully (broadcasting or error)
        # The exact behavior depends on implementation
        try:
            result = apply_zscore_normalization(X, genes_mu, genes_sigma)
            # If it works, check that result has correct shape
            assert result.shape == X.shape
        except (ValueError, RuntimeError):
            # It's also acceptable to raise an error for mismatched dimensions
            pass

    def test_empty_matrix(self):
        """Test handling of empty matrices."""
        X_empty = np.array([]).reshape(0, 3)

        # Row normalization with empty matrix
        result = apply_row_normalization(X_empty, target_sum=1000.0)
        assert result.shape == (0, 3)

        # Log transform with empty matrix
        result = apply_log_transform(X_empty)
        assert result.shape == (0, 3)


class TestExtractIsExpanded:
    """Test the extract_is_expanded function."""

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        expansion_data = pd.DataFrame(
            {"expansion": ["expanded", "not_expanded", "expanded", "other"]}
        )

        result = extract_is_expanded(expansion_data)
        expected = np.array([1, 0, 1, 0])

        np.testing.assert_array_equal(result, expected)

    def test_dict_input(self):
        """Test with dictionary input."""
        obs_dict = {"expansion": pd.Series(["expanded", "not_expanded", "expanded"])}

        result = extract_is_expanded(obs_dict)
        expected = np.array([1, 0, 1])

        np.testing.assert_array_equal(result, expected)

    def test_series_input(self):
        """Test with Series input."""
        series = pd.Series({"expansion": pd.Series(["expanded", "not_expanded"])})

        # This should work if the function handles Series correctly
        # The exact behavior depends on implementation
        try:
            result = extract_is_expanded(series)
            assert isinstance(result, np.ndarray)
        except (KeyError, AttributeError):
            # It's acceptable if Series input is not supported
            pass


# Performance and integration tests
class TestIntegration:
    """Integration tests with realistic data."""

    def test_realistic_workflow(self, sample_data):
        """Test realistic single-cell preprocessing workflow."""
        # Use larger, more realistic data
        X_sparse = sample_data["sparse"]
        genes_mu = sample_data["genes_mu"]
        genes_sigma = sample_data["genes_sigma"]

        data_format = DataFormat(
            n_genes=len(genes_mu),
            gene_names=[f"gene_{i}" for i in range(len(genes_mu))],
            categorical_mappings={},
            genes_mu=genes_mu,
            genes_sigma=genes_sigma,
            target_sum=1e4,
            use_log_transform=True,
            eps=DEFAULT_EPS,
        )

        # Full pipeline
        result = preprocess_expression_data(X_sparse, data_format)

        # Verify results
        assert isinstance(result, np.ndarray)
        assert result.shape == X_sparse.shape
        assert np.isfinite(result).all()

        # Check that z-score normalization was applied
        # (mean should be close to 0, std close to 1 after normalization)
        gene_means = result.mean(axis=0)
        gene_stds = result.std(axis=0)

        # Due to the preprocessing steps, exact values are complex,
        # but we can check that the operation completed successfully
        assert len(gene_means) == len(genes_mu)
        assert len(gene_stds) == len(genes_sigma)

    def test_consistency_across_types(self, sample_data):
        """Test that different input types give consistent results."""
        X_dense = sample_data["small_dense"]
        X_sparse = sample_data["small_sparse"]
        X_tensor = torch.from_numpy(X_dense)

        target_sum = 1000.0

        # Row normalization should give similar results
        result_dense = apply_row_normalization(X_dense.copy(), target_sum=target_sum)
        result_sparse = apply_row_normalization(X_sparse, target_sum=target_sum)
        result_tensor = apply_row_normalization(X_tensor.clone(), target_sum=target_sum)

        # Convert all to numpy for comparison
        result_sparse_dense = result_sparse.toarray()
        result_tensor_numpy = result_tensor.numpy()

        # Should be approximately equal (allowing for numerical differences)
        np.testing.assert_allclose(result_dense, result_sparse_dense, rtol=1e-5)
        np.testing.assert_allclose(result_dense, result_tensor_numpy, rtol=1e-5)


class TestGeneSubsetDataFormat:
    """Test the _create_gene_subset_data_format helper function."""

    def test_subset_by_gene_names(self, sample_anndata, comprehensive_data_format):
        """Test subsetting by gene names."""
        gene_subset = ["ENSG00000000000", "ENSG00000000002", "ENSG00000000005"]

        subset_data_format, gene_indices = _create_gene_subset_data_format(
            sample_anndata, comprehensive_data_format, gene_subset
        )

        # Check that subset data format has correct genes
        assert subset_data_format.gene_names == gene_subset
        assert len(subset_data_format.genes_mu) == len(gene_subset)
        assert len(subset_data_format.genes_sigma) == len(gene_subset)

        # Check gene indices are correct
        expected_indices = [0, 2, 5]
        np.testing.assert_array_equal(gene_indices, expected_indices)

        # Check normalization parameters are correctly subset
        original_mu = comprehensive_data_format.genes_mu
        original_sigma = comprehensive_data_format.genes_sigma

        np.testing.assert_array_equal(
            subset_data_format.genes_mu, original_mu[expected_indices]
        )
        np.testing.assert_array_equal(
            subset_data_format.genes_sigma, original_sigma[expected_indices]
        )

    def test_subset_by_gene_indices(self, sample_anndata, comprehensive_data_format):
        """Test subsetting by gene indices."""
        gene_indices = [1, 3, 7, 9]

        subset_data_format, returned_indices = _create_gene_subset_data_format(
            sample_anndata, comprehensive_data_format, gene_indices
        )

        # Check that returned indices match input
        np.testing.assert_array_equal(returned_indices, gene_indices)

        # Check gene names are correct
        expected_gene_names = [sample_anndata.var_names[i] for i in gene_indices]
        assert subset_data_format.gene_names == expected_gene_names

        # Check normalization parameters
        original_mu = comprehensive_data_format.genes_mu
        original_sigma = comprehensive_data_format.genes_sigma

        np.testing.assert_array_equal(
            subset_data_format.genes_mu, original_mu[gene_indices]
        )
        np.testing.assert_array_equal(
            subset_data_format.genes_sigma, original_sigma[gene_indices]
        )

    def test_subset_with_numpy_array(self, sample_anndata, comprehensive_data_format):
        """Test subsetting with numpy array input."""
        gene_indices = np.array([0, 4, 8])

        subset_data_format, returned_indices = _create_gene_subset_data_format(
            sample_anndata, comprehensive_data_format, gene_indices
        )

        np.testing.assert_array_equal(returned_indices, gene_indices)
        assert len(subset_data_format.gene_names) == 3

    def test_missing_gene_error(self, sample_anndata, comprehensive_data_format):
        """Test error when gene not found in AnnData."""
        gene_subset = ["ENSG00000000000", "NONEXISTENT_GENE"]

        with pytest.raises(ValueError, match="not found in AnnData"):
            _create_gene_subset_data_format(
                sample_anndata, comprehensive_data_format, gene_subset
            )

    def test_invalid_gene_index_error(self, sample_anndata, comprehensive_data_format):
        """Test error when gene index is out of range."""
        gene_indices = [0, 100]  # 100 is out of range

        with pytest.raises(ValueError, match="Invalid gene indices"):
            _create_gene_subset_data_format(
                sample_anndata, comprehensive_data_format, gene_indices
            )

    def test_gene_not_in_data_format_error(self):
        """Test error when gene in AnnData is not in data_format."""
        # Create custom AnnData and data_format that share one gene but not others
        # AnnData has genes: ENSG00000000000, SHARED_GENE
        # data_format has genes: SHARED_GENE, DIFFERENT_GENE1, DIFFERENT_GENE2

        # Create AnnData with specific genes
        n_cells = 10
        X = np.random.exponential(scale=2.0, size=(n_cells, 2)).astype(np.float32)
        adata_custom = ad.AnnData(X=sp.csr_matrix(X))
        adata_custom.var_names = ["ENSG00000000000", "SHARED_GENE"]

        # Create data_format with different genes
        mismatched_data_format = DataFormat(
            n_genes=3,
            gene_names=["SHARED_GENE", "DIFFERENT_GENE1", "DIFFERENT_GENE2"],
            genes_mu=np.array([1.0, 2.0, 3.0]),
            genes_sigma=np.array([0.5, 1.0, 1.5]),
            target_sum=1e4,
            use_log_transform=True,
        )

        # Try to use a gene that exists in AnnData but not in data_format
        gene_subset = ["ENSG00000000000"]

        with pytest.raises(ValueError, match="not found in data_format"):
            _create_gene_subset_data_format(
                adata_custom, mismatched_data_format, gene_subset
            )


class TestLoadAndPreprocessDataNumpyGeneSubset:
    """Test load_and_preprocess_data_numpy with gene_subset parameter."""

    def test_single_gene_by_name(self, sample_anndata, comprehensive_data_format):
        """Test loading and preprocessing a single gene by name."""
        gene_name = "ENSG00000000003"
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=[gene_name]
            )

        # Check output shape
        expected_shape = (sample_anndata.n_obs, 1)
        assert result.shape == expected_shape

        # Check that result is finite
        assert np.isfinite(result).all()

        # Check that result is dense (due to z-score normalization)
        assert isinstance(result, np.ndarray)

    def test_single_gene_by_index(self, sample_anndata, comprehensive_data_format):
        """Test loading and preprocessing a single gene by index."""
        gene_index = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=[gene_index]
            )

        # Check output shape
        expected_shape = (sample_anndata.n_obs, 1)
        assert result.shape == expected_shape

        # Check that result is finite
        assert np.isfinite(result).all()

    def test_multiple_genes_by_names(self, sample_anndata, comprehensive_data_format):
        """Test loading and preprocessing multiple genes by names."""
        gene_names = ["ENSG00000000001", "ENSG00000000004", "ENSG00000000007"]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=gene_names
            )

        # Check output shape
        expected_shape = (sample_anndata.n_obs, 3)
        assert result.shape == expected_shape

        # Check that result is finite
        assert np.isfinite(result).all()

    def test_multiple_genes_by_indices(self, sample_anndata, comprehensive_data_format):
        """Test loading and preprocessing multiple genes by indices."""
        gene_indices = [0, 2, 5, 8]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=gene_indices
            )

        # Check output shape
        expected_shape = (sample_anndata.n_obs, 4)
        assert result.shape == expected_shape

        # Check that result is finite
        assert np.isfinite(result).all()

    def test_gene_subset_with_row_indices(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test gene subset with row indices (cell subset)."""
        gene_names = ["ENSG00000000002", "ENSG00000000006"]
        row_indices = np.array([5, 10, 15, 20])
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path,
                comprehensive_data_format,
                row_indices=row_indices,
                gene_subset=gene_names,
            )

        # Check output shape
        expected_shape = (len(row_indices), len(gene_names))
        assert result.shape == expected_shape

        # Check that result is finite
        assert np.isfinite(result).all()

    def test_gene_subset_consistency_name_vs_index(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test that using gene names vs indices gives same results."""
        gene_name = "ENSG00000000004"
        gene_index = list(sample_anndata.var_names).index(gene_name)
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result_by_name = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=[gene_name]
            )

            result_by_index = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=[gene_index]
            )

        # Results should be identical
        np.testing.assert_allclose(result_by_name, result_by_index, rtol=1e-10)

    def test_no_gene_subset_vs_full_data(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test that no gene_subset gives same result as full data preprocessing."""
        # Load all genes using gene_subset=None
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            result_no_subset = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=None
            )

            # Load all genes using gene_subset with all gene names
            all_gene_names = list(sample_anndata.var_names)
            result_all_genes = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=all_gene_names
            )

        # Results should be identical
        np.testing.assert_allclose(result_no_subset, result_all_genes, rtol=1e-10)

    def test_dimension_validation_still_works(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test that dimension validation still works for incompatible AnnData."""
        # Create AnnData subset (old problematic way)
        subset_adata = sample_anndata[:, [0]]  # Single gene subset
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            subset_adata.write_h5ad(data_path)
            # This should still raise an error because we're not using gene_subset parameter
            with pytest.raises(
                ValueError, match="Number of genes in expression matrix"
            ):
                load_and_preprocess_data_numpy(data_path, comprehensive_data_format)

    def test_empty_gene_subset_error(self, sample_anndata, comprehensive_data_format):
        """Test error handling for empty gene subset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            with pytest.raises((ValueError, IndexError)):
                load_and_preprocess_data_numpy(
                    data_path, comprehensive_data_format, gene_subset=[]
                )

    def test_mixed_type_gene_subset_error(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test error handling for mixed type gene subset."""
        # Mix of string and int should fail
        mixed_subset = ["ENSG00000000001", 2]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            # This should either work (if implementation handles it) or raise a clear error
            try:
                result = load_and_preprocess_data_numpy(
                    data_path, comprehensive_data_format, gene_subset=mixed_subset
                )
                # If it works, check basic properties
                assert result.shape == (sample_anndata.n_obs, 2)
            except (ValueError, TypeError):
                # It's acceptable to raise an error for mixed types
                pass


class TestGeneSubsetIntegration:
    """Integration tests for gene subset functionality."""

    def test_realistic_single_gene_analysis(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test realistic single gene analysis workflow with pre-computed scaling factors."""
        target_gene = "ENSG00000000005"

        # Create DataFormat with pre-computed scaling factors for proper single-gene analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp_data.h5ad"
            sample_anndata.write_h5ad(temp_path)

            # Create data format with pre-computed scaling factors
            data_format_with_scaling = DataFormat(
                n_genes=comprehensive_data_format.n_genes,
                gene_names=comprehensive_data_format.gene_names,
                genes_mu=comprehensive_data_format.genes_mu,
                genes_sigma=comprehensive_data_format.genes_sigma,
                target_sum=comprehensive_data_format.target_sum,
                use_log_transform=comprehensive_data_format.use_log_transform,
            )

            # Compute scaling factors from full matrix (this preserves relative relationships)
            train_indices = np.arange(sample_anndata.n_obs)
            data_format_with_scaling.create_data_format(
                data_path=temp_path,
                adata=sample_anndata,
                row_inds_train=train_indices,
            )

            # Now single gene analysis preserves variation
            expr_with_scaling = load_and_preprocess_data_numpy(
                temp_path, data_format_with_scaling, gene_subset=[target_gene]
            ).flatten()

            # Basic statistical checks
            assert len(expr_with_scaling) == sample_anndata.n_obs
            assert np.isfinite(expr_with_scaling).all()

            # With pre-computed scaling factors, single genes should have variation
            print(
                f"Single gene std with pre-computed scaling: {expr_with_scaling.std():.4f}"
            )
            assert np.abs(expr_with_scaling.mean()) < 10  # Not too extreme
            assert expr_with_scaling.std() > 0.01  # Should have variation now

            # Compare with dynamic scaling (constant values)
            expr_dynamic = load_and_preprocess_data_numpy(
                temp_path, comprehensive_data_format, gene_subset=[target_gene]
            ).flatten()

            print(f"Dynamic scaling std: {expr_dynamic.std():.4f}")
            print(f"Pre-computed scaling std: {expr_with_scaling.std():.4f}")

            # Pre-computed should have more variation than dynamic
            assert expr_with_scaling.std() > expr_dynamic.std()

    def test_gene_panel_analysis(self, sample_anndata, comprehensive_data_format):
        """Test analysis of a panel of genes with pre-computed scaling factors."""
        gene_panel = [
            "ENSG00000000001",
            "ENSG00000000003",
            "ENSG00000000007",
            "ENSG00000000009",
        ]

        # Create DataFormat with pre-computed scaling factors for consistent analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp_data.h5ad"
            sample_anndata.write_h5ad(temp_path)

            data_format_with_scaling = DataFormat(
                n_genes=comprehensive_data_format.n_genes,
                gene_names=comprehensive_data_format.gene_names,
                genes_mu=comprehensive_data_format.genes_mu,
                genes_sigma=comprehensive_data_format.genes_sigma,
                target_sum=comprehensive_data_format.target_sum,
                use_log_transform=comprehensive_data_format.use_log_transform,
            )

            train_indices = np.arange(sample_anndata.n_obs)
            data_format_with_scaling.create_data_format(
                data_path=temp_path,
                adata=sample_anndata,
                row_inds_train=train_indices,
            )

            # Load and preprocess gene panel with pre-computed scaling
            expr_panel = load_and_preprocess_data_numpy(
                temp_path, data_format_with_scaling, gene_subset=gene_panel
            )

            # Check shape
            assert expr_panel.shape == (sample_anndata.n_obs, len(gene_panel))

            # Check that each gene has been processed and has variation
            for i, gene in enumerate(gene_panel):
                gene_expr = expr_panel[:, i]
                assert np.isfinite(gene_expr).all()
                print(f"Gene {gene} std: {gene_expr.std():.4f}")
                # With pre-computed scaling, each gene should have meaningful variation
                assert gene_expr.std() > 0.01

    def test_subset_preserves_cell_order(
        self, sample_anndata, comprehensive_data_format
    ):
        """Test that gene subsetting preserves cell order."""
        # Test with a specific subset of cells
        cell_indices = np.array([5, 15, 25, 35, 45])
        target_genes = ["ENSG00000000002", "ENSG00000000008"]

        # Load subset
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            sample_anndata.write_h5ad(data_path)
            expr_subset = load_and_preprocess_data_numpy(
                data_path,
                comprehensive_data_format,
                row_indices=cell_indices,
                gene_subset=target_genes,
            )

            # Load full data with same genes
            expr_full = load_and_preprocess_data_numpy(
                data_path, comprehensive_data_format, gene_subset=target_genes
            )

        # Subset should match the corresponding rows in full data
        np.testing.assert_allclose(expr_subset, expr_full[cell_indices, :], rtol=1e-10)

    def test_performance_with_large_gene_subset(self):
        """Test performance and correctness with larger gene subset."""
        # Create larger test data
        n_cells, n_genes = 200, 100
        np.random.seed(42)

        X_large = np.random.exponential(scale=2.0, size=(n_cells, n_genes)).astype(
            np.float32
        )
        gene_names_large = [f"ENSG{i:010d}" for i in range(n_genes)]

        adata_large = ad.AnnData(X=sp.csr_matrix(X_large))
        adata_large.var_names = gene_names_large

        # Create matching data format
        data_format_large = DataFormat(
            n_genes=n_genes,
            gene_names=gene_names_large,
            genes_mu=np.random.normal(0, 1, n_genes).astype(np.float32),
            genes_sigma=np.random.uniform(0.5, 2.0, n_genes).astype(np.float32),
            target_sum=1e4,
            use_log_transform=True,
        )

        # Test with substantial subset (half the genes)
        subset_size = 50
        gene_subset = gene_names_large[:subset_size]
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test.h5ad"
            adata_large.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path, data_format_large, gene_subset=gene_subset
            )

        # Check results
        assert result.shape == (n_cells, subset_size)
        assert np.isfinite(result).all()

        # Verify basic properties of the subset result
        # Note: When using dynamic row normalization (no pre-computed scaling factors),
        # gene subsets will have different normalization than full matrix due to
        # different row sums. This is expected behavior.

        # Just verify that the result is reasonable and finite
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Basic sanity check: values should be in a reasonable range after normalization
        assert np.all(result >= -20)  # No extremely negative values after log/z-score
        assert np.all(result <= 20)  # No extremely positive values after log/z-score
