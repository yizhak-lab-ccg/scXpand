"""Tests for the normalization module functions."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from scxpand.data_util.data_format import DataFormat
from scxpand.data_util.transforms import (
    apply_log_transform,
    apply_row_normalization,
    apply_zscore_normalization,
    load_and_preprocess_data_numpy,
    preprocess_expression_data,
)


class TestRowNormalization:
    """Test row normalization functions."""

    def test_numpy_row_normalization_basic(self):
        """Test basic numpy row normalization."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        target_sum = 100.0

        # Test in-place
        X_copy = X.copy()
        result = apply_row_normalization(X_copy, target_sum=target_sum)

        assert result is X_copy  # Should return same object when
        assert np.allclose(result.sum(axis=1), target_sum)

    def test_numpy_row_normalization_copy(self):
        """Test numpy row normalization with copy."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        target_sum = 100.0

        # Test with copy (since function is always in-place)
        X_copy = X.copy()
        result = apply_row_normalization(X_copy, target_sum=target_sum)

        assert result is X_copy  # Always returns same object (in-place)
        assert np.allclose(result.sum(axis=1), target_sum)
        assert not np.allclose(
            X.sum(axis=1), target_sum
        )  # Original should be unchanged

    def test_torch_row_normalization_basic(self):
        """Test basic torch row normalization."""
        X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        target_sum = 100.0

        # Test in-place
        X_copy = X.clone()
        result = apply_row_normalization(X_copy, target_sum=target_sum)

        assert result is X_copy  # Should return same object when
        assert torch.allclose(result.sum(dim=1), torch.tensor(target_sum))

    def test_torch_row_normalization_copy(self):
        """Test torch row normalization with copy."""
        X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        target_sum = 100.0

        # Test with copy (since function is always in-place)
        X_copy = X.clone()
        result = apply_row_normalization(X_copy, target_sum=target_sum)

        assert result is X_copy  # Always returns same object (in-place)
        assert torch.allclose(result.sum(dim=1), torch.tensor(target_sum))
        assert not torch.allclose(
            X.sum(dim=1), torch.tensor(target_sum)
        )  # Original should be unchanged

    def test_zero_row_handling(self):
        """Test handling of rows with zero sum."""
        X_np = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        X_torch = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.float32)
        target_sum = 100.0

        # Test numpy
        result_np = apply_row_normalization(X_np.copy(), target_sum=target_sum)
        assert result_np[0].sum() == 0  # Zero row should remain zero
        assert np.allclose(result_np[1].sum(), target_sum)

        # Test torch
        result_torch = apply_row_normalization(X_torch.clone(), target_sum=target_sum)
        assert result_torch[0].sum() == 0  # Zero row should remain zero
        assert torch.allclose(result_torch[1].sum(), torch.tensor(target_sum))


class TestLogTransform:
    """Test log transformation functions."""

    def test_numpy_log_transform(self):
        """Test numpy log transformation."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        # Test in-place
        X_copy = X.copy()
        result = apply_log_transform(X_copy)
        expected = np.log1p(X)

        assert result is X_copy
        assert np.allclose(result, expected)

    def test_torch_log_transform(self):
        """Test torch log transformation."""
        X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

        # Test in-place
        X_copy = X.clone()
        result = apply_log_transform(X_copy)
        expected = torch.log1p(X)

        assert result is X_copy
        assert torch.allclose(result, expected)

    def test_log_transform_copy(self):
        """Test log transformation with copy."""
        X_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        X_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

        # Test numpy copy
        result_np = apply_log_transform(X_np, in_place=False)
        assert result_np is not X_np
        assert np.allclose(result_np, np.log1p(X_np))
        assert not np.allclose(X_np, np.log1p(X_np))  # Original unchanged

        # Test torch copy
        result_torch = apply_log_transform(X_torch, in_place=False)
        assert result_torch is not X_torch
        assert torch.allclose(result_torch, torch.log1p(X_torch))


class TestZScoreNormalization:
    """Test z-score normalization functions."""

    def test_numpy_zscore_normalization(self):
        """Test numpy z-score normalization."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        genes_mu = np.array([2.5, 3.5, 4.5], dtype=np.float32)
        genes_sigma = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        eps = 1e-6

        # Test in-place
        X_copy = X.copy()
        result = apply_zscore_normalization(X_copy, genes_mu, genes_sigma, eps=eps)
        expected = (X - genes_mu) / (genes_sigma + eps)

        assert result is X_copy
        assert np.allclose(result, expected)

    def test_torch_zscore_normalization(self):
        """Test torch z-score normalization."""
        X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        genes_mu = torch.tensor([2.5, 3.5, 4.5], dtype=torch.float32)
        genes_sigma = torch.tensor([1.5, 1.5, 1.5], dtype=torch.float32)
        eps = 1e-6

        # Test in-place
        X_copy = X.clone()
        result = apply_zscore_normalization(X_copy, genes_mu, genes_sigma, eps=eps)
        expected = (X - genes_mu) / (genes_sigma + eps)

        assert result is X_copy
        assert torch.allclose(result, expected)

    def test_zscore_prevents_division_by_zero(self):
        """Test that eps prevents division by zero."""
        X_np = np.array([[1, 2, 3]], dtype=np.float32)
        X_torch = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        genes_mu = np.array([1, 2, 3], dtype=np.float32)
        genes_sigma = np.array([0, 0, 0], dtype=np.float32)  # Zero std
        genes_mu_torch = torch.tensor([1, 2, 3], dtype=torch.float32)
        genes_sigma_torch = torch.tensor([0, 0, 0], dtype=torch.float32)
        eps = 1e-6

        # Should not raise error due to eps
        result_np = apply_zscore_normalization(X_np, genes_mu, genes_sigma, eps=eps)
        result_torch = apply_zscore_normalization(
            X_torch, genes_mu_torch, genes_sigma_torch, eps=eps
        )

        assert not np.any(np.isnan(result_np))
        assert not torch.any(torch.isnan(result_torch))


class TestNumpyTorchConsistency:
    """Test that numpy and torch functions produce consistent results."""

    def test_row_normalization_consistency(self):
        """Test that numpy and torch row normalization give same results."""
        X_np = np.random.rand(10, 5).astype(np.float32)
        X_torch = torch.from_numpy(X_np.copy())
        target_sum = 1000.0

        result_np = apply_row_normalization(X_np.copy(), target_sum=target_sum)
        result_torch = apply_row_normalization(X_torch.clone(), target_sum=target_sum)

        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-6)

    def test_log_transform_consistency(self):
        """Test that numpy and torch log transform give same results."""
        X_np = np.random.rand(10, 5).astype(np.float32)
        X_torch = torch.from_numpy(X_np.copy())

        result_np = apply_log_transform(X_np)
        result_torch = apply_log_transform(X_torch)

        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-6)

    def test_zscore_normalization_consistency(self):
        """Test that numpy and torch z-score normalization give same results."""
        X_np = np.random.rand(10, 5).astype(np.float32)
        X_torch = torch.from_numpy(X_np.copy())
        genes_mu = np.random.rand(5).astype(np.float32)
        genes_sigma = np.random.rand(5).astype(np.float32) + 0.1
        genes_mu_torch = torch.from_numpy(genes_mu.copy())
        genes_sigma_torch = torch.from_numpy(genes_sigma.copy())
        eps = 1e-6

        result_np = apply_zscore_normalization(X_np, genes_mu, genes_sigma, eps=eps)
        result_torch = apply_zscore_normalization(
            X_torch, genes_mu_torch, genes_sigma_torch, eps=eps
        )

        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-6)


class TestCompletePreprocessingPipeline:
    """Test complete preprocessing pipeline functions."""

    @pytest.fixture
    def mock_data_format(self):
        """Create a mock DataFormat for testing."""
        return DataFormat(
            gene_names=["gene1", "gene2", "gene3", "gene4", "gene5"],
            n_genes=5,
            genes_mu=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            genes_sigma=np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            target_sum=1e4,
            use_log_transform=True,
            eps=1e-6,
            aux_categorical_types=(),
        )

    def test_preprocess_expression_data(self, mock_data_format):
        """Test complete numpy preprocessing pipeline."""
        X = np.random.rand(10, 5).astype(np.float32) * 100  # Raw counts

        # Test in-place
        X_copy = X.copy()
        result = preprocess_expression_data(X=X_copy, data_format=mock_data_format)

        assert result is X_copy
        assert result.dtype == np.float32
        assert result.shape == X.shape

        # Verify preprocessing steps were applied
        assert not np.allclose(result, X)  # Should be different from original

    def test_preprocess_expression_data_torch(self, mock_data_format):
        """Test complete torch preprocessing pipeline."""
        X = torch.rand(10, 5).float() * 100  # Raw counts

        # Test in-place
        X_copy = X.clone()
        result = preprocess_expression_data(X=X_copy, data_format=mock_data_format)

        assert result is X_copy
        assert result.dtype == torch.float32
        assert result.shape == X.shape

        # Verify preprocessing steps were applied
        assert not torch.allclose(result, X)  # Should be different from original

    def test_numpy_torch_pipeline_consistency(self, mock_data_format):
        """Test that numpy and torch pipelines give consistent results."""
        X_np = np.random.rand(10, 5).astype(np.float32) * 100
        X_torch = torch.from_numpy(X_np.copy())

        result_np = preprocess_expression_data(
            X=X_np.copy(), data_format=mock_data_format
        )
        result_torch = preprocess_expression_data(
            X=X_torch.clone(), data_format=mock_data_format
        )

        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-5)

    def test_preprocessing_with_log_transform_disabled(self, mock_data_format):
        """Test preprocessing with log transform disabled."""
        # Modify data format to disable log transform
        data_format_no_log = DataFormat(
            gene_names=mock_data_format.gene_names,
            n_genes=mock_data_format.n_genes,
            genes_mu=mock_data_format.genes_mu,
            genes_sigma=mock_data_format.genes_sigma,
            target_sum=mock_data_format.target_sum,
            use_log_transform=False,  # Disabled
            aux_categorical_types=(),
        )

        X_np = np.random.rand(10, 5).astype(np.float32) * 100
        X_torch = torch.from_numpy(X_np.copy())

        result_np = preprocess_expression_data(X_np.copy(), data_format_no_log)
        result_torch = preprocess_expression_data(X_torch.clone(), data_format_no_log)

        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-5)


class TestLoadAndPreprocessData:
    """Test the load_and_preprocess_data_numpy function."""

    @pytest.fixture
    def mock_adata_and_format(self):
        """Create mock AnnData and DataFormat for testing."""
        n_cells, n_genes = 50, 10
        X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32) * 100)

        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        data_format = DataFormat(
            gene_names=[f"gene_{i}" for i in range(n_genes)],
            n_genes=n_genes,
            genes_mu=np.random.rand(n_genes).astype(np.float32),
            genes_sigma=np.random.rand(n_genes).astype(np.float32) + 0.1,
            target_sum=1e4,
            use_log_transform=True,
            eps=1e-6,
            aux_categorical_types=(),
        )

        return adata, data_format

    def test_load_and_preprocess_basic(self, mock_adata_and_format):
        """Test basic load and preprocess functionality."""
        adata, data_format = mock_adata_and_format
        row_indices = np.array([0, 1, 2, 3, 4])
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "test.h5ad"
            adata.write_h5ad(data_path)
            result = load_and_preprocess_data_numpy(
                data_path=data_path, data_format=data_format, row_indices=row_indices
            )

        assert result.shape == (len(row_indices), data_format.n_genes)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_load_and_preprocess_with_file(self, mock_adata_and_format):
        """Test load and preprocess with backed AnnData file."""
        adata, data_format = mock_adata_and_format

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.h5ad"
            adata.write_h5ad(file_path)

            # Load as backed
            row_indices = np.array([0, 1, 2, 3, 4])

            result = load_and_preprocess_data_numpy(
                data_path=file_path, data_format=data_format, row_indices=row_indices
            )

            assert result.shape == (len(row_indices), data_format.n_genes)
            assert result.dtype == np.float32
            assert not np.any(np.isnan(result))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Test handling of empty input arrays."""
        X_np = np.array([], dtype=np.float32).reshape(0, 3)
        X_torch = torch.tensor([], dtype=torch.float32).reshape(0, 3)

        # Row normalization should handle empty arrays
        result_np = apply_row_normalization(X_np, target_sum=100)
        result_torch = apply_row_normalization(X_torch, target_sum=100)

        assert result_np.shape == (0, 3)
        assert result_torch.shape == (0, 3)

    def test_single_cell(self):
        """Test handling of single cell input."""
        X_np = np.array([[1, 2, 3]], dtype=np.float32)
        X_torch = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        target_sum = 100.0

        result_np = apply_row_normalization(X_np.copy(), target_sum=target_sum)
        result_torch = apply_row_normalization(X_torch.clone(), target_sum=target_sum)

        assert np.allclose(result_np.sum(), target_sum)
        assert torch.allclose(result_torch.sum(), torch.tensor(target_sum))
        assert np.allclose(result_np, result_torch.numpy())

    def test_large_values(self):
        """Test handling of large values."""
        X_np = np.array([[1e6, 2e6, 3e6]], dtype=np.float32)
        X_torch = torch.tensor([[1e6, 2e6, 3e6]], dtype=torch.float32)
        target_sum = 1e4

        result_np = apply_row_normalization(X_np.copy(), target_sum=target_sum)
        result_torch = apply_row_normalization(X_torch.clone(), target_sum=target_sum)

        assert np.allclose(result_np.sum(), target_sum)
        assert torch.allclose(result_torch.sum(), torch.tensor(target_sum))
        assert np.allclose(result_np, result_torch.numpy(), rtol=1e-5)
