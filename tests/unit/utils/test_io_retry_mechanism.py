"""Tests for the retry mechanism in I/O operations.

Tests verify that:
1. Retry mechanisms work correctly for HDF5 errors
2. Settings can be configured properly
3. Jitter functionality works as expected
"""

import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.util.io import (
    exponential_backoff_delay,
    is_hdf5_error,
    open_adata_multiprocessing_safe,
    retry_hdf5_operation,
    safe_read_adata_slice,
    settings,
)


class TestRetryMechanism:
    """Test the retry mechanism functions."""

    def test_is_hdf5_error(self):
        """Test HDF5 error detection."""
        # Should detect HDF5 errors
        assert is_hdf5_error(OSError("Can't synchronously read data"))
        assert is_hdf5_error(OSError("filter returned failure"))
        assert is_hdf5_error(OSError("HDF5 error occurred"))

        # Should not detect non-HDF5 errors
        assert not is_hdf5_error(ValueError("Invalid value"))
        assert not is_hdf5_error(OSError("File not found"))

    def test_exponential_backoff_without_jitter(self):
        """Test exponential backoff calculation without jitter."""
        # Test without jitter for predictable results
        delay1 = exponential_backoff_delay(0, jitter=False)
        delay2 = exponential_backoff_delay(1, jitter=False)
        delay3 = exponential_backoff_delay(2, jitter=False)

        assert delay1 == 0.1  # initial_delay
        assert delay2 == 0.2  # initial_delay * backoff_factor^1
        assert delay3 == 0.4  # initial_delay * backoff_factor^2

    def test_exponential_backoff_with_jitter(self):
        """Test that jitter produces different delays."""
        delays = [exponential_backoff_delay(1, jitter=True) for _ in range(5)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1

        # All delays should be within expected range (50% to 100% of base delay)
        base_delay = 0.2  # initial_delay * backoff_factor^1
        for delay in delays:
            assert base_delay * 0.5 <= delay <= base_delay

    def test_retry_hdf5_operation_success(self):
        """Test successful operation without retries."""
        call_count = 0

        def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_hdf5_operation(successful_operation)
        assert result == "success"
        assert call_count == 1

    def test_settings_configuration(self):
        """Test that settings can be configured."""
        original_max_retries = settings.max_retries
        original_jitter = settings.enable_jitter

        try:
            # Test changing settings
            settings.max_retries = 10
            settings.enable_jitter = False

            assert settings.max_retries == 10
            assert settings.enable_jitter is False

        finally:
            # Restore original settings
            settings.max_retries = original_max_retries
            settings.enable_jitter = original_jitter


class TestDataAccess:
    """Test data access functions."""

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        n_cells, n_genes = 20, 10
        X = np.random.poisson(3, size=(n_cells, n_genes)).astype(np.float32)

        obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
        var = pd.DataFrame({"gene_id": [f"gene_{i}" for i in range(n_genes)]})

        return ad.AnnData(X=X, obs=obs, var=var)

    def test_safe_read_adata_slice(self, sample_adata):
        """Test safe reading of AnnData slices."""
        indices = np.array([0, 5, 10, 15])

        result = safe_read_adata_slice(sample_adata, indices)

        assert result.shape == (4, 10)
        assert isinstance(result, np.ndarray)

        # Verify data integrity
        expected = (
            sample_adata.X[indices, :].toarray()
            if hasattr(sample_adata.X, "toarray")
            else sample_adata.X[indices, :]
        )
        np.testing.assert_array_equal(result, expected)

    def test_open_adata_with_in_memory_object(self, sample_adata):
        """Test context manager with in-memory AnnData object."""
        indices = np.array([1, 2, 3])

        with open_adata_multiprocessing_safe(
            None, adata=sample_adata, indices=indices
        ) as (adata, returned_indices):
            assert adata is sample_adata
            assert np.array_equal(returned_indices, indices)

    def test_open_adata_with_file_path(self, sample_adata):
        """Test context manager with file path."""
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save AnnData to file
            sample_adata.write(tmp_path)

            indices = np.array([1, 2])

            with open_adata_multiprocessing_safe(tmp_path, indices=indices) as (
                adata,
                returned_indices,
            ):
                assert adata is not None
                assert adata.n_obs == sample_adata.n_obs
                assert adata.n_vars == sample_adata.n_vars
                assert np.array_equal(returned_indices, indices)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_open_adata_invalid_arguments(self):
        """Test error handling for invalid arguments."""
        with pytest.raises(
            ValueError, match="Either data_path or adata must be provided"
        ):
            with open_adata_multiprocessing_safe(None, adata=None):
                pass


if __name__ == "__main__":
    pytest.main([__file__])
