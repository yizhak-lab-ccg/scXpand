"""Tests for multiprocessing-safe AnnData file operations.

This module tests the open_adata_multiprocessing_safe function to ensure
it correctly handles various scenarios and prevents multiprocessing conflicts.
"""

import tempfile

from pathlib import Path
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scxpand.util.io import open_adata_multiprocessing_safe


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    # Create some sample data
    n_obs, n_vars = 100, 50
    X = sp.random(n_obs, n_vars, density=0.1, format="csr")

    adata = ad.AnnData(
        X=X,
        obs={
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "expansion": np.random.choice([0, 1], size=n_obs),
        },
        var={
            "gene_name": [f"gene_{i}" for i in range(n_vars)],
        },
    )
    return adata


@pytest.fixture
def temp_h5ad_file(sample_adata):
    """Create a temporary H5AD file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)

    # Save the sample data to the temporary file
    sample_adata.write_h5ad(temp_path)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestOpenAdataMultiprocessingSafe:
    """Test cases for open_adata_multiprocessing_safe function."""

    def test_with_in_memory_adata(self, sample_adata):
        """Test using pre-loaded AnnData object."""
        test_indices = np.array([0, 1, 2])

        with open_adata_multiprocessing_safe(data_path=None, adata=sample_adata, indices=test_indices) as (
            adata,
            indices,
        ):
            assert adata is sample_adata
            assert np.array_equal(indices, test_indices)
            assert adata.shape == (100, 50)

    def test_with_file_path_backed_mode(self, temp_h5ad_file):
        """Test loading from file with backed='r' mode."""
        test_indices = np.array([5, 10, 15])

        with open_adata_multiprocessing_safe(data_path=temp_h5ad_file, adata=None, indices=test_indices) as (
            adata,
            indices,
        ):
            assert adata is not None
            assert np.array_equal(indices, test_indices)
            assert adata.shape == (100, 50)
            # Check that data can be accessed
            X_subset = adata.X[indices, :]
            assert X_subset.shape == (3, 50)

    def test_without_indices(self, temp_h5ad_file):
        """Test that function works when indices is None."""
        with open_adata_multiprocessing_safe(data_path=temp_h5ad_file, adata=None, indices=None) as (adata, indices):
            assert adata is not None
            assert indices is None
            assert adata.shape == (100, 50)

    def test_error_when_no_data_source(self):
        """Test that appropriate error is raised when neither data_path nor adata is provided."""
        with pytest.raises(ValueError, match="Either data_path or adata must be provided"):
            with open_adata_multiprocessing_safe(
                data_path=None,
                adata=None,
            ) as (adata, indices):
                pass

    def test_file_not_found_error(self):
        """Test behavior when file path doesn't exist."""
        nonexistent_path = "/tmp/nonexistent_file.h5ad"

        with pytest.raises(FileNotFoundError):
            with open_adata_multiprocessing_safe(
                data_path=nonexistent_path,
                adata=None,
            ) as (adata, indices):
                pass

    @patch("anndata.read_h5ad")
    def test_file_handle_cleanup_backed_mode(self, mock_read_h5ad, temp_h5ad_file):
        """Test that file handles are properly closed in backed mode."""
        # Create a mock AnnData object with a file handle
        mock_adata = MagicMock()
        mock_file = MagicMock()
        mock_adata.file = mock_file
        mock_read_h5ad.return_value = mock_adata

        with open_adata_multiprocessing_safe(
            data_path=temp_h5ad_file,
            adata=None,
        ) as (adata, indices):
            assert adata is mock_adata

        # Verify that the file was closed
        mock_file.close.assert_called_once()

    @patch("anndata.read_h5ad")
    def test_file_handle_cleanup_no_file_attr(self, mock_read_h5ad, temp_h5ad_file):
        """Test cleanup when AnnData object has no file attribute."""
        # Create a mock AnnData object without a file handle
        mock_adata = MagicMock()
        del mock_adata.file  # Remove file attribute
        mock_read_h5ad.return_value = mock_adata

        # Should not raise an error
        with open_adata_multiprocessing_safe(
            data_path=temp_h5ad_file,
            adata=None,
        ) as (adata, indices):
            assert adata is mock_adata

    @patch("anndata.read_h5ad")
    def test_file_handle_cleanup_none_file(self, mock_read_h5ad, temp_h5ad_file):
        """Test cleanup when AnnData object has file=None."""
        # Create a mock AnnData object with file=None
        mock_adata = MagicMock()
        mock_adata.file = None
        mock_read_h5ad.return_value = mock_adata

        # Should not raise an error
        with open_adata_multiprocessing_safe(
            data_path=temp_h5ad_file,
            adata=None,
        ) as (adata, indices):
            assert adata is mock_adata

    def test_multiple_concurrent_access_simulation(self, temp_h5ad_file):
        """Test simulating multiple concurrent accesses (single-threaded but multiple calls)."""
        test_indices_list = [
            np.array([0, 1, 2]),
            np.array([10, 11, 12]),
            np.array([20, 21, 22]),
        ]

        results = []

        # Simulate multiple workers accessing the same file
        for test_indices in test_indices_list:
            with open_adata_multiprocessing_safe(data_path=temp_h5ad_file, adata=None, indices=test_indices) as (
                adata,
                indices,
            ):
                X_subset = adata.X[indices, :]
                results.append(X_subset.shape)

        # All workers should successfully access the data
        assert len(results) == 3
        assert all(shape == (3, 50) for shape in results)

    @patch("scxpand.util.io.logger")
    def test_logging_debug_messages(self, mock_logger, temp_h5ad_file):
        """Test that appropriate debug messages are logged."""
        with open_adata_multiprocessing_safe(
            data_path=temp_h5ad_file,
            adata=None,
        ) as (adata, indices):
            pass

        # Check that debug messages were logged
        debug_calls = list(mock_logger.debug.call_args_list)
        assert len(debug_calls) >= 2  # Should have opening and closing debug messages

        # Check opening message
        opening_call = debug_calls[0][0][0]
        assert "Opening AnnData file with backed='r'" in opening_call

        # Check closing message
        closing_call = debug_calls[1][0][0]
        assert "Closed AnnData file handle" in closing_call


class TestIntegrationWithDataset:
    """Integration tests with CellsDataset to ensure the utility works in practice."""

    def test_dataset_integration_smoke_test(self, temp_h5ad_file):
        """Smoke test to ensure the utility works with CellsDataset."""
        # This test would require importing and setting up CellsDataset
        # For now, we'll just test that the utility function can be imported

        # Test that the function exists and is callable
        assert callable(open_adata_multiprocessing_safe)

        # Test basic functionality
        with open_adata_multiprocessing_safe(
            data_path=temp_h5ad_file,
            adata=None,
        ) as (adata, indices):
            assert adata is not None
            assert adata.shape == (100, 50)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
