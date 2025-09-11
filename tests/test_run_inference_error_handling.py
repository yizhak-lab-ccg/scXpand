"""Error handling tests for run_inference function.

This module tests various error scenarios and edge cases for the run_inference function.
"""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.inference import DEFAULT_MODEL_NAME, run_inference
from scxpand.core.inference_results import InferenceResults


class TestRunInferenceErrorHandling:
    """Test error handling scenarios for run_inference function."""

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        n_cells, n_genes = 50, 25

        # Create random gene expression data
        X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        # Create observation data
        obs_data = {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "expansion": np.random.choice(["expanded", "non-expanded"], n_cells),
        }
        obs_df = pd.DataFrame(obs_data)

        # Create variable data
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    def test_run_inference_no_data_input(self):
        """Test run_inference with no data input."""
        with pytest.raises(ValueError, match="Either adata or data_path must be provided"):
            run_inference(model_path="fake_path")

    @patch("scxpand.core.inference.fetch_model_and_run_inference")
    def test_run_inference_no_model_source(self, mock_fetch, sample_adata):
        """Test run_inference with no model source (should use default registry model)."""
        mock_fetch.return_value = InferenceResults(
            predictions=np.array([0.5] * len(sample_adata)),
            model_info=None,
            metrics=None,
        )

        result = run_inference(adata=sample_adata)

        # Should call fetch_model_and_run_inference with default model
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[1]["model_name"] == DEFAULT_MODEL_NAME
        assert call_args[1]["model_url"] is None
        assert result.predictions is not None

    def test_run_inference_multiple_model_sources_path_and_name(self, sample_adata):
        """Test run_inference with both model_path and model_name."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(adata=sample_adata, model_path="fake_path", model_name="fake_model")

    def test_run_inference_multiple_model_sources_path_and_url(self, sample_adata):
        """Test run_inference with both model_path and model_url."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(adata=sample_adata, model_path="fake_path", model_url="https://fake.com/model.zip")

    def test_run_inference_multiple_model_sources_name_and_url(self, sample_adata):
        """Test run_inference with both model_name and model_url."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(adata=sample_adata, model_name="fake_model", model_url="https://fake.com/model.zip")

    def test_run_inference_all_three_model_sources(self, sample_adata):
        """Test run_inference with all three model sources specified."""
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(
                adata=sample_adata,
                model_path="fake_path",
                model_name="fake_model",
                model_url="https://fake.com/model.zip",
            )

    def test_run_inference_nonexistent_data_path(self):
        """Test run_inference with nonexistent data_path."""
        with pytest.raises(FileNotFoundError):
            run_inference(data_path="nonexistent_file.h5ad", model_path="fake_path")

    def test_run_inference_invalid_batch_size(self, sample_adata):
        """Test run_inference with invalid batch size (passed through to pipeline)."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test negative batch size - should pass through to pipeline
            run_inference(adata=sample_adata, model_path="fake_path", batch_size=-1)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["batch_size"] == -1

            mock_pipeline.reset_mock()

            # Test zero batch size - should pass through to pipeline
            run_inference(adata=sample_adata, model_path="fake_path", batch_size=0)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["batch_size"] == 0

    def test_run_inference_invalid_num_workers(self, sample_adata):
        """Test run_inference with invalid num_workers (passed through to pipeline)."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test negative num_workers - should pass through to pipeline
            run_inference(adata=sample_adata, model_path="fake_path", num_workers=-1)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["num_workers"] == -1

    def test_run_inference_invalid_eval_row_inds(self, sample_adata):
        """Test run_inference with invalid eval_row_inds (passed through to pipeline)."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test eval_row_inds with negative indices - should pass through to pipeline
            invalid_indices = np.array([-1, 0, 1])
            run_inference(adata=sample_adata, model_path="fake_path", eval_row_inds=invalid_indices)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], invalid_indices)

            mock_pipeline.reset_mock()

            # Test eval_row_inds with indices out of range - should pass through to pipeline
            out_of_range_indices = np.array([0, 1, 100])  # 100 is out of range for 50 cells
            run_inference(adata=sample_adata, model_path="fake_path", eval_row_inds=out_of_range_indices)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], out_of_range_indices)

    def test_run_inference_empty_adata(self):
        """Test run_inference with empty AnnData object."""
        # Create empty AnnData
        empty_adata = ad.AnnData()

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Should handle empty adata gracefully
            run_inference(adata=empty_adata, model_path="fake_path")

            mock_pipeline.assert_called_once()

    def test_run_inference_malformed_adata(self):
        """Test run_inference with malformed AnnData object."""
        # Create malformed AnnData (no var_names)
        malformed_adata = ad.AnnData(X=np.random.rand(10, 5))
        # Don't set var_names to simulate malformed data

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Should handle malformed adata gracefully
            run_inference(adata=malformed_adata, model_path="fake_path")

            mock_pipeline.assert_called_once()

    def test_run_inference_with_none_parameters(self, sample_adata):
        """Test run_inference with None parameters."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test with None parameters (should use defaults)
            run_inference(adata=sample_adata, model_path="fake_path", save_path=None, eval_row_inds=None, device=None)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["save_path"] is None
            assert call_kwargs["eval_row_inds"] is None
            assert call_kwargs["device"] is None

    def test_run_inference_pipeline_exception(self, sample_adata):
        """Test run_inference when prediction pipeline raises exception."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            # Mock pipeline to raise exception
            mock_pipeline.side_effect = RuntimeError("Pipeline failed")

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Pipeline failed"):
                run_inference(adata=sample_adata, model_path="fake_path")

    def test_run_inference_pretrained_exception(self, sample_adata):
        """Test run_inference when pretrained inference raises exception."""
        with patch("scxpand.core.inference.fetch_model_and_run_inference") as mock_pretrained:
            # Mock pretrained function to raise exception
            mock_pretrained.side_effect = RuntimeError("Download failed")

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Download failed"):
                run_inference(adata=sample_adata, model_name="fake_model")

    def test_run_inference_invalid_device(self, sample_adata):
        """Test run_inference with invalid device parameter (passed through to pipeline)."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test with invalid device string - should pass through to pipeline
            run_inference(adata=sample_adata, model_path="fake_path", device="invalid_device")

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["device"] == "invalid_device"

    def test_run_inference_large_batch_size(self, sample_adata):
        """Test run_inference with very large batch size."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Test with batch size larger than number of cells
            large_batch_size = len(sample_adata) * 2

            # Should handle large batch size gracefully
            run_inference(adata=sample_adata, model_path="fake_path", batch_size=large_batch_size)

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["batch_size"] == large_batch_size

    def test_run_inference_very_small_dataset(self):
        """Test run_inference with very small dataset."""
        # Create dataset with only 1 cell
        X = np.random.rand(1, 10).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": ["expanded"]})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(10)])
        small_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Should handle very small dataset gracefully
            run_inference(adata=small_adata, model_path="fake_path")

            mock_pipeline.assert_called_once()

    def test_run_inference_very_large_dataset(self):
        """Test run_inference with very large dataset."""
        # Create dataset with many cells (but still manageable for testing)
        n_cells, n_genes = 10000, 100
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs_df = pd.DataFrame({"expansion": np.random.choice(["expanded", "non-expanded"], n_cells)})
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        large_adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = None

            # Should handle large dataset gracefully
            run_inference(
                adata=large_adata,
                model_path="fake_path",
                batch_size=1000,  # Use reasonable batch size
            )

            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["batch_size"] == 1000

    def test_run_inference_memory_efficiency(self, sample_adata):
        """Test run_inference memory efficiency with different configurations."""
        configurations = [
            {"batch_size": 1, "num_workers": 0},  # Minimal memory
            {"batch_size": 16, "num_workers": 1},  # Low memory
            {"batch_size": 64, "num_workers": 2},  # Medium memory
            {"batch_size": 256, "num_workers": 4},  # High memory
        ]

        for config in configurations:
            with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = None

                # Should handle different memory configurations
                run_inference(adata=sample_adata, model_path="fake_path", **config)

                mock_pipeline.assert_called_once()
                call_kwargs = mock_pipeline.call_args[1]
                assert call_kwargs["batch_size"] == config["batch_size"]
                assert call_kwargs["num_workers"] == config["num_workers"]

                mock_pipeline.reset_mock()

    def test_run_inference_edge_case_parameters(self, sample_adata):
        """Test run_inference with edge case parameter values."""
        edge_cases = [
            {"batch_size": 1},  # Minimum batch size
            {"num_workers": 0},  # No multiprocessing
            {"device": "cpu"},  # CPU device
            {"eval_row_inds": np.array([])},  # Empty eval indices
        ]

        for case in edge_cases:
            with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = None

                # Should handle edge cases gracefully
                run_inference(adata=sample_adata, model_path="fake_path", **case)

                mock_pipeline.assert_called_once()
                mock_pipeline.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__])
