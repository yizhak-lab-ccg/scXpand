"""Comprehensive integration tests for run_inference function.

This module tests the run_inference function with real models and data
to ensure end-to-end functionality works correctly.
"""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.inference import DEFAULT_MODEL_NAME, run_inference
from scxpand.core.inference_results import InferenceResults
from scxpand.data_util.data_format import DataFormat


class TestRunInferenceIntegration:
    """Integration tests for run_inference function with real models."""

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object for testing."""
        n_cells, n_genes = 100, 50

        # Create random gene expression data
        X = np.random.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        # Create observation data with expansion labels
        obs_data = {
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
            "expansion": np.random.choice(["expanded", "non-expanded"], n_cells),
            "tissue_type": np.random.choice(["tumor", "normal"], n_cells),
            "imputed_labels": np.random.choice(["T_cell", "B_cell"], n_cells),
        }
        obs_df = pd.DataFrame(obs_data)

        # Create variable data
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    @pytest.fixture
    def mock_data_format(self):
        """Create a mock DataFormat for testing."""
        return DataFormat(
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
            genes_mu=np.zeros(50, dtype=np.float32),
            genes_sigma=np.ones(50, dtype=np.float32),
            use_log_transform=True,
            use_zscore_norm=True,
            target_sum=1e4,
        )

    @pytest.fixture
    def mock_model_path(self, tmp_path, mock_data_format):
        """Create a mock model directory with necessary files."""
        model_path = tmp_path / "test_model"
        model_path.mkdir()

        # Create model_type.txt file
        (model_path / "model_type.txt").write_text("mlp")

        # Create data_format.json file
        mock_data_format.save(model_path / "data_format.json")

        # Create a mock model file
        (model_path / "model.pth").write_bytes(b"mock_model_data")

        return model_path

    def test_run_inference_local_model_success(self, sample_adata, mock_model_path):
        """Test run_inference with local model - successful case."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            # Mock successful prediction pipeline
            mock_predictions = np.random.random(len(sample_adata))
            mock_metrics = {
                "AUROC": 0.85,
                "error_rate": 0.15,
                "harmonic_avg": {"AUROC": 0.85},
                "T_cell__tumor": {"AUROC": 0.82},
                "B_cell__normal": {"AUROC": 0.88},
            }

            mock_pipeline.return_value = InferenceResults(
                predictions=mock_predictions,
                metrics=mock_metrics,
            )

            # Run inference
            results = run_inference(
                adata=sample_adata, model_path=str(mock_model_path), batch_size=32, num_workers=2, device="cpu"
            )

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(sample_adata)
            assert results.metrics == mock_metrics

            # Verify pipeline was called with correct parameters
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["model_path"] == str(mock_model_path)
            assert call_kwargs["adata"] is sample_adata
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["num_workers"] == 2
            assert call_kwargs["device"] == "cpu"

    def test_run_inference_with_data_path(self, sample_adata, mock_model_path, tmp_path):
        """Test run_inference with data_path instead of adata."""
        # Save adata to file
        data_path = tmp_path / "test_data.h5ad"
        sample_adata.write_h5ad(data_path)

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(sample_adata))
            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

            # Run inference with data_path
            results = run_inference(data_path=str(data_path), model_path=str(mock_model_path), batch_size=64)

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(sample_adata)

            # Verify pipeline was called with data_path
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["data_path"] == str(data_path)
            assert call_kwargs["adata"] is None
            assert call_kwargs["batch_size"] == 64

    def test_run_inference_registry_model(self, sample_adata):
        """Test run_inference with registry model."""
        with patch("scxpand.core.inference.fetch_model_and_run_inference") as mock_pretrained:
            mock_predictions = np.random.random(len(sample_adata))
            mock_metrics = {"AUROC": 0.90}

            mock_pretrained.return_value = InferenceResults(
                predictions=mock_predictions,
                metrics=mock_metrics,
            )

            # Run inference with registry model
            results = run_inference(adata=sample_adata, model_name=DEFAULT_MODEL_NAME, batch_size=128)

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(sample_adata)
            assert results.metrics == mock_metrics

            # Verify pretrained function was called
            mock_pretrained.assert_called_once()
            call_kwargs = mock_pretrained.call_args[1]
            assert call_kwargs["model_name"] == DEFAULT_MODEL_NAME
            assert call_kwargs["adata"] is sample_adata
            assert call_kwargs["batch_size"] == 128

    def test_run_inference_url_model(self, sample_adata):
        """Test run_inference with URL model."""
        with patch("scxpand.core.inference.fetch_model_and_run_inference") as mock_pretrained:
            mock_predictions = np.random.random(len(sample_adata))
            mock_metrics = {"AUROC": 0.88}

            mock_pretrained.return_value = InferenceResults(
                predictions=mock_predictions,
                metrics=mock_metrics,
            )

            # Run inference with URL model
            results = run_inference(adata=sample_adata, model_url="https://example.com/model.zip", batch_size=256)

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(sample_adata)
            assert results.metrics == mock_metrics

            # Verify pretrained function was called
            mock_pretrained.assert_called_once()
            call_kwargs = mock_pretrained.call_args[1]
            assert call_kwargs["model_url"] == "https://example.com/model.zip"
            assert call_kwargs["adata"] is sample_adata
            assert call_kwargs["batch_size"] == 256

    def test_run_inference_with_save_path(self, sample_adata, mock_model_path, tmp_path):
        """Test run_inference with save_path specified."""
        save_path = tmp_path / "results"

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(sample_adata))
            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

            # Run inference with save_path
            results = run_inference(adata=sample_adata, model_path=str(mock_model_path), save_path=str(save_path))

            # Verify results
            assert isinstance(results, InferenceResults)

            # Verify pipeline was called with save_path
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["save_path"] == str(save_path)

    def test_run_inference_with_eval_row_inds(self, sample_adata, mock_model_path):
        """Test run_inference with eval_row_inds specified."""
        eval_row_inds = np.array([0, 5, 10, 15, 20])

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(eval_row_inds))
            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

            # Run inference with eval_row_inds
            results = run_inference(adata=sample_adata, model_path=str(mock_model_path), eval_row_inds=eval_row_inds)

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(eval_row_inds)

            # Verify pipeline was called with eval_row_inds
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], eval_row_inds)

    @pytest.mark.slow
    def test_run_inference_parameter_validation(self, sample_adata):
        """Test run_inference parameter validation."""
        # Test no data input
        with pytest.raises(ValueError, match="Either adata or data_path must be provided"):
            run_inference(model_path="fake_path")

        # Test no model source (should use default registry model)
        # Mock the model fetch to avoid network calls
        with patch("scxpand.core.inference.fetch_model_and_run_inference") as mock_fetch:
            mock_fetch.side_effect = Exception("Mocked network error")

            with pytest.raises(Exception, match="Mocked network error"):
                run_inference(adata=sample_adata)

        # Test multiple model sources
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(adata=sample_adata, model_path="fake_path", model_name="fake_model")

        # Test all three model sources
        with pytest.raises(ValueError, match="Cannot specify multiple model sources"):
            run_inference(
                adata=sample_adata,
                model_path="fake_path",
                model_name="fake_model",
                model_url="https://fake.com/model.zip",
            )

    def test_run_inference_default_parameters(self, sample_adata, mock_model_path):
        """Test run_inference with default parameters."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(sample_adata))
            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

            # Run inference with minimal parameters
            results = run_inference(adata=sample_adata, model_path=str(mock_model_path))

            # Verify results
            assert isinstance(results, InferenceResults)

            # Verify default parameters were used
            mock_pipeline.assert_called_once()
            call_kwargs = mock_pipeline.call_args[1]
            assert call_kwargs["batch_size"] == 1024  # Default batch size
            assert call_kwargs["num_workers"] == 4  # Default num_workers
            assert call_kwargs["device"] is None  # Default device (auto-detect)
            assert call_kwargs["eval_row_inds"] is None  # Default eval_row_inds

    def test_run_inference_with_missing_expansion_column(self, sample_adata, mock_model_path):
        """Test run_inference with missing expansion column."""
        # Remove expansion column
        adata_no_expansion = sample_adata.copy()
        del adata_no_expansion.obs["expansion"]

        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(adata_no_expansion))
            # Empty metrics due to missing expansion column
            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={})

            # Run inference
            results = run_inference(adata=adata_no_expansion, model_path=str(mock_model_path))

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(adata_no_expansion)
            assert results.metrics == {}

    def test_run_inference_with_complete_dataset(self, sample_adata, mock_model_path):
        """Test run_inference with complete dataset (all columns present)."""
        with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
            mock_predictions = np.random.random(len(sample_adata))
            # Complete metrics with stratified results
            mock_metrics = {
                "AUROC": 0.85,
                "error_rate": 0.15,
                "harmonic_avg": {"AUROC": 0.85},
                "T_cell__tumor": {"AUROC": 0.82},
                "B_cell__normal": {"AUROC": 0.88},
                "T_cell__normal": {"AUROC": 0.83},
                "B_cell__tumor": {"AUROC": 0.87},
            }

            mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics=mock_metrics)

            # Run inference
            results = run_inference(adata=sample_adata, model_path=str(mock_model_path))

            # Verify results
            assert isinstance(results, InferenceResults)
            assert len(results.predictions) == len(sample_adata)
            assert results.metrics == mock_metrics

            # Should have stratified metrics
            stratified_keys = [key for key in results.metrics if "__" in key]
            assert len(stratified_keys) > 0

    def test_run_inference_device_parameter(self, sample_adata, mock_model_path):
        """Test run_inference with different device parameters."""
        test_devices = ["cpu", "cuda", "mps", None]

        for device in test_devices:
            with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
                mock_predictions = np.random.random(len(sample_adata))
                mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

                # Run inference with specific device
                _results = run_inference(adata=sample_adata, model_path=str(mock_model_path), device=device)

                # Verify device parameter was passed
                mock_pipeline.assert_called_once()
                call_kwargs = mock_pipeline.call_args[1]
                assert call_kwargs["device"] == device

    def test_run_inference_batch_size_variations(self, sample_adata, mock_model_path):
        """Test run_inference with different batch sizes."""
        test_batch_sizes = [1, 16, 64, 256, 1024, 2048]

        for batch_size in test_batch_sizes:
            with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
                mock_predictions = np.random.random(len(sample_adata))
                mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

                # Run inference with specific batch size
                _results = run_inference(adata=sample_adata, model_path=str(mock_model_path), batch_size=batch_size)

                # Verify batch size parameter was passed
                mock_pipeline.assert_called_once()
                call_kwargs = mock_pipeline.call_args[1]
                assert call_kwargs["batch_size"] == batch_size

    def test_run_inference_num_workers_variations(self, sample_adata, mock_model_path):
        """Test run_inference with different num_workers values."""
        test_num_workers = [0, 1, 2, 4, 8]

        for num_workers in test_num_workers:
            with patch("scxpand.core.inference.run_prediction_pipeline") as mock_pipeline:
                mock_predictions = np.random.random(len(sample_adata))
                mock_pipeline.return_value = InferenceResults(predictions=mock_predictions, metrics={"AUROC": 0.85})

                # Run inference with specific num_workers
                _results = run_inference(adata=sample_adata, model_path=str(mock_model_path), num_workers=num_workers)

                # Verify num_workers parameter was passed
                mock_pipeline.assert_called_once()
                call_kwargs = mock_pipeline.call_args[1]
                assert call_kwargs["num_workers"] == num_workers


if __name__ == "__main__":
    pytest.main([__file__])
