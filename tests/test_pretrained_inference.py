"""Tests for pretrained inference functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.inference_results import InferenceResults
from scxpand.pretrained.inference_api import fetch_model_and_run_inference


class TestFetchModelAndRunInference:
    """Tests for the fetch_model_and_run_inference function."""

    @pytest.fixture
    def mock_adata(self):
        """Create a mock AnnData object for testing."""
        obs_data = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                "expansion": ["expanded", "non-expanded", "expanded"],
            }
        )
        X = np.random.randn(3, 100)  # 3 cells, 100 genes
        return ad.AnnData(X=X, obs=obs_data)

    @pytest.fixture
    def mock_model_info(self):
        """Mock model info object."""
        mock_info = Mock()
        mock_info.name = "test_model"
        mock_info.version = "1.0.0"
        return mock_info

    def test_fetch_model_and_run_inference_model_name_success(self, tmp_path, mock_adata, mock_model_info):
        """Test successful inference with model name."""
        data_path = tmp_path / "test_data.h5ad"
        mock_adata.write_h5ad(data_path)

        mock_results = InferenceResults(predictions=np.array([0.8, 0.2, 0.9]), metrics={"AUROC": 0.85})

        with (
            patch("scxpand.pretrained.inference_api.get_pretrained_model_info") as mock_get_info,
            patch("scxpand.pretrained.inference_api.download_pretrained_model") as mock_download,
            patch("scxpand.pretrained.inference_api.run_prediction_pipeline") as mock_pipeline,
            patch("scxpand.pretrained.inference_api.load_model_type") as mock_load_model_type,
        ):
            # Setup mocks
            mock_get_info.return_value = mock_model_info
            mock_download.return_value = Path("/cache/test_model")
            mock_pipeline.return_value = mock_results
            mock_load_model_type.return_value = "mlp"  # Auto-detected model type

            result = fetch_model_and_run_inference(
                model_name="test_model",
                data_path=str(data_path),
                batch_size=512,
            )

            # Verify function calls
            mock_get_info.assert_called_once_with("test_model")
            mock_download.assert_called_once_with(model_name="test_model")
            mock_pipeline.assert_called_once()

            # Check pipeline arguments
            pipeline_kwargs = mock_pipeline.call_args[1]
            assert pipeline_kwargs["model_path"] == Path("/cache/test_model")
            assert pipeline_kwargs["data_path"] == data_path  # Path object, not string
            assert pipeline_kwargs["batch_size"] == 512
            # model_type is now auto-detected from model_type.txt, not passed as parameter

            # Check result includes model info
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, np.array([0.8, 0.2, 0.9]))
            assert result.metrics == {"AUROC": 0.85}
            assert result.model_info is not None
            assert result.model_info.model_name == "test_model"
            assert result.model_info.version == "1.0.0"
            assert result.model_info.model_type == "mlp"  # Auto-detected from model_type.txt
            assert result.model_info.source == "registry"

    def test_fetch_model_and_run_inference_model_doi_success(self, tmp_path, mock_adata):
        """Test successful inference with model DOI."""
        data_path = tmp_path / "test_data.h5ad"
        mock_adata.write_h5ad(data_path)

        mock_results = InferenceResults(predictions=np.array([0.7, 0.3, 0.8]), metrics={"AUROC": 0.82})

        with (
            patch("scxpand.pretrained.inference_api.download_pretrained_model") as mock_download,
            patch("scxpand.pretrained.inference_api.run_prediction_pipeline") as mock_pipeline,
            patch("scxpand.pretrained.inference_api.load_model_type") as mock_load_model_type,
        ):
            # Setup mocks
            mock_download.return_value = Path("/cache/external_model")
            mock_pipeline.return_value = mock_results
            mock_load_model_type.return_value = "autoencoder"  # Auto-detected model type

            _result = fetch_model_and_run_inference(
                model_url="https://your-platform.com/model.zip",
                data_path=str(data_path),
                num_workers=2,
            )

            # Verify function calls
            mock_download.assert_called_once_with(model_url="https://your-platform.com/model.zip")
            mock_pipeline.assert_called_once()

            # Check pipeline arguments
            pipeline_kwargs = mock_pipeline.call_args[1]
            assert pipeline_kwargs["model_path"] == Path("/cache/external_model")
            assert pipeline_kwargs["data_path"] == data_path  # Path object, not string
            assert pipeline_kwargs["num_workers"] == 2
            # model_type is now auto-detected from model_type.txt, not passed as parameter

    def test_fetch_model_and_run_inference_adata_input(self, mock_adata, mock_model_info):
        """Test inference with in-memory AnnData input."""
        mock_results = InferenceResults(predictions=np.array([0.6, 0.4, 0.7]), metrics={"AUROC": 0.80})

        with (
            patch("scxpand.pretrained.inference_api.get_pretrained_model_info") as mock_get_info,
            patch("scxpand.pretrained.inference_api.download_pretrained_model") as mock_download,
            patch("scxpand.pretrained.inference_api.run_prediction_pipeline") as mock_pipeline,
            patch("scxpand.pretrained.inference_api.load_model_type") as mock_load_model_type,
        ):
            # Setup mocks
            mock_get_info.return_value = mock_model_info
            mock_download.return_value = Path("/cache/test_model")
            mock_pipeline.return_value = mock_results
            mock_load_model_type.return_value = "svm"  # Auto-detected model type

            _result = fetch_model_and_run_inference(
                model_name="test_model",
                adata=mock_adata,
                save_path="/custom/save/path",
            )

            # Verify adata was passed correctly
            pipeline_kwargs = mock_pipeline.call_args[1]
            assert pipeline_kwargs["adata"] is mock_adata
            assert pipeline_kwargs["data_path"] is None
            assert pipeline_kwargs["save_path"] == "/custom/save/path"

    def test_fetch_model_and_run_inference_no_model_source_error(self):
        """Test error when no model source is provided."""
        with pytest.raises(ValueError, match="Either model_name or model_url must be provided"):
            fetch_model_and_run_inference(data_path="test.h5ad")

    def test_fetch_model_and_run_inference_both_model_sources_error(self):
        """Test error when both model sources are provided."""
        with pytest.raises(ValueError, match="Cannot specify both model_name and model_url"):
            fetch_model_and_run_inference(
                model_name="test_model",
                model_url="https://your-platform.com/model.zip",
                data_path="test.h5ad",
            )

    def test_fetch_model_and_run_inference_no_data_source_error(self):
        """Test error when no data source is provided."""
        with pytest.raises(ValueError, match="Either adata or data_path must be provided"):
            fetch_model_and_run_inference(model_name="test_model")

    def test_fetch_model_and_run_inference_missing_data_file_error(self, tmp_path):
        """Test error when data file doesn't exist."""
        missing_file = tmp_path / "missing.h5ad"

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            fetch_model_and_run_inference(model_name="test_model", data_path=str(missing_file))

    def test_fetch_model_and_run_inference_default_batch_size(self, tmp_path, mock_adata, mock_model_info):
        """Test that default batch size is applied correctly."""
        data_path = tmp_path / "test_data.h5ad"
        mock_adata.write_h5ad(data_path)

        mock_results = InferenceResults(predictions=np.array([0.5]), metrics={})

        with (
            patch("scxpand.pretrained.inference_api.get_pretrained_model_info") as mock_get_info,
            patch("scxpand.pretrained.inference_api.download_pretrained_model") as mock_download,
            patch("scxpand.pretrained.inference_api.run_prediction_pipeline") as mock_pipeline,
        ):
            # Setup mocks
            mock_get_info.return_value = mock_model_info
            mock_download.return_value = Path("/cache/test_model")
            mock_pipeline.return_value = mock_results

            fetch_model_and_run_inference(
                model_name="test_model",
                data_path=str(data_path),
                # No batch_size specified - should use default
            )

            # Verify default batch size was used
            pipeline_kwargs = mock_pipeline.call_args[1]
            assert pipeline_kwargs["batch_size"] == 1024  # Default value

    def test_fetch_model_and_run_inference_no_ground_truth(self, tmp_path, mock_adata, mock_model_info):
        """Test inference when ground truth is not available."""
        data_path = tmp_path / "test_data.h5ad"
        mock_adata.write_h5ad(data_path)

        mock_results = InferenceResults(
            predictions=np.array([0.5]),
            metrics={},  # Empty metrics when no ground truth
        )

        with (
            patch("scxpand.pretrained.inference_api.get_pretrained_model_info") as mock_get_info,
            patch("scxpand.pretrained.inference_api.download_pretrained_model") as mock_download,
            patch("scxpand.pretrained.inference_api.run_prediction_pipeline") as mock_pipeline,
        ):
            # Setup mocks
            mock_get_info.return_value = mock_model_info
            mock_download.return_value = Path("/cache/test_model")
            mock_pipeline.return_value = mock_results

            result = fetch_model_and_run_inference(
                model_name="test_model",
                data_path=str(data_path),
            )

            # Result should still include model info even without metrics
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, np.array([0.5]))
            assert result.metrics == {}
            assert result.model_info is not None
