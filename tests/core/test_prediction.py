"""Tests for core.prediction module."""

from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.core.inference_results import InferenceResults
from scxpand.core.prediction import run_prediction_pipeline
from scxpand.util.classes import ModelType


class TestRunPredictionPipeline:
    """Test suite for run_prediction_pipeline function."""

    @pytest.fixture
    def mock_adata(self):
        """Create a mock AnnData object for testing."""
        obs_data = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                "expansion": ["expanded", "non-expanded", "expanded"],
                "clone_id_size": [5, 2, 8],
                "median_clone_size": [3, 3, 3],
            }
        )
        X = np.random.randn(3, 100)  # 3 cells, 100 genes
        return ad.AnnData(X=X, obs=obs_data)

    @pytest.fixture
    def mock_predictions(self):
        """Mock prediction probabilities."""
        return np.array([0.8, 0.2, 0.9])

    @pytest.fixture
    def mock_model(self):
        """Mock trained model."""
        model = Mock()
        model.predict_proba = Mock(
            return_value=np.array([[0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
        )
        return model

    @pytest.fixture
    def mock_data_format(self):
        """Mock data format object."""
        data_format = Mock()
        data_format.gene_names = [f"gene_{i}" for i in range(100)]
        return data_format

    def test_run_prediction_pipeline_with_data_path_success(
        self, tmp_path, mock_adata, mock_model, mock_data_format, mock_predictions
    ):
        """Test successful prediction pipeline with data_path."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        data_path = tmp_path / "data.h5ad"
        save_path = tmp_path / "predictions"

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch(
                "scxpand.core.prediction.evaluate_predictions_and_save"
            ) as mock_evaluate,
            patch("scxpand.core.prediction.ensure_directory_exists"),
            patch("anndata.read_h5ad") as mock_read_h5ad,
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions
            mock_evaluate.return_value = {"AUROC": 0.85, "accuracy": 0.8}
            mock_read_h5ad.return_value = mock_adata

            # Run the function
            result = run_prediction_pipeline(
                model_type=ModelType.AUTOENCODER,
                model_path=str(model_path),
                data_path=str(data_path),
                save_path=str(save_path),
                batch_size=512,
                num_workers=2,
            )

            # Verify mocks were called correctly
            mock_setup.assert_called_once_with(
                model_type=ModelType.AUTOENCODER, model_path=model_path
            )

            mock_inference.assert_called_once_with(
                model_type=ModelType.AUTOENCODER,
                model=mock_model,
                data_format=mock_data_format,
                adata=None,
                data_path=str(data_path),
                device="cpu",
                batch_size=512,
                num_workers=2,
                eval_row_inds=None,
            )
            mock_read_h5ad.assert_called_once_with(str(data_path), backed="r")
            mock_evaluate.assert_called_once()

            # Verify result
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.metrics == {"AUROC": 0.85, "accuracy": 0.8}
            assert result.model_info is None  # Local models don't have model_info

    def test_run_prediction_pipeline_with_adata_success(
        self, tmp_path, mock_adata, mock_model, mock_data_format, mock_predictions
    ):
        """Test successful prediction pipeline with in-memory adata."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        save_path = tmp_path / "predictions"

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch(
                "scxpand.core.prediction.evaluate_predictions_and_save"
            ) as mock_evaluate,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions
            mock_evaluate.return_value = {"AUROC": 0.92, "F1": 0.88}

            # Run the function
            result = run_prediction_pipeline(
                model_type="mlp",  # Test string input
                model_path=str(model_path),
                adata=mock_adata,
                save_path=str(save_path),
            )

            # Verify mocks were called correctly
            mock_setup.assert_called_once_with(model_type="mlp", model_path=model_path)
            mock_inference.assert_called_once_with(
                model_type="mlp",
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                data_path=None,
                device="cpu",
                batch_size=1024,  # default
                num_workers=0,  # default
                eval_row_inds=None,
            )
            # Should not read from file since adata provided
            mock_evaluate.assert_called_once()

            # Verify result
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.metrics == {"AUROC": 0.92, "F1": 0.88}
            assert result.model_info is None  # Local models don't have model_info

    def test_run_prediction_pipeline_without_ground_truth(
        self, tmp_path, mock_model, mock_data_format, mock_predictions
    ):
        """Test prediction pipeline when ground truth is not available."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        # Create adata without expansion column
        obs_data_no_expansion = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                "clone_id_size": [5, 2, 8],
                "median_clone_size": [3, 3, 3],
            }
        )
        X = np.random.randn(3, 100)
        adata_no_expansion = ad.AnnData(X=X, obs=obs_data_no_expansion)

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch(
                "scxpand.core.prediction.evaluate_predictions_and_save"
            ) as mock_evaluate,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cuda")
            mock_inference.return_value = mock_predictions
            mock_evaluate.return_value = {}  # Empty metrics when no ground truth

            # Run the function
            result = run_prediction_pipeline(
                model_type=ModelType.LIGHTGBM,
                model_path=str(model_path),
                adata=adata_no_expansion,
            )

            # Verify evaluation was called but returned empty metrics
            mock_evaluate.assert_called_once()
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.metrics == {}  # Empty metrics when no ground truth
            assert result.model_info is None

    def test_run_prediction_pipeline_with_eval_indices(
        self, tmp_path, mock_adata, mock_model, mock_data_format
    ):
        """Test prediction pipeline with specific evaluation indices."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        eval_indices = np.array([0, 2])  # Only evaluate first and third cells
        mock_predictions = np.array([0.8, 0.9])  # Predictions for selected cells

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch(
                "scxpand.core.prediction.evaluate_predictions_and_save"
            ) as mock_evaluate,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions
            mock_evaluate.return_value = {"AUROC": 0.75}

            # Run the function
            result = run_prediction_pipeline(
                model_type=ModelType.SVM,
                model_path=str(model_path),
                adata=mock_adata,
                eval_row_inds=eval_indices,
            )

        # Verify the result is returned
        assert isinstance(result, InferenceResults)
        assert np.array_equal(result.predictions, mock_predictions)
        assert result.metrics == {"AUROC": 0.75}
        assert result.model_info is None

        # Verify inference was called with eval indices
        mock_inference.assert_called_once()
        call_kwargs = mock_inference.call_args[1]
        np.testing.assert_array_equal(call_kwargs["eval_row_inds"], eval_indices)

        # Verify evaluation used subset of obs data
        mock_evaluate.assert_called_once()
        eval_call_kwargs = mock_evaluate.call_args[1]
        expected_obs_df = mock_adata.obs.iloc[eval_indices]
        pd.testing.assert_frame_equal(eval_call_kwargs["obs_df"], expected_obs_df)

    def test_run_prediction_pipeline_validation_errors(self, tmp_path):
        """Test validation errors in prediction pipeline."""
        model_path = tmp_path / "model"

        # Test missing both adata and data_path
        with pytest.raises(
            ValueError, match="Either adata or data_path must be provided"
        ):
            run_prediction_pipeline(
                model_type=ModelType.MLP,
                model_path=str(model_path),
                adata=None,
                data_path=None,
            )

        # Test non-existent model path
        with pytest.raises(FileNotFoundError, match="Model path not found"):
            run_prediction_pipeline(
                model_type=ModelType.MLP,
                model_path=str(tmp_path / "nonexistent"),
                data_path="dummy.h5ad",
            )

    def test_run_prediction_pipeline_default_save_path(
        self, tmp_path, mock_adata, mock_model, mock_data_format, mock_predictions
    ):
        """Test prediction pipeline with default save path."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch(
                "scxpand.core.prediction.evaluate_predictions_and_save"
            ) as mock_evaluate,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions
            mock_evaluate.return_value = {}

            # Run without explicit save_path
            result = run_prediction_pipeline(
                model_type=ModelType.AUTOENCODER,
                model_path=str(model_path),
                adata=mock_adata,
            )

            # Verify result
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.metrics == {}
            assert result.model_info is None

    def test_run_prediction_pipeline_model_type_conversion(
        self, tmp_path, mock_adata, mock_model, mock_data_format, mock_predictions
    ):
        """Test that string model_type is properly handled."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions

            # Run with string model_type
            result = run_prediction_pipeline(
                model_type="autoencoder",  # String instead of enum
                model_path=str(model_path),
                adata=mock_adata,
            )

            # Verify result
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.model_info is None

    def test_run_prediction_pipeline_integration(
        self, tmp_path, mock_adata, mock_model, mock_data_format, mock_predictions
    ):
        """Test basic integration of prediction pipeline components."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with (
            patch(
                "scxpand.util.inference_utils.setup_inference_environment"
            ) as mock_setup,
            patch("scxpand.util.inference_utils.run_model_inference") as mock_inference,
            patch("scxpand.core.prediction.ensure_directory_exists"),
        ):
            # Setup mocks
            mock_setup.return_value = (mock_data_format, mock_model, "cpu")
            mock_inference.return_value = mock_predictions

            # Run pipeline - should complete without error
            result = run_prediction_pipeline(
                model_type=ModelType.MLP, model_path=str(model_path), adata=mock_adata
            )

            # Verify mocks were called
            mock_setup.assert_called_once()
            mock_inference.assert_called_once()

            # Should return InferenceResults object
            assert isinstance(result, InferenceResults)
            assert np.array_equal(result.predictions, mock_predictions)
            assert result.model_info is None

    @patch("scxpand.util.inference_utils.setup_inference_environment")
    @patch("scxpand.util.inference_utils.run_model_inference")
    def test_run_prediction_pipeline_exception_handling(
        self, mock_inference, mock_setup, tmp_path, mock_adata
    ):
        """Test exception handling in prediction pipeline."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        # Mock setup to raise an exception
        mock_setup.side_effect = FileNotFoundError("Data format not found")

        with pytest.raises(FileNotFoundError, match="Data format not found"):
            run_prediction_pipeline(
                model_type=ModelType.AUTOENCODER,
                model_path=str(model_path),
                adata=mock_adata,
            )

        # Verify inference was not called due to setup failure
        mock_inference.assert_not_called()
