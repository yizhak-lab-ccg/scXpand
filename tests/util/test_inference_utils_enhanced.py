"""Enhanced tests for the refactored util.inference_utils module."""

from unittest.mock import Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scxpand.util.classes import ModelType
from scxpand.util.inference_utils import load_model, run_model_inference, setup_inference_environment


class TestSetupInferenceEnvironment:
    """Test suite for setup_inference_environment function."""

    def test_setup_inference_environment_success(self, tmp_path):
        """Test successful inference environment setup."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        data_format_path = model_path / "data_format.json"
        data_format_path.write_text('{"genes": ["gene1", "gene2"]}')

        mock_data_format = Mock()
        mock_model = Mock()
        mock_device = "cuda"

        with (
            patch("scxpand.util.inference_utils.load_data_format") as mock_load_format,
            patch("scxpand.util.inference_utils.get_device") as mock_get_device,
            patch("scxpand.util.inference_utils.load_model") as mock_load_model,
        ):
            # Setup mocks
            mock_load_format.return_value = mock_data_format
            mock_get_device.return_value = mock_device
            mock_load_model.return_value = mock_model

            # Run function
            data_format, model, device = setup_inference_environment(
                model_type=ModelType.AUTOENCODER, model_path=str(model_path)
            )

            # Verify calls
            mock_load_format.assert_called_once_with(data_format_path)
            mock_get_device.assert_called_once()
            mock_load_model.assert_called_once_with(
                model_type=ModelType.AUTOENCODER, model_path=model_path, device=mock_device
            )

            # Verify results
            assert data_format == mock_data_format
            assert model == mock_model
            assert device == mock_device

    def test_setup_inference_environment_missing_data_format(self, tmp_path):
        """Test error when data format file is missing."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        # No data_format.json file created

        with pytest.raises(FileNotFoundError, match="Data format file not found"):
            setup_inference_environment(model_type=ModelType.MLP, model_path=str(model_path))

    def test_setup_inference_environment_with_string_model_type(self, tmp_path):
        """Test setup with string model type."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        data_format_path = model_path / "data_format.json"
        data_format_path.write_text("{}")

        with (
            patch("scxpand.util.inference_utils.load_data_format"),
            patch("scxpand.util.inference_utils.get_device", return_value="cpu"),
            patch("scxpand.util.inference_utils.load_model") as mock_load_model,
        ):
            setup_inference_environment(
                model_type="lightgbm",  # String instead of enum
                model_path=str(model_path),
            )

            # Verify load_model was called with string
            mock_load_model.assert_called_once()
            call_args = mock_load_model.call_args[1]
            assert call_args["model_type"] == "lightgbm"

    def test_setup_inference_environment_path_conversion(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        data_format_path = model_path / "data_format.json"
        data_format_path.write_text("{}")

        with (
            patch("scxpand.util.inference_utils.load_data_format") as mock_load_format,
            patch("scxpand.util.inference_utils.get_device", return_value="cpu"),
            patch("scxpand.util.inference_utils.load_model"),
        ):
            setup_inference_environment(
                model_type=ModelType.SVM,
                model_path=str(model_path),  # String path
            )

            # Verify load_data_format was called with Path object
            mock_load_format.assert_called_once_with(data_format_path)

    def test_setup_inference_environment_path_validation(self, tmp_path):
        """Test path handling in setup_inference_environment."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        data_format_path = model_path / "data_format.json"
        data_format_path.write_text("{}")

        with (
            patch("scxpand.util.inference_utils.load_data_format") as mock_load,
            patch("scxpand.util.inference_utils.get_device", return_value="cpu"),
            patch("scxpand.util.inference_utils.load_model"),
        ):
            setup_inference_environment(model_type=ModelType.LOGISTIC, model_path=str(model_path))

            # Verify correct path was used
            mock_load.assert_called_once_with(data_format_path)


class TestLoadModelEnhanced:
    """Enhanced tests for load_model function."""

    def test_load_model_type_conversion(self, tmp_path):
        """Test that model_type is properly converted to string."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        with patch("scxpand.util.inference_utils.load_ae_model") as mock_load_ae:
            mock_model = Mock()
            mock_load_ae.return_value = mock_model

            # Test with enum
            result = load_model(model_type=ModelType.AUTOENCODER, model_path=results_path, device="cpu")

            mock_load_ae.assert_called_once_with(model_path=results_path, device="cpu")
            assert result == mock_model

    def test_load_model_unsupported_type(self, tmp_path):
        """Test error handling for unsupported model type."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        with pytest.raises(ValueError, match="Unsupported model_type for loading"):
            load_model(model_type="unsupported_model", model_path=results_path, device="cpu")

    def test_load_model_all_supported_types(self, tmp_path):
        """Test loading all supported model types."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        test_cases = [
            (ModelType.AUTOENCODER, "scxpand.util.inference_utils.load_ae_model"),
            (ModelType.MLP, "scxpand.util.inference_utils.load_nn_model"),
            (ModelType.LIGHTGBM, "scxpand.util.inference_utils.load_sklearn_model"),
            (ModelType.LOGISTIC, "scxpand.util.inference_utils.load_sklearn_model"),
            (ModelType.SVM, "scxpand.util.inference_utils.load_sklearn_model"),
        ]

        for model_type, mock_path in test_cases:
            with patch(mock_path) as mock_loader:
                mock_model = Mock()
                mock_loader.return_value = mock_model

                result = load_model(model_type=model_type, model_path=results_path, device="cuda")

                mock_loader.assert_called_once()
                assert result == mock_model

    def test_load_model_sklearn_routing(self, tmp_path):
        """Test that sklearn models are routed correctly."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        with patch("scxpand.util.inference_utils.load_sklearn_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            result = load_model(model_type=ModelType.LIGHTGBM, model_path=results_path, device="cpu")

            mock_load.assert_called_once_with(results_path=results_path)
            assert result is mock_model


class TestRunInferenceEnhanced:
    """Enhanced tests for run_inference function."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData object."""
        obs_data = pd.DataFrame(
            {"cell_id": ["cell_1", "cell_2", "cell_3"], "expansion": ["expanded", "non-expanded", "expanded"]}
        )
        X = np.random.randn(3, 10)
        return ad.AnnData(X=X, obs=obs_data)

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return Mock()

    @pytest.fixture
    def mock_data_format(self):
        """Create mock data format."""
        return Mock()

    def test_run_inference_with_both_data_sources(self, mock_model, mock_data_format, mock_adata):
        """Test that both adata and data_path can be provided and are passed through."""
        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_ae_inference:
            mock_ae_inference.return_value = np.array([0.8, 0.2, 0.9])

            result = run_model_inference(
                model_type=ModelType.AUTOENCODER,
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                data_path="some_path.h5ad",  # Both provided
                device="cpu",
            )

            # Should use adata and data_path (both are passed through)
            mock_ae_inference.assert_called_once()
            call_kwargs = mock_ae_inference.call_args[1]
            assert call_kwargs["adata"] is mock_adata  # Use 'is' for object identity
            assert call_kwargs["data_path"] == "some_path.h5ad"

            # Verify the result is returned correctly
            np.testing.assert_array_equal(result, np.array([0.8, 0.2, 0.9]))

    def test_run_inference_validation_neither_data_source(self, mock_model, mock_data_format):
        """Test validation error when neither adata nor data_path is provided."""
        with pytest.raises(ValueError, match="Either adata or data_path must be provided"):
            run_model_inference(
                model_type=ModelType.MLP,
                model=mock_model,
                data_format=mock_data_format,
                adata=None,
                data_path=None,
                device="cpu",
            )

    def test_run_inference_model_type_conversion(self, mock_model, mock_data_format, mock_adata):
        """Test that model_type enum is converted to string."""
        with patch("scxpand.util.inference_utils.run_mlp_inference") as mock_mlp_inference:
            mock_mlp_inference.return_value = np.array([0.7, 0.3, 0.8])

            run_model_inference(
                model_type=ModelType.MLP,  # Enum
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                device="cpu",
            )

            # Should route to MLP inference
            mock_mlp_inference.assert_called_once()

    def test_run_inference_all_model_types(self, mock_model, mock_data_format, mock_adata):
        """Test inference routing for all model types."""
        test_cases = [
            (ModelType.AUTOENCODER, "run_ae_inference"),
            (ModelType.MLP, "run_mlp_inference"),
            (ModelType.LIGHTGBM, "run_lightgbm_inference"),
            (ModelType.LOGISTIC, "run_linear_inference"),
            (ModelType.SVM, "run_linear_inference"),
        ]

        for model_type, inference_func in test_cases:
            with patch(f"scxpand.util.inference_utils.{inference_func}") as mock_inference:
                mock_inference.return_value = np.array([0.5, 0.6, 0.7])

                result = run_model_inference(
                    model_type=model_type,
                    model=mock_model,
                    data_format=mock_data_format,
                    adata=mock_adata,
                    device="cpu",
                    batch_size=256,
                    num_workers=4,
                )

                mock_inference.assert_called_once()
                # Verify common parameters are passed
                call_kwargs = mock_inference.call_args[1]
                assert call_kwargs["model"] is mock_model
                assert call_kwargs["data_format"] is mock_data_format
                assert call_kwargs["adata"] is mock_adata

                # Verify device parameter for neural networks
                if model_type in [ModelType.AUTOENCODER, ModelType.MLP]:
                    assert call_kwargs["device"] == "cpu"
                    assert call_kwargs["batch_size"] == 256
                    assert call_kwargs["num_workers"] == 4

                np.testing.assert_array_equal(result, np.array([0.5, 0.6, 0.7]))

    def test_run_inference_unsupported_model_type(self, mock_model, mock_data_format, mock_adata):
        """Test error for unsupported model type."""
        with pytest.raises(ValueError, match="Unsupported model_type for inference"):
            run_model_inference(
                model_type="unsupported", model=mock_model, data_format=mock_data_format, adata=mock_adata, device="cpu"
            )

    def test_run_inference_with_eval_indices(self, mock_model, mock_data_format, mock_adata):
        """Test inference with evaluation indices."""
        eval_indices = np.array([0, 2])

        with patch("scxpand.util.inference_utils.run_lightgbm_inference") as mock_lgb_inference:
            mock_lgb_inference.return_value = np.array([0.8, 0.9])

            _result = run_model_inference(
                model_type=ModelType.LIGHTGBM,
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                eval_row_inds=eval_indices,
            )

            # Verify eval_row_inds was passed
            mock_lgb_inference.assert_called_once()
            call_kwargs = mock_lgb_inference.call_args[1]
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], eval_indices)

    def test_run_inference_svm_routing(self, mock_model, mock_data_format, mock_adata):
        """Test that SVM inference is routed correctly."""
        with patch("scxpand.util.inference_utils.run_linear_inference") as mock_linear_inference:
            mock_predictions = np.array([0.4, 0.6, 0.8])
            mock_linear_inference.return_value = mock_predictions

            result = run_model_inference(
                model_type=ModelType.SVM, model=mock_model, data_format=mock_data_format, adata=mock_adata, device="cpu"
            )

            mock_linear_inference.assert_called_once()
            np.testing.assert_array_equal(result, mock_predictions)

    def test_run_inference_parameter_forwarding(self, mock_model, mock_data_format, mock_adata):
        """Test that all parameters are properly forwarded to inference functions."""
        with patch("scxpand.util.inference_utils.run_ae_inference") as mock_ae_inference:
            mock_ae_inference.return_value = np.array([0.1, 0.2, 0.3])

            run_model_inference(
                model_type=ModelType.AUTOENCODER,
                model=mock_model,
                data_format=mock_data_format,
                adata=mock_adata,
                data_path="backup_path.h5ad",
                device="cuda:1",
                batch_size=2048,
                num_workers=8,
                eval_row_inds=np.array([1, 2]),
            )

            # Verify all parameters were forwarded
            mock_ae_inference.assert_called_once()
            call_kwargs = mock_ae_inference.call_args[1]
            assert call_kwargs["model"] is mock_model
            assert call_kwargs["data_format"] is mock_data_format
            assert call_kwargs["adata"] is mock_adata
            assert call_kwargs["data_path"] == "backup_path.h5ad"
            assert call_kwargs["device"] == "cuda:1"
            assert call_kwargs["batch_size"] == 2048
            assert call_kwargs["num_workers"] == 8
            np.testing.assert_array_equal(call_kwargs["eval_row_inds"], np.array([1, 2]))
